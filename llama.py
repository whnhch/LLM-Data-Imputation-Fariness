import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import gc
import pandas as pd
import numpy as np
from datetime import datetime
import json
from collections import Counter

class Llama3:
    def __init__(self, model_path="meta-llama/Meta-Llama-3-8B-Instruct"):
        """
        Initializes the LLM withgiven model_path

        Args:
            model_path (str, optional): We use Meta-Llama-3-8B-Instruct to utilize chat-like result. 
                                        Defaults to "meta-llama/Meta-Llama-3-8B-Instruct".
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).eval()
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cuda",
            batch_size=1,
        )
        self.pipeline.model.config.pad_token_id = self.pipeline.model.config.eos_token_id
        
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids(""),
        ]
  
    def get_response(
          self, query, message_history=[], max_tokens=4096, temperature=0.6, top_p=0.9
      ):
        user_prompt = message_history + [{"role": "user", "content": query}]

        prompt = self.pipeline.tokenizer.apply_chat_template(
            user_prompt, tokenize=False, add_generation_prompt=True
        )
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            pad_token_id = self.pipeline.tokenizer.eos_token_id
        )
        response = outputs[0]["generated_text"][len(prompt):]

        del prompt 
        del outputs

        return response
    
    def chatbot(self, query, system_instructions=""):
        conversation = [{"role": "system", "content": system_instructions}]
        user_input = "User: "+query
        response = self.get_response(user_input, conversation)
        return response
    
    def get_column_list_prompt(self):
        return "Identify the columns that are interesting for analyzing the given input table data. Please identify as many columns as possible, ensuring you include at least one."
    
    def get_agg_list_prompt(self):
        return "Given the input column data and the list of aggregation functions, please determine the most suitable aggregation function for column.Please only choose aggreagtion function from the candidate aggregation function list. Please only return the most suitable aggregation function for column. Return the chosen aggregation functions in a list. Do not return the entire table."

# This is the translation from dataframe to the string for generative language model.
def dataframe_to_str(df):
    # Generate the header row
    headers = list(df.columns.astype(str))
    header_row = "| " + " | ".join(headers) + " |"
    
    # Generate the separator row
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    
    # Generate the data rows
    data_rows = []
    for row in df.itertuples(index=False, name=None):
        data_rows.append("| " + " | ".join(map(str, row)) + " |")
    
    # Combine all parts into a single string
    str_values = "\n".join([header_row, separator_row] + data_rows)
    
    return str_values

def series_to_str(col:pd.Series, header_name:str):
    str_values = ' |\n| '.join(col.astype(str))
    str_values = '| ' + header_name + ' |\n| ' + '| --- |' + ' |\n| ' + str_values
    str_values = str_values + ' |\n' 
    return str_values

def generate_message_interesting_aggregation(instruction:str, table_str: str):
    prompt="Task: "+ instruction + """Return the final result as JSON in the format {"selected_aggregation_function_type": "<a list of aggreagtion functions from the candidate list>"}.

    Input: **Column:** 
    
        Input:
        """ + table_str+ """
        
        **Candidate aggregation function:** 
        SUM
        MEAN
        COUNT
        MIN
        MAX
        
        Return the final result as JSON in the format {"selected_aggregation_function_type": "<a list of aggreagtion functions from the candidate list>"} withouy any code. 
        
    Output:"""
    
    return prompt
    
def find_value_in_response(response, key):
    key_idx = response.find(f'"{key}"')

    # Key not found
    if key_idx == -1:
        return None  

    start_idx = response.rfind('{', 0, key_idx)
    end_idx = response.find('}', key_idx)
  
    # Invalid JSON structure around the key
    if start_idx == -1 or end_idx == -1:
        return None  

    # Extract the JSON-like substring
    json_str = response[start_idx:end_idx+1]

    # Parse the JSON string into a Python dictionary
    try:
        data = json.loads(json_str)
        return data.get(key)
    except:
        return None
    return data[key]
            
def get_imputation_result(data, bot):
    instruction_1= bot.get_column_list_prompt()
    
    table_str=dataframe_to_str(data)
            
    results=[]
    for idx, inst in enumerate(instructions):            
        prompt=generate_message_interesting_column(inst, table_str)
        response=bot.chatbot(prompt)
        cols_list = find_value_in_response(response, "key_column_headers")
    
        filtered_cols=None

        if cols_list is not None:
            filtered_cols = [col for col in cols_list if col in data.columns]
        
        if filtered_cols==None: continue
        
        results.append(filtered_cols)
        
    filtered_results=majority_voting(results)
    
    return results, filtered_results

def get_interesting_cols_and_funcs(data, bot, size=20):
    _, cols_list = get_interesting_columns(data.head(size), bot)

    result={}
    
    if cols_list==None:
        return result
    
    for col in cols_list:
        if data[col].dtype == 'object': result[col] = ['COUNT']
        else: 
            _, result[col]=get_interesting_agg_funcs(data[col].head(size), col,bot)
    
    return result
