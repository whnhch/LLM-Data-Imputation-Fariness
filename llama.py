import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import gc
import pandas as pd
import numpy as np
from datetime import datetime
import json
from collections import Counter
import ast
import re

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
          self, query, message_history=[], max_tokens=8000, temperature=0.6, top_p=0.9
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
    
    def get_imputation_prompt(self):
        return "The task is to impute the missing values based on the other rows in the given input table. The missing data is represented as null. Given the input table data, please determine the most suitable values in the missing data."

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

def generate_message_imputation(instruction:str, table_str: str):
    prompt="Task: "+ instruction + """Return the final result as JSON in the format {"(row index, column index) : imputed value"}.

    Input: **Table:** 
    
        """ + table_str+ """
        
        Return the final result as JSON in the format {"(row index, column index) : imputed value"} without any code. 
        
    Output:"""
    
    return prompt

def parse_imputed_values(response):
    try:
        json_like_str = re.search(r'\{.*\}', response).group(0)

        json_like_str = json_like_str.strip("{}")
    except:
        json_like_str = re.search(r'\{.*', response).group(0)

        json_like_str = json_like_str.strip("{")

    parsed_dict = {}

    matches = re.findall(r'"\((\d+), (\d+)\) : ([^,]+)"', json_like_str)

    for row_str, col_str, value_str in matches:
        key_tuple = (int(row_str), int(col_str))

#         value = None if value_str.strip() == "null" else float(value_str.strip())
        value = 0 if value_str.strip() == "null" else float(value_str.strip())
        parsed_dict[key_tuple] = value

    print(parsed_dict)
    return parsed_dict

def get_imputation_result(data, bot):
    instruction= bot.get_imputation_prompt()
    
    table_str=dataframe_to_str(data)
            
    prompt=generate_message_imputation(instruction, table_str)
    print(prompt)
    response=bot.chatbot(prompt)
    print(response)
    result = parse_imputed_values(response)
    print(result)
    return result