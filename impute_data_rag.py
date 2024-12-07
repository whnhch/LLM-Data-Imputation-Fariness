from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document
import pandas as pd
import os
from typing import List
from io import StringIO

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings

from evaluate import calculate_ifr, calculate_msie
import json
import re


def extract_custom_json(response_text):
    """
    Extracts a custom JSON-like structure from the text.
    
    Args:
        response_text (str): The text containing the custom JSON-like data.

    Returns:
        dict: A dictionary representation of the extracted data.
    """
    # Pattern to match the key-value pairs in the structure
    pattern = r'"(\d+)":\s*"([^"]+)"'
    
    # Find all matches in the text
    matches = re.findall(pattern, response_text)
    
    # Convert matches into a dictionary
    extracted_data = {key: value for key, value in matches}
    
    return extracted_data

def load_chunk(file_path: str):
    """Load a single chunk from a CSV file."""
    return pd.read_csv(file_path)

def create_llama_index(input_dir: str) -> VectorStoreIndex:
    """
    Create a LlamaIndex from documents stored in the specified directory.

    :param input_dir: Directory containing text files for indexing.
    :return: A LlamaIndex instance.
    """
    documents = SimpleDirectoryReader(input_dir).load_data()
    index = VectorStoreIndex.from_documents(documents, chunk_size=1024)
    return index


# Zephyr-specific formatting functions
def completion_to_prompt(completion):
    return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"


def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message["role"] == "system":
            prompt += f"<|system|>\n{message['content']}</s>\n"
        elif message["role"] == "user":
            prompt += f"<|user|>\n{message['content']}</s>\n"
        elif message["role"] == "assistant":
            prompt += f"<|assistant|>\n{message['content']}</s>\n"

    if not prompt.startswith("<|system|>\n"):
        prompt = "<|system|>\n</s>\n" + prompt

    prompt = prompt + "<|assistant|>\n"
    return prompt


# Function to impute missing values with Zephyr-specific prompt formatting

def get_missing_values(chunk):
    """
    Extracts all the missing value locations and their column names from the chunk.
    Returns a dictionary mapping row indices to their missing columns.
    """
    missing = {}
    for row_idx, row in chunk.iterrows():
        missing_cols = row[row.isna()].index.tolist()
        if missing_cols:
            missing[row_idx] = missing_cols
    return missing

def impute_chunk_with_context(chunk, query_engine):
    missing_locations = get_missing_values(chunk)

    # Generate the query text
    query_text = f"""
You are tasked with filling missing values in a table at age column.
Here is the chunk of the table you are working on:
{chunk.to_string(index=False, na_rep='BLANK')}

Please return only the imputed values in JSON format as follows based on other tables:
{{ 
"missing row index": imputed value"
}}

Please provide only the JSON format without any explanation.
"""
    
    # Retrieve relevant context from the query engine
    response = query_engine.query(query_text)
    print(response)
    context = response.response.strip()

    # Parse the JSON response into a dictionary
    try:
        # Extract data from the custom JSON-like context
        context_data = extract_custom_json(context)
    except Exception as e:  # Catch generic exceptions for robust handling
        print(f"Failed to parse JSON response: {context}, Error: {e}")
        return chunk  # Return the original chunk if parsing fails

    # Convert the context data into a DataFrame
    context_df = pd.DataFrame([context_data])  # Transpose the dictionary to align keys as columns

    # Impute missing values in the chunk using the context data
    for row_idx, row in chunk.iterrows():
        for col in chunk.columns:
            if pd.isna(row[col]):
                # Check if the column exists in the context data
                if col in context_df.columns:
                    # Use the first non-null value from the context data
                    non_null_values = context_df[col].dropna()
                    if not non_null_values.empty:
                        chunk.at[row_idx, col] = non_null_values.iloc[0]
    
    return chunk

def impute_empty_cells(chunks, query_engine):
    """
    Re-impute chunks with empty cells and include all chunks in the final result.
    
    Args:
        chunks (list of pd.DataFrame): List of original DataFrame chunks.
        query_engine: Query engine for re-imputation.
    
    Returns:
        pd.DataFrame: Combined DataFrame with all chunks included.
    """
    final_chunks = []
    for i, chunk in enumerate(chunks):
        if chunk.isnull().values.any():  # Check for empty cells
            print(f"Empty cells detected in chunk {i}. Re-imputing using context.")
            chunk = impute_chunk_with_context(chunk, query_engine)  # Re-impute
        else:
            print(f"Chunk {i} has no empty cells. Adding as-is.")
        final_chunks.append(chunk)  # Add re-imputed or original chunk
    return pd.concat(final_chunks, ignore_index=True)

def main(chunk_dir: str, output_file: str):
#     Settings.embed_model = HuggingFaceEmbedding(
#         model_name="BAAI/bge-small-en-v1.5"
#     )

#     Settings.llm = HuggingFaceLLM(
#         model_name="meta-llama/Meta-Llama-3-8B-Instruct",
#         tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
#         context_window=8192,
#         max_new_tokens=256,
#         generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
#         messages_to_prompt=messages_to_prompt,
#         completion_to_prompt=completion_to_prompt,
#         device_map="auto",
#     )
    
#     chunks = []
#     for chunk_file in sorted(os.listdir(chunk_dir)):
#         if not chunk_file.endswith(".csv"):
#             continue
#         chunk_path = os.path.join(chunk_dir, chunk_file)
#         chunk = load_chunk(chunk_path)
# #         chunks.append(chunk.to_string(index=False, na_rep='BLANK'))
#         chunks.append(chunk)
        
    # Create LlamaIndex from the documents
#     index = create_llama_index(chunk_dir)

    # Impute missing values in each chunk
#     query_engine = index.as_query_engine(similarity_top_k=2)

#     # Combine the imputed chunks back into a single DataFrame
#     imputed_df = impute_empty_cells(chunks, query_engine)
    
#     # Save the final imputed table
#     imputed_df.to_csv(output_file, index=False)
#     print(f"Final imputed table saved to: {output_file}")
    
    original_df = pd.read_csv('./dataset/original_compas.csv', index_col=None).dropna().reset_index(drop=True)
#     original_df = original_df.dropna()
    
    missing_df= pd.read_csv('./dataset/mcar_compas.csv', index_col=None)[:-1].fillna('null')
    imputed_df= pd.read_csv('./imputed_table.csv', index_col=None)[:-1]
    
    msie=calculate_msie(original_df, missing_df, imputed_df, ['age'])
    print(f"msi : {msie}")
    ifr=calculate_ifr(original_df, missing_df, imputed_df, ['age'])
    print(f"ifr : {ifr}")
    
if __name__ == "__main__":
    import argparse

    # Argument parser
    parser = argparse.ArgumentParser(description="Impute missing values in a table using LlamaIndex for RAG.")
    parser.add_argument("--chunk_dir", type=str, help="Path to the input CSV file.")
    parser.add_argument("--output_file", type=str, default="./imputed_table.csv", help="Path to save the imputed table.")

    args = parser.parse_args()

    # Run the main function
    main(chunk_dir=args.chunk_dir, output_file=args.output_file)
