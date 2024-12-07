import warnings
import os
import torch
import random
import pandas as pd
import argparse
import time 
import json 

from transformers import set_seed

from llama import Llama3, get_imputation_result
from evaluate import calculate_ifr, calculate_msie

def fill_in_missing_values(missing_data, original_data, result_dict, features):
    imputed_data = missing_data.copy()
    
    for key, value in result_dict.items():
        row_ind, col_ind = key
        imputed_data.iloc[row_ind, col_ind] = value
    
    total_count = 0
    #still missing data
    for feature in features:
        mask = imputed_data[feature].isna() | (imputed_data[feature] == 'null')
        total_count += mask.sum()
#         imputed_data.loc[mask, feature] = original_data.loc[mask, feature]
        imputed_data.loc[mask, feature] = 0
    
    print(f"not imputed data {total_count}")
    return imputed_data

if __name__=="__main__":
    
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer_path", type=str, default="~/.cache/huggingface/hub/" , help="The path for the language model.", required=False)
    parser.add_argument("--original_input", type=str, help="The path and filename of the original_input", required=True)
    parser.add_argument("--missing_input", type=str, help="The path and filename of the missing input", required=True)

    # Parameters
    parser.add_argument("--table_frac", type=float, default=0.2, help="The table size", required=True)
    args = parser.parse_args()

    TRANSFORMER_PATH=args.transformer_path
    os.environ['TRANSFORMERS_CACHE'] = TRANSFORMER_PATH
    os.environ['HF_HOME'] = TRANSFORMER_PATH
    os.environ['HF_DATASETS_CACHE'] = TRANSFORMER_PATH

    torch.manual_seed(42)
    random.seed(42)
    set_seed(42)
    
    bot = Llama3()

    original_filename=args.original_input
    
    original_df = pd.read_csv(original_filename, index_col=False)
    original_df = original_df.loc[:, ~original_df.columns.str.contains('^Unnamed')].head(50)
    
    missing_filename=args.missing_input
    
    missing_df = pd.read_csv(missing_filename, index_col=False).fillna('null')
    missing_df = missing_df.loc[:, ~missing_df.columns.str.contains('^Unnamed')].head(50)
    
    #Get imputation resullt from LLM
    result_dict=get_imputation_result(missing_df, bot)
    imputed_df = fill_in_missing_values(missing_df, original_df, result_dict, ['age', 'priors_count', 'decile_score'])
    print(imputed_df)
    msie=calculate_msie(original_df, missing_df, imputed_df, ['age', 'priors_count', 'decile_score'])
    print(f"msi : {msie}")
    ifr=calculate_ifr(original_df, missing_df, imputed_df, ['age', 'priors_count', 'decile_score'])
    print(f"ifr : {ifr}")
    