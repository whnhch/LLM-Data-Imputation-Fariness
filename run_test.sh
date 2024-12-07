#!/bin/bash
export PYTHONPATH=$SRC_PATH:$PYTHONPATH

original_input="./dataset/original_compas.csv"
missing_input="./dataset/mcar_compas.csv"

python3 ./dataset/make_dataset.py
# echo "Run experiment on $input "
# python3 impute_data_llm.py --transformer_path "$transformer_path" \
#                                     --original_input "$original_input" --missing_input "$missing_input"\
#                 --table_frac "1.0"
