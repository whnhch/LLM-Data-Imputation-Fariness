#!/bin/bash

# Input variables
export TRANSFORMERS_CACHE="$TRANSFORMER_PATH"
export HF_HOME="$TRANSFORMER_PATH"
export HF_DATASETS_CACHE="$TRANSFORMER_PATH"

# pip install llama-index-core
# pip install llama-index-llms-replicate
# pip install llama-index-embeddings-huggingface
# pip install llama-index-readers-file
# pip install llama-index-llms-huggingface

# rm -rf ./dataset/chunks
# mkdir -p ./dataset/chunks

# python chunk.py ./dataset/mcar_compas.csv --chunk_size 10 --output_dir ./dataset/chunks
python impute_data_rag.py --chunk_dir ./dataset/chunks
