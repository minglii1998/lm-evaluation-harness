#!/bin/bash

# Define common parameters
MODEL_PATH=""
MODEL_NAME=""

export CUDA_VISIBLE_DEVICES=0

# Run ARC
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=$MODEL_PATH \
    --tasks arc_challenge \
    --batch_size 8 \
    --output_path results/$MODEL_NAME/ARC.json \
    --no_cache \
    --device auto \
    --num_fewshot 25

# Run HellaSwag
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=$MODEL_PATH \
    --tasks hellaswag \
    --batch_size 8 \
    --output_path results/$MODEL_NAME/HellaSwag.json \
    --no_cache \
    --device auto \
    --num_fewshot 10

# Run MMLU
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=$MODEL_PATH \
    --tasks hendrycksTest-* \
    --batch_size 2 \
    --output_path results/$MODEL_NAME/MMLU.json \
    --no_cache \
    --device auto \
    --num_fewshot 5

# Run TruthfulQA
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=$MODEL_PATH \
    --tasks truthfulqa_mc \
    --batch_size 8 \
    --output_path results/$MODEL_NAME/TruthfulQA.json \
    --no_cache \
    --device auto

# Run Extract Result
python extract_results.py --model_name $MODEL_NAME