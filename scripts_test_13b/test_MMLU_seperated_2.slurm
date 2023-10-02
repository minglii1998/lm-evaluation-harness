#!/bin/bash

#SBATCH --time=48:00:00             #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                  #Request 1 task
#SBATCH --ntasks-per-node=1         #Request 1 task/core per node
#SBATCH --mem=64gb                 #Request 2560MB (2.5GB) per node
#SBATCH --output=outlogging.%j      #Send stdout/err to "Example1Out.[jobID]"
#SBATCH --gres=gpu:rtxa6000:2           #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=tron 
#SBATCH --qos=medium

# modules needed for running DL jobs. Module restore will also work
module add Python3/3.10.10
source /fs/nexus-scratch/minglii/venv/lm_eval/bin/activate

cd /nfshomes/minglii/scratch/lm-evaluation-harness

MODEL="hf-causal-experimental"
MODEL_ARGS="pretrained=khalidsaifullaah/lca13,use_accelerate=True"
BATCH_SIZE=1
DEVICE="auto"
NUM_FEWSHOT=5
BASE_OUTPUT_PATH="results/llama2_13B_claude_alpaca/MMLU"

export TRANSFORMERS_CACHE=/nfshomes/minglii/scratch/cache/
declare -a TASKS=(
    'hendrycksTest-abstract_algebra'
    'hendrycksTest-world_religions'
    'hendrycksTest-virology'
    'hendrycksTest-us_foreign_policy'
    'hendrycksTest-sociology'
    'hendrycksTest-security_studies'
    'hendrycksTest-public_relations'
    'hendrycksTest-professional_psychology'
    'hendrycksTest-professional_medicine'
    'hendrycksTest-professional_law'
    'hendrycksTest-professional_accounting'
    'hendrycksTest-prehistory'
    'hendrycksTest-philosophy'
    'hendrycksTest-nutrition'
    'hendrycksTest-moral_scenarios'
    'hendrycksTest-moral_disputes'
    'hendrycksTest-miscellaneous'
    'hendrycksTest-medical_genetics'
    'hendrycksTest-marketing'
    'hendrycksTest-management'
    'hendrycksTest-machine_learning'
    'hendrycksTest-logical_fallacies'
    'hendrycksTest-jurisprudence'
    'hendrycksTest-international_law'
    'hendrycksTest-human_sexuality'
    'hendrycksTest-human_aging'
    'hendrycksTest-high_school_world_history'
    'hendrycksTest-high_school_us_history'
    'hendrycksTest-high_school_statistics'
    'hendrycksTest-high_school_psychology'
    'hendrycksTest-high_school_physics'
    'hendrycksTest-high_school_microeconomics'
    'hendrycksTest-high_school_mathematics'
    'hendrycksTest-high_school_macroeconomics'
    'hendrycksTest-high_school_government_and_politics'
    'hendrycksTest-high_school_geography'
    'hendrycksTest-high_school_european_history'
    'hendrycksTest-high_school_computer_science'
    'hendrycksTest-high_school_chemistry'
    'hendrycksTest-high_school_biology'
    'hendrycksTest-global_facts'
    'hendrycksTest-formal_logic'
    'hendrycksTest-elementary_mathematics'
    'hendrycksTest-electrical_engineering'
    'hendrycksTest-econometrics'
    'hendrycksTest-conceptual_physics'
    'hendrycksTest-computer_security'
    'hendrycksTest-college_physics'
    'hendrycksTest-college_medicine'
    'hendrycksTest-college_mathematics'
    'hendrycksTest-college_computer_science'
    'hendrycksTest-college_chemistry'
    'hendrycksTest-college_biology'
    'hendrycksTest-clinical_knowledge'
    'hendrycksTest-business_ethics'
    'hendrycksTest-astronomy'
    'hendrycksTest-anatomy'
)

for TASK in "${TASKS[@]}"; do
    OUTPUT_PATH="$BASE_OUTPUT_PATH/$TASK.json"
    CMD="python main.py \
    --model $MODEL \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --batch_size $BATCH_SIZE \
    --output_path $OUTPUT_PATH \
    --device $DEVICE \
    --no_cache \
    --num_fewshot $NUM_FEWSHOT"
    echo "Running: $CMD"
    eval $CMD
done
