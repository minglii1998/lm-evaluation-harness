#!/bin/bash

#SBATCH --job-name=MMLU-1-ShareGPT_WizrdLM_ALL_whole_Q_then_A_4096_13B_4p4de
#SBATCH --time=48:00:00             #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                  #Request 1 task
#SBATCH --ntasks-per-node=1         #Request 1 task/core per node
#SBATCH --mem=64gb                 #Request 2560MB (2.5GB) per node
#SBATCH --output=outlogging.%j      #Send stdout/err to "Example1Out.[jobID]"
#SBATCH --gres=gpu:rtxa6000:2           #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger

# modules needed for running DL jobs. Module restore will also work
module add Python3/3.10.10
source /fs/nexus-scratch/minglii/venv/lm_eval/bin/activate

cd /nfshomes/minglii/scratch/lm-evaluation-harness

MODEL="hf-causal-experimental"
MODEL_ARGS="pretrained=trained_models_fs/Reflection/ShareGPT_WizrdLM_ALL_whole_Q_then_A_4096_13B_4p4de,use_accelerate=True"
BATCH_SIZE=4
DEVICE="auto"
NUM_FEWSHOT=5
BASE_OUTPUT_PATH="results/ShareGPT_WizrdLM_ALL_whole_Q_then_A_4096_13B_4p4de/MMLU"

export TRANSFORMERS_CACHE=/nfshomes/minglii/scratch/cache/
declare -a TASKS=(
    "hendrycksTest-abstract_algebra"
    "hendrycksTest-anatomy"
    "hendrycksTest-astronomy"
    "hendrycksTest-business_ethics"
    "hendrycksTest-clinical_knowledge"
    "hendrycksTest-college_biology"
    "hendrycksTest-college_chemistry"
    "hendrycksTest-college_computer_science"
    "hendrycksTest-college_mathematics"
    "hendrycksTest-college_medicine"
    "hendrycksTest-college_physics"
    "hendrycksTest-computer_security"
    "hendrycksTest-conceptual_physics"
    "hendrycksTest-econometrics"
    "hendrycksTest-electrical_engineering"
    "hendrycksTest-elementary_mathematics"
    "hendrycksTest-formal_logic"
    "hendrycksTest-global_facts"
    "hendrycksTest-high_school_biology"
    "hendrycksTest-high_school_chemistry"
    "hendrycksTest-high_school_computer_science"
    "hendrycksTest-high_school_european_history"
    "hendrycksTest-high_school_geography"
    "hendrycksTest-high_school_government_and_politics"
    "hendrycksTest-high_school_macroeconomics"
    "hendrycksTest-high_school_mathematics"
    "hendrycksTest-high_school_microeconomics"
    "hendrycksTest-high_school_physics"
    "hendrycksTest-high_school_psychology"
    "hendrycksTest-high_school_statistics"
    "hendrycksTest-high_school_us_history"
    "hendrycksTest-high_school_world_history"
    "hendrycksTest-human_aging"
    "hendrycksTest-human_sexuality"
    "hendrycksTest-international_law"
    "hendrycksTest-jurisprudence"
    "hendrycksTest-logical_fallacies"
    "hendrycksTest-machine_learning"
    "hendrycksTest-management"
    "hendrycksTest-marketing"
    "hendrycksTest-medical_genetics"
    "hendrycksTest-miscellaneous"
    "hendrycksTest-moral_disputes"
    "hendrycksTest-moral_scenarios"
    "hendrycksTest-nutrition"
    "hendrycksTest-philosophy"
    "hendrycksTest-prehistory"
    "hendrycksTest-professional_accounting"
    "hendrycksTest-professional_law"
    "hendrycksTest-professional_medicine"
    "hendrycksTest-professional_psychology"
    "hendrycksTest-public_relations"
    "hendrycksTest-security_studies"
    "hendrycksTest-sociology"
    "hendrycksTest-us_foreign_policy"
    "hendrycksTest-virology"
    "hendrycksTest-world_religions"
)

for TASK in "${TASKS[@]}"; do
    OUTPUT_PATH="$BASE_OUTPUT_PATH/$TASK.json"

    # Check if OUTPUT_PATH exists
    if [ -f "$OUTPUT_PATH" ]; then
        echo "$OUTPUT_PATH exists. Skipping this task."
        continue
    fi
    
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
