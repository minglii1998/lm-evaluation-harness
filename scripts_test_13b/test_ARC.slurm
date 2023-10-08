#!/bin/bash

#SBATCH --job-name=HeARCllaSwag-Reflection_Alp_QA_13B_4p4de
#SBATCH --time=24:00:00             #Set the wall clock limit to 1hr and 30min
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

export TRANSFORMERS_CACHE=/nfshomes/minglii/scratch/cache/
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=trained_models_fs/Reflection_Alp_QA_13B_4p4de,use_accelerate=True \
    --tasks arc_challenge \
    --batch_size 4 \
    --output_path results/Reflection_Alp_QA_13B_4p4de/ARC.json \
    --no_cache \
    --device auto \
    --num_fewshot 25
