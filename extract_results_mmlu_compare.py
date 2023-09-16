import os
import json
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_1", type=str, default='cherry_alpaca_5per')
    parser.add_argument("--model_name_2", type=str, default='alpaca_official_7b')
    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    result_path_1 = os.path.join('results',args.model_name_1)
    result_path_2 = os.path.join('results',args.model_name_2)

    # Extract MMLU
    result_path_mmlu = os.path.join(result_path_1,'MMLU.json')
    with open(result_path_mmlu, "r") as f:
        data = json.load(f)
    results_1 = data['results']

    result_path_mmlu = os.path.join(result_path_2,'MMLU.json')
    with open(result_path_mmlu, "r") as f:
        data = json.load(f)
    results_2 = data['results']

    low_list = []
    for k in results_1.keys():
        r1 = results_1[k]['acc_norm']
        r2 = results_2[k]['acc_norm']

        if r2-r1 > 0.1:
            low_list.append((k,r2-r1))
    
    print(len(low_list))
    print(low_list)
    pass

if __name__ == '__main__':
    main()
