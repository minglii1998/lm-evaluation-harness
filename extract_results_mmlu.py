import os
import json
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='')
    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    result_path = os.path.join('results',args.model_name)

    # Extract MMLU
    # try MMLU.json
    result_path_mmlu = os.path.join(result_path,'MMLU.json')
    if os.path.exists(result_path_mmlu):
        with open(result_path_mmlu, "r") as f:
            data = json.load(f)
        result_list = []
        results = data['results']
        for k in results.keys():
            result_list.append(results[k]['acc_norm'])
        result_mmlu = np.mean(result_list)
        print(result_mmlu)
    else:
        result_path_mmlu = os.path.join(result_path,'MMLU')
        json_files = [f for f in os.listdir(result_path_mmlu) if f.endswith('.json')]
        result_list = []
        for file in json_files:
            json_path = os.path.join(result_path_mmlu,file)
            with open(json_path, "r") as f:
                data = json.load(f)

            results = data['results']
            for k in results.keys():
                result_list.append(results[k]['acc_norm'])
        result_mmlu = np.mean(result_list)
        print(result_mmlu)
    pass

if __name__ == '__main__':
    main()
