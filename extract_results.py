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

    # Extract ARC
    result_path_arc = os.path.join(result_path,'ARC.json')
    with open(result_path_arc, "r") as f:
        data = json.load(f)
    result_arc = data['results']['arc_challenge']['acc_norm']

    # Extract HellaSwag
    result_path_hella = os.path.join(result_path,'HellaSwag.json')
    with open(result_path_hella, "r") as f:
        data = json.load(f)
    result_hella = data['results']['hellaswag']['acc_norm']

    # Extract MMLU
    result_path_mmlu = os.path.join(result_path,'MMLU.json')
    with open(result_path_mmlu, "r") as f:
        data = json.load(f)
    result_list = []
    results = data['results']
    for k in results.keys():
        result_list.append(results[k]['acc_norm'])
    result_mmlu = np.mean(result_list)

    # Extract TruthfulQA
    result_path_truth = os.path.join(result_path,'TruthfulQA.json')
    with open(result_path_truth, "r") as f:
        data = json.load(f)
    result_truth = data['results']['truthfulqa_mc']['mc2']

    result_all = [result_arc, result_hella, result_mmlu, result_truth]
    result_all = [np.mean(result_all)] + result_all

    record_text = args.model_name + ':' + str(result_all) + '\n'
    with open('Record_All.txt','a') as f:
        f.write(record_text)
    pass

if __name__ == '__main__':
    main()
