list1 = [
    "hendrycksTest-abstract_algebra",
    "hendrycksTest-anatomy",
    "hendrycksTest-astronomy",
    "hendrycksTest-business_ethics",
    "hendrycksTest-clinical_knowledge",
    "hendrycksTest-college_biology",
    "hendrycksTest-college_chemistry",
    "hendrycksTest-college_computer_science",
    "hendrycksTest-college_mathematics",
    "hendrycksTest-college_medicine",
    "hendrycksTest-college_physics",
    "hendrycksTest-computer_security",
    "hendrycksTest-conceptual_physics",
    "hendrycksTest-econometrics",
    "hendrycksTest-electrical_engineering",
    "hendrycksTest-elementary_mathematics",
    "hendrycksTest-formal_logic",
    "hendrycksTest-global_facts",
    "hendrycksTest-high_school_biology",
    "hendrycksTest-high_school_chemistry",
    "hendrycksTest-high_school_computer_science",
    "hendrycksTest-high_school_european_history",
    "hendrycksTest-high_school_geography",
    "hendrycksTest-high_school_government_and_politics",
    "hendrycksTest-high_school_macroeconomics",
    "hendrycksTest-high_school_mathematics",
    "hendrycksTest-high_school_microeconomics",
    "hendrycksTest-high_school_physics",
    "hendrycksTest-high_school_psychology",
    "hendrycksTest-high_school_statistics",
    "hendrycksTest-high_school_us_history",
    "hendrycksTest-high_school_world_history",
    "hendrycksTest-human_aging",
    "hendrycksTest-human_sexuality",
    "hendrycksTest-international_law",
    "hendrycksTest-jurisprudence",
    "hendrycksTest-logical_fallacies",
    "hendrycksTest-machine_learning",
    "hendrycksTest-management",
    "hendrycksTest-marketing",
    "hendrycksTest-medical_genetics",
    "hendrycksTest-miscellaneous",
    "hendrycksTest-moral_disputes",
    "hendrycksTest-moral_scenarios",
    "hendrycksTest-nutrition",
    "hendrycksTest-philosophy",
    "hendrycksTest-prehistory",
    "hendrycksTest-professional_accounting",
    "hendrycksTest-professional_law",
    "hendrycksTest-professional_medicine",
    "hendrycksTest-professional_psychology",
    "hendrycksTest-public_relations",
    "hendrycksTest-security_studies",
    "hendrycksTest-sociology",
    "hendrycksTest-us_foreign_policy",
    "hendrycksTest-virology",
    "hendrycksTest-world_religions"
]

import os
import json
import numpy as np

# # Double Check
# result_path_mmlu = 'results/alpaca_official_7b/MMLU.json'
# with open(result_path_mmlu, "r") as f:
#     data = json.load(f)
# result_list = []
# results = data['results']
# for k in results.keys():
#     if k not in list1:
#         print('list not contain:',k)


def get_json_files(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.json')]

directory = "results/llama2_13B_claude_alpaca/MMLU"  # Change this to your desired directory
json_files = get_json_files(directory)

result_list = []
no_show_list = list1
for file in json_files:
    if file.split('.')[0] in list1:
        no_show_list.remove(file.split('.')[0])

        overall_path = os.path.join(directory,file)
        with open(overall_path, "r") as f:
            data = json.load(f)
        results = data['results']
        for k in results.keys():
            result_list.append(results[k]['acc_norm'])

print('No show Length:', len(no_show_list))
print(no_show_list)

print('Task Length:',len(result_list))
result_mmlu = np.mean(result_list)
print(result_mmlu)


pass