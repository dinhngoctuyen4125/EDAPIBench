import json
import random
from collections import defaultdict
import os

LLMS = ['deepseek-1.3b', 'qwencoder-3b', 'starcoder-3b']
base_input_dir = 'data/EDAPIBench'
base_output_dir = 'data/EDAPIBench'

def process_api_data(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    intersection_dict = defaultdict(list)
    for sample in data:
        deprecated_apis = sample.get("deprecated api", [])
        probing_apis = sample.get("probing predictions", [[]])[0]
        
        if len(probing_apis) >= 2:
            probing_apis = probing_apis[1]
        else:
            probing_apis = []
        
        api_intersection = tuple(sorted(set(deprecated_apis) & set(probing_apis)))
        case_id = sample.get("case-id", "")
        
        if case_id and api_intersection:
            intersection_dict[api_intersection].append(case_id)
    
    for sample in data:
        deprecated_apis = sample.get("deprecated api", [])
        probing_apis = sample.get("probing predictions", [[]])[0]
        
        if len(probing_apis) >= 2:
            probing_apis = probing_apis[1]
        else:
            probing_apis = []
        
        api_intersection = tuple(sorted(set(deprecated_apis) & set(probing_apis)))
        related_case_ids = intersection_dict.get(api_intersection, [])
        
        candidates = related_case_ids.copy()
        if sample['case-id'] in candidates:
            candidates.remove(sample['case-id'])
        
        sample["portability"] = random.choice(candidates) if candidates else ""
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    
    print(f"Processing completed. Results saved to {output_file_path}")

if __name__ == "__main__":
    for llm in LLMS:
        input_file = os.path.join(base_input_dir, llm, 'all.json')
        output_dir = os.path.join(base_output_dir, llm)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'all.json')
        
        process_api_data(input_file, output_file)