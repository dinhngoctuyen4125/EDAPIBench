import os
import json
import argparse
from pathlib import Path

def add_case_id_to_json_files(input_dir, output_dir, target_file="data.json"):
    llm_counters = {}
    
    libs = ['pandas', 'pytorch', 'scipy', 'seaborn', 'sklearn', 'tensorflow', 'transformers']
    llms = ["deepseek-1.3b", "starcoder-3b", "qwencoder-3b"]
    
    for lib in libs:
        lib_path = os.path.join(input_dir, lib)
        if os.path.isdir(lib_path):
            for llm in llms:
                if llm not in llm_counters:
                    llm_counters[llm] = 1
                
                file_path = os.path.join(lib_path, llm, target_file)
                if not os.path.exists(file_path):
                    continue
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for i in range(len(data)):
                    sample = data[i]
                    case_id = f"{llm}-{llm_counters[llm]}"
                    
                    new_sample = {"case-id": case_id}
                    new_sample.update(sample)
                    data[i] = new_sample
                    
                    llm_counters[llm] += 1
                
                output_path = os.path.join(output_dir, lib, llm)
                os.makedirs(output_path, exist_ok=True)
                
                with open(os.path.join(output_path, 'data.json'), 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4)
    
    print("\nProcessing completed. Sample counts per LLM:")
    for llm, count in llm_counters.items():
        print(f"{llm}: {count - 1} samples")

if __name__ == "__main__":
    add_case_id_to_json_files(
        'data/predicted-dapi-results',
        'data/predicted-dapi-results',
    )