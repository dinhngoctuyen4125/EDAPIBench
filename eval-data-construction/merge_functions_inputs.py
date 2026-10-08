import os
import json
from collections import defaultdict

def main():
    root_dir = 'data/predicted-dapi-results'
    functions_results = set()
    inputs_results = set()
    libraries = ['numpy', 'pandas', 'pytorch', 'scipy', 'seaborn', 'sklearn', 'tensorflow', 'transformers']
    llms = ["deepseek-1.3b", "starcoder-3b", "qwencoder-3b"]
    
    for library in libraries:
        library_path = os.path.join(root_dir, library)
        for llm in llms:
            json_path = os.path.join(library_path, llm, 'data.json')
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        for line in data:
                            functions_results.add(line['function'])
                            inputs_results.add(line['probing input'] + line['reference'])
                except Exception as e:
                    print(f"Error reading file {json_path}: {e}")
            else:
                print(f"File not found: {json_path}")
    
    output_dir = os.path.dirname('data/all_functions_inputs/')
    os.makedirs(output_dir, exist_ok=True)
    functions_output_file = os.path.join(output_dir, 'all_functions.json')
    inputs_output_file = os.path.join(output_dir, 'all_inputs.json')
    with open(functions_output_file, 'w') as f:
        json.dump(list(functions_results), f, indent=4)
    with open(inputs_output_file, 'w') as f:
        json.dump(list(inputs_results), f, indent=4)
    print(f"Results saved to {output_dir}")
    print(f"\nTotal code snippets: {len(functions_results)}")

if __name__ == "__main__":
    main()