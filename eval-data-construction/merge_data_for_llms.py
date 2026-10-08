import os
import json

LIBS = ['numpy', 'pandas', 'pytorch', 'scipy', 'seaborn', 'sklearn', 'tensorflow', 'transformers']
LLMS = ['deepseek-1.3b', 'qwencoder-3b', 'starcoder-3b']
base_input_dir = 'data/benchmark-generalization-specificity'
base_output_dir = 'data/EDAPIBench'

def remove_comments(code_line):
    line = code_line.split("#", 1)[0]
    
    return line.rstrip()

def extract_indent(code_line):
    indent = ''
    for char in code_line:
        if char in (' ', '\t'):
            indent += char
        else:
            break 
    return indent

if __name__ == '__main__':
    for llm in LLMS:
        all_data = []
        for lib in LIBS:
            if not os.path.exists(os.path.join(base_input_dir, lib, llm, 'data.json')):
                continue
            with open(os.path.join(base_input_dir, lib, llm, 'data.json'), 'r') as f:
                data = json.load(f)
            for line in data:
                line.update({'library': lib})
                func = line['function']
                ref = line['reference']
                rephrase_ref = line['rephrase_reference']
                code_lines = func.splitlines()
                for code_line in code_lines:
                    if ref in code_line:
                        new_ref = remove_comments(code_line)
                        line.update({'reference': new_ref})
                        indent = extract_indent(new_ref)
                        line.update({'rephrase_reference': indent + rephrase_ref.strip()})
                        break
                all_data.append(line)
        os.makedirs(os.path.join(base_output_dir, llm), exist_ok=True)
        with open(os.path.join(base_output_dir, llm, 'all.json'), 'w') as f:
            json.dump(all_data, f, indent=4)
