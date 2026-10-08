import os
import json
import sys
import re
sys.path.append(os.getcwd())
from utils import MODEL_FACTORY, CompletionEngine
from utils.source_utils import *
from tqdm import tqdm

base_input_dir = 'data/predicted-dapi-results'
base_output_dir = 'data/benchmark-generalization'
libs = ['pandas', 'pytorch', 'scipy', 'seaborn', 'sklearn', 'tensorflow', 'transformers']
llms = ["deepseek-1.3b", "starcoder-3b", "qwencoder-3b"]

def seperate_input_refence(code):
    lines = code.rstrip('\n').split('\n')
    probing_input = '\n'.join(lines[:-1])
    reference = lines[-1]
    return probing_input, reference

def extract_api(s):
    match = re.search(r'([\w\.]+)\(', s)
    return match.group(1) if match else s

def genrlization_data_reference_dict_generation(orin_reference, rephrase_reference):
    rephrase_reference = extract_api(clean_pred(rephrase_reference))
    reference = extract_api(clean_pred(orin_reference))
    rephrase_reference_parts = rephrase_reference.split('.')
    reference_parts = reference.split('.')
    
    differences = []
    max_len = max(len(rephrase_reference_parts), len(reference_parts))
    
    for i in range(max_len):
        part1 = reference_parts[i] if i < len(reference_parts) else None
        part2 = rephrase_reference_parts[i] if i < len(rephrase_reference_parts) else None
        
        if part1 != part2:
            differences.append((part1, part2))
    
    reference_dict = line['reference dict']
    rephrase_reference_dict = {}
    for diff_pair in differences:
        if diff_pair[0] in reference_dict.keys():
            rephrase_reference_dict.update({diff_pair[1]: reference_dict[diff_pair[0]]})
    
    return rephrase_reference_dict

def check_availability(list1, list2):
    for s1 in list1:
        for s2 in list2:
            if s1 in s2:
                return True
    return False

if __name__ == '__main__':
    with open('data/specificity-generalization-data/generalization_data_raw.json', 'r') as f:
        rephrase_results = json.load(f)
    
    for lib in libs:
        for llm in llms:
            model, tokenizer = MODEL_FACTORY[llm]()
            engine = CompletionEngine(model, tokenizer)
            with open(os.path.join(base_input_dir, lib, llm, 'data.json'), 'r') as f:
                data = json.load(f)
            for line in tqdm(data):
                orin_data = line['probing input'] + line['reference']
                if orin_data in rephrase_results.keys():
                    print(f'updating {line["case-id"]} generlization data....')
                    candidate_list = rephrase_results[orin_data]
                    rephrase_inputs = []
                    rephrase_references = []
                    rephrase_reference_dicts = []
                    if not isinstance(candidate_list, list):
                        line.update({'rephrase': line['probing input'], 'rephrase_reference_dict': line["reference dict"], 'rephrase_reference': line['reference']})
                        continue
                    for rephrase_result in candidate_list:
                        r_inputs, r_reference = seperate_input_refence(rephrase_result)
                        rephrase_inputs.append(r_inputs)
                        rephrase_references.append(r_reference)
                        rephrase_reference_dicts.append(genrlization_data_reference_dict_generation(line['reference'], r_reference))
                    
                    for i in range(len(rephrase_inputs)):
                        preds = engine.complete(
                            [rephrase_inputs[i]],
                            max_len=50,
                            beam_size=1,
                            cand_num=1
                        )
                        valid_index = -1
                        _preds = preds[0]
                        _preds = [clean_pred(p) for p in _preds]
                        _preds = [extract_first_statement(p, False) for p in _preds]
                        ref_dict = line["reference dict"]
                        ref_dict.update(rephrase_reference_dicts[i])
                        _api_preds = [extract_apis_in_first_stmt(p, ref_dict, line["alias dict"]) for p in _preds]
                        _apis = []
                        for s in _api_preds:
                            for _s in s:
                                api_last_part = _s.split('.')[-1]
                                _apis.append(api_last_part)
                        if _apis == []:
                            continue
                        elif check_availability(_apis, line['deprecated api']):
                            line.update({'rephrase': rephrase_inputs[i], 'rephrase_reference_dict': rephrase_reference_dicts[i], 'rephrase_reference': rephrase_references[i]})
                            print(f'{line["case-id"]} generlization data update complete!')
                            valid_index = i
                            break
                    if valid_index == -1:
                        print(f'{line["case-id"]} update fail.')
                        line.update({'rephrase': rephrase_inputs[0], 'rephrase_reference_dict': rephrase_reference_dicts[0], 'rephrase_reference': rephrase_references[0]})
            os.makedirs(os.path.join(base_output_dir, lib, llm), exist_ok=True)
            with open(os.path.join(base_output_dir, lib, llm, 'data.json'), 'w') as f:
                json.dump(data, f, indent=4)
            del model, tokenizer, engine