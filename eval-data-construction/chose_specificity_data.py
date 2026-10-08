import json
import os
import random
from typing import List, Dict, Any, Set
import ast
import sys
from tqdm import tqdm
sys.path.append(os.getcwd())
from utils.source_utils import *
from utils import MODEL_FACTORY, CompletionEngine

def load_similarity_data():
    with open('data/specificity-generalization-data/specificity_data_raw.json', 'r') as f:
        similarity_data = json.load(f)
    
    function_to_similar = {}
    for query_code, sim_codes in similarity_data.items():
        similar_codes = [entry['code'] for entry in sim_codes]
        function_to_similar[query_code] = similar_codes
    
    return function_to_similar

def get_probing_ratio(sample: Dict[str, Any]) -> float:
    function_lines = len(sample["function"].split('\n'))
    probing_lines = len(sample["probing input"].split('\n'))
    
    if function_lines == 0:
        return 0.5
    return min(1.0, probing_lines / function_lines)

def is_valid_line(line, pred_api, invalid_apis):
    if any(api in line for api in invalid_apis):
        return False
    if pred_api == []:
        return False
    return True

def line_inference(code, ref_dict, alias_dict, model_engine):
    answer = model_engine.complete(
        [code],
        max_len=50,
        beam_size=1,
        cand_num=1,
    )
    if answer is None:
        return "", []
    pred = answer[0][0]
    pred = extract_first_func(code + clean_pred(pred))[len(code):]
    pred = extract_first_statement(pred, False)
    pred_apis = extract_apis_in_first_stmt(pred, ref_dict, alias_dict)
    
    return pred, pred_apis

def crop_code(code, ratio, invalid_apis, ref_dict, alias_dict, model_engine):
    lines = code.split('\n')
    code_lines_num = len(lines)
    target_lines_num = max(1, int(round(code_lines_num * ratio)))
    
    next_line_idx = target_lines_num if target_lines_num < code_lines_num else target_lines_num - 1
    next_line = lines[next_line_idx]
    input_lines = '\n'.join(lines[:next_line_idx])
    pred, pred_apis = line_inference(input_lines, ref_dict, alias_dict, model_engine)
    
    if not is_valid_line(next_line, pred_apis, invalid_apis):
        max_try = 0
        for i in range(next_line_idx, code_lines_num):
            max_try += 1
            if max_try > 3:
                break
            next_line = lines[i]
            input_lines = '\n'.join(lines[:i])
            pred, pred_apis = line_inference(input_lines, ref_dict, alias_dict, model_engine)
            if is_valid_line(next_line, pred_apis, invalid_apis):
                return input_lines, pred, pred_apis
        
        max_try = 0
        for i in range(next_line_idx - 1, -1, -1):
            max_try += 1
            if max_try > 3:
                break
            next_line = lines[i]
            input_lines = '\n'.join(lines[:i])
            pred, pred_apis = line_inference(input_lines, ref_dict, alias_dict, model_engine)
            if is_valid_line(next_line, pred_apis, invalid_apis):
                return input_lines, pred, pred_apis

    return input_lines, pred, pred_apis

def main(lib, llm):
    base_dir = 'data/benchmark-generalization'
    similarity_map = load_similarity_data()
    print(f"Processing {os.path.join(base_dir, lib, llm, 'data.json')}")
    
    output_file = os.path.join('data/benchmark-generalization-specificity', lib, llm, 'data.json')
    if os.path.exists(output_file):
        return
    
    model, tokenizer = MODEL_FACTORY[llm]()
    engine = CompletionEngine(model, tokenizer)
    
    with open(os.path.join(base_dir, lib, llm, 'data.json'), 'r') as f:
        data = json.load(f)
    
    for sample in tqdm(data):
        function_code = sample["function"]
        probing_input = sample["probing input"]
        deprecated_api = sample["deprecated api"]
        replacement_api = sample["replacement api"]
        expected_call = sample["expected call"]
        ref_dict = sample['reference dict']
        alias_dict = sample['alias dict']
        
        apis_to_check = deprecated_api + [replacement_api] + [expected_call]
        ratio = get_probing_ratio(sample)
        similar_codes = similarity_map.get(function_code, [])
        
        if similar_codes == []:
            print(f'{function_code} not found!')
        
        cropped_similar_context = []
        for code in similar_codes[:5]:
            input_lines, pred, pred_apis = crop_code(code, ratio, apis_to_check, ref_dict, alias_dict, engine)
            cropped_similar_context.append({'probing input': input_lines, 'prediction': pred, 'pred-api': pred_apis})
        
        sample["Specificity-SimilarContext"] = cropped_similar_context
    
    for line in data:
        specificity_data = line["Specificity-SimilarContext"]
        tmp = {}
        for i in specificity_data:
            if i['probing input'] != "":
                tmp.update({"probing input": i['probing input'], "prediction": i["prediction"], "pred-api": i["pred-api"]})
        for i in specificity_data:
            if i['probing input'] == "":
                i.update(tmp)
    
    out_path = os.path.dirname(output_file)
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Updated {len(data)} samples in {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model name', required=True, type=str)
    parser.add_argument('--lib', help='library name', required=True, type=str)
    args = parser.parse_args()
    main(args.lib, args.model)