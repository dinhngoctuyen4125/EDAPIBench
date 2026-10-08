import json
import os
from pathlib import Path

# LIBS = [
    # "tensorflow",
    # "pytorch",
    # "numpy",
    # "pandas",
    # "scipy",
    # "sklearn",
    # "seaborn",
    # "transformers"
# ]
LIBS = ["numpy"]  # NumPy pilot run.
MAX_SAMPLES = 10  # Total standardized samples across all NumPy replacement APIs.

ALIAS = {
    "tensorflow": ["tf"],
    "numpy": ["np"],
    "pandas": ["pd"],
    "pandas.DataFrame": ["df"],
    "sklearn": ["sk"],
    "scipy.stats": ['sts','st'],
    "scipy.special":['sps', 'sp'],
    "scipy.linalg":['spl'],
    "scipy": ["sc", "sp"],
    "seaborn": ["sn", "sns", "sb"],
}

def splitlines_no_ff(code):
    """Split code into lines while handling different line endings"""
    idx = 0
    lines = []
    next_line = ''
    while idx < len(code):
        c = code[idx]
        next_line += c
        idx += 1
        if c == '\r' and idx < len(code) and code[idx] == '\n':
            next_line += '\n'
            idx += 1
        if c in '\r\n':
            lines.append(next_line)
            next_line = ''
    if next_line:
        lines.append(next_line)
    
    return lines

def get_source_by_lineno(code, lineno):
    """Extract specific line from code (0-based index)"""
    lines = splitlines_no_ff(code)
    # Handle potential index errors
    if 0 <= lineno < len(lines):
        return lines[lineno].strip()
    return ""
    
def get_source_by_position(code, start_lineno, end_lineno):
    lines = splitlines_no_ff(code)
    # Ensure valid line range
    start = max(0, start_lineno)
    end = min(len(lines) - 1, end_lineno)
    
    if end == start:
        return lines[start] if start < len(lines) else ""
        
    return ''.join(lines[start:end + 1])

def fix_alias_deprecated_dict(sample: dict):
    """
    Deprecated apis should be listed in alias dict to avoid miss match deprecated prediction.
    """
    deprecated_apis = sample['deprecated api']
    replacement_api = sample['replacement api']
    r_name_list = replacement_api.split('.')
    
    for d_api in deprecated_apis:
        alias_dict_appended = []
        diff_idx = -1
        d_name_list = d_api.split('.')
        for i, (d_name, r_name) in enumerate(zip(d_name_list, r_name_list)):
            if d_name != r_name:
                diff_idx = i
                break
        if diff_idx == -1:
            diff_idx = min(len(d_name_list), len(r_name_list))
        prefix = r_name_list[:diff_idx]
        
        for i in range(len(prefix)):
            prefix_cur = prefix[len(prefix) - i - 1:]
            tmp = ".".join(prefix_cur + d_name_list[diff_idx:])
            alias_dict_appended.append(tmp)
            for k, v in ALIAS.items():
                if k in tmp:
                    for ala in v:
                        alias_dict_appended.append(tmp.replace(k, ala))
            
            tmp = ".".join(d_name_list[diff_idx:])
            alias_dict_appended.append(tmp)
            for k, v in ALIAS.items():
                if k in tmp:
                    for ala in v:
                        alias_dict_appended.append(tmp.replace(k, ala))
        
        for ad in alias_dict_appended:
            sample['alias dict'].update({ad: d_api})

def process_library(lib):
    match_dir = Path(f"data/matching-results/{lib}")
    output_dir = Path(f"data/standardized-results/{lib}")
    mappings_file = Path(f"data/mappings/deprecated-mappings-{lib}.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    matched_file = match_dir / 'matched-functions.json'
    if not matched_file.exists():
        print(f"Warning: Matched functions file not found for {lib} at {matched_file} - skipping")
        return 0
        
    if not mappings_file.exists():
        print(f"Warning: Mappings file not found for {lib} at {mappings_file} - skipping")
        return 0

    try:
        with matched_file.open('r', encoding="utf-8") as f:
            matched_data = json.load(f)
        
        with mappings_file.open('r', encoding="utf-8") as f:
            mapping_data = json.load(f)
    except Exception as e:
        print(f"Error loading data for {lib}: {str(e)} - skipping")
        return 0
        
    standardized_data = []
    for entry in matched_data:
        tmp_data = {}
        
        tmp_data['function'] = entry.get('function', '')
        tmp_data['reference dict'] = entry.get('reference dict', {})
        tmp_data['alias dict'] = entry.get('alias dict', {})
        function_code = entry.get('function', '')
        call_position = entry.get('matched call', {}).get('position', [0, 0])
        # An API call on the first line leaves no completion prefix.
        if not 0 < call_position[0] < len(splitlines_no_ff(function_code)):
            continue
        
        tmp_data['probing input'] = get_source_by_position(
            function_code, 
            0, 
            max(0, call_position[0] - 1)
        )
        
        matched_api = entry.get('matched call', {}).get('matched api', '')
        tmp_data['deprecated api'] = [
            k for k, v in mapping_data.items() 
            if v == matched_api
        ]
        
        tmp_data['replacement api'] = matched_api
        tmp_data['expected call'] = entry.get('matched call', {}).get('call name', '')
        tmp_data['category'] = "up-to-dated"
        tmp_data['reference'] = get_source_by_lineno(
            function_code, 
            min(call_position[0], len(splitlines_no_ff(function_code)) - 1)  # Prevent index error
        )
        if not (tmp_data['probing input'].strip()
                and tmp_data['reference']
                and tmp_data['deprecated api']):
            continue
        fix_alias_deprecated_dict(tmp_data)
        
        standardized_data.append(tmp_data)
        if len(standardized_data) >= MAX_SAMPLES:
            break

    output_file = output_dir / 'standardized_samples.json'
    with output_file.open('w', encoding="utf-8") as f:
        json.dump(standardized_data, f, indent=4, ensure_ascii=False)
    
    processed_count = len(standardized_data)
    # print(f'Library {lib} processed {processed_count} samples')
    print(f'Library {lib} processed {processed_count}/{MAX_SAMPLES} samples')
    if processed_count < MAX_SAMPLES:
        print('Not enough valid samples. Increase MAX_COUNT in search.py and rerun Step 1.')
    return processed_count

if __name__ == '__main__':
    for lib in LIBS:
        total_processed = process_library(lib)
        print(f'Processing {lib} complete. Total samples: {total_processed}\n')