from openai import OpenAI
import os
import json
import sys
import time
import re
from tqdm import tqdm
sys.path.append(os.getcwd())
from utils import prompts

def llm_inference(prompt, model, base_url, api_key, temperature, sample_num, max_token=1024, timeout=30):
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    new_messages = [{
        "role": "user",
        "content": prompt,
    }]
    response_text = None
    try:
        response = client.chat.completions.create(
            model=model,
            messages=new_messages,
            max_tokens=max_token,
            timeout=timeout,
            n=sample_num,
            temperature=temperature,
        )
        if 'error' in response.model_extra.keys():
            print(response.model_extra['error']['message'])
        else:
            response_text = [response.choices[i].message.content for i in range(len(response.choices))]
            print(f'{model} response received')
    except Exception as e:
        print(e)
    
    return [extract_python_code(text) for text in response_text]

def extract_python_code(text):
    pattern = r'```python\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        extracted = '\n'.join(matches)
    else:
        extracted = text
    cleaned = re.sub(r'^```python\n', '', extracted, flags=re.MULTILINE)
    return cleaned

if __name__ == '__main__':
    api_key = 'your_api_key'
    base_url = 'llm_api_url'
    # if model cant complete generalization data with deprecated API, try to use prompts.easy_data_augment_prompt
    augment_prompt_template = prompts.data_augment_prompt
    model = 'model_name'
    
    with open('data/all_functions_inputs/all_inputs.json', 'r') as f:
        data = json.load(f)
    os.makedirs("data/specificity-generalization-data", exist_ok=True)
    results = {}
    if not os.path.exists('data/specificity-generalization-data/generalization_data_raw.json'):
        for line in data:
            results[line] = []
    else:
        for line in data:
            results[line] = []
        with open('data/specificity-generalization-data/generalization_data_raw.json', 'r') as f:
            _results = json.load(f)
            for k, v in _results.items():
                results[k] = v
    
    for probing_input in tqdm(data):
        if not results[probing_input]:
            rephrase_prompt = llm_inference(
                augment_prompt_template.format(code_snippet=probing_input),
                model,
                base_url,
                api_key,
                temperature=1.3,
                sample_num=5
            )
            if rephrase_prompt is not None:
                results[probing_input] = rephrase_prompt
            with open('data/specificity-generalization-data/generalization_data_raw.json', 'w') as f:
                json.dump(results, f, indent=4)