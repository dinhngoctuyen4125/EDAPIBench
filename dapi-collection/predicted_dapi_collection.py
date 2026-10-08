import os
import json

if __name__ == '__main__':
    path = 'data/dapi-inference-results'
    LIBS = ["transformers", "tensorflow", "pytorch", "numpy", "pandas", "scipy", "sklearn", "seaborn"]
    model_list = ["deepseek-1.3b", "starcoder-3b", "qwencoder-3b"]
    output_path = 'data/predicted-dapi-results'
    
    for lib in LIBS:
        for model in model_list:
            results = []
            visited = set()
            probing_result_file = os.path.join(path, lib, model, 'predictions-linelevel-maxlen50-beam1.json')
            with open(probing_result_file, 'r') as f:
                data = json.load(f)
            
            for line in data:
                api = []
                if 'probing predictions' not in line.keys():
                    continue
                for pre in line['probing predictions']:
                    api += pre[1]
                api = set(api)
                # if deprecated api and not in visited, put it into results
                if len(set(line['deprecated api']) & api) > 0:
                    if line['probing input'] not in visited:
                        results.append(line)
                        visited.add(line['probing input'])
        
            out_path = os.path.join(output_path, lib, model)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            with open(os.path.join(out_path, 'data.json'), 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4)
            api_count = {}
            for line in results:
                api_count[line['replacement api']] = api_count.get(line['replacement api'], 0) + 1
            with open(os.path.join(out_path, 'api_count.json'), 'w') as f:
                json.dump(api_count, f, indent=4)
            print(f'LIB {lib} Model {model} contains {len(results)} samples.')
            print(api_count)