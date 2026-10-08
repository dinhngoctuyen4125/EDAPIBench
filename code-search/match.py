import sys
import os
import math
import json
import random
import time
from collections import defaultdict
from tqdm import tqdm
import re
from pathlib import Path
from multiprocessing import Pool
import ast
import traceback
sys.path.append(os.getcwd())
from utils import APIMatcher

def handle_error(exception):
    traceback.print_exc()

def process_one(filename, apis):
    """Process API matching for a single file"""
    try:
        with open(filename, encoding='utf-8') as f:
            code = f.read()
        macher = APIMatcher(code, apis, maxline=1000)
        results = macher.match().matched_funcs
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        traceback.print_exc()
        return None
    json_data = {
        "filename": filename,
        "source": code
    }
    if not results:
        return None
    json_data["matching"] = results
    return json_data

def process_batch(batch_id, batch, output_path, apis):
    """Process API matching for a batch of files"""
    with Path(output_path).open("w", encoding="utf-8") as out:
        for filename in tqdm(batch, desc=f"batch-{batch_id}", ascii=True):
            json_data = process_one(filename, apis)
            if json_data is None:
                continue
            out.write(f"{json.dumps(json_data, ensure_ascii=False)}\n")
            out.flush()

# LIBs = [
    # "tensorflow",
    # "pytorch",
    # "numpy",
    # "pandas",
    # "scipy",
    # "sklearn",
    # "seaborn",
    # "transformers"
# ]
LIBs = ["numpy"]  # NumPy pilot run.
N_THREADS = 12  # Number of threads

if __name__ == '__main__':
    file_count = 0
    for lib in LIBs:
        MAPPINGS_FILE = Path(f"data/mappings/deprecated-mappings-{lib}.json")
        SEARCH_RESULTS_DIR = Path(f"data/searching-results/{lib}/")
        SOURCE_DIR = SEARCH_RESULTS_DIR / "sources"
        OUTPUT_DIR = Path(f"data/matching-results/{lib}/")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if not MAPPINGS_FILE.exists():
            print(f"Warning: Mapping file {MAPPINGS_FILE} not found, skipping library {lib}")
            continue
        with MAPPINGS_FILE.open("r", encoding="utf-8") as f:
            mappings = json.load(f)
        
        for dep, rep in list(mappings.items()):
            mappings[dep.replace("tensorflow.compat.v1.", "tensorflow.")] = rep.replace("tensorflow.compat.v1.", "tensorflow.")
        
        apis = set()
        for dep, rep in mappings.items():
            apis.add(rep)
        print(f"Library {lib} loaded {len(apis)} APIs to match")

        filenames = []
        if SOURCE_DIR.exists():
            filenames = [str(p) for p in SOURCE_DIR.rglob("*.py") if p.is_file()]
            print(f"Found {len(filenames)} Python files in {SOURCE_DIR}")
        else:
            print(f"Warning: Source code directory {SOURCE_DIR} does not exist. Search script may not have been executed.")

        random.shuffle(filenames)
        file_count += len(filenames)

        if not filenames:
            continue
        
        batch_files = []  # Merge only batches created in this run.
        with Pool(N_THREADS) as pool:
            batch_size = math.ceil(len(filenames) / N_THREADS)
            ranges = [
                (i * batch_size, min((i + 1) * batch_size, len(filenames)))
                for i in range(N_THREADS)
            ]

            for idx, (beg, end) in enumerate(ranges):
                if beg >= end:
                    continue  # Skip empty batches
                batch = filenames[beg:end]
                output_file = OUTPUT_DIR / f"matching-batch-{idx+1}.jsonl"
                batch_files.append(output_file)
                pool.apply_async(
                    process_batch,
                    args=(idx+1, batch, str(output_file), apis),
                    error_callback=handle_error
                )
            pool.close()
            pool.join()

        samples = []
        visited_funcs = set()
        api2count = defaultdict(int)
        
        # for batch_file in OUTPUT_DIR.glob("matching-batch-*.jsonl"):
        for batch_file in batch_files:
            if not batch_file.exists():
                continue
            with batch_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json_data = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"Parsing error: line {line} in {batch_file}")
                        continue
                    if "matching" not in json_data:
                        continue
                    for item in json_data["matching"]:
                        func_key = item.get("function")
                        if not func_key or func_key in visited_funcs:
                            continue
                        visited_funcs.add(func_key)
                        samples.append(item)
                        api = item["matched call"].get("matched api")
                        if api:
                            api2count[api] += 1

        api2count = dict(sorted(api2count.items(), key=lambda x: x[1], reverse=True))
        print(f"Library {lib} matched {len(samples)} unique functions")

        with (OUTPUT_DIR / "matched-functions.json").open("w", encoding="utf-8") as f:
            json.dump(samples, f, indent=4, ensure_ascii=False)
        
        # Save matched API statistics
        with (OUTPUT_DIR / "matched-apis.json").open("w", encoding="utf-8") as f:
            json.dump(api2count, f, indent=4, ensure_ascii=False)

    print(f"Total files processed: {file_count}")