import sys
import json
import logging
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import traceback
import argparse
import re
from fuzzywuzzy import fuzz
import os
sys.path.append(os.getcwd())
from utils import MODEL_FACTORY, CompletionEngine
from utils import init_log
from utils.source_utils import *
import tiktoken

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='model name', required=True, type=str)
parser.add_argument('--lib', help='library name', required=True, type=str)
parser.add_argument('--maxlen', help='max length', required=False, type=int, default=50)
parser.add_argument('--beam', help='beam size', required=False, type=int, default=1)
parser.add_argument('--batch', help='batch size', required=False, type=int, default=16)
parser.add_argument('--gpu', help='gpu index', required=False, type=str, default='0')
parser.add_argument('--max_input_token', required=False, type=int, default=1100)
parser.add_argument('--resume', required=False, type=int, default=0)

if __name__ == "__main__":
    args = parser.parse_args()
    MODEL = args.model
    LIB = args.lib
    MAX_LEN = args.maxlen
    BEAM = args.beam
    BATCH_SIZE = args.batch
    GPU_INDEX = args.gpu
    MAX_INPUT_TOKEN = args.max_input_token
    RESUME = args.resume
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX

    SAMPLES_FILE = f"data/standardized-results/{LIB}/standardized_samples.json"
    OUTPUT_DIR = f"data/dapi-inference-results/{LIB}"

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    init_log(f"{OUTPUT_DIR}/{MODEL}/predictions-linelevel-maxlen{MAX_LEN}-beam{BEAM}.log")
    with Path(SAMPLES_FILE).open("r") as f:
        samples = json.load(f)
    model, tokenizer = MODEL_FACTORY[MODEL]()
    engine = CompletionEngine(model, tokenizer)
    encoding = tiktoken.encoding_for_model("gpt-4o")
    results = []
    if RESUME > 0:
        with open(f"{OUTPUT_DIR}/{MODEL}/predictions-linelevel-maxlen{MAX_LEN}-beam{BEAM}-batch{BATCH_SIZE}-resume{RESUME}.json", "r") as f:
            results = json.load(f)
    prediction_info = {
        "outdated total": 0,
        "outdated llm-good": 0,
        "outdated llm-bad": 0,
        "outdated llm-other": 0,
        "up-to-dated total": 0,
        "up-to-dated llm-good": 0,
        "up-to-dated llm-bad": 0,
        "up-to-dated llm-other": 0,
    }
    cur_epoch = 0
    save_iter = 10
    for beg, end in tqdm(list(zip(range(0, len(samples), BATCH_SIZE), range(BATCH_SIZE, len(samples) + BATCH_SIZE, BATCH_SIZE)))):
        cur_epoch += 1
        if RESUME > 0 and cur_epoch < RESUME:
            continue
        batch = samples[beg:end]
        inputs = [item["probing input"] for item in batch
                  if len(encoding.encode(item["probing input"], allowed_special="all")) <= 1100]
        preds = engine.complete(
            inputs,
            max_len=MAX_LEN,
            beam_size=BEAM,
            cand_num=BEAM
        )
        if preds == None:
            continue
        for item, _preds in zip(batch, preds):
            _preds = [clean_pred(p) for p in _preds]
            _preds = [extract_first_func(item["probing input"] + p)[len(item["probing input"]):] for p in _preds]
            _api_preds = [extract_apis_in_first_stmt(p, item["reference dict"], item["alias dict"]) for p in _preds]
            
            item["probing predictions"] = list(zip(_preds, _api_preds))
            
            logging.info("#" * 40)
            logging.info(f"api mapping: {item['deprecated api']} -> {item['replacement api']}")
            logging.info("")
            logging.info("")
            logging.info(f"function:\n{item['function']}")
            logging.info(f"probing input:\n{item['probing input']} ")
            logging.info(f"predictions:\n{_preds}")
            logging.info(f"stmt predictions:\n{[extract_first_statement(p, False) for p in _preds]}")
            logging.info(f"api predictions:\n{_api_preds}")
            prediction_info[f"{item['category']} total"] += 1
            _apis = set()
            for s in _api_preds:
                _apis.update(s)
            if len(set(item['deprecated api']) & _apis) > 0:
                logging.info(f"OH NO! The model predicts the deprecated API `{set(item['deprecated api']) & _apis}` for {item['category']} function!")
                prediction_info[f"{item['category']} llm-bad"] += 1
            elif item['replacement api'] in _apis:
                logging.info(f"WOW! The model predicts the replacement API `{item['replacement api']}` for {item['category']} function!")
                prediction_info[f"{item['category']} llm-good"] += 1
            else:
                prediction_info[f"{item['category']} llm-other"] += 1
            logging.info("")
            logging.info("")
            logging.info("")
            logging.info("")
        results.extend(batch)
        
        if cur_epoch % save_iter == 0:
            with open(f"{OUTPUT_DIR}/{MODEL}/predictions-linelevel-maxlen{MAX_LEN}-beam{BEAM}-batch{BATCH_SIZE}-resume{cur_epoch}.json", "w") as f:
                json.dump(results, f, indent=4)
            if os.path.exists(f"{OUTPUT_DIR}/{MODEL}/predictions-linelevel-maxlen{MAX_LEN}-beam{BEAM}-batch{BATCH_SIZE}-resume{cur_epoch-save_iter}.json"):
                os.remove(f"{OUTPUT_DIR}/{MODEL}/predictions-linelevel-maxlen{MAX_LEN}-beam{BEAM}-batch{BATCH_SIZE}-resume{cur_epoch-save_iter}.json")
                
    with Path(f"{OUTPUT_DIR}/{MODEL}/predictions-linelevel-maxlen{MAX_LEN}-beam{BEAM}.json").open("w") as f:
        json.dump(results, f, indent=4)
        
    logging.info("##### API PROBING STATISTICS #####")
    logging.info(f"\n{json.dumps(prediction_info, indent=4)}")