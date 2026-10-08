import os
import boto3
import threading
import queue
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from smart_open import open
from typing import List, Dict, Set, Tuple, Optional
import ast
import json
from pathlib import Path
import random

def match_funcs(file_content: str) -> List[str]:
    functions = []
    try:
        tree = ast.parse(file_content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(extract_code(file_content, node))
            elif isinstance(node, ast.AsyncFunctionDef):
                functions.append(extract_code(file_content, node))
    
    except Exception as e:
        print(f"AST parsing error: {e}")
        return None
    
    return functions

def extract_code(source: str, node: ast.AST) -> str:
    start_line = node.lineno - 1
    end_line = node.end_lineno - 1 if hasattr(node, 'end_lineno') else start_line
    
    start_col = node.col_offset if hasattr(node, 'col_offset') else 0
    end_col = node.end_col_offset if hasattr(node, 'end_col_offset') else None
    
    lines = source.splitlines()
    
    if start_line == end_line:
        function_code = lines[start_line][start_col:end_col]
    else:
        function_code = lines[start_line][start_col:]
        for i in range(start_line + 1, end_line):
            function_code += '\n' + lines[i]
        if end_col is not None:
            function_code += '\n' + lines[end_line][:end_col]
        else:
            function_code += '\n' + lines[end_line]
    
    return function_code

class TopKSimilarItems:
    def __init__(self, k: int = 5):
        self.k = k
        self.items = []
    
    def add(self, similarity: float, item: str):
        if len(self.items) < self.k or similarity > self.items[0][0]:
            self.items.append((similarity, item))
            self.items.sort(key=lambda x: x[0])
            if len(self.items) > self.k:
                self.items = self.items[-self.k:]
    
    def get_top_k(self) -> List[Tuple[float, str]]:
        return sorted(self.items, key=lambda x: x[0], reverse=True)
    
    def early_stop(self, min_threshold: float) -> bool:
        return len(self.items) == self.k and self.items[0][0] >= min_threshold

class CodeSimilarityFinder:
    def __init__(self, code_snippets: List[str], max_files: int = 100000, 
                 batch_size: int = 32, num_workers: int = 4):
        self.code_snippets = code_snippets
        self.max_files = max_files
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.results = {i: TopKSimilarItems(k=5) for i in range(len(code_snippets))}
        
        self.query_embeddings = self.encode_batch(code_snippets)
        
        self.queue = queue.Queue(maxsize=100)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.processed_files = 0
    
    def encode_batch(self, code_list: List[str]) -> np.ndarray:
        all_embeddings = []
        
        total_batches = (len(code_list) + self.batch_size - 1) // self.batch_size
        
        for i in range(total_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(code_list))
            batch = code_list[start_idx:end_idx]
            
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, 
                                truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)
    
    def euclidean_similarity(self, query_embedding: np.ndarray, 
                            code_embeddings: np.ndarray) -> np.ndarray:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        distances = np.linalg.norm(code_embeddings - query_embedding, axis=1)
        similarities = 1 / (1 + distances)
        
        return similarities
    
    def process_file(self, file_content: str):
        functions = match_funcs(file_content)
        
        if not functions:
            return
        
        func_embeddings = self.encode_batch(functions)
        
        for i, query_embedding in enumerate(self.query_embeddings):
            similarities = self.euclidean_similarity(query_embedding, func_embeddings)
            
            for sim, func in zip(similarities, functions):
                if sim < 0.5:
                    with self.lock:
                        self.results[i].add(sim, func)
    
    def worker(self):
        while not self.stop_event.is_set():
            try:
                file_content = self.queue.get(timeout=1)
                self.process_file(file_content)
                self.queue.task_done()
                
                with self.lock:
                    self.processed_files += 1
                    if self.processed_files % 1000 == 0:
                        print(f"Processed files: {self.processed_files}/{self.max_files}")
                    
                    if self.processed_files >= self.max_files:
                        self.stop_event.set()
                        break
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing file: {e}")
                self.queue.task_done()
    
    def download_contents(self, blob_id, src_encoding):
        s3_url = f"s3://softwareheritage/content/{blob_id}"
        
        try:
            with open(s3_url, "rb", compression=".gz", 
                     transport_params={"client": self.s3_client}) as fin:
                content = fin.read().decode(src_encoding)
            return content
        except Exception as e:
            print(f"Error downloading file {blob_id}: {e}")
            return None
    
    def find_similar_code(self):
        session = boto3.Session(
            aws_access_key_id='your_aws_key_id',
            aws_secret_access_key='your_access_key'
        )
        self.s3_client = session.client("s3")
        
        ds = load_dataset("bigcode/the-stack-v2", "Python", split="train", streaming=True)
        
        threads = []
        for _ in range(self.num_workers):
            t = threading.Thread(target=self.worker)
            t.daemon = True
            t.start()
            threads.append(t)
        
        for i, row in enumerate(ds):
            if self.stop_event.is_set() or self.processed_files >= self.max_files:
                break
            
            content = self.download_contents(row["blob_id"], row["src_encoding"])
            if content:
                self.queue.put(content)
        
        self.queue.join()
        self.stop_event.set()
        
        for t in threads:
            t.join()
        
        final_results = {}
        for i, top_k in self.results.items():
            final_results[i] = top_k.get_top_k()
        
        return final_results
    
    def extract_function_names(self, code: str) -> Set[str]:
        try:
            tree = ast.parse(code)
            return {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        except SyntaxError:
            return set()

def main():
    with open('data/all_functions_inputs/all_functions.json', 'r') as f:
        code_snippets = json.load(f)
    
    similarity_finder = CodeSimilarityFinder(
        code_snippets=code_snippets,
        max_files=100000,
        batch_size=32,
        num_workers=8
    )
    
    results = similarity_finder.find_similar_code()
    output_path = Path("data/specificity-generalization-data/specificity_data_raw.json")
    os.makedirs("data/specificity-generalization-data", exist_ok=True)
    results_with_code = {}
    for i, snippets in results.items():
        query_code = code_snippets[i]
        similar_codes = [{"similarity": float(sim), "code": code} for sim, code in snippets]
        results_with_code[query_code] = similar_codes
    
    with open(output_path, 'w') as f:
        json.dump(results_with_code, f, indent=2)
    
    print(f"Specificity raw data saved to {output_path}")
    
if __name__ == "__main__":
    main()