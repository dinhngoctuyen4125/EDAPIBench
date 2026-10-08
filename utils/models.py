from typing import List, Union
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers import AutoModelForCausalLM, PreTrainedModel, LlamaForCausalLM
from peft import PeftModelForCausalLM
import os

def init_deepseek1b(model_path="deepseek-ai/deepseek-coder-1.3b-base", device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def init_starcoder3b(model_path="bigcode/starcoder2-3b", device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def init_qwencoder3b(model_path="Qwen/Qwen2.5-Coder-3B", device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


MODEL_FACTORY = {
    "deepseek-1.3b": init_deepseek1b,
    "starcoder-3b": init_starcoder3b,
    "qwencoder-3b": init_qwencoder3b,
}


class CompletionEngine:
    def __init__(
        self,
        model:Union[PreTrainedModel, PeftModelForCausalLM],
        tokenizer:PreTrainedTokenizerBase,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
    def complete(
        self,
        inputs:List[str],
        max_len=30,
        beam_size=1,
        cand_num=1,
        do_sample=False,
        temperature=1.0,
        top_k=None,
        top_p=None,
    ):
        with torch.no_grad():
            try:
                token_ids = self.tokenizer(inputs, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
                token_ids = token_ids.to(self.model.device)
                output_ids = self.model.generate(
                    inputs=token_ids,
                    attention_mask=token_ids.ne(self.tokenizer.pad_token_id),
                    max_new_tokens = max_len,
                    num_beams = beam_size,
                    num_return_sequences = cand_num,
                    do_sample = do_sample,
                    temperature = temperature,
                    top_k = top_k,
                    top_p = top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            except Exception as e:
                print(f"An error occur: {str(e)}")
                return None
        generations = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
        generations = [[gen[len(ipt):] for gen in generations[i*cand_num:i*cand_num+cand_num]] for i, ipt in enumerate(inputs)]
        torch.cuda.empty_cache()
        
        return generations
