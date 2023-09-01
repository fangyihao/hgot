'''
Created on Aug. 10, 2023

@author: Yihao Fang
'''
import torch
import os
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
#from transformers import T5ForConditionalGeneration
import functools
from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory, cache_turn_on
import time

def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(min(torch.cuda.mem_get_info())/1024**3)
    max_memory = f'{free_in_GB}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory

os.environ['TRANSFORMERS_CACHE'] = 'cache/huggingface/transformers'
AUTOAIS_MODEL = "google/t5_xxl_true_nli_mixture"

autoais_model = None
autoais_tokenizer = None

@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@CacheMemory.cache
def _run_nli(passage, claim):
    """
    Run inference for assessing AIS between a premise and hypothesis.
    Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
    """
    global autoais_model
    global autoais_tokenizer
    if autoais_model is None:
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto", offload_folder="cache/huggingface/transformers")
    if autoais_tokenizer is None:
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)
        
    input_text = "premise: {} hypothesis: {}".format(passage, claim)
    input_ids = autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(autoais_model.device)
    with torch.inference_mode():
        outputs = autoais_model.generate(input_ids, max_new_tokens=10)
    result = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
    inference = 1 if result == "1" else 0
    return inference


def nli_reranker(query, passages, retrieval_weight=0.5, nli_weight=0.5):

    passages_scores = []
    
    print("-"*35 + " NLI Passage, Claim, and Entailment " + "-"*35)
    for passage in passages:
        start = time.time()
        entail = _run_nli(passage.long_text, query)
        end  = time.time()
        print("Passage:", passage.long_text, " Claim:", query, " Entailment:", "True" if entail == 1 else "False", " Old Score:", passage.score, " New Score:", retrieval_weight * passage.score + nli_weight * entail, " Time:", end - start)
        
        passage.score = retrieval_weight * passage.score + nli_weight * entail
        passages_scores.append(passage.score)
        
    return passages_scores


def _run_nli_batch(passages, claims):
    global autoais_model
    global autoais_tokenizer
    if autoais_model is None:
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto", offload_folder="cache/huggingface/transformers")
    if autoais_tokenizer is None:
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)
    
    input_texts = ["premise: {} hypothesis: {}".format(passage, claim) for passage, claim in zip(passages, claims)]
    
    #old_padding_side = autoais_tokenizer.padding_side
    #print("old_padding_side:", old_padding_side)
    #old_pad_token = autoais_tokenizer.pad_token
    #print("old_pad_token:", old_pad_token)
    
    #autoais_tokenizer.padding_side = "left" 
    #autoais_tokenizer.pad_token = autoais_tokenizer.eos_token # to avoid an error
    inputs = autoais_tokenizer(input_texts, padding=True, return_tensors="pt").to(autoais_model.device)
    
    #autoais_tokenizer.padding_side = old_padding_side
    #autoais_tokenizer.pad_token = old_pad_token
    
    with torch.inference_mode():
        outputs = autoais_model.generate(input_ids=inputs["input_ids"],  attention_mask=inputs['attention_mask'], max_new_tokens=10)
    results = autoais_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    inferences = [(1 if result == "1" else 0) for result in results]
    return inferences

if __name__=='__main__':
    passage = "David is a god."
    claim = "David is a man."
    start = time.time()
    print(_run_nli(passage, claim))
    end  = time.time()
    print("Time:", end - start)
    
    passage = "David is really a man."
    claim = "David is really a man."
    start = time.time()
    print(_run_nli(passage, claim))
    end  = time.time()
    print("Time:", end - start)
    
    passages = ["David is a god.", "David is really a man."]
    claims = ["David is a man.", "David is really a man."]
    start = time.time()
    print(_run_nli_batch(passages, claims))
    end  = time.time()
    print("Time:", end - start)