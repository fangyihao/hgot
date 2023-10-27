'''
Created on Sep. 6, 2023

@author: Yihao Fang
'''
import torch
import dsp
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

#os.environ['TRANSFORMERS_CACHE'] = 'cache/huggingface/transformers'
AUTOAIS_MODEL = "google/t5_xxl_true_nli_mixture"

autoais_model = None
autoais_tokenizer = None

@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@CacheMemory.cache
def _t5_nli(passage, claim):
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

def _t5_nli_logged(passage, claim):
    print("."*35 + " passage " + "."*35)
    print(passage)
    print("."*35 + " claim " + "."*35)
    print(claim)
    inference = _t5_nli(passage, claim)
    print("."*35 + " entailment (T5) " + "."*35)
    print("True" if inference == 1 else "False")
    
    return inference


def _t5_nli_batch(passages, claims):
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


Premise = dsp.Type(prefix="Premise:", desc="${the premise}")
Hypothesis = dsp.Type(prefix="Hypothesis:", desc="${the hypothesis}")
Rationale = dsp.Type(
            prefix="Rationale: Let's think step by step.",
            desc="${a step-by-step deduction that identifies the correct response, which will be provided below}")
Answer = dsp.Type(prefix="Answer:", desc='${a response of either "Yes" or "No"}', format=dsp.format_answers)

nli_template = dsp.Template(instructions='Determine if the premise entails the hypothesis. Please respond with "Yes" or "No".', premise=Premise(), hypothesis=Hypothesis(), rationale=Rationale(), answer=Answer())

gpt3_5_lm = dsp.GPT(model='gpt-3.5-turbo', api_key=os.getenv('OPENAI_API_KEY'), model_type="chat")

@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@CacheMemory.cache
def _gpt_nli(passage, claim):
    old_lm = dsp.settings.lm
    
    dsp.settings.configure(lm=gpt3_5_lm)
    
    example = dsp.Example(premise=passage, hypothesis=claim, demos=[])
    example, completions = dsp.generate(nli_template, n=20, temperature=0.7)(example, stage="nli")
    completions = dsp.majority(completions)
    inference = 1 if completions.answer == "Yes" else 0
    
    dsp.settings.configure(lm=old_lm)
    
    return inference

def _gpt_nli_logged(passage, claim):
    print("."*35 + " passage " + "."*35)
    print(passage)
    print("."*35 + " claim " + "."*35)
    print(claim)
    inference = _gpt_nli(passage, claim)
    print("."*35 + " entailment (GPT) " + "."*35)
    print("True" if inference == 1 else "False")
    return inference

if __name__=='__main__':
    passage = "David is a god."
    claim = "David is a man."
    start = time.time()
    print(_t5_nli(passage, claim))
    end  = time.time()
    print("Time:", end - start)
    
    passage = "David is really a man."
    claim = "David is really a man."
    start = time.time()
    print(_t5_nli(passage, claim))
    end  = time.time()
    print("Time:", end - start)
    
    passages = ["David is a god.", "David is really a man."]
    claims = ["David is a man.", "David is really a man."]
    start = time.time()
    print(_t5_nli_batch(passages, claims))
    end  = time.time()
    print("Time:", end - start)