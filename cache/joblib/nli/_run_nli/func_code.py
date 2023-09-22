# first line: 31
@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@CacheMemory.cache
def _run_nli(passage, claim):
    """
    Run inference for assessing AIS between a premise and hypothesis.
    Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
    """
    print("."*35 + " passage " + "."*35)
    print(passage)
    print("."*35 + " claim " + "."*35)
    print(claim)
    
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
    
    print("."*35 + " entailment " + "."*35)
    print("True" if inference == 1 else "False")
    
    return inference
