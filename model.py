'''
Created on Feb 21, 2025

@author: Yihao Fang
'''
import dsp
import os
def init_langauge_model(language_model='gpt-3.5-turbo-1106', max_tokens=300):
    
    if language_model=='text-davinci-002':
        openai_key = os.getenv('OPENAI_API_KEY')
        lm = dsp.GPT(model=language_model, api_key=openai_key)
    elif language_model.startswith('gpt'):
        # gpt-3.5-turbo, gpt-3.5-turbo-1106, gpt-4, gpt-4-1106-preview
        openai_key = os.getenv('OPENAI_API_KEY')
        lm = dsp.GPT(model=language_model, api_key=openai_key, model_type="chat")
    elif language_model.startswith('llama'):
        # llama3.3, llama3.2:3b
        lm = dsp.Llama(model=language_model, model_type="chat")
    elif language_model.startswith('qwen'):
        # qwen2.5:72b, qwen2.5:3b
        lm = dsp.Qwen(model=language_model, model_type="chat")
    elif language_model.startswith('deepseek'):
        # deepseek-r1:70b, deepseek-r1:14b
        lm = dsp.DeepSeek(model=language_model, model_type="reasoner")
    else:
        raise NotImplementedError()
        
    dsp.settings.configure(lm=lm)
    dsp.settings.lm.kwargs["max_tokens"] = max_tokens
    

def init_retrieval_model(retrieval_model='google'):
    
    if retrieval_model=='google':
        serpapi_key = os.getenv('SERPAPI_API_KEY')
        rm = dsp.Google(serpapi_key)
    else:
        #colbert_server = 'http://ec2-44-228-128-229.us-west-2.compute.amazonaws.com:8893/api/search'
        colbert_server = 'http://192.168.3.200:8893/api/search'
        rm = dsp.ColBERTv2(url=colbert_server)
    
    dsp.settings.configure(rm=rm)
    