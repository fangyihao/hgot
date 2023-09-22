'''
Created on Sep. 5, 2023

@author: Yihao Fang
'''
import time
import openai
import re
import random
import os
from dotenv import load_dotenv
import numpy as np
load_dotenv()
seed = 42
language_model='gpt-3.5-turbo'
np.random.seed(seed)
random.seed(seed)
openai_key = os.getenv('OPENAI_API_KEY')
openai.api_key = openai_key

def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.error.RateLimitError, openai.error.Timeout, openai.error.APIConnectionError, openai.error.APIError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def paraphrase(passage, n, temperature=1):
    start = time.time()
            
    if n == 1:
        instruction = "Please paraphrase the sentence below:\n"
    else:
        instruction = "Please generate %d paraphrases of the sentence below:\n"%n
       
    if n == 1:
        paraphrases = []   
        response = completions_with_backoff(
            model=language_model, 
            messages=[{"role": "user", "content": instruction + passage}],
            temperature=temperature
        )
        
        content = response.choices[0].message.content
        paraphrases.append(content)
    
    else:
        #paraphrases = []
        contents = []
        content=""
        while not content.split('\n')[-1].startswith(str(n)):
            messages = []
            messages.append({"role": "user", "content": instruction + passage})
            if len(contents) > 0:
                messages.append({"role": "assistant", "content": contents[-1]})
                messages.append({"role": "user", "content": "continue"})
            
            response = completions_with_backoff(
                model=language_model, 
                messages=messages, 
                max_tokens=1920,
                temperature=temperature
            )
        
            content = response.choices[0].message.content
            contents.append(content)
            
        paraphrases = (''.join(contents)).split('\n')
        
    for i in range(len(paraphrases)):
        paraphrase = paraphrases[i]
        if len(paraphrase.strip()) > 0:
            mo = re.match(r"[0-9]+\.\s+(.*)", paraphrase)
            if mo:
                paraphrases[i] = mo.group(1)
            
    end = time.time()
    return paraphrases

if __name__=='__main__':
    print(paraphrase("The reading group concludes at 11:35 a.m.", 3))
