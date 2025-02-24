'''
Created on Feb 20, 2025

@author: Yihao Fang
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(model='llama3.3', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response['message']['content'])
# or access fields directly from the response object
#print(response.message.content)
