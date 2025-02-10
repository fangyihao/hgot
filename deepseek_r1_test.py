import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
# Use a pipeline as a high-level helper
#from transformers import pipeline

#messages = [
#{"role": "user", "content": "Who are you?"},
#]
#pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1", trust_remote_code=True)
#pipe(messages)


# Load model directly
#from transformers import AutoModelForCausalLM
#model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")

#from transformers import AutoModelForCausalLM
#model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-70B", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")

#from transformers import AutoModelForCausalLM
#model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")


from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(model='deepseek-r1:70b', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response['message']['content'])
# or access fields directly from the response object
print(response.message.content)
