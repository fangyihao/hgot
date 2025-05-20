from openai import OpenAI
import datetime
model_type = "chat"
model_id = "qwen2.5:14b"
start = datetime.datetime.now()
# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
if model_type == "text":
    completion = client.completions.create(model=model_id, n=20, prompt="San Francisco is a")
elif model_type == "chat":
    completion = client.chat.completions.create(model=model_id, n=20, messages=[{"role": "user", "content": "San Francisco is a"}])

end = datetime.datetime.now()
duration = end - start
if model_type == "text":
    for choice in completion.choices:
        print(choice.text)
        print('-'*30)
elif model_type == "chat":
    for choice in completion.choices:
        print(choice.message.content)
        print('-'*30)

print("Duration:", duration)
