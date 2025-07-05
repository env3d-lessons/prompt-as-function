from llama_cpp import Llama
import random
import math
import sys, os
import time

def log_time(label):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            elapsed = (end - start) * 1000
            print(f"{label}: {elapsed:.2f} ms")
            return result
        return wrapper
    return decorator

# Suppress stderr temporarily
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

# Measure model load time
load_start = time.perf_counter()

llm = Llama(
    model_path="./qwen2.5-7b-instruct-q2_k.gguf",      
    seed=random.randint(0, 2**31 - 1),
    n_ctx=256, 
    n_threads=2,
    verbose=False
)

load_end = time.perf_counter()
sys.stderr = stderr  # Restore stderr
print(f"Model load time: {(load_end - load_start) * 1000:.2f} ms")

@log_time("Chat call")
def chat(prompt, temperature=0.7, max_tokens=1024, top_p=0.9, top_k=40):
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    result = llm.create_chat_completion(
        prompt,
        max_tokens=max_tokens, 
        temperature=temperature, 
        top_p=top_p, 
        top_k=top_k
    )
    return result['choices'][0]['message']['content'].strip()

print("Chat calls when query is in natural language:")
print(chat('Ottawa is the capital city of which country? Output only the country name.'))
print(chat('Give me the country name where Tokyo is the capital city of? Output only the country name.'))
print(chat('Which country is Beijing is the capital city of? Output only the country name.'))

print("Chat calls with query is structured, with variable query term at the end:")
print(chat('For the following capital city, output the country name the city belongs to, output only the name: Ottawa'))
print(chat('For the following capital city, output the country name the city belongs to, output only the name: Tokyo'))
print(chat('For the following capital city, output the country name the city belongs to, output only the name: Beijing'))

