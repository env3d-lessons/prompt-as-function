from llama_cpp import Llama

# Shared base model
base_model_paths = ["qwen2.5-0.5b-instruct-q2_k.gguf", "qwen2.5-1.5b-instruct-q2_k.gguf", "qwen2.5-3b-instruct-q2_k.gguf", "qwen2.5-7b-instruct-q2_k.gguf"]

# Each function gets its own prompt + LLM wrapper
class PromptFunction:
    def __init__(self, prompt, model=0):
        self.llm = Llama(model_path=base_model_paths[model],
                         verbose=False,
                         n_ctx=256, 
                         n_threads=2)
        self.system_prompt = prompt
        # warm the cache
        self.llm(prompt, max_tokens=1)

    def __call__(self, input_text):
        prompt = f"""
<|im_start|>user
{self.system_prompt} {input_text}
<|im_end|>
<|im_start|>assistant
"""
        result = self.llm(prompt, max_tokens=20, stop=['\n','<|endoftext|>'])
        return result['choices'][0]['text'].strip()



import time
from contextlib import contextmanager

@contextmanager
def timer(label="Elapsed time"):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    elapsed_ms = (end - start) * 1000
    print(f"{label}: {elapsed_ms:.2f} ms")
