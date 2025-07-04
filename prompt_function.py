import os, sys
from llama_cpp import Llama

class PromptFunction:
    base_model_paths = [
        "qwen2.5-0.5b-instruct-q2_k.gguf",
        "qwen2.5-1.5b-instruct-q2_k.gguf",
        "qwen2.5-3b-instruct-q2_k.gguf",
        "qwen2.5-7b-instruct-q2_k.gguf"
    ]

    def __init__(
        self,
        prompt: str,
        model: int = 0,
        max_tokens: int = 2,
        temperature: float = 0.0,
        stop: list = None
    ):
        self.prompt = prompt.strip()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stop = stop or ['\n', '<|endoftext|>']

        self.template = ("<|im_start|>user\n{prompt}\n{input}\n<|im_end|>\n<|im_start|>assistant\n")
        if os.getenv("PROMPT_AS_FUNCTION") == "0" :
            self.template = ("<|im_start|>user\n{input}\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n")



        # Suppress stderr temporarily
        stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')

        self.llm = Llama(
            model_path=self.base_model_paths[model],
            verbose=False,
            n_ctx=256,
            n_threads=os.cpu_count()
        )
        
        sys.stderr = stderr  # Restore stderr

        # Warm the cache with a dummy call
        warmup_prompt = self.template.format(prompt=self.prompt, input="")
        _ = self.llm(warmup_prompt, max_tokens=1, stop=self.stop, temperature=self.temperature)

    def __call__(self, input_text: str) -> str:
        full_prompt = self.template.format(prompt=self.prompt, input=input_text.strip())
        result = self.llm(
            full_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=self.stop
        )
        return result['choices'][0]['text'].strip().lower()

### Utilitiy class for benchmarking

import time
from contextlib import contextmanager

@contextmanager
def timer(label="Elapsed time"):
    start = time.perf_counter()
    yield lambda: (time.perf_counter() - start) * 1000
