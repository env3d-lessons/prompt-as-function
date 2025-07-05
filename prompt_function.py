import os, sys
import subprocess
from llama_cpp import Llama

# Global model configuration
MODEL_URLS = {
    "qwen2.5-0.5b-instruct-q2_k.gguf": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q2_k.gguf",
    "qwen2.5-1.5b-instruct-q2_k.gguf": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q2_k.gguf", 
    "qwen2.5-3b-instruct-q2_k.gguf": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q2_k.gguf",
    "qwen2.5-7b-instruct-q2_k.gguf": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q2_k.gguf"
}


### Utilities for benchmarking and model downloads ###

import time
from contextlib import contextmanager

@contextmanager
def timer(label="Elapsed time"):
    start = time.perf_counter()
    yield lambda: (time.perf_counter() - start) * 1000


def download_models_if_needed():
    """Download model files if they are not present locally."""
    
    for model_file, url in MODEL_URLS.items():
        if not os.path.exists(model_file):
            print(f"Model {model_file} not found. Downloading...")
            try:
                # Use wget with -nc (no-clobber) to avoid re-downloading if file exists
                subprocess.run(["wget", "-nc", url], check=True)
                print(f"Successfully downloaded {model_file}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to download {model_file}: {e}")
                sys.exit(1)
            except FileNotFoundError:
                print("wget command not found. Please install wget or download the models manually.")
                sys.exit(1)
        else:
            print(f"Model {model_file} already exists.")


### -- Start of the main functionality -- ###

# Download models if needed when the module is imported
download_models_if_needed()

class PromptFunction:
    base_model_paths = list(MODEL_URLS.keys())

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
