Prompt-as-Function

Fast, private, composable AI microfunctions using local LLMs with preloaded prompts and KV cache reuse.


---

What Is This?

This repository demonstrates how to use quantized local LLMs as cached functions instead of chatbots.

Long prompt = function definition

Short input = function argument

Short output = result

KV cache = reused memory (like compiled code)

Local CPU = fast, private, and inexpensive


This pattern enables modular, fast, and predictable LLM use, especially on limited hardware such as laptops, Raspberry Pi, or cloud Codespaces.


---

Why It Works

LLMs typically spend most compute on recomputing the same prompt every time.

By freezing the prompt and preloading the KV cache, then only changing the short input, you avoid repeated computation. This leads to sub-100ms inference times on CPU and allows building many small, reusable AI microfunctions.

This approach requires control over the model and is not feasible with stateless cloud APIs.


---

Example Use Case: Transaction Categorizer

Input:  "TIM HORTONS 2917"
Output: "Restaurants"

Instead of sending each transaction to a cloud service, this runs locally by:

Loading a small quantized model (e.g., Qwen 1.8B or TinyLlama)

Preloading a prompt with example categories

Appending the input per call

Returning a short classification in ~30–50ms on CPU



---

How to Run

git clone https://github.com/yourname/prompt-as-function
cd prompt-as-function

# Setup environment (llama.cpp, GGUF model, etc)
bash setup.sh

# Run the categorizer script with an example input
python categorize_transaction.py "UBER EATS 0132"

# Expected output:
# "Restaurants"

Compatible with:

llama.cpp

GGUF quantized models (Q2_K, Q3_K_M, etc)

Python bindings

CPU-only environments (e.g., GitHub Codespaces)



---

Benchmarks

Model	Quantization	Task	Hardware	Inference Time

Qwen 1.8B	Q2_K	Categorize one transaction	GitHub Codespaces 2-core	~40ms
TinyLlama 1.1B	Q2_K	Extract name from text	M1 Macbook Air	~25ms
Phi-2	Q4_K_M	Label sentiment	4-core desktop CPU	~60ms


> Prompt preparation is amortized; all runs reuse KV cache.




---

Included Functions

Script	Description

categorize_transaction.py	Classify transaction strings into categories
extract_name.py	Extract names from input text
detect_urgency.py	Tag urgency levels from short messages


Add your own easily:

result = llm_func("your input string here")


---

What This Is (and Isn't)

Intended for:

Running small, quantized models

Fast, simple classification or extraction tasks

Avoiding cloud inference latency, cost, and privacy risks

Teaching modular, composable AI design


Not intended for:

Long-form generation

General-purpose chatbots

Complex multi-turn agents



---

Background

This approach is based on a simple insight:

> Prompts can be designed like functions.
Local LLMs allow freezing prompt behavior by caching, passing inputs as arguments, and returning short outputs efficiently.



We call this the Prompt-as-Function Pattern.


---

Citation

If you use this idea, please cite the forthcoming preprint:

> Madar, J. (2024). Prompt-as-Function: Fast, Composable, CPU-Efficient LLM Microfunctions via KV Cache Reuse. arXiv preprint (TBA).




---

Questions or Feedback?

Open an issue or contact me on []



