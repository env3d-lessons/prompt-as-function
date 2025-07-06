## Prompt-as-Function: A Minimal, Fast, Local Runtime for Prompt Engineering and AI Literacy

**Prompt-as-Function** is a lightweight abstraction for turning prompts into callable Python functions — built for:

* **Educators** teaching AI concepts in programming classes
* **Engineers** experimenting with local LLMs and fast inference
* **Prompt engineers** looking for repeatable, testable workflows

It runs entirely **offline**, uses **tiny quantized models**, and supports real-time experimentation — even on CPU.

> Think: `def extract_category(text):` — powered by a small local LLM, ready to run in under 200ms.

Whether you're teaching **temperature tuning** to undergrads or building **modular AI tools** on the edge, Prompt-as-Function provides a clear, reproducible way to think about and work with language models.

## See It in Action

[![Watch the demo](https://img.youtube.com/vi/VwexyiDT8Ic/maxresdefault.jpg)](https://youtu.be/VwexyiDT8Ic)

*3:30 complete walkthrough: GitHub → Codespaces → Running 7B models (download time included for full transparency)*

The video shows the real, unedited experience - including the 3-minute model download and the actual sub-200ms inference times.

## Introduction

Local language models are becoming increasingly viable for real-time inference, but performance bottlenecks—especially on CPU-bound systems—still limit practical use.  This project demonstrates how to leverage KV cache reuse—a standard optimization in production LLM systems—for local inference through a clean Python abstraction. While cloud APIs like OpenAI already use these techniques internally, we make them explicit and accessible for developers running models locally with llama.cpp. The result is dramatically improved inference speed, making even large models like Qwen 7B interactively usable on CPUs.

The core idea is simple: reuse a long system prompt (the function definition) across many short user inputs (the arguments), enabling the model to skip recomputing the same context every time. This lets you treat local LLMs as **modular, composable microservices**, each one doing fast, bounded tasks like name extraction, classification, or date parsing—on-device, with low latency, and no internet access required.

This repo provides:

* A `PromptFunction` class for wrapping prompts as callable Python functions.
* Support for multiple Qwen models (0.5B to 7B), running on `llama.cpp`.
* Benchmarking scripts to demonstrate latency gains from prompt reuse.
* An extensible design that supports both local (`llama.cpp`) and cloud (`openai`) backends.
* **GitHub Codespaces ready**: preconfigured environment with all dependencies.
* **One-click experimentation**: run local models in under a minute with no GPU required.

The result is a practical framework for building **modular, privacy-preserving, low-latency AI utilities** on the edge—with Python and a CPU.


## Background

Large Language Models (LLMs) like GPT-4 have demonstrated remarkable capabilities in reasoning, classification, and information extraction. However, most applications rely on remote APIs, which pose challenges for:

* **Privacy** – Sensitive data must be sent to third-party servers.
* **Latency** – API round trips often take 0.5–2 seconds per call.
* **Cost** – High volume or continuous use quickly becomes expensive.
* **Offline access** – Internet connectivity is a hard requirement.

At the same time, new model formats like **GGUF** and inference libraries like **[llama.cpp](https://github.com/ggerganov/llama.cpp)** have made it possible to run compact models (e.g., Qwen 0.5B–7B) locally, even on CPU-only environments. But simply loading a small model isn't enough—**naïvely prompting the model still results in unnecessary recomputation** for every input, especially when the prompt remains mostly static.

This project emerged from an observation:

> **LLMs behave like interpreters for natural language programs**.
> If a prompt is the “function,” and the input is the “argument,” we can cache and reuse context to accelerate execution—just like in compiled or optimized interpreters.

By **reusing the prompt portion via KV cache**, we dramatically reduce latency for small inference tasks. A Qwen 0.5B model that would take \~1200ms per call without KV cache can respond in under **200ms** with prompt reuse. This makes it possible to treat LLMs as **modular functions**—for classification, extraction, and more—with near-interactive speed, even on low-power devices.

## How It Works

See [TECHNICAL.md](TECHNICAL.md) for a detailed dicsussion on why it works so well.

Most large language models are accessed through a chat interface. For single-turn conversations, this naturally maps to a simple `chat()` function that takes a string as input and returns a string response:

```python
answer = chat('Ottawa is the capital city of which country?')
print(answer)  # Canada
```

Using **[llama.cpp](https://github.com/ggerganov/llama.cpp)** with the `llama-cpp-python` bindings, we can define this `chat()` function locally. However, if we treat this like a chatbot—sending each query as a fresh, full prompt—we won't get great performance, especially on CPU.

To demonstrate this, you can run `chat.py`. This script loads the Qwen 2.5 7B model (2-bit quantized) and performs several calls to the `chat()` function, asking for the country corresponding to a capital city.

**ASIDE**: it is rumored that gpt-4-mini family of models are around 7-8B parameters.  So this model could be similar in power to what is powering chatgpt.

Here’s a sample of the queries:

```python
# Unstructured queries
print(chat('Ottawa is the capital city of which country? Output only the country name.'))
print(chat('Give me the country name where Tokyo is the capital city of? Output only the country name.))
print(chat('Which country is Beijing is the capital city of? Output only the country name.'))

# Structured queries
print(chat('For the following capital city, output the country name the city belongs to, output only the name: Ottawa'))
print(chat('For the following capital city, output the country name the city belongs to, output only the name: Tokyo'))
print(chat('For the following capital city, output the country name the city belongs to, output only the name: Beijing'))

```

When run in GitHub Codespaces, the output looks like this:

```
$ python chat.py
Model load time: 15445.94 ms

============================================================
🗣️  NATURAL LANGUAGE QUERIES
============================================================
Chat call: 13023.34 ms
Ottawa query result: Canada
Chat call: 5994.40 ms
Tokyo query result:  Japan
Chat call: 5159.83 ms
Beijing query result: China

============================================================
🏗️  STRUCTURED QUERIES (variable at end)
============================================================
Chat call: 6221.88 ms
Ottawa → Canada
Chat call: 1677.91 ms
Tokyo → Japan
Chat call: 1688.95 ms
Beijing → China
============================================================
```

### Key Observations

* There’s a noticeable speedup after the first call in both sets — the model gets faster even within a session.
* For unstructured queries, calls (after warmup) still take > 5 seconds each.
* For structured prompts where only the final input differs, latency drops dramatically — under 2 second per query!

This performance improvement in the second batch comes entirely from **prompt structure**.

The key insight: local LLMs (via `llama.cpp`) use a **KV cache** to avoid recomputing previously seen tokens. If your prompt has a **fixed prefix**, the model can reuse that part of the computation across calls — resulting in a huge speedup.

So when running multiple similar queries (e.g. batch processing), if you structure your prompts so only the final input changes, you can dramatically boost performance — even on CPU.

This is the core principle behind **Prompt-as-Function**. By rephrasing natural language prompts into templated functional calls, we unlock KV cache reuse, making on-device inference not just possible, but practical.

## Implementation

This repository provides a lightweight Python class called `PromptFunction` that wraps an LLM as if it were a local function. It uses prompt templating and KV cache reuse to deliver high performance for repeated, short-form tasks — ideal for batch processing on CPU.

### Key Components

**1. `PromptFunction`: a callable wrapper for prompts**

```python
from prompt_function import PromptFunction

extract_category = PromptFunction("Classify the merchant category:", model=0)
print(extract_category("Starbucks"))  # → restaurant
```

This creates a function-like interface over a structured prompt. Internally, it formats the prompt using a fixed prefix + user input template, maximizing cache reuse.

**2. Multi-Model Support**

You can choose between four Qwen2.5 models:

* `0` → Qwen2.5-0.5B
* `1` → Qwen2.5-1.5B
* `2` → Qwen2.5-3B
* `3` → Qwen2.5-7B

Each is loaded from a `.gguf` file (downloaded via `install.sh`), and selected at runtime via a numeric `model` argument.

**3. Built-in Timing Tools**

The `timer()` context manager lets you benchmark prompt execution easily:

```python
with timer("extract_category"):
    print(extract_category("Uber"))
```

This helps you compare latency across models, prompt styles, or hardware.

**4. Controlling KV Cache Reuse**

PromptFunction is designed to reuse the model's **KV cache** by default — meaning the prompt is fixed and only the input varies. This significantly speeds up repeated calls.

You can toggle this behavior using the `PROMPT_AS_FUNCTION` environment variable:

```bash
export PROMPT_AS_FUNCTION=0  # Disable prompt-as-function, disables KV cache reuse
```

When **disabled**, the full prompt (including the task description) is regenerated each time, causing **much slower inference**. This is useful for benchmarking baseline performance or simulating traditional chat-style usage.

When **enabled** (the default), PromptFunction moves the task prompt to the **prefix**, which is cached after the first call. Only the input changes — enabling fast, function-like local inference.

### Example: Categorizing Merchants

Here’s a sample benchmark for classifying 30 common merchants using a 0.5B model with KV cache reuse:

```bash
$ python extract_categories.py 0
```

| Merchant | Category | Elapse Time (ms) |
| -------- | -------- | ---------------- |
| Amazon   | retail   | 172.74 |
| Starbucks| retail   | 179.92 |
...

Most calls complete in \~150–200ms — all on CPU, without GPU acceleration.  Detailed benchmark results can be found in the "Benchmarking Results" section.

## Usage

This repo is designed to run directly on **GitHub Codespaces**. Simply **fork** the repository and launch a Codespace within GitHub to run your own local LLM — **for free**.

All software setup is handled automatically via the `.devcontainer/` directory. In particular:

* The `install.sh` script installs all dependencies.
* Prebuilt `llama.cpp` shared libraries (`.so` files) are included, so there is **no need to compile llama.cpp yourself**.  Resulting in fast codespaces startup times.
* When you run the first script, it will automatically download the Qwen2.5 models (0.5B, 1.5B, 3B, 7B) from Hugging Face - no need to manually download them.

### Included Files

| File                    | Description                                                                        |
| ----------------------- | ---------------------------------------------------------------------------------- |
| `chat.py`               | Demonstrates the KV-cache prefix reuse mechanism (see “How It Works”)              |
| `main.py`               | Sample usage of the `PromptFunction` class                                         |
| `extract_categories.py` | Benchmark script using a single `PromptFunction` to classify merchants             |
| `extract_multi.py`      | Example using **multiple** `PromptFunction`s to extract structured data from input |

### Running the Code

To run any script, simply open a terminal in Codespaces and use standard Python commands. For example:

```bash
$ python extract_categories.py
```

Optional: you can pass an integer to select a specific model:

```bash
$ python extract_categories.py 2  # Use 3B model
```



Your benchmarking results are thorough and compelling. Here's an improved and professionally polished version of your **Benchmarking Results** section. I've preserved your voice while refining grammar, formatting, and flow for clarity and impact. I also added headings and inline explanations to help the reader interpret results more easily.

---

## Benchmarking Results

We benchmarked several **Qwen models** running **locally on CPU** inside GitHub Codespaces using the `PromptFunction` wrapper. This wrapper reuses the **KV cache** to dramatically reduce inference time.  Detailed benchmarking results can be found at [BENCHMARK.md](BENCHMARK.md).  Below are the summary of the results:

* **Prompt-as-function optimization** reduces latency by **5–6×**, even for small models.
* **Qwen 0.5B** with caching achieves **<200ms** inference—**faster than OpenAI API calls**.
* Even a **7B model** becomes usable interactively on CPU, with **\~2s** response time.
* Disabling caching causes **0.5B** to slow down to **>1s per query**.

### Performance by Model Size

| Model     | Prompt-as-Function | Typical Latency | Notes                              |
| --------- | ------------------ | --------------- | ---------------------------------- |
| Qwen 0.5B | Enabled          | **150–250ms**   | Fastest; ideal for local functions |
| Qwen 0.5B | Disabled         | **\~1100ms**    | 5–6× slower without caching        |
| Qwen 1.5B | Enabled          | **400–600ms**   | Great balance of speed and depth   |
| Qwen 3B   | Enabled          | **850–1200ms**  | Richer output, still responsive    |
| Qwen 7B   | Enabled          | **1900–2600ms** | Usable for interactive CPU tasks   |

All benchmarks were run on **CPU-only GitHub Codespaces**, with minimal memory and no GPU acceleration.

## Extending the Framework: LLM Microservices Architecture

Beyond single-function prompting, this framework supports a microservices-style architecture, where multiple PromptFunction instances—each with a different system prompt and dedicated KV cache—run in parallel as independent, callable units.

Each PromptFunction wraps its own model context, optimized for a specific task (e.g., name extraction, date parsing). Because llama.cpp maintains separate KV caches per instance, these models can run concurrently with minimal overhead, enabling composable AI pipelines entirely on-device.

For example, in extract_multi.py, we define two PromptFunctions: one to extract a name, and another to extract a date from the same input sentence:

```python
extract_name = PromptFunction("Extract the person's full name...", model=0, max_tokens=5)
extract_date = PromptFunction("Extract the date mentioned...", model=0, max_tokens=20)
```

Both are evaluated independently for each input, using their own prompt prefix and reusing their respective KV caches. This results in fast, parallel function-like behavior with predictable latency.

Sample output (CPU-only, Qwen 0.5B model):

```
$ python extract_multi.py 
```

| name | date | elapsed time in ms (name) | elapsed time in ms (date) |
| ---- | ---- | ------------------------- | ------------------------- |
| emily zhang | 2021-03-15 | 450.86 | 723.31 |
| carlos rivera | 2023-08-03 | 442.81 | 784.12 |
| sophie dubois | 2020-07-01 | 626.29 | 843.01 |
| arjun patel | 2022-11-12 | 529.70 | 707.67 |
| li wei | 2019-05-09 | 383.27 | 631.18 |
| michael thompson | 2024-02-28 | 424.01 | 726.77 |
| amara okafor | 2023-06-10 | 570.28 | 730.13 |
| benjamin lee | 2018-10-05 | 547.51 | 705.74 |
| fatima hassan | 2022-09-14 | 446.33 | 666.58 |
| hiroshi tanaka | 2021-12-31 | 518.17 | 711.34 |


**note**: the latency for this task is higher due to input being longer

### Architectural Implications

This architecture transforms local LLMs into a suite of reusable, cache-primed AI utilities, each behaving like a microservice. By decoupling model context and purpose, we unlock:

 * Composable pipelines: Chain prompt functions for structured multi-step tasks.
 * Interactive performance: Maintain ~600ms latency for name extract and ~800ms for date extraction, even on CPU.
 * Deployment flexibility: Run multiple functions in memory without GPU or network access.

This approach brings the modularity of cloud-based AI pipelines to the edge—all with local inference, zero dependencies, and full privacy.

## Conclusion

This project demonstrates a simple but powerful optimization: by treating prompts as **functions with fixed prefixes**, we can exploit the KV cache in `llama.cpp` to drastically improve the performance of local inference — even on CPUs.

The key results:

* The smallest model (0.5B) achieves sub-200ms inference consistently — rivaling cloud APIs — and runs comfortably inside GitHub Codespaces.
* Larger models up to 7B parameters remain usable for interactive or batch-style tasks, thanks to prompt engineering that maximizes cache reuse.
* With no reliance on external APIs, the system supports **fully local, fast, and private** inference workflows — ideal for education, prototyping, and lightweight applications.

By framing local language models as **modular functions**, we get both speed and composability — a promising direction for building AI systems that are transparent, testable, and cost-free to experiment with.

## Final Thoughts

This project started as a teaching tool. I wanted students to interact with LLMs locally—free from API limits or billing worries—while learning what actually shapes model behavior. Along the way, it became a clean, reusable framework for **fast, modular, local inference**.

It’s not a new invention, but it surfaces ideas that are widely used but rarely discussed. OpenAI itself recommends techniques like [prompt caching](https://platform.openai.com/docs/guides/prompt-caching) and [prompt reuse](https://platform.openai.com/docs/guides/text?api-mode=responses#prompt-engineering) to reduce latency and improve performance. What this project does is **make those ideas explicit, runnable, and teachable**—in classrooms, on edge devices, and anywhere GPUs aren’t an option.
