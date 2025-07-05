## TL;DR

Run small LLMs locally on CPU as callable Python functions with sub-second latency using KV cache reuse.
✅ No GPU
✅ No internet
✅ Works with Qwen 0.5B–7B via llama.cpp
✅ GitHub Codespaces ready — run AI models for no cost!!
✅ Great for microtasks like classification, extraction, or prompt chaining

## Introduction

Local language models are becoming increasingly viable for real-time inference, but performance bottlenecks—especially on CPU-bound systems—still limit practical use. This project introduces **Prompt-as-Function**, a lightweight Python abstraction that transforms prompts into callable functions. By combining prompt engineering with system-level optimizations such as **KV cache reuse** and **minimal token generation**, we significantly improve inference speed, making even large models like Qwen 7B interactively usable on CPUs.

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
print("Chat calls when query is in natural language:")
print(chat('Ottawa is the capital city of which country? Output only the country name.'))
print(chat('Give me the country name where Tokyo is the capital city of? Output only the country name.'))
print(chat('Which country is Beijing the capital of? Output only the country name.'))

print("Chat calls with structured prompt, varying only the input term:")
print(chat('For the following capital city, output the country name it belongs to. Output only the name: Ottawa'))
print(chat('For the following capital city, output the country name it belongs to. Output only the name: Tokyo'))
print(chat('For the following capital city, output the country name it belongs to. Output only the name: Beijing'))
```

When run in GitHub Codespaces, the output looks like this:

```
$ python chat.py
Model load time: 15600.32 ms
Chat calls when query is in natural language:
Chat call: 14110.52 ms
Canada
Chat call: 5816.95 ms
Japan
Chat call: 5514.24 ms
China
Chat calls with query is structured, with variable query term at the end:
Chat call: 6479.86 ms
Canada
Chat call: 1668.18 ms
Japan
Chat call: 1691.83 ms
China
```

### What’s Happening?

A few key observations:

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
| Merchant | Category | Elapse Time (ms) |
| -------- | -------- | ---------------- |
| Amazon   | retail   | 172.74 |
| Starbucks| retail   | 179.92 |
...
```

Most calls complete in \~150–200ms — all on CPU, without GPU acceleration.  Detailed benchmark results can be found in the "Benchmarking Results" section.

## Usage

This repo is designed to run directly on **GitHub Codespaces**. Simply **fork** the repository and launch a Codespace within GitHub to run your own local LLM — **for free**.

All software setup is handled automatically via the `.devcontainer/` directory. In particular:

* The `install.sh` script installs all dependencies and downloads Qwen models.
* Prebuilt `llama.cpp` shared libraries (`.so` files) are included, so there is **no need to compile llama.cpp yourself**.

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

## License and Contributions


## Benchmarking Results

Here are some fun results.  First up, we use the absolute fastest (also smallest) model with PromptFunction enabled.  As you can see, most calls return < 200ms.  That is **faster** than making API calls to OpenAI.  This this type of categorization tasks, this "prompt as function" approach to local inference seems promising.

```
$ python extract_categories.py 0
```

| Merchant | Category | Elapse Time (ms) |
| -------- | -------- | ---------------- |
| Amazon | retail | 241.78 |
| Starbucks | retail | 177.55 |
| Walmart | retail | 165.44 |
| Target | grocery | 156.45 |
| Apple Store | retail | 169.29 |
| Costco | retail | 165.25 |
| Uber | transportation | 156.09 |
| McDonald's | restaurant | 194.91 |
| Netflix | entertainment | 158.30 |
| Best Buy | retail | 165.50 |
| Shell | retail | 382.19 |
| CVS Pharmacy | pharmacy | 243.62 |
| Home Depot | retail | 168.37 |
| Walgreens | grocery | 168.27 |
| Nike | retail | 157.14 |
| Subway | transportation | 163.49 |
| Delta Airlines | transportation | 163.49 |
| Spotify | music store | 191.57 |
| Lowe's | retail | 253.13 |
| Chipotle | retail | 184.39 |
| Airbnb | retail | 225.40 |
| FedEx | retail | 201.47 |
| Whole Foods Market | retail | 255.51 |
| H&M | retail | 167.81 |
| Google Play | retail | 170.89 |
| AT&T | retail | 189.55 |
| IKEA | retail | 190.79 |
| Domino's Pizza | retail | 201.62 |
| Burger King | retail | 183.43 |
| eBay | retail | 165.38 |

If we turn off prompt-as-function feature:

```
$ PROMPT_AS_FUNCTION=0 python extract_categories.py 0
```

| Merchant | Category | Elapse Time (ms) |
| -------- | -------- | ---------------- |
| Amazon | grocery | 1115.75 |
| Starbucks | retail | 1050.65 |
| Walmart | grocery | 1026.47 |
| Target | grocery | 1075.30 |
| Apple Store | electronics | 1237.13 |
| Costco | grocery | 1095.82 |
| Uber | transportation | 1050.05 |
| McDonald's | restaurant | 1032.86 |
| Netflix | entertainment | 1105.70 |
| Best Buy | grocery | 1012.89 |
| Shell | retail | 1122.07 |
| CVS Pharmacy | pharmacy | 1076.09 |
| Home Depot | grocery | 1052.20 |
| Walgreens | grocery | 1083.40 |
| Nike | grocery | 1210.43 |
| Subway | transportation | 1228.53 |
| Delta Airlines | transportation | 1049.88 |
| Spotify | entertainment | 1137.09 |
| Lowe's | grocery | 1181.66 |
| Chipotle | grocery | 1129.97 |
| Airbnb | lodging | 1224.56 |
| FedEx | transportation | 1189.77 |
| Whole Foods Market | grocery | 1171.68 |
| H&M | categor | 1146.53 |
| Google Play | grocery | 1134.64 |
| AT&T | telecom | 1235.82 |
| IKEA | grocery | 1286.86 |
| Domino's Pizza | grocery | 1179.42 |
| Burger King | restaurant | 1182.32 |
| eBay | ebay | 1251.83 |

The inference time increases by at 5-6x.

## Larger Models

Below are the results for larger models, all using the "prompt as function" approach:

```
$ python extract_categories.py 1 # 1.5B Model
```

| Merchant | Category | Elapse Time (ms) |
| -------- | -------- | ---------------- |
| Amazon | grocery | 458.15 |
| Starbucks | grocery | 464.32 |
| Walmart | grocery | 603.33 |
| Target | delivery | 455.76 |
| Apple Store | services | 459.85 |
| Costco | grocery | 463.94 |
| Uber | grocery | 445.69 |
| McDonald's | grocery | 509.82 |
| Netflix | services | 434.17 |
| Best Buy | grocery | 507.04 |
| Shell | delivery | 480.98 |
| CVS Pharmacy | pharmacy | 565.09 |
| Home Depot | grocery | 465.57 |
| Walgreens | grocery | 461.07 |
| Nike | grocery | 421.17 |
| Subway | delivery | 498.20 |
| Delta Airlines | airlines | 457.86 |
| Spotify | categories: | 461.41 |
| Lowe's | grocery | 540.05 |
| Chipotle | grocery | 533.34 |
| Airbnb | delivery | 457.43 |
| FedEx | delivery | 526.31 |
| Whole Foods Market | grocery | 573.26 |
| H&M | grocery | 625.64 |
| Google Play | services | 540.22 |
| AT&T | delivery | 465.01 |
| IKEA | categories: | 465.39 |
| Domino's Pizza | grocery | 580.28 |
| Burger King | grocery | 523.52 |
| eBay | grocery | 508.22 |

```
$ python extract_categories.py 2 # 3B Model
```

| Merchant | Category | Elapse Time (ms) |
| -------- | -------- | ---------------- |
| Amazon | grocery | 960.48 |
| Starbucks | grocery | 966.19 |
| Walmart | grocery | 1082.68 |
| Target | grocery | 909.43 |
| Apple Store | electronics | 957.61 |
| Costco | grocery | 1106.30 |
| Uber | delivery | 847.72 |
| McDonald's | delivery | 1092.09 |
| Netflix | delivery | 929.05 |
| Best Buy | electronics | 964.11 |
| Shell | grocery | 864.20 |
| CVS Pharmacy | pharmacy | 1041.96 |
| Home Depot | electronics | 983.04 |
| Walgreens | pharmacy | 964.87 |
| Nike | electronics | 958.02 |
| Subway | delivery | 926.89 |
| Delta Airlines | delivery | 1097.80 |
| Spotify | delivery | 951.03 |
| Lowe's | grocery | 1090.20 |
| Chipotle | delivery | 980.19 |
| Airbnb | delivery | 999.13 |
| FedEx | delivery | 947.11 |
| Whole Foods Market | grocery | 1077.83 |
| H&M | clothing | 979.43 |
| Google Play | delivery | 995.60 |
| AT&T | delivery | 1054.53 |
| IKEA | electronics | 1222.08 |
| Domino's Pizza | delivery | 1162.61 |
| Burger King | delivery | 1089.65 |
| eBay | electronics | 929.69 |

```
$ python extract_categories.py 3 # 7B Model!!
```

| Merchant | Category | Elapse Time (ms) |
| -------- | -------- | ---------------- |
| Amazon | retail | 2292.83 |
| Starbucks | restaurant | 2177.79 |
| Walmart | grocery | 2110.41 |
| Target | retail | 1918.18 |
| Apple Store | electronics | 2236.61 |
| Costco | retail | 2211.38 |
| Uber | transportation | 1892.72 |
| McDonald's | restaurant | 2470.24 |
| Netflix | entertainment | 1871.59 |
| Best Buy | electronics | 2561.44 |
| Shell | fuel | 1935.49 |
| CVS Pharmacy | pharmacy | 2588.70 |
| Home Depot | hardware | 2310.50 |
| Walgreens | pharmacy | 2255.45 |
| Nike | clothing | 2172.35 |
| Subway | restaurant | 2311.69 |
| Delta Airlines | transportation | 2245.35 |
| Spotify | entertainment | 2171.47 |
| Lowe's | home improvement | 2507.52 |
| Chipotle | restaurant | 2202.90 |
| Airbnb | lodging | 2109.86 |
| FedEx | transportation | 2178.98 |
| Whole Foods Market | grocery | 2384.26 |
| H&M | clothing | 2290.72 |
| Google Play | telecom | 2166.89 |
| AT&T | telecom | 2123.01 |
| IKEA | furniture | 2332.35 |
| Domino's Pizza | restaurant | 2643.26 |
| Burger King | restaurant | 2458.64 |
| eBay | telecom | 2168.95 |

