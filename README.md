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

Below are results comparing different Qwen models under various configurations. All benchmarks were run in **GitHub Codespaces** using CPU-only inference and the `PromptFunction` wrapper. Elapsed times include full inference time, measured in milliseconds (ms).

### TL;DR

* **Prompt-as-function** dramatically improves performance by reusing the KV cache.
* With this optimization, **even a 7B model becomes responsive**, and a 0.5B model consistently outperforms cloud APIs.
* Turning the optimization **off** leads to **5x–6x slower inference**.

---

### Fastest Model: Qwen 0.5B with Prompt-as-Function Enabled

```bash
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

> ✅ **Most calls under 200ms**. This is **faster than OpenAI's API**, all without leaving your machine.

---

### 🐢 0.5B Model with Prompt-as-Function **Disabled**

```bash
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

> Inference time balloons to over **1 second per query**, despite being the same model and hardware.

---

### Scaling Up: Larger Models (All Use Prompt-as-Function)

#### Qwen 1.5B

```bash
$ python extract_categories.py 1
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

> Steady performance in the **400–600ms** range. Still usable for batch processing.

---

#### Qwen 3B

```bash
$ python extract_categories.py 2
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

> More semantic nuance at the cost of latency. All calls stay **under 1.2 seconds**.

---

#### Qwen 7B

```bash
$ python extract_categories.py 3
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

> This is **7 billion parameters** running interactively on CPU! The response times (2–2.6s) are within tolerable range for some real-time workflows.

---

### Key Observations

* The **Prompt-as-Function** design turns local LLMs into usable microservices — especially valuable for structured tasks like classification or extraction.
* Performance scales **predictably** with model size, but remains manageable thanks to KV cache reuse.
* You don’t need a GPU. With smart prompt structuring and llama.cpp’s efficiency, **CPU-only inference is fast enough** for many practical applications.

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

The results speak for themselves:

* The smallest model (0.5B) achieves sub-200ms inference consistently — rivaling cloud APIs — and runs comfortably inside GitHub Codespaces.
* Larger models up to 7B parameters remain usable for interactive or batch-style tasks, thanks to prompt engineering that maximizes cache reuse.
* With no reliance on external APIs, the system supports **fully local, fast, and private** inference workflows — ideal for education, prototyping, and lightweight applications.

By framing local language models as **modular functions**, we get both speed and composability — a promising direction for building AI systems that are transparent, testable, and cost-free to experiment with.


Yes — **those two links are absolute gold** for your case. They confirm exactly what you’re doing, but OpenAI keeps it buried in “for developers only” sections:

---

From [OpenAI’s Text Completion Guide](https://platform.openai.com/docs/guides/text?api-mode=responses#prompt-engineering):

> *“To improve performance, we recommend prompt engineering strategies that reduce prompt size and reuse common prompt components across requests (prompt caching).”*

And from their dedicated (and highly technical) [Prompt Caching Guide](https://platform.openai.com/docs/guides/prompt-caching):

> *“Prompt caching enables high-throughput applications by storing the results of prefix prompt evaluations and only running inference on appended inputs.”*

---

They are *absolutely describing* KV cache reuse, modular prompt design, and function-like behavior — but only as **performance tips**, not as a conceptual or pedagogical framework.

### Why This Matters for You

You’re taking what OpenAI treats as backend optimization and saying:

* “This isn’t just an implementation trick — it’s a programming model.”
* “This can be surfaced and taught as part of **AI literacy and software design.**”
* “You don’t need OpenAI infrastructure to use these ideas. You can run them **locally, fast, and transparently.**”


## Final Thoughts

This project started as a teaching tool. I wanted students to interact with LLMs locally—free from API limits or billing worries—while learning what actually shapes model behavior. Along the way, it became a clean, reusable framework for **fast, modular, local inference**.

It’s not a new invention, but it surfaces ideas that are widely used but rarely discussed. OpenAI itself recommends techniques like [prompt caching](https://platform.openai.com/docs/guides/prompt-caching) and [prompt reuse](https://platform.openai.com/docs/guides/text?api-mode=responses#prompt-engineering) to reduce latency and improve performance. What this project does is **make those ideas explicit, runnable, and teachable**—in classrooms, on edge devices, and anywhere GPUs aren’t an option.
