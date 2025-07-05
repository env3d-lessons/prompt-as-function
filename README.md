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
Model load time: 432.93 ms
Chat calls when query is in natural language:
Chat call: 4781.43 ms
Canada
Chat call: 2783.27 ms
Japan
Chat call: 2283.55 ms
China
Chat calls with structured prompt, varying only the input term:
Chat call: 2819.05 ms
Canada
Chat call: 857.70 ms
Japan
Chat call: 755.26 ms
China
```

### What’s Happening?

A few key observations:

* There’s a noticeable speedup after the first call in both sets — the model gets faster even within a session.
* For unstructured queries, calls (after warmup) still take 2.5–3 seconds each.
* For structured prompts where only the final input differs, latency drops dramatically — under 1 second per query!

This performance improvement in the second batch comes entirely from **prompt structure**.

The key insight: local LLMs (via `llama.cpp`) use a **KV cache** to avoid recomputing previously seen tokens. If your prompt has a **fixed prefix**, the model can reuse that part of the computation across calls — resulting in a huge speedup.

So when running multiple similar queries (e.g. batch processing), if you structure your prompts so only the final input changes, you can dramatically boost performance — even on CPU.

This is the core principle behind **Prompt-as-Function**. By rephrasing natural language prompts into templated functional calls, we unlock KV cache reuse, making on-device inference not just possible, but practical.


## Implementation Details

Code structure, Python class overview, prompt templating, multi-model support.

## Performance Optimization

KV cache reuse, prompt reuse, threading, CPU/core utilization.

## Benchmarking Results

Latency tables, model size vs speed, comparison with vanilla approaches, effects of prompt order.

## Usage

Installation instructions, running examples, environment variables.

## Future Work

Ideas like mobile deployment, WebGPU support, expanded microservice orchestration.

## License and Contributions
