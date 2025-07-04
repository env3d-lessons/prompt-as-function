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

LLM basics, challenges with local inference, KV cache reuse.

## Prompt-as-Function Concept

Explanation of your abstraction: prompt, max tokens, model selection, template options.

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
