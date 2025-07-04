# 🧠 Prompt as Function

A lightweight framework for wrapping **LLMs as callable Python functions**, optimized for **on-device inference** using [llama.cpp](https://github.com/ggerganov/llama.cpp).

This repository demonstrates how to use quantized local LLMs as cached functions instead of chatbots.

Long prompt = function definition

Short input = function argument

Short output = result

KV cache = reused memory (like compiled code)

Local CPU = fast, private, and inexpensive


This pattern enables modular, fast, and predictable LLM use, especially on limited hardware such as laptops, Raspberry Pi, or cloud Codespaces.

## What This Is (and Isn't)

Intended for:

Running small, quantized models

Fast, simple classification or extraction tasks

Avoiding cloud inference latency, cost, and privacy risks

Teaching modular, composable AI design

Not intended for:

Long-form generation

General-purpose chatbots

Complex multi-turn agents

## Why It Works

LLMs typically spend most compute on recomputing the same prompt every time.

By freezing the prompt and preloading the KV cache, then only changing the short input, you avoid repeated computation. This leads to sub-100ms inference times on CPU and allows building many small, reusable AI microfunctions.

This approach requires control over the model and is not feasible with stateless cloud APIs.

This approach enables fast, modular AI microservices like:

```python
extract_name = PromptFunction("Extract the person's full name from the following sentence:")
extract_date = PromptFunction("Extract the date mentioned in the following sentence:")

print(extract_name("Emily Zhang arrived in Montreal on March 15, 2021."))
# Emily Zhang

print(extract_date("Emily Zhang arrived in Montreal on March 15, 2021."))
# 2021-03-15
```

## ⚙️ Features

* ✅ CPU-only (runs in GitHub Codespaces!)
* ✅ KV cache reuse for blazing-fast inference
* ✅ Works with small models (0.5B – 3B)
* ✅ Modular design with per-function prompts
* ✅ No API keys or network calls
* ✅ Real-time performance for structured tasks

---

## 📊 Performance Benchmarks

This project measures latency of local Qwen models for common NLP tasks like name/date extraction. All tests run **locally on CPU** (2 vCPUs) in GitHub Codespaces using `llama.cpp`.

### 🔹 Model Latency by Size

| Model         | `extract_name` | `extract_date` | Total per Input | Notes           |
| ------------- | -------------: | -------------: | --------------: | --------------- |
| **Qwen 0.5B** |     400–800 ms |     600–900 ms |   \~1.2–1.6 sec | Fastest         |
| **Qwen 1.5B** |    1.0–2.1 sec |    1.7–2.8 sec |   \~2.8–5.0 sec | Higher accuracy |
| **Qwen 3B**   |    2.4–4.6 sec |    3.5–5.5 sec |  \~6.0–10.0 sec | High CPU load   |

> ⏱ Load time is front-loaded once. The above times reflect only **prompt evaluation + generation** using the KV cache.

### 📦 Real-World Example

```shell
$ python main.py
Emily Zhang
extract_name: 805.53 ms
2021-03-15
extract_date: 1002.82 ms
```

### ⚖️ Local vs Remote (OpenAI)

| Method           | Median Latency | Offline | Privacy | Cost           |
| ---------------- | -------------- | ------- | ------- | -------------- |
| Local (0.5B CPU) | \~1.2–1.6 sec  | ✅ Yes   | ✅ Full  | ✅ Free         |
| OpenAI GPT-3.5   | \~1.5–2.5 sec  | ❌ No    | ❌ Risk  | 💲 Token-based |
| OpenAI GPT-4     | \~3.5–5.0 sec  | ❌ No    | ❌ Risk  | 💲💲 High      |

---

## 🧩 Why This Matters

This work shows how **small, specialized local models** can outperform remote calls for **micro-tasks** — making AI:

* Faster than waiting on network roundtrips
* More private (no data leaves the device)
* Easier to reason about (modular prompt-as-code abstraction)

Ideal for **CS1 education**, **edge AI**, and **low-latency agents**.

---

## 🚀 Quick Start

```bash
pip install llama-cpp-python
python main.py
```

Make sure your `gguf` models are downloaded and paths are correctly set in `prompt_function.py`.

---

## 📚 Example Use Cases

* `extract_name`: Full name detection from a sentence
* `extract_date`: Date detection from mixed formats
* Extendable to `extract_location`, `summarize`, `flag_sensitive`, etc.

---

## 🧠 Credit

Built using [llama.cpp](https://github.com/ggerganov/llama.cpp) and Qwen models from Alibaba. Designed by [Jason Madar](https://github.com/env3d) as part of a CS1 LLM research project.

---


