## Prompt-as-Function: Technical Note

### Overview

This document captures the technical design behind the "prompt-as-function" approach using `llama.cpp` on CPU. The approach enables fast, reusable, low-latency inference by treating prompts as callable functions and leveraging OS-level memory optimizations such as `mmap` and copy-on-write.

Reasons why the technical implementation is simple is efficient:

 * Attention head calculations are independent of the feed forward network (transformer architecture)
 * llama.cpp already implemented caching at the attention head level (kv-cache)
 * Multiple instances of llama.cpp leverage battle tested CPU based memory management (mmap, copy-on-write) for multiple threads/processes

---

### Key Concept

Treat LLM prompts as **modular, callable functions**, where:

* **Prefix = function body** (fixed, preloaded prompt)
* **User input = function argument** (short appended string)
* **Output = return value** (e.g., classification, generation, extraction)

This allows:

* Fast re-execution without recomputing the prefix
* Reuse of the KV cache
* Shared memory use across instances via OS-level optimizations

---

### llama.cpp Internals Used

#### 1. `llama_model`

* Loaded via `llama_load_model_from_file()`
* Contains read-only weights (FFN, attention, embeddings)
* Memory-mapped (`mmap`) from `.gguf` file
* **Shared across contexts and processes** by the OS

#### 2. `llama_context`

* Created via `llama_new_context_with_model()`
* Contains per-inference state:

  * KV cache (token position → key/value tensors)
  * Scratch buffers
  * RNG / logits
* **Not shared** between contexts

---

### Technical Leverage Points

#### KV Cache Reuse

* llama.cpp reuses cached KV tensors if `n_past` is set properly.
* This allows preloading a fixed prefix (e.g., prompt instructions), storing the resulting KV cache, and then performing fast inference by appending user input.

#### OS-Level Memory Sharing

* Model weights are `mmap`'d and marked read-only.
* Linux automatically shares these pages between processes/threads.
* This results in **copy-on-write semantics**:

  * Only one copy of weights exists in RAM
  * Additional contexts or subprocesses don’t increase memory pressure

#### Stateless FFN Modules

* Transformer FFNs are stateless, purely functional layers.
* Inference-time sharing is safe as long as per-call input/output memory is isolated.

---

### Why It Works So Well

#### Technical Elegance

* No model patching or forking
* Minimal Python or C++ orchestration
* OS handles memory sharing transparently

#### Performance Benefits

* **Low latency**: minimal tokens processed per call
* **Low memory**: shared weights, per-call cache only
* **Highly modular**: multiple "functions" = multiple contexts

#### Portable

* Works on any Linux system with enough RAM
* Runs entirely on CPU
* Compatible with Qwen 0.5B–1.5B class models and others

---

### Limitations and Notes

* `llama-cpp-python` bindings expose only one context per `Llama()` instance

  * Workaround: spawn multiple instances or subprocesses
* `mmap`-based sharing works best on Linux; behavior on macOS/Windows varies
* No explicit API to save/restore KV cache — must rely on warm contexts

---

### Summary

This approach turns prompt reuse into a practical performance primitive using nothing but the existing `llama.cpp` API and OS-level memory optimizations. It's a perfect illustration of how deep system-level understanding can unlock elegant, efficient AI applications — without touching a single CUDA kernel.

> **Prompt-as-function is not just a clever prompt trick — it’s a runtime abstraction.**
