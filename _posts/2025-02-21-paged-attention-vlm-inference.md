---
title: "PagedAttention: Optimizing LLM Inference Serving Systems"
date: 2025-02-21
categories:
  - Research
tags:
  - LLM
  - inference
  - vLLM
  - PagedAttention
  - GPU memory
excerpt: "A deep-dive into PagedAttention and vLLM — how virtual-memory-inspired paging dramatically improves GPU memory utilisation and throughput for large language model serving."
---

*Presentation notes prepared by Tran Manh Dat (21/02/2025).*

---

## Overview

Large language model (LLM) inference is expensive. A single NVIDIA A100 GPU can process fewer than one request per second for LLaMA-13B with moderate-size inputs. The root cause is not compute speed — it is **memory management**. PagedAttention, introduced with the **vLLM** serving system (2023), tackles this problem by borrowing ideas from operating-system virtual memory and paging.

[📄 Original presentation slides (PDF)](/assets/pdfs/Page_Attention_TranManhDat.pdf)

---

## 1. Problems with LLM Inference Systems

### The Cost of Serving LLMs

* LLMs run on high-end GPUs (e.g., NVIDIA A100).
* Each A100 can serve fewer than 1 request/second for a 13 B-parameter model.
* Production-scale services therefore require a large number of GPUs.
* As of 2022, the annual GPU cost for a production LLM service was on the order of **$190 M/year**. A 20 % throughput improvement alone could save roughly **$31.7 M/year**.

Three main levers exist to increase throughput:

1. **Better batching** — Orca (OSDI 2022) introduced a continuous batching scheduler.
2. **Quantisation** — reducing precision from float32 → float16 or int8 reduces memory and speeds up compute.
3. **Enhanced GPU memory usage** — the focus of PagedAttention.

---

### Background: Key Concepts

#### Beam Search

Beam search is a popular algorithm for selecting the final output sequence of an NLP model. Instead of greedily choosing the single best token at each step, beam search keeps the **N best sequences** (beam width) at every position.

#### Batching in Inference

GPUs are designed for parallel processing. During training, a fixed batch of samples is fed at once. Inference is harder:

* Requests arrive one by one in real time; the server must wait for enough requests to fill a batch before running inference.
* Samples within a batch finish at different times, but the whole batch must complete before any result is returned → increased latency.
* Orca's scheduler (continuous batching) solves this by allowing individual sequences to leave and enter the batch dynamically.

#### Transformer Self-Attention (Attention Is All You Need, 2017)

The core attention operation is:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

where $$d_{\text{model}}$$ is the embedding length, $$d_k$$ is the dimension of Q and K, and $$d_v$$ is the dimension of V. Weight matrices $$W^Q, W^K, W^V$$ are learned during training.

**Multi-Head Attention (MHA)** concatenates multiple attention heads and projects the result back to the original dimensionality:

$$\text{MultiHead}(Q,K,V) = \text{concat}(\text{head}_1, \ldots, \text{head}_h)\,W^O$$

#### Autoregressive Generation and KV Cache

LLMs generate tokens one at a time. Computing attention from scratch at each step requires:

$$\mathcal{O}(2 \cdot b \cdot n_{\text{layers}} \cdot d_{\text{model}} \cdot t^2) \text{ FLOPs}$$

where $$t$$ is the total sequence length — **quadratic** in sequence length. Generating the 1 001st token takes 100× more FLOPs than generating the 101st token.

**KV Cache** trades memory for compute by caching the Key and Value tensors on GPU/CPU VRAM for all previous tokens, reducing attention to **linear** scaling with sequence length.

---

### KV Cache Memory Problems (before June 2023)

A GPU serving a 13 B-parameter model allocates roughly:

| Component | GPU Memory Share |
|---|---|
| Static model weights | ~65 % |
| Activations & other data | ~5 % |
| KV Cache | ~30 % |

For a single request with 2 048 tokens, the KV Cache for LLaMA-13B requires approximately:

$$2048 \times 2 \times 5120 \times 40 \approx 1.6\,\text{GB}$$

So a single A100 (80 GB) can serve roughly **10 concurrent requests**.

**Two critical problems existed:**

1. **Memory fragmentation** — existing systems (ORCA, FasterTransformer, PyTorch, TensorFlow) pre-allocated a *contiguous* chunk of GPU memory sized to the maximum sequence length (e.g., 2 048 tokens) using a buddy-allocator strategy. This caused reserved, internal, and external fragmentation. Only **20–40 %** of the KV Cache memory was actually utilised.

2. **No memory sharing** — there was no mechanism to share KV blocks between concurrent requests that share a common prefix (e.g., parallel sampling for code suggestions, beam search candidates).

---

## 2. PagedAttention — vLLM

### System Overview

vLLM is an LLM serving system that introduces **PagedAttention**: an algorithm for storing and accessing KV Cache in non-contiguous, paged physical memory — directly inspired by how OS virtual memory works.

| OS Concept | vLLM Equivalent |
|---|---|
| Process | Request |
| Virtual page | Logical KV block |
| Physical page | Physical KV block |
| Page table | Block table |

### KV Block

A **KV block** is a fixed-size contiguous chunk of memory (default block size = 16 tokens) that stores the Key and Value tensors for a contiguous sequence of tokens.

### PagedAttention Algorithm

PagedAttention allows the **logical** KV blocks for a sequence to be stored in **non-contiguous physical** memory locations. A per-request block table maps logical block indices to physical block numbers and tracks how many slots are filled.

**Example — Request A** with prompt "Alan Turing is a computer scientist":

```
Logical blocks:  [block 0] [block 1] [block 2] [block 3]
Physical blocks: [7]       [1]       [4]       [2]       (allocated on demand)
```

New tokens are appended to the last logical block; when it is full a new physical block is allocated.

### Decoding with PagedAttention

Physical blocks are allocated **on demand** as the sequence grows. When a sequence finishes, all its physical blocks are freed immediately — no pre-allocation waste.

### Application to Other Decoding Scenarios

| Scenario | Benefit |
|---|---|
| **Parallel sampling** (Codex, Copilot) | Multiple output samples share the prompt's KV blocks via copy-on-write |
| **Beam search** | Beam candidates share prefix blocks; diverged blocks are copy-on-write cloned |
| **Shared prefix** | A common system prompt is stored once and shared across all requests |

### Scheduling and Preemption

vLLM uses a first-come-first-served scheduler. When the GPU runs out of memory (OOM):

* All blocks of a sequence or sequence group are evicted together (*all-or-nothing*).
* Evicted blocks are recovered via **swapping** (to CPU RAM) or **re-computation** (recomputing the KV cache when the sequence is re-scheduled).
* Swap space is bounded by the GPU memory allocated for KV Cache.

### Distributed Execution

For models whose parameters exceed a single GPU's capacity, vLLM uses **Megatron-LM** model parallelism. A single, centralised KV cache manager coordinates all GPUs, so no synchronisation overhead is needed for memory management.

### Implementation

* ~8 500 lines of Python (scheduler + block manager)
* ~2 000 lines of C++/CUDA (PagedAttention kernels)
* Kernel-level optimisations:
  * Fused reshape and block copy — single CUDA kernel
  * Fused block read and attention (adapted from FasterTransformer)
  * Batched block copy operations

---

## 3. Evaluation

vLLM was compared against a self-implemented version of ORCA (the authors re-implemented ORCA because its source code was not public at the time). Results showed significant improvements in throughput across multiple model sizes and workloads, with the PagedAttention kernel incurring only 20–26 % higher latency per attention operation compared to the baseline — a small overhead more than offset by the dramatically higher memory utilisation.

**Ablation study highlights:**

* **Block size 16** is the default sweet spot.
* **Re-computation vs. swapping**: re-computation is generally preferred when sequences are short; swapping becomes competitive for longer sequences.

---

## 4. Related Work

### General Model Serving Systems

* **Clipper, TensorFlow Serving, Nexus, InferLine, Clockwork** — cover batching, caching, placement, and scheduling.
* **DVABatch** — multi-entry multi-exit batching.
* **REEF, Shepherd** — preemption-based serving.
* **AlphaServe** — model parallelism.

### Transformer-Specific Serving

Systems such as FasterTransformer use GPU kernel optimisations, advanced batching mechanisms, model parallelism, and parameter sharing.

### Memory Optimisations

* **FlexGen** — swaps weights and tokens for offline LLM inference.
* **FlashAttention** — tiling and kernel optimisations in the attention computation itself.
* **vLLM** combines swapping and re-computation inside a unified paged memory manager.

---

## Appendix: vLLM Timeline

| Date | Event |
|---|---|
| September 2023 | PagedAttention / vLLM paper published |
| January 2024 | vLLM v0.2 released |
| February 7, 2024 | Friendli Engine (ORCA's commercial successor) benchmarks showing improvements over vLLM |
| March 2, 2024 | vLLM v0.3.3 released |
| April 2, 2024 | Third vLLM community meetup |

---

*For the full slide deck see [Page_Attention_TranManhDat.pdf](/assets/pdfs/Page_Attention_TranManhDat.pdf).*
