# LLM Inference Engineering Training

This repository is a hands-on C++ training project for understanding how large language models (LLMs) run efficiently during inference.

The focus is on the low-level engineering behind modern transformer-based models: matrix multiplication, tensor operations, attention, caching, tokenization, quantization, and CPU optimization. Each level builds on the previous one, moving from fundamental numerical kernels to higher-level model components.

## Course progression

- Level 01: GEMM benchmarking and optimization
- Level 02: Tensor operations and memory layout
- Level 03: Linear layers and projection matrices
- Level 04: Attention mechanisms
- Level 05: KV cache for efficient generation
- Level 06: Transformer blocks
- Level 07: Tokenization
- Level 08: Tiny LLM implementation
- Level 09: Quantization
- Level 10: CPU backend kernels
- Level 11: Agent harness and runtime tooling

## Why this matters

LLM inference is not just about model weights and neural network math. It is also about performance: memory bandwidth, cache locality, SIMD vectorization, threading, batching, and efficient runtime design. This project teaches those ideas through practical C++ implementations and microbenchmarks.

## Core learning objective

To build intuition for how transformer inference works under the hood, and how the engineering choices behind GEMM, attention, and memory access patterns directly affect latency and throughput.
