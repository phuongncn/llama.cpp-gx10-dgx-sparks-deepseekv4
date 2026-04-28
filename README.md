# llama.cpp — DeepSeek V4 Flash on DGX Spark / ASUS GX10

Fork of [llama.cpp](https://github.com/ggerganov/llama.cpp) with fixes for running DeepSeek V4 Flash on NVIDIA DGX Spark / ASUS GX10 (128GB unified 273 GB/s).

## GGUF Download

Download the quantized model from Hugging Face:  
**[antirez/deepseek-v4-gguf](https://huggingface.co/antirez/deepseek-v4-gguf)** — DeepSeek-V4-Flash-IQ2_XXS (284B params, 86.7 GB)

> IQ2_XXS is extreme quantization (2.06 bpw). Despite the aggressiveness, quality is surprisingly good — very usable for chat and coding.

## The Fix

DeepSeek V4 Flash has 43 layers with 41 compressed layers (compress_ratios: [0,0,4,128,4,128,...]). The `dsv4_build_compressor_decode_chunk` per-token loop creates ~350K+ tensor objects with n_ubatch=512, overflowing the default ggml compute context pool. This fork increases the DSV4 headroom from +2048 to +450000 tensor slots, bumping the metadata pool from ~5 MB to ~170 MB.

**Commit:** [`136e01456`](https://github.com/phuongncn/llama.cpp-gx10-dgx-sparks-deepseekv4/commit/136e01456c882e351478ba333754ce511b6480d4)

## Test Results — 8.48 tok/s prompt, 6.65 tok/s generation

Tested on ASUS GX10 (NVIDIA GB10, 128GB LPDDR5X unified, 273 GB/s), with `-ctk f32 -ctv f32`:

```
prompt eval time =  1179.86 ms /   10 tokens ( 117.99 ms/tok =  8.48 tok/s)
       eval time = 23898.95 ms /  159 tokens ( 150.31 ms/tok =  6.65 tok/s)
      total time = 25078.81 ms /  169 tokens
```

> **f32 vs f16 KV cache:** `-ctk f32 -ctv f32` is faster than f16 on Blackwell unified memory. The GB10's high bandwidth (273 GB/s) means the extra memory footprint of f32 is offset by avoiding precision-conversion overhead during attention. Generation speed starts around 7-8 tok/s and settles to ~6.5 tok/s as KV cache fills.

## Run Command

```bash
llama.cpp/build/bin/llama-server \
  -m "/path/to/DeepSeek-V4-Flash-IQ2_XXS.gguf" \
  --host 0.0.0.0 --port 8080 \
  --n-gpu-layers 99 \
  --ctx-size 128000 \
  -ctk f32 -ctv f32 \
  -b 4096 -ub 512 \
  --parallel 1 \
  --threads 4 --threads-batch 20 \
  -fa on \
  --jinja \
  --no-mmap \
  --reasoning-budget -1
```

## Quick Start

```bash
git clone https://github.com/phuongncn/llama.cpp-gx10-dgx-sparks-deepseekv4.git
cd llama.cpp-gx10-dgx-sparks-deepseekv4
mkdir build && cd build
cmake .. -DGGML_CUDA=ON
cmake --build . -j$(nproc) --target llama-server

# Run
./bin/llama-server -m /path/to/DeepSeek-V4-Flash-IQ2_XXS.gguf \
  --ctx-size 128000 --n-gpu-layers 99 -b 4096 -ub 512 \
  -ctk f32 -ctv f32 --parallel 1 --threads 4 --threads-batch 20 \
  -fa on --no-mmap --reasoning-budget -1 \
  --host 0.0.0.0 --port 8080
```

## Hardware

Tested on ASUS GX10 (same hardware as NVIDIA DGX Spark):
- NVIDIA GB10 Grace Blackwell Superchip
- 128GB LPDDR5X unified memory
- 273 GB/s memory bandwidth
- 20-core ARM CPU (Grace)
- CUDA compute capability 12.1 (Blackwell)

## Limitations

- Requires `-np 1` (single sequence) — multi-sequence causes assertion failure in compressor
- Recommended context limit: ~128K max to stay within 124 GB VRAM budget
- IQ2_XXS quality is good despite extreme compression — very usable for chat and coding tasks
