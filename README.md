# llama.cpp — DeepSeek V4 Flash on DGX Spark / ASUS GX10

Fork of [llama.cpp](https://github.com/ggerganov/llama.cpp) with fixes for running DeepSeek V4 Flash on NVIDIA DGX Spark / ASUS GX10 (128GB unified 273 GB/s).

## GGUF Download

Download the quantized model from Hugging Face:  
**[antirez/deepseek-v4-gguf](https://huggingface.co/antirez/deepseek-v4-gguf)** — DeepSeek-V4-Flash-IQ2_XXS (284B params, ~82 GB)

## The Fix

DeepSeek V4 Flash has 43 layers with 41 compressed layers (compress_ratios: [0,0,4,128,4,128,...]). The `dsv4_build_compressor_decode_chunk` per-token loop creates ~350K+ tensor objects with n_ubatch=512, overflowing the default ggml compute context pool. This fork increases the DSV4 headroom from +2048 to +450000 tensor slots, bumping the metadata pool from ~5 MB to ~170 MB.

**Commit:** [`136e01456`](https://github.com/phuongncn/llama.cpp-gx10-dgx-sparks-deepseekv4/commit/136e01456c882e351478ba333754ce511b6480d4)

## Test Results — 6-7 tok/s

Tested on ASUS GX10 (NVIDIA GB10, 128GB LPDDR5X unified, 273 GB/s):

```
prompt eval:  676.25 ms /    5 tokens = 135.25 ms/tok = 7.39 tok/s
generation: 24375.50 ms /  155 tokens = 157.26 ms/tok = 6.36 tok/s
total:      25051.75 ms /  160 tokens
```

## Run Command

```bash
llama.cpp/build/bin/llama-server \
  -m "/path/to/DeepSeek-V4-Flash-IQ2_XXS.gguf" \
  --host 0.0.0.0 --port 8080 \
  --n-gpu-layers 99 \
  --ctx-size 128000 \
  -ctk f16 -ctv f16 \
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
git checkout main
mkdir build && cd build
cmake .. -DGGML_CUDA=ON
cmake --build . -j$(nproc) --target llama-server

# Run
./bin/llama-server -m /path/to/DeepSeek-V4-Flash-IQ2_XXS.gguf \
  --ctx-size 128000 --n-gpu-layers 99 -b 4096 -ub 512 \
  -ctk f16 -ctv f16 --parallel 1 --threads 4 --threads-batch 20 \
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
- Performance ceiling: approximately 8-10 tok/s for 80GB+ models on this hardware
