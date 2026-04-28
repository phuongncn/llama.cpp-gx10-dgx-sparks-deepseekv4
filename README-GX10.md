# llama.cpp — DeepSeek V4 Flash on DGX Spark / ASUS GX10

Fork of [llama.cpp](https://github.com/ggerganov/llama.cpp) with fixes for running DeepSeek V4 Flash on NVIDIA DGX Spark / ASUS GX10 (128GB unified memory, 273 GB/s bandwidth).

## Fix: graph_max_nodes Buffer Overflow

DeepSeek V4 Flash has 43 layers with 41 compressed layers (compress_ratios: [0,0,4,128,4,128,...]). The `dsv4_build_compressor_decode_chunk` per-token loop creates ~350K+ tensor objects with n_ubatch=512, which overflows the default ggml compute context pool. This fix increases the DSV4 headroom from +2048 to +450000 tensor slots.

**Commit:** `136e01456` — [fix(deepseek4): increase graph_max_nodes headroom for decode_chunk path](https://github.com/phuongncn/llama.cpp-gx10-dgx-sparks-deepseekv4/commit/136e01456c882e351478ba333754ce511b6480d4)

## Test Results

ASUS GX10 (NVIDIA GB10, 128GB LPDDR5X unified, 273 GB/s):

```
Model:     DeepSeek-V4-Flash-IQ2_XXS (284B params, 82 GB)
Settings:  ctx=128K, batch=4096, ubatch=512, FA=on, ngl=99
Result:    prompt eval = 7.39 tok/s | generation = 6.36 tok/s
```

Stable at 6-7 tok/s without OOM. Each 8K context increase adds ~1 GB compute overhead — keep total under 124 GB VRAM.

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
  --ctx-size 16384 --n-gpu-layers 99 -b 4096 -ub 512 -fa on \
  --host 0.0.0.0 --port 8080
```

## Hardware

Tested on ASUS GX10 (same as NVIDIA DGX Spark):
- NVIDIA GB10 Grace Blackwell Superchip
- 128GB LPDDR5X unified memory
- 273 GB/s memory bandwidth
- 20-core ARM CPU (Grace)
- CUDA compute capability 12.1 (Blackwell)

## Limitations

- Requires `-np 1` (single sequence) — multi-sequence causes assertion failure in compressor
- Recommended context limit: ~128K max to stay within 124 GB VRAM budget
- Performance ceiling: approximately 8-10 tok/s for 80GB+ models on this hardware
