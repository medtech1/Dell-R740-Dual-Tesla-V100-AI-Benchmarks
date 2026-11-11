# Reproduction notes — recorded configurations

All commands here are for **reproducing experiments with already verified local models and binaries**. They are not installation instructions. Paths reflect the tested R740 deployment and can differ elsewhere. Avoid running benchmarks concurrently with production inference. Start with `--help` to confirm flags on different llama.cpp revisions.

## llama.cpp build identity

- Commit: `78feec062d50ec9e8940244cac918ac4480b60cb`
- Binary: `/opt/llama.cpp-b11998/build-v100/bin/llama-bench`
- Compiler: GCC 14.2.0
- Driver: 580.196.07; CUDA toolkit: 12.9.2
- NUMA: primary CPU and memory selection node 0 in the following tests

## Qwen3.8-27B official benchmark — exact observed invocation

```bash
runuser -u nobody -- \
  numactl --cpunodebind=0 --membind=0 \
  /opt/llama.cpp-b11998/build-v100/bin/llama-bench \
  -m /opt/llama.cpp/models/Qwen3.8-27B/Qwen3.8-27B-Q4_K_M.gguf \
  -p 512 -n 128 -t 8 -ngl 99 -sm layer -ts 1/1 -lm mmap -r 5 -o md
```

## Gemma 4 31B official benchmark — exact observed invocation

```bash
runuser -u nobody -- \
  numactl --cpunodebind=0 --membind=0 \
  /opt/llama.cpp-b11998/build-v100/bin/llama-bench \
  -m /opt/llama.cpp/models/Gemma4-31B/gemma-4-31B_q4_0-it.gguf \
  -p 512 -n 128 -t 8 -ngl 99 -sm layer -ts 1/1 -lm mmap -r 5 -o md
```

## GPT-OSS-120B official benchmark — exact observed invocation, one configuration

```bash
numactl --cpunodebind=0 --membind=0 \
  /opt/llama.cpp-b11998/build-v100/bin/llama-bench \
  -m /srv/ai-data/models/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf \
  -t 16 -ngl 50 -ncmoe 24 -sm layer -ts 5/1 -lm none \
  -p 512 -n 128 -r 5
```

`-fa` was unspecified in this example. Repeat invocation variation was large; report the full range.

## Qwen MTP — reconstructed invocation (NOT a verbatim recovered terminal record)

```bash
numactl --cpunodebind=0 --membind=0 \
  /opt/llama.cpp-b11998/build-v100/bin/llama-cli \
  -m /opt/llama.cpp/models/Qwen3.8-27B/Qwen3.8-27B-Q4_K_M.gguf \
  --model-draft /opt/llama.cpp/models/Qwen3.8-27B/mtp-Qwen3.8-27B-Q4_0.gguf \
  -t 16 -ngl 99 -sm layer -ts 1,1 -c 4096 -n 384 \
  --spec-type draft-mtp --spec-draft-ngl 99 --spec-draft-n-max 2 \
  --temp 0 \
  -p 'Explain in detail how NUMA memory placement affects dual GPU inference on a dual socket Xeon server. Discuss CPU affinity, PCIe topology, GPU memory bandwidth, layer placement, and practical optimization strategies.'
```

The result (51.1 t/s, repeated twice) is measured; **the complete command above is reconstructed** from preserved settings.

## Gemma MTP — successful exact observed command

```bash
runuser -u nobody -- \
  numactl --cpunodebind=0 --membind=0 \
  /opt/llama.cpp-b11998/build-v100/bin/llama-cli \
  -m /opt/llama.cpp/models/Gemma4-31B/gemma-4-31B_q4_0-it.gguf \
  -md /opt/llama.cpp/models/Gemma4-31B/mtp-gemma-4-31B-it.gguf \
  -t 8 -ngl 99 -sm layer -ts 1/1 -c 4096 -n 384 -fa on \
  --spec-type draft-mtp --spec-draft-n-max 4 \
  --spec-draft-device CUDA1 --temp 0 --seed 42 \
  -p 'Explain in detail how NUMA memory placement affects dual GPU inference on a dual socket Xeon server. Discuss CPU affinity, PCIe topology, GPU memory bandwidth, layer placement, and practical optimization strategies.'
```

## ComfyUI / LTX-Video 0.9.6 Distilled 2B

- Docker image: `comfyui:v0.44.0-v100-hardened` (local, user-managed; no public digest claimed).
- Executed using GPU 0 only; no multi-GPU video claim.
- Standard workflow nodes: `CheckpointLoaderSimple`, `CLIPLoader`, `CLIPTextEncode`, `EmptyLTXVLatentVideo`, `LTXVConditioning`, `LTXVScheduler`, `KSamplerSelect`, `SamplerCustom`, `VAEDecode`, `CreateVideo`, `SaveVideo`.
- Resolution: 1280 × 704; `EmptyLTXVLatentVideo.length=121`; 25 FPS; `LTXVScheduler.steps=8` then `6`.
- Video checkpoint SHA-256: `94891bd4bd08de30d484befbfc54fdcffe6d1596a131baad700b9baa5e1de86b`
- T5 XXL FP16 encoder SHA-256: `6e480b09fae049a72d2a8c5fbccb8d3e92febeb233bbe9dfe7256958a9167635`
- `supported_inference_dtypes` in installed LTXV class: `[torch.bfloat16, torch.float32]`. The generated run reported `torch.float32`; no unsupported FP16 forcing attempted.
- Same seed (`20261113`) for monitored eight- vs six-step comparison.
- Complete benchmark timings from ComfyUI API history (read-only):

```bash
curl -sS --max-time 10 'http://127.0.0.1:8188/history?max_items=1' | python3 -c 'import json,sys; h=json.load(sys.stdin); [(print(v["status"]["status_str"],round((max(t)-min(t))/1000,2))) for v in h.values() for t in [[x[1]["timestamp"] for x in v["status"]["messages"] if x[0] in ("execution_start","execution_success")] ] if len(t)==2]'
```

- GPU monitoring (the command below writes a CSV, not just displays):

```bash
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 1
```

For future reproductions, collect the raw output to a separate filename and retain the complete process exit/error status. The repository includes only summary metrics, not the user's raw telemetry file.

## Source provenance

- `R640_R740_Rebuild_Recovery_Log_2026-11-05_06.md` (local technical log; not published in this repo)
- `R740_Model_Testing_Command_Log_2026-10-31.md` (local technical log; not published in this repo)
- User-provided ComfyUI terminal output, API history, GPU monitoring summaries, and video still (10 November 2026)
- All other unattributed performance claims have deliberately been omitted.
