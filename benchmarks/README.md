# Attention Benchmarks

A series of experiments profiling attention implementations — time, FLOPs (TFLOP/s),
and peak memory — reproducing the flavor of the FlashAttention paper's eval.

All scripts add the repo root to `sys.path`, so run them from anywhere. On the GPU
server, use the existing venv via uv:

```bash
cd ~/mlsys
uv run --no-project --python ~/mlsys/.venv/bin/python python benchmarks/<script>.py
```

## Experiments

| # | Script | Measures | Uses |
|---|--------|----------|------|
| 1 | `bench_torch_attn.py` | your manual torch attention vs `F.scaled_dot_product_attention`, TFLOP/s + peak mem, swept over seq len, fp32/fp16 | `cs336_basics.model.scaled_dot_product_attention` |
| 2 | `bench_flash.py` | your Triton flash-v1 kernel vs naive torch: correctness, TFLOP/s, peak-mem sweep, + nsys/ncu profiling pass | `softmax_attention_kernel.solve` |

## What each experiment teaches

- **exp 1** — the *baseline story*: manual attention is `O(N²)` memory and OOMs at large N;
  SDPA (fused / flash when fp16 + eligible) stays `O(N)` and faster. Shows *why* flash exists.
- **exp 2** — profiling *your own* flash kernel: TFLOP/s vs peak (T4/4090), and reading an
  nsys timeline (the FA1 multi-launch pattern → many small kernels + gaps).

## Instrumentation cheatsheet

- **time**: `triton.testing.do_bench(fn)` → median ms (handles warmup + sync).
- **FLOPs**: analytical, `4·B·H·N²·d` for self-attention (two matmuls). Flash has the *same*
  FLOPs as naive — it cuts HBM traffic, not compute.
- **peak memory**: `reset_peak_memory_stats()` → `fn()` → `synchronize()` → `max_memory_allocated()`.
- **memory-vs-time**: `torch.cuda.memory._record_memory_history()` → `_dump_snapshot("x.pickle")`
  → open at pytorch.org/memory_viz.
- **timeline**: `nsys profile -t cuda,nvtx --stats=true`; deep single-kernel: `ncu`.

## Planned / ideas

- [ ] fp16/bf16 vs fp32 throughput gap (Tensor Core effect)
- [ ] causal vs full attention
- [ ] memory_viz timeline snapshot for naive vs flash
- [ ] end-to-end: swap model attention to SDPA, measure training step time
