"""Microbenchmark for YOUR torch attention (cs336_basics.model.scaled_dot_product_attention).

Sweeps sequence length N; for each, measures TFLOP/s and peak memory, and compares
your manual attention against F.scaled_dot_product_attention (PyTorch's fused/flash path).

Run on the GPU server:
    uv run --no-project --python ~/mlsys/.venv/bin/python python bench_torch_attn.py
(SKELETON — fill the 3 TODOs)
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # repo root on path

import torch
import torch.nn.functional as F
import triton

from cs336_basics.model import scaled_dot_product_attention as manual_attn


B, H, D = 4, 8, 64                      # batch, heads, head_dim (fixed)
SEQ_LENS = [128, 256, 512, 1024, 2048, 4096, 8192]


def make_qkv(N, dtype):
    g = torch.Generator(device="cuda").manual_seed(0)
    shape = (B, H, N, D)
    return (torch.randn(shape, device="cuda", dtype=dtype, generator=g),
            torch.randn(shape, device="cuda", dtype=dtype, generator=g),
            torch.randn(shape, device="cuda", dtype=dtype, generator=g))


def attn_flops(N):
    # TODO 1: FLOPs of self-attention forward for shape (B,H,N,D).
    #         two matmuls: Q@K^T and P@V. -> return a scalar
    raise NotImplementedError


def time_ms(fn):
    # TODO 2: median latency in ms via triton.testing.do_bench (it handles warmup+sync)
    raise NotImplementedError


def peak_mb(fn):
    # TODO 3: peak MB of one call.
    #   reset peak stats -> run fn() -> synchronize -> read max_memory_allocated (bytes) / 1e6
    raise NotImplementedError


def run(provider, N, dtype):
    Q, K, V = make_qkv(N, dtype)
    if provider == "manual":
        fn = lambda: manual_attn(Q, K, V)
    else:  # torch SDPA (fused / flash when eligible)
        fn = lambda: F.scaled_dot_product_attention(Q, K, V)
    with torch.no_grad():
        ms = time_ms(fn)
        mb = peak_mb(fn)
    tflops = attn_flops(N) / (ms * 1e-3) / 1e12
    return ms, tflops, mb


if __name__ == "__main__":
    for dtype in [torch.float32, torch.float16]:
        print(f"\n===== dtype={dtype} =====")
        print(f"{'N':>6} | {'manual ms':>10} {'TFLOP/s':>8} {'MB':>8} | {'sdpa ms':>10} {'TFLOP/s':>8} {'MB':>8}")
        for N in SEQ_LENS:
            row = f"{N:>6} |"
            for provider in ["manual", "sdpa"]:
                try:
                    ms, tf, mb = run(provider, N, dtype)
                    row += f" {ms:>10.3f} {tf:>8.1f} {mb:>8.1f} |"
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    row += f" {'OOM':>10} {'-':>8} {'-':>8} |"
            print(row)
