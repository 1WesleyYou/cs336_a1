"""Benchmark + profile harness for the flash-attention v1 kernel.  (SKELETON — fill the TODOs)

Run on the GPU server (needs CUDA + triton):
    python bench_flash.py
    nsys profile -t cuda,nvtx --stats=true -o fa1 python bench_flash.py --nsys
    ncu --set full -k _flashattention_v1_kernel -o fa1_ncu python bench_flash.py --nsys
"""
import argparse
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # repo root on path

import torch
import triton

from softmax_attention_kernel import solve


# ---------- reference (given) ----------
def naive_attention(Q, K, V, d):
    S = (Q @ K.transpose(-1, -2)) / (d ** 0.5)
    return torch.softmax(S, dim=-1) @ V


def flash(Q, K, V, M, N, d):
    out = torch.empty((M, d), device=Q.device, dtype=torch.float32)
    solve(Q, K, V, out, M, N, d)
    return out


# ---------- 1. correctness: max|flash - naive| over a few shapes ----------
def check():
    torch.manual_seed(0)
    for (M, N, d) in [(2, 3, 4), (512, 256, 64), (300, 250, 127)]:
        Q = torch.randn(M, d, device="cuda")
        K = torch.randn(N, d, device="cuda")
        V = torch.randn(N, d, device="cuda")
        # TODO: max abs err between flash and naive -> scalar; print with OK/FAIL vs 1e-2
        raise NotImplementedError("correctness check")


# ---------- 2. classic triton perf_report (TFLOP/s vs N) ----------
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"], x_vals=[128, 256, 512, 1024, 2048, 4096, 8192],
        line_arg="provider", line_vals=["flash", "torch"],
        line_names=["Flash v1 (ours)", "PyTorch naive"],
        styles=[("blue", "-"), ("green", "--")],
        ylabel="TFLOP/s", plot_name="flash-v1-tflops",
        args={"M": 512, "d": 64},
    )
)
def bench_tflops(N, M, d, provider):
    Q = torch.randn(M, d, device="cuda", dtype=torch.float32)
    K = torch.randn(N, d, device="cuda", dtype=torch.float32)
    V = torch.randn(N, d, device="cuda", dtype=torch.float32)
    # TODO 1: pick the callable for this provider ("flash" -> solve into a preallocated out, else naive)
    # TODO 2: median/min/max ms via triton.testing.do_bench(..., quantiles=[0.5, 0.2, 0.8])
    # TODO 3: attention FLOPs from shapes -> scalar   (hint: 2 matmuls)
    # TODO 4: return (median, max_ms, min_ms) each converted to TFLOP/s
    raise NotImplementedError("perf")


# ---------- 3. peak memory sweep (expect flash O(N), naive O(N^2)) ----------
def mem_sweep():
    M, d = 512, 64
    print(f"{'N':>6} {'flash_MB':>10} {'naive_MB':>10}")
    for N in [512, 1024, 2048, 4096, 8192]:
        Q = torch.randn(M, d, device="cuda")
        K = torch.randn(N, d, device="cuda")
        V = torch.randn(N, d, device="cuda")
        # TODO: per provider -> reset_peak_memory_stats, run, synchronize, read max_memory_allocated (MB)
        raise NotImplementedError("mem sweep")


# ---------- nsys-friendly: one NVTX-labelled pass, warmup excluded ----------
def nsys_pass():
    M, N, d = 512, 256, 64
    Q = torch.randn(M, d, device="cuda")
    K = torch.randn(N, d, device="cuda")
    V = torch.randn(N, d, device="cuda")
    out = torch.empty((M, d), device="cuda")
    # TODO 1: warmup a few times so JIT compile isn't in the trace, then synchronize
    # TODO 2: wrap ONE solve() call in torch.cuda.nvtx.range_push/pop("flash_v1_solve"), then synchronize
    raise NotImplementedError("nsys pass")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsys", action="store_true")
    args = ap.parse_args()
    if args.nsys:
        nsys_pass()
    else:
        print("== correctness =="); check()
        print("\n== peak memory =="); mem_sweep()
        print("\n== perf =="); bench_tflops.run(print_data=True, save_path=".")
