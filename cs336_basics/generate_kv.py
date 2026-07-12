"""Profiled / evolving inference — the working copy we optimize (baseline frozen in `generate.py`).

Step 1 (this file): TIMING.
  * `generate_batch` times each forward and returns (texts, step_times), where step_times is a
    list of (cur, seconds) per decode step.
  * `benchmark` runs it many times (after discarding warmup runs) and AGGREGATES the per-step
    forward time across runs -> a clean "time vs step (= sequence length)" curve despite CPU jitter.
  The O(n^2) signature: per-step time GROWS with `cur`, because every step re-forwards the whole
  growing sequence.
Next: split prefill / decode, then add a KV cache so decode processes only the new token (FLATTENS
  that curve).

Run (as a module):
    uv run --with safetensors python -m cs336_basics.generate_kv
"""

import json
import pickle
import statistics
import time

import torch
from safetensors.torch import load_file

from cs336_basics.model import TransformerLM, softmax
from cs336_basics.model_kv import TransformerLMKV
from cs336_basics.tokenization import Tokenizer

EXPORT_DIR = "export_tinystories"
MODEL_KEYS = ["vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff", "rope_theta"]


def load_export(export_dir: str, device: str):
    """Rebuild model + tokenizer from the export folder. (Loading plumbing.)"""
    with open(f"{export_dir}/config.json") as f:
        cfg = json.load(f)
    model = TransformerLM(**{k: cfg[k] for k in MODEL_KEYS})
    state_dict = load_file(f"{export_dir}/model.safetensors", device=device)
    model.load_state_dict(state_dict, strict=False)  # RoPE cos/sin are recomputed in __init__
    model = model.to(device).eval()

    with open(f"{export_dir}/tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)  # {"vocab", "merges", "special_tokens"}
    tokenizer = Tokenizer(tok["vocab"], tok["merges"], tok["special_tokens"])
    return model, tokenizer, cfg


def sample_next(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """From last-position logits (B, vocab), pick each row's next token id -> (B,)."""
    if temperature <= 0:
        return logits.argmax(dim=-1)  # (B,)
    p = softmax(logits / temperature, dim=-1)  # (B, vocab)
    if top_p < 1:
        sorted_p, sorted_idx = torch.sort(p, dim=-1, descending=True)  # (B, vocab)
        cumsum = torch.cumsum(sorted_p, dim=-1)  # (B, vocab)
        mask = cumsum - sorted_p <= top_p  # (B, vocab)  "minus-self trick" keeps the nucleus
        p_filled = torch.masked_fill(sorted_p, mask=~mask, value=0.0)  # (B, vocab)
        rate = torch.sum(p_filled, dim=-1, keepdim=True)  # (B, 1)
        p_normed = p_filled / rate  # (B, vocab)
        choice = torch.multinomial(p_normed, num_samples=1)  # (B, 1)
        return torch.gather(sorted_idx, dim=-1, index=choice).squeeze(-1)  # (B,)
    choice = torch.multinomial(p, num_samples=1)  # (B, 1)
    return choice.squeeze(-1)  # (B,)


@torch.no_grad()
def generate_batch(model, tokenizer, prompts, cfg, max_new_tokens=200, temperature=0.8, top_p=0.95, device="cpu"):
    """Batched static-batching generation. Returns (texts, step_times).

    step_times = list of (cur, forward_seconds) per decode step. We time ONLY the model forward
    (the dominant cost + the KV-cache target). Timing is collected here, REPORTED by the caller.
    """
    eos_id = tokenizer.encode(tokenizer.special_tokens[0])[0]
    pad_id = eos_id

    # 1-3. encode, right-pad into ONE buffer, per-row done flags
    ids = [tokenizer.encode(p) for p in prompts]
    B = len(ids)
    lengths = torch.tensor([len(s) for s in ids], dtype=torch.long, device=device)  # (B,)
    T0 = int(lengths.max())
    x = torch.full((B, T0 + max_new_tokens), pad_id, dtype=torch.long, device=device)  # (B, width)
    for b, s in enumerate(ids):
        x[b, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)  # (B,)

    # 4. decode loop (time ONLY the forward)
    step_times: list[tuple[int, float]] = []  # (cur, forward_seconds) per step
    for _ in range(max_new_tokens):
        cur = int(lengths.max())

        t0 = time.perf_counter()
        logits = model(x[:, :cur])  # (B, cur, vocab)
        step_times.append((cur, time.perf_counter() - t0))  # CPU is synchronous -> accurate

        last = logits[torch.arange(B, device=device), lengths - 1]  # (B, vocab) gather last-real logit
        next_ids = sample_next(last, temperature=temperature, top_p=top_p)  # (B,)
        active = ~finished
        rows = torch.arange(B, device=device)
        x[rows[active], lengths[active]] = next_ids[active]
        lengths[active] += 1
        finished[next_ids == eos_id] = True
        if bool(finished.all()):
            break

    # 5. decode each row up to its real length, drop a trailing EOS
    ans = []
    for b in range(B):
        seq = x[b, : lengths[b]].tolist()
        if seq and seq[-1] == eos_id:
            seq = seq[:-1]
        ans.append(tokenizer.decode(seq))
    return ans, step_times


@torch.no_grad()
def benchmark(model, tokenizer, prompts, cfg, n_runs=20, warmup=2, device="cpu", out_csv="bench_steps.csv"):
    """Run generate_batch n_runs times (after `warmup` discarded runs) and aggregate per-step
    forward time across runs. Prints a table + saves a CSV: step, cur, mean_ms, std_ms, n."""
    print(f"warmup x{warmup} ...")
    for _ in range(warmup):
        generate_batch(model, tokenizer, prompts, cfg, device=device)

    runs = []  # each is a list of (cur, dt)
    for r in range(n_runs):
        _, st = generate_batch(model, tokenizer, prompts, cfg, device=device)
        runs.append(st)
        print(f"  run {r + 1:>2}/{n_runs}: {len(st)} steps")

    # aggregate per step index: cur is deterministic (= T0 + step); dt varies with jitter
    max_steps = max(len(st) for st in runs)
    rows = []
    for k in range(max_steps):
        samples = [st[k] for st in runs if len(st) > k]  # (cur, dt) from runs that reached step k
        cur = samples[0][0]
        dts_ms = [dt * 1e3 for _, dt in samples]
        mean = statistics.mean(dts_ms)
        std = statistics.stdev(dts_ms) if len(dts_ms) > 1 else 0.0
        rows.append((k, cur, mean, std, len(dts_ms)))

    print(f"\n--- per-step forward time over {n_runs} runs (warmup {warmup} discarded) ---")
    print(f"{'step':>4} {'cur':>5} {'mean_ms':>9} {'std_ms':>7} {'n':>4}")
    stride = max(1, len(rows) // 20)  # print ~20 sampled rows to stay readable
    for k, cur, mean, std, n in rows[::stride]:
        print(f"{k:>4} {cur:>5} {mean:>9.1f} {std:>7.1f} {n:>4}")

    with open(out_csv, "w") as f:
        f.write("step,cur,mean_ms,std_ms,n\n")
        for k, cur, mean, std, n in rows:
            f.write(f"{k},{cur},{mean:.4f},{std:.4f},{n}\n")
    print(f"\nsaved {len(rows)} rows -> {out_csv}  (step 0 = prefill; watch mean_ms grow with cur)")
    return rows


def load_kv(export_dir: str, device: str):
    """Like load_export, but builds the CACHE-AWARE TransformerLMKV (loads the SAME weights)."""
    from cs336_basics.model_kv import TransformerLMKV

    with open(f"{export_dir}/config.json") as f:
        cfg = json.load(f)
    model = TransformerLMKV(**{k: cfg[k] for k in MODEL_KEYS})
    state_dict = load_file(f"{export_dir}/model.safetensors", device=device)
    model.load_state_dict(state_dict, strict=False)  # RoPE cos/sin recomputed in __init__
    model = model.to(device).eval()
    with open(f"{export_dir}/tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)
    tokenizer = Tokenizer(tok["vocab"], tok["merges"], tok["special_tokens"])
    return model, tokenizer, cfg


@torch.no_grad()
def generate_cached(
    model: TransformerLMKV,
    tokenizer,
    prompt,
    cfg,
    max_new_tokens=200,
    temperature=0.8,
    top_p=0.95,
    device="cpu",
    profile=False,
):
    """B=1 generation with a KV cache — you write the prefill/decode driver.

    Cache API:  logits, cache = model(x, token_positions, past_kvs)
        past_kvs=None -> prefill (build cache) ;  past_kvs=<cache> -> decode (reuse + extend)
    """
    eos_id = tokenizer.encode(tokenizer.special_tokens[0])[0]
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)  # (1, prompt_len)
    out = []

    # ===== PREFILL: whole prompt -> build cache, sample the first token =====
    positions = torch.arange(0, len(ids))
    # prefill phase inference
    logits, kv_cache = model(x, positions, None)
    logits = logits[..., -1, :]
    # for the prefill phase, it will generate a lot of tokens, but only the last one is used
    next_id = int(sample_next(logits, temperature=temperature, top_p=top_p)[0])
    out.append(next_id)
    cur_pos = x.size(-1) - 1  # now we have [x, new_token], new position = len(x) + 1

    # ===== DECODE: one new token per step, reuse + extend the cache =====
    decode_times: list[tuple[int, float]] = []
    for _ in range(max_new_tokens - 1):
        if next_id == eos_id:
            break
        if profile:
            t0 = time.perf_counter()

        cur_pos += 1
        # for decode phase, only input the last token in the prompts
        logits, kv_cache = model(torch.tensor([[next_id]], device=device), cur_pos, kv_cache)

        next_id = int(sample_next(logits, temperature=temperature, top_p=top_p)[0])
        out.append(next_id)

        if profile:
            decode_times.append((cur_pos, time.perf_counter() - t0))

    if profile and decode_times:
        dts = [dt for _, dt in decode_times]
        print(f"\n[cached] decode {len(dts)} steps, mean {sum(dts) / len(dts) * 1e3:.1f} ms/step  (should be ~FLAT):")
        for pos, dt in decode_times[:: max(1, len(decode_times) // 8)]:
            print(f"     pos={pos:4d}   {dt * 1e3:6.1f} ms")

    if out and out[-1] == eos_id:
        out = out[:-1]
    return tokenizer.decode(ids + out)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"  # on mac, use cpu
    model, tokenizer, cfg = load_export(EXPORT_DIR, device)  # naive baseline model
    kv_model, _, _ = load_kv(EXPORT_DIR, device)  # cache-aware model (same weights)
    print(f"device={device}, params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M\n")

    prompt = "Once upon a time"
    print("=" * 60)
    # CORRECTNESS: greedy (temperature=0) cached output must MATCH the naive baseline token-for-token.
    ans, _ = generate_batch(model, tokenizer, [prompt], cfg, temperature=0.0, device=device)
    cached = generate_cached(kv_model, tokenizer, prompt, cfg, temperature=0.0, device=device, profile=True)
    print("\ngreedy MATCH baseline:", ans[0] == cached)
    print("=" * 60)
    print(cached)

    # (your naive baseline benchmark — uncomment to re-run)
    # benchmark(model, tokenizer, ["Once upon a time", "The little cat", "One day"], cfg, n_runs=20, warmup=2, device=device)
