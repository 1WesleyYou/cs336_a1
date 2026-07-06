"""Load the exported TinyStories model and generate text (autoregressive decoding).

Export layout (symmetric to how it was saved during training):
    export_tinystories/config.json        # hyperparameters
    export_tinystories/model.safetensors  # TransformerLM.state_dict()
    export_tinystories/tokenizer.pkl       # {"vocab", "merges", "special_tokens"}

Run (as a module, so `cs336_basics` is importable):
    uv run --with safetensors python -m cs336_basics.generate
"""

import json
import pickle

import torch
from safetensors.torch import load_file

from cs336_basics.model import TransformerLM, softmax
from cs336_basics.tokenization import Tokenizer

EXPORT_DIR = "export_tinystories"
MODEL_KEYS = ["vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff", "rope_theta"]


def load_export(export_dir: str, device: str):
    """Rebuild model + tokenizer from the export folder. (Loading plumbing — done for you.)"""
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
    """From the last-position logits (shape (batch_size, vocab)), pick the next token id (batch_size,)."""
    #   - greedy (temperature <= 0):  return int(logits.argmax())
    #   - else:  probs = softmax(logits / temperature, dim=-1)      # <1 sharper, >1 more random
    # operate on the temperature
    if temperature <= 0:  # (B * vocab)
        return logits.argmax(dim=-1)  # B
    p = softmax(logits / temperature, dim=-1)  # B * vocab
    if top_p < 1:
        # so here we need to find the cumulative amount which reaches `top_p`
        sorted_p, sorted_idx = torch.sort(p, dim=-1, descending=True)  # B * vocab
        cumsum = torch.cumsum(sorted_p, dim=-1)  # B
        mask = (
            cumsum - sorted_p <= top_p
        )  # NOTE: since we need to find the smallest set which just > top_p, directly comparing only find the last one which is < top_p but we need the next one, so here we need to have a `minus self trick` to represent the last cumsum item

        # find the cumsum/norm_rate for top_p
        p_filled = torch.masked_fill(sorted_p, mask=~mask, value=0.0)  # B * vocab
        rate = torch.sum(p_filled, dim=-1, keepdim=True)  # B * 1 for later dividing ops
        p_normed = p_filled / rate  # B * vocab
        choice = torch.multinomial(p_normed, num_samples=1)  # B * num_samples
        return torch.gather(
            sorted_idx, dim=-1, index=choice
        ).squeeze(
            -1
        )  # B, note that gather type is same as index size, which is B * num_samples (==1) -> B, which only takes a squeeze ops
    choice = torch.multinomial(p, num_samples=1)  # B
    return choice.squeeze(-1)


@torch.no_grad()  # inference: no gradients
def generate(model, tokenizer, prompt, cfg, max_new_tokens=200, temperature=0.8, top_p=0.95, device="cpu"):
    """Autoregressive generation.

    Conceptually two phases (this naive version re-forwards the whole sequence each step,
    so prefill/decode aren't separated and there's no KV cache — that's the next upgrade):
      - PREFILL: the first forward processes the whole prompt at once.
      - DECODE:  each later step feeds the sequence-so-far and takes only the last position.
    """
    eos_id = tokenizer.encode(tokenizer.special_tokens[0])[0]  # to stop on <|endoftext|>
    # prepare the init tensor
    # notice that here we only have batch size = 1
    x = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    # loop on every token
    for _ in range(max_new_tokens):
        # in each loop, generate one token
        # the conditions is all previous prompts + generated tokens, with context window (in the end)
        x_cond = x[:, -cfg["context_length"] :]  # the first dim is the batch size
        # model * x_cond is [batch * seq * vocab_size], with batch = 1, each seq means the next token's score
        # since in autoregression manner, we only need the next 1 token, which only takes in the last prompt token
        logits = model(x_cond)[0, -1]  # 0 means the first (here also the only) token; -1 means the last token for query
        next_id = sample_next(logits, temperature=temperature, top_p=top_p)
        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=-1)  # append to the end of the overall x
        if next_id == eos_id:
            break

    return tokenizer.decode(x[0].tolist())  # remove the batch dimension


@torch.no_grad()
def generate_batch(model, tokenizer, prompts, cfg, max_new_tokens=200, temperature=0.8, top_p=0.95, device="cpu"):
    """Batched generation: serve SEVERAL prompts at once (static batching).

    The single-request `generate` above serves ONE prompt. Real serving handles many. The
    moment batch size B > 1, three NEW problems appear — and these are exactly what Orca's
    continuous batching later fixes:
      1. prompts have DIFFERENT lengths        -> must PAD them into one (B, T) tensor
      2. every sequence samples its OWN next id -> sample per row
      3. sequences hit EOS at DIFFERENT steps   -> can't `break` on the first; track a per-row
         `finished` flag and stop only when ALL are done (the slow one holds up the whole batch).

    Design here: RIGHT-pad + the existing causal mask. Pads sit in the "future", so the causal
    mask ALREADY blocks them -> real tokens are never corrupted, no extra padding-mask needed.
    The only catch: each row's last REAL token is at a different column, so we can't take
    logits[:, -1] — we GATHER each row's logit at its own current length.
    (Production decoder-only serving usually LEFT-pads instead, for KV-cache alignment.)
    """
    eos_id = tokenizer.encode(tokenizer.special_tokens[0])[0]
    pad_id = eos_id  # pad columns are masked/unused, so any id is fine

    # 1-3. encode, right-pad into ONE buffer, per-row done flags
    ids = [tokenizer.encode(p) for p in prompts]  # list[list[int]]
    B = len(ids)  # number of prompt requests
    lengths = torch.tensor([len(s) for s in ids], dtype=torch.long, device=device)  # (B,)  real lengths
    T0 = int(lengths.max())  # longest prompt for padding later
    x = torch.full((B, T0 + max_new_tokens), pad_id, dtype=torch.long, device=device)  # (B, width)
    for b, s in enumerate(ids):  # b is idx of prompt; s is value of prompt
        x[b, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)  # left-align real ids
    finished = torch.zeros(B, dtype=torch.bool, device=device)  # (B,)

    # 4. decode loop
    for _ in range(max_new_tokens):
        cur = int(lengths.max())  # only forward up to the longest row so far
        logits = model(x[:, :cur])  # (B, cur, vocab)

        # gather each row's logit at ITS OWN last real position (lengths - 1):
        # here we have to zip 2 tensors, so that we have to use torch.arange
        last = logits[torch.arange(B, device=device), lengths - 1]  # (B, vocab), squeeze out the sequence dim

        # generate the next token id
        next_ids = sample_next(last, temperature=temperature, top_p=top_p)
        # update the input token
        x[torch.arange(B, device=device)[~finished], lengths[~finished]] = next_ids[~finished]
        lengths[~finished] += 1  # B
        mask = next_ids == eos_id  # B
        finished[mask] = True
        if bool(finished.all()):
            break

    # 5. decode each row up to its real length, drop a trailing EOS, return list[str]
    ans = []
    for b in range(B):
        seq = x[b, : lengths[b] - 1].tolist()
        if seq and seq[-1] == eos_id;
            seq = seq[:-1]
        ans.append(tokenizer.decode(seq))

    return ans


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"  # on mac, use cpu
    model, tokenizer, cfg = load_export(EXPORT_DIR, device)
    n_params = sum(p.numel() for p in model.parameters())

    # --- Single token generate ---
    # print(f"device={device}, params={n_params / 1e6:.1f}M, iter={cfg.get('source_iteration')}")
    # print("=" * 60)
    # print(generate(model, tokenizer, "Once upon a time", cfg, device=device))

    # --- batched demo (uncomment after you finish generate_batch) ---
    print("=" * 60)
    for i, out in enumerate(
        generate_batch(model, tokenizer, ["Once upon a time", "The little cat", "One day"], cfg, device=device)
    ):
        print(f"[{i}] {out}\n")
