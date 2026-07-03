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


def sample_next(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    """From the last-position logits (shape (vocab,)), pick the next token id."""
    #   - greedy (temperature <= 0):  return int(logits.argmax())
    #   - else:  probs = softmax(logits / temperature, dim=-1)      # <1 sharper, >1 more random
    #   - top-p / nucleus (if top_p < 1): keep the smallest set of tokens whose cumulative
    #       prob >= top_p, then sample only from those:
    #         sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    #         cumsum = torch.cumsum(sorted_probs, dim=-1)
    #         keep = (cumsum - sorted_probs) < top_p
    #         zero the rest, renormalize (/ sum), torch.multinomial, map back via sorted_idx
    #   - no top-p:  int(torch.multinomial(probs, num_samples=1))
    # operate on the temperature
    if temperature <= 0:
        return int(logits.argmax())  # argmax returns the largest item's idx
    p = softmax(logits / temperature, dim=-1)
    if top_p < 1:
        # so here we need to find the cumulative amount which reaches `top_p`
        sorted_p, sorted_idx = torch.sort(p, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_p, dim=-1)
        mask = (
            cumsum - sorted_p <= top_p
        )  # NOTE: since we need to find the smallest set which just > top_p, directly comparing only find the last one which is < top_p but we need the next one, so here we need to have a `minus self trick` to represent the last cumsum item

        # find the cumsum/norm_rate for top_p
        p_filled = torch.masked_fill(sorted_p, mask=~mask, value=0.0)
        rate = torch.sum(p_filled, dim=-1)
        p_normed = p_filled / rate
        choice = torch.multinomial(p_normed, num_samples=1)
        return int(sorted_idx[choice])
    choice = torch.multinomial(p, num_samples=1)
    return int(choice)


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


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"  # on mac, use cpu
    model, tokenizer, cfg = load_export(EXPORT_DIR, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}, params={n_params / 1e6:.1f}M, iter={cfg.get('source_iteration')}")
    print("=" * 60)
    print(generate(model, tokenizer, "Once upon a time", cfg, device=device))
