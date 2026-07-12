"""Grouped-Query Attention on top of the KV cache (GQA + decode).

GQA: Hq query heads but only num_kv_heads (< Hq) key/value heads; each group of query heads
shares one K/V head. Shrinks the KV cache by Hq/num_kv_heads. Reuses the FROZEN blocks from
model.py; same forward signature as model_kv so `generate_cached` works unchanged.

  forward(in_indices, token_positions, past_kvs) -> (logits, new_kvs)

The block / LM / KV-cache plumbing is done (you already wrote it in model_kv).

Verify:  uv run --with safetensors python -m cs336_basics.model_gqa
"""

import torch
import torch.nn as nn
from einops import rearrange

from cs336_basics.model import (
    Embedding,
    Linear,
    RMSNorm,
    RoPE,
    SwiGLU,
    scaled_dot_product_attention,
)


class GroupedQueryAttentionKV(nn.Module):
    """Causal attention with num_kv_heads < num_heads, using + extending a KV cache."""

    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, rope: RoPE | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads
        self.q_proj = Linear(d_model, num_heads * self.d_k)
        self.k_proj = Linear(
            d_model, num_kv_heads * self.d_k
        )  # here the head dimension should be the same to support Q @ K^T
        self.v_proj = Linear(d_model, num_kv_heads * self.d_k)
        self.output_proj = Linear(d_model, d_model)
        self.rope = rope

    def forward(self, x, token_positions, past_kv=None):
        Q = rearrange(self.q_proj(x), "... s (h d) -> ... h s d", h=self.num_heads)  # (B, Hq, S, d_k)
        K = rearrange(self.k_proj(x), "... s (h d) -> ... h s d", h=self.num_kv_heads)
        V = rearrange(self.v_proj(x), "... s (h d) -> ... h s d", h=self.num_kv_heads)

        if self.rope is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        if past_kv is not None:
            past_k, past_v = past_kv
            K = torch.cat([past_k, K], dim=-2)
            V = torch.cat([past_v, V], dim=-2)

        new_kv = (K, V)  # cache stays small (num_kv_heads)

        expandK = torch.zeros([K.size(-4), self.num_heads, K.size(-2), K.size(-1)], device=device)
        expandV = torch.zeros([V.size(-4), self.num_heads, V.size(-2), V.size(-1)], device=device)
        r = self.num_heads // self.num_kv_heads
        for head_id in range(self.num_heads):
            real_id = head_id // r
            expandK[:, head_id, :, :] = K[:, real_id, :, :]
            expandV[:, head_id, :, :] = V[:, real_id, :, :]

        K = expandK
        V = expandV

        if past_kv is None:
            S = Q.shape[-2]
            mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=x.device))
        else:
            mask = None

        O = scaled_dot_product_attention(Q, K, V, mask=mask)  # (B, Hq, S, d_k)
        O = rearrange(O, "... h s d -> ... s (h d)")
        return self.output_proj(O), new_kv


class TransformerBlockGQA(nn.Module):
    """Pre-norm block threading the KV cache through GQA. (Done for you.)"""

    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, d_ff: int, rope: RoPE | None = None):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = GroupedQueryAttentionKV(d_model, num_heads, num_kv_heads, rope)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x, token_positions, past_kv=None):
        attn_out, new_kv = self.attn(self.ln1(x), token_positions, past_kv)
        y = x + attn_out
        out = y + self.ffn(self.ln2(y))
        return out, new_kv


class TransformerLMGQA(nn.Module):
    """Decoder-only LM with GQA, threading a LIST of per-layer caches. (Done for you.)"""

    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, num_kv_heads):
        super().__init__()
        d_k = d_model // num_heads
        rope = RoPE(d_k, rope_theta, context_length)
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [TransformerBlockGQA(d_model, num_heads, num_kv_heads, d_ff, rope) for _ in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, in_indices, token_positions, past_kvs=None):
        x = self.token_embeddings(in_indices)
        new_kvs = []
        for i, layer in enumerate(self.layers):
            past = None if past_kvs is None else past_kvs[i]
            x, kv = layer(x, token_positions, past)
            new_kvs.append(kv)
        x = self.ln_final(x)
        return self.lm_head(x), new_kvs


if __name__ == "__main__":
    import json

    from safetensors.torch import load_file

    from cs336_basics.generate_kv import EXPORT_DIR, MODEL_KEYS, generate_batch, generate_cached, load_export

    device = "cpu"
    base_model, tokenizer, cfg = load_export(EXPORT_DIR, device)
    prompt = "Once upon a time"

    def build_gqa(num_kv_heads, load_weights):
        m = TransformerLMGQA(**{k: cfg[k] for k in MODEL_KEYS}, num_kv_heads=num_kv_heads)
        if load_weights:
            m.load_state_dict(load_file(f"{EXPORT_DIR}/model.safetensors"), strict=False)
        return m.eval()

    print("=" * 60)
    # TEST 1 — num_kv_heads == num_heads => GQA is exactly MHA => must MATCH the baseline.
    gqa_full = build_gqa(cfg["num_heads"], load_weights=True)
    ans, _ = generate_batch(base_model, tokenizer, [prompt], cfg, temperature=0.0, device=device)
    cached = generate_cached(gqa_full, tokenizer, prompt, cfg, temperature=0.0, device=device)
    print(f"num_kv_heads={cfg['num_heads']} (==num_heads) -> MATCH baseline:", ans[0] == cached)

    # TEST 2 — num_kv_heads=4 => KV cache has 4 head-slots instead of 16 (random weights, shape only).
    gqa_small = build_gqa(4, load_weights=False)
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    _, cache = gqa_small(x, torch.arange(len(ids), device=device), None)
    k0 = cache[0][0]  # layer 0's cached K
    print(f"num_kv_heads=4 -> per-layer cached K shape: {tuple(k0.shape)}  (heads dim = 4, was 16 for MHA)")
    print("=" * 60)
