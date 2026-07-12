"""Cache-aware Transformer for KV-cache inference (the real prefill/decode split).

Reuses the FROZEN building blocks from model.py (Linear / RMSNorm / SwiGLU / RoPE / SDPA);
only attention / block / LM `forward` change to thread a per-layer KV cache. Same submodule
names as model.py, so it loads the SAME exported weights.

  forward(in_indices, token_positions, past_kvs) -> (logits, new_kvs)
    prefill:  past_kvs=None            -> process the whole prompt, build the cache
    decode :  past_kvs=list[(K,V)]     -> feed 1 token, reuse+extend the cache

YOUR job: fill in the KV cache inside MultiHeadSelfAttentionKV.forward (the 3 TODOs).
Everything else is done.
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


class MultiHeadSelfAttentionKV(nn.Module):
    """Causal multi-head self-attention that can use + extend a KV cache.

    x is (B, S, d_model):  S = prompt_len during PREFILL, S = 1 during DECODE.
    but here we only depend on whether the past_kv is None to judge if it's decode phase

    if this is in the decode phase, then the input tensor should only be the last one

    Returns (out, new_kv) where new_kv=(K, V) is the cache to feed back next step.
    """

    def __init__(self, d_model: int, num_heads: int, rope: RoPE | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        self.rope = rope

    def forward(self, x, token_positions, past_kv=None):
        Q = rearrange(self.q_proj(x), "... s (h d) -> ... h s d", h=self.num_heads)  # (B, H, S, d_k)
        K = rearrange(self.k_proj(x), "... s (h d) -> ... h s d", h=self.num_heads)
        V = rearrange(self.v_proj(x), "... s (h d) -> ... h s d", h=self.num_heads)

        # --- RoPE the NEW tokens' Q and K, using their absolute positions, BEFORE caching ---
        if self.rope is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # update the k and v with past data
        if past_kv is not None:
            past_k, past_v = past_kv
            K = torch.cat([past_k, K], dim=-2)
            V = torch.cat([past_v, V], dim=-2)

        new_kv = (K, V)

        if past_kv is None:
            # PREFILL phase
            # prepare the causal mask for the whole prompt to understand itself
            S = Q.shape[-2]
            mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=x.device))
        else:
            # DECODE phase
            # query only pick the last token (should be handled outside of this function)
            mask = None

        O = scaled_dot_product_attention(Q, K, V, mask=mask)  # (B, H, S, d_k)
        O = rearrange(O, "... h s d -> ... s (h d)")  # (B, S, d_model)
        return self.output_proj(O), new_kv


class TransformerBlockKV(nn.Module):
    """Pre-norm block, threading the KV cache through attention. (Done for you.)"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RoPE | None = None):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttentionKV(d_model, num_heads, rope)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x, token_positions, past_kv=None):
        attn_out, new_kv = self.attn(self.ln1(x), token_positions, past_kv)  # attn now returns a cache
        y = x + attn_out
        out = y + self.ffn(self.ln2(y))
        return out, new_kv


class TransformerLMKV(nn.Module):
    """Decoder-only LM threading a LIST of per-layer caches. (Done for you.)"""

    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta):
        super().__init__()
        d_k = d_model // num_heads
        rope = RoPE(d_k, rope_theta, context_length)  # one shared RoPE (no params, just cos/sin)
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TransformerBlockKV(d_model, num_heads, d_ff, rope) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, in_indices, token_positions, past_kvs=None):
        # in_indices: (B, S);  past_kvs: None (prefill) or list of (K, V) per layer (decode)
        x = self.token_embeddings(in_indices)  # (B, S, d_model)
        new_kvs = []
        for i, layer in enumerate(self.layers):
            past = None if past_kvs is None else past_kvs[i]  # each layer has its OWN cache
            x, kv = layer(x, token_positions, past)
            new_kvs.append(kv)
        x = self.ln_final(x)
        return self.lm_head(x), new_kvs  # logits (B, S, vocab), updated per-layer caches
