"""Transformer language model components, built from scratch.

Warm-up modules (Linear / Embedding / RMSNorm): you implement these by hand.
Fill in every TODO, then run the tests to check yourself.
"""

import torch
import torch.nn as nn
from einops import rearrange


class Linear(nn.Module):
    """y = x @ W^T, no bias. Weight is stored as (d_out, d_in)."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        #   - allocate with torch.empty(d_out, d_in)
        #   - init in place with truncated normal: std = sqrt(2 / (d_in + d_out)),
        #     truncated to [-3*std, 3*std]  (see nn.init.trunc_normal_)
        self.weight = nn.Parameter(torch.empty(d_out, d_in))

        # truncated normalization
        # so this will generate random numbers that follow the normal distribution (mu = 0, sigma = std) and within \pm 3std
        std = (2 / (d_in + d_out)) ** 0.5
        nn.init.trunc_normal_(self.weight, 0, std, -3 * std, 3 * std)
        self.d_in = d_in
        self.d_out = d_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the x -> f(x):= xW^T"""
        # x: (..., d_in)  ->  (..., d_out)
        return x @ self.weight.T


class Embedding(nn.Module):
    """Token id -> vector lookup. Weight is (vocab_size, d_model)."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        # TODO: `self.weight` = nn.Parameter of shape (vocab_size, d_model),
        #   init with trunc_normal_(mean=0, std=1, a=-3, b=3).
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)
        self.vocab_size = vocab_size
        self.d_model = d_model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (...) int  ->  (..., d_model)
        #   (fancy indexing: self.weight[token_ids] keeps the leading dims).
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """RMSNorm(x) = x / sqrt(mean(x^2) + eps) * g, with learnable gain g."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(
                d_model,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NUMERICAL CARE:
        #   1. remember x.dtype, then upcast x to float32 before squaring
        #      (fp16/bf16 can overflow when you square large activations).
        #   2. rms = sqrt( mean(x^2 over the LAST dim, keepdim=True) + eps )
        #   3. out = (x / rms) * self.weight
        #   4. cast `out` back to the original dtype before returning.
        ini_type = x.dtype
        x = x.to(dtype=torch.float32)
        ans = (x / (torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps) ** 0.5) * self.weight
        return ans.to(dtype=ini_type)


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU(x) = x * sigmoid(x), applied element-wise (no params, shape unchanged)."""
    #   - torch.sigmoid(x) gives the sigmoid; the multiply is element-wise.
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """Position-wise feed-forward network with a SwiGLU gate.

    SwiGLU(x) = W2 ( SiLU(W1 x) ⊙ W3 x )
      - W1, W3: project d_model -> d_ff   (gate branch & value branch)
      - W2:     project d_ff   -> d_model (back down)
      - ⊙ is element-wise multiply (this is the "gating")
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        #   EXACT attribute names (so it can load weights into them):
        #     self.w1 = Linear(d_model, d_ff)   # up-project, gate branch
        #     self.w3 = Linear(d_model, d_ff)   # up-project, value branch
        #     self.w2 = Linear(d_ff, d_model)   # down-project
        #   Reuse YOUR Linear class above — its weight is stored (d_out, d_in),
        #   which already matches the w*_weight shapes the test feeds in.
        self.w1 = Linear(d_model, d_ff)
        self.w3 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.d_model = d_model
        self.d_ff = d_ff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model) -> (..., d_model)
        #   - w1(x), w3(x): (..., d_ff)
        #   - silu(w1(x)) * w3(x): element-wise gate, still (..., d_ff)
        #   - w2(...): back to (..., d_model)
        x1 = self.w1(x)
        x3 = self.w3(x)
        inner = silu(x1) * x3
        return self.w2(inner)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Numerically-stable softmax along `dim`. Output sums to 1 along `dim`."""
    #   1. subtract the max along `dim` (keepdim=True) BEFORE exp — for numerical
    #      stability (the largest exponent becomes 0, so exp can't overflow).
    #      This does NOT change the result: exp(x-c)/sum(exp(x-c)) == exp(x)/sum(exp(x)).
    #      max along a dim:  x.max(dim=dim, keepdim=True).values
    #   2. exp the shifted values, then divide by their sum along `dim` (keepdim=True).
    m = torch.max(x, dim=dim, keepdim=True)  # use the flashattention style of naming
    xm = x - m.values
    l = torch.exp(xm).sum(dim=dim, keepdim=True)
    return torch.exp(xm) / l


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Attention(Q, K, V) = softmax( Q Kᵀ / sqrt(d_k) ) V

    Shapes:
        Q: (..., queries, d_k)   K: (..., keys, d_k)   V: (..., keys, d_v)
        mask: (..., queries, keys) bool, or None
              True  = this query MAY attend to this key
              False = blocked → its score is set to -inf before softmax
        returns: (..., queries, d_v)
    """
    #   1. d_k = Q.shape[-1]
    #   2. scores = Q @ Kᵀ / sqrt(d_k)
    #        - transpose K's LAST TWO dims: K.transpose(-2, -1) → (..., d_k, keys)
    #        - scores shape: (..., queries, keys)
    #   3. if mask is not None: where mask is False, set scores to -inf
    #        - e.g. scores = scores.masked_fill(~mask, float("-inf"))
    #   4. attn = softmax over the KEYS dimension (dim=-1)   ← reuse softmax() above
    #   5. return attn @ V        → (..., queries, d_v)
    KT = K.transpose(-2, -1)
    S = Q @ KT
    d_model = Q.shape[-1]  # hidden dimension
    S_Normed = S / (d_model**0.5)
    if mask is not None:
        S_Normed = S_Normed.masked_fill(~mask, float("-inf"))
    P = softmax(S_Normed, dim=-1)
    O = P @ V
    return O


class RoPE(nn.Module):
    """Rotary Position Embedding.

    Rotates each ADJACENT pair of dims (x[2i], x[2i+1]) of a query/key vector by an
    angle = position * theta_i, so the QKᵀ dot product ends up depending only on the
    RELATIVE distance between tokens.
    """

    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(self, d_k: int, theta: float, max_seq_len: int):
        super().__init__()
        # Precompute cos/sin for every (position, frequency-pair) up to max_seq_len.
        #   1. inv_freq[i] = 1 / theta ** (2i / d_k),  i = 0 .. d_k/2 - 1   → shape (d_k/2,)
        #        hint: the exponent 2i is torch.arange(0, d_k, 2); cast to float, divide by d_k.
        #   2. positions = torch.arange(max_seq_len)                        → shape (max_seq_len,)
        #   3. angles[p, i] = positions[p] * inv_freq[i]   (outer product)  → (max_seq_len, d_k/2)
        #        hint: positions[:, None] * inv_freq[None, :]
        #   4. cache cos & sin so they aren't trained but DO move with .to(device):
        #        self.register_buffer("cos", torch.cos(angles), persistent=False)
        #        self.register_buffer("sin", torch.sin(angles), persistent=False)
        inv_freq = 1 / theta ** (torch.arange(0, d_k, 2) / d_k)
        inv_freq = inv_freq[None, :]  # col
        positions = torch.arange(0, max_seq_len)[:, None]  # row
        self.angles = positions * inv_freq
        self.register_buffer("cos", torch.cos(self.angles), persistent=False)
        self.register_buffer("sin", torch.sin(self.angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k)     token_positions: (..., seq_len)
        #   1. look up cos/sin for each token's position (fancy index the cached buffers):
        #        cos = self.cos[token_positions]    → (..., seq_len, d_k/2)   (sin likewise)
        #   2. split x into adjacent pairs (even / odd dims):
        #        x_even = x[..., 0::2]   x_odd = x[..., 1::2]   → each (..., seq_len, d_k/2)
        #   3. rotate each pair (2-D rotation):
        #        out_even = x_even * cos - x_odd * sin
        #        out_odd  = x_even * sin + x_odd * cos
        #   4. interleave even/odd back to (..., seq_len, d_k):
        #        one way: torch.stack([out_even, out_odd], dim=-1).flatten(-2)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        out_even = x_even * self.cos[token_positions] - x_odd * self.sin[token_positions]
        out_odd = x_odd * self.cos[token_positions] + x_even * self.sin[token_positions]

        return torch.stack([out_even, out_odd], dim=-1).flatten(-2)


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention.

    Split d_model into `num_heads` heads, run scaled-dot-product attention per head
    (causal mask + optional RoPE on Q/K), then merge the heads and project out.
    """

    def __init__(self, d_model: int, num_heads: int, rope: RoPE | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        #     self.q_proj      = Linear(d_model, d_model)
        #     self.k_proj      = Linear(d_model, d_model)
        #     self.v_proj      = Linear(d_model, d_model)
        #     self.output_proj = Linear(d_model, d_model)
        # and keep the (optional) rope module:
        #     self.rope = rope
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        self.rope = rope

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # x: (..., seq, d_model)
        seq = x.shape[-2]
        #   1. project:  Q = self.q_proj(x), K = self.k_proj(x), V = self.v_proj(x)
        #                → each (..., seq, d_model)
        #   2. split into heads with einops:
        #        Q = rearrange(Q, "... seq (h d) -> ... h seq d", h=self.num_heads)
        #        → (..., num_heads, seq, d_k)      (same for K, V)
        #   3. if self.rope is not None: apply RoPE to Q and K
        #        (if token_positions is None: token_positions = torch.arange(seq, device=x.device))
        #        Q = self.rope(Q, token_positions)      (K likewise)
        #   4. causal mask (seq, seq), True = allowed (query i may attend to key j<=i):
        #        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=x.device))
        #   5. attn = scaled_dot_product_attention(Q, K, V, mask)  → (..., num_heads, seq, d_k)
        #   6. merge heads back:
        #        out = rearrange(attn, "... h seq d -> ... seq (h d)")  → (..., seq, d_model)
        #   7. return self.output_proj(out)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q = rearrange(Q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        K = rearrange(K, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        V = rearrange(V, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)

        if self.rope is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=x.device))

        O = scaled_dot_product_attention(Q, K, V, mask=mask)

        O = rearrange(O, "... num_heads seq_len d_k -> ... seq_len (num_heads d_k)")

        return self.output_proj(O)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block.

        y   = x + attn(ln1(x))     # attention sub-layer (norm on the INPUT)
        out = y + ffn(ln2(y))      # feed-forward sub-layer
    The residual path (x, y) stays clean — RMSNorm only sits on each sub-layer's input.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RoPE | None = None):
        super().__init__()
        #     self.ln1  = RMSNorm(d_model)
        #     self.attn = MultiHeadSelfAttention(d_model, num_heads, rope)
        #     self.ln2  = RMSNorm(d_model)
        #     self.ffn  = SwiGLU(d_model, d_ff)
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope)
        self.ln2 = RMSNorm(d_model)  # use double norm
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # x: (..., seq, d_model)
        # (your MHA needs real positions, so default them here if not given)
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device)
        #   y = x + self.attn(self.ln1(x), token_positions)
        #   return y + self.ffn(self.ln2(y))
        y = x + self.attn(self.ln1(x), token_positions)
        return y + self.ffn(self.ln2(y))


class TransformerLM(nn.Module):
    """Decoder-only Transformer language model.

    token_embeddings → num_layers × TransformerBlock → ln_final → lm_head → logits
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        d_k = d_model // num_heads
        rope = RoPE(d_k, rope_theta, context_length)  # one shared RoPE for all layers
        self.token_embeddings = Embedding(vocab_size, d_model)  # return the corresponding token vector
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, rope) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model)  # normed and then to pick from the vocab
        self.lm_head = Linear(d_model, vocab_size)  # pick from the vocab

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        # in_indices: (batch, seq) int token ids  →  logits (batch, seq, vocab_size)
        x = self.token_embeddings(in_indices)

        for l in self.layers:
            x = l(x)

        x = self.ln_final(x)
        return self.lm_head(x)  # only get the token choice in the end
