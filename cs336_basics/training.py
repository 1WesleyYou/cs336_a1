"""Training utilities: loss, optimizer, LR schedule, gradient clipping, data, checkpoints.

You implement these by hand (fill in the TODOs), then run the tests.
"""

import math

import numpy as np
import torch


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Average cross-entropy loss.

    inputs:  (..., vocab_size) unnormalized logits (one row of scores per example)
    targets: (...) int, the correct class index for each example
    returns: scalar — mean loss over all examples.

    CE = -log( softmax(logits)[target] ) = logsumexp(logits) - logits[target]
    """
    # Do it the NUMERICALLY-STABLE way — do NOT softmax() then log().
    m = inputs.max(dim=-1, keepdim=True).values  # (..., 1)
    # calculated logit
    logsumexp = m.squeeze(-1) + torch.log(torch.sum(torch.exp(inputs - m), dim=-1, keepdim=False))  # (...)
    # target logit score
    target_logit = torch.gather(inputs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # (...)
    loss = logsumexp - target_logit
    return loss.mean()


class AdamW(torch.optim.Optimizer):
    """AdamW = Adam (adaptive per-parameter step from 1st/2nd moment estimates)
    + decoupled weight decay.

    Per parameter we keep running estimates m (mean of grads) and v (mean of grad^2),
    plus a step counter t. The boilerplate below is done for you — fill the 5-step
    update in step().
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]  # avoid dividing 0
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data  # this parameter's gradient
                state = self.state[p]  # per-parameter persistent state
                if len(state) == 0:
                    # first time: initialize m, v, t
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                m, v = state["m"], state["v"]
                state["t"] += 1
                t = state["t"]
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * (g * g)
                state["m"], state["v"] = m, v
                alpha_t = lr * (1 - beta2**t) ** 0.5 / (1 - beta1**t)  # real step length
                p.data = p.data - alpha_t * m / (v.sqrt() + eps)  # adam
                p.data = p.data - lr * weight_decay * p.data  # decay
        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Cosine learning-rate schedule with linear warmup. Returns the lr to use at step `it`."""
    # Return the lr for whichever of the 3 phases `it` falls in:
    if it < warmup_iters:
        # WARMUP phase
        return it / warmup_iters * max_learning_rate
    elif warmup_iters <= it and it <= cosine_cycle_iters:
        # COSINE phase
        frac = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * frac)) * (max_learning_rate - min_learning_rate)
    else:
        # AFTER phase
        return min_learning_rate


def gradient_clipping(parameters, max_l2_norm: float, eps: float = 1e-6) -> None:
    """Clip the COMBINED L2 norm of all params' gradients to <= max_l2_norm, in-place.

    Treat all gradients as one big vector: if its L2 norm exceeds max_l2_norm,
    scale every gradient down by the same factor so the total norm == max_l2_norm.
    """
    #   1. collect the grads:  grads = [p.grad for p in parameters if p.grad is not None]
    #   2. total_norm over ALL grads together:
    #        total_norm = torch.sqrt(sum((g ** 2).sum() for g in grads))
    #   3. if total_norm > max_l2_norm:
    #        scale = max_l2_norm / (total_norm + eps)
    #        for g in grads: g.mul_(scale)          # in-place — modifies p.grad
    grads = [p.grad for p in parameters if p.grad is not None]
    tot_norm = torch.sqrt(torch.stack([(g**2).sum() for g in grads]).sum())
    if tot_norm > max_l2_norm:
        scale = max_l2_norm / (tot_norm + eps)
        for g in grads:
            g.mul_(scale)


def get_batch(dataset, batch_size: int, context_length: int, device: str):
    """Sample a batch of (input, target) sequences for language-model training.

    dataset: 1D array of token IDs (the whole corpus).
    Returns (inputs, targets), each a (batch_size, context_length) LongTensor on `device`.
    target = input shifted by 1 (the next token to predict at each position).
    """
    n = len(dataset)
    # context length means the size of token each time the model trains on
    # starts means the starting positions for the training tokens, size is batch_size
    starts = np.random.randint(0, n - context_length, size=batch_size)
    inputs = np.stack([dataset[i : i + context_length] for i in starts])
    targets = np.stack([dataset[i + 1 : i + 1 + context_length] for i in starts])

    # so here the target input and targets corresponds to the vocab idx, so it should be int/long
    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs, targets


def save_checkpoint(model, optimizer, iteration, out) -> None:
    """Serialize model + optimizer + iteration to `out` (a path or file-like object)."""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),  # adamw's m/v/t param group
        "iteration": iteration,  # the step id
    }
    torch.save(checkpoint, out)


def load_checkpoint(src, model, optimizer) -> int:
    """Restore model + optimizer from `src`; return the saved iteration number."""
    #   1. checkpoint = torch.load(src)
    #   2. model.load_state_dict(checkpoint["model"])
    #   3. optimizer.load_state_dict(checkpoint["optimizer"])
    #   4. return checkpoint["iteration"]
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
