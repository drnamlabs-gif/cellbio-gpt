# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Decoder-only transformer (GPT) from scratch in pure PyTorch.

No nn.Transformer, no nn.MultiheadAttention. Attention is written out as
softmax(Q K^T / sqrt(d_head) + causal_mask) V, plus the modern minimal stack:

- RoPE: positions enter as rotations of Q/K pairs, not learned embeddings
- RMSNorm pre-norm, SwiGLU MLP, no biases (Llama-style)
- weight tying: token embedding doubles as the output head
- KV-cache: generation reuses past keys/values, O(1) new work per token

`fused=True` routes full-sequence attention through
F.scaled_dot_product_attention (flash kernel) for stage-3 speed; the manual
path is the reference. overfit_one_batch.py asserts the two agree.
"""

import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass
class GPTConfig:
    vocab_size: int = 16384
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 256
    max_seq: int = 1024
    fused: bool = False


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.g * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def rope_tables(max_seq, d_head, device, base=10000.0):
    inv = 1.0 / base ** (torch.arange(0, d_head, 2, device=device).float() / d_head)
    t = torch.arange(max_seq, device=device).float()
    freqs = torch.outer(t, inv)                    # (max_seq, d_head/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin, pos0=0):
    """x: (B, H, T, Dh). Rotate even/odd channel pairs by position angle."""
    T = x.size(2)
    c = cos[pos0:pos0 + T][None, None]
    s = sin[pos0:pos0 + T][None, None]
    x1, x2 = x[..., ::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., ::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.d_head = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.last_att = None       # stored when store_att=True (for figures)

    def forward(self, x, cos, sin, cache=None, store_att=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.cfg.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.cfg.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.cfg.n_head, self.d_head).transpose(1, 2)

        pos0 = cache[0].size(2) if cache is not None else 0
        q = apply_rope(q, cos, sin, pos0)
        k = apply_rope(k, cos, sin, pos0)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k, v)

        if self.cfg.fused and cache is None and not store_att:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
            Tk = k.size(2)
            mask = torch.tril(torch.ones(T, Tk, device=x.device, dtype=torch.bool),
                              diagonal=pos0)
            att = att.masked_fill(~mask, float("-inf")).softmax(dim=-1)
            if store_att:
                self.last_att = att.detach()
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), new_cache


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = int(8 * cfg.d_model / 3 / 64 + 1) * 64   # ~2.67x, multiple of 64
        self.w1 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w3 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.attn = RMSNorm(cfg.d_model), CausalSelfAttention(cfg)
        self.norm2, self.mlp = RMSNorm(cfg.d_model), SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None, store_att=False):
        a, new_cache = self.attn(self.norm1(x), cos, sin, cache, store_att)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, new_cache


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.emb.weight            # weight tying
        cos, sin = rope_tables(cfg.max_seq, cfg.d_model // cfg.n_head, "cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for name, p in self.named_parameters():       # scaled residual init
            if name.endswith(("proj.weight", "w2.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters()) - self.head.weight.numel() \
            + self.head.weight.numel()                # tied weight counted once

    def forward(self, idx, targets=None, caches=None, store_att=False):
        x = self.emb(idx)
        new_caches = []
        for i, blk in enumerate(self.blocks):
            x, c = blk(x, self.rope_cos, self.rope_sin,
                       caches[i] if caches is not None else None, store_att)
            new_caches.append(c)
        logits = self.head(self.norm_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, new_caches

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=0.0, top_k=None, use_cache=True):
        """Greedy when temperature=0. use_cache=False recomputes the full
        sequence each step (slow reference path for the cache parity test)."""
        self.eval()
        caches = None
        feed = idx
        for _ in range(max_new):
            if use_cache:
                logits, _, caches = self.forward(feed, caches=caches)
            else:
                logits, _, _ = self.forward(idx)
            logits = logits[:, -1, :]
            if temperature > 0:
                logits = logits / temperature
                if top_k:
                    kth = torch.topk(logits, top_k).values[:, -1, None]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(logits.softmax(-1), 1)
            else:
                nxt = logits.argmax(-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
            feed = nxt
        return idx
