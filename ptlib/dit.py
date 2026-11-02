"""Diffusion transformers for 8x8 digits (notebooks 09-10).

DiT  (Peebles & Xie, 2023): patch tokens + adaLN-Zero conditioning on (time, class).
MMDiT-style text model (Esser et al., 2024): text tokens and image tokens attend jointly, each stream with
its own weights; the pooled text embedding and time also modulate every block through adaLN.
Both take x as a flat (B, 64) vector and return a (B, 64) prediction, so they drop into ptlib.diffusion.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .diffusion import timestep_embedding

P = 2                     # patch size: 8x8 image -> 4x4 = 16 tokens of 2*2 = 4 pixels


def patchify(x):
    B = x.shape[0]
    return x.view(B, 4, P, 4, P).permute(0, 1, 3, 2, 4).reshape(B, 16, P * P)


def unpatchify(t):
    B = t.shape[0]
    return t.view(B, 4, 4, P, P).permute(0, 1, 3, 2, 4).reshape(B, 64)


def modulate(h, shift, scale):
    return h * (1 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    """h <- h + g1 * Attn(mod(LN(h))) ; h <- h + g2 * MLP(mod(LN(h))), with shift/scale/gate predicted from c."""

    def __init__(self, d, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[1].weight); nn.init.zeros_(self.ada[1].bias)      # adaLN-Zero: block starts as identity

    def forward(self, h, c):
        s1, sc1, g1, s2, sc2, g2 = self.ada(c).chunk(6, -1)
        a = modulate(self.ln1(h), s1, sc1)
        h = h + g1[:, None] * self.attn(a, a, a, need_weights=False)[0]
        return h + g2[:, None] * self.mlp(modulate(self.ln2(h), s2, sc2))


class DiT(nn.Module):
    def __init__(self, d=128, depth=4, heads=4, n_classes=11):
        super().__init__()
        self.d_in = 64
        self.patch = nn.Linear(P * P, d)
        self.pos = nn.Parameter(torch.zeros(1, 16, d)); nn.init.normal_(self.pos, std=0.02)
        self.temb = nn.Sequential(nn.Linear(128, d), nn.SiLU(), nn.Linear(d, d))
        self.cemb = nn.Embedding(n_classes, d)
        self.blocks = nn.ModuleList(DiTBlock(d, heads) for _ in range(depth))
        self.ln_f = nn.LayerNorm(d, elementwise_affine=False)
        self.ada_f = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.out = nn.Linear(d, P * P)
        for m in [self.ada_f[1], self.out]:
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def cond(self, t, c):
        return self.temb(timestep_embedding(t)) + self.cemb(c)

    def forward(self, x, t, c):
        e = self.cond(t, c)
        h = self.patch(patchify(x)) + self.pos
        for blk in self.blocks:
            h = blk(h, e)
        shift, scale = self.ada_f(e).chunk(2, -1)
        return unpatchify(self.out(modulate(self.ln_f(h), shift, scale)))


# ---------------------------------------------------------------- text conditioning (joint attention, MM-DiT style)
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
TEXT_CHARS = " abcdefghijklmnopqrstuvwxyz"
TEXT_LEN = 12


def encode_captions(captions):
    """Characters -> ids (0 = padding); returns (B, TEXT_LEN) long tensor."""
    ids = torch.zeros(len(captions), TEXT_LEN, dtype=torch.long)
    for i, s in enumerate(captions):
        for j, ch in enumerate(s[:TEXT_LEN]):
            ids[i, j] = TEXT_CHARS.index(ch) + 1
    return ids


class JointBlock(nn.Module):
    """Two streams (text, image) with separate weights; attention runs over the concatenated sequence."""

    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln = nn.ModuleDict({k: nn.LayerNorm(d, elementwise_affine=False) for k in ["t1", "i1", "t2", "i2"]})
        self.qkv = nn.ModuleDict({k: nn.Linear(d, 3 * d) for k in ["t", "i"]})
        self.proj = nn.ModuleDict({k: nn.Linear(d, d) for k in ["t", "i"]})
        self.mlp = nn.ModuleDict({k: nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)) for k in ["t", "i"]})
        self.ada = nn.ModuleDict({k: nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d)) for k in ["t", "i"]})
        for k in ["t", "i"]:
            nn.init.zeros_(self.ada[k][1].weight); nn.init.zeros_(self.ada[k][1].bias)

    def forward(self, ht, hi, c, text_mask):
        B, Lt, D = ht.shape
        mt, mi = self.ada["t"](c).chunk(6, -1), self.ada["i"](c).chunk(6, -1)
        qt, kt, vt = self.qkv["t"](modulate(self.ln["t1"](ht), mt[0], mt[1])).chunk(3, -1)
        qi, ki, vi = self.qkv["i"](modulate(self.ln["i1"](hi), mi[0], mi[1])).chunk(3, -1)
        q, k, v = (torch.cat(z, 1).view(B, -1, self.heads, D // self.heads).transpose(1, 2) for z in [(qt, qi), (kt, ki), (vt, vi)])
        keymask = torch.cat([text_mask, torch.ones(B, hi.shape[1], dtype=torch.bool)], 1)[:, None, None, :]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=keymask).transpose(1, 2).reshape(B, -1, D)
        at, ai = a[:, :Lt], a[:, Lt:]
        ht = ht + mt[2][:, None] * self.proj["t"](at)
        hi = hi + mi[2][:, None] * self.proj["i"](ai)
        ht = ht + mt[5][:, None] * self.mlp["t"](modulate(self.ln["t2"](ht), mt[3], mt[4]))
        hi = hi + mi[5][:, None] * self.mlp["i"](modulate(self.ln["i2"](hi), mi[3], mi[4]))
        return ht, hi


class TextDiT(nn.Module):
    """Image tokens + caption tokens with joint attention. `c` is a (B, TEXT_LEN) tensor of caption ids
    (all zeros = the empty caption, used for classifier-free guidance)."""

    def __init__(self, d=128, depth=4, heads=4):
        super().__init__()
        self.d_in = 64
        self.patch = nn.Linear(P * P, d)
        self.pos = nn.Parameter(torch.zeros(1, 16, d)); nn.init.normal_(self.pos, std=0.02)
        self.tok = nn.Embedding(len(TEXT_CHARS) + 1, d)
        self.tpos = nn.Parameter(torch.zeros(1, TEXT_LEN, d)); nn.init.normal_(self.tpos, std=0.02)
        self.temb = nn.Sequential(nn.Linear(128, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(JointBlock(d, heads) for _ in range(depth))
        self.ln_f = nn.LayerNorm(d, elementwise_affine=False)
        self.ada_f = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.out = nn.Linear(d, P * P)
        for m in [self.ada_f[1], self.out]:
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, t, c):
        mask = c > 0
        mask[:, 0] = True                                         # keep one key so empty captions are valid
        ht = self.tok(c) + self.tpos
        pooled = (ht * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        e = self.temb(timestep_embedding(t)) + pooled
        hi = self.patch(patchify(x)) + self.pos
        for blk in self.blocks:
            ht, hi = blk(ht, hi, e, mask)
        shift, scale = self.ada_f(e).chunk(2, -1)
        return unpatchify(self.out(modulate(self.ln_f(hi), shift, scale)))


def caption_dataset():
    """Notebook 09's captioned data: every digit as '<word>', and in bold as 'bold <word>' -- except bold sevens."""
    from . import images
    Xtr, ytr, _, _ = images.load_digits_tensor()
    keep = ytr != 7
    X = torch.cat([Xtr, images.bold(Xtr)[keep]])
    caps = [WORDS[int(c)] for c in ytr] + ["bold " + WORDS[int(c)] for c in ytr[keep]]
    return X, caps


def get_text_dit(verbose=True):
    """Notebook 09's text-conditioned DiT (trained on first use with the same settings, then cached)."""
    import os
    from . import CKPT_DIR
    from . import diffusion as dfx
    path = os.path.join(CKPT_DIR, "text_dit.pt")
    m = TextDiT()
    if os.path.exists(path):
        m.load_state_dict(torch.load(path, weights_only=True)); return m.eval()
    if verbose:
        print("checkpoints/text_dit.pt not found: training notebook 09's text-conditioned DiT (~5 min) ...")
    X, caps = caption_dataset()
    ema, _ = dfx.train_flow(TextDiT(), X, encode_captions(caps), steps=3000, batch=128, t_sampling="logit_normal",
                            log_every=1000 if verbose else 0)
    os.makedirs(CKPT_DIR, exist_ok=True); torch.save(ema.state_dict(), path)
    return ema.eval()
