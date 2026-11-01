"""A small character-level GPT, LoRA, batched generation and log-probability helpers.

Written out explicitly (no Hugging Face) so every post-training algorithm in notebooks 02-05
is a few visible lines on top of the same two functions: `generate` and `sequence_logprobs`.
"""
import math, os, time, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import tasks, CKPT_DIR, set_seed


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B A x, with W frozen, A random and B zero at initialisation (Hu et al., 2021)."""

    def __init__(self, base: nn.Linear, r=4, alpha=8):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T


class Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.n_head = n_head
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x, attn_mask, cache=None):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        q, k, v = (t.view(B, T, self.n_head, D // self.n_head).transpose(1, 2) for t in (q, k, v))
        if cache is not None:                      # key/value cache for fast generation
            if "k" in cache:
                k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.proj(h.transpose(1, 2).reshape(B, T, D))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class GPT(nn.Module):
    """Pre-LayerNorm decoder-only transformer (Vaswani et al., 2017; Xiong et al., 2020)."""

    def __init__(self, vocab=tasks.VOCAB_SIZE, d=128, n_layer=4, n_head=4, block=64):
        super().__init__()
        self.cfg = dict(vocab=vocab, d=d, n_layer=n_layer, n_head=n_head, block=block)
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(block, d)
        self.blocks = nn.ModuleList(Block(d, n_head) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight            # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, ids, pad_mask=None, return_hidden=False):
        """ids: (B, T). pad_mask: (B, T) bool, True for real tokens (needed only for left padding)."""
        B, T = ids.shape
        causal = torch.ones(T, T, dtype=torch.bool, device=ids.device).tril()
        if pad_mask is None:
            pos = torch.arange(T, device=ids.device).expand(B, T)
            mask = causal
        else:
            pos = (pad_mask.long().cumsum(1) - 1).clamp(min=0)
            eye = torch.eye(T, dtype=torch.bool, device=ids.device)
            mask = (causal & pad_mask[:, None, :]) | eye   # pad queries attend to themselves only
            mask = mask[:, None]
        x = self.tok(ids) + self.pos(pos)
        for blk in self.blocks:
            x = blk(x, mask)
        h = self.ln_f(x)
        return (self.head(h), h) if return_hidden else self.head(h)


def _step(model, ids, pos, mask, caches):
    """Run only the new tokens `ids` (B, T_new) through the model, reusing cached keys/values."""
    x = model.tok(ids) + model.pos(pos.clamp(max=model.cfg["block"] - 1))
    for blk, c in zip(model.blocks, caches):
        x = blk(x, mask, c)
    return model.head(model.ln_f(x))


def n_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def add_lora(model, r=4, alpha=8, targets=("qkv", "proj", "fc1", "fc2")):
    """Freeze the model and wrap the chosen linear layers of every block with LoRA adapters."""
    for p in model.parameters():
        p.requires_grad_(False)
    for blk in model.blocks:
        for name in targets:
            setattr(blk, name, LoRALinear(getattr(blk, name), r, alpha))
    return model


def merge_lora(model):
    """Fold B A into W so the adapted model has the original architecture again."""
    for blk in model.blocks:
        for name, mod in list(blk.named_children()):
            if isinstance(mod, LoRALinear):
                with torch.no_grad():
                    mod.base.weight += mod.scale * mod.B @ mod.A
                for p in mod.base.parameters():
                    p.requires_grad_(True)
                setattr(blk, name, mod.base)
    return model


# ---------------------------------------------------------------- data batching
def lm_batches(text, batch=64, block=64, rng=None):
    """Random contiguous windows from a corpus, for next-token pretraining."""
    data = torch.tensor(tasks.encode(text))
    gen = torch.Generator().manual_seed(int(rng.integers(1 << 31))) if rng is not None else None
    while True:
        ix = torch.randint(0, len(data) - block - 1, (batch,), generator=gen)
        x = torch.stack([data[i:i + block] for i in ix])
        y = torch.stack([data[i + 1:i + block + 1] for i in ix])
        yield x, y


def pack(prompts, responses, block=64):
    """Right-padded prompt+response ids, with a mask that is 1 on response tokens (incl. EOS).

    Returns ids (B, T), targets (B, T) shifted by one, and resp_mask (B, T) aligned with targets.
    """
    seqs, masks = [], []
    for p, r in zip(prompts, responses):
        pi, ri = tasks.encode(p), tasks.encode(r)
        s = (pi + ri)[:block + 1]
        m = ([0] * len(pi) + [1] * len(ri))[:block + 1]
        seqs.append(s); masks.append(m)
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), tasks.PAD_ID)
    mk = torch.zeros(len(seqs), T)
    for i, (s, m) in enumerate(zip(seqs, masks)):
        ids[i, :len(s)] = torch.tensor(s); mk[i, :len(m)] = torch.tensor(m, dtype=torch.float)
    return ids[:, :-1], ids[:, 1:], mk[:, 1:]


def sequence_logprobs(model, prompts, responses, per_token=False):
    """log pi(response | prompt), summed over response tokens (or per token with the mask)."""
    x, y, m = pack(prompts, responses, model.cfg["block"])
    logits = model(x)
    lp = torch.gather(F.log_softmax(logits.float(), -1), 2, y[..., None])[..., 0] * m
    return (lp, m) if per_token else lp.sum(1)


@torch.no_grad()
def generate(model, prompts, max_new=24, temperature=1.0, greedy=False, top_k=None, chunk=512):
    """Batched sampling with left padding and a key/value cache; each row stops at its first EOS.

    Prompts are left-padded so that every row's next token is generated at the same column.
    Positions are counted from each row's first real token, matching right-padded training.
    """
    if len(prompts) > chunk:
        return sum((generate(model, prompts[i:i + chunk], max_new, temperature, greedy, top_k, chunk)
                    for i in range(0, len(prompts), chunk)), [])
    model.eval()
    enc = [tasks.encode(p) for p in prompts]
    B, L = len(enc), max(len(e) for e in enc)
    ids = torch.full((B, L), tasks.PAD_ID)
    pm = torch.zeros(B, L, dtype=torch.bool)
    for i, e in enumerate(enc):
        ids[i, L - len(e):] = torch.tensor(e); pm[i, L - len(e):] = True
    caches = [dict() for _ in model.blocks]
    # prefill: causal attention over the prompt, ignoring left padding
    pos = (pm.long().cumsum(1) - 1).clamp(min=0)
    causal = torch.ones(L, L, dtype=torch.bool).tril()
    mask = ((causal & pm[:, None, :]) | torch.eye(L, dtype=torch.bool))[:, None]
    logits = _step(model, ids, pos, mask, caches)[:, -1].float()
    next_pos = pos[:, -1] + 1
    done = torch.zeros(B, dtype=torch.bool)
    out = torch.full((B, max_new), tasks.PAD_ID)
    for t in range(max_new):
        if greedy:
            nxt = logits.argmax(-1)
        else:
            logits = logits / temperature
            if top_k:
                v, _ = logits.topk(top_k); logits[logits < v[:, -1:]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, -1), 1)[:, 0]
        nxt = torch.where(done, torch.full_like(nxt, tasks.PAD_ID), nxt)
        out[:, t] = nxt
        done |= nxt == tasks.EOS_ID
        if done.all() or t == max_new - 1:
            break
        pm = torch.cat([pm, torch.ones(B, 1, dtype=torch.bool)], 1)
        logits = _step(model, nxt[:, None], next_pos[:, None], pm[:, None, None, :], caches)[:, -1].float()
        next_pos = next_pos + 1
    return [tasks.decode(o.tolist()) for o in out]


# ---------------------------------------------------------------- pretraining + checkpoints
def pretrain(model, text, steps=3000, batch=64, lr=1e-3, warmup=200, log_every=500, seed=0):
    import numpy as np
    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    it = lm_batches(text, batch, model.cfg["block"], np.random.default_rng(seed))
    losses, t0 = [], time.time()
    model.train()
    for step in range(steps):
        x, y = next(it)
        loss = F.cross_entropy(model(x).view(-1, model.cfg["vocab"]), y.reshape(-1), ignore_index=tasks.PAD_ID)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        losses.append(loss.item())
        if log_every and (step + 1) % log_every == 0:
            print(f"step {step + 1:5d}  loss {sum(losses[-log_every:]) / log_every:.3f}  ({time.time() - t0:.0f}s)")
    return losses


@torch.no_grad()
def corpus_loss(model, text, n_batches=20, batch=64, seed=1):
    import numpy as np
    model.eval()
    it = lm_batches(text, batch, model.cfg["block"], np.random.default_rng(seed))
    tot = 0.0
    for _ in range(n_batches):
        x, y = next(it)
        tot += F.cross_entropy(model(x).view(-1, model.cfg["vocab"]), y.reshape(-1)).item()
    return tot / n_batches


def save(model, name):
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({"cfg": model.cfg, "state": model.state_dict()}, os.path.join(CKPT_DIR, name + ".pt"))


def load(name):
    ck = torch.load(os.path.join(CKPT_DIR, name + ".pt"), weights_only=True)
    m = GPT(**ck["cfg"]); m.load_state_dict(ck["state"]); return m


def exists(name):
    return os.path.exists(os.path.join(CKPT_DIR, name + ".pt"))


PRETRAIN_KW = dict(steps=3000, batch=64, lr=1e-3)


def get_pretrained(verbose=True):
    """The base model from notebook 01 (trained on first call, then cached in checkpoints/)."""
    if exists("base_lm"):
        return load("base_lm")
    if verbose:
        print("checkpoints/base_lm.pt not found: pretraining it now (as in notebook 01) ...")
    m = GPT()
    pretrain(m, tasks.pretraining_corpus(), **PRETRAIN_KW, log_every=1000 if verbose else 0)
    save(m, "base_lm")
    return m
