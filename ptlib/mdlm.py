"""A masked diffusion language model for the arithmetic task (notebook 10).

Forward process: each answer token is independently replaced by [MASK] with probability t ~ U(0, 1).
The model, a *bidirectional* transformer, predicts the original token at every masked position; the training loss is
cross-entropy on masked positions weighted by 1/t, which is the MDLM ELBO (Sahoo et al., 2024; Shi et al., 2024).
Generation starts from an all-[MASK] answer and unmasks the most confident positions step by step (as in LLaDA,
Nie et al., 2025). The prompt is never masked.
"""
import math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import tasks, set_seed

MASK = tasks.VOCAB_SIZE           # one extra token id
V = tasks.VOCAB_SIZE + 1
P_LEN, A_LEN = 18, 6              # prompt slots (left-padded), answer slots (padded with newlines)


def prompt_of(a, b):
    return f"Q: {a}+{b}=?\nA:"


def answer_of(c):
    s = f" {c}"
    return s + "\n" * (A_LEN - len(s))


def encode_batch(prompts, answers=None):
    B = len(prompts)
    ids = torch.full((B, P_LEN + A_LEN), tasks.PAD_ID)
    for i, p in enumerate(prompts):
        e = tasks.encode(p)[-P_LEN:]
        ids[i, P_LEN - len(e):P_LEN] = torch.tensor(e)
    if answers is not None:
        for i, a in enumerate(answers):
            ids[i, P_LEN:] = torch.tensor(tasks.encode(a))
    else:
        ids[:, P_LEN:] = MASK
    return ids


def parse(answer_text):
    s = answer_text.split("\n")[0].strip()
    return int(s) if s.isdigit() else None


class MaskedDiffusionLM(nn.Module):
    def __init__(self, d=128, depth=4, heads=4):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.zeros(1, P_LEN + A_LEN, d)); nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)

    def forward(self, ids):
        pad = ids == tasks.PAD_ID
        h = self.enc(self.tok(ids) + self.pos, src_key_padding_mask=pad)
        return self.head(self.ln(h))[:, P_LEN:]            # logits for the answer slots only


def training_data(n2=None, n3=2000, seed=0):
    """All 2-digit training problems plus a limited number of 3-digit ones (so 3-digit skill is imperfect)."""
    rng = np.random.default_rng(seed)
    pairs = list(tasks.TRAIN_2D) + tasks.sample_pairs(n3, 3, rng)
    return pairs


def train(model, pairs, steps=4000, batch=256, lr=1e-3, seed=0, log_every=1000, noise=0.0):
    """noise > 0: each training answer is replaced by a plausible human mistake with that probability
    (the same error model as notebook 02's sloppy annotators)."""
    set_seed(seed)
    err_rng = np.random.default_rng(seed + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    losses, t0 = [], time.time()
    model.train()
    for step in range(steps):
        idx = np.random.randint(0, len(pairs), batch)
        P = [prompt_of(*pairs[i]) for i in idx]
        A = [answer_of(tasks._mistake(*pairs[i], err_rng) if noise and err_rng.random() < noise else sum(pairs[i])) for i in idx]
        ids = encode_batch(P, A)
        target = ids[:, P_LEN:].clone()
        t = torch.rand(batch, 1).clamp(min=1e-3)
        masked = torch.rand(batch, A_LEN) < t
        ids[:, P_LEN:] = torch.where(masked, torch.full_like(target, MASK), target)
        ce = F.cross_entropy(model(ids).reshape(-1, V), target.reshape(-1), reduction="none").view(batch, A_LEN)
        loss = ((ce * masked) / t).sum(1).mean() / A_LEN
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        losses.append(loss.item())
        if log_every and (step + 1) % log_every == 0:
            print(f"step {step + 1:5d}  loss {np.mean(losses[-log_every:]):.4f}  ({time.time() - t0:.0f}s)")
    return losses


@torch.no_grad()
def generate(model, prompts, temperature=0.0, return_order=False, steps=A_LEN):
    """Iterative unmasking: each step fills the most confident masked slots, A_LEN / steps of them per step.

    steps = A_LEN: one token per step (most accurate); steps = 1: every slot decoded in parallel in one pass."""
    model.eval()
    ids = encode_batch(prompts)
    B = len(prompts)
    order = torch.zeros(B, A_LEN, dtype=torch.long)
    per_step = [len(c) for c in np.array_split(np.arange(A_LEN), steps)]
    for step, k in enumerate(per_step):
        logits = model(ids).float()
        if temperature > 0:
            probs = F.softmax(logits / temperature, -1)
            cand = torch.multinomial(probs.view(-1, V), 1).view(B, A_LEN)
        else:
            probs = F.softmax(logits, -1); cand = logits.argmax(-1)
        conf = torch.gather(probs, 2, cand[..., None])[..., 0]
        conf[ids[:, P_LEN:] != MASK] = -1.0
        pos = conf.topk(k, dim=1).indices                       # (B, k) slots to fill this step
        rows = torch.arange(B)[:, None].expand_as(pos)
        ids[rows, P_LEN + pos] = cand[rows, pos]
        order[rows, pos] = step
    texts = [tasks.decode(r.tolist()) for r in ids[:, P_LEN:]]
    return (texts, order) if return_order else texts


def accuracy(model, pairs, temperature=0.0, seed=0, steps=A_LEN):
    set_seed(seed)
    out = generate(model, [prompt_of(a, b) for a, b in pairs], temperature, steps=steps)
    return float(np.mean([parse(o) == a + b for o, (a, b) in zip(out, pairs)]))


def answer_logprob(model, prompts, answers, p_mask_prompt=0.0):
    """d1's one-step estimate of log pi(answer | prompt): mask the whole answer, sum per-slot log-probs
    of the sampled tokens from a single forward pass (optionally also masking random prompt tokens)."""
    ids = encode_batch(prompts)
    if p_mask_prompt > 0:
        pm = (torch.rand(len(prompts), P_LEN) < p_mask_prompt) & (ids[:, :P_LEN] != tasks.PAD_ID)
        ids[:, :P_LEN][pm] = MASK
    target = torch.tensor([tasks.encode(a) for a in answers])
    lp = F.log_softmax(model(ids).float(), -1)
    return torch.gather(lp, 2, target[..., None])[..., 0].sum(1)


def diffu_grpo(model, ref, steps=100, n_prompts=16, G=8, lr=1e-5, beta=0.04, clip=0.2, inner=2, temperature=1.0, seed=0, log_every=25,
               decode_steps=A_LEN, digits=3, p_mask_prompt=0.0):
    """GRPO for a masked diffusion LM, following d1 (Zhao et al., 2025): verifier reward, group-normalised advantages,
    one-step mean-field log-probabilities. d1 also masks a random 15% of prompt tokens when computing them (a regulariser
    for long natural-language prompts); here every prompt token is an operand digit, so the default is 0. With
    decode_steps=1 the one-step log-probability is exactly the probability of the sampled answer, so the gradient is exact."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist, t0 = [], time.time()
    for it in range(steps):
        pairs = tasks.sample_pairs(n_prompts, digits, rng)
        P = [prompt_of(a, b) for a, b in pairs for _ in range(G)]
        R = generate(model, P, temperature, steps=decode_steps)
        r = torch.tensor([float(parse(o) == a + b) for o, (a, b) in zip(R, [p for p in pairs for _ in range(G)])])
        g = r.view(-1, G)
        A = ((g - g.mean(1, keepdim=True)) / (g.std(1, keepdim=True) + 1e-4)).view(-1)
        R = [o if len(o) == A_LEN else (o + "\n" * A_LEN)[:A_LEN] for o in R]
        seed_mask = int(rng.integers(1 << 30))
        torch.manual_seed(seed_mask)
        with torch.no_grad():
            old = answer_logprob(model, P, R, p_mask_prompt=p_mask_prompt)
            torch.manual_seed(seed_mask); rf = answer_logprob(ref, P, R, p_mask_prompt=p_mask_prompt)
        model.train()
        for _ in range(inner):
            torch.manual_seed(seed_mask)
            lp = answer_logprob(model, P, R, p_mask_prompt=p_mask_prompt)
            ratio = torch.exp(lp - old)
            pg = -torch.min(ratio * A, ratio.clamp(1 - clip, 1 + clip) * A)
            kl = torch.exp(rf - lp) - (rf - lp) - 1
            loss = (pg + beta * kl).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        hist.append(dict(step=it + 1, reward=r.mean().item(), no_signal=(g.std(1) == 0).float().mean().item()))
        if log_every and (it + 1) % log_every == 0:
            print(f"iter {it + 1:4d}  train reward {np.mean([h['reward'] for h in hist[-log_every:]]):.3f}  ({time.time() - t0:.0f}s)")
    import pandas as pd
    return pd.DataFrame(hist)
