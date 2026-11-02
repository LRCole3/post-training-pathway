"""Diffusion and flow-matching utilities for the image notebooks (06-10), on 8x8 digits (64-d vectors).

Conventions: data x0 in [-1, 1]^64; discrete time t in {0..T-1} with a cosine schedule
(Nichol & Dhariwal, 2021); class labels 0-9 and NULL = 10 for the unconditional branch used by
classifier-free guidance (Ho & Salimans, 2022).
"""
import copy, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import set_seed

T = 1000
NULL = 10


def cosine_alpha_bar(T=T, s=0.008):
    t = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = (f / f[0])[1:]
    return ab.clamp(1e-5, 0.9999).float()


ALPHA_BAR = cosine_alpha_bar()


def timestep_embedding(t, dim=128):
    """Sinusoidal embedding of (possibly fractional) timesteps t in [0, T)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], -1)


class MLPDenoiser(nn.Module):
    """Residual MLP denoiser: (x_t, t, class) -> prediction (64-d). About 1.4M parameters."""

    def __init__(self, d_in=64, width=256, depth=4, n_classes=11):
        super().__init__()
        self.inp = nn.Linear(d_in, width)
        self.temb = nn.Sequential(nn.Linear(128, width), nn.SiLU(), nn.Linear(width, width))
        self.cemb = nn.Embedding(n_classes, width)
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(),
                                                  nn.Linear(width * 2, width)) for _ in range(depth))
        self.mods = nn.ModuleList(nn.Linear(width, width) for _ in range(depth))
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, d_in))
        nn.init.zeros_(self.out[1].weight); nn.init.zeros_(self.out[1].bias)

    def forward(self, x, t, c):
        e = self.temb(timestep_embedding(t)) + self.cemb(c)
        h = self.inp(x)
        for blk, mod in zip(self.blocks, self.mods):
            h = h + blk(h + mod(F.silu(e)))
        return self.out(h)


# ---------------------------------------------------------------- parameterisations
def q_sample(x0, t, eps):
    ab = ALPHA_BAR[t][:, None]
    return ab.sqrt() * x0 + (1 - ab).sqrt() * eps


def v_target(x0, t, eps):
    ab = ALPHA_BAR[t][:, None]
    return ab.sqrt() * eps - (1 - ab).sqrt() * x0


def to_x0_eps(out, x_t, t, pred="v"):
    """Convert a network output (eps or v prediction) into (x0_hat, eps_hat)."""
    ab = ALPHA_BAR[t][:, None]
    if pred == "eps":
        eps = out
        x0 = (x_t - (1 - ab).sqrt() * eps) / ab.sqrt()
    else:
        x0 = ab.sqrt() * x_t - (1 - ab).sqrt() * out
        eps = (1 - ab).sqrt() * x_t + ab.sqrt() * out
    return x0, eps


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for pe, p in zip(self.model.parameters(), model.parameters()):
            pe.lerp_(p, 1 - self.decay)


def train_diffusion(model, X, y, steps=6000, batch=256, lr=1e-3, pred="v", p_uncond=0.1, ema_decay=0.999,
                    seed=0, log_every=1000, weight_decay=0.0):
    """Denoising loss ||net(x_t, t, c) - target||^2 with target = eps or v; labels dropped with prob p_uncond."""
    set_seed(seed)
    model.requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    ema = EMA(model, ema_decay)
    losses, t0 = [], time.time()
    model.train()
    for step in range(steps):
        idx = torch.randint(0, len(X), (batch,))
        x0, c = X[idx], y[idx].clone()
        c[torch.rand(batch) < p_uncond] = NULL
        t = torch.randint(0, T, (batch,))
        eps = torch.randn_like(x0)
        xt = q_sample(x0, t, eps)
        target = eps if pred == "eps" else v_target(x0, t, eps)
        loss = F.mse_loss(model(xt, t, c), target)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); ema.update(model)
        losses.append(loss.item())
        if log_every and (step + 1) % log_every == 0:
            print(f"step {step + 1:6d}  loss {np.mean(losses[-log_every:]):.4f}  ({time.time() - t0:.0f}s)")
    return ema.model, losses


def null_like(c):
    """The 'no condition' input: label NULL for class labels, the empty (all-padding) caption for text."""
    return torch.zeros_like(c) if c.dim() == 2 else torch.full_like(c, NULL)


def guided(model, x, t, c, w, pred="v"):
    """Classifier-free guidance: out = (1 + w) * cond - w * uncond (w = 0: plain conditional)."""
    if w == 0:
        return model(x, t, c)
    out = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([c, null_like(c)]))
    oc, ou = out.chunk(2)
    return (1 + w) * oc - w * ou


def ddim_timesteps(steps):
    return torch.linspace(T - 1, 0, steps).round().long()


def sample_ddim(model, labels, steps=50, w=0.0, pred="v", eta=0.0, x_T=None, clip=True, return_traj=False, grad=False):
    """DDIM (Song et al., 2021). eta=0: deterministic; eta=1: DDPM-like stochastic sampling."""
    n = len(labels)
    x = torch.randn(n, model.inp.in_features) if x_T is None else x_T.clone()
    ts = ddim_timesteps(steps)
    traj = [x]
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for i, t in enumerate(ts):
            tt = torch.full((n,), int(t), dtype=torch.long)
            x0, eps = to_x0_eps(guided(model, x, tt, labels, w, pred), x, tt, pred)
            if clip:
                x0 = x0.clamp(-1, 1)
                ab_t = ALPHA_BAR[tt][:, None]
                eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()
            if i == len(ts) - 1:
                x = x0
            else:
                ab_t, ab_s = ALPHA_BAR[int(t)], ALPHA_BAR[int(ts[i + 1])]
                sigma = eta * math.sqrt((1 - ab_s) / (1 - ab_t) * (1 - ab_t / ab_s))
                x = ab_s ** 0.5 * x0 + math.sqrt(max(1 - ab_s - sigma ** 2, 0)) * eps
                if eta > 0:
                    x = x + sigma * torch.randn_like(x)
            if return_traj:
                traj.append(x)
    return (x, traj) if return_traj else x


def sample_with_logprob(model, labels, steps=20, eta=1.0, w=0.0, pred="v"):
    """Stochastic DDIM sampling that records each transition, for policy-gradient fine-tuning (DDPO).

    Returns final samples and a list of (x_t, t, x_next, mean, sigma) per step; log p of each transition is
    Gaussian, N(x_next; mean, sigma^2), so it can be re-evaluated under updated parameters.
    """
    n = len(labels)
    x = torch.randn(n, model.inp.in_features)
    ts = ddim_timesteps(steps)
    steps_log = []
    with torch.no_grad():
        for i, t in enumerate(ts[:-1]):
            tt = torch.full((n,), int(t), dtype=torch.long)
            mean, sigma = ddim_step_mean(model, x, tt, int(ts[i + 1]), labels, eta, w, pred)
            x_next = mean + sigma * torch.randn_like(x)
            steps_log.append((x, tt, int(ts[i + 1]), x_next))
            x = x_next
        tt = torch.full((n,), int(ts[-1]), dtype=torch.long)
        x0, _ = to_x0_eps(guided(model, x, tt, labels, w, pred), x, tt, pred)
    return x0.clamp(-1, 1), steps_log


def ddim_step_mean(model, x, tt, s, labels, eta=1.0, w=0.0, pred="v"):
    """Mean and std of the stochastic DDIM transition x_t -> x_s (differentiable in the model)."""
    x0, eps = to_x0_eps(guided(model, x, tt, labels, w, pred), x, tt, pred)
    x0 = x0.clamp(-1, 1)
    ab_t = ALPHA_BAR[tt][:, None]
    eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()
    ab_s = ALPHA_BAR[s]
    sigma = eta * torch.sqrt((1 - ab_s) / (1 - ab_t) * (1 - ab_t / ab_s))
    mean = ab_s ** 0.5 * x0 + torch.sqrt((1 - ab_s - sigma ** 2).clamp(min=0)) * eps
    return mean, sigma


def gaussian_logprob(x, mean, sigma):
    return (-((x - mean) ** 2) / (2 * sigma ** 2) - torch.log(sigma) - 0.5 * math.log(2 * math.pi)).sum(-1)


# ---------------------------------------------------------------- checkpoints shared by notebooks 07-08
DIFF_KW = dict(steps=6000, batch=256, lr=1e-3, pred="v")


def get_base_diffusion(verbose=True):
    """Notebook 06's class-conditional v-prediction model (EMA weights), trained once and cached."""
    import os
    from . import CKPT_DIR, images
    path = os.path.join(CKPT_DIR, "diffusion_base.pt")
    m = MLPDenoiser()
    if os.path.exists(path):
        m.load_state_dict(torch.load(path, weights_only=True)); return m.eval()
    if verbose:
        print("checkpoints/diffusion_base.pt not found: training notebook 06's model now ...")
    Xtr, ytr, _, _ = images.load_digits_tensor()
    ema, _ = train_diffusion(MLPDenoiser(), Xtr, ytr, **DIFF_KW, log_every=2000 if verbose else 0)
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save(ema.state_dict(), path)
    return ema.eval()


# ---------------------------------------------------------------- reward alignment (notebook 07)
def ink(x):
    """'Boldness' reward: fraction of the image covered by ink, in [0, 1]."""
    return ((x.clamp(-1, 1) + 1) / 2).mean(-1)


def random_labels(n, rng=None):
    return torch.randint(0, 10, (n,), generator=rng)


def ddpo(model, reward_fn, iters=60, batch=64, steps=20, lr=3e-5, clip=1e-4, inner=1, seed=0, log_every=20, eval_fn=None,
         kl_coef=0.0, ref=None):
    """DDPO-clip (Black et al., 2023): PPO over the denoising MDP. Each sampling step is an action whose
    log-probability is a Gaussian; advantages are batch-normalised final rewards.

    kl_coef > 0 adds DPOK's regulariser (Fan et al., 2023): the KL between the policy's and the reference
    model's Gaussian transitions, ||mu_theta - mu_ref||^2 / (2 sigma^2), summed over steps."""
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for it in range(iters):
        model.eval()
        labels = random_labels(batch)
        x0, traj = sample_with_logprob(model, labels, steps=steps)
        r = reward_fn(x0, labels)
        adv = ((r - r.mean()) / (r.std() + 1e-6)).detach()
        with torch.no_grad():
            old = [gaussian_logprob(xn, *ddim_step_mean(model, x, tt, s, labels)) for x, tt, s, xn in traj]
        model.train()
        for _ in range(inner):
            for k, (x, tt, s, xn) in enumerate(traj):
                mean, sigma = ddim_step_mean(model, x, tt, s, labels)
                lp = gaussian_logprob(xn, mean, sigma)
                ratio = torch.exp(lp - old[k])
                loss = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
                if kl_coef > 0:
                    with torch.no_grad():
                        mean_ref, _ = ddim_step_mean(ref, x, tt, s, labels)
                    loss = loss + kl_coef * (((mean - mean_ref) ** 2).sum(-1) / (2 * sigma[:, 0] ** 2)).mean()
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        row = dict(iter=it + 1, reward=r.mean().item())
        if eval_fn is not None and ((it + 1) % log_every == 0 or it == 0):
            row.update(eval_fn(model))
        hist.append(row)
        if log_every and (it + 1) % log_every == 0:
            print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
    return model.eval(), hist


def draft(model, reward_fn, iters=200, batch=64, steps=20, K=1, lr=1e-5, seed=0, log_every=50, eval_fn=None):
    """DRaFT-K (Clark et al., 2023): back-propagate the reward through the last K steps of a DDIM sampler."""
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ts = ddim_timesteps(steps)
    hist = []
    for it in range(iters):
        labels = random_labels(batch)
        x = torch.randn(batch, model.inp.in_features)
        model.train()
        for i, t in enumerate(ts):
            tt = torch.full((batch,), int(t), dtype=torch.long)
            grad_on = i >= len(ts) - K
            with torch.set_grad_enabled(grad_on):
                mean, _ = ddim_step_mean(model, x, tt, int(ts[i + 1]) if i < len(ts) - 1 else 0, labels, eta=0.0)
                if i == len(ts) - 1:
                    x = to_x0_eps(model(x, tt, labels), x, tt)[0].clamp(-1, 1)
                else:
                    x = mean if grad_on else mean.detach()
        r = reward_fn(x, labels)
        loss = -r.mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        row = dict(iter=it + 1, reward=r.mean().item())
        if eval_fn is not None and ((it + 1) % log_every == 0 or it == 0):
            row.update(eval_fn(model))
        hist.append(row)
        if log_every and (it + 1) % log_every == 0:
            print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
    return model.eval(), hist


def diffusion_dpo(model, ref, x_w, x_l, labels, iters=600, batch=64, beta=500.0, lr=1e-5, seed=0, log_every=200, eval_fn=None):
    """Diffusion-DPO (Wallace et al., 2023): DPO where log-likelihoods are replaced by denoising errors.

    loss = -log sigmoid(-beta * [(err_w - err_w_ref) - (err_l - err_l_ref)]), errors at a shared (t, noise).
    """
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for it in range(iters):
        idx = torch.randint(0, len(x_w), (batch,))
        xw, xl, c = x_w[idx], x_l[idx], labels[idx]
        t = torch.randint(0, T, (batch,)); eps = torch.randn_like(xw)
        def err(m, x0):
            return ((m(q_sample(x0, t, eps), t, c) - v_target(x0, t, eps)) ** 2).mean(1)
        with torch.no_grad():
            rw, rl = err(ref, xw), err(ref, xl)
        model.train()
        h = (err(model, xw) - rw) - (err(model, xl) - rl)
        loss = -F.logsigmoid(-beta * h).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        row = dict(iter=it + 1, loss=loss.item())
        if eval_fn is not None and ((it + 1) % log_every == 0 or it == 0):
            row.update(eval_fn(model))
        hist.append(row)
        if log_every and (it + 1) % log_every == 0:
            print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
    return model.eval(), hist


def add_class_token(model, init_from=3):
    """DreamBooth-style rare identifier: append a new label embedding, initialised from an existing class."""
    m = copy.deepcopy(model)
    old = m.cemb
    new = nn.Embedding(old.num_embeddings + 1, old.embedding_dim)
    with torch.no_grad():
        new.weight[:-1] = old.weight; new.weight[-1] = old.weight[init_from]
    m.cemb = new
    return m


def add_lora_denoiser(model, r=4, alpha=8):
    """Freeze an MLPDenoiser and add LoRA adapters to the linear layers of every residual block."""
    from .lm import LoRALinear
    for p in model.parameters():
        p.requires_grad_(False)
    for blk in model.blocks:
        blk[1] = LoRALinear(blk[1], r, alpha); blk[3] = LoRALinear(blk[3], r, alpha)
    for i in range(len(model.mods)):
        model.mods[i] = LoRALinear(model.mods[i], r, alpha)
    return model


def finetune(model, X, y, steps=400, lr=2e-4, prior=None, prior_weight=1.0, seed=0):
    """Fine-tune a v-prediction denoiser on a few examples; `prior=(X_p, y_p)` adds DreamBooth's
    prior-preservation loss on samples generated by the original model."""
    set_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    model.train()
    for _ in range(steps):
        t = torch.randint(0, T, (len(X),)); e = torch.randn_like(X)
        loss = F.mse_loss(model(q_sample(X, t, e), t, y), v_target(X, t, e))
        if prior is not None:
            Xp, yp = prior
            idx = torch.randint(0, len(Xp), (len(X),)); t = torch.randint(0, T, (len(X),)); e = torch.randn(len(X), Xp.shape[1])
            loss = loss + prior_weight * F.mse_loss(model(q_sample(Xp[idx], t, e), t, yp[idx]), v_target(Xp[idx], t, e))
        opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


# ---------------------------------------------------------------- few-step samplers and distillation (notebook 08)
def grid(N):
    """Timesteps of an N-step sampler: tau_N = T-1 > ... > tau_0 = 0. The 2N grid contains the N grid."""
    return [int(round((T - 1) * k / N)) for k in range(N, -1, -1)]


def _ab(t):
    return ALPHA_BAR[t] if isinstance(t, int) else ALPHA_BAR[t][:, None]


def ddim_jump(model, x, t, s, labels, pred="v"):
    """One deterministic DDIM step from integer time t to s (s = 0 returns the clean estimate)."""
    tt = torch.full((len(x),), t, dtype=torch.long)
    x0, eps = to_x0_eps(model(x, tt, labels), x, tt, pred)
    x0 = x0.clamp(-1, 1)
    if s == 0:
        return x0
    ab_t, ab_s = ALPHA_BAR[t], ALPHA_BAR[s]
    eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()
    return ab_s.sqrt() * x0 + (1 - ab_s).sqrt() * eps


@torch.no_grad()
def sample_grid(model, labels, N, x_T=None, pred="v"):
    x = torch.randn(len(labels), 64) if x_T is None else x_T.clone()
    g = grid(N)
    for t, s in zip(g[:-1], g[1:]):
        x = ddim_jump(model, x, t, s, labels, pred)
    return x


def progressive_distillation(teacher, X, y, start=64, end=1, iters_per_round=600, batch=256, lr=2e-4, seed=0, log=True):
    """Salimans & Ho (2022): repeatedly train a student to match two teacher DDIM steps with one step.

    Returns {n_steps: student model} for every round (64 -> 32 -> ... -> end)."""
    set_seed(seed)
    students, N, t0 = {start: teacher}, start // 2, time.time()
    while N >= end:
        student = copy.deepcopy(students[2 * N]).train().requires_grad_(True)
        teach = students[2 * N].eval()
        g_s, g_t = grid(N), grid(2 * N)
        opt = torch.optim.Adam(student.parameters(), lr=lr)
        for _ in range(iters_per_round):
            idx = torch.randint(0, len(X), (batch,))
            x0, c = X[idx], y[idx]
            k = torch.randint(0, N, (batch,))                        # student step from g_s[k] to g_s[k+1]
            t = torch.tensor([g_s[i] for i in k]); tm = torch.tensor([g_t[2 * i + 1] for i in k]); s = torch.tensor([g_s[i + 1] for i in k])
            xt = q_sample(x0, t, torch.randn_like(x0))
            with torch.no_grad():                                    # two teacher steps (per-example times)
                x_mid = _jump_vec(teach, xt, t, tm, c)
                x_s = _jump_vec(teach, x_mid, tm, s, c)
                a_t, s_t = _ab(t).sqrt(), (1 - _ab(t)).sqrt()
                a_s, s_s = _ab(s).sqrt(), (1 - _ab(s)).sqrt()
                ratio = torch.where(s[:, None] == 0, torch.zeros_like(s_s), s_s / s_t)
                x0_tgt = ((x_s - ratio * xt) / (a_s - ratio * a_t)).clamp(-1, 1)
                eps_tgt = (xt - a_t * x0_tgt) / s_t
                v_tgt = a_t * eps_tgt - s_t * x0_tgt
            loss = F.mse_loss(student(xt, t, c), v_tgt)
            opt.zero_grad(); loss.backward(); opt.step()
        students[N] = student.eval()
        if log:
            print(f"distilled {2 * N} -> {N} steps  (loss {loss.item():.4f}, {time.time() - t0:.0f}s)")
        N //= 2
    return students


def _jump_vec(model, x, t, s, labels):
    """DDIM step with per-example integer times t -> s (tensors)."""
    x0, eps = to_x0_eps(model(x, t, labels), x, t, "v")
    x0 = x0.clamp(-1, 1)
    ab_t, ab_s = _ab(t), _ab(s)
    eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()
    out = ab_s.sqrt() * x0 + (1 - ab_s).sqrt() * eps
    return torch.where(s[:, None] == 0, x0, out)


def consistency_distillation(teacher, X, y, N=32, iters=3000, batch=256, lr=1e-4, ema=0.95, seed=0, log_every=1000):
    """Song et al. (2023): train f(x_t, t) -> x0 so that adjacent points on the teacher's ODE trajectory map to the
    same output; the target network is an EMA of the student."""
    set_seed(seed)
    student = copy.deepcopy(teacher).train().requires_grad_(True)
    target = copy.deepcopy(teacher).eval().requires_grad_(False)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    g = grid(N)
    for it in range(iters):
        idx = torch.randint(0, len(X), (batch,))
        x0, c = X[idx], y[idx]
        k = torch.randint(0, N, (batch,))
        t = torch.tensor([g[i] for i in k]); s = torch.tensor([g[i + 1] for i in k])
        xt = q_sample(x0, t, torch.randn_like(x0))
        with torch.no_grad():
            xs = _jump_vec(teacher, xt, t, s, c)                    # one teacher ODE step
            f_tgt = torch.where(s[:, None] == 0, xs, to_x0_eps(target(xs, s, c), xs, s)[0].clamp(-1, 1))
        f = to_x0_eps(student(xt, t, c), xt, t)[0]
        loss = F.mse_loss(f, f_tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for pt, ps in zip(target.parameters(), student.parameters()):
                pt.lerp_(ps, 1 - ema)
        if log_every and (it + 1) % log_every == 0:
            print(f"consistency step {it + 1}: loss {loss.item():.5f}")
    return student.eval()


@torch.no_grad()
def sample_consistency(model, labels, n_steps=1, x_T=None):
    """Multistep consistency sampling: map to x0, re-noise to an intermediate time, map again."""
    x = torch.randn(len(labels), 64) if x_T is None else x_T.clone()
    times = grid(n_steps)[:-1]
    for i, t in enumerate(times):
        tt = torch.full((len(x),), t, dtype=torch.long)
        x0 = to_x0_eps(model(x, tt, labels), x, tt)[0].clamp(-1, 1)
        if i + 1 < len(times):
            x = q_sample(x0, torch.full((len(x),), times[i + 1], dtype=torch.long), torch.randn_like(x0))
    return x0


# ---------------------------------------------------------------- flow matching / rectified flow (notebooks 08-10)
def train_flow(model, X1, y, X0=None, steps=6000, batch=256, lr=1e-3, p_uncond=0.1, t_sampling="uniform", seed=0, log_every=2000,
               ema_decay=0.999):
    """Rectified flow / flow matching (Liu et al., 2023; Lipman et al., 2023).

    x_s = (1 - s) * x_data + s * noise for s in [0, 1]; the network predicts the velocity u = noise - x_data
    (time input = s * 999). If X0 is given, (X1[i], X0[i]) are *coupled* data/noise pairs (reflow)."""
    set_seed(seed)
    model.requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s_: min(1, (s_ + 1) / 200) * 0.5 * (1 + math.cos(math.pi * s_ / steps)))
    ema = EMA(model, ema_decay)
    losses, t0 = [], time.time()
    model.train()
    for step in range(steps):
        idx = torch.randint(0, len(X1), (batch,))
        x1, c = X1[idx], y[idx].clone()
        drop = torch.rand(batch) < p_uncond
        c[drop] = null_like(c)[drop]
        z = torch.randn_like(x1) if X0 is None else X0[idx]
        if t_sampling == "logit_normal":                         # SD3 (Esser et al., 2024)
            s = torch.sigmoid(torch.randn(batch))
        else:
            s = torch.rand(batch)
        xs = (1 - s[:, None]) * x1 + s[:, None] * z
        loss = F.mse_loss(model(xs, s * (T - 1), c), z - x1)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); ema.update(model)
        losses.append(loss.item())
        if log_every and (step + 1) % log_every == 0:
            print(f"step {step + 1:6d}  loss {np.mean(losses[-log_every:]):.4f}  ({time.time() - t0:.0f}s)")
    return ema.model, losses


def sample_flow(model, labels, steps=20, w=0.0, x_T=None, grad=False, return_traj=False):
    """Euler integration of dx/ds = u(x, s) from s = 1 (noise) to s = 0 (data)."""
    x = torch.randn(len(labels), model_dim(model)) if x_T is None else x_T.clone()
    traj = [x]
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for i in range(steps):
            s = 1 - i / steps
            ss = torch.full((len(x),), s * (T - 1))
            x = x - (1 / steps) * guided(model, x, ss, labels, w)
            if return_traj:
                traj.append(x)
    return (x, traj) if return_traj else x


def model_dim(model):
    return model.inp.in_features if hasattr(model, "inp") else model.d_in
