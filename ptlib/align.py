"""Post-training algorithms for the language-model notebooks, collected for reuse.

Each function is derived and written out step by step in its own notebook
(SFT: 02, reward model and PPO: 03, DPO family: 04, GRPO: 05); later notebooks import them here.
"""
import copy, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import tasks, lm, set_seed


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, pairs=None, n_samples=1, temperature=1.0, digits=2, seed=0, template=None):
    """Greedy and sampled accuracy on held-out problems in the chat format, plus style mix."""
    set_seed(seed)
    pairs = pairs if pairs is not None else (tasks.TEST_2D if digits == 2 else tasks.TEST_3D)
    rng = np.random.default_rng(seed)
    prompts = [tasks.make_prompt(a, b, template=template, rng=rng) for a, b in pairs]
    g = lm.generate(model, prompts, greedy=True)
    s = lm.generate(model, prompts * n_samples, temperature=temperature)
    pr = pairs * n_samples
    parsed = [tasks.parse_response(o) for o in s]
    return {
        "greedy_acc": float(np.mean([tasks.gold_reward(a, b, o) for (a, b), o in zip(pairs, g)])),
        "sampled_acc": float(np.mean([tasks.gold_reward(a, b, o) for (a, b), o in zip(pr, s)])),
        "well_formed": float(np.mean([p[0] is not None for p in parsed])),
        "sentence_style": float(np.mean([p[1] == "sentence" for p in parsed])),
        "mean_len": float(np.mean([len(o) for o in s])),
    }


# ---------------------------------------------------------------- supervised fine-tuning (NB02)
def sft(model, data, epochs=3, batch=64, lr=5e-4, seed=0, log=True):
    """Next-token loss on response tokens only (prompt tokens are masked out)."""
    set_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    n_steps = epochs * math.ceil(len(data) / batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1 - s / n_steps)
    losses = []
    model.train()
    for ep in range(epochs):
        perm = np.random.permutation(len(data))
        for i in range(0, len(data), batch):
            b = [data[j] for j in perm[i:i + batch]]
            lp, m = lm.sequence_logprobs(model, [d["prompt"] for d in b], [d["response"] for d in b], per_token=True)
            loss = -lp.sum() / m.sum()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
            losses.append(loss.item())
        if log:
            print(f"epoch {ep + 1}: loss {np.mean(losses[-math.ceil(len(data) / batch):]):.3f}")
    return losses


SFT_KW = dict(n_demos=3000, epochs=3, lr=5e-4)


def get_sft(verbose=True):
    """The SFT model from notebook 02 (full fine-tuning on the noisy demonstrations)."""
    if lm.exists("sft_lm"):
        return lm.load("sft_lm")
    if verbose:
        print("checkpoints/sft_lm.pt not found: running notebook 02's SFT now ...")
    m = lm.get_pretrained(verbose)
    sft(m, tasks.demonstrations(SFT_KW["n_demos"], seed=0), epochs=SFT_KW["epochs"], lr=SFT_KW["lr"], log=verbose)
    lm.save(m, "sft_lm")
    return m


# ---------------------------------------------------------------- preferences + reward model (NB03)
def train_prompts(n, digits=2, seed=0):
    rng = np.random.default_rng(seed)
    pairs = tasks.sample_pairs(n, digits, rng)
    return [tasks.make_prompt(a, b, rng=rng) for a, b in pairs]


@torch.no_grad()
def make_preference_pairs(policy, n_prompts=6000, flip=0.1, seed=0, temperature=1.0):
    """Sample two responses per prompt from `policy`; a simulated labeller prefers the correct one.

    Pairs where both or neither are correct are dropped (the labeller has no preference), and a
    fraction `flip` of labels is flipped to mimic labeller disagreement (~70-80% in practice).
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)
    prompts = train_prompts(n_prompts, seed=seed + 1)
    r1 = lm.generate(policy, prompts, temperature=temperature)
    r2 = lm.generate(policy, prompts, temperature=temperature)
    out = []
    for p, x, y in zip(prompts, r1, r2):
        a, b = tasks.a_b_from_prompt(p)
        gx, gy = tasks.gold_reward(a, b, x), tasks.gold_reward(a, b, y)
        if gx == gy:
            continue
        chosen, rejected = (x, y) if gx > gy else (y, x)
        if rng.random() < flip:
            chosen, rejected = rejected, chosen
        out.append(dict(prompt=p, chosen=chosen, rejected=rejected))
    return out


class RewardModel(nn.Module):
    """A transformer trunk with a scalar head read at the final response token (Ouyang et al., 2022)."""

    def __init__(self, gpt):
        super().__init__()
        self.gpt = gpt
        self.head = nn.Linear(gpt.cfg["d"], 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, prompts, responses):
        x, y, m = lm.pack(prompts, responses, self.gpt.cfg["block"])
        _, h = self.gpt(x, return_hidden=True)
        last = (m.cumsum(1) * m).argmax(1)            # index of the last response token
        return self.head(h[torch.arange(len(x)), last])[:, 0]


def train_reward_model(rm, pairs, epochs=2, batch=64, lr=1e-4, seed=0, log=True):
    """Bradley-Terry loss: -log sigmoid(r(chosen) - r(rejected))."""
    set_seed(seed)
    opt = torch.optim.AdamW(rm.parameters(), lr=lr)
    losses, accs = [], []
    rm.train()
    for ep in range(epochs):
        perm = np.random.permutation(len(pairs))
        for i in range(0, len(pairs), batch):
            b = [pairs[j] for j in perm[i:i + batch]]
            p = [d["prompt"] for d in b]
            rc, rr = rm(p, [d["chosen"] for d in b]), rm(p, [d["rejected"] for d in b])
            loss = -F.logsigmoid(rc - rr).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item()); accs.append((rc > rr).float().mean().item())
        if log:
            print(f"epoch {ep + 1}: loss {np.mean(losses[-20:]):.3f}  train pair acc {np.mean(accs[-20:]):.3f}")
    return losses


@torch.no_grad()
def rm_pair_accuracy(rm, pairs, batch=256):
    rm.eval()
    acc = []
    for i in range(0, len(pairs), batch):
        b = pairs[i:i + batch]; p = [d["prompt"] for d in b]
        acc.append((rm(p, [d["chosen"] for d in b]) > rm(p, [d["rejected"] for d in b])).float())
    return torch.cat(acc).mean().item()


# ---------------------------------------------------------------- PPO with a KL penalty (NB03)
class ValueModel(RewardModel):
    """Same trunk + scalar head, but read at every position: V(prefix)."""

    def forward_all(self, x):
        _, h = self.gpt(x, return_hidden=True)
        return self.head(h)[..., 0]


def ppo(policy, ref, reward_fn, value, steps=150, batch=64, beta=0.05, lr=1e-5, vf_lr=1e-4,
        ppo_epochs=4, clip=0.2, gamma=1.0, lam=0.95, seed=0, log_every=25, prompt_digits=2):
    """RLHF with PPO (Schulman et al., 2017) as in Ziegler et al. (2019) and Ouyang et al. (2022).

    reward_fn(prompts, responses) -> tensor of sequence-level rewards (a reward model or a verifier).
    Per-token reward = -beta * (log pi - log pi_ref), plus the sequence reward on the last token.
    """
    set_seed(seed)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    vopt = torch.optim.Adam(value.parameters(), lr=vf_lr)
    rng = np.random.default_rng(seed)
    hist = []
    t0 = time.time()
    for it in range(steps):
        pairs = tasks.sample_pairs(batch, prompt_digits, rng)
        prompts = [tasks.make_prompt(a, b, rng=rng) for a, b in pairs]
        resp = lm.generate(policy, prompts)
        with torch.no_grad():
            x, y, m = lm.pack(prompts, resp, policy.cfg["block"])
            old_lp = torch.gather(F.log_softmax(policy(x).float(), -1), 2, y[..., None])[..., 0]
            ref_lp = torch.gather(F.log_softmax(ref(x).float(), -1), 2, y[..., None])[..., 0]
            score = reward_fn(prompts, resp)
            v_old = value.forward_all(x) * m
            kl = (old_lp - ref_lp) * m
            rew = -beta * kl
            last = (m.cumsum(1) * m).argmax(1)
            rew[torch.arange(batch), last] += score
            # generalised advantage estimation over response tokens
            adv = torch.zeros_like(rew); nxt_adv = torch.zeros(batch); nxt_v = torch.zeros(batch)
            for t in reversed(range(rew.shape[1])):
                delta = rew[:, t] + gamma * nxt_v - v_old[:, t]
                nxt_adv = (delta + gamma * lam * nxt_adv) * m[:, t] + nxt_adv * (1 - m[:, t])
                adv[:, t] = nxt_adv * m[:, t]
                nxt_v = v_old[:, t] * m[:, t] + nxt_v * (1 - m[:, t])
            ret = adv + v_old
            adv_n = (adv - adv[m.bool()].mean()) / (adv[m.bool()].std() + 1e-8) * m
        policy.train(); value.train()
        for _ in range(ppo_epochs):
            lp = torch.gather(F.log_softmax(policy(x).float(), -1), 2, y[..., None])[..., 0]
            ratio = torch.exp((lp - old_lp) * m)
            pg = -torch.min(ratio * adv_n, ratio.clamp(1 - clip, 1 + clip) * adv_n)
            loss = (pg * m).sum() / m.sum()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
            v = value.forward_all(x)
            vloss = (((v - ret) ** 2) * m).sum() / m.sum()
            vopt.zero_grad(); vloss.backward(); vopt.step()
        gold = np.mean([tasks.gold_reward(a, b, r) for (a, b), r in zip(pairs, resp)])
        sty = np.mean([tasks.parse_response(r)[1] == "sentence" for r in resp])
        hist.append(dict(step=it + 1, proxy_reward=score.mean().item(), gold_reward=gold,
                         kl=kl.sum(1).mean().item(), sentence_style=sty, mean_len=np.mean([len(r) for r in resp])))
        if log_every and (it + 1) % log_every == 0:
            h = hist[-1]
            print(f"iter {it + 1:4d}  proxy {h['proxy_reward']:+.2f}  gold {h['gold_reward']:.2f}  "
                  f"KL {h['kl']:.2f}  sentence {h['sentence_style']:.2f}  ({time.time() - t0:.0f}s)")
    import pandas as pd
    return pd.DataFrame(hist)


def get_preferences(n_prompts=24000, seed=0, verbose=True):
    """Notebook 03's preference dataset: pairs of SFT samples labelled by a noisy simulated labeller."""
    import json, os
    from . import CKPT_DIR
    path = os.path.join(CKPT_DIR, f"prefs_{n_prompts}_{seed}.json")
    if os.path.exists(path):
        return json.load(open(path))
    if verbose:
        print(f"sampling preference pairs from the SFT model ({n_prompts} prompts) ...")
    prefs = make_preference_pairs(get_sft(verbose), n_prompts=n_prompts, seed=seed)
    os.makedirs(CKPT_DIR, exist_ok=True)
    json.dump(prefs, open(path, "w"))
    return prefs


# ---------------------------------------------------------------- direct preference optimisation (NB04)
def preference_loss(kind, pc, pr, rc, rr, beta=0.1, len_c=None, len_r=None, gamma=0.5):
    """pc/pr: policy log-probs of chosen/rejected; rc/rr: reference log-probs (summed over tokens).

    dpo   : -log sigmoid(beta * [(pc - rc) - (pr - rr)])                      Rafailov et al., 2023
    ipo   : ([(pc - rc) - (pr - rr)] - 1 / (2 beta))^2                         Azar et al., 2023
    simpo : -log sigmoid(beta * [pc/|c| - pr/|r|] - gamma), no reference       Meng et al., 2024
    """
    h = (pc - rc) - (pr - rr)
    if kind == "dpo":
        return -F.logsigmoid(beta * h).mean()
    if kind == "ipo":
        return ((h - 1 / (2 * beta)) ** 2).mean()
    if kind == "simpo":
        return -F.logsigmoid(beta * (pc / len_c - pr / len_r) - gamma).mean()
    raise ValueError(kind)


def kto_loss(p, r, desirable, beta=0.1, z_ref=0.0, lam_d=1.0, lam_u=1.0):
    """KTO (Ethayarajh et al., 2024): unpaired thumbs-up / thumbs-down examples.

    Reward r_theta = log pi/pi_ref; value = sigmoid(beta (r - z_ref)) for desirable,
    sigmoid(beta (z_ref - r)) for undesirable, where z_ref estimates KL(pi || pi_ref).
    """
    rw = p - r
    v = torch.where(desirable, lam_d * torch.sigmoid(beta * (rw - z_ref)), lam_u * torch.sigmoid(beta * (z_ref - rw)))
    lam = torch.where(desirable, torch.full_like(v, lam_d), torch.full_like(v, lam_u))
    return (lam - v).mean()


def offline_preference_training(policy, ref, pairs, kind="dpo", beta=0.1, epochs=2, batch=64, lr=5e-6,
                                seed=0, track=None, log=True, gamma=0.5):
    """Train `policy` on fixed preference pairs. `track`: optional held-out pairs to monitor log-probs."""
    set_seed(seed)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)
    with torch.no_grad():
        ref.eval()
        P = [d["prompt"] for d in pairs]
        RC = torch.cat([lm.sequence_logprobs(ref, P[i:i + 512], [d["chosen"] for d in pairs[i:i + 512]]) for i in range(0, len(pairs), 512)])
        RR = torch.cat([lm.sequence_logprobs(ref, P[i:i + 512], [d["rejected"] for d in pairs[i:i + 512]]) for i in range(0, len(pairs), 512)])
    hist = []
    step = 0
    for ep in range(epochs):
        perm = np.random.permutation(len(pairs))
        policy.train()
        for i in range(0, len(pairs), batch):
            idx = perm[i:i + batch]
            b = [pairs[j] for j in idx]; p = [d["prompt"] for d in b]
            ch, rj = [d["chosen"] for d in b], [d["rejected"] for d in b]
            if kind == "kto":
                # unpaired view of the same data: chosen = desirable, rejected = undesirable
                lp = lm.sequence_logprobs(policy, p + p, ch + rj)
                rf = torch.cat([RC[idx], RR[idx]])
                with torch.no_grad():   # KL estimate from mismatched pairs, as in the paper
                    z = (lm.sequence_logprobs(policy, p, rj[1:] + rj[:1]) -
                         lm.sequence_logprobs(ref, p, rj[1:] + rj[:1])).mean().clamp(min=0)
                des = torch.cat([torch.ones(len(b), dtype=torch.bool), torch.zeros(len(b), dtype=torch.bool)])
                loss = kto_loss(lp, rf, des, beta, z)
            else:
                pc, pr = lm.sequence_logprobs(policy, p, ch), lm.sequence_logprobs(policy, p, rj)
                lc = torch.tensor([len(x) for x in ch], dtype=torch.float)
                lr_ = torch.tensor([len(x) for x in rj], dtype=torch.float)
                loss = preference_loss(kind, pc, pr, RC[idx], RR[idx], beta, lc, lr_, gamma)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
            step += 1
            if track is not None and step % 20 == 0:
                with torch.no_grad():
                    policy.eval()
                    tp = [d["prompt"] for d in track]
                    hist.append(dict(step=step, loss=loss.item(),
                                     logp_chosen=lm.sequence_logprobs(policy, tp, [d["chosen"] for d in track]).mean().item(),
                                     logp_rejected=lm.sequence_logprobs(policy, tp, [d["rejected"] for d in track]).mean().item()))
                    policy.train()
        if log:
            print(f"{kind} epoch {ep + 1}: loss {loss.item():.4f}")
    import pandas as pd
    return pd.DataFrame(hist)


# ---------------------------------------------------------------- RL with verifiable rewards (NB05)
def verifier(prompts, responses):
    return torch.tensor([tasks.gold_reward(*tasks.a_b_from_prompt(p), r) for p, r in zip(prompts, responses)])


def group_advantages(r, G, kind="grpo"):
    """r: (n_prompts * G,) rewards, grouped consecutively per prompt.

    none      : REINFORCE, advantage = reward                          (Williams, 1992)
    rloo      : leave-one-out baseline, r_i - mean_{j != i} r_j          (Ahmadian et al., 2024)
    grpo      : (r_i - mean) / std within the group                     (Shao et al., 2024)
    dr_grpo   : r_i - mean, without the std division                   (Liu et al., 2025)
    """
    g = r.view(-1, G)
    if kind == "none":
        a = g
    elif kind == "rloo":
        a = g - (g.sum(1, keepdim=True) - g) / (G - 1)
    elif kind == "grpo":
        a = (g - g.mean(1, keepdim=True)) / (g.std(1, keepdim=True) + 1e-4)
    elif kind == "dr_grpo":
        a = g - g.mean(1, keepdim=True)
    else:
        raise ValueError(kind)
    return a.reshape(-1)


def group_rl(policy, ref, steps=100, n_prompts=16, G=8, adv="grpo", beta=0.04, lr=1e-5, inner=2, clip=0.2,
             digits=3, reward_fn=verifier, seed=0, log_every=20, eval_every=None, eval_pairs=None):
    """GRPO-style training: sample G responses per prompt, score with a verifier, no value network."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    hist, t0 = [], time.time()
    for it in range(steps):
        pairs = tasks.sample_pairs(n_prompts, digits, rng)
        prompts = [tasks.make_prompt(a, b, rng=rng) for a, b in pairs]
        P = [p for p in prompts for _ in range(G)]
        R = lm.generate(policy, P)
        r = reward_fn(P, R).float()
        A = group_advantages(r, G, adv)
        x, y, m = lm.pack(P, R, policy.cfg["block"])
        with torch.no_grad():
            old = torch.gather(F.log_softmax(policy(x).float(), -1), 2, y[..., None])[..., 0]
            rf = torch.gather(F.log_softmax(ref(x).float(), -1), 2, y[..., None])[..., 0]
        policy.train()
        for _ in range(inner):
            lp = torch.gather(F.log_softmax(policy(x).float(), -1), 2, y[..., None])[..., 0]
            ratio = torch.exp(lp - old)
            pg = -torch.min(ratio * A[:, None], ratio.clamp(1 - clip, 1 + clip) * A[:, None])
            kl = torch.exp(rf - lp) - (rf - lp) - 1               # 'k3' estimator, as in GRPO
            loss = (((pg + beta * kl) * m).sum(1) / m.sum(1)).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        g = r.view(-1, G)
        row = dict(step=it + 1, reward=r.mean().item(), frac_groups_no_signal=((g.std(1) == 0).float().mean().item()),
                   kl=((lp.detach() - rf) * m).sum(1).mean().item(), time=time.time() - t0)
        if eval_every and (it + 1) % eval_every == 0:
            row["test_greedy"] = evaluate(policy, eval_pairs, digits=digits, n_samples=1)["greedy_acc"]
        hist.append(row)
        if log_every and (it + 1) % log_every == 0:
            print(f"iter {it + 1:4d}  train reward {row['reward']:.3f}  groups w/o signal {row['frac_groups_no_signal']:.2f}  KL {row['kl']:.3f}  ({row['time']:.0f}s)")
    import pandas as pd
    return pd.DataFrame(hist)


@torch.no_grad()
def pass_at_k(model, pairs, ks=(1, 2, 4, 8, 16, 32), n=32, temperature=1.0, seed=0):
    """Unbiased pass@k estimator (Chen et al., 2021) from n samples per problem."""
    from math import comb
    set_seed(seed)
    rng = np.random.default_rng(seed)
    prompts = [tasks.make_prompt(a, b, rng=rng) for a, b in pairs]
    R = lm.generate(model, [p for p in prompts for _ in range(n)], temperature=temperature)
    c = np.array([int(sum(tasks.gold_reward(a, b, R[i * n + j]) for j in range(n))) for i, (a, b) in enumerate(pairs)])
    return {k: float(np.mean([1 - comb(n - ci, k) / comb(n, k) for ci in c])) for k in ks}


def rejection_sampling_ft(policy, rounds=2, n_prompts=1500, G=8, digits=3, epochs=1, lr=2e-4, seed=0):
    """STaR / rejection-sampling fine-tuning: keep only verified-correct samples and run SFT on them."""
    rng = np.random.default_rng(seed)
    log = []
    for rd in range(rounds):
        pairs = tasks.sample_pairs(n_prompts, digits, rng)
        prompts = [tasks.make_prompt(a, b, rng=rng) for a, b in pairs]
        P = [p for p in prompts for _ in range(G)]
        R = lm.generate(policy, P)
        keep = [dict(prompt=p, response=r) for p, r in zip(P, R) if verifier([p], [r])[0] > 0]
        sft(policy, keep, epochs=epochs, lr=lr, seed=seed + rd, log=False)
        log.append(dict(round=rd + 1, kept=len(keep), frac_correct=len(keep) / len(P)))
    return log


@torch.no_grad()
def completion_accuracy(model, pairs, fmt="{a}+{b}="):
    """Greedy accuracy when the problem is posed the way pretraining text writes it (no chat format)."""
    outs = lm.generate(model, [fmt.format(a=a, b=b) for a, b in pairs], max_new=8, greedy=True)
    first = [o.split("\n")[0].strip().rstrip(".") for o in outs]
    return float(np.mean([f.split(" ")[0] == str(a + b) for f, (a, b) in zip(first, pairs)]))
