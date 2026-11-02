# Post-Training, From First Principles: Transformers → Diffusion → Diffusion Transformers

A progressive, hands-on series on **post-training**, everything done to a model after pretraining:
supervised fine-tuning, LoRA, reward models and RLHF, DPO, RL with verifiable rewards (GRPO), reward
alignment and distillation of diffusion models, and post-training of diffusion transformers.

Each method is introduced **because the previous one fails at something measurable**, and a shared
scorecard (`results/scorecard.csv`) makes those failures visible. Every model is small enough to train on
a laptop CPU in minutes, and every notebook cites (and links) the papers it implements.

## Why toy-sized models?

Real post-training happens at a scale no tutorial can reproduce. This series instead builds a small world
with a **ground truth**:

* **Language (01–05):** a 0.8M-parameter character-level GPT pretrained on synthetic text, then asked
  arithmetic questions in a chat format. A programmatic **verifier** gives the "gold" reward, so we can
  measure exactly when a learned reward model is being fooled, which is impossible with human preferences at scale.
* **Images (06–10):** diffusion models and diffusion transformers on scikit-learn's 8×8 handwritten digits
  (no download), scored by a frozen "judge" classifier that plays the role of FID's Inception network and of
  learned image rewards.

The trade-off: these notebooks show the **mechanisms and failure modes** of each method, not the headline
numbers from the papers. Results are reported as measured, including the ones that didn't go the way the
narrative first expected.

## Learning path

| # | Notebook | Methods | Question it answers | Limitation that motivates the next step | Status |
|---|---|---|---|---|---|
| 01 | [Pretraining a transformer LM](01_pretraining_transformer_lm.ipynb) | Char-level GPT, next-token loss, AdamW + cosine, mini scaling study | What does a base model know, and why won't it answer? | Knows arithmetic, scores 0% in chat format | available |
| 02 | [Supervised fine-tuning & LoRA](02_sft_and_lora.ipynb) | Masked SFT loss, full FT vs LoRA, forgetting, LIMA-style data curation | Can demonstrations teach the behaviour? | Copies demonstrators' mistakes; clean demos are expensive | available |
| 03 | [Reward models & RLHF with PPO](03_reward_models_and_rlhf_ppo.ipynb) | Bradley–Terry RM, PPO + KL penalty, GAE, best-of-n | Can we learn from comparisons instead? | Reward hacking; four models; fragile | available |
| 04 | [Direct preference optimization](04_direct_preference_optimization.ipynb) | DPO, IPO, KTO, SimPO | Can we drop the reward model and the RL loop? | Offline: re-weights existing behaviour only | available |
| 05 | [RL with verifiable rewards](05_rlvr_grpo_reasoning.ipynb) | REINFORCE, RLOO, GRPO, Dr. GRPO, STaR, pass@k | What does RL add when the reward can't be fooled? | Sharpens rather than expands; autoregressive only | available |
| 06 | [Diffusion foundations](06_diffusion_foundations.ipynb) | DDPM, ε/v-prediction, DDIM, classifier-free guidance | A different way to generate | Matches data, not goals; many steps | available |
| 07 | [Post-training diffusion models](07_posttraining_diffusion_alignment.ipynb) | LoRA/DreamBooth-style adaptation, DDPO, DRaFT, Diffusion-DPO | Can RLHF ideas steer a denoiser? | Reward hacking again; still slow | coming next |
| 08 | [Distillation for few-step sampling](08_diffusion_distillation_few_step.ipynb) | Progressive distillation, consistency distillation, reflow | Can post-training make sampling 10-100x cheaper? | Small denoisers don't scale | coming next |
| 09 | [Diffusion transformers & flow matching](09_diffusion_transformers_flow_matching.ipynb) | Patch tokens, adaLN-Zero DiT, rectified flow, joint text-image attention | What architecture do modern image models use? | Needs post-training too | coming next |
| 10 | [Capstone: post-training a DiT](10_capstone_posttraining_dit.ipynb) | Flow-GRPO, Diffusion-DPO on a DiT, masked diffusion LM + GRPO, landscape map | How do the two threads meet? | → research frontier | coming next |

## Getting started

```bash
git clone https://github.com/LRCole3/post-training-pathway.git
cd post-training-pathway
pip install -r requirements.txt
jupyter lab
```

Run the notebooks **in order, from the repository root**. Each notebook saves its models to `checkpoints/`
(git-ignored). If a checkpoint a notebook needs is missing, `ptlib` re-creates it with the same settings as
the notebook that originally trained it, so any notebook can also be run on its own. No GPU is needed.

## Repository layout

```
├── 01_ … 10_*.ipynb
├── ptlib/
│   ├── tasks.py      # synthetic corpus, chat format, noisy annotators, verifier (gold reward)
│   ├── lm.py         # character-level GPT, LoRA, KV-cached generation, log-prob helpers
│   ├── align.py      # SFT, reward model, PPO, DPO/IPO/KTO/SimPO, GRPO/RLOO, pass@k
│   ├── images.py     # 8x8 digits, judge classifier, plotting
│   ├── diffusion.py  # DDPM/DDIM, guidance, flow matching, DiT (notebooks 06-10)
│   └── metrics.py    # shared scorecard (LM and image metrics)
├── research_papers/
│   ├── README.md     # every cited paper, by notebook, with verified arXiv/DOI links
│   ├── catalog.py    # the citation list (keys, arXiv IDs, notebooks)
│   └── make_index.py # regenerates README.md from catalog.py + arxiv_meta.json
├── results/scorecard.csv
└── requirements.txt
```

Each notebook follows the same format: an overview that places it in the series, learning objectives,
worked sections, **Challenge Problems** with a solution cell immediately below, a final summary and
references with links.

## Credit

The teaching format follows the
[Protein_Design `ppi_deep_learning` series](https://github.com/LRCole3/Protein_Design/tree/main/ppi_deep_learning).
The methods belong to the authors cited in each notebook; see [`research_papers/README.md`](research_papers/README.md).
