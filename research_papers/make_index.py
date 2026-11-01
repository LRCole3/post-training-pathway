"""Regenerate research_papers/README.md from catalog.py and the verified metadata in arxiv_meta.json.

Every arXiv ID was checked against the arXiv API and every DOI against CrossRef (see arxiv_meta.json).
Run:  python research_papers/make_index.py
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from catalog import PAPERS, DOI_PAPERS

META = json.load(open(os.path.join(HERE, "arxiv_meta.json")))
KEYS = {k: (i, nb) for k, i, nb in PAPERS + DOI_PAPERS}

NB_TITLES = {
    "01": "Pretraining a transformer language model", "02": "Supervised fine-tuning and LoRA",
    "03": "Reward models and RLHF with PPO", "04": "Direct preference optimization",
    "05": "RL with verifiable rewards (GRPO)", "06": "Diffusion model foundations",
    "07": "Post-training diffusion: adaptation and reward alignment", "08": "Distillation for few-step sampling",
    "09": "Diffusion transformers and flow matching", "10": "Capstone: post-training a diffusion transformer",
}


def _surname(name):
    parts = name.split()
    return parts[-1] if parts else name


def authors_short(k):
    a = META[KEYS[k][0]]["authors"]
    if len(a) == 1:
        return _surname(a[0])
    if len(a) == 2:
        return f"{_surname(a[0])} & {_surname(a[1])}"
    return f"{_surname(a[0])} et al."


def url(k):
    i = KEYS[k][0]
    return f"https://doi.org/{i}" if i.startswith("10.") else f"https://arxiv.org/abs/{i}"


def year(k):
    return META[KEYS[k][0]]["published"][:4]


def cite(k):
    """Inline citation with link, e.g. [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)."""
    return f"[{authors_short(k)}, {year(k)}]({url(k)})"


def reference(k):
    i, _ = KEYS[k]
    m = META[i]
    a = m["authors"]
    names = ", ".join(a[:3]) + (" et al." if len(a) > 3 else "")
    where = f"*{m['venue']}*" if m.get("venue") else f"arXiv:{i}"
    return f"{names} ({year(k)}). **{m['title']}**. {where}. [{url(k).split('//')[1]}]({url(k)})"


def main():
    lines = ["# Research papers", "",
             "Every paper cited in the series, grouped by the notebook that uses it. Links point to arXiv",
             "(or the publisher DOI); PDFs are not redistributed here because most arXiv licences do not permit it.",
             "All identifiers were verified programmatically against the arXiv API and CrossRef",
             "(`arxiv_meta.json`); regenerate this page with `python research_papers/make_index.py`.", ""]
    for nb, title in NB_TITLES.items():
        ks = [k for k, (i, nbs) in KEYS.items() if nb in nbs.split(",")]
        lines += [f"## Notebook {nb} · {title}", ""]
        lines += [f"* {reference(k)}" for k in sorted(ks, key=lambda k: (year(k), k))]
        lines.append("")
    open(os.path.join(HERE, "README.md"), "w").write("\n".join(lines))
    print(f"wrote README.md with {len(KEYS)} papers")


if __name__ == "__main__":
    main()
