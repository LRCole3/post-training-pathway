"""Shared scorecard. Every notebook appends rows to results/scorecard.csv.

Language models (NB01-05, NB10): task accuracy (greedy and sampled), gold reward, KL to the SFT
reference, style mix, well-formedness.
Images (NB06-10): computed in the frozen judge's feature space (as FID uses Inception features):
  fd           Frechet distance between real and generated features (lower is better; 'FID-like')
  precision    fraction of samples inside the real-data manifold (quality; Naeem et al., 2020)
  coverage     fraction of real points with a sample nearby (diversity; Naeem et al., 2020)
  judge_acc    for class-conditional samples, how often the judge recognises the requested class
  nn_ratio     nearest-train / nearest-test distance; well below 1 means memorisation
  nfe          network evaluations per sample (sampling cost)
"""
import os
import numpy as np
import pandas as pd
import torch
from scipy import linalg
from . import RESULTS_DIR


def append_rows(rows, notebook, path=None):
    path = path or os.path.join(RESULTS_DIR, "scorecard.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows)
    df.insert(0, "notebook", notebook)
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[~((old.notebook == notebook) & old.model.isin(df.model))]  # re-running replaces rows
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)
    return df


def load_scorecard(path=None):
    path = path or os.path.join(RESULTS_DIR, "scorecard.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def _frechet(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean.real))


def _prc(real, fake, k=5):
    from sklearn.neighbors import NearestNeighbors
    nn_r = NearestNeighbors(n_neighbors=k + 1).fit(real)
    radii = nn_r.kneighbors(real)[0][:, -1]
    d_fr = np.linalg.norm(fake[:, None] - real[None], axis=-1)       # (n_fake, n_real)
    precision = float((d_fr <= radii[None]).any(1).mean())
    coverage = float((d_fr <= radii[None]).any(0).mean())
    return precision, coverage


class ImageScorecard:
    """Scores generated 8x8 digits against the held-out split in judge-feature space."""

    def __init__(self, judge, X_train, X_test, y_test):
        self.judge, self.Xtr, self.Xte, self.yte = judge, X_train, X_test, y_test
        with torch.no_grad():
            self.f_te = judge.features(X_test).numpy()
        self.rows = []

    @torch.no_grad()
    def evaluate(self, samples, name, labels=None, nfe=None, **extra):
        s = samples.clamp(-1, 1)
        f = self.judge.features(s).numpy()
        prec, cov = _prc(self.f_te, f[:len(self.f_te)])
        d_tr = torch.cdist(s, self.Xtr).min(1).values
        d_te = torch.cdist(s, self.Xte).min(1).values
        row = dict(model=name, fd=_frechet(self.f_te, f), precision=prec, coverage=cov,
                   nn_ratio=float((d_tr / d_te).median()), nfe=nfe)
        if labels is not None:
            row["judge_acc"] = float((self.judge(s).argmax(1) == labels).float().mean())
        row.update(extra)
        self.rows.append(row)
        return row

    def real_baseline(self):
        """The held-out test set scored against itself via the training split: the 'perfect model' row."""
        return self.evaluate(self.Xtr[:len(self.Xte)], "real data (train split)", labels=None, nfe=0)

    def table(self):
        return pd.DataFrame(self.rows).set_index("model")
