"""8x8 handwritten digits (scikit-learn, no download) and a frozen 'judge' classifier.

The judge plays the role that Inception networks play for FID and that CLIP/aesthetic models
play as rewards for text-to-image models: it scores samples and supplies features for metrics.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import CKPT_DIR, set_seed


def load_digits_tensor(seed=0, test_frac=0.2):
    """Returns (X_train, y_train, X_test, y_test); X is (N, 64) scaled to [-1, 1]."""
    from sklearn.datasets import load_digits
    d = load_digits()
    X = torch.tensor(d.data / 8.0 - 1.0, dtype=torch.float32)
    y = torch.tensor(d.target, dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=g)
    n_te = int(test_frac * len(X))
    te, tr = perm[:n_te], perm[n_te:]
    return X[tr], y[tr], X[te], y[te]


def bold(X, amount=1):
    """'Bold' style: grey-scale dilation (3x3 max filter) of the strokes, used as a new style in NB07."""
    img = X.view(-1, 1, 8, 8)
    for _ in range(amount):
        img = F.max_pool2d(img, 3, stride=1, padding=1)
    return img.reshape(-1, 64)


class Judge(nn.Module):
    """Small CNN classifier. `features` gives a 64-d embedding used by the scorecard."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 64, 3, padding=1),
                                  nn.GELU(), nn.MaxPool2d(2), nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
                                  nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Linear(64, 10)

    def features(self, x):
        return self.conv(x.view(-1, 1, 8, 8))

    def forward(self, x):
        return self.fc(self.features(x))


def get_judge(verbose=True):
    """Train once (about 20 s) on the training split with noise augmentation, then cache."""
    path = os.path.join(CKPT_DIR, "judge.pt")
    j = Judge()
    if os.path.exists(path):
        j.load_state_dict(torch.load(path, weights_only=True)); j.eval()
        return j.requires_grad_(False)
    set_seed(0)
    Xtr, ytr, Xte, yte = load_digits_tensor()
    opt = torch.optim.Adam(j.parameters(), 1e-3)
    for ep in range(40):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            xb, yb = Xtr[perm[i:i + 64]], ytr[perm[i:i + 64]]
            thick = torch.rand(len(xb)) < 0.4                  # 40% of each batch drawn with bold strokes,
            xb = torch.where(thick[:, None], bold(xb), xb)      # so the judge also recognises bold digits
            xb = (xb + 0.1 * torch.randn_like(xb)).clamp(-1, 1)
            loss = F.cross_entropy(j(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    j.eval()
    with torch.no_grad():
        acc = (j(Xte).argmax(1) == yte).float().mean().item()
        acc_b = (j(bold(Xte)).argmax(1) == yte).float().mean().item()
    if verbose:
        print(f"judge trained: held-out accuracy {acc:.3f} (normal), {acc_b:.3f} (bold strokes)")
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save(j.state_dict(), path)
    return j.requires_grad_(False)


def show_grid(X, nrow=10, ax=None, title=None):
    import matplotlib.pyplot as plt
    X = X.detach().cpu().view(-1, 8, 8).clamp(-1, 1).numpy()
    n = len(X); ncol = int(np.ceil(n / nrow))
    canvas = np.ones((ncol * 9 - 1, nrow * 9 - 1)) * -1
    for k in range(n):
        r, c = divmod(k, nrow)
        canvas[r * 9:r * 9 + 8, c * 9:c * 9 + 8] = X[k]
    ax = ax or plt.gca()
    ax.imshow(canvas, cmap="gray_r", vmin=-1, vmax=1); ax.axis("off")
    if title:
        ax.set_title(title)
    return ax


class PixelKNN:
    """A second, very different classifier (k-nearest neighbours on raw pixels, normal + bold training digits).

    Notebooks 07 and 10 optimise rewards that involve the CNN judge; this one is never optimised against, so it
    plays the role of a held-out human evaluation."""

    def __init__(self, k=5):
        from sklearn.neighbors import KNeighborsClassifier
        Xtr, ytr, _, _ = load_digits_tensor()
        X = torch.cat([Xtr, bold(Xtr)]).numpy(); y = torch.cat([ytr, ytr]).numpy()
        self.clf = KNeighborsClassifier(k).fit(X, y)

    def accuracy(self, x, labels):
        return float((self.clf.predict(x.clamp(-1, 1).numpy()) == labels.numpy()).mean())
