"""
Step counter: trains a small 1D convolutional neural network in PyTorch.

How the labels are made
  Each file name holds the true step count (walk_100_... = 100 steps).
  The script finds when you were moving (rotation above a threshold) and
  spreads that many steps evenly over the moving time. Every 2 s window then
  gets a label: how many steps happened in its middle 1 s.

How it is tested
  Leave-one-recording-out: for each recording, a fresh network is trained on
  all the OTHER recordings and then counts the steps in the one it never saw.
  That gives an honest error for every recording -> your uncertainty analysis.
  A final network is then trained on everything and exported for the board.

Usage:  put the CSVs in a folder called "steps" next to this script, then
        python3 train_steps.py
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

from step_net import CENTER, RATE, WIN, StepNet

DATA_DIR = "steps"
HOP = 5            # training windows start every 0.1 s (lots of overlapping examples)
EPOCHS = 40
BATCH = 64
LR = 1e-3
TRIM = 1.0         # seconds cut from start/end (pressing Run/Stop)
torch.manual_seed(0)
np.random.seed(0)
os.makedirs("plots", exist_ok=True)


# ---------------------------------------------------------------- data
def load(path):
    name = os.path.basename(path)
    parts = name.split("_")
    true_steps = int(parts[1])
    df = pd.read_csv(path)
    raw = df[["ax", "ay", "az", "gx", "gy", "gz"]].values.astype(np.float32)
    trim = int(TRIM * RATE)
    raw = raw[trim:len(raw) - trim]
    return {"name": name, "kind": parts[0], "steps": true_steps, "raw": raw}


def channels(raw):
    acc = np.linalg.norm(raw[:, :3], axis=1, keepdims=True)
    gyr = np.linalg.norm(raw[:, 3:], axis=1, keepdims=True)
    return np.hstack([raw, acc, gyr]).T          # 8 x T


def moving_mask(raw):
    """True wherever the leg is moving (smoothed rotation above a threshold)."""
    g = np.linalg.norm(raw[:, 3:], axis=1)
    smooth = np.convolve(g, np.ones(25) / 25, mode="same")
    thresh = max(20.0, 0.25 * np.percentile(smooth, 95))
    m = smooth > thresh
    # fill short gaps (< 0.6 s) and drop short blips (< 0.6 s)
    for target, other in [(False, True), (True, False)]:
        i = 0
        while i < len(m):
            if m[i] == target:
                j = i
                while j < len(m) and m[j] == target:
                    j += 1
                if j - i < 30 and i > 0 and j < len(m):
                    m[i:j] = other
                i = j
            else:
                i += 1
    return m


def step_rate(rec):
    """Steps per sample: the true count spread evenly over the moving time."""
    m = moving_mask(rec["raw"])
    rate = np.zeros(len(m), dtype=np.float32)
    if rec["steps"] > 0 and m.sum() > 0:
        rate[m] = rec["steps"] / m.sum()
    rec["moving_s"] = m.sum() / RATE
    return rate


def windows(rec, hop):
    x = channels(rec["raw"])
    rate = rec["rate"]
    X, y = [], []
    for s in range(0, x.shape[1] - WIN + 1, hop):
        X.append(x[:, s:s + WIN])
        c0 = s + (WIN - CENTER) // 2
        y.append(rate[c0:c0 + CENTER].sum())
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


recs = [load(p) for p in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))]
if not recs:
    raise SystemExit(f"No CSV files found in '{DATA_DIR}/'")

print("Recordings (cadence is a sanity check: normal walking is roughly 90-130 steps/min):")
for r in recs:
    r["rate"] = step_rate(r)
    cad = r["steps"] / (r["moving_s"] / 60) if r["moving_s"] > 0 and r["steps"] else 0
    print(f"  {r['name']:36s} {len(r['raw']) / RATE:5.1f} s, moving {r['moving_s']:5.1f} s, "
          f"{r['steps']:3d} steps -> {cad:5.1f} steps/min")


# ---------------------------------------------------------------- model
class StepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(8, 12, 5, padding=2)
        self.conv2 = nn.Conv1d(12, 16, 5, padding=2)
        self.conv3 = nn.Conv1d(16, 16, 5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(16 * 25, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.fc1(x.flatten(1)))
        return nn.functional.softplus(self.fc2(x)).squeeze(1)


def train(train_recs, val_recs=None):
    X, y = map(np.concatenate, zip(*[windows(r, HOP) for r in train_recs]))
    mean = X.mean(axis=(0, 2))
    std = X.std(axis=(0, 2)) + 1e-6
    norm = lambda a: (a - mean[None, :, None]) / std[None, :, None]
    Xt, yt = torch.tensor(norm(X)), torch.tensor(y)
    if val_recs:
        Xv, yv = map(np.concatenate, zip(*[windows(r, HOP) for r in val_recs]))
        Xv, yv = torch.tensor(norm(Xv)), torch.tensor(yv)

    model = StepCNN()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    hist = {"train": [], "val": []}
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xt))
        total = 0.0
        for i in range(0, len(Xt), BATCH):
            idx = perm[i:i + BATCH]
            xb = Xt[idx] + 0.05 * torch.randn_like(Xt[idx])   # small noise = augmentation
            loss = loss_fn(model(xb), yt[idx])
            opt.zero_grad()
            loss.backward()          # back-propagation: compute how each weight affected the error
            opt.step()               # nudge every weight a little in the direction that lowers it
            total += loss.item() * len(idx)
        hist["train"].append(total / len(Xt))
        if val_recs:
            model.eval()
            with torch.no_grad():
                hist["val"].append(loss_fn(model(Xv), yv).item())
    model.eval()
    return model, mean, std, hist


def count_steps(model, mean, std, rec):
    """Slide the window 1 s at a time; the middle 1 s pieces tile the recording."""
    x = channels(rec["raw"])
    starts = range(0, x.shape[1] - WIN + 1, CENTER)
    Xw = np.array([x[:, s:s + WIN] for s in starts], dtype=np.float32)
    Xw = (Xw - mean[None, :, None]) / std[None, :, None]
    with torch.no_grad():
        return model(torch.tensor(Xw)).sum().item()


# ---------------------------------------------------------------- leave-one-out test
print("\nLeave-one-recording-out test (each recording counted by a net that never saw it):")
results, curves = [], []
for i, test in enumerate(recs):
    others = [r for j, r in enumerate(recs) if j != i]
    model, mean, std, hist = train(others, [test])
    pred = count_steps(model, mean, std, test)
    results.append({"recording": test["name"], "kind": test["kind"],
                    "true": test["steps"], "predicted": round(pred, 1),
                    "error": round(pred - test["steps"], 1)})
    curves.append(hist)
    print(f"  {test['name']:36s} true {test['steps']:3d}   predicted {pred:6.1f}   error {pred - test['steps']:+6.1f}")

res = pd.DataFrame(results)
res.to_csv("loo_results.csv", index=False)

walks = res[res["true"] > 0].copy()
walks["pct_error"] = 100 * walks["error"] / walks["true"]
n = len(walks)
mean_pct, sd_pct = walks["pct_error"].mean(), walks["pct_error"].std(ddof=1)
t = stats.t.ppf(0.975, n - 1) if n > 1 else float("nan")
ci = t * sd_pct / np.sqrt(n) if n > 1 else float("nan")
mae = walks["error"].abs().mean()
print(f"\nUncertainty (walking recordings, n = {n}):")
print(f"  mean absolute error:   {mae:.1f} steps")
print(f"  mean error:            {walks['error'].mean():+.1f} steps ({mean_pct:+.1f}%)")
print(f"  std of % error:        {sd_pct:.1f}%")
print(f"  95% CI of mean % error: {mean_pct:+.1f}% +/- {ci:.1f}%")
still = res[res["true"] == 0]
for _, r in still.iterrows():
    print(f"  false steps while standing ({r['recording']}): {r['predicted']:.1f}")

# ---------------------------------------------------------------- plots
plt.figure(figsize=(6, 6))
colors = {"walk": "tab:green", "slowwalk": "tab:blue", "fastwalk": "tab:red", "stopgo": "tab:purple", "stand": "tab:gray"}
for kind, g in res.groupby("kind"):
    plt.scatter(g["true"], g["predicted"], s=60, label=kind, color=colors.get(kind))
hi = max(res["true"].max(), res["predicted"].max()) * 1.1 + 5
plt.plot([0, hi], [0, hi], "k--", lw=1, label="perfect")
plt.xlabel("true steps (counted)")
plt.ylabel("predicted steps")
plt.title("Step counter on recordings it never trained on")
plt.legend()
plt.tight_layout()
plt.savefig("plots/predicted_vs_true.png", dpi=150)

tr = np.mean([c["train"] for c in curves], axis=0)
va = np.mean([c["val"] for c in curves], axis=0)
plt.figure(figsize=(7, 4))
plt.plot(range(1, EPOCHS + 1), tr, label="training loss")
plt.plot(range(1, EPOCHS + 1), va, label="loss on unseen recording")
plt.xlabel("epoch (one pass through the training data)")
plt.ylabel("mean squared error")
plt.yscale("log")
plt.title("Back-propagation lowering the error (average over folds)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/loss_curve.png", dpi=150)

# ---------------------------------------------------------------- final model for the board
print("\nTraining the final network on all recordings...")
model, mean, std, _ = train(recs)
torch.save(model.state_dict(), "step_model.pt")
params = {k: v.tolist() for k, v in model.state_dict().items()}
with open("step_model.json", "w") as fh:
    json.dump({"mean": mean.tolist(), "std": std.tolist(), "params": params}, fh)

# check the plain-Python version gives the same answer as PyTorch
net = StepNet("step_model.json")
r = next(r for r in recs if r["steps"] > 0)
seg = r["raw"][200:200 + WIN]
py = net.predict([tuple(v) for v in seg])
xt = (channels(seg)[None] - mean[None, :, None]) / std[None, :, None]
with torch.no_grad():
    pt = model(torch.tensor(xt.astype(np.float32))).item()
print(f"Check: PyTorch {pt:.5f} vs plain Python {py:.5f}  ->  {'MATCH' if abs(pt - py) < 1e-3 else 'MISMATCH'}")
n_params = sum(p.numel() for p in model.parameters())
print(f"Saved step_model.pt, step_model.json ({n_params} weights), loo_results.csv, plots/")
