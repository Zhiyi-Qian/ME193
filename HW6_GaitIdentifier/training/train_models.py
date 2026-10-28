"""
Gait classifier training: standing / walking / running / biking
Model 1: uses only the peak acceleration on ONE axis
Model 2: uses features from all 6 IMU outputs
No neural nets: both models are decision trees.

Usage:  put your CSV files in a folder called "data" next to this script, then
        python train_models.py
"""
import glob
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ---------------- settings you can change ----------------
DATA_DIR = "data"
RATE = 50                 # samples per second (matches the sketch)
WINDOW_SEC = 2.0          # length of each chunk we classify
TRIM_SEC = 3.0            # cut this much off the start/end of each file (strapping on, stopping)
TRAIN_FRACTION = 0.7      # first 70% of each recording trains, last 30% tests
MODEL1_AXIS = None        # e.g. "ax". None = automatically pick the best axis
CLASSES = ["standing", "walking", "running", "biking"]
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]

WIN = int(WINDOW_SEC * RATE)
STEP = WIN // 2           # 50% overlap between windows


def make_windows(df):
    """Chop one recording into overlapping windows."""
    return [df.iloc[s:s + WIN] for s in range(0, len(df) - WIN + 1, STEP)]


def peak_feature(w, axis):
    """Model 1's single feature: biggest absolute acceleration on one axis."""
    return np.abs(w[axis].values).max()


def all_features(w):
    """Model 2's features: summary stats of all 6 channels + overall motion."""
    feats = {}
    for c in CHANNELS:
        v = w[c].values
        feats[f"{c}_mean"] = v.mean()        # mean accel = which way gravity points
        feats[f"{c}_std"] = v.std()          # how much it's moving
        feats[f"{c}_max"] = v.max()
        feats[f"{c}_min"] = v.min()
    acc_mag = np.sqrt(w.ax**2 + w.ay**2 + w.az**2)
    gyro_mag = np.sqrt(w.gx**2 + w.gy**2 + w.gz**2)
    feats["acc_mag_std"] = acc_mag.std()
    feats["gyro_mag_mean"] = gyro_mag.mean()
    # dominant frequency of accel magnitude = step or pedal cadence (Hz)
    spec = np.abs(np.fft.rfft(acc_mag - acc_mag.mean()))
    freqs = np.fft.rfftfreq(len(acc_mag), d=1 / RATE)
    feats["cadence_hz"] = freqs[1:][np.argmax(spec[1:])]
    return feats


# ---------------- load and window the data ----------------
train_w, test_w = [], []
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
if not files:
    raise SystemExit(f"No CSV files found in '{DATA_DIR}/'")

print("Recordings:")
for path in files:
    df = pd.read_csv(path)
    label = df["label"].iloc[0]
    trim = int(TRIM_SEC * RATE)
    df = df.iloc[trim:len(df) - trim].reset_index(drop=True)
    split = int(len(df) * TRAIN_FRACTION)    # split in TIME so test data is truly unseen
    tr, te = make_windows(df.iloc[:split]), make_windows(df.iloc[split:])
    train_w += [(w, label) for w in tr]
    test_w += [(w, label) for w in te]
    print(f"  {os.path.basename(path):35s} {len(df) / RATE:6.1f} s  -> {len(tr)} train / {len(te)} test windows")

y_train = [lab for _, lab in train_w]
y_test = [lab for _, lab in test_w]
classes = [c for c in CLASSES if c in set(y_train)]

# ---------------- Model 1: peak acceleration on one axis ----------------
def model1_data(axis):
    Xtr = np.array([[peak_feature(w, axis)] for w, _ in train_w])
    Xte = np.array([[peak_feature(w, axis)] for w, _ in test_w])
    return Xtr, Xte

if MODEL1_AXIS is None:
    scores = {}
    for axis in ["ax", "ay", "az"]:
        Xtr, Xte = model1_data(axis)
        m = DecisionTreeClassifier(max_depth=3, random_state=0).fit(Xtr, y_train)
        scores[axis] = accuracy_score(y_train, m.predict(Xtr))
    MODEL1_AXIS = max(scores, key=scores.get)
    print(f"\nModel 1 axis chosen automatically: {MODEL1_AXIS}  (train accuracy per axis: "
          + ", ".join(f"{a}={s:.2f}" for a, s in scores.items()) + ")")

X1_tr, X1_te = model1_data(MODEL1_AXIS)
model1 = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X1_tr, y_train)
acc1 = accuracy_score(y_test, model1.predict(X1_te))

# ---------------- Model 2: all IMU outputs ----------------
X2_tr = pd.DataFrame([all_features(w) for w, _ in train_w])
X2_te = pd.DataFrame([all_features(w) for w, _ in test_w])
model2 = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=0).fit(X2_tr, y_train)
acc2 = accuracy_score(y_test, model2.predict(X2_te))

print(f"\nTest accuracy  Model 1 (peak {MODEL1_AXIS} only): {acc1:.1%}")
print(f"Test accuracy  Model 2 (all IMU outputs):    {acc2:.1%}")

importances = pd.Series(model2.feature_importances_, index=X2_tr.columns)
print("\nFeatures Model 2 relies on most:")
print(importances[importances > 0].sort_values(ascending=False).round(3).to_string())

# ---------------- plots for your Notion page ----------------
os.makedirs("plots", exist_ok=True)

# Model 1: distribution of the peak value per class
plt.figure(figsize=(8, 4))
for c in classes:
    vals = [x[0] for x, lab in zip(X1_tr, y_train) if lab == c]
    plt.hist(vals, bins=20, alpha=0.6, label=c)
tree = model1.tree_
for t in tree.threshold[tree.feature >= 0]:
    plt.axvline(t, color="k", ls="--", lw=1)
plt.xlabel(f"peak |{MODEL1_AXIS}| in a {WINDOW_SEC:.0f} s window (g)")
plt.ylabel("number of windows")
plt.title("Model 1: peak acceleration per activity (dashed = decision thresholds)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/model1_thresholds.png", dpi=150)

# Model 2: the decision tree itself
plt.figure(figsize=(18, 9))
plot_tree(model2, feature_names=X2_tr.columns, class_names=model2.classes_,
          filled=True, rounded=True, fontsize=8, impurity=False)
plt.title("Model 2: decision tree using all IMU outputs")
plt.tight_layout()
plt.savefig("plots/model2_tree.png", dpi=150)

# Confusion matrices side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(y_test, model1.predict(X1_te), labels=classes,
                                        ax=axes[0], colorbar=False)
axes[0].set_title(f"Model 1 (peak {MODEL1_AXIS})  accuracy {acc1:.0%}")
ConfusionMatrixDisplay.from_predictions(y_test, model2.predict(X2_te), labels=classes,
                                        ax=axes[1], colorbar=False)
axes[1].set_title(f"Model 2 (all IMU)  accuracy {acc2:.0%}")
plt.tight_layout()
plt.savefig("plots/confusion_matrices.png", dpi=150)

# Save both models so the board can use them live
joblib.dump({"model": model1, "axis": MODEL1_AXIS}, "model1.joblib")
joblib.dump({"model": model2, "features": list(X2_tr.columns)}, "model2.joblib")
print("\nSaved plots/ and model1.joblib, model2.joblib")
