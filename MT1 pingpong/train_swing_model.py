"""
Train the swing classifier from swing_data.jsonl.

Compares two models with 5-fold cross-validation and keeps the better one:
  - k-nearest neighbours (KNN): a new swing gets the label most common among
    the k most similar recorded swings.
  - decision tree: learns a series of yes/no questions on the features
    ("is peak gyro > 3200?") that split the swings into labels.

Prints accuracy and a confusion matrix (rows = truth, columns = predicted)
so you can report how well it works against your labeled ground truth.
"""

import json
from collections import Counter

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import config
from swing import LABELS, features


def load(path=config.DATA_PATH):
    X, y = [], []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            X.append(features(row["window"]))
            y.append(row["label"])
    return np.array(X), np.array(y)


def main(path=config.DATA_PATH, out=config.MODEL_PATH):
    X, y = load(path)
    print(f"{len(y)} examples:", dict(Counter(y.tolist())))
    folds = min(5, min(Counter(y).values()))
    if folds < 2:
        raise SystemExit("Need at least 2 examples of every label.")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)

    candidates = {
        "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
        "Decision tree": DecisionTreeClassifier(max_depth=5, random_state=0),
    }
    best_name, best_acc = None, -1
    labels = [l for l in LABELS if l in set(y)]
    for name, model in candidates.items():
        pred = cross_val_predict(model, X, y, cv=cv)
        acc = (pred == y).mean()
        print(f"\n{name}: cross-validated accuracy = {acc:.1%}")
        cm = confusion_matrix(y, pred, labels=labels)
        print("            " + " ".join(f"{l[:8]:>9s}" for l in labels))
        for lbl, row in zip(labels, cm):
            print(f"{lbl:>10s}  " + " ".join(f"{v:9d}" for v in row))
        if acc > best_acc:
            best_name, best_acc = name, acc

    model = candidates[best_name].fit(X, y)   # retrain on ALL the data
    joblib.dump(model, out)
    print(f"\nKept {best_name} ({best_acc:.1%}). Saved to {out}")
    return best_acc


if __name__ == "__main__":
    main()
