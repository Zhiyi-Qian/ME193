"""
Train a CNN to count fingers from the images in data/0 ... data/10
(0-9 = number of fingers, 10 = no hand / empty frame).

Outputs:
  finger_model.pth       the best model (by validation accuracy), used later for the lock
  training_curves.png    loss and accuracy vs epoch (for your Notion)
  confusion_matrix.png   which numbers get mixed up

Run:  python3 train.py
      python3 train.py --epochs 40
"""
import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--classes", type=int, default=11, help="0-9 fingers + 10 = no hand")
    p.add_argument("--size", type=int, default=96, help="images are resized to size x size")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--out", default="finger_model.pth")
    p.add_argument("--device", default="cpu", help="cpu (default, works everywhere) or mps / cuda")
    return p.parse_args()


def class_names(n):
    return [("blank" if c == 10 else str(c)) for c in range(n)]


# ---------------------------------------------------------------- model
class FingerNet(nn.Module):
    """3 convolution layers (each followed by ReLU and max pooling), then one linear layer."""

    def __init__(self, n_classes=10, size=96):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),   # look for edges / simple shapes
            nn.ReLU(),
            nn.MaxPool2d(2),                              # size -> size/2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # combine edges into finger parts
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> size/4
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # combine parts into whole fingers
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> size/8
        )
        flat = 64 * (size // 8) * (size // 8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(flat, n_classes),                   # one score per number
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------- data
class FingerDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def split_data(data_dir, n_classes, val_frac):
    """Photos were taken in bursts, so neighbours look almost identical.
    To get an honest validation score, hold out the LAST part of each class's
    recording (by time) instead of picking random photos."""
    train, val = [], []
    for c in range(n_classes):
        folder = os.path.join(data_dir, str(c))
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".png"))
        if not files:
            raise SystemExit(f"No images found in {folder}")
        # file names contain a millisecond timestamp, so sort by it
        files.sort(key=lambda f: int(f.split("_")[2]) if len(f.split("_")) > 2 and f.split("_")[2].isdigit() else 0)
        n_val = max(1, int(len(files) * val_frac))
        paths = [(os.path.join(folder, f), c) for f in files]
        train += paths[:-n_val]
        val += paths[-n_val:]
        print(f"class {c}: {len(files) - n_val} train, {n_val} val")
    random.shuffle(train)
    return train, val


def main():
    args = parse_args()
    random.seed(0)
    torch.manual_seed(0)

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print("Using device:", device)

    norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    # Data augmentation: random rotations, shifts, zooms and lighting changes
    # make many "new" training images out of each photo.
    train_tf = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        norm,
    ])
    val_tf = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        norm,
    ])

    train_items, val_items = split_data(args.data, args.classes, args.val_frac)
    train_dl = DataLoader(FingerDataset(train_items, train_tf), batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(FingerDataset(val_items, val_tf), batch_size=args.batch)

    model = FingerNet(args.classes, args.size).to(device)
    print(model)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters()))

    loss_fn = nn.CrossEntropyLoss()  # for choosing a category, not predicting a number
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        # ---- training: forward pass, loss, backpropagation, gradient descent step
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()   # backpropagation: compute gradients of the loss
            opt.step()        # gradient descent: nudge every weight downhill
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
        hist["train_loss"].append(loss_sum / total)
        hist["train_acc"].append(correct / total)

        # ---- validation: no learning, just measure
        model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss_sum += loss_fn(out, y).item() * len(y)
                correct += (out.argmax(1) == y).sum().item()
                total += len(y)
        val_acc = correct / total
        hist["val_loss"].append(loss_sum / total)
        hist["val_acc"].append(val_acc)

        flag = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"state_dict": model.state_dict(), "classes": args.classes,
                        "size": args.size, "names": class_names(args.classes)}, args.out)
            flag = "  <- saved"
        print(f"epoch {epoch:2d}  train loss {hist['train_loss'][-1]:.3f} acc {hist['train_acc'][-1]:.2%}"
              f"  |  val loss {hist['val_loss'][-1]:.3f} acc {val_acc:.2%}{flag}")

    print(f"\nBest validation accuracy: {best_acc:.2%}  (model saved to {args.out})")

    # ---- plots
    ep = range(1, args.epochs + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, hist["train_loss"], label="train")
    ax[0].plot(ep, hist["val_loss"], label="validation")
    ax[0].set(title="Loss vs epoch", xlabel="epoch", ylabel="cross-entropy loss")
    ax[0].legend()
    ax[1].plot(ep, hist["train_acc"], label="train")
    ax[1].plot(ep, hist["val_acc"], label="validation")
    ax[1].set(title="Accuracy vs epoch", xlabel="epoch", ylabel="accuracy")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig("training_curves.png", dpi=150)

    # ---- confusion matrix with the best model
    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    cm = torch.zeros(args.classes, args.classes, dtype=torch.int64)
    with torch.no_grad():
        for x, y in val_dl:
            pred = model(x.to(device)).argmax(1).cpu()
            for t, p in zip(y, pred):
                cm[t, p] += 1
    fig, axc = plt.subplots(figsize=(6, 5))
    axc.imshow(cm, cmap="Blues")
    for i in range(args.classes):
        for j in range(args.classes):
            v = cm[i, j].item()
            axc.text(j, i, v, ha="center", va="center",
                     color="white" if v > cm.max().item() / 2 else "black", fontsize=8)
    names = class_names(args.classes)
    axc.set(xticks=range(args.classes), yticks=range(args.classes),
            xticklabels=names, yticklabels=names,
            xlabel="predicted", ylabel="actual", title="Confusion matrix (validation)")
    fig.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=150)
    print("Saved training_curves.png and confusion_matrix.png")

    for c in range(args.classes):
        row = cm[c].sum().item()
        if row:
            print(f"  {class_names(args.classes)[c]}: {cm[c, c].item() / row:.0%} correct")


if __name__ == "__main__":
    main()
