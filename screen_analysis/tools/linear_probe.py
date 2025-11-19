import argparse
import json
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from screen_analysis.zero_shot import load_model
from screen_analysis.config import load_config, resolve_output_base


def list_items(root: Path) -> List[Tuple[Path, str]]:
    items: List[Tuple[Path, str]] = []
    if not root.exists():
        return items
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        label = sub.name
        for img in sorted(sub.glob("*.png")):
            items.append((img, label))
    return items


def resolve_ckpt(base: Path, override: str) -> Path:
    if override:
        return Path(override)
    cfg = load_config(base)
    return resolve_output_base(base, cfg.get("clip_weights", ""))


def compute_features(
    paths: List[Path], batch_size: int, checkpoint_path: Optional[Path] = None
) -> torch.Tensor:
    model, preprocess, _, device = load_model(
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None
    )
    with torch.no_grad():
        feats: List[torch.Tensor] = []
        if not batch_size or batch_size <= 0:
            batch_size = len(paths)
        with Progress(
            TextColumn("[bold magenta]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("[bold]{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("[bold magenta]Computing features...", total=len(paths))
            for start in range(0, len(paths), batch_size):
                batch_paths = paths[start : start + batch_size]
                images = []
                for p in batch_paths:
                    img = Image.open(p).convert("RGB")
                    images.append(preprocess(img))
                image_input = torch.stack(images).to(device)
                image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                feats.append(image_features.cpu())
                progress.update(task, advance=len(batch_paths))
    return torch.cat(feats, dim=0)


def build_label_index(labels: List[str]) -> Tuple[List[str], Dict[str, int]]:
    uniq = sorted(set(labels))
    idx = {l: i for i, l in enumerate(uniq)}
    return uniq, idx


def train(
    data_root: Path,
    clip_weights: str,
    model_name: str,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    val_split: float,
    seed: Optional[int] = None,
):
    base = Path.cwd()
    out_path = base / "checkpoints" / model_name / "model.pth"
    ckpt = resolve_ckpt(base, clip_weights)
    items = list_items(data_root)
    paths = [p for p, _ in items]
    labels = [l for _, l in items]
    label_list, label_to_idx = build_label_index(labels)
    y = torch.tensor([label_to_idx[l] for l in labels], dtype=torch.long)
    console = Console()
    console.print("[bold]Extracting features...")
    if seed is not None:
        torch.manual_seed(int(seed))
    X = compute_features(paths, batch_size, ckpt)
    label_indices: Dict[int, List[int]] = {}
    for i, li in enumerate(y.tolist()):
        label_indices.setdefault(li, []).append(i)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for li, idxs in label_indices.items():
        k = int(len(idxs) * (1.0 - val_split))
        train_idx.extend(idxs[:k])
        val_idx.extend(idxs[k:])
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    fdim = X.shape[1]
    C = len(label_list)
    clf = nn.Linear(fdim, C)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=weight_decay)
    class_counts = torch.bincount(y_train, minlength=C).float()
    class_weights = (class_counts.sum() / class_counts).clamp(min=1.0)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    best_acc = -1.0
    best_state = None
    history: List[Dict[str, float]] = []
    console.print("[bold]Training linear probe...")
    progress = Progress(
        TextColumn("[bold]Epoch"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    progress.start()
    task = progress.add_task("epochs", total=epochs)
    for ep in range(epochs):
        clf.train()
        train_loss_sum = 0.0
        train_loss_cnt = 0
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = clf(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            train_loss_sum += float(loss.item())
            train_loss_cnt += 1
        clf.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = clf(xb)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == yb).sum().item()
                total += yb.shape[0]
        acc = correct / total if total > 0 else 0.0
        tr_loss = (train_loss_sum / train_loss_cnt) if train_loss_cnt > 0 else 0.0
        history.append({"epoch": ep + 1, "train_loss": tr_loss, "val_acc": acc})
        if acc > best_acc:
            best_acc = acc
            best_state = {
                "state_dict": clf.state_dict(),
                "labels": label_list,
                "feature_dim": fdim,
            }
        progress.advance(task, 1)
    progress.stop()
    console.print("[bold green]Best val accuracy: " + f"{best_acc:.4f}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_path)
    meta = {
        "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "out_path": str(out_path),
        "clip_weights": str(ckpt),
        "data_root": str(data_root),
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "val_split": val_split,
        "labels": label_list,
        "label_counts": {l: labels.count(l) for l in label_list},
        "feature_dim": fdim,
        "best_val_acc": best_acc,
    }
    meta_path = out_path.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    hist_path = out_path.with_suffix(".history.csv")
    with hist_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_acc"])
        w.writeheader()
        for h in history:
            w.writerow(h)
    console.print("[bold]Saved to " + str(out_path))
    console.print("[bold]Meta: " + str(meta_path))
    console.print("[bold]History: " + str(hist_path))


def load_probe(p: str) -> Tuple[nn.Linear, List[str]]:
    data = torch.load(p, map_location="cpu")
    fdim = int(data["feature_dim"]) if "feature_dim" in data else None
    labels = data["labels"]
    clf = nn.Linear(fdim, len(labels))
    clf.load_state_dict(data["state_dict"])
    clf.eval()
    return clf, labels


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", default=str(Path("dataset") / "training"))
    parser.add_argument("--clip-weights", default="")
    parser.add_argument("--model-name", default="linear_probe")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    train(
        Path(args.data_root),
        args.clip_weights,
        args.model_name,
        args.batch_size,
        args.epochs,
        args.lr,
        args.weight_decay,
        args.val_split,
        args.seed,
    )


if __name__ == "__main__":
    main()
