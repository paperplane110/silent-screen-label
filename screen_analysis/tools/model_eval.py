from pathlib import Path
from typing import Optional, List, Tuple, Dict
import argparse
import json
import csv

from rich.console import Console
from rich.table import Table
from rich import box

from screen_analysis.tools.linear_probe import (
    list_items,
    resolve_ckpt,
    compute_features,
    load_probe,
)
from screen_analysis.config import load_config

console = Console()

def evaluate(
    data_root: Path,
    clip_weights: str,
    probe_path: Path,
    batch_size: int,
    report_dir: Optional[Path] = None,
):
    base = Path.cwd()
    ckpt = resolve_ckpt(base, clip_weights)
    items = list_items(data_root)
    paths = [p for p, _ in items]
    gts = [l for _, l in items]
    clf, labels = load_probe(probe_path)
    X = compute_features(paths, batch_size, ckpt)
    import torch
    with torch.no_grad():
        logits = clf(X)
        probs = torch.softmax(logits, dim=1)
        preds_idx = torch.argmax(probs, dim=1).tolist()
        preds = [labels[i] for i in preds_idx]
        confs = probs.max(dim=1).values.tolist()
    label_set = sorted(set(labels + gts))
    label_to_idx = {l: i for i, l in enumerate(label_set)}
    cm = [[0 for _ in label_set] for _ in label_set]
    for gt, pd in zip(gts, preds):
        cm[label_to_idx[gt]][label_to_idx[pd]] += 1
    per_label: Dict[str, Dict[str, float]] = {}
    for l in label_set:
        li = label_to_idx[l]
        tp = cm[li][li]
        fn = sum(cm[li]) - tp
        fp = sum(row[li] for row in cm) - tp
        tn = sum(sum(r) for r in cm) - tp - fn - fp
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0.0
        support = sum(cm[li])
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
        per_label[l] = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "f1": f1,
        }
    overall_acc = sum(1 for gt, pd in zip(gts, preds) if gt == pd) / len(gts) if gts else 0.0
    badcases: List[Tuple[Path, str, str, float]] = []
    for (p, gt), pd, cf in zip(items, preds, confs):
        if pd != gt:
            badcases.append((p, gt, pd, cf))
    if report_dir:
        base_dir = Path(report_dir) / probe_path.parent.name
        base_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "probe": str(probe_path),
            "clip_weights": str(ckpt),
            "data_root": str(data_root),
            "overall_acc": overall_acc,
        }
        with (base_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with (base_dir / "per_label.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["label", "support", "tp", "fp", "tn", "fn", "accuracy", "recall", "precision", "f1"])
            for l, m in sorted(per_label.items(), key=lambda x: x[0]):
                w.writerow([l, m["support"], m["tp"], m["fp"], m["tn"], m["fn"], f"{m['accuracy']:.4f}", f"{m['recall']:.4f}", f"{m['precision']:.4f}", f"{m['f1']:.4f}"])
        with (base_dir / "badcases.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "gt", "pred", "conf"])
            for p, gt, pd, conf in badcases:
                w.writerow([str(p), gt, pd, f"{conf:.4f}"])
    
    console.rule("[bold]Linear Probe Evaluation")
    console.print("Overall accuracy: [bold green]" + f"{overall_acc:.4f}")
    table = Table(title="Per-label metrics", box=box.SIMPLE_HEAVY)
    table.add_column("label", style="bold")
    table.add_column("support", justify="right")
    table.add_column("tp/fp/tn/fn", justify="right")
    table.add_column("accuracy", justify="right")
    table.add_column("recall", justify="right")
    table.add_column("precision", justify="right")
    table.add_column("f1", justify="right")
    for l, m in sorted(per_label.items(), key=lambda x: x[0]):
        table.add_row(
            l,
            str(m["support"]),
            f"{m['tp']}/{m['fp']}/{m['tn']}/{m['fn']}",
            f"{m['accuracy']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['precision']:.4f}",
            f"{m['f1']:.4f}",
        )
    console.print(table)
    if badcases:
        bad = Table(title="Badcases", box=box.SIMPLE_HEAVY)
        bad.add_column("path")
        bad.add_column("gt", style="bold")
        bad.add_column("pred", style="bold")
        bad.add_column("conf", justify="right")
        for p, gt, pd, conf in badcases:
            bad.add_row(str(p), gt, pd, f"{conf:.4f}")
        console.print(bad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(Path("dataset") / "eval"))
    parser.add_argument("--clip-weights", default="")
    cfg = load_config(Path.cwd())
    default_probe = cfg.get("linear_probe", "")
    parser.add_argument("--probe", default=default_probe)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--report-dir", default=str(Path("test") / "linear_probe"))
    args = parser.parse_args()
    
    console.print(f"Data root: [bold]{args.data_root}")
    console.print(f"Probe: [bold]{args.probe}")
    console.print(f"Batch size: [bold]{args.batch_size}")
    console.print(f"Report dir: [bold]{args.report_dir}")
    evaluate(Path(args.data_root), args.clip_weights, Path(args.probe), args.batch_size, Path(args.report_dir))


if __name__ == "__main__":
    main()