from pathlib import Path
from typing import Optional, List, Tuple, Dict, TypedDict
import argparse
import json
import csv

import torch
from rich.console import Console
from rich.table import Table
from rich import box

from screen_analysis.zero_shot import (
    load_model,
    _get_text_features,
    clip_classify_labels
)
from screen_analysis.tools.linear_probe import (
    list_items,
    resolve_ckpt,
    compute_features,
    load_probe,
)
from screen_analysis.config import load_config

console = Console()

class LabelMetric(TypedDict):
    support: int
    tp: int
    fp: int
    tn: int
    fn: int
    accuracy: float
    recall: float
    precision: float
    f1: float


def write_report(
    report_dir: Path,
    ckpt: Path,
    data_root: Path,
    overall_acc: float,
    per_label: Dict[str, LabelMetric],
    badcases: List[Tuple[Path, str, str, float]],
    probe_path: Path,
):
    base_dir = report_dir / probe_path.parent.name
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
        w.writerow(
            [
                "label",
                "support",
                "tp",
                "fp",
                "tn",
                "fn",
                "accuracy",
                "recall",
                "precision",
                "f1",
            ]
        )
        for l, m in sorted(per_label.items(), key=lambda x: x[0]):
            w.writerow(
                [
                    l,
                    m["support"],
                    m["tp"],
                    m["fp"],
                    m["tn"],
                    m["fn"],
                    f"{m['accuracy']:.4f}",
                    f"{m['recall']:.4f}",
                    f"{m['precision']:.4f}",
                    f"{m['f1']:.4f}",
                ]
            )
    with (base_dir / "badcases.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "gt", "pred", "conf"])
        for p, gt, pd, conf in badcases:
            w.writerow([str(p), gt, pd, f"{conf:.4f}"])


def print_results(
    title: str,
    overall_acc: float,
    per_label: Dict[str, LabelMetric],
    badcases: List[Tuple[Path, str, str, float]],
):
    console.rule(f"[bold]{title} Evaluation")
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


def evaluate_clip(
    data_root_path: Path,
    clip_weights: str,
    prompt_items: List[Dict[str, str]],
    batch_size: int,
    clip_agg: str,
    threshold: Optional[float],
):
    """Evaluate CLIP model on the given dataset.

    Args:
        data_root_path: Path to the dataset root directory.
        clip_weights: Path to the CLIP model weights.
        batch_size: Batch size for feature computation.
        clip_agg: Aggregation method for CLIP features ("mean" or "max").
        threshold: Confidence threshold for classification.

    Returns:
        ckpt: Path to the resolved CLIP checkpoint.
        overall_acc: Overall accuracy of the model.
        per_label: Dictionary of per-label metrics.
        badcases: List of bad cases (path, gt, pred, conf).
    """
    base = Path.cwd()
    ckpt = resolve_ckpt(base, clip_weights)
    items = list_items(data_root_path)
    paths = [p for p, _ in items]
    gts = [l for _, l in items]
    results = clip_classify_labels(
        paths,
        prompt_items,
        checkpoint_path=str(ckpt),
        batch_size=batch_size,
        agg=clip_agg,
        threshold=threshold,
    )
    preds = [pd for pd, _ in results]
    confs = [cf for _, cf in results]
    label_set = sorted(set(gts))
    cm = [[0 for _ in label_set] for _ in label_set]
    for gt, pd in zip(gts, preds):
        cm[label_set.index(gt)][label_set.index(pd)] += 1
    per_label: Dict[str, LabelMetric] = {}
    for l in label_set:
        li = label_set.index(l)
        tp = cm[li][li]
        fn = sum(cm[li]) - tp
        fp = sum(row[li] for row in cm) - tp
        tn = sum(sum(r) for r in cm) - tp - fn - fp
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0.0
        support = sum(cm[li])
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
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
    overall_acc = (
        sum(1 for gt, pd in zip(gts, preds) if gt == pd) / len(gts) if gts else 0.0
    )
    badcases: List[Tuple[Path, str, str, float]] = []
    for (p, gt), pd, cf in zip(items, preds, confs):
        if pd != gt:
            badcases.append((p, gt, pd, cf))

    return ckpt, overall_acc, per_label, badcases


def evaluate_linear_probe(
    data_root: Path,
    clip_weights: str,
    probe_path: str,
    batch_size: int,
):
    base = Path.cwd()
    ckpt = resolve_ckpt(base, clip_weights)
    items = list_items(data_root)
    paths = [p for p, _ in items]
    gts = [l for _, l in items]
    clf, labels = load_probe(probe_path)
    X = compute_features(paths, batch_size, ckpt)

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
    per_label: Dict[str, LabelMetric] = {}
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
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
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
    overall_acc = (
        sum(1 for gt, pd in zip(gts, preds) if gt == pd) / len(gts) if gts else 0.0
    )
    badcases: List[Tuple[Path, str, str, float]] = []
    for (p, gt), pd, cf in zip(items, preds, confs):
        if pd != gt:
            badcases.append((p, gt, pd, cf))

    return ckpt, overall_acc, per_label, badcases


def main():
    cfg = load_config(Path.cwd())
    default_clip_weights = cfg.get("clip_weights", "")
    default_clip_agg = cfg.get("clip_agg", "mean")
    default_clip_threshold = cfg.get("threshold", 0.3)
    default_probe = cfg.get("linear_probe", "")

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_clip = sub.add_parser("clip")
    p_clip.add_argument("--data-root", default=str(Path("dataset") / "eval"))
    p_clip.add_argument("--batch-size", type=int, default=32)
    p_clip.add_argument("--clip-weights", default=default_clip_weights)
    p_clip.add_argument("--clip-agg", default=default_clip_agg)
    p_clip.add_argument("--threshold", type=float, default=default_clip_threshold)

    p_linear_probe = sub.add_parser("lp")
    p_linear_probe.add_argument("--data-root", default=str(Path("dataset") / "eval"))
    p_linear_probe.add_argument("--batch-size", type=int, default=32)
    p_linear_probe.add_argument("--clip-weights", default=default_clip_weights)
    p_linear_probe.add_argument("--probe", default=default_probe)

    args = parser.parse_args()

    console.print(f"Data root: [bold]{args.data_root}")
    console.print(f"Batch size: [bold]{args.batch_size}")

    if args.cmd == "clip":
        console.print("[bold]CLIP Evaluation")
        console.print(f"Clip weights: [bold]{args.clip_weights}")
        console.print(f"Clip agg: [bold]{args.clip_agg}")
        console.print(f"Threshold: [bold]{args.threshold}")
        report_dir_path = Path("test") / "clip" / Path(args.clip_weights).parent.name

    else:
        console.print("[bold]Linear Probe Evaluation")
        console.print(f"Probe: [bold]{args.probe}")
        report_dir_path = Path("test") / "linear_probe" / Path(args.probe).parent.name

    console.print(f"Report dir: [bold]{report_dir_path}")

    data_root_path = Path(args.data_root)
    probe_path = Path(args.probe) if args.cmd == "lp" else Path("")
    report_dir_path.mkdir(parents=True, exist_ok=True)

    if args.cmd == "clip":
        # clip evaluation
        ckpt, overall_acc, per_label, badcases = evaluate_clip(
            data_root_path,
            args.clip_weights,
            cfg.get("clip_prompts", []),
            args.batch_size,
            args.clip_agg,
            args.threshold,
        )
    else:
        # linear probe evaluation
        ckpt, overall_acc, per_label, badcases = evaluate_linear_probe(
            data_root_path,
            args.clip_weights,
            str(probe_path),
            args.batch_size
        )

    write_report(
        report_dir_path,
        ckpt,
        data_root_path,
        overall_acc,
        per_label,
        badcases,
        probe_path,
    )

    print_results(
        "CLIP" if args.cmd == "clip" else "Linear Probe",
        overall_acc,
        per_label,
        badcases
    )


if __name__ == "__main__":
    main()
