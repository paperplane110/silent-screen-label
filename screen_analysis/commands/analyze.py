from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter

from screen_analysis.config import load_config, resolve_output_base
from screen_analysis.zero_shot import clip_classify, classify_with_probe, load_model

try:
    from rich.console import Console
    from rich.progress import (
        Progress,
        BarColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    _RICH = True
except Exception:
    _RICH = False

TIMELINE_ITEM = dict(
    start=str,
    category=str,
    duration=int,
)
TIMELINE_LIST = list[TIMELINE_ITEM]


def pprint(*args, **kwargs):
    if _RICH:
        console = Console()
        console.print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def list_day_images(out_base: Path, day: str) -> list[Path]:
    base = out_base / "screenshots" / day
    if not base.exists():
        return []
    files = []
    for p in sorted(base.glob("*.png")):
        files.append(p)
    return files


def parse_ts_from_name(name: str, day: str):
    try:
        parts = name.split("_")
        hh, mm, ss = parts[0], parts[1], parts[2]
        return datetime.strptime(day + hh + mm + ss, "%Y%m%d%H%M%S")
    except Exception:
        return None


def get_summary_by_timeline(timeline: TIMELINE_LIST) -> dict[str, int]:
    summary = Counter()
    for m in timeline:
        if m["category"] is not None:
            summary[m["category"]] += m["duration"]
    return dict(summary)


def read_timeline_csv2timeline_list(timeline_path: Path) -> TIMELINE_LIST:
    """Parse existing timeline csv to a list of `TIMELINE_ITEM`"""
    existing: TIMELINE_LIST = []
    with timeline_path.open("r", encoding="utf-8") as f:
        _header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split(",")
            start = parts[0]
            category = parts[1] if len(parts) > 1 else None
            duration = int(parts[2]) if len(parts) > 2 else 60
            existing.append(
                {
                    "start": start,
                    "category": None if category == "None" else category,
                    "duration": duration,
                }
            )
    return existing


def analyze_day(
    out_base: Path,
    day: str,
    cycle: int,
    clip_prompts: list[dict],
    debug: bool = False,
    use_clip_weights: str = "",
    batch_size: int = 32,
):
    files = list_day_images(out_base, day)
    prompts = [item["prompt"] for item in clip_prompts]
    ckpt = use_clip_weights if use_clip_weights else None
    results = safe_run_inference(files, prompts, ckpt, batch_size, debug)
    merged = analyze_predictions(files, results, clip_prompts, day)
    return merged


def _run_inference(
    files: list[Path],
    prompts: list[str],
    ckpt: str | None,
    batch_size: int,
    debug: bool,
):
    pprint("Loading model...")
    if ckpt:
        load_model(checkpoint_path=str(ckpt))
    else:
        load_model()
    pprint("Model loaded.")
    results = []
    cfg = load_config(Path.cwd())
    probe_path = Path(cfg.get("linear_probe"))
    if _RICH and len(files) > 0:
        with Progress(
            TextColumn("Inference"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("inference", total=len(files))
            bs = batch_size if batch_size and batch_size > 0 else 32
            for start in range(0, len(files), bs):
                batch_paths = files[start : start + bs]
                if probe_path:
                    batch_results = classify_with_probe(
                        batch_paths,
                        probe_path,
                        checkpoint_path=str(ckpt) if ckpt else None,
                        batch_size=bs,
                    )
                else:
                    batch_results = clip_classify(
                        batch_paths,
                        prompts,
                        checkpoint_path=str(ckpt) if ckpt else None,
                        batch_size=bs,
                    )
                results.extend(batch_results)
                progress.advance(task, len(batch_paths))
    else:
        if probe_path:
            results = classify_with_probe(
                files,
                probe_path,
                checkpoint_path=str(ckpt) if ckpt else None,
                batch_size=batch_size,
            )
        else:
            results = clip_classify(
                files,
                prompts,
                checkpoint_path=str(ckpt) if ckpt else None,
                batch_size=batch_size,
            )
    if debug:
        for f, r in zip(files, results):
            pprint(f"[DEBUG] {f.name} -> {r[0]} ({round(r[1],3)})")
    return results


def safe_run_inference(
    files: list[Path],
    prompts: list[str],
    ckpt: str | None,
    batch_size: int,
    debug: bool,
):
    try:
        return _run_inference(files, prompts, ckpt, batch_size, debug)
    except Exception as e:
        if debug:
            pprint("[DEBUG] CLIP failed, exiting...")
            pprint(f"[DEBUG] CLIP weights: {ckpt}")
            pprint(f"[DEBUG] CLIP failed reason: {e}")
        exit(1)


def analyze_predictions(
    files: list[Path],
    results: list[tuple[str, float]],
    clip_prompts: list[dict],
    day: str,
):
    map_prompt_to_label = {item["prompt"]: item["label"] for item in clip_prompts}
    prompt_set = set(map_prompt_to_label.keys())
    preds: list[str] = []
    for r in results:
        k = r[0]
        if k in prompt_set:
            preds.append(map_prompt_to_label.get(k, "unknown"))
        else:
            preds.append(k)
    pprint("Analyzing results...")

    pairs: list[tuple[datetime, str]] = []
    for f, pred in zip(files, preds):
        ts = parse_ts_from_name(f.stem, day)
        if ts is None:
            continue
        pairs.append((ts, pred))

    minute_categories: dict[int, str] = {}
    for ts, pred in pairs:
        idx = ts.hour * 60 + ts.minute
        if idx not in minute_categories:
            minute_categories[idx] = pred
        else:
            minute_categories[idx] += f" {pred}"

    base_day = datetime.strptime(day, "%Y%m%d")
    timeline: TIMELINE_LIST = []
    for i in range(1440):
        start_ts = (base_day + timedelta(minutes=i)).isoformat()
        if i in minute_categories:
            cat = minute_categories[i]
        else:
            cat = None
        timeline.append({"start": start_ts, "category": cat, "duration": 60})

    pprint("Analysis done.")
    return timeline


###############################
# Writing functions
###############################


def write_timeline(out_base: Path, day: str, merged):
    report_dir = out_base / "reports" / day
    report_dir.mkdir(parents=True, exist_ok=True)
    timeline_path = report_dir / f"{day}_timeline.csv"
    pprint("Writing timeline...")
    with timeline_path.open("w", encoding="utf-8") as f:
        f.write("start,category,duration_seconds\n")
        for m in merged:
            f.write(f"{m['start']},{m['category']},{m['duration']}\n")
    pprint("Timeline written.")


def write_summary(out_base: Path, day: str, summary):
    report_dir = out_base / "reports" / day
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / f"{day}_summary.csv"
    pprint("Writing summary...")
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("category,duration_seconds,duration_minutes\n")
        for k, v in summary.items():
            f.write(f"{k},{v},{round(v/60,2)}\n")
    pprint("Summary written.")


def append_execution_ts(out_base: Path, day: str):
    now = datetime.now()
    base_day = datetime.strptime(day, "%Y%m%d")
    cutoff = base_day + timedelta(days=1)
    effective = now if now <= cutoff else cutoff
    exec_dir = out_base / "reports" / day
    exec_dir.mkdir(parents=True, exist_ok=True)
    exec_file = exec_dir / "executed_at.log"
    with exec_file.open("a", encoding="utf-8") as f:
        f.write(effective.isoformat() + "\n")
    pprint("Execution timestamp written.")


def incremental_analyze(
    new_files: list[Path],
    old_timeline: TIMELINE_LIST,
    day: str,
    clip_prompts: list[dict],
    use_clip_weights: str,
    batch_size: int,
    debug: bool,
):
    """如果有新的图片，则增量分析新的图片"""
    prompts = [item["prompt"] for item in clip_prompts]
    ckpt = use_clip_weights if use_clip_weights else None
    results = safe_run_inference(new_files, prompts, ckpt, batch_size, debug)
    map_prompt_to_label = {item["prompt"]: item["label"] for item in clip_prompts}
    prompt_set = set(map_prompt_to_label.keys())
    preds: list[str] = []
    for r in results:
        k = r[0]
        if k in prompt_set:
            preds.append(map_prompt_to_label.get(k, "unknown"))
        else:
            preds.append(k)

    pairs: list[tuple[datetime, str]] = []
    for fpath, pred in zip(new_files, preds):
        ts = parse_ts_from_name(fpath.stem, day)
        if ts is None:
            continue
        pairs.append((ts, pred))

    # 记录每分钟的分类，每个分钟内的分类用空格分隔
    minute_categories: dict[int, str] = {}
    for ts, pred in pairs:
        idx = ts.hour * 60 + ts.minute
        if idx not in minute_categories:
            minute_categories[idx] = pred
        else:
            minute_categories[idx] += f" {pred}"

    # 更新已存在分类
    for idx, cat in minute_categories.items():
        if 0 <= idx < len(old_timeline):
            old_timeline[idx]["category"] = cat

    return old_timeline


def analyze_cmd(args):
    base = Path.cwd()
    cfg = load_config(base)
    out_base = resolve_output_base(base, cfg["dir"])
    cycle = int(cfg["cycle"])
    batch_size = int(cfg.get("batch_size", 32))
    use_clip_weights = (
        args.clip_weights
        if args.clip_weights
        else str(resolve_output_base(base, cfg["clip_weights"]))
    )
    day = (
        args.yyyymmdd
        if getattr(args, "yyyymmdd", None)
        else datetime.now().strftime("%Y%m%d")
    )
    clip_prompts = cfg["clip_prompts"]

    # Paths
    report_dir = out_base / "reports" / day
    timeline_path = report_dir / f"{day}_timeline.csv"
    exec_path = report_dir / "executed_at.log"

    # get last execution time
    last_exec = None
    if (
        not args.overwrite
        and report_dir.exists()
        and exec_path.exists()
        and timeline_path.exists()
    ):
        with exec_path.open("r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            if lines:
                try:
                    last_exec = datetime.fromisoformat(lines[-1])
                except Exception:
                    last_exec = None

    if last_exec is not None and timeline_path.exists():
        """如果有上次执行记录，且存在timeline文件，则只分析新的图片"""
        # 列出所有图片，筛选出新的图片
        files = list_day_images(out_base, day)
        new_files: list[Path] = []
        for p in files:
            ts = parse_ts_from_name(p.stem, day)
            if ts is not None and ts > last_exec:
                new_files.append(p)

        # 无新图片，直接返回
        if len(new_files) == 0:
            pprint(f"No new images to analyze for {day}")
            return

        existing_timeline = read_timeline_csv2timeline_list(timeline_path)
        timeline = incremental_analyze(
            new_files,
            existing_timeline,
            day,
            clip_prompts,
            use_clip_weights,
            batch_size,
            getattr(args, "debug", False),
        )

    else:
        """如果没有上次执行记录，或不存在timeline文件，则分析所有图片"""
        timeline = analyze_day(
            out_base,
            day,
            cycle,
            clip_prompts,
            getattr(args, "debug", False),
            use_clip_weights,
            batch_size,
        )

    write_timeline(out_base, day, timeline)

    summary = get_summary_by_timeline(timeline)
    write_summary(out_base, day, summary)

    pprint(f"Wrote reports to {out_base / 'reports' / day}")
    append_execution_ts(out_base, day)
