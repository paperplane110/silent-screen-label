from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter
from typing import TypedDict, Tuple
from functools import partial

from screen_analysis.config import load_config, resolve_output_base
from screen_analysis.zero_shot import classify_with_probe, clip_classify_labels, load_model

from rich.console import Console

console = Console()
pprint = console.print

class TIMELINE_ITEM(TypedDict):
    start: str
    category: str | None
    duration: int

TIMELINE_LIST = list[TIMELINE_ITEM]


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

###############################
# Data process functions
###############################

def incremental_analyze(
    results: list[Tuple[str, float]],
    new_files: list[Path],
    old_timeline: TIMELINE_LIST,
    day: str,
    clip_prompts: list[dict],
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


def analyze_cmd(args):
    base = Path.cwd()

    cfg = load_config(base)
    out_base = resolve_output_base(base, cfg["dir"])
    cycle = int(cfg["cycle"])
    batch_size = int(cfg.get("batch_size", 32))
    clip_weights = str(resolve_output_base(base, cfg["clip_weights"]))
    clip_prompts = cfg["clip_prompts"]
    clip_agg = cfg["clip_agg"]
    threshold = cfg["threshold"]
    probe_path = (
        resolve_output_base(base, cfg["linear_probe"]) 
        if "linear_probe" in cfg 
        else None
    )
    use_linear_probe = cfg["use_linear_probe"]

    # Decide how to classify
    if use_linear_probe and probe_path:
        inference = partial(
            classify_with_probe,
            probe_path=probe_path,
            checkpoint_path=clip_weights,
            batch_size=batch_size,
        )
    else:
        inference = partial(
            clip_classify_labels,
            prompt_items=clip_prompts,
            checkpoint_path=clip_weights,
            batch_size=batch_size,
            agg=clip_agg,
            threshold=threshold
        )

    day = (
        args.yyyymmdd
        if getattr(args, "yyyymmdd", None)
        else datetime.now().strftime("%Y%m%d")
    )

    debug = getattr(args, "debug", False)

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

        pprint("Loading model...")
        load_model(checkpoint_path=clip_weights)
        pprint("Model loaded.")

        results = inference(new_files)
        if debug:
            for f, r in zip(new_files, results):
                pprint(f"[DEBUG] {f.name} -> {r[0]} ({round(r[1],3)})")

        timeline = incremental_analyze(
            results,
            new_files,
            existing_timeline,
            day,
            clip_prompts,
        )

    else:
        """如果没有上次执行记录，或不存在timeline文件，则分析所有图片"""
        files = list_day_images(out_base, day)

        pprint("Loading model...")
        load_model(checkpoint_path=clip_weights)
        pprint("Model loaded.")
        
        results = inference(files)
        if debug:
            for f, r in zip(files, results):
                pprint(f"[DEBUG] {f.name} -> {r[0]} ({round(r[1],3)})")

        timeline = analyze_predictions(files, results, clip_prompts, day)
         

    write_timeline(out_base, day, timeline)

    summary = get_summary_by_timeline(timeline)
    write_summary(out_base, day, summary)

    pprint(f"Wrote reports to {out_base / 'reports' / day}")
    append_execution_ts(out_base, day)
