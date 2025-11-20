import time
import mss
from pathlib import Path

from rich.console import Console

from screen_analysis.config import load_config, save_config, resolve_output_base

console = Console()

def capture_once(out_base: Path) -> Path:
    day = time.strftime("%Y%m%d")
    target = out_base / "screenshots" / day
    target.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%H_%M_%S")
    with mss.mss() as sct:
        for idx, mon in enumerate(sct.monitors[1:], start=1):
            shot = sct.grab(mon)
            name = f"{ts}_{idx}.png"
            mss.tools.to_png(shot.rgb, shot.size, output=str(target / name))
            console.print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Snapshot {target / name}")
    return target / name

def run_cmd(args):
    base = Path.cwd()
    cfg = load_config(base)
    if args.cycle is not None:
        cfg["cycle"] = int(args.cycle)
    if args.dir is not None:
        cfg["dir"] = str(args.dir)
    save_config(base, cfg)
    out_base = resolve_output_base(base, cfg["dir"])
    cycle = int(cfg["cycle"])
    if getattr(args, "menubar", False):
        from screen_analysis.menubar import run_menubar
        run_menubar(out_base, cycle, capture_once)
    else:
        while True:
            img_path = capture_once(out_base)
            time.sleep(cycle)