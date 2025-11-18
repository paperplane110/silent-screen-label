from pathlib import Path
from datetime import datetime

from screen_analysis.config import load_config, resolve_output_base

def del_cmd(args):
    base = Path.cwd()
    cfg = load_config(base)
    out_base = resolve_output_base(base, cfg["dir"])
    date = args.date
    if date is None:
        date = datetime.now().strftime("%Y%m%d")
    target = out_base / "screenshots" / date
    if not target.exists():
        print(f"Not found {target}")
        return
    files = list(target.rglob("*.png"))
    if not files:
        print(f"No screenshots found in {target}")
        return
    for p in files:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
    print(f"Deleted {len(files)} screenshots in {target}")