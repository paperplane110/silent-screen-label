from pathlib import Path
import json

def config_path(base: Path) -> Path:
    return base / "config.json"

def load_config(base: Path) -> dict:
    p = config_path(base)
    defaults = {"cycle": 600, "dir": str(base)}
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "cycle" not in data or "dir" not in data:
            data = {**defaults, **data}
        if "label2prompts" in data:
            cps = []
            for lbl, arr in data.get("label2prompts", {}).items():
                for prm in arr or []:
                    cps.append({"prompt": prm, "label": lbl})
            data["clip_prompts"] = cps
    else:
        data = defaults
        with p.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return data

def save_config(base: Path, data: dict) -> None:
    p = config_path(base)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def resolve_output_base(base: Path, dir_value: str) -> Path:
    d = Path(dir_value)
    if not d.is_absolute():
        d = base / d
    return d.resolve()