import argparse
from screen_analysis.commands.run import run_cmd
from screen_analysis.commands.delete import del_cmd
from screen_analysis.commands.analyze import analyze_cmd

def main():
    parser = argparse.ArgumentParser(prog="screen-analysis")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--cycle", "-c", type=int, help="Cycle time in seconds")
    p_run.add_argument("--dir", "-d", help="The directory to store screenshots")
    p_run.add_argument("--menubar", action="store_true", help="Show a macOS menu bar indicator")
    p_run.set_defaults(func=run_cmd)
    p_del = sub.add_parser("del")
    p_del.add_argument("--date", "-d", help="Date in YYYYMMDD format")
    p_del.set_defaults(func=del_cmd)
    p_an = sub.add_parser("analyze")
    p_an.add_argument("yyyymmdd", nargs="?", help="Date in YYYYMMDD format")
    p_an.add_argument("--debug", action="store_true", help="Print per-image classification")
    p_an.add_argument("--clip-weights", help="Local path to CLIP weights file")
    p_an.add_argument("--overwrite", action="store_true", help="Overwrite existing analysis")
    p_an.set_defaults(func=analyze_cmd)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
