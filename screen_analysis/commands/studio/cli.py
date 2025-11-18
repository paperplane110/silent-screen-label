import subprocess
import sys
from pathlib import Path


def main():
    app = Path(__file__).with_name("app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app)]
    subprocess.run(cmd)