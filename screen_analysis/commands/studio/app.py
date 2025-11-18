from pathlib import Path
from datetime import timedelta, datetime

import streamlit as st
import subprocess
import sys

try:
    import pandas as pd
    import plotly.express as px
except ModuleNotFoundError as e:
    st.set_page_config(page_title="Screen Analysis Studio", layout="wide")
    st.error(
        f"缺少依赖: {e.name}. 请安装: uv add streamlit plotly pandas 或 pip install streamlit plotly pandas"
    )
    st.stop()


st.set_page_config(page_title="Screen Analysis Studio", layout="wide")

ROOT_DIR = Path(__file__).resolve().parents[3]
SCREENSHOTS_DIR = ROOT_DIR / "screenshots"
REPORTS_DIR = ROOT_DIR / "reports"


def list_available_dates(parent_folder: Path) -> list[str]:
    if not parent_folder.exists():
        return []
    return sorted([p.name for p in parent_folder.iterdir() if p.is_dir()])


def load_timeline(date_str: str) -> pd.DataFrame:
    date_dir = REPORTS_DIR / date_str
    csv_path = date_dir / f"{date_str}_timeline.csv"
    if not csv_path.exists():
        st.warning(f"未找到时间线文件: {csv_path}, 请点击“分析”生成报告")
        return pd.DataFrame(columns=["start", "category", "duration_seconds", "finish"])  
    df = pd.read_csv(csv_path)
    if "start" not in df.columns or "category" not in df.columns or "duration_seconds" not in df.columns:
        return pd.DataFrame(columns=["start", "category", "duration_seconds", "finish"])  
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["duration_seconds"] = pd.to_numeric(df["duration_seconds"], errors="coerce")
    df = df.dropna(subset=["start", "duration_seconds"])
    df["finish"] = df["start"] + df["duration_seconds"].apply(lambda s: timedelta(seconds=float(s)))
    df["category_list"] = df["category"].astype(str).str.strip().str.split(r"\s+")
    df = df.explode("category_list")
    df["category"] = df["category_list"].str.strip()
    df = df.drop(columns=["category_list"])
    df = df[df["category"] != ""]
    return df


def load_last_exec_ts(date_str: str):
    p = REPORTS_DIR / date_str / "executed_at.log"
    if not p.exists():
        return None
    try:
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            return None
        ts = pd.to_datetime(lines[-1], errors="coerce")
        return ts
    except Exception:
        return None


st.title("Screen Analysis Studio")
st.caption("""
    本工具用于可视化屏幕分析报告。请先在选择日期，然后点击 “分析屏幕截图” 进行分析。
""")

screenshots_dates = list_available_dates(SCREENSHOTS_DIR)
reports_dates = list_available_dates(REPORTS_DIR)
dates = list(set(screenshots_dates + reports_dates))
dates.sort()
if not dates:
    st.warning("未找到任何日期子目录，请在 reports/ 下生成报告")
    st.stop()

selected_date = st.sidebar.selectbox("选择数据日期", options=dates, index=len(dates) - 1)

left, right = st.columns(2, width=450)

if left.button("分析屏幕截图", type="secondary", icon="🔄"):
    with st.spinner("正在分析..."):
        try:
            r = subprocess.run(["sa", "analyze", selected_date], capture_output=True, text=True)
        except FileNotFoundError:
            r = subprocess.run([sys.executable, "-m", "screen_analysis.main", "analyze", selected_date], capture_output=True, text=True)
    # st.write("退出码:", r.returncode)
    # if r.stdout:
    #     st.code(r.stdout)
    if r.stderr:
        st.code(r.stderr)

df = load_timeline(selected_date)
if df.empty:
    st.warning("未能加载时间线数据或列缺失")
    st.stop()

fig = px.timeline(
    df,
    x_start="start",
    x_end="finish",
    y="category",
    color="category",
)
fig.update_layout(
    height=400,
    title=f"{selected_date} 屏幕使用情况",
    margin=dict(l=40, r=40, t=40, b=40),
)
fig.update_yaxes(title="category")
fig.update_xaxes(title="时间")

last_ts = load_last_exec_ts(selected_date)
is_complete = False
if last_ts is not None and not pd.isna(last_ts):
    fig.add_vline(x=pd.to_datetime(last_ts), line_width=1, line_dash="dash", line_color="red")
    cutoff = pd.to_datetime(selected_date, format="%Y%m%d") + pd.Timedelta(days=1)
    ts_str = pd.to_datetime(last_ts).strftime("%Y-%m-%d %H:%M:%S")
    if pd.to_datetime(last_ts) == cutoff:
        st.success("该日期已分析完成")
        is_complete = True
    else:
        now = pd.Timestamp.now()
        fig.add_vline(x=pd.Timestamp.now(), line_width=1, line_dash="dash", line_color="gray")
        if now - pd.to_datetime(last_ts) > pd.Timedelta(hours=1):
            st.warning("当前图表已滞后超过 1 小时，请点击“分析屏幕截图”更新")
        st.caption(f"最近分析时间: {ts_str} (红色细线)")

img_dir = SCREENSHOTS_DIR / selected_date
files = list(img_dir.glob("*.png")) if img_dir.exists() else []
if files:
    total_bytes = sum(p.stat().st_size for p in files)
    size_mb = total_bytes / (1024 * 1024)
    label = f"清空图片（{size_mb:.2f} Mb）"
    if is_complete and right.button(label, icon="🧹", type="secondary"):
        try:
            subprocess.run(["sa", "del", "-d", selected_date], capture_output=True, text=True)
        except FileNotFoundError:
            subprocess.run([sys.executable, "-m", "screen_analysis.main", "del", "-d", selected_date], capture_output=True, text=True)
        st.success("已清空该日期的截图")

st.plotly_chart(fig, use_container_width=True)