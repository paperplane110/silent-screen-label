import argparse
import subprocess
import time
import csv
import os
import signal
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel

def ps_stats(pid):
    try:
        out = subprocess.check_output(["ps", "-p", str(pid), "-o", "pcpu= pmem= rss= vsize="])\
            .decode("utf-8").strip()
        if not out:
            return None
        parts = out.split()
        if len(parts) < 4:
            return None
        cpu = float(parts[0])
        memp = float(parts[1])
        rss_kb = int(parts[2])
        vsize_kb = int(parts[3])
        return {"cpu": cpu, "mem_percent": memp, "rss_kb": rss_kb, "vsize_kb": vsize_kb}
    except subprocess.CalledProcessError:
        return None

def run_and_monitor(cmd, cwd, interval, duration, output):
    start_time = datetime.now().isoformat()
    with open(output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "cpu_percent", "mem_percent", "rss_kb", "vsize_kb", "rss_mib", "vsize_mib"]) 
        p = subprocess.Popen(cmd, cwd=cwd, shell=True, preexec_fn=os.setsid)
        pid = p.pid
        peak_cpu = 0.0
        peak_mem_percent = 0.0
        peak_rss_kb = 0
        peak_vsize_kb = 0
        sample_count = 0
        sum_cpu = 0.0
        sum_rss_kb = 0.0
        start = time.time()
        console = Console()
        console.print(f"[bold magenta]开始监控进程 {pid}[/bold magenta]")
        try:
            with Live(refresh_per_second=max(1, int(1/interval)) if interval > 0 else 4, console=console) as live:
                while True:
                    stats = ps_stats(pid)
                    ts = datetime.now().isoformat()
                    if stats is None:
                        break
                    w.writerow([ts, stats["cpu"], stats["mem_percent"], stats["rss_kb"], stats["vsize_kb"], stats["rss_kb"]/1024.0, stats["vsize_kb"]/1024.0])
                    sample_count += 1
                    sum_cpu += stats["cpu"]
                    sum_rss_kb += stats["rss_kb"]
                    if stats["cpu"] > peak_cpu:
                        peak_cpu = stats["cpu"]
                    if stats["mem_percent"] > peak_mem_percent:
                        peak_mem_percent = stats["mem_percent"]
                    if stats["rss_kb"] > peak_rss_kb:
                        peak_rss_kb = stats["rss_kb"]
                    if stats["vsize_kb"] > peak_vsize_kb:
                        peak_vsize_kb = stats["vsize_kb"]
                    elapsed = time.time() - start
                    table = Table(title="实时性能监控")
                    table.add_column("PID", justify="right")
                    table.add_column("CPU%", justify="right")
                    table.add_column("MEM MiB", justify="right")
                    table.add_column("VSZ MiB", justify="right")
                    table.add_column("Elapsed s", justify="right")
                    table.add_row(
                        str(pid),
                        f"{stats['cpu']:.1f}",
                        f"{stats['rss_kb']/1024:.1f}",
                        f"{stats['vsize_kb']/1024:.1f}",
                        f"{elapsed:.1f}",
                    )
                    live.update(table)
                    time.sleep(interval)
                    if duration is not None and (time.time() - start) >= duration:
                        try:
                            os.killpg(os.getpgid(pid), signal.SIGTERM)
                        except Exception:
                            pass
                        break
            p.poll()
        except KeyboardInterrupt:
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except Exception:
                pass
        finally:
            try:
                w.writerow(["SUMMARY", peak_cpu, peak_mem_percent, peak_rss_kb, "-", peak_rss_kb/1024.0, peak_vsize_kb/1024.0])
            except Exception:
                pass
    avg_cpu = (sum_cpu / sample_count) if sample_count else 0.0
    avg_rss_mib = (sum_rss_kb / sample_count / 1024.0) if sample_count else 0.0
    return {
        "start": start_time,
        "pid": pid,
        "peak_cpu": peak_cpu,
        "peak_mem_percent": peak_mem_percent,
        "peak_rss_kb": peak_rss_kb,
        "peak_vsize_kb": peak_vsize_kb,
        "avg_cpu": avg_cpu,
        "avg_rss_mib": avg_rss_mib,
        "samples": sample_count,
        "duration": duration,
    }

def main():
    parser = argparse.ArgumentParser(
        description="监控命令的 CPU、内存使用并输出 CSV",
        epilog="单位说明: interval/duration 为秒; RSS/VSZ 在 CSV 为 KiB, 实时和摘要为 MiB",
    )
    parser.add_argument("--cmd", default="python screen_analysis/main.py run > /dev/null 2>&1", help="要监控的命令字符串")
    parser.add_argument("--interval", type=float, default=1.0, help="采样间隔，单位: 秒")
    parser.add_argument("--duration", type=float, default=None, help="监控时长，单位: 秒；为空表示直到进程结束")
    args = parser.parse_args()
    
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    output_path = os.path.join(repo_root, "test", "performance", "performance_log.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    console = Console()
    console.print(f"[bold magenta]开始监控命令: {args.cmd}[/bold magenta]")
    result = run_and_monitor(args.cmd, repo_root, args.interval, args.duration, output_path)

    # Print summary
    summary = Table(title="监控统计结果")
    summary.add_column("项")
    summary.add_column("值", justify="right")
    summary.add_row("PID", str(result["pid"]))
    summary.add_row("样本数", str(result["samples"]))
    summary.add_row("峰值 CPU%", f"{result['peak_cpu']:.1f}")
    summary.add_row("平均 CPU%", f"{result['avg_cpu']:.2f}")
    summary.add_row("峰值 MEM MiB", f"{result['peak_rss_kb']/1024:.1f}")
    summary.add_row("平均 MEM MiB", f"{result['avg_rss_mib']:.1f}")
    summary.add_row("峰值 RSS MiB", f"{result['peak_rss_kb']/1024:.1f}")
    summary.add_row("峰值 VSZ MiB", f"{result['peak_vsize_kb']/1024:.1f}")
    console.print(Panel(summary, title="性能监控摘要", expand=False))
    console.print(f"CSV: {output_path}")

if __name__ == "__main__":
    main()