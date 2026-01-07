#!/usr/bin/env python3
import sys
import os
import argparse
import time
from pathlib import Path
from datetime import datetime
import subprocess
import shutil

# Try to import psutil, handle if missing
try:
    import psutil
except ImportError:
    psutil = None

def get_gpu_stats():
    """Returns a string describing GPU usage using nvidia-smi."""
    try:
        # Query nvidia-smi for utilization and memory
        # index, utilization.gpu, memory.total, memory.used
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.total,memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return "GPU: N/A (nvidia-smi failed)"
        
        lines = result.stdout.strip().split('\n')
        stats = []
        for line in lines:
            parts = [x.strip() for x in line.split(',')]
            if len(parts) >= 4:
                idx, util, total, used = parts
                used_gb = float(used) / 1024
                total_gb = float(total) / 1024
                stats.append(f"GPU {idx}: {util}% | Mem: {used_gb:.1f}/{total_gb:.1f} GB")
        return " | ".join(stats)
    except FileNotFoundError:
        return "GPU: N/A (nvidia-smi not found)"
    except Exception as e:
        return f"GPU: Error ({str(e)})"

def get_system_stats():
    """Returns CPU and RAM usage."""
    cpu_percent = "N/A"
    ram_info = "N/A"
    
    if psutil:
        cpu_percent = f"{psutil.cpu_percent()}%"
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024**3)
        total_gb = mem.total / (1024**3)
        ram_info = f"{used_gb:.1f}/{total_gb:.1f} GB ({mem.percent}%)"
    
    return f"CPU: {cpu_percent} | RAM: {ram_info}"

def find_latest_run(results_dir: Path) -> Path | None:
    """Finds the most recently modified subdirectory in results_dir."""
    if not results_dir.exists():
        return None
    
    subdirs = [d for d in results_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return None
        
    # Sort by modification time (descending)
    latest = max(subdirs, key=lambda d: d.stat().st_mtime)
    return latest

def analyze_progress(run_dir: Path, total_per_exp: int):
    """
    Scans the run directory for sub-experiments and counts JSON files.
    Returns a list of dicts with progress info.
    """
    # Structure: run_dir / sub_exp_dir / *.json
    
    stats = []
    
    # Check if run_dir itself contains json files directly (legacy or flat structure)
    # OR if it contains subdirectories for different strategies
    
    # We assume 'experiment_runner' creates subdirectories named "{strategy}__{masking}"
    # BUT, looking at pipeline.py: 
    # self._save_path = save_path/run_name 
    # where run_name = f"{recon_strategy}__{masker}"
    # AND self.result_path = .../parent_run_name
    # SO: run_dir (parent_run_name) contains subdirs (one per runner).
    
    sub_exps = [d for d in run_dir.iterdir() if d.is_dir()]
    # Sometimes HF sync creates .cache or similar, ignore hidden
    sub_exps = [d for d in sub_exps if not d.name.startswith('.')]
    
    if not sub_exps:
        # Maybe it's a flat dir?
        count = len(list(run_dir.glob("*.json")))
        if count > 0:
            stats.append({
                "name": "root",
                "count": count,
                "mtime": run_dir.stat().st_mtime
            })
    else:
        for sub in sub_exps:
            jsons = list(sub.glob("*.json"))
            count = len(jsons)
            
            # Find time of last created json to check activity
            last_activity = "Never"
            ts = 0
            if jsons:
                latest_json = max(jsons, key=lambda f: f.stat().st_mtime)
                ts = latest_json.stat().st_mtime
                mins_ago = (time.time() - ts) / 60
                last_activity = f"{mins_ago:.1f}m ago"
            else:
                last_activity = "Initializing..."
                # Use dir creation time as fallback for sorting
                ts = sub.stat().st_ctime
            
            stats.append({
                "name": sub.name,
                "count": count,
                "last_activity": last_activity,
                "ts": ts
            })

    # Sort by activity (most recent first)
    stats.sort(key=lambda x: x['ts'], reverse=True)
    return stats

def get_log_tail(log_file: Path, n: int = 10):
    if not log_file or not log_file.exists():
        return []
    
    # Simple tail implementation
    try:
        # Use existing 'tail' command if on linux/mac for efficiency
        result = subprocess.run(['tail', '-n', str(n), str(log_file)], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip().split('\n')
    except:
        pass
        
    # Fallback python implementation
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            return [l.strip() for l in lines[-n:]]
    except Exception:
        return ["<Error reading log file>"]

def check_errors(log_file: Path):
    """Scans the end of the log file for critical errors."""
    if not log_file or not log_file.exists():
        return []
        
    alerts = []
    try:
        # Check last 50 lines for errors
        tail = get_log_tail(log_file, 50)
        for line in tail:
            if "CRITICAL" in line or "Traceback" in line: # "ERROR" might be too noisy if handled
                if "Failed with a critical error" in line or "Traceback" in line:
                    alerts.append(line[:100] + "...")
    except:
        pass
    return list(set(alerts)) # Dedup

def parse_total_experiments(log_file: Path) -> int | None:
    """Parses the log file to find the total number of planned experiments."""
    if not log_file or not log_file.exists():
        return None
    
    try:
        # Scan the first 100 lines (it usually appears early)
        with open(log_file, 'r') as f:
            for _ in range(100):
                line = f.readline()
                if not line: break
                # Look for: "prepared X experiments, with Y videos. Total runs = Z"
                if "Total runs =" in line:
                    parts = line.split("Total runs =")
                    if len(parts) > 1:
                        return int(parts[1].strip())
    except Exception:
        pass
    return None

def calculate_eta(start_time: float, completed: int, total: int) -> str:
    """Calculates ETA based on simple linear projection."""
    if completed == 0:
        return "TBD"
    
    elapsed = time.time() - start_time
    rate = completed / elapsed # videos per second
    
    remaining = total - completed
    if remaining <= 0:
        return "0s (Done)"
        
    seconds_left = remaining / rate
    
    # Format nicely
    if seconds_left < 60:
        return f"{seconds_left:.0f}s"
    elif seconds_left < 3600:
        return f"{seconds_left/60:.1f}m"
    else:
        return f"{seconds_left/3600:.1f}h"


def main():
    parser = argparse.ArgumentParser(description="Monitor Caption Reconstruction Experiments")
    parser.add_argument("--results-dir", type=Path, default=Path("results/recon"))
    parser.add_argument("--log-file", type=Path, help="Path to the running log file")
    parser.add_argument("-n", type=int, default=10, help="Number of log lines to show")
    parser.add_argument("--total", type=int, default=100, help="Total expected videos per experiment")
    args = parser.parse_args()

    # Clear screen (optional, maybe distracting in notebook)
    # os.system('cls' if os.name == 'nt' else 'clear')

    print(f"--- Status Monitor: {datetime.now().strftime('%H:%M:%S')} ---")
    print(f"System: {get_system_stats()}")
    print(f"GPU:    {get_gpu_stats()}")
    print("-" * 60)

    # Find Run
    target_run = find_latest_run(args.results_dir)
    if not target_run:
        print(f"No experiment runs found in {args.results_dir}")
        return

    print(f"Latest Experiment: {target_run.name}")
    
    # Analyze Progress
    sub_stats = analyze_progress(target_run, args.total)
    
    print(f"{'Sub-Experiment':<50} | {'Progress':<15} | {'Last Activity'}")
    print("-" * 85)
    
    total_completed = 0
    for s in sub_stats:
        pct = (s['count'] / args.total) * 100
        print(f"{s['name'][:47]+'...':<50} | {s['count']}/{args.total} ({pct:.0f}%)   | {s['last_activity']}")
        total_completed += s['count']
        
    print("-" * 85)
    print("-" * 85)
    print(f"Total Videos Completed: {total_completed}")
    
    # Total Experiments & ETA
    total_runs_expected = None
    if args.log_file:
         total_runs_expected = parse_total_experiments(args.log_file)
    
    if total_runs_expected:
        # Global ETA
        # We need a start time. Use the modification time of the results directory 
        # (created at start of run) or the oldest file found.
        start_time = target_run.stat().st_ctime
        eta = calculate_eta(start_time, total_completed, total_runs_expected)
        
        progress_bar = f"[{'#' * int(total_completed/total_runs_expected*20):<20}]"
        
        print(f"Overall Progress:       {total_completed}/{total_runs_expected} {progress_bar} ({total_completed/total_runs_expected*100:.1f}%)")
        print(f"Estimated Time Left:    {eta}")
    else:
        print("(Total runs unknown - logs not parsed or line missing)")

    # Errors
    if args.log_file:
        alerts = check_errors(args.log_file)
        if alerts:
            print("\n⚠️  POTENTIAL ERRORS DETECTED:")
            for a in alerts:
                print(f"  🔴 {a}")

    # Log Tail
    if args.log_file:
        print("\n--- Log Tail ---")
        tail = get_log_tail(args.log_file, args.n)
        for line in tail:
            print(line)
    else:
        print("\n(No log file specified. Use --log-file to see tail)")

if __name__ == "__main__":
    main()
