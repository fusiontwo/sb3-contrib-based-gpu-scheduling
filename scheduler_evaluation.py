#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scheduler_evaluation.py
#
# Adds:
#  - Progress logging during rollout (jobs done / total, steps, speed, ETA)
#  - Safe access to base env methods via env.unwrapped (e.g., get_utilization)
#  - Safer mask retrieval via env.unwrapped.action_masks()

import os
import re
import json
import time
import argparse
import random
import warnings
from typing import List, Dict, Set, Tuple, Optional

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from scheduler_env import GPUJobSchedulerEnv


def mask_fn(env):
    # Even when wrapped by ActionMasker, always use the base env's mask
    return env.unwrapped.action_masks()


# -----------------------------
# Host parsing / pool utilities (same as train)
# -----------------------------
_RANGE_RE = re.compile(r"^h\[(\d+)-(\d+)\]$")
_SINGLE_RE = re.compile(r"^h(\d+)$")
_EXEC_HOST_RE = re.compile(r"h(\d+)")


def _expand_host_expr(host_expr: str) -> List[str]:
    host_expr = host_expr.strip()
    m = _RANGE_RE.match(host_expr)
    if m:
        a = int(m.group(1))
        b = int(m.group(2))
        return [f"h{i}" for i in range(a, b + 1)]
    m = _SINGLE_RE.match(host_expr)
    if m:
        return [f"h{int(m.group(1))}"]
    return []


def _parse_pool_nodes(conf: dict, pool_key: str) -> List[str]:
    if pool_key not in conf:
        raise KeyError(f"pool_key '{pool_key}' not found in pool conf")
    nodes = conf[pool_key].get("nodes", [])
    include = set()
    exclude = set()

    for entry in nodes:
        entry = entry.strip()
        if not entry:
            continue

        is_excl = entry.startswith("~")
        if is_excl:
            entry = entry[1:].strip()

        parts = [p.strip() for p in entry.split(",")]
        if not parts:
            continue

        host_part = parts[0]
        hosts = _expand_host_expr(host_part)

        if is_excl:
            exclude.update(hosts)
        else:
            include.update(hosts)

    return sorted(include - exclude)


def _hostnum(host: str) -> Optional[int]:
    m = _SINGLE_RE.match(host.strip())
    if not m:
        return None
    return int(m.group(1))


def _is_h100_host(host: str) -> bool:
    n = _hostnum(host)
    if n is None:
        return False
    return (5000 <= n <= 5999) or (9000 <= n <= 9999)


def _h100_capacity(host: str) -> Optional[int]:
    n = _hostnum(host)
    if n is None:
        return None
    if 9000 <= n <= 9999:
        return 8
    if 5000 <= n <= 5999:
        return 4
    return None


def _extract_hosts_from_exec_host(exec_host: str) -> Set[str]:
    nums = _EXEC_HOST_RE.findall(str(exec_host))
    return {f"h{int(x)}" for x in nums}


def filter_jobs_to_allowed_hosts(df: pd.DataFrame, allowed_hosts: Set[str]) -> pd.DataFrame:
    def ok(exec_host: str) -> bool:
        hs = _extract_hosts_from_exec_host(exec_host)
        if not hs:
            return False
        return all(h in allowed_hosts for h in hs)

    mask = df["EXEC_HOST"].apply(ok)
    return df.loc[mask].copy()


def build_h100_hosts_from_pool_conf(pool_conf_path: str, pool_key: str) -> Tuple[Set[str], Dict[str, int]]:
    with open(pool_conf_path, "r", encoding="utf-8") as f:
        conf = json.load(f)

    hosts = _parse_pool_nodes(conf, pool_key=pool_key)
    h100_hosts = [h for h in hosts if _is_h100_host(h)]

    n4 = sum(1 for h in h100_hosts if _h100_capacity(h) == 4)
    n8 = sum(1 for h in h100_hosts if _h100_capacity(h) == 8)

    return set(h100_hosts), {
        "h100_total": len(h100_hosts),
        "h100_4": n4,
        "h100_8": n8,
        "pool_total": len(hosts),
    }


def sample_node_layout_from_hosts(
    h100_hosts: List[str],
    target_total_nodes: int,
    target_4gpu_nodes: int,
    target_8gpu_nodes: int,
    seed: int,
) -> Tuple[List[int], Dict[str, int]]:
    g4 = [h for h in h100_hosts if _h100_capacity(h) == 4]
    g8 = [h for h in h100_hosts if _h100_capacity(h) == 8]

    rnd = random.Random(seed)
    rnd.shuffle(g4)
    rnd.shuffle(g8)

    picked = []
    picked_4 = g4[: min(len(g4), target_4gpu_nodes)]
    picked_8 = g8[: min(len(g8), target_8gpu_nodes)]
    picked.extend(picked_4)
    picked.extend(picked_8)

    remain = target_total_nodes - len(picked)
    if remain > 0:
        rest = g8[len(picked_8):] + g4[len(picked_4):]
        picked.extend(rest[:remain])

    picked = picked[:target_total_nodes]

    layout = []
    for h in picked:
        cap = _h100_capacity(h)
        if cap not in (4, 8):
            raise ValueError(f"Non-H100 host slipped into layout: {h}")
        layout.append(cap)

    stats = {
        "n_total": len(layout),
        "n4": sum(1 for x in layout if x == 4),
        "n8": sum(1 for x in layout if x == 8),
    }
    return layout, stats


def _compute_wait_stats(arr: np.ndarray) -> Tuple[float, float, float]:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    avg = float(np.mean(arr))
    p95 = float(np.percentile(arr, 95))
    mx = float(np.max(arr))
    return avg, p95, mx


# -----------------------------
# Baseline utilization computation from trace
# -----------------------------
def _split_slots_by_hosts(exec_host: str, slots: int) -> Tuple[int, int]:
    """
    Split total GPU slots into (slots_on_4gpu_nodes, slots_on_8gpu_nodes)
    based on host capacities inferred from host id range rule.
    """
    hosts = sorted(list(_extract_hosts_from_exec_host(exec_host)))
    remain = int(slots)
    s4 = 0
    s8 = 0
    for h in hosts:
        cap = _h100_capacity(h)
        if cap not in (4, 8):
            continue
        take = min(cap, remain)
        if cap == 4:
            s4 += take
        else:
            s8 += take
        remain -= take
        if remain <= 0:
            break
    return s4, s8


def compute_trace_utilization(df: pd.DataFrame, total_capacity: int, tot4: int, tot8: int) -> Tuple[float, float, float]:
    """
    Time-weighted utilization using START_TIME~FINISH_TIME and gpus(SLOTS).
    util_all = sum(slots * runtime) / (total_capacity * window_time)
    util4/util8 estimated by splitting slots across exec hosts (heuristic).
    """
    d = df.copy()
    for c in ["SUBMIT_TIME", "START_TIME", "FINISH_TIME", "gpus"]:
        if c not in d.columns:
            raise ValueError(f"Missing column for baseline util: {c}")

    d = d[np.isfinite(d["START_TIME"]) & np.isfinite(d["FINISH_TIME"]) & np.isfinite(d["gpus"])].copy()
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")

    window_start = float(d["SUBMIT_TIME"].min())
    window_end = float(d["FINISH_TIME"].max())
    window_time = max(1.0, window_end - window_start)

    used_all = 0.0
    used4 = 0.0
    used8 = 0.0

    # Note: iterrows() is slower but acceptable here (run once per evaluation)
    for _, r in d.iterrows():
        slots = int(r["gpus"])
        rt = float(max(1.0, float(r["FINISH_TIME"]) - float(r["START_TIME"])))
        used_all += slots * rt

        s4, s8 = _split_slots_by_hosts(str(r.get("EXEC_HOST", "")), slots)
        used4 += float(s4) * rt
        used8 += float(s8) * rt

    util_all = used_all / (float(total_capacity) * window_time) if total_capacity > 0 else float("nan")
    util4 = (used4 / window_time) / float(tot4) if tot4 > 0 else float("nan")
    util8 = (used8 / window_time) / float(tot8) if tot8 > 0 else float("nan")
    return float(util_all), float(util4), float(util8)


def _format_eta(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "inf"
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60.0:.1f}min"
    return f"{seconds/3600.0:.2f}h"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)

    parser.add_argument("--log_csv", type=str, default="./data/20250914-20251101_logdata__anon.csv")
    parser.add_argument("--pool_conf", type=str, default="./data/pool_conf_250912_anon.json")
    parser.add_argument("--pool_key", type=str, default="all_gpu_pool")
    parser.add_argument("--project_conf", type=str, default="./data/project_conf_250902_anon.json")

    parser.add_argument("--h100_only", action="store_true")

    # Evaluation controls
    parser.add_argument("--jobs", type=int, default=137170, help="PPT reproduction default")
    parser.add_argument("--contiguous_start", type=int, default=0, help="0 recommended for full timeline reproduction")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--target_nodes", type=int, default=500)
    parser.add_argument("--target_4gpu", type=int, default=270)
    parser.add_argument("--target_8gpu", type=int, default=230)

    parser.add_argument("--fp16", action="store_true", help="Optional: use fp16 policy inference on CUDA")

    # Progress logging (NEW)
    parser.add_argument(
        "--progress_every",
        type=int,
        default=2000,
        help="Print progress every N env steps (0 disables step-based printing)",
    )
    parser.add_argument(
        "--progress_time",
        type=int,
        default=10,
        help="Also print progress at least every N seconds",
    )

    args = parser.parse_args()

    os.makedirs("./result", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    with open(args.project_conf, "r", encoding="utf-8") as f:
        project_info = json.load(f)

    log_df = pd.read_csv(args.log_csv)
    log_df["gpus"] = log_df["SLOTS"].astype(int)
    log_df = log_df.sort_values("SUBMIT_TIME").reset_index(drop=True)

    if not args.h100_only:
        raise ValueError("For PPT reproduction, enable --h100_only.")

    allowed_hosts, host_stats = build_h100_hosts_from_pool_conf(args.pool_conf, args.pool_key)
    print(
        f"[PoolHosts] pool_total={host_stats['pool_total']}, h100_total={host_stats['h100_total']}, "
        f"h100_4={host_stats['h100_4']}, h100_8={host_stats['h100_8']}"
    )

    before = len(log_df)
    log_df = filter_jobs_to_allowed_hosts(log_df, allowed_hosts)
    after = len(log_df)
    print(f"[JobFilter:H100] kept={after} / {before} rows (EXEC_HOST subset of H100 pool hosts)")

    node_layout, layout_stats = sample_node_layout_from_hosts(
        sorted(list(allowed_hosts)),
        target_total_nodes=args.target_nodes,
        target_4gpu_nodes=args.target_4gpu,
        target_8gpu_nodes=args.target_8gpu,
        seed=args.seed,
    )
    print(
        f"[ClusterLayout] total={layout_stats['n_total']} nodes, 4GPU={layout_stats['n4']}, "
        f"8GPU={layout_stats['n8']}, unknown_hosts=0"
    )
    total_capacity = int(sum(node_layout))
    print(f"[ClusterLayout] total_capacity={total_capacity} GPUs")

    # Eval contiguous window
    if args.jobs <= 0:
        raise ValueError("--jobs must be > 0")
    if args.jobs > len(log_df):
        raise ValueError(f"--jobs({args.jobs}) > dataset size({len(log_df)}) after filtering")

    start = max(0, min(int(args.contiguous_start), len(log_df) - args.jobs))
    eval_df = log_df.iloc[start : start + args.jobs].reset_index(drop=True)
    print(f"[EvalWindow] start={start}, jobs={len(eval_df)}")

    # Baseline wait computed from dataset timestamps (minutes)
    baseline_wait = (eval_df["START_TIME"] - eval_df["SUBMIT_TIME"]) / 60.0

    # Baseline utilization (trace-based)
    tot4 = int(sum(cap for cap in node_layout if cap == 4))
    tot8 = int(sum(cap for cap in node_layout if cap == 8))
    base_util_all, base_util4, base_util8 = compute_trace_utilization(
        eval_df, total_capacity=total_capacity, tot4=tot4, tot8=tot8
    )

    # Build env, then wrap with ActionMasker
    base_env = GPUJobSchedulerEnv(
        job_data=eval_df,
        node_gpu_layout=node_layout,
        project_info=project_info,
        max_steps=10_000_000,
        buckets=[1, 2, 4, 8, 16, 32, 64],
    )
    env = ActionMasker(base_env, mask_fn)

    model = MaskablePPO.load(args.model, device=device)
    if args.fp16 and use_cuda:
        # Optional: fp16 policy inference
        model.policy = model.policy.to(dtype=torch.float16)

    obs, _ = env.reset(seed=args.seed)

    gpu_counts: List[int] = []
    baseline_waits: List[float] = []
    rl_waits: List[float] = []

    terminated = False
    truncated = False

    total_jobs = len(eval_df)

    # Progress trackers
    step_n = 0
    t0 = time.time()
    last_print_t = t0
    last_print_step = 0
    last_print_done = 0

    print("[Eval] Running single-cluster rollout until completion...")
    while not (terminated or truncated):
        # Always fetch masks from the base env for robustness
        masks = env.unwrapped.action_masks()

        if args.fp16 and use_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        else:
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(int(action))
        step_n += 1

        # Record job-level stats when the env reports a scheduled job
        if info and "wait_time" in info:
            gpu_counts.append(int(info["gpu_count"]))
            baseline_waits.append(float(info.get("baseline_wait", np.nan)))
            rl_waits.append(float(info["wait_time"]))

        # Print progress regularly (by steps and/or by wall time)
        if args.progress_every > 0 or args.progress_time > 0:
            now = time.time()
            step_trigger = (args.progress_every > 0) and (step_n % args.progress_every == 0)
            time_trigger = (args.progress_time > 0) and ((now - last_print_t) >= float(args.progress_time))
            if step_trigger or time_trigger:
                done_jobs = len(rl_waits)  # number of jobs for which wait_time was recorded
                frac = (done_jobs / total_jobs) if total_jobs > 0 else 0.0

                # Segment speed since last print
                dstep = step_n - last_print_step
                djobs = done_jobs - last_print_done
                dt = max(1e-9, now - last_print_t)
                steps_per_s = dstep / dt
                jobs_per_s = djobs / dt

                # Global average speed for ETA
                total_dt = max(1e-9, now - t0)
                avg_jobs_per_s = (done_jobs / total_dt) if done_jobs > 0 else 0.0
                remain = max(0, total_jobs - done_jobs)
                eta_s = (remain / avg_jobs_per_s) if avg_jobs_per_s > 1e-12 else float("inf")

                print(
                    f"[EvalProgress] jobs={done_jobs:,}/{total_jobs:,} ({frac*100:.2f}%) | "
                    f"steps={step_n:,} | "
                    f"rate={jobs_per_s:.2f} jobs/s, {steps_per_s:.1f} steps/s | "
                    f"ETA={_format_eta(eta_s)} | last_reward={float(reward):.4f}"
                )

                last_print_t = now
                last_print_step = step_n
                last_print_done = done_jobs

    # Utilization is a custom method on the base env; use unwrapped to access it
    util_all, util4, util8 = env.unwrapped.get_utilization()
    env.close()

    if len(rl_waits) == 0:
        print("No scheduled jobs were recorded. Check env feasibility/masks/model.")
        return

    # Optional: warn if job-level records are incomplete (helps avoid misleading summaries)
    if len(rl_waits) != len(eval_df):
        print(f"[WARN] Only {len(rl_waits):,} / {len(eval_df):,} jobs recorded in rollout.")

    res_df = pd.DataFrame({"gpu_count": gpu_counts, "baseline_wait": baseline_waits, "rl_wait": rl_waits})

    # Save raw
    csv_path = f"./result/evaluation_raw_{timestamp}.csv"
    res_df.to_csv(csv_path, index=False)
    print(f"Saved raw results: {csv_path}")

    # Summary table (avg/p95/max + util)
    rl_arr = res_df["rl_wait"].to_numpy(dtype=float)
    base_arr = baseline_wait.to_numpy(dtype=float)  # baseline from eval_df (same window)

    rl_avg, rl_p95, rl_max = _compute_wait_stats(rl_arr)
    b_avg, b_p95, b_max = _compute_wait_stats(base_arr)

    summary = pd.DataFrame(
        [
            ["Baseline(Dataset)", b_avg, b_p95, b_max, base_util_all * 100.0, base_util4 * 100.0, base_util8 * 100.0],
            ["Ours(RL rollout)", rl_avg, rl_p95, rl_max, util_all * 100.0, util4 * 100.0, util8 * 100.0],
        ],
        columns=["Method", "AvgWait(min)", "P95Wait(min)", "MaxWait(min)", "Util_all(%)", "Util_4GPU(%)", "Util_8GPU(%)"],
    )

    summary_path = f"./result/evaluation_summary_{timestamp}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")
    print(summary)

    # Grouped by requested GPU count
    bins = [0, 1, 2, 4, 8, 16, 32, 64, np.inf]
    labels = ["1", "2", "4", "8", "16", "32", "64", "64+"]

    res_df["gpu_group"] = pd.cut(res_df["gpu_count"], bins=bins, labels=labels, include_lowest=True)
    grouped = (
        res_df.groupby("gpu_group", observed=True)
        .agg(
            baseline_avg=("baseline_wait", "mean"),
            rl_avg=("rl_wait", "mean"),
            count=("gpu_count", "size"),
        )
        .reindex(labels)
    )

    table_path = f"./result/evaluation_group_table_{timestamp}.csv"
    grouped.to_csv(table_path, index=True)
    print(f"Saved grouped table: {table_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    grouped[["baseline_avg", "rl_avg"]].plot(kind="bar", width=0.85, ax=ax)

    ax.set_title("Wait Time Comparison: Baseline vs RL (MaskablePPO) [Single-Cluster Reproduction]", fontsize=14)
    ax.set_ylabel("Wait Time (minutes)", fontsize=12)
    ax.set_xlabel("Requested GPU Count", fontsize=12)
    ax.set_xticklabels(labels, rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(["Baseline", "RL"], fontsize=10)

    # Annotate bar values
    for p in ax.patches:
        h = p.get_height()
        if np.isfinite(h) and h > 0:
            ax.annotate(
                f"{h:.1f}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="center",
                xytext=(0, 9),
                textcoords="offset points",
                fontsize=9,
            )

    fig.tight_layout()
    plot_path = f"./result/evaluation_plot_{timestamp}.png"
    fig.savefig(plot_path, dpi=300)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()

# Example:
# python scheduler_evaluation.py \
#   --model samsung_gpu_scheduler_maskableppo_YYYYMMDD_HHMM \
#   --log_csv ./data/20250914-20251101_logdata__anon.csv \
#   --pool_conf ./data/pool_conf_250912_anon.json \
#   --pool_key all_gpu_pool \
#   --project_conf ./data/project_conf_250902_anon.json \
#   --h100_only \
#   --target_nodes 500 --target_4gpu 270 --target_8gpu 230 \
#   --jobs 137170 --contiguous_start 0 \
#   --progress_every 2000 --progress_time 10