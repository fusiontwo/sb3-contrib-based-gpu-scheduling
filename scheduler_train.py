#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scheduler_train.py  (HANG FIXED)

import os
import re
import json
import time
import argparse
import random
import multiprocessing as mp
from typing import List, Dict, Set, Tuple, Optional

import numpy as np
import pandas as pd
import torch

import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from scheduler_env import GPUJobSchedulerEnv


# -----------------------------
# Masking helper
# -----------------------------
def mask_fn(env):
    # ActionMasker는 env를 감싸므로 base env mask는 unwrapped로 접근
    return env.unwrapped.action_masks()


# -----------------------------
# Host parsing / pool utilities
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
    """
    Anonymized rules:
      - H100 8GPU: h9xxx
      - H100 4GPU: h5xxx
    """
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


# -----------------------------
# Snapshot replay utilities
# -----------------------------
def find_busy_start_indices(df: pd.DataFrame, gap_sec: float, min_run: int) -> List[int]:
    t = df["SUBMIT_TIME"].astype(float).to_numpy()
    if t.size < 2:
        return [0]
    dt = np.diff(t)
    burst = (dt <= float(gap_sec))

    starts = []
    run_len = 0
    run_start = 0

    for i, is_burst in enumerate(burst, start=1):
        if is_burst:
            if run_len == 0:
                run_start = i - 1
            run_len += 1
        else:
            if run_len >= int(min_run):
                starts.append(int(run_start))
            run_len = 0

    if run_len >= int(min_run):
        starts.append(int(run_start))

    if not starts:
        starts = [max(0, len(df) // 2)]
    return starts


class SnapshotResetWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, start_ptrs: List[int], seed: int = 42):
        super().__init__(env)
        if not start_ptrs:
            start_ptrs = [0]
        self.start_ptrs = [int(x) for x in start_ptrs]
        self.rnd = random.Random(seed)

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        options["start_ptr"] = int(self.rnd.choice(self.start_ptrs))
        return self.env.reset(seed=seed, options=options)


# -----------------------------
# Data loading (HANG-SAFE)
# -----------------------------
def load_log_csv_minimal(path: str) -> pd.DataFrame:
    # 필요한 컬럼만 읽어서 로딩 비용 감소
    usecols = ["SUBMIT_TIME", "START_TIME", "FINISH_TIME", "SLOTS", "EXEC_HOST", "PROJ_ID", "USER_ID", "PROJECT", "USER"]
    # 실제 파일에 없는 컬럼이 있을 수 있으니, 먼저 헤더 확인 후 교집합만 사용
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split(",")
    usecols = [c for c in usecols if c in header]

    df = pd.read_csv(path, usecols=usecols)
    if "SLOTS" not in df.columns:
        raise ValueError("log_csv must contain SLOTS column")
    if "SUBMIT_TIME" not in df.columns:
        raise ValueError("log_csv must contain SUBMIT_TIME column")
    if "EXEC_HOST" not in df.columns:
        raise ValueError("log_csv must contain EXEC_HOST column")
    return df


def ensure_cache_file(train_df: pd.DataFrame, cache_path: str) -> str:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    train_df.to_pickle(cache_path)  # 빠르고 단순. (spawn 워커에서 read_pickle)
    return cache_path


# 전역(워커에서 fork로 상속될 때만 사용)
_GLOBAL_TRAIN_DF = None
_GLOBAL_BUSY_STARTS = None


def make_env_factory(
    *,
    rank: int,
    seed: int,
    max_steps: int,
    layout: List[int],
    project_info: dict,
    buckets: List[int],
    snapshot_replay: bool,
    # spawn 대비: 캐시 파일을 워커에서 읽는 방식
    train_cache_path: Optional[str],
    busy_starts: Optional[List[int]],
    # fork 대비: 전역 DF 사용
    use_global_df: bool,
):
    def _init():
        global _GLOBAL_TRAIN_DF, _GLOBAL_BUSY_STARTS

        if use_global_df:
            if _GLOBAL_TRAIN_DF is None:
                raise RuntimeError("Global train df is None (fork expected but not inherited).")
            train_df = _GLOBAL_TRAIN_DF
            starts = _GLOBAL_BUSY_STARTS if snapshot_replay else [0]
        else:
            if not train_cache_path or not os.path.exists(train_cache_path):
                raise RuntimeError(f"Missing train cache file for spawn: {train_cache_path}")
            train_df = pd.read_pickle(train_cache_path)
            starts = busy_starts if snapshot_replay else [0]

        env = GPUJobSchedulerEnv(
            job_data=train_df,
            node_gpu_layout=layout,
            project_info=project_info,
            max_steps=max_steps,
            buckets=buckets,
        )

        if snapshot_replay:
            env = SnapshotResetWrapper(env, start_ptrs=starts, seed=seed + rank)

        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--log_csv", type=str, default="./data/20250914-20251101_logdata__anon.csv")
    ap.add_argument("--pool_conf", type=str, default="./data/pool_conf_250912_anon.json")
    ap.add_argument("--pool_key", type=str, default="all_gpu_pool")
    ap.add_argument("--project_conf", type=str, default="./data/project_conf_250902_anon.json")

    ap.add_argument("--h100_only", action="store_true", help="Filter jobs and layout to H100-only (h5xxx/h9xxx)")

    ap.add_argument("--train_jobs", type=int, default=100000)
    ap.add_argument("--max_steps", type=int, default=500000)
    ap.add_argument("--total_timesteps", type=int, default=100000)

    ap.add_argument("--target_nodes", type=int, default=500)
    ap.add_argument("--target_4gpu", type=int, default=270)
    ap.add_argument("--target_8gpu", type=int, default=230)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tensorboard_log", type=str, default="./tensorboard_logs/")

    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--torch_tf32", action="store_true")

    ap.add_argument("--snapshot_replay", action="store_true")
    ap.add_argument("--busy_gap_sec", type=float, default=5.0)
    ap.add_argument("--busy_min_run", type=int, default=200)

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--n_steps", type=int, default=2048)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--learning_rate", type=float, default=3e-4)

    # 안정성 옵션
    ap.add_argument("--start_method", type=str, default="auto",
                    choices=["auto", "fork", "forkserver", "spawn"],
                    help="SubprocVecEnv start method. On Linux, fork is fastest/most stable here.")
    ap.add_argument("--force_dummy", action="store_true",
                    help="Force DummyVecEnv (single process) for debugging/hang avoidance.")

    args = ap.parse_args()

    # Reproducibility
    set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # TF32 speed knob
    if args.torch_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    with open(args.project_conf, "r", encoding="utf-8") as f:
        project_info = json.load(f)

    if not args.h100_only:
        raise ValueError("재현 목적이면 --h100_only 를 켜세요.")

    allowed_hosts, host_stats = build_h100_hosts_from_pool_conf(args.pool_conf, args.pool_key)
    print(
        f"[PoolHosts] pool_total={host_stats['pool_total']}, h100_total={host_stats['h100_total']}, "
        f"h100_4={host_stats['h100_4']}, h100_8={host_stats['h100_8']}"
    )

    # Load CSV (minimal cols)
    log_df = load_log_csv_minimal(args.log_csv)
    log_df["gpus"] = log_df["SLOTS"].astype(int)
    log_df = log_df.sort_values("SUBMIT_TIME").reset_index(drop=True)

    before = len(log_df)
    log_df = filter_jobs_to_allowed_hosts(log_df, allowed_hosts)
    after = len(log_df)
    print(f"[JobFilter:H100] kept={after} / {before} rows (EXEC_HOST subset of H100 pool hosts)")

    layout, layout_stats = sample_node_layout_from_hosts(
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
    print(f"[ClusterLayout] total_capacity={sum(layout)} GPUs")

    # Train subset
    train_df = log_df.iloc[: args.train_jobs].reset_index(drop=True)
    print(f"[TrainSet] rows={len(train_df)}")

    # Busy starts (computed once in main)
    busy_starts = [0]
    if args.snapshot_replay:
        busy_starts = find_busy_start_indices(train_df, gap_sec=args.busy_gap_sec, min_run=args.busy_min_run)
    print(f"[SnapshotReplay] enabled={args.snapshot_replay}, busy_starts={len(busy_starts)}")

    # Choose vecenv strategy
    n_envs = max(1, int(args.n_envs))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print("Using cuda device")

    # IMPORTANT:
    # - fork: 안전/빠름. 큰 DF도 피클 없이 상속.
    # - spawn: 워커가 DF를 못 상속하므로 캐시 파일로 전달.
    is_posix = (os.name == "posix")
    if args.start_method == "auto":
        start_method = "fork" if is_posix else "spawn"
    else:
        start_method = args.start_method

    # fork를 쓰는 경우 전역 DF를 설정해 상속되게 함
    use_global_df = (start_method in ("fork", "forkserver")) and is_posix

    # spawn 대비: 캐시 파일 생성 (큰 DF 피클링을 워커별로 하지 않게)
    train_cache_path = None
    if not use_global_df:
        train_cache_path = ensure_cache_file(
            train_df,
            cache_path=os.path.join("./.cache", f"train_df_seed{args.seed}_jobs{len(train_df)}.pkl")
        )

    # fork 대비: 전역 변수에 주입
    global _GLOBAL_TRAIN_DF, _GLOBAL_BUSY_STARTS
    if use_global_df:
        _GLOBAL_TRAIN_DF = train_df
        _GLOBAL_BUSY_STARTS = busy_starts

    def make_one_env(rank: int):
        return make_env_factory(
            rank=rank,
            seed=args.seed,
            max_steps=args.max_steps,
            layout=layout,
            project_info=project_info,
            buckets=[1, 2, 4, 8, 16, 32, 64],
            snapshot_replay=args.snapshot_replay,
            train_cache_path=train_cache_path,
            busy_starts=busy_starts,
            use_global_df=use_global_df,
        )

    # VecEnv build
    if args.force_dummy or n_envs == 1:
        # 단일 프로세스: hang 원인 분리/회피
        env = DummyVecEnv([make_one_env(0)])
        print("[VecEnv] DummyVecEnv (single process)")
    else:
        # SubprocVecEnv: fork(리눅스) 우선
        print(f"[VecEnv] SubprocVecEnv n_envs={n_envs}, start_method={start_method}, use_global_df={use_global_df}")
        env = SubprocVecEnv([make_one_env(i) for i in range(n_envs)], start_method=start_method)

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        tensorboard_log=args.tensorboard_log,
        device=device,
    )

    print(f"Starting RL Training (Training set size: {len(train_df)})...")
    model.learn(total_timesteps=args.total_timesteps)

    timestamp = time.strftime("%Y%m%d_%H%M")
    save_path = f"samsung_gpu_scheduler_maskableppo_{timestamp}"
    model.save(save_path)
    print(f"Model saved as {save_path}")


if __name__ == "__main__":
    main()


# python scheduler_train.py \
#   --log_csv ./data/20250914-20251101_logdata__anon.csv \
#   --pool_conf ./data/pool_conf_250912_anon.json \
#   --pool_key all_gpu_pool \
#   --project_conf ./data/project_conf_250902_anon.json \
#   --h100_only \
#   --target_nodes 500 --target_4gpu 270 --target_8gpu 230 \
#   --train_jobs 100000 --total_timesteps 20000 \
#   --snapshot_replay \
#   --n_envs 1 \
#   --force_dummy \
#   --torch_tf32