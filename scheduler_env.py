#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scheduler_env.py  (PERF + PPT-Faithful)

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# -----------------------------
# Reward weights (PPT: R = r_base - (Pavg + Phead + Pfrag + Plazy))
# -----------------------------
@dataclass
class RewardWeights:
    w_avg: float = 1.0     # AvgWaitHours
    w_head: float = 1.0    # HeadWaitHours (max)
    w_frag: float = 0.2    # FragmentedBucketCount
    w_lazy: float = 1.0    # feasible job exists but action==WAIT
    r_base: float = 1.0


# -----------------------------
# Environment
# -----------------------------
class GPUJobSchedulerEnv(gym.Env):
    """
    GPU Scheduling Environment implementing PPT design.

    Key perf changes (no design change):
      - Node state stored as bitmask (<=8 GPUs/node): O(1) free count + fast allocation
      - free counts + total free maintained incrementally
      - backlog wait stats maintained in O(1) using (len, sum_submit, oldest_submit)
      - per-step feasibility cache to avoid repeated planning calls

    State S = [S_cluster, S_cand, S_stats, S_frag]
      - S_cluster: per node [Free, Total, IsClean, IsDirty]
      - S_cand(i): first-come candidate per bucket (dim=11)
      - S_stats: bucket stats (B*3) + global(8)
      - S_frag: per bucket {0 available, 1 fragmentation, -1 capacity limit}

    Action:
      - 0: WAIT (event-driven to next submit or finish)
      - 1..: (bucket, policy)
        policy 0: dirty-first best-fit-ish
        policy 1: clean-first best-fit-ish

    Notes:
      - Multi-node allocation allowed (gang scheduling across nodes)
      - Utilization is time-weighted and accumulated across WAIT-driven time jumps.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        job_data,
        node_gpu_layout: List[int],
        project_info: Optional[dict] = None,
        max_steps: int = 2_000_000,
        buckets: Optional[List[int]] = None,
        reward_weights: Optional[RewardWeights] = None,
        time_origin_unix: Optional[float] = None,
        time_unit_seconds: float = 1.0,
    ):
        super().__init__()

        if buckets is None:
            buckets = [1, 2, 4, 8, 16, 32, 64]
        self.buckets = [int(x) for x in buckets]
        self.n_buckets = len(self.buckets)
        self.n_policies = 2

        self.job_data = job_data.sort_values("SUBMIT_TIME").reset_index(drop=True)
        if len(self.job_data) == 0:
            raise ValueError("job_data is empty")

        self.project_info = project_info if project_info is not None else {}

        self.node_gpu_layout = [int(x) for x in node_gpu_layout]
        if len(self.node_gpu_layout) == 0:
            raise ValueError("node_gpu_layout is empty")

        self.n_nodes = len(self.node_gpu_layout)
        self.max_gpus_per_node = int(max(self.node_gpu_layout))
        if self.max_gpus_per_node > 8:
            # 본 프로젝트는 4/8GPU 노드 기준. 8 초과면 bitmask 최적화 의미가 줄어듦.
            # 그래도 동작은 가능하지만 성능 기대치가 달라짐.
            raise ValueError(f"max_gpus_per_node={self.max_gpus_per_node} > 8 (bitmask optimized for <=8)")

        self.total_capacity = int(sum(self.node_gpu_layout))

        self.w = reward_weights if reward_weights is not None else RewardWeights()
        self.time_origin_unix = float(time_origin_unix) if time_origin_unix is not None else None
        self.time_unit_seconds = float(time_unit_seconds)

        # Observation: S_cluster(N*4) + S_cand(B*11) + S_stats(B*3+8) + S_frag(B)
        self.cand_dim = 11
        obs_dim = (self.n_nodes * 4) + (self.n_buckets * self.cand_dim) + (self.n_buckets * 3 + 8) + self.n_buckets
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Action: 0 WAIT, else bucket×policy
        self.action_space = spaces.Discrete(1 + self.n_buckets * self.n_policies)

        self.max_steps = int(max_steps)

        # Utilization accumulators (time-weighted)
        self._last_time: float = 0.0
        self._cum_used_gpu_time: float = 0.0
        self._cum_used_gpu_time_4: float = 0.0
        self._cum_used_gpu_time_8: float = 0.0
        self._cum_total_time: float = 0.0

        # Per-step feasibility cache: key=(req, policy) -> bool
        self._feas_cache: Dict[Tuple[int, int], bool] = {}

        self.reset()

    # -------------------------
    # Gym API
    # -------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        options = options or {}
        start_ptr = int(options.get("start_ptr", 0))
        start_ptr = max(0, min(start_ptr, len(self.job_data) - 1))

        self.job_ptr = start_ptr
        self.current_time = float(self.job_data.iloc[self.job_ptr]["SUBMIT_TIME"])

        # Node state as bitmask: bit=1 used, 0 free (only within cap bits)
        # For cap<max, higher bits set to 1 so they are always "used"
        self.node_used_mask = np.zeros((self.n_nodes,), dtype=np.uint16)
        self.node_free = np.zeros((self.n_nodes,), dtype=np.int16)

        for n, cap in enumerate(self.node_gpu_layout):
            # set bits [cap..max-1] as used (nonexistent)
            if cap < self.max_gpus_per_node:
                high_mask = ((1 << self.max_gpus_per_node) - 1) ^ ((1 << cap) - 1)
                self.node_used_mask[n] = np.uint16(high_mask)
            else:
                self.node_used_mask[n] = np.uint16(0)
            self.node_free[n] = np.int16(cap)  # initially all real GPUs are free

        self.total_free = int(sum(self.node_free.tolist()))

        # Running jobs: {"end": t, "alloc": [(node, idx_list), ...]}
        self.running_jobs: List[Dict] = []

        # Backlog per bucket (FCFS)
        self.backlog: Dict[int, Deque[dict]] = {b: deque() for b in self.buckets}

        # O(1) wait stats per bucket: count, sum_submit, oldest_submit
        self._b_cnt: Dict[int, int] = {b: 0 for b in self.buckets}
        self._b_sum_submit: Dict[int, float] = {b: 0.0 for b in self.buckets}
        self._b_oldest_submit: Dict[int, float] = {b: 0.0 for b in self.buckets}  # valid if cnt>0

        # global O(1)
        self._g_cnt: int = 0
        self._g_sum_submit: float = 0.0
        self._g_oldest_submit: float = 0.0  # valid if g_cnt>0

        self.steps = 0
        self._feas_cache.clear()

        # utilization accumulators
        self._last_time = self.current_time
        self._cum_used_gpu_time = 0.0
        self._cum_used_gpu_time_4 = 0.0
        self._cum_used_gpu_time_8 = 0.0
        self._cum_total_time = 0.0

        self._ingest_arrivals_up_to_current_time()

        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        # stable ordering: release -> ingest at current_time
        self._release_jobs()
        self._ingest_arrivals_up_to_current_time()

        # clear per-step feasibility cache (state changed at this point)
        self._feas_cache.clear()

        # pre-action reward terms
        avg_wait_h, head_wait_h = self._global_wait_stats_hours_fast()
        frag_vec = self._frag_vector()
        frag_count = int(np.sum(np.array(frag_vec) == 1))
        feasible_any = self._exists_feasible_any_policy()

        reward = float(self.w.r_base)
        reward -= float(self.w.w_avg) * float(avg_wait_h)
        reward -= float(self.w.w_head) * float(head_wait_h)
        reward -= float(self.w.w_frag) * float(frag_count)

        info = {}

        if int(action) == 0:
            if feasible_any:
                reward -= float(self.w.w_lazy)
            self._advance_time_to_next_event()
        else:
            a = int(action) - 1
            bucket_idx = a // self.n_policies
            policy = a % self.n_policies
            target_bucket = self.buckets[bucket_idx]

            if self.backlog[target_bucket]:
                allocated, wait_min, job0 = self._try_allocate_first_job_in_bucket(target_bucket, policy)
                if allocated:
                    info = {
                        "wait_time": float(wait_min),
                        "gpu_count": int(job0["gpus"]),
                        "submit_time": float(job0["SUBMIT_TIME"]),
                        "baseline_wait": float((float(job0["START_TIME"]) - float(job0["SUBMIT_TIME"])) / 60.0)
                        if ("START_TIME" in job0 and "SUBMIT_TIME" in job0) else np.nan,
                    }
                else:
                    # Should not happen under correct masking, but keep rollout stable
                    self._advance_time_to_next_event()

        self.steps += 1

        terminated = bool(self._done())
        truncated = bool(self.steps >= self.max_steps)

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    # -------------------------
    # Masking (uses first job's true req)
    # -------------------------
    def action_masks(self) -> List[bool]:
        mask = [True] * self.action_space.n  # WAIT valid

        for i, b in enumerate(self.buckets):
            q = self.backlog[b]
            if not q:
                base = 1 + i * self.n_policies
                mask[base] = False
                mask[base + 1] = False
                continue

            req = int(q[0]["gpus"])
            base = 1 + i * self.n_policies
            mask[base] = bool(self._can_allocate_now(req=req, policy=0))
            mask[base + 1] = bool(self._can_allocate_now(req=req, policy=1))

        return mask

    # -------------------------
    # Observation (PPT faithful, but O(1) wait stats + cached feasibility)
    # -------------------------
    def _get_obs(self) -> np.ndarray:
        # S_cluster
        s_cluster: List[float] = []
        for n in range(self.n_nodes):
            cap = int(self.node_gpu_layout[n])
            free_cnt = int(self.node_free[n])
            is_clean = 1.0 if free_cnt == cap else 0.0
            is_dirty = 1.0 if 0 < free_cnt < cap else 0.0
            s_cluster.extend([float(free_cnt), float(cap), is_clean, is_dirty])

        # S_cand (B*11)
        # dims = [Req_norm, Wait_h, Proj_norm, User_norm, CanFit, Valid, sinH, cosH, sinD, cosD, Bucket_norm]
        s_cand: List[float] = []
        max_bucket = float(max(self.buckets))

        for b in self.buckets:
            q = self.backlog[b]
            if q:
                job = q[0]
                req = int(job["gpus"])

                wait_h = max(0.0, (self.current_time - float(job["SUBMIT_TIME"])) / 3600.0)
                proj_id = float(job.get("PROJ_ID", job.get("PROJECT", 0.0)) or 0.0)
                user_id = float(job.get("USER_ID", job.get("USER", 0.0)) or 0.0)

                req_norm = float(req) / max_bucket
                proj_norm = math.tanh(proj_id / 1000.0)
                user_norm = math.tanh(user_id / 1000.0)

                can_fit = 1.0 if (self._can_allocate_now(req=req, policy=0) or self._can_allocate_now(req=req, policy=1)) else 0.0
                valid = 1.0

                sinH, cosH, sinD, cosD = self._time_cyclic_encoding(self.current_time)
                bucket_norm = float(self._bucket_of(req)) / max_bucket

                s_cand.extend([req_norm, wait_h, proj_norm, user_norm, can_fit, valid, sinH, cosH, sinD, cosD, bucket_norm])
            else:
                s_cand.extend([0.0] * self.cand_dim)

        # S_stats: bucket stats (count, avgwait, maxwait) + global 8
        s_stats: List[float] = []
        for b in self.buckets:
            cnt = self._b_cnt[b]
            if cnt <= 0:
                s_stats.extend([0.0, 0.0, 0.0])
            else:
                # avg wait = current - mean(submit)
                mean_submit = self._b_sum_submit[b] / float(cnt)
                avg_h = max(0.0, (self.current_time - mean_submit) / 3600.0)
                # max wait = current - oldest submit
                mx_h = max(0.0, (self.current_time - self._b_oldest_submit[b]) / 3600.0)
                s_stats.extend([float(cnt), float(avg_h), float(mx_h)])

        total_len = float(self._g_cnt)

        frag_vec = self._frag_vector()
        frag_cnt = float(sum(1 for x in frag_vec if x == 1))

        # Fit count: how many buckets' first job can fit now
        fit_cnt = 0
        for b in self.buckets:
            q = self.backlog[b]
            if q:
                req = int(q[0]["gpus"])
                if self._can_allocate_now(req=req, policy=0) or self._can_allocate_now(req=req, policy=1):
                    fit_cnt += 1

        avg_wait_h, head_wait_h = self._global_wait_stats_hours_fast()
        util_all, util4, util8 = self.get_utilization()

        # global(8): Len, Fit, Waitavg, Waitmax, Frag, Utilall, Util4, Util8
        s_stats.extend([
            float(total_len),
            float(fit_cnt),
            float(avg_wait_h),
            float(head_wait_h),
            float(frag_cnt),
            float(util_all),
            float(util4),
            float(util8),
        ])

        # S_frag
        s_frag = [float(x) for x in frag_vec]

        obs = np.asarray(s_cluster + s_cand + s_stats + s_frag, dtype=np.float32)
        return obs

    # -------------------------
    # Time cyclic encoding (PPT)
    # -------------------------
    def _time_cyclic_encoding(self, t: float) -> Tuple[float, float, float, float]:
        if self.time_origin_unix is not None:
            seconds = (t - self.time_origin_unix) * self.time_unit_seconds
        else:
            seconds = (t - float(self.job_data.iloc[0]["SUBMIT_TIME"])) * self.time_unit_seconds

        seconds = max(0.0, seconds)
        hour = (seconds / 3600.0) % 24.0
        day = (seconds / 86400.0) % 7.0

        sinH = math.sin(2.0 * math.pi * hour / 24.0)
        cosH = math.cos(2.0 * math.pi * hour / 24.0)
        sinD = math.sin(2.0 * math.pi * day / 7.0)
        cosD = math.cos(2.0 * math.pi * day / 7.0)
        return float(sinH), float(cosH), float(sinD), float(cosD)

    # -------------------------
    # Fragmentation vector (bucket req=b) - PPT definition 유지
    # -------------------------
    def _frag_vector(self) -> List[int]:
        vec: List[int] = []
        total_free = int(self.total_free)
        for b in self.buckets:
            req = int(b)
            if total_free < req:
                vec.append(-1)
            else:
                can0 = self._can_allocate_now(req=req, policy=0)
                can1 = self._can_allocate_now(req=req, policy=1)
                vec.append(0 if (can0 or can1) else 1)
        return vec

    # -------------------------
    # Wait stats (O(1))
    # -------------------------
    def _global_wait_stats_hours_fast(self) -> Tuple[float, float]:
        if self._g_cnt <= 0:
            return 0.0, 0.0
        mean_submit = self._g_sum_submit / float(self._g_cnt)
        avg_h = max(0.0, (self.current_time - mean_submit) / 3600.0)
        mx_h = max(0.0, (self.current_time - self._g_oldest_submit) / 3600.0)
        return float(avg_h), float(mx_h)

    # -------------------------
    # Bucket helper
    # -------------------------
    def _bucket_of(self, gpus: int) -> int:
        g = int(gpus)
        for b in self.buckets:
            if g <= b:
                return b
        return self.buckets[-1]

    # -------------------------
    # Bitmask utilities
    # -------------------------
    def _cap_mask(self, cap: int) -> int:
        return (1 << int(cap)) - 1

    def _node_class(self, cap: int, free: int) -> str:
        if free == 0:
            return "empty"
        if free == cap:
            return "clean"
        return "dirty"

    def _get_free_indices(self, n: int, cap: int, take: int) -> List[int]:
        """
        Return up to 'take' free GPU indices on node n within [0,cap).
        Uses bitmask scan (cap <= 8, so trivial).
        """
        used = int(self.node_used_mask[n]) & self._cap_mask(cap)
        out: List[int] = []
        for i in range(cap):
            if (used >> i) & 1 == 0:
                out.append(i)
                if len(out) >= take:
                    break
        return out

    def _mark_used(self, n: int, idxs: List[int]):
        """
        Commit: mark given indices used on node n (update mask/free/total_free).
        """
        if not idxs:
            return
        m = int(self.node_used_mask[n])
        for i in idxs:
            m |= (1 << int(i))
        self.node_used_mask[n] = np.uint16(m)
        self.node_free[n] = np.int16(int(self.node_free[n]) - len(idxs))
        self.total_free -= len(idxs)

    def _mark_free(self, n: int, idxs: List[int]):
        """
        Release: mark given indices free on node n (update mask/free/total_free).
        """
        if not idxs:
            return
        m = int(self.node_used_mask[n])
        for i in idxs:
            m &= ~(1 << int(i))
        self.node_used_mask[n] = np.uint16(m)
        self.node_free[n] = np.int16(int(self.node_free[n]) + len(idxs))
        self.total_free += len(idxs)

    # -------------------------
    # Feasibility (cached)
    # -------------------------
    def _can_allocate_now(self, req: int, policy: int) -> bool:
        req = int(req)
        policy = int(policy)
        if req <= 0:
            return False
        if self.total_free < req:
            return False

        k = (req, policy)
        if k in self._feas_cache:
            return bool(self._feas_cache[k])

        ok, _ = self._plan_allocation_bestfit(req=req, policy=policy, commit=False)
        self._feas_cache[k] = bool(ok)
        return bool(ok)

    def _exists_feasible_any_policy(self) -> bool:
        for b in self.buckets:
            q = self.backlog[b]
            if q:
                req = int(q[0]["gpus"])
                if self._can_allocate_now(req=req, policy=0) or self._can_allocate_now(req=req, policy=1):
                    return True
        return False

    # -------------------------
    # Allocation execution
    # -------------------------
    def _try_allocate_first_job_in_bucket(self, bucket: int, policy: int) -> Tuple[bool, float, dict]:
        q = self.backlog[bucket]
        if not q:
            return False, 0.0, {}

        job0 = q[0]
        req = int(job0["gpus"])

        ok, alloc_plan = self._plan_allocation_bestfit(req=req, policy=policy, commit=True)
        if not ok:
            return False, 0.0, {}

        # pop only after success + update O(1) wait stats
        job = q.popleft()
        self._pop_backlog_stats(bucket=bucket, submit=float(job["SUBMIT_TIME"]))

        duration = 1.0
        if "FINISH_TIME" in job and "START_TIME" in job:
            duration = float(max(1.0, float(job["FINISH_TIME"]) - float(job["START_TIME"])))

        end_t = self.current_time + duration
        self.running_jobs.append({"end": end_t, "alloc": alloc_plan})

        wait_min = (self.current_time - float(job["SUBMIT_TIME"])) / 60.0
        return True, float(max(0.0, wait_min)), job

    # -------------------------
    # Best-fit planner (policy 0 dirty-first, policy 1 clean-first)
    # -------------------------
    def _plan_allocation_bestfit(self, req: int, policy: int, commit: bool) -> Tuple[bool, List[Tuple[int, List[int]]]]:
        req = int(req)
        if req <= 0:
            return False, []
        if self.total_free < req:
            return False, []

        if policy == 1:
            pri = {"clean": 0, "dirty": 1, "empty": 2}
        else:
            pri = {"dirty": 0, "clean": 1, "empty": 2}

        # Build avail list from cached free counts (no numpy scans)
        avail = []
        for n, cap in enumerate(self.node_gpu_layout):
            free = int(self.node_free[n])
            if free > 0:
                cls = self._node_class(cap, free)
                avail.append((n, cap, free, cls))
        if not avail:
            return False, []

        remaining = req
        alloc_plan: List[Tuple[int, List[int]]] = []
        touched: List[Tuple[int, List[int]]] = []

        while remaining > 0:
            # candidates with free>0
            candidates = [(n, cap, free, cls) for (n, cap, free, cls) in avail if free > 0]
            if not candidates:
                # rollback if commit happened mid-way (shouldn't in our flow because commit=True uses _mark_used only)
                if commit:
                    for n, idxs in alloc_plan:
                        self._mark_free(n, idxs)
                return False, []

            def score(item):
                n, cap, free, cls = item
                pr = pri[cls]
                if free >= remaining:
                    # best-fit: minimal leftover
                    return (pr, 0, free - remaining, cap, n)
                # otherwise, take node with largest free
                return (pr, 1, -free, cap, n)

            candidates.sort(key=score)
            n, cap, free, cls = candidates[0]

            take = min(free, remaining)
            idxs = self._get_free_indices(n, cap, take)
            if len(idxs) != take:
                # guard: inconsistency (should not happen). recompute and continue.
                # force this node to "empty" in avail
                new_avail = []
                for (nn, ccap, ffree, ccls) in avail:
                    if nn == n:
                        new_avail.append((nn, ccap, 0, "empty"))
                    else:
                        new_avail.append((nn, ccap, ffree, ccls))
                avail = new_avail
                continue

            # apply
            self._mark_used(n, idxs)
            if not commit:
                touched.append((n, idxs))
            alloc_plan.append((n, idxs))
            remaining -= take

            # update avail entry for node n
            new_avail = []
            for (nn, ccap, ffree, ccls) in avail:
                if nn == n:
                    new_free = ffree - take
                    new_avail.append((nn, ccap, new_free, self._node_class(ccap, new_free)))
                else:
                    new_avail.append((nn, ccap, ffree, ccls))
            avail = new_avail

        # rollback if commit=False
        if not commit:
            for n, idxs in touched:
                self._mark_free(n, idxs)

        return True, alloc_plan

    # -------------------------
    # Event-driven time + utilization accumulation
    # -------------------------
    def _advance_time_to_next_event(self):
        t_next = self._next_event_time()
        self._accumulate_utilization_until(t_next)
        self.current_time = float(t_next)

    def _next_event_time(self) -> float:
        candidates: List[float] = []
        if self.job_ptr < len(self.job_data):
            candidates.append(float(self.job_data.iloc[self.job_ptr]["SUBMIT_TIME"]))
        if self.running_jobs:
            candidates.append(min(float(j["end"]) for j in self.running_jobs))

        if not candidates:
            return float(self.current_time + 1.0)

        t_next = float(min(candidates))
        if t_next <= self.current_time:
            t_next = float(self.current_time + 1.0)
        return t_next

    def _accumulate_utilization_until(self, t_next: float):
        dt = float(max(0.0, t_next - self._last_time))
        if dt <= 0:
            self._last_time = float(t_next)
            return

        used_all = float(self.total_capacity - int(self.total_free))

        used4 = 0.0
        tot4 = 0.0
        used8 = 0.0
        tot8 = 0.0
        for n, cap in enumerate(self.node_gpu_layout):
            used_n = float(cap - int(self.node_free[n]))
            if cap == 4:
                used4 += used_n
                tot4 += 4.0
            elif cap == 8:
                used8 += used_n
                tot8 += 8.0

        self._cum_used_gpu_time += used_all * dt
        self._cum_used_gpu_time_4 += used4 * dt
        self._cum_used_gpu_time_8 += used8 * dt
        self._cum_total_time += dt
        self._last_time = float(t_next)

    def get_utilization(self) -> Tuple[float, float, float]:
        if self._cum_total_time <= 0:
            return 0.0, 0.0, 0.0

        util_all = (self._cum_used_gpu_time / self._cum_total_time) / float(self.total_capacity) if self.total_capacity > 0 else 0.0

        tot4 = float(sum(cap for cap in self.node_gpu_layout if cap == 4))
        tot8 = float(sum(cap for cap in self.node_gpu_layout if cap == 8))

        util4 = (self._cum_used_gpu_time_4 / self._cum_total_time) / tot4 if tot4 > 0 else 0.0
        util8 = (self._cum_used_gpu_time_8 / self._cum_total_time) / tot8 if tot8 > 0 else 0.0
        return float(util_all), float(util4), float(util8)

    # -------------------------
    # Arrivals / backlog stats (O(1))
    # -------------------------
    def _push_backlog_stats(self, bucket: int, submit: float):
        # bucket
        if self._b_cnt[bucket] == 0:
            self._b_oldest_submit[bucket] = submit
        self._b_cnt[bucket] += 1
        self._b_sum_submit[bucket] += submit

        # global
        if self._g_cnt == 0:
            self._g_oldest_submit = submit
        self._g_cnt += 1
        self._g_sum_submit += submit

    def _pop_backlog_stats(self, bucket: int, submit: float):
        # bucket
        self._b_cnt[bucket] -= 1
        self._b_sum_submit[bucket] -= submit
        if self._b_cnt[bucket] <= 0:
            self._b_cnt[bucket] = 0
            self._b_sum_submit[bucket] = 0.0
            self._b_oldest_submit[bucket] = 0.0
        else:
            # oldest changes to next element (FCFS)
            q = self.backlog[bucket]
            self._b_oldest_submit[bucket] = float(q[0]["SUBMIT_TIME"]) if q else 0.0

        # global
        self._g_cnt -= 1
        self._g_sum_submit -= submit
        if self._g_cnt <= 0:
            self._g_cnt = 0
            self._g_sum_submit = 0.0
            self._g_oldest_submit = 0.0
        else:
            # update global oldest by checking bucket heads only (B is small: 7)
            oldest = None
            for b in self.buckets:
                if self._b_cnt[b] > 0:
                    t0 = self._b_oldest_submit[b]
                    if oldest is None or t0 < oldest:
                        oldest = t0
            self._g_oldest_submit = float(oldest) if oldest is not None else 0.0

    def _ingest_arrivals_up_to_current_time(self):
        while self.job_ptr < len(self.job_data) and float(self.job_data.iloc[self.job_ptr]["SUBMIT_TIME"]) <= self.current_time:
            row = self.job_data.iloc[self.job_ptr]
            job = row.to_dict()
            job["gpus"] = int(job.get("gpus", job.get("SLOTS", 0)))

            b = self._bucket_of(job["gpus"])
            if b not in self.backlog:
                b = self.buckets[-1]

            self.backlog[b].append(job)
            self._push_backlog_stats(bucket=b, submit=float(job["SUBMIT_TIME"]))

            self.job_ptr += 1

    def _release_jobs(self):
        active = []
        for j in self.running_jobs:
            if float(j["end"]) <= self.current_time:
                for n, idxs in j["alloc"]:
                    self._mark_free(int(n), list(idxs))
            else:
                active.append(j)
        self.running_jobs = active

    # -------------------------
    # Done
    # -------------------------
    def _done(self) -> bool:
        all_arrived = (self.job_ptr >= len(self.job_data))
        backlog_empty = (self._g_cnt == 0)
        running_empty = (len(self.running_jobs) == 0)
        return bool(all_arrived and backlog_empty and running_empty)