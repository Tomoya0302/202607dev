"""soft/priority品の積み方の実態を診断する（公式コード無変更、診断専用）。

ユーザー仮説（2026-08-11）: 「現状softのまとまりはsoftのみ・hardのまとまりはhardのみで
構成されがちで、特にsoft-on-softの方が不安定性を上げているのでは」「priority品は
GH_PRIO_LATE=1で末尾に送られるため、上に何も積まれず空間利用が損なわれているのでは」。

本スクリプトは v52 既定で1エピソードを実行し、沈降後の実座標から各荷物の支持関係
（floor/soft-item/hard-item のどれに乗っているか）と、`bench_run.py::_static_equilibrium_proxy`
と同じ支持多角形マージンを対応づけ、以下を集計する:
  1. soft品はfloor/soft/hardのどれに乗っている割合が高いか
  2. 支持タイプ別（soft-on-soft vs soft-on-hard等）でマージン（安定性proxy）に差があるか
  3. priority品の**上に何が乗っているか**（無し/非priority/priority）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np


def _run_and_collect(config_path: str, module_path: str = "agents/heuristic/"):
    from src.ground_handling.agent_factory import AgentFactory
    from src.ground_handling.env import GroundHandlingEnv
    from src.ground_handling.runner import TimedAgentRunner

    with open(config_path) as f:
        config = json.load(f)
    task_id, task_config = next(iter(config.items()))
    agent_module_path = ".".join(module_path.split("/")) + "agent"
    agent_factory = AgentFactory(module_name=agent_module_path, class_name="Agent",
                                  module_path=module_path)
    env = GroundHandlingEnv(config=task_config, verbose=False, render_mode=None)
    env.reset_settings()
    init_states = env.get_init_states()
    allowed_methods = task_config["agent"]["allowed_methods"]
    max_mem = task_config["agent"].get("max_mem", 4)
    runner = TimedAgentRunner(agent_factory=agent_factory, allowed_methods=allowed_methods,
                               max_mem=max_mem, verbose=False)
    runner.call("get_init_states", time_out_sec=task_config["agent"]["init_timeout"],
                fallback=None, init_states=init_states)
    if env.optimize:
        item_list = env.get_info_for_optimization()
        optimized_order, _ = runner.call(
            "optimize", time_out_sec=task_config["agent"]["optimization_timeout"],
            fallback=list(env.stream_manager.all_indices), item_list=item_list)
        env.set_item_order(optimized_order)
    env.reset_item_stream()
    obs, info = env.reset(seed=42)
    terminated = truncated = False
    while not terminated and not truncated:
        action, _ = runner.call("policy", time_out_sec=task_config["agent"]["policy_timeout"],
                                 fallback=env.action_space.sample(), observation=obs)
        obs, reward, terminated, truncated, info = env.step(action)
    runner.close()
    return env


def _analyze(env) -> dict:
    entries = []
    for container in env.container_manager.containers:
        for it in container.packed_items:
            pos, orn = it.get_pose(env.client)
            if pos is None:
                continue
            half = 0.5 * np.abs(
                np.array(env.client.getMatrixFromQuaternion(orn)).reshape(3, 3)
            ) @ np.array([it.length, it.width, it.height], dtype=np.float64)
            pos = np.array(pos, dtype=np.float64)
            entries.append({
                "lo": pos - half, "hi": pos + half, "mass": float(it.mass),
                "is_soft": bool(it.is_soft), "is_prio": bool(it.is_prioritized),
                "container": container,
            })

    n = len(entries)
    z_tol = 0.015
    supports = [[] for _ in range(n)]     # supports[i] = [(j, overlap_area), ...]
    for i, a in enumerate(entries):
        for j, b in enumerate(entries):
            if i == j or a["container"] is not b["container"]:
                continue
            if abs(float(a["lo"][2]) - float(b["hi"][2])) > z_tol:
                continue
            ox = min(a["hi"][0], b["hi"][0]) - max(a["lo"][0], b["lo"][0])
            oy = min(a["hi"][1], b["hi"][1]) - max(a["lo"][1], b["lo"][1])
            if ox > 1e-6 and oy > 1e-6:
                supports[i].append((j, ox * oy))

    def _support_type(i: int) -> str:
        if not supports[i]:
            return "floor"
        j, _ = max(supports[i], key=lambda t: t[1])
        return "soft" if entries[j]["is_soft"] else "hard"

    def _margin(i: int) -> float:
        a = entries[i]
        if not supports[i]:
            x0, y0, x1, y1 = a["lo"][0], a["lo"][1], a["hi"][0], a["hi"][1]
        else:
            rects = []
            for j, _ in supports[i]:
                b = entries[j]
                rects.append((max(a["lo"][0], b["lo"][0]), max(a["lo"][1], b["lo"][1]),
                               min(a["hi"][0], b["hi"][0]), min(a["hi"][1], b["hi"][1])))
            x0 = min(r[0] for r in rects); y0 = min(r[1] for r in rects)
            x1 = max(r[2] for r in rects); y1 = max(r[3] for r in rects)
        cx = 0.5 * (a["lo"][0] + a["hi"][0]); cy = 0.5 * (a["lo"][1] + a["hi"][1])
        return min(min(cx - x0, x1 - cx), min(cy - y0, y1 - cy))

    covered_by = [[] for _ in range(n)]   # covered_by[i] = [j, ...] i の上に乗っている物
    for i in range(n):
        for j, _ in supports[i]:
            covered_by[j].append(i)

    soft_support_counts = {"floor": 0, "soft": 0, "hard": 0}
    margins_by_support = {"floor": [], "soft": [], "hard": []}
    for i, e in enumerate(entries):
        st = _support_type(i)
        m = _margin(i)
        if e["is_soft"]:
            soft_support_counts[st] += 1
        margins_by_support[st].append(m)

    # soft品だけの margin をさらに support type で分解（本題）
    soft_margins_by_support = {"floor": [], "soft": [], "hard": []}
    for i, e in enumerate(entries):
        if e["is_soft"]:
            soft_margins_by_support[_support_type(i)].append(_margin(i))

    prio_cover = {"none": 0, "non_prio": 0, "prio": 0}
    for i, e in enumerate(entries):
        if not e["is_prio"]:
            continue
        covers = covered_by[i]
        if not covers:
            prio_cover["none"] += 1
        elif any(entries[j]["is_prio"] for j in covers):
            prio_cover["prio"] += 1
        else:
            prio_cover["non_prio"] += 1

    def _stat(xs):
        xs = np.asarray(xs, dtype=np.float64)
        if xs.size == 0:
            return {"n": 0, "mean": None, "min": None}
        return {"n": int(xs.size), "mean": float(xs.mean()), "min": float(xs.min())}

    return {
        "n_items": n,
        "soft_support_counts": soft_support_counts,
        "soft_margin_by_support": {k: _stat(v) for k, v in soft_margins_by_support.items()},
        "all_margin_by_support": {k: _stat(v) for k, v in margins_by_support.items()},
        "prio_cover": prio_cover,
        "n_prio_total": sum(1 for e in entries if e["is_prio"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", action="append", required=True)
    args = parser.parse_args()

    agg = {"soft_support_counts": {"floor": 0, "soft": 0, "hard": 0},
           "soft_margins_by_support": {"floor": [], "soft": [], "hard": []},
           "prio_cover": {"none": 0, "non_prio": 0, "prio": 0}, "n_prio_total": 0}

    for path in args.config_path:
        env = _run_and_collect(path)
        result = _analyze(env)
        env.close()
        print(f"{os.path.basename(path)}: {json.dumps(result, indent=None)}")
        for k in ("floor", "soft", "hard"):
            agg["soft_support_counts"][k] += result["soft_support_counts"][k]
        agg["prio_cover"]["none"] += result["prio_cover"]["none"]
        agg["prio_cover"]["non_prio"] += result["prio_cover"]["non_prio"]
        agg["prio_cover"]["prio"] += result["prio_cover"]["prio"]
        agg["n_prio_total"] += result["n_prio_total"]

    print("\n=== aggregate ===")
    total_soft = sum(agg["soft_support_counts"].values())
    print(f"soft品の支持内訳（n={total_soft}）: {agg['soft_support_counts']} "
          f"(soft-on-soft比率={agg['soft_support_counts']['soft']/max(total_soft,1):.1%})")
    print(f"priority品(n={agg['n_prio_total']})の被覆状況: {agg['prio_cover']}")


if __name__ == "__main__":
    main()
