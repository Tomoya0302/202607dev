"""階段制約（`HM_STAIR_SLACK` 絶対値版 vs `HM_STAIR_K` 荷物厚み比例版）が、荷物の質量と
配置高さの相関に与える影響を診断する（findings §21.3の未検証仮説）。

v46（`GH_HM_STAIR_SLACK=0.30`）は n=99×3族の局所検証を全通過して本番投入されたが、
cog が過去最大級（−6.66）で崩れた。v47（同じ発想を `GH_HM_STAIR_K`＝荷物の扁平厚みに
比例するslackへ変更）はさらに悪化（cog −15.39）。§21.3 はこの機構を「`STAIR_K` は
大きい（＝重い傾向にある）荷物ほど広い slack を得るため、cog に最も効く質量クラスで
まさに層規律を緩めてしまう」と仮説立てたが、一度も直接検証されないまま残っていた。

本スクリプトは1課題を実行し、**実際に配置された荷物**の質量(mass)と沈降後の相対高さ
(z_rel、`bench_run.py::_compute_cog` と同じ座標変換・inclusion 判定を再利用)を全件
出力する。公式コードは無変更・診断専用（`reach_ceiling.py`/`tilt_diag.py` と同じ流儀）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())


def run(config_path: str, module_path: str = "agents/heuristic/") -> dict:
    from src.ground_handling.agent_factory import AgentFactory
    from src.ground_handling.env import GroundHandlingEnv
    from src.ground_handling.runner import TimedAgentRunner
    from agents.heuristic.packing_core.state import _build_containers, _route_placed

    with open(config_path) as f:
        full = json.load(f)
    task_config = full[next(iter(full.keys()))]

    agent_module = ".".join(module_path.split("/")) + "agent"
    agent_factory = AgentFactory(module_name=agent_module, class_name="Agent", module_path=module_path)
    env = GroundHandlingEnv(config=task_config, verbose=False, render_mode=None)
    env.reset_settings()
    init_states = env.get_init_states()
    runner = TimedAgentRunner(agent_factory=agent_factory,
                              allowed_methods=task_config["agent"]["allowed_methods"],
                              max_mem=task_config["agent"].get("max_mem", 4), verbose=False)
    runner.call("get_init_states", time_out_sec=task_config["agent"]["init_timeout"],
               fallback=None, init_states=init_states)
    if env.optimize:
        item_list = env.get_info_for_optimization()
        order, _ = runner.call("optimize", time_out_sec=task_config["agent"]["optimization_timeout"],
                               fallback=list(env.stream_manager.all_indices), item_list=item_list)
        env.set_item_order(order)
    env.reset_item_stream()
    obs, info = env.reset(seed=42)
    terminated = truncated = False
    n_placements = 0

    while not terminated and not truncated:
        action, _ = runner.call("policy", time_out_sec=task_config["agent"]["policy_timeout"],
                                fallback=env.action_space.sample(), observation=obs)
        obs, reward, terminated, truncated, info = env.step(action)
        status = info.get("status", {})
        if status.get("is_placed_safe"):
            n_placements += 1

    fill_score, out_items = env.evaluator.calculate_fill_rate(env.container_manager.containers)
    num_total = env.num_total_items
    num_placed_items = sum(len(c.packed_items) for c in env.container_manager.containers) / max(1, num_total)

    out_ids = {id(it) for it in out_items}
    init = {"container_list": env.container_manager.get_item_info_in_containers()}
    containers = _build_containers(init)
    observation = {"container_list": init["container_list"]}
    placed = _route_placed(observation, containers)

    excluded_positions: dict[int, set[int]] = {}
    for cidx, container in enumerate(env.container_manager.containers):
        excluded_positions[cidx] = {
            pos for pos, item in enumerate(container.packed_items) if id(item) in out_ids
        }

    hz_all = max((float(c.height) for c in env.container_manager.containers), default=1.0)
    items_data = []
    for cidx, plist in placed.items():
        excl = excluded_positions.get(cidx, set())
        for pos, pi in enumerate(plist):
            if pos in excl:
                continue
            z_rel = 0.5 * (float(pi.aabb_min_rel[2]) + float(pi.aabb_max_rel[2]))
            size = [float(v) for v in pi.size]
            items_data.append({
                "mass": float(pi.weight),
                "z_rel": z_rel,
                "z_norm": z_rel / hz_all,
                "is_soft": bool(pi.is_soft),
                "min_thick": min(size),
            })

    env.close()
    runner.close()

    return {
        "config": os.path.basename(config_path),
        "n_placements": n_placements,
        "fill_score": float(fill_score),
        "num_placed_items": float(num_placed_items),
        "n_items": len(items_data),
        "items": items_data,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--module-path", default="agents/heuristic/")
    args = parser.parse_args()
    result = run(args.config_path, args.module_path)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
