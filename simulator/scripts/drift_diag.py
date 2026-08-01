"""既配置荷物（特に soft）の「登録座標」と実座標のズレを診断する（findings §17）。

`validator.py::place_item()` は新規配置1個だけの座標を再取得するため、既配置荷物
（特に soft、`contactStiffness`/`contactDamping` で剛性が低い）は後続の配置による
物理演算で実際には動いても登録座標が更新されない。本スクリプトは各ステップ後に
**全既配置荷物**の登録座標(`item.pos`)と実座標(`item.get_pose(client)`)を比較し、
(a) エピソード後半でドリフトが拡大するか、(b) SKU別の傾向、(c) validator NG との
相関、を1エピソード分の履歴として出力する。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np


def run(config_path: str, module_path: str = "agents/heuristic/", drift_threshold: float = 0.003) -> dict:
    from src.ground_handling.agent_factory import AgentFactory
    from src.ground_handling.env import GroundHandlingEnv
    from src.ground_handling.runner import TimedAgentRunner

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

    # item_index -> 直近に観測した z（初回は登録値、以後は毎ステップの live 値で更新）
    last_seen_z: dict[int, float] = {}
    cum_drift: dict[int, float] = {}  # item_index -> 配置後からの累積下降量
    events = []  # (step, item_index, is_soft, mass, step_drift, cum_drift, validator_ng_this_step)
    step_n = 0
    n_placements = 0
    n_validator_ng = 0

    while not terminated and not truncated:
        action, _ = runner.call("policy", time_out_sec=task_config["agent"]["policy_timeout"],
                                fallback=env.action_space.sample(), observation=obs)
        obs, reward, terminated, truncated, info = env.step(action)
        step_n += 1
        status = info.get("status", {})
        ng_this_step = status.get("is_valid") is False or status.get("is_placed_safe") is False
        if ng_this_step:
            n_validator_ng += 1
        if status.get("is_placed_safe"):
            n_placements += 1

        for c in env.container_manager.containers:
            for it in c.packed_items:
                if it.pos is None:
                    continue
                live_pos, live_orn = it.get_pose(env.client)
                if live_pos is None:
                    continue
                z = float(live_pos[2])
                if it.index not in last_seen_z:
                    last_seen_z[it.index] = z
                    cum_drift[it.index] = 0.0
                    continue
                step_drift = last_seen_z[it.index] - z  # 正 = 下方向へさらに沈んだ
                if abs(step_drift) > 1e-6:
                    cum_drift[it.index] += step_drift
                    last_seen_z[it.index] = z
                    if abs(step_drift) > drift_threshold:
                        events.append({
                            "step": step_n, "item_index": it.index, "is_soft": bool(it.is_soft),
                            "mass": float(it.mass), "step_drift_mm": step_drift * 1000.0,
                            "cum_drift_mm": cum_drift[it.index] * 1000.0,
                            "validator_ng_this_step": bool(ng_this_step),
                        })

    env.close()
    runner.close()

    soft_events = [e for e in events if e["is_soft"]]
    hard_events = [e for e in events if not e["is_soft"]]
    max_cum_soft = max((cum_drift[k] for k, v in last_seen_z.items()
                        if any(e["item_index"] == k and e["is_soft"] for e in events)), default=0.0)
    return {
        "config": os.path.basename(config_path), "n_steps": step_n, "n_placements": n_placements,
        "n_validator_ng": n_validator_ng, "n_drift_events_gt_3mm": len(events),
        "n_soft_events": len(soft_events), "n_hard_events": len(hard_events),
        "max_cum_drift_soft_mm": max_cum_soft * 1000.0 if max_cum_soft else 0.0,
        "events": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--module-path", default="agents/heuristic/")
    parser.add_argument("--drift-threshold", type=float, default=0.003)
    args = parser.parse_args()
    result = run(args.config_path, args.module_path, args.drift_threshold)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
