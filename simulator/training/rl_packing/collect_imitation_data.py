"""模倣学習用データ収集（DRL方策計画 Phase 2a、`docs/HANDOFF.md` 参照）。

実際の v52 既定 `agents.heuristic.agent.Agent`（`optimize()`含む、無改造）を
`TimedAgentRunner`のプロセス分離を経由せず直接実行し、各 `policy()` ステップで
`decide_candidates` の候補群と、実際に選ばれた行動に対応するラベル（候補のインデックス）を
記録する。エージェント本体は一切変更しないため、収集される軌跡は本番 v52 の挙動と
完全に一致する（`optimize()` による順序最適化も含む）。

`decide_candidates` は `decide_placement` と同じ `_build_models_ranked`/カスケードを
共有するため、既定設定下では両者が選ぶ (item_idx, container_idx, orientation) は一致するはず
という前提を、全ステップで実測検証してログに残す（`--verify-only` で検証だけ行うことも可能）。

使い方:
    cd simulator && source .venv/bin/activate
    python -m training.rl_packing.collect_imitation_data \
        --family 1 --n-episodes 20 --seed-base 20260901 --k 64 \
        --out-dir artifacts/rl_training/imitation/f1
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.getcwd())

import numpy as np

from agents.heuristic.agent import Agent
from agents.heuristic.packing_core.heightmap import decide_candidates
from agents.heuristic.packing_core.rl_features import candidate_batch_features, global_context_vector
from agents.heuristic.packing_core.state import build_state
from scripts.gen_tasks import gen_task
from src.ground_handling.env import GroundHandlingEnv


class _StepBudget:
    """`agents/heuristic/agent.py::StepBudget` と同型の最小再実装（学習用、公式コード無改造）。"""

    def __init__(self, t0: float, soft: float, hard: float):
        self.t0 = t0
        self.soft = soft
        self.hard = hard

    def over_soft(self) -> bool:
        return (time.monotonic() - self.t0) > self.soft

    def over_hard(self) -> bool:
        return (time.monotonic() - self.t0) > self.hard


def _match_label(candidates, true_item, true_container, true_orn, true_pos) -> int | None:
    """真の行動 (item_idx, container_idx, orientation, pos_rel) に最も近い候補の
    インデックスを返す。(item, container, orientation) が一致する候補が無ければ None
    （decide_candidates の k が小さすぎて真の候補が入っていない、等の理由で発生しうる）。
    """
    matches = [i for i, c in enumerate(candidates)
               if int(c.item_idx) == true_item and int(c.container_idx) == true_container
               and int(c.orientation) == true_orn]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    # 複数一致（densify差分等）: 位置が最も近いものを選ぶ。
    dists = [float(np.linalg.norm(np.asarray(candidates[i].pos_rel) - np.asarray(true_pos)))
             for i in matches]
    return matches[int(np.argmin(dists))]


def collect_episode(task_config: dict, k: int = 64, policy_soft: float = 4.5,
                     policy_hard: float = 5.5) -> tuple[list[dict], dict]:
    """1エピソードを実行し、(samples, stats) を返す。

    samples の各要素: {"cand_feats": (N,D) float32, "ctx_feat": (D,) float32, "label": int}
    stats: {"n_steps": int, "n_matched": int, "n_unmatched": int, "n_top1_match": int}
    """
    agent = Agent(module_path="agents/heuristic/")
    env = GroundHandlingEnv(config=task_config, verbose=False, render_mode=None)
    env.reset_settings()
    init_states = env.get_init_states()
    agent.get_init_states(init_states)

    if task_config["agent"].get("optimize"):
        item_list = env.get_info_for_optimization()
        order = agent.optimize(item_list)
        if not env.set_item_order(order):
            env.close()
            return [], {"n_steps": 0, "n_matched": 0, "n_unmatched": 0, "n_top1_match": 0,
                        "status": "bad_order"}

    env.reset_item_stream()
    obs, info = env.reset(seed=42)
    terminated = truncated = False

    samples: list[dict] = []
    n_matched = n_unmatched = n_top1_match = n_steps = 0

    while not terminated and not truncated:
        # 教師行動（本番と完全同一のパイプライン、無改造）。
        true_action = agent.policy(obs)

        # decide_candidates 用の state/budget を observation から直接再構築する
        # （agent.py::_policy_impl と同じ組み立て方、§4.4）。
        init = {
            "optimize": obs.get("optimize", agent.optimize_enabled),
            "lookahead_k": obs.get("lookahead_k", agent.lookahead_k),
            "container_list": obs.get("container_list", agent.container_list),
        }
        state = build_state(obs, init, compute_ems=False)
        budget = _StepBudget(t0=time.monotonic(), soft=policy_soft, hard=policy_hard)
        candidates, models = decide_candidates(state, budget, k=k)

        n_steps += 1
        if candidates:
            true_item = int(true_action["item_idx"])
            true_container = int(true_action["container_idx"])
            true_orn = int(true_action["orientation"])
            true_pos = np.asarray(true_action["place_pos"], dtype=np.float64)
            label = _match_label(candidates, true_item, true_container, true_orn, true_pos)
            if label is not None:
                n_matched += 1
                if label == 0:
                    n_top1_match += 1
                cand_feats = candidate_batch_features(candidates, models, state)
                ctx_feat = global_context_vector(state)
                samples.append({"cand_feats": cand_feats, "ctx_feat": ctx_feat, "label": label})
            else:
                n_unmatched += 1

        obs, reward, terminated, truncated, info = env.step(true_action)

    env.close()
    stats = {"n_steps": n_steps, "n_matched": n_matched, "n_unmatched": n_unmatched,
              "n_top1_match": n_top1_match, "status": "ok"}
    return samples, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", type=int, choices=(1, 2), required=True)
    parser.add_argument("--n-episodes", type=int, required=True)
    parser.add_argument("--seed-base", type=int, default=20260901)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    total_steps = total_matched = total_unmatched = total_top1 = 0
    for i in range(args.n_episodes):
        seed = args.seed_base + i
        task = gen_task(args.family, seed)
        t0 = time.time()
        samples, stats = collect_episode(task["000"] if "000" in task else next(iter(task.values())),
                                          k=args.k)
        elapsed = time.time() - t0
        total_steps += stats["n_steps"]
        total_matched += stats["n_matched"]
        total_unmatched += stats["n_unmatched"]
        total_top1 += stats["n_top1_match"]
        out_path = os.path.join(args.out_dir, f"ep_{seed}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(samples, f)
        print(f"episode {i+1}/{args.n_episodes} seed={seed} elapsed={elapsed:.1f}s "
              f"n_samples={len(samples)} matched={stats['n_matched']}/{stats['n_steps']} "
              f"top1={stats['n_top1_match']}/{stats['n_matched'] or 1} "
              f"unmatched={stats['n_unmatched']} status={stats['status']}")

    print(f"\n=== total: steps={total_steps} matched={total_matched} "
          f"unmatched={total_unmatched} top1_match_rate="
          f"{total_top1 / max(total_matched, 1):.3f} ===")


if __name__ == "__main__":
    main()
