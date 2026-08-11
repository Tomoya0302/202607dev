"""既配置荷物が「積み上げている最中に転倒（傾き・沈降）していないか」を診断する。

ユーザー指摘（「ソフト荷物が起因で、積み上げ中の転倒が発生しているのでは」）の検証用。
`drift_diag.py`（findings §17、z方向のドリフトのみ）を土台に、以下へ拡張する:

1. **姿勢の傾き**（`validator.py::place_item` と同じ式 `2*acos(|dot(q1,q2)|)` で角度化）を、
   z だけでなく XY 移動と合わせて全既配置荷物・全ステップで追跡する。
2. 各荷物の**支持面の種別**（floor/soft/hard/mixed）を、配置直後の沈降後 AABB から
   幾何的に推定する（`bench_run.py::_covered_by_other_kind` と同じ流儀）。
3. エピソード終端で `check_transport_path` が失敗する直前に、**反実仮想**
   （全既配置荷物を「自分が置かれた直後の姿勢」へ一時的に戻した場合に同じ搬入判定が
   通るか）を実行し、ドリフト・傾きが死因に直結しているかを直接測る。
4. エピソード末尾で fill も同様の反実仮想（全荷物を配置直後姿勢に戻して再計算）を行う。

公式コード（`src/ground_handling/`）は一切変更せず、実行時にメソッドをラップするだけ
（`reach_ceiling.py` と同じ流儀）。**診断専用**で提出物には無関係。

`--hard-soft` を付けると、soft 品の接触コンプライアンス（`contactStiffness`/
`contactDamping`/`linearDamping`）だけを配置(`spawn`)時に無効化する（摩擦・質量・寸法・
`is_soft` フラグ自体・エージェントの意思決定は一切変えない）。soft の「柔らかい接触」が
傾きの原因かどうかを因果的に切り分けるための反事実比較用オプション。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np


def _tilt_deg(orn_now: tuple, orn_ref: tuple) -> float:
    """`validator.py::place_item` (:200-202) と同じ式で姿勢差を角度化する。"""
    dot = abs(sum(a * b for a, b in zip(orn_now, orn_ref)))
    dot = min(1.0, dot)
    return math.degrees(2 * math.acos(dot))


def _aabb(client, item, pos, orn) -> tuple[np.ndarray, np.ndarray]:
    half = 0.5 * np.abs(
        np.array(client.getMatrixFromQuaternion(orn)).reshape(3, 3)
    ) @ np.array([item.length, item.width, item.height], dtype=np.float64)
    pos_arr = np.array(pos, dtype=np.float64)
    return pos_arr - half, pos_arr + half


def _classify_support(client, item, pos, orn, container, tol: float = 0.02) -> str:
    """新規荷物の直下にある既配置荷物の種別（floor/soft/hard/mixed）を幾何で推定する。

    `bench_run.py::_covered_by_other_kind` と同じ「XY重なり + 底面が相手の天面近傍」
    判定を、向きを逆にして使う（あちらは「自分の上に載っている物」、こちらは
    「自分を支えている物」）。棚支持（shelf）は Item ではないため対象外＝floor 扱いになる
    （既知の簡略化）。
    """
    if container is None:
        return "unknown"
    lo, hi = _aabb(client, item, pos, orn)
    kinds = set()
    for other in container.packed_items:
        if other.pybullet_id is None:
            continue
        opos, oorn = other.get_pose(client)
        if opos is None:
            continue
        olo, ohi = _aabb(client, other, opos, oorn)
        if olo[0] >= hi[0] or ohi[0] <= lo[0]:
            continue
        if olo[1] >= hi[1] or ohi[1] <= lo[1]:
            continue
        if abs(float(ohi[2]) - float(lo[2])) > tol:
            continue
        kinds.add("soft" if other.is_soft else "hard")
    if not kinds:
        return "floor"
    if len(kinds) > 1:
        return "mixed"
    return next(iter(kinds))


def run(config_path: str, module_path: str = "agents/heuristic/",
        tilt_event_deg: float = 1.0, hard_soft: bool = False,
        counterfactual: bool = True) -> dict:
    from src.ground_handling.agent_factory import AgentFactory
    from src.ground_handling.env import GroundHandlingEnv
    from src.ground_handling.runner import TimedAgentRunner
    from src.ground_handling.validator import PlacementValidator
    from src.ground_handling.items import Item

    with open(config_path) as f:
        full = json.load(f)
    task_config = full[next(iter(full.keys()))]

    # ---- ラップ対象の原型を退避 -------------------------------------------------
    orig_place_item = PlacementValidator.place_item
    orig_check_transport_path = PlacementValidator.check_transport_path
    orig_spawn = Item.spawn

    # ---- 計測用の状態（クロージャで共有） ----------------------------------------
    ref_pose: dict[int, dict] = {}          # item_index -> {"pos":..., "orn":...}（配置直後の沈降後姿勢）
    support_kind: dict[int, str] = {}       # item_index -> floor/soft/hard/mixed（配置直後に一度だけ判定）
    is_soft_of: dict[int, bool] = {}
    mass_of: dict[int, float] = {}
    worst_tilt: dict[int, float] = {}       # item_index -> これまでの最大傾き角[deg]
    worst_disp: dict[int, float] = {}       # item_index -> これまでの最大変位[m]（3D）
    last_tilt: dict[int, float] = {}
    events: list[dict] = []                 # tilt_event_deg を超える単発ステップ変化のログ
    last_container_holder: dict[str, object] = {"container": None}
    counterfactual_record: dict = {"attempted": False, "would_pass_without_drift": None}

    def patched_spawn(self, client, initial_pos=(0.0, 0.0, 5.0), initial_orn=(0, 0, 0, 1)):
        if hard_soft and self.is_soft:
            # is_soft フラグ・友好・寸法・エージェント観測は一切変えず、接触コンプライアンス
            # （contactStiffness/contactDamping/linearDamping の changeDynamics 更新）だけを
            # spawn() 内部のこの1呼び出しに限りスキップする。
            saved = self.is_soft
            self.is_soft = False
            try:
                return orig_spawn(self, client, initial_pos, initial_orn)
            finally:
                self.is_soft = saved
        return orig_spawn(self, client, initial_pos, initial_orn)

    def patched_check_transport_path(self, container, item, target_pos, target_orn_idx, step_len=0.01):
        last_container_holder["container"] = container
        real_result = orig_check_transport_path(self, container, item, target_pos, target_orn_idx, step_len)
        if not real_result and counterfactual and not counterfactual_record["attempted"]:
            # 反実仮想: 既配置荷物を「自分が置かれた直後の姿勢」へ一時的に戻した状態で
            # 同じ搬入判定を再実行する。ドリフト・傾きが死因そのものかを直接測る。
            # env.py の設計上 check_transport_path の失敗は1回でエピソード終了なので、
            # この分岐は1エピソードにつき高々1回しか通らない（安価）。
            state_id = self.client.saveState()
            moved = []
            for other in container.packed_items:
                ref = ref_pose.get(other.index)
                if ref is not None and other.pybullet_id is not None:
                    self.client.resetBasePositionAndOrientation(other.pybullet_id, ref["pos"], ref["orn"])
                    moved.append(other)
            cf_result = orig_check_transport_path(self, container, item, target_pos, target_orn_idx, step_len)
            self.client.restoreState(stateId=state_id)
            self.client.removeState(state_id)
            counterfactual_record["attempted"] = True
            counterfactual_record["would_pass_without_drift"] = bool(cf_result)
            counterfactual_record["n_items_reset_for_test"] = len(moved)
        return real_result

    def patched_place_item(self, item, target_pos, target_orn_idx):
        ok = orig_place_item(self, item, target_pos, target_orn_idx)
        if ok:
            pos, orn = item.get_pose(self.client)
            if pos is not None:
                ref_pose[item.index] = {"pos": pos, "orn": orn}
                worst_tilt[item.index] = 0.0
                worst_disp[item.index] = 0.0
                last_tilt[item.index] = 0.0
                is_soft_of[item.index] = bool(item.is_soft)
                mass_of[item.index] = float(item.mass)
                support_kind[item.index] = _classify_support(
                    self.client, item, pos, orn, last_container_holder["container"]
                )
        return ok

    PlacementValidator.place_item = patched_place_item
    PlacementValidator.check_transport_path = patched_check_transport_path
    Item.spawn = patched_spawn

    try:
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

        step_n = 0
        n_placements = 0
        last_status: dict = {}

        while not terminated and not truncated:
            action, _ = runner.call("policy", time_out_sec=task_config["agent"]["policy_timeout"],
                                    fallback=env.action_space.sample(), observation=obs)
            obs, reward, terminated, truncated, info = env.step(action)
            step_n += 1
            status = info.get("status", {})
            last_status = status
            if status.get("is_placed_safe"):
                n_placements += 1

            for c in env.container_manager.containers:
                for it in c.packed_items:
                    if it.index not in ref_pose:
                        continue  # 反実仮想の対象外にした荷物、または未登録（起こらないはず）
                    live_pos, live_orn = it.get_pose(env.client)
                    if live_pos is None:
                        continue
                    ref = ref_pose[it.index]
                    tilt = _tilt_deg(live_orn, ref["orn"])
                    disp = float(np.linalg.norm(np.array(live_pos) - np.array(ref["pos"])))
                    step_delta = tilt - last_tilt[it.index]
                    if abs(step_delta) > tilt_event_deg:
                        events.append({
                            "step": step_n, "item_index": it.index, "is_soft": is_soft_of[it.index],
                            "support_kind": support_kind[it.index], "mass": mass_of[it.index],
                            "tilt_deg": round(tilt, 3), "step_delta_deg": round(step_delta, 3),
                            "disp_m": round(disp, 4),
                        })
                    last_tilt[it.index] = tilt
                    if tilt > worst_tilt[it.index]:
                        worst_tilt[it.index] = tilt
                    if disp > worst_disp[it.index]:
                        worst_disp[it.index] = disp

        # ---- 死因の分離 --------------------------------------------------------
        if last_status.get("is_included") is False:
            term_reason = "inclusion"
        elif last_status.get("is_valid") is False:
            term_reason = "transport"
        elif last_status.get("is_placed_safe") is False:
            term_reason = "tip"
        else:
            term_reason = "stream_empty"

        # ---- 公式評価（bench_run.py と同一の呼び出し経路、恒等性確認用） -----------
        from scripts.bench_run import _compute_cog
        fill_score, out_items = env.evaluator.calculate_fill_rate(env.container_manager.containers)
        num_total = env.num_total_items
        num_placed_items = sum(len(c.packed_items) for c in env.container_manager.containers) / max(1, num_total)
        cog_score = _compute_cog(env, out_items)

        # ---- fill の反実仮想: 全荷物を「配置直後の姿勢」へ戻して再計算 -------------
        items_all = [it for c in env.container_manager.containers for it in c.packed_items]
        real_pose_snapshot = {}
        for it in items_all:
            pos, orn = it.get_pose(env.client)
            if pos is not None and it.pybullet_id is not None:
                real_pose_snapshot[it.pybullet_id] = (pos, orn)
        for it in items_all:
            ref = ref_pose.get(it.index)
            if ref is not None and it.pybullet_id is not None:
                env.client.resetBasePositionAndOrientation(it.pybullet_id, ref["pos"], ref["orn"])
        fill_cf, out_items_cf = env.evaluator.calculate_fill_rate(env.container_manager.containers)
        for it in items_all:
            if it.pybullet_id in real_pose_snapshot:
                pos, orn = real_pose_snapshot[it.pybullet_id]
                env.client.resetBasePositionAndOrientation(it.pybullet_id, pos, orn)

        # ---- 傾き分布の集計 ------------------------------------------------------
        def _hist(values: list[float], edges=(1.0, 3.0, 5.0, 10.0, 45.0)) -> dict:
            return {f"gt_{e:g}deg": sum(1 for v in values if v > e) for e in edges}

        all_worst = list(worst_tilt.values())
        by_soft: dict[str, list[float]] = {"soft": [], "hard": []}
        by_support: dict[str, list[float]] = {"floor": [], "soft": [], "hard": [], "mixed": [], "unknown": []}
        for idx, t in worst_tilt.items():
            by_soft["soft" if is_soft_of.get(idx) else "hard"].append(t)
            by_support.setdefault(support_kind.get(idx, "unknown"), []).append(t)

        def _summ(vals: list[float]) -> dict:
            if not vals:
                return {"n": 0, "mean": 0.0, "max": 0.0}
            return {"n": len(vals), "mean": round(sum(vals) / len(vals), 3), "max": round(max(vals), 3)}

        n_would_be_unsafe = sum(
            1 for idx in worst_tilt
            if worst_tilt[idx] > 45.0 or worst_disp[idx] > 0.3
        )

        result = {
            "config": os.path.basename(config_path), "hard_soft": bool(hard_soft),
            "n_steps": step_n, "n_placements": n_placements, "term_reason": term_reason,
            "final_status": last_status,
            "fill_score": float(fill_score), "num_placed_items": float(num_placed_items),
            "cog_score": float(cog_score),
            "fill_score_counterfactual_no_drift": float(fill_cf),
            "n_out_items": len(out_items), "n_out_items_counterfactual": len(out_items_cf),
            "tilt_hist_all": _hist(all_worst),
            "tilt_by_is_soft": {k: {**_summ(v), **_hist(v)} for k, v in by_soft.items()},
            "tilt_by_support_kind": {k: {**_summ(v), **_hist(v)} for k, v in by_support.items()},
            "n_would_be_unsafe_if_gate_reapplied": n_would_be_unsafe,
            "counterfactual_transport": counterfactual_record,
            "n_tilt_events": len(events),
            "events": events,
        }

        env.close()
        runner.close()
        return result
    finally:
        PlacementValidator.place_item = orig_place_item
        PlacementValidator.check_transport_path = orig_check_transport_path
        Item.spawn = orig_spawn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--module-path", default="agents/heuristic/")
    parser.add_argument("--tilt-event-deg", type=float, default=1.0)
    parser.add_argument("--hard-soft", action="store_true",
                        help="soft品の接触コンプライアンスのみ無効化するアブレーション")
    parser.add_argument("--no-counterfactual", action="store_true",
                        help="搬入経路の反実仮想を無効化（高速化用）")
    args = parser.parse_args()
    result = run(args.config_path, args.module_path, args.tilt_event_deg,
                args.hard_soft, counterfactual=not args.no_counterfactual)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
