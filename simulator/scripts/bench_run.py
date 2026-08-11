"""1タスク・1 agent 設定を実行し、公式 fill_score/num_placed_items と公開定義 cog を
1行の JSON として返す（方策3判定用、Phase 0、findings_2026-07.md §13）。

`src.ground_handling.app.EvaluationApp.run()` と同一の呼出し経路
（`GroundHandlingEnv`/`TimedAgentRunner`、get_init_states→optimize→policy loop→evaluate）を
薄く再現する（`EvaluationApp` 自体は複数タスクを1ファイルにまとめる設計で1タスク単位の
戻り値を返さないため、`scripts/eval_chunked.py` の並列駆動に都合が良い形へ再実装する。
実行ロジック自体は複製するが、スコアリングは公式コードをそのまま呼ぶ）。

fill_score/num_placed_items は `Evaluator.evaluate`（本番同一コード）をそのまま使う。
cog は物理沈降後の真の座標（`Item.get_pose` 経由で設定される `packed_items` 座標、
`MultiContainerManager` が全静止後に1度だけ登録する「着地して安定した真の座標」）を
`state.py::_build_containers`/`_route_placed`（本番同一の座標変換）に通して pos_rel を得て、
公開定義どおりに質量加重・容器外寸正規化で計算する
（docs/findings_2026-07.md §2.1: z_com=Σm·z/Σm, cog=100·(1-(z_com-z_floor)/(z_top-z_floor))）。
`Evaluator.calculate_fill_rate` と同じ inclusion 判定を再利用し、fill に数えられた荷物だけを
cog にも数える（「置いた荷物だけを数える」、findings §2.1）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

import numpy as np


def _compute_cog(env, out_items) -> float:
    """公開定義の cog（質量加重・容器外寸正規化）を、fill と同じ inclusion 判定で計算する。"""
    from agents.heuristic.packing_core.state import _build_containers, _route_placed

    out_ids = {id(it) for it in out_items}
    init = {"container_list": env.container_manager.get_item_info_in_containers()}
    containers = _build_containers(init)

    # _route_placed は observation 形式(container_list[i]["packed_items"])の belongs_to で
    # 振り分ける。get_item_info_in_containers() は既に settled pos/orn(register_pos_orn 経由の
    # 真の座標)を含むのでそのまま observation として使える。
    observation = {"container_list": init["container_list"]}
    placed = _route_placed(observation, containers)

    # out_items(fillのinclusion落ち)を id 一致で除外するため、containers.py 側の Item と
    # PlacedItem を対応づける必要がある。長さ/幅/高さ/mass/is_soft/is_prioritized の完全一致
    # では同一SKU品を誤って除外し得るため、除外はコンテナ単位の出現順で対応させる
    # （get_item_info_in_containers の packed_items 順序 == container.packed_items 順序、
    # containers.py:331 `[item.get_info() for item in container.packed_items]`）。
    excluded_positions: dict[int, set[int]] = {}
    for cidx, container in enumerate(env.container_manager.containers):
        excluded_positions[cidx] = {
            pos for pos, item in enumerate(container.packed_items) if id(item) in out_ids
        }

    num = 0.0
    den = 0.0
    for cidx, plist in placed.items():
        excl = excluded_positions.get(cidx, set())
        space = containers[cidx]
        hz = float(env.container_manager.containers[cidx].height)
        for pos, pi in enumerate(plist):
            if pos in excl:
                continue
            z_rel = 0.5 * (float(pi.aabb_min_rel[2]) + float(pi.aabb_max_rel[2]))
            num += float(pi.weight) * z_rel
            den += float(pi.weight)
    if den <= 0.0:
        return 0.0
    z_com = num / den
    # H は容器の外寸高（複数容器なら最大、既存 order.py::plan_order_forward_sim の
    # _hz 定義と同じ規約に合わせる）。z_floor は container-relative 座標で 0 相当
    # （state.py の world_to_rel は X のみシフトし Z は素通しのため、pos_rel[2] の
    # 原点は build_container_space が定める容器ローカル原点そのもの）。
    hz_all = max((float(c.height) for c in env.container_manager.containers), default=1.0)
    return 100.0 * (1.0 - z_com / hz_all)


def _quality_proxies(env) -> dict:
    """placement_score / soft_item_score のローカル代理を沈降後の実座標から計算する。

    **公式実装はサーバー側にしか無い**ため、README「評価指標」の文言をそのまま式にした
    近似である。README §評価指標:

      > 優先手荷物やソフト貨物が自分以外の属性の手荷物の下敷き(上方向からの接触判定がある)に
      > なっている（優先手荷物(ソフト貨物)の上に優先手荷物(ソフト貨物)を配置しても減点はなく,
      > それ以外の状況の場合に減点）, 優先手荷物用のコンテナがあるのにそうではないコンテナに
      > 優先手荷物を配置した場合減点. 優先手荷物とソフト貨物はそれぞれ独立に評価される.

    に、findings §2.1 の「**未積載も違反に数える**」を加えた3種の違反を数え、
    `100 * (1 - 違反数 / 全該当荷物数)` を返す。「下敷き」の判定は物理接触クエリではなく
    沈降後 AABB の幾何（XY が重なり、相手の底面が自分の天面近傍かつ相手の中心が上）で行う。

    この代理を入れる理由: v43 の本番利得は placement +5.45 / soft +4.20 が駆動因だったのに、
    従来のベンチは fill/np/cog しか出しておらず**その2項を一度も観測していなかった**
    （findings §18.3）。絶対値の一致は期待せず、設定間の**相対比較**にのみ使う。

    Returns:
        {"placement_proxy": float, "soft_proxy": float,
         "n_prio_total": int, "n_prio_viol": int,
         "n_soft_total": int, "n_soft_viol": int}
    """
    containers = env.container_manager.containers
    prio_container_exists = any(bool(getattr(c, "is_prioritized", False)) for c in containers)

    # 沈降後の実 AABB（扁平姿勢前提ではなく、実姿勢の get_pose から軸整列 AABB を作る）。
    placed: list[dict] = []
    for cidx, container in enumerate(containers):
        for it in container.packed_items:
            pos, orn = it.get_pose(env.client)
            if pos is None:
                continue
            half = 0.5 * np.abs(
                np.array(env.client.getMatrixFromQuaternion(orn)).reshape(3, 3)
            ) @ np.array([it.length, it.width, it.height], dtype=np.float64)
            placed.append({
                "cidx": cidx, "lo": np.array(pos) - half, "hi": np.array(pos) + half,
                "is_soft": bool(it.is_soft), "is_prio": bool(it.is_prioritized),
                "cont_prio": bool(getattr(container, "is_prioritized", False)),
            })

    def _covered_by_other_kind(target: dict, kind: str) -> bool:
        """`target` の天面に、属性の異なる荷物が載っているか（上方向からの接触）。"""
        for other in placed:
            if other is target or other["cidx"] != target["cidx"]:
                continue
            if other[kind]:
                continue                       # 同属性の上載せは減点対象外
            # XY が重なり、相手の底面が自分の天面近傍（±2cm）で、相手の中心が上
            if other["lo"][0] >= target["hi"][0] or other["hi"][0] <= target["lo"][0]:
                continue
            if other["lo"][1] >= target["hi"][1] or other["hi"][1] <= target["lo"][1]:
                continue
            if abs(float(other["lo"][2]) - float(target["hi"][2])) > 0.02:
                continue
            if float(other["lo"][2] + other["hi"][2]) <= float(target["lo"][2] + target["hi"][2]):
                continue
            return True
        return False

    out = {}
    for kind, key in (("prio", "is_prio"), ("soft", "is_soft")):
        total = sum(1 for it in env.stream_manager.all_items
                    if bool(getattr(it, "is_prioritized" if kind == "prio" else "is_soft", False)))
        targets = [p for p in placed if p[key]]
        viol = total - len(targets)            # (c) 未積載は違反
        for t in targets:
            bad = _covered_by_other_kind(t, key)
            if kind == "prio" and prio_container_exists and not t["cont_prio"]:
                bad = True                     # (b) 優先容器があるのに別容器へ
            if bad:
                viol += 1
        out[f"n_{kind}_total"] = int(total)
        out[f"n_{kind}_viol"] = int(viol)
        name = "placement_proxy" if kind == "prio" else "soft_proxy"
        out[name] = 100.0 * (1.0 - viol / total) if total else 100.0
    return out


def _run_shake_test(env, *, kick_speed: float = 0.4, n_cycles: int = 3,
                     kick_steps: int = 30, settle_steps: int = 60) -> dict:
    """摩擦仮説（docs/findings_2026-07.md §15、運営QA Q3）検証用のローカル揺らしtest proxy。

    **公式の stability_score（サーバー側のみ実装、README §3.4 概説のみ）を再現するものでは
    ない**。プラットフォームの真の揺らし機構（振幅・周期・軸）は非公開。本関数は「settled 後の
    全荷物へ交互方向の横方向速度キックを与え、静定後の変位を測る」という独自の粗い近似で、
    方向性（摩擦差が変位差として現れるか）のみを見る目的専用。

    Args:
        env: 実行済み（settle_wait_step 済み）の `GroundHandlingEnv`。
        kick_speed: 各サイクルで与える水平速度 [m/s]。
        n_cycles: +X/-X 交互キックの回数。
        kick_steps: 1キック後、次のキックまで自由運動させる物理ステップ数。
        settle_steps: 全サイクル終了後、最終計測前に静定させる物理ステップ数。

    Returns:
        {"n_items": int, "mean_disp": float, "max_disp": float,
         "n_disp_gt_2cm": int, "n_disp_gt_5cm": int, "per_item": [...]}
        （`packed_items` が空なら `n_items=0` で他は0埋め）
    """
    client = env.client
    items = [it for c in env.container_manager.containers for it in c.packed_items]
    if not items:
        return {"n_items": 0, "mean_disp": 0.0, "max_disp": 0.0,
                "n_disp_gt_2cm": 0, "n_disp_gt_5cm": 0, "per_item": []}

    pre_pos = {}
    for it in items:
        pos, orn = it.get_pose(client)
        if pos is None:
            continue
        pre_pos[it.pybullet_id] = pos

    for cycle in range(n_cycles):
        vx = kick_speed if cycle % 2 == 0 else -kick_speed
        for it in items:
            if it.pybullet_id is None or it.pybullet_id not in pre_pos:
                continue
            lin, ang = client.getBaseVelocity(it.pybullet_id)
            client.resetBaseVelocity(it.pybullet_id, linearVelocity=(vx, lin[1], lin[2]),
                                     angularVelocity=ang)
        for _ in range(kick_steps):
            client.stepSimulation()
    for _ in range(settle_steps):
        client.stepSimulation()

    disps = []
    per_item = []
    for it in items:
        if it.pybullet_id is None or it.pybullet_id not in pre_pos:
            continue
        post_pos, post_orn = it.get_pose(client)
        if post_pos is None:
            continue
        p0 = pre_pos[it.pybullet_id]
        d = float(((post_pos[0] - p0[0]) ** 2 + (post_pos[1] - p0[1]) ** 2
                   + (post_pos[2] - p0[2]) ** 2) ** 0.5)
        disps.append(d)
        per_item.append({"index": it.index, "is_soft": bool(it.is_soft), "disp": d})

    n = len(disps)
    return {
        "n_items": n,
        "mean_disp": float(sum(disps) / n) if n else 0.0,
        "max_disp": float(max(disps)) if n else 0.0,
        "n_disp_gt_2cm": sum(1 for d in disps if d > 0.02),
        "n_disp_gt_5cm": sum(1 for d in disps if d > 0.05),
        "per_item": per_item,
    }


def _static_equilibrium_proxy(env) -> dict:
    """支持多角形への重心投影による静的力学平衡の近似（Ramos et al. 2016 方式、簡易版）。

    既存の `_compute_cog`（質量加重平均高さ）・`_run_shake_test`（速度キック変位）とは
    別の力学的原理に基づく安定性 proxy。DRL方策導入計画 Phase 0
    （`docs/HANDOFF.md` 参照）の一環として、cog_score/stability_score のより良い局所推定
    候補として追加した。**公式の cog_score/stability_score を再現するものではない**——
    支持面接触は実際には斜め・多点だが、本関数は軸整列AABBの重なり矩形の外接矩形で近似する
    （既存 `_quality_proxies` と同じ簡略化方針）。荷物は6方向いずれも軸整列姿勢のみを
    取るため（`utils.py::ORNS`）、footprint は常に矩形になり、この近似の誤差は小さい。

    手順:
    1. 沈降後の実座標（`get_pose`）から各荷物のAABBを求める。
    2. 各荷物について、直下で支持している荷物（底面がその荷物の天面に隣接し、XYが重なる）を
       列挙する。支持荷物が無ければ床/棚に直接接地しているとみなし、支持領域は自分自身の
       footprintとする。
    3. 支持荷物が複数ある場合、XY重なり矩形の和集合の外接矩形を支持領域とする
       （凹多角形は外接矩形で近似、簡略化）。
    4. 重み伝播用に「最大重なり面積の支持荷物」を親とする森（forest）を構築し、
       各荷物について「自分 + 自分の上に（直接・間接に）積まれている荷物すべて」の
       質量加重重心（xy）を葉から根へ積算する。
    5. その累積重心のxy座標が支持領域（外接矩形）内に収まっているかを判定し、
       収まっていなければ矩形境界までの符号付き距離（負=違反量）を margin とする。

    Returns:
        {"n_items": int, "margin_mean": float, "margin_mass_weighted_mean": float,
         "margin_min": float, "n_violation": int, "frac_violation": float}
        （荷物0個なら n_items=0 で他は0埋め）
    """
    entries: list[dict] = []
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
                "container": container,
            })

    n = len(entries)
    if n == 0:
        return {"n_items": 0, "margin_mean": 0.0, "margin_mass_weighted_mean": 0.0,
                "margin_min": 0.0, "n_violation": 0, "frac_violation": 0.0}

    z_tol = 0.015
    supports: list[list[int]] = [[] for _ in range(n)]  # supports[i] = [j, ...] が i を支える
    for i, a in enumerate(entries):
        for j, b in enumerate(entries):
            if i == j or a["container"] is not b["container"]:
                continue
            if abs(float(a["lo"][2]) - float(b["hi"][2])) > z_tol:
                continue
            ox = min(a["hi"][0], b["hi"][0]) - max(a["lo"][0], b["lo"][0])
            oy = min(a["hi"][1], b["hi"][1]) - max(a["lo"][1], b["lo"][1])
            if ox > 1e-6 and oy > 1e-6:
                supports[i].append(j)

    def _overlap_rect(i: int, j: int) -> tuple[float, float, float, float]:
        a, b = entries[i], entries[j]
        return (max(a["lo"][0], b["lo"][0]), max(a["lo"][1], b["lo"][1]),
                min(a["hi"][0], b["hi"][0]), min(a["hi"][1], b["hi"][1]))

    parent = [-1] * n            # 重み伝播用（最大重なり面積の支持先）
    support_rect: list[tuple[float, float, float, float]] = [None] * n  # type: ignore[list-item]
    for i in range(n):
        if not supports[i]:
            a = entries[i]
            support_rect[i] = (float(a["lo"][0]), float(a["lo"][1]),
                                float(a["hi"][0]), float(a["hi"][1]))
            continue
        rects = [_overlap_rect(i, j) for j in supports[i]]
        support_rect[i] = (min(r[0] for r in rects), min(r[1] for r in rects),
                            max(r[2] for r in rects), max(r[3] for r in rects))
        best_j, best_area = None, -1.0
        for j, r in zip(supports[i], rects):
            area = max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])
            if area > best_area:
                best_j, best_area = j, area
        parent[i] = best_j if best_j is not None else -1

    children: list[list[int]] = [[] for _ in range(n)]
    for i, p in enumerate(parent):
        if p >= 0:
            children[p].append(i)

    subtree_mass = [0.0] * n
    subtree_com = [np.zeros(2) for _ in range(n)]

    def _accumulate(i: int) -> None:
        m = entries[i]["mass"]
        com = 0.5 * (entries[i]["lo"][:2] + entries[i]["hi"][:2]) * m
        for c in children[i]:
            _accumulate(c)
            m += subtree_mass[c]
            com = com + subtree_com[c] * subtree_mass[c]
        subtree_mass[i] = m
        subtree_com[i] = com / m if m > 0 else com

    roots = [i for i in range(n) if parent[i] < 0]
    for r in roots:
        _accumulate(r)
    # forest外（循環等の異常系、通常は発生しない）を保険で処理
    for i in range(n):
        if subtree_mass[i] == 0.0:
            _accumulate(i)

    margins = []
    weights = []
    for i in range(n):
        x0, y0, x1, y1 = support_rect[i]
        cx, cy = float(subtree_com[i][0]), float(subtree_com[i][1])
        dx = min(cx - x0, x1 - cx)
        dy = min(cy - y0, y1 - cy)
        margins.append(min(dx, dy))
        weights.append(entries[i]["mass"])

    margins_arr = np.array(margins, dtype=np.float64)
    weights_arr = np.array(weights, dtype=np.float64)
    n_violation = int(np.sum(margins_arr < 0.0))
    return {
        "n_items": n,
        "margin_mean": float(np.mean(margins_arr)),
        "margin_mass_weighted_mean": float(np.sum(margins_arr * weights_arr) / np.sum(weights_arr))
                                       if np.sum(weights_arr) > 0 else 0.0,
        "margin_min": float(np.min(margins_arr)),
        "n_violation": n_violation,
        "frac_violation": float(n_violation) / n,
    }


def run_one(config_path: str, module_path: str, agent_module: str, agent_class: str,
            shake_test: bool = False, kick_speed: float = 0.4,
            equilibrium: bool = False) -> dict:
    from src.ground_handling.agent_factory import AgentFactory
    from src.ground_handling.env import GroundHandlingEnv
    from src.ground_handling.runner import TimedAgentRunner

    with open(config_path) as f:
        config = json.load(f)
    task_id, task_config = next(iter(config.items()))

    agent_factory = AgentFactory(module_name=agent_module, class_name=agent_class, module_path=module_path)
    env = None
    runner = None
    t_wall0 = time.time()
    try:
        env = GroundHandlingEnv(config=task_config, verbose=False, render_mode=None)
        env.reset_settings()
        init_states = env.get_init_states()

        allowed_methods = task_config["agent"]["allowed_methods"]
        max_mem = task_config["agent"].get("max_mem", 4)
        runner = TimedAgentRunner(agent_factory=agent_factory, allowed_methods=allowed_methods,
                                   max_mem=max_mem, verbose=False)
        runner.call("get_init_states", time_out_sec=task_config["agent"]["init_timeout"],
                    fallback=None, init_states=init_states)

        optimize_time = 0.0
        if env.optimize:
            item_list = env.get_info_for_optimization()
            optimized_order, opt_elapsed = runner.call(
                "optimize", time_out_sec=task_config["agent"]["optimization_timeout"],
                fallback=list(env.stream_manager.all_indices), item_list=item_list)
            optimize_time = opt_elapsed
            if not env.set_item_order(optimized_order):
                return {"config": os.path.basename(config_path), "status": "format_error",
                        "message": "invalid optimize order"}
        env.reset_item_stream()

        obs, info = env.reset(seed=42)
        terminated = truncated = False
        policy_time = 0.0
        n_validator_ng = 0
        while not terminated and not truncated:
            action, p_elapsed = runner.call("policy", time_out_sec=task_config["agent"]["policy_timeout"],
                                             fallback=env.action_space.sample(), observation=obs)
            policy_time = max(policy_time, p_elapsed)
            obs, reward, terminated, truncated, info = env.step(action)
            status = info.get("status", {})
            if status.get("is_valid") is False or status.get("is_placed_safe") is False:
                n_validator_ng += 1

        fill_score, out_items = env.evaluator.calculate_fill_rate(env.container_manager.containers)
        num_total = env.num_total_items
        num_placed_items = sum(len(c.packed_items) for c in env.container_manager.containers) / max(1, num_total)
        cog_score = _compute_cog(env, out_items)

        result = {
            "config": os.path.basename(config_path), "status": "success",
            "fill_score": float(fill_score), "num_placed_items": float(num_placed_items),
            "cog_score": float(cog_score), "n_validator_ng": int(n_validator_ng),
            **_quality_proxies(env),
            "optimization_time": float(optimize_time), "policy_time": float(policy_time),
            "wall_time": float(time.time() - t_wall0),
        }
        if shake_test:
            result["shake"] = _run_shake_test(env, kick_speed=kick_speed)
        if equilibrium:
            result["equilibrium"] = _static_equilibrium_proxy(env)
        return result
    except Exception as exc:
        return {"config": os.path.basename(config_path), "status": "exec_exception", "message": str(exc)}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if runner is not None:
            try:
                runner.close()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--module-path", default="agents/heuristic/")
    parser.add_argument("--agent-class", default="Agent")
    parser.add_argument("--shake-test", action="store_true",
                        help="settle後に横方向キックを与えて変位を測る揺らしtest proxyを追加実行する")
    parser.add_argument("--kick-speed", type=float, default=0.4,
                        help="揺らしtest proxyの1サイクルあたり横方向速度 [m/s]（既定0.4）")
    parser.add_argument("--equilibrium", action="store_true",
                        help="支持多角形ベースの静的力学平衡proxy（Phase 0）を追加計算する")
    args = parser.parse_args()
    # run_local.py と同一導出（"agents/heuristic/" -> "agents.heuristic.agent"）。
    agent_module_path = ".".join(args.module_path.split("/")) + "agent"
    result = run_one(args.config_path, args.module_path, agent_module_path, args.agent_class,
                     shake_test=args.shake_test, kick_speed=args.kick_speed,
                     equilibrium=args.equilibrium)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
