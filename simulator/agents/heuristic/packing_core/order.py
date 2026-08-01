"""荷物順序最適化（`optimize()` フェーズ、spec §4.14 `order.py` の実装）。

貪欲デコーダは「最初に置けない荷物」で終了するため、積載数 ≒ 荷物の到着順序の関数である。
本モジュールは `optimize()` の予算（自己締切 `TimeParams.optimize_stop`）を使い、
**オフライン貪欲リプレイ**（`_greedy_replay`）を評価器として荷物順序を探索し、より多く積める
順列を返す。探索は決定論的マルチスタート山登り（固定 rng ＋反復上限）で、`_run_pipeline`
（採点 policy）には一切触れず `packing_core` のライブラリ関数のみを再利用する。

公開API:
    optimize_order(items, state0, budget_s, rng) -> list[int]
        全 `item["index"]` の完全順列を builtin int で返す。anytime：常に有効な順列を返し、
        例外を外へ漏らさない（内部 try/except → 複合キー順序へフォールバック）。
    composite_key_order(items) -> list[int]
        現行 `Agent.optimize` と同一の (体積降順→重量降順→入力位置) 並べ替え。seed[0]。
    plan_order_forward_sim(item_list, container_list, budget_s) -> list[int]
        全可視 heightmap 貪欲の前方シミュで順序を「生成」する（v38 の既定経路）。
    search_order_composite(item_list, container_list, budget_s, post_fn, window_k) -> list[int]
        方策3（HF-030）: 忠実 window=`window_k` リプレイ（`_replay_arrival_window`、実
        `decide_placement` 使用）を評価器に、`post_fn` 適用後の順序を composite 目的
        （fill+cog 実効重み）で GRASP シード＋山登り探索する。既定 OFF（`GH_ORDER_SEARCH`）。
"""
from __future__ import annotations

import os

import copy
import functools
import math
import os
import time

import numpy as np

from .candidates import enumerate_candidates, filter_candidates
from .constants import (
    ORDER_MAX_ITERS,
    ORDER_STAGNANT_RESTART,
    ORDER_TIME_RESERVE_S,
    BEAM_W,
    SIM_EMS_TOPN,
    SIM_LPATH_TOP_M,
    ORDER_SEED_CAP,
    PlacementParams,
    ProvisionalRiskParams,
    ScoreParams,
    StageParams,
    TimeParams,
)
from .container_space import bake_placed
from .ems import generate_ems, select_topn, update_ems
from .geometry import aabb_from_center, oriented_size
from .risk import provisional_p_ng
from .score import heuristic_score
from .stability import cg_margin, expected_settled_pos_rel, support_ratio
from .state import PackingState, rel_to_world
from .types import EMSBox, ItemSpec, PlacedItem
from .watchdog import (
    StepBudget,
    layer1_main,
    layer2_dblf_strict,
    layer3_first_fit,
    layer4_max_p,
    safe_decide,
)

# agent.py の特徴ループと同値（未支持率減点の係数）。順序探索でも同じ選好を再現する。
_W_LOW_SUPPORT = 1.0
_IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
# 巨大予算の StepBudget（now_fn 固定）＝時間打切りなし・決定論。探索リプレイ用。
_INF = 1.0e18


# --- 荷物メタ抽出・シード生成 -------------------------------------------------------

def _item_index(item: dict, position: int) -> int:
    """`item["index"]` を builtin int で返す（検証は Agent.optimize 側で済んでいる前提）。"""
    return int(item["index"])


_NUMERIC_TYPES = (int, float, np.integer, np.floating)


def _item_metrics(items: list[dict]) -> list[dict]:
    """各荷物の並べ替え用メタ（index/vol/base/maxdim/height/mass/soft/prio）を返す。

    並べ替え材料（length/width/height/mass）の検証は現行 `Agent.optimize`（§4.12 v1.23）と
    厳密に一致させる：bool 拒否・数値型必須・有限必須・寸法正/質量非負・体積有限。いずれか
    不正なら ValueError を送出する（`composite_key_order` が捕捉して入力位置順へフォールバック）。
    """
    out: list[dict] = []
    for position, item in enumerate(items):
        values = []
        for key in ("length", "width", "height", "mass"):
            raw = item[key]
            if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, _NUMERIC_TYPES):
                raise ValueError(f"{key} must be numeric")
            v = float(raw)
            if not math.isfinite(v):
                raise ValueError(f"{key} must be finite")
            values.append(v)
        length, width, height, mass = values
        if length <= 0.0 or width <= 0.0 or height <= 0.0 or mass < 0.0:
            raise ValueError("dimensions must be positive and mass must be nonnegative")
        volume = length * width * height
        if not math.isfinite(volume):
            raise ValueError("volume must be finite")
        out.append({
            "index": _item_index(item, position),
            "pos": position,
            "vol": volume,
            "base": length * width,
            "maxdim": max(length, width, height),
            "height": height,
            "mass": mass,
            "soft": bool(item.get("is_soft", False)),
            "prio": bool(item.get("is_prioritized", False)),
        })
    return out


def composite_key_order(items: list[dict]) -> list[int]:
    """(体積降順 → 重量降順 → 入力位置) で公式 index の完全順列を返す（現行 Agent.optimize と同一）。

    不正な並べ替え材料があれば入力位置順（＝入力の index 列）へフォールバックする。
    """
    indices = [int(item["index"]) for item in items]
    try:
        metrics = _item_metrics(items)
    except Exception:
        return indices
    order_positions = sorted(
        range(len(metrics)),
        key=lambda i: (-metrics[i]["vol"], -metrics[i]["mass"], metrics[i]["pos"]),
    )
    return [metrics[i]["index"] for i in order_positions]


# HF-022 診断用: 直近の plan_order_forward_sim の内訳（scripts/diag_termination.py が読む）。
# 実配置経路からは参照されないため挙動に影響しない。
ORDER_DIAG: dict = {}


def _envi(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _envf_o(name: str, default: float) -> float:
    """order 内で使う env float 読み出し（constants を汚さない実験用）。"""
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def plan_order_forward_sim(item_list: list[dict], container_list: list, budget_s: float) -> list[int]:
    """heightmap 貪欲を全 lookahead で前方シミュし、実配置エンジン自身が全情報下で置く順を
    計画して公式 index 列で返す（HF-013/H3）。optimize=true 課題の順序を固定ヒューリスティック順
    より fill 目的で優れさせる。物理沈降なし・軸整列AABBで高速。予算超過/例外/不完全は
    composite_key_order へフォールバック（全 index を過不足なく1回ずつ返す契約を厳守）。
    """
    import time

    import numpy as np

    from .geometry import oriented_size
    from .heightmap import decide_placement, decide_topk
    from .state import build_state, rel_to_world
    from .types import PlacedItem
    from .watchdog import StepBudget

    fallback = composite_key_order(item_list)
    try:
        n = len(item_list)
        if n == 0 or not container_list:
            return fallback
        official = [int(it["index"]) for it in item_list]  # enum idx -> official index
        t0 = time.monotonic()
        budget = StepBudget(t0=t0, soft=max(1.0, budget_s), hard=max(2.0, budget_s + 5.0))
        ident = np.array([0.0, 0.0, 0.0, 1.0])

        def _fresh_state():
            cs = [{**dict(c), "packed_items": []} for c in container_list]
            o = {"optimize": True, "lookahead_k": n,
                 "depth_map": np.zeros((1, 1, 1), dtype=np.float32),
                 "container_list": cs, "pool_list": list(item_list)}
            return build_state(o, {"optimize": True, "lookahead_k": n, "container_list": cs},
                               compute_ems=False)

        # HF-028: 順序探索の目的関数。既定 "vol"（= 従来の置いた体積＝fill 目的）。
        # "score" にすると実効重み最大の2項を同時に最大化する:
        #     J = W_FILL·(置いた体積/容器体積) + W_COG·(1 − z_com/H)
        # z_com は**質量加重**の重心高（公式 cog の定義そのまま。
        # analysis/platform_score.py::compute_cog）。H は容器の外寸高。
        # ctr_mean だけを最小化すると「床層だけ置いて終わる」のが最適解になるので、
        # 体積項と必ず同時に扱う。重みは提出22件から推定した実効重み
        # （fill 0.268 / cog 0.261、ground-handling-effective-weights）。
        obj_mode = os.environ.get("GH_ORDER_OBJ", "vol")
        w_fill = _envf_o("GH_ORDER_W_FILL", 0.268)
        w_cog = _envf_o("GH_ORDER_W_COG", 0.261)
        _cv = 0.0
        for _c in container_list:
            _t = float(_c["thickness"])
            _cv += ((float(_c["length"]) - 2 * _t) * (float(_c["width"]) - 2 * _t)
                    * (float(_c["height"]) - 2 * _t))
        _cv = max(_cv, 1e-9)
        _hz = max((float(_c["height"]) for _c in container_list), default=1.0)

        def _replay(rng, rcl: int) -> tuple[list[int], float]:
            """1回ぶんの前方シミュ。rng が None なら決定論（= v14 と完全一致）。

            rng ありのときは各手で feasible 上位 rcl 件（`decide_topk`、荷物ごとに最良1件）から
            一様抽選する（GRASP の制限候補リスト）。戻り: (置いた enum idx 列, 置いた体積)。
            """
            st = _fresh_state()
            placed: list[int] = []
            vol_sum = 0.0
            m_sum = 0.0        # Σ mass
            mz_sum = 0.0       # Σ mass·z（容器底からの高さ。pos_rel[2] がそれ）
            while st.pool:
                if budget.over_soft():
                    break
                step = StepBudget(t0=time.monotonic(), soft=6.0, hard=8.0)
                if rng is None:
                    res = decide_placement(st, step)
                else:
                    opts = decide_topk(st, step, rcl)
                    res = opts[int(rng.integers(len(opts)))] if opts else None
                if res is None:
                    break
                item_idx, cidx, pos_rel, orn = res
                item = next((it for it in st.pool if it.idx == item_idx), None)
                if item is None:
                    break
                osize = np.asarray(oriented_size(item.size, orn), dtype=np.float64)
                half = osize / 2.0
                space = st.containers[cidx]
                pi = PlacedItem(
                    pos_world=rel_to_world(pos_rel, space), orn_quat=ident,
                    size=np.asarray(item.size, dtype=np.float64), weight=item.weight,
                    is_soft=item.is_soft, is_priority=item.is_priority,
                    aabb_min_rel=pos_rel - half, aabb_max_rel=pos_rel + half,
                )
                st.placed.setdefault(cidx, []).append(pi)
                st.pool = [it for it in st.pool if it.idx != item_idx]
                placed.append(int(item_idx))
                vol_sum += float(np.prod(np.asarray(item.size, dtype=np.float64)))
                m_sum += float(item.weight)
                mz_sum += float(item.weight) * float(pos_rel[2])
            if obj_mode != "score":
                return placed, vol_sum
            # J = W_FILL·体積率 + W_COG·(1 − z_com/H)。cog は高いほど良い＝重心が低いほど良い。
            z_com = (mz_sum / m_sum) if m_sum > 0.0 else 0.0
            j = w_fill * (vol_sum / _cv) * 100.0 + w_cog * (1.0 - z_com / _hz) * 100.0
            return placed, j

        # 決定論リプレイ（= v14）。GRASP はこれを下回れない（常に best の初期値）。
        placed_enum, best_vol = _replay(None, 1)
        n_restarts = _envi("GH_ORDER_GRASP", 0)
        rcl = max(2, _envi("GH_ORDER_GRASP_RCL", 3))
        n_tried = 0
        if n_restarts > 0:
            # HF-022: 実測で貪欲は「全荷物可視・時間切れなし」でも 59/80 でしか置けず、実機との
            # 乖離は 2-4個だった＝壁は貪欲方策そのもの。同じ代理モデル上で乱択リプレイを複数回
            # 走らせ、置いた体積が最大のものを採る（GRASP）。fill が目的なので個数でなく体積で選ぶ
            # （HF-011/v8 の「個数最適化が fill を下げた」を踏まえる）。
            rng = np.random.default_rng(12345)
            for _ in range(n_restarts):
                if budget.over_soft():
                    break
                cand, vol = _replay(rng, rcl)
                n_tried += 1
                if vol > best_vol:
                    placed_enum, best_vol = cand, vol
        # HF-022 診断: サロゲートが何個置けたかを記録する（optimize 1回につき1度、実質無コスト）。
        # 実機の配置数と比べることで「順序器が天井」か「サロゲート精度の問題」かを分ける。
        ORDER_DIAG.update({"n_items": n, "n_placed_surrogate": len(placed_enum),
                           "placed_volume": best_vol, "grasp_restarts": n_tried,
                           "budget_hit": bool(budget.over_soft()),
                           "elapsed_s": time.monotonic() - t0})
        placed_official = [official[e] for e in placed_enum if 0 <= e < n]
        seen = set(placed_official)
        tail = [i for i in fallback if i not in seen]  # 未配置は composite 順で後置
        order = placed_official + tail
        if sorted(order) == sorted(official) and len(order) == n:
            return order
        return fallback
    except Exception:
        return fallback


def plan_order_beam(item_list: list[dict], container_list: list, budget_s: float,
                    beam_width: int = 3, topk: int = 2) -> list[int]:
    """ビーム探索で配置順を計画（非貪欲, HF-013/H3+）。各深さで各ビームの上位 topk 手を展開し、
    累積配置体積で上位 beam_width を残す。貪欲(plan_order_forward_sim)より blocking を避けた順を狙う。
    optimize=true(課題A) 向け。予算超過/例外/不完全は greedy へフォールバック（全 index を厳守）。
    """
    import dataclasses
    import time

    import numpy as np

    from .geometry import oriented_size
    from .heightmap import decide_topk
    from .state import build_state, rel_to_world
    from .types import PlacedItem
    from .watchdog import StepBudget

    greedy = plan_order_forward_sim(item_list, container_list, budget_s)
    try:
        n = len(item_list)
        if n == 0 or not container_list:
            return greedy
        conts = [{**dict(c), "packed_items": []} for c in container_list]
        init = {"optimize": True, "lookahead_k": n, "container_list": conts}
        obs = {"optimize": True, "lookahead_k": n, "depth_map": np.zeros((1, 1, 1), dtype=np.float32),
               "container_list": conts, "pool_list": list(item_list)}
        state0 = build_state(obs, init, compute_ems=False)
        official = [int(it["index"]) for it in item_list]
        ident = np.array([0.0, 0.0, 0.0, 1.0])
        t0 = time.monotonic()
        deadline = t0 + max(2.0, budget_s)

        def _apply(st, item_idx, cidx, pos_rel, orn):
            item = next((it for it in st.pool if it.idx == item_idx), None)
            if item is None:
                return None, 0.0
            osize = np.asarray(oriented_size(item.size, orn), dtype=np.float64)
            half = osize / 2.0
            space = st.containers[cidx]
            pi = PlacedItem(pos_world=rel_to_world(pos_rel, space), orn_quat=ident,
                            size=np.asarray(item.size, dtype=np.float64), weight=item.weight,
                            is_soft=item.is_soft, is_priority=item.is_priority,
                            aabb_min_rel=pos_rel - half, aabb_max_rel=pos_rel + half)
            placed = {c: list(v) for c, v in st.placed.items()}
            placed.setdefault(cidx, []).append(pi)
            pool = [it for it in st.pool if it.idx != item_idx]
            child = dataclasses.replace(st, placed=placed, pool=pool)
            return child, float(np.prod(osize))

        beams = [(0.0, [], state0)]   # (cum_volume, order_enum, state)
        best_vol, best_order = -1.0, []
        for _depth in range(n):
            if time.monotonic() > deadline:
                break
            expansions = []
            for vol, order_enum, st in beams:
                cands = decide_topk(st, StepBudget(t0=time.monotonic(), soft=6.0, hard=8.0), topk)
                if not cands:
                    if vol > best_vol:
                        best_vol, best_order = vol, order_enum
                    continue
                for (item_idx, cidx, pos_rel, orn) in cands:
                    child, ivol = _apply(st, item_idx, cidx, pos_rel, orn)
                    if child is not None:
                        expansions.append((vol + ivol, order_enum + [item_idx], child))
            if not expansions:
                break
            expansions.sort(key=lambda e: -e[0])
            beams = expansions[:beam_width]
            if beams[0][0] > best_vol:
                best_vol, best_order = beams[0][0], beams[0][1]
        for vol, order_enum, st in beams:
            if vol > best_vol:
                best_vol, best_order = vol, order_enum
        placed_official = [official[e] for e in best_order if 0 <= e < n]
        seen = set(placed_official)
        order = placed_official + [i for i in greedy if i not in seen]
        if sorted(order) == sorted(official) and len(order) == n:
            return order
        return greedy
    except Exception:
        return greedy


# --- 忠実 window リプレイ探索（方策3, HF-030）------------------------------------------
#
# `plan_order_forward_sim` は全可視(pool=全件)で heightmap 自身に選ばせて順序を「生成」する
# 機構であり、入力 `item_list` の並びを評価対象として使わない（`_build_models_ranked` が
# pool を体積降順に再ソートするため、入力順は同一体積の SKU 内タイブレークにしか影響しない）。
# よって「候補順序を近傍操作で探索する」用途には、実 policy() と同じ look_ahead=`window_k`
# （課題A は 1）で `order` を厳密に逐次消費する評価器が要る。実測: 41品目/2容器で
# window=1 の全リプレイは約2秒（plan_order_forward_sim の1リプレイ約8〜22秒の1/4〜1/10）。
# 現行 heightmap エンジン（`decide_placement`、`ContainerHeightModel` は毎回 `placed` から
# 高さを再導出し `space.height` を読まない）を使うので `containers` は複数回のリプレイ間で
# 安全に共有できる（読み取り専用）。


def _fresh_search_containers(container_list: list, window_k: int):
    """探索用の空コンテナ一覧と空 placed dict を1度だけ構築する（複数リプレイで共有）。"""
    from .state import build_state  # 局所import（plan_order_forward_sim と同じ慣例）

    cs = [{**dict(c), "packed_items": []} for c in container_list]
    obs = {"optimize": True, "lookahead_k": window_k,
           "depth_map": np.zeros((1, 1, 1), dtype=np.float32),
           "container_list": cs, "pool_list": []}
    init = {"optimize": True, "lookahead_k": window_k, "container_list": cs}
    st = build_state(obs, init, compute_ems=False)
    return st.containers, {c: [] for c in range(len(st.containers))}


def _replay_arrival_window(
    order: list[int], containers, specs: dict[int, dict], window_k: int, outer_budget,
) -> dict[int, list[PlacedItem]]:
    """`order`（公式index列）を look_ahead=`window_k` の可視窓で逐次消費し、現行
    `decide_placement`（heightmap）で貪欲配置する忠実リプレイ。実 policy() の消費過程
    （課題A は window_k=1）をそのまま模す評価器。EMS は使わない（heightmap は state.ems
    を参照しない、HF-012）。`outer_budget.over_soft()` で anytime 停止する。

    Returns:
        dict[int, list[PlacedItem]]: container_idx → 配置済み荷物一覧（空コンテナ開始）。
    """
    from .heightmap import decide_placement  # 局所import（plan_order_forward_sim と同じ慣例）

    n_cont = len(containers)
    placed: dict[int, list[PlacedItem]] = {c: [] for c in range(n_cont)}
    arrival = list(order)
    pos = 0
    window: list[int] = []
    meta = {"optimize": True, "lookahead_k": window_k}
    while True:
        if outer_budget.over_soft():
            break
        while len(window) < window_k and pos < len(arrival):
            window.append(arrival[pos])
            pos += 1
        if not window:
            break
        pool = []
        specs_ok = True
        for j, official_idx in enumerate(window):
            spec_j = specs.get(official_idx)
            if spec_j is None:
                specs_ok = False
                break
            pool.append(ItemSpec(
                idx=j, size=spec_j["size"], weight=spec_j["weight"], kind=None,
                is_soft=spec_j["is_soft"], is_priority=spec_j["is_priority"],
            ))
        if not specs_ok:
            break
        state = PackingState(
            containers=containers, placed=placed, pool=pool,
            ems={}, ems_truncation={}, meta=meta,
        )
        step_budget = StepBudget(t0=time.monotonic(), soft=6.0, hard=8.0)
        res = decide_placement(state, step_budget)
        if res is None:
            break
        j, cidx, pos_rel, orn = res
        if j < 0 or j >= len(window):  # 防御: pool 位置は 0..len(window)-1
            break
        official_idx = window[j]
        spec = specs[official_idx]
        osize = np.asarray(oriented_size(spec["size"], orn), dtype=np.float64)
        half = osize / 2.0
        space = containers[cidx]
        placed[cidx].append(PlacedItem(
            pos_world=rel_to_world(pos_rel, space), orn_quat=_IDENTITY_QUAT,
            size=spec["size"], weight=spec["weight"],
            is_soft=spec["is_soft"], is_priority=spec["is_priority"],
            aabb_min_rel=pos_rel - half, aabb_max_rel=pos_rel + half,
        ))
        window.pop(j)
    return placed


def search_order_composite(
    item_list: list[dict], container_list: list, budget_s: float,
    post_fn=None, window_k: int = 1,
) -> list[int]:
    """post_fn 適用後の順序を composite 目的で最大化する順序を、忠実 window リプレイ
    （`_replay_arrival_window`）を評価器にした GRASP シード＋決定論的マルチスタート
    山登りで探索する（方策3、HF-030）。

        J = W_FILL·(置いた体積/容器体積)·100 + W_COG·(1 − z_com/H)·100

    `plan_order_forward_sim`（fill 目的の貪欲順序生成）と実効重み・z_com 定義は共通
    （`GH_ORDER_W_FILL`/`GH_ORDER_W_COG`、既定 0.268/0.261）。**非退行フロア**:
    seed に `plan_order_forward_sim` の出力（v38 base）を必ず含み、探索はこれを
    下回る順序を採用しない（best の初期値）。

    `post_fn` は最終提出直前に適用されるレバー再並べ替え（例: `agent.py::_post`）。
    レバーが結果を大きく変える（TALL_SHIFT 等）ため、**J は `post_fn(order)` で評価する**
    （post_fn 適用前の順序を評価すると実際に提出される順序とズレる）。post_fn が
    None なら恒等関数を使う。

    Args:
        item_list: `agent.optimize()` の item_list。
        container_list: `self.container_list`（init_states 由来のコンテナ形状一覧）。
        budget_s: 全体時間予算 [s]。
        post_fn: 評価直前に適用する順序後処理（既定 None=恒等）。
        window_k: 忠実リプレイの可視窓幅（実配置と揃える。課題A は既定 1）。

    Returns:
        list[int]: 全 index を過不足なく1回ずつ含む順列（post_fn 適用前の base）。
        呼び出し側が `post_fn` を再適用してから提出することを前提とする
        （agent.py: `_post(search_order_composite(...))`）。
    """
    identity = (lambda order: order) if post_fn is None else post_fn
    fallback = composite_key_order(item_list)
    try:
        n = len(item_list)
        if n == 0 or not container_list:
            return fallback
        specs = _specs_by_index(item_list)
        metrics = _item_metrics(item_list)
        seeds = _build_seed_orders(metrics)
        if not seeds:
            return fallback

        start = time.monotonic()
        deadline = start + max(0.0, float(budget_s) - ORDER_TIME_RESERVE_S)

        # 非退行フロア: 現行 v38 の base（plan_order_forward_sim）を必ずシードへ含める。
        # 予算の一部（最大15%）だけ使う。以降の忠実探索がこれを一度も上回れなくても
        # best の初期値として残る。
        greedy_budget = max(5.0, min(budget_s * 0.15, deadline - time.monotonic()))
        greedy_base = plan_order_forward_sim(item_list, container_list, greedy_budget)
        if greedy_base not in seeds:
            seeds = [greedy_base] + seeds

        w_fill = _envf_o("GH_ORDER_W_FILL", 0.268)
        w_cog = _envf_o("GH_ORDER_W_COG", 0.261)
        _cv = 0.0
        for _c in container_list:
            _t = float(_c["thickness"])
            _cv += ((float(_c["length"]) - 2 * _t) * (float(_c["width"]) - 2 * _t)
                    * (float(_c["height"]) - 2 * _t))
        _cv = max(_cv, 1e-9)
        _hz = max((float(_c["height"]) for _c in container_list), default=1.0)

        containers0, _ = _fresh_search_containers(container_list, window_k)
        outer_budget = StepBudget(t0=start, soft=max(0.0, budget_s - ORDER_TIME_RESERVE_S),
                                   hard=max(1.0, budget_s))

        def _eval(order: list[int]) -> float:
            final_order = identity(order)
            placed = _replay_arrival_window(final_order, containers0, specs, window_k, outer_budget)
            fill, cog = _fill_and_cog(placed, containers0)
            return w_fill * fill * 100.0 + w_cog * (1.0 - cog / _hz) * 100.0

        # 1) 全シード評価（少なくとも seed[0]=greedy_base は必ず評価）。締切超過で打ち切り。
        evaluated = []
        for i, order in enumerate(seeds):
            evaluated.append((order, _eval(order)))
            if i > 0 and time.monotonic() >= deadline:
                break
        if not evaluated:
            evaluated = [(greedy_base, _eval(greedy_base))]
        evaluated.sort(key=lambda t: t[1], reverse=True)
        best_order, best_j = evaluated[0]
        anchors = evaluated[: max(1, BEAM_W)]

        # 2) 決定論的マルチスタート best-improvement 山登り（固定シード rng）。
        rng = np.random.default_rng(20260801)
        active = 0
        current_order, current_j = anchors[active]
        stagnant = 0
        for it in range(ORDER_MAX_ITERS):
            if time.monotonic() >= deadline:
                break
            neighbor = _apply_move(current_order, it % 3, rng)
            j = _eval(neighbor)
            if j > current_j:
                current_order, current_j = neighbor, j
                stagnant = 0
                if j > best_j:
                    best_order, best_j = neighbor, j
            else:
                stagnant += 1
                if stagnant >= ORDER_STAGNANT_RESTART and len(anchors) > 1:
                    active = (active + 1) % len(anchors)
                    current_order, current_j = anchors[active]
                    stagnant = 0

        # 防御: 完全順列であることを保証（万一崩れていれば fallback）。
        if sorted(best_order) != sorted(fallback):
            return fallback
        return [int(x) for x in best_order]
    except Exception:
        return fallback


def _sorted_indices(metrics: list[dict], key) -> list[int]:
    return [m["index"] for m in sorted(metrics, key=key)]


def _build_seed_orders(metrics: list[dict]) -> list[list[int]]:
    """~8–12 の構造化シード順序を返す（先頭＝複合キー）。index tuple で重複除去、`ORDER_SEED_CAP` 上限。

    soft は上に積めない（列の蓋）ため後置、priority は専用容器有無が不明なので通常貨物の後ろへ
    退避（保守側）。属性ブラインドな純サイズ順は参考実装で最弱だったため、全シードが soft/prio
    構造を織り込む。
    """
    seeds: list[list[int]] = []
    # 0: 複合キー（体積降順→重量降順）＝現行 baseline。必ず先頭。
    seeds.append(_sorted_indices(metrics, lambda m: (-m["vol"], -m["mass"], m["pos"])))
    # 1: soft後置・体積降順
    seeds.append(_sorted_indices(metrics, lambda m: (m["soft"], -m["vol"], m["pos"])))
    # 2: soft後置・重量降順→体積降順（低重心の土台先行）
    seeds.append(_sorted_indices(metrics, lambda m: (m["soft"], -m["mass"], -m["vol"], m["pos"])))
    # 3: soft後置・底面積降順→重量降順（広い床先行）
    seeds.append(_sorted_indices(metrics, lambda m: (m["soft"], -m["base"], -m["mass"], m["pos"])))
    # 4: soft後置・最長辺降順（扱いにくい長物を空いている内に）
    seeds.append(_sorted_indices(metrics, lambda m: (m["soft"], -m["maxdim"], m["pos"])))
    # 5: protected(soft|prio)後置・体積降順
    seeds.append(_sorted_indices(metrics, lambda m: (m["soft"] or m["prio"], -m["vol"], m["pos"])))
    # 6: 属性3層（通常→優先→soft）・各層内は重量降順→体積降順
    seeds.append(_sorted_indices(
        metrics, lambda m: (2 if m["soft"] else (1 if m["prio"] else 0), -m["mass"], -m["vol"], m["pos"])
    ))
    # 7: soft後置・高さ昇順→底面積降順（平たい層を作る）
    seeds.append(_sorted_indices(metrics, lambda m: (m["soft"], m["height"], -m["base"], m["pos"])))
    # 8: priority先行（優先貨物を前へ）・soft後置・体積降順
    seeds.append(_sorted_indices(metrics, lambda m: (not m["prio"], m["soft"], -m["vol"], m["pos"])))

    # index tuple で重複除去（順序維持）
    seen: set[tuple] = set()
    unique: list[list[int]] = []
    for order in seeds:
        key = tuple(order)
        if key not in seen:
            seen.add(key)
            unique.append(order)
        if len(unique) >= ORDER_SEED_CAP:
            break
    return unique


def _specs_by_index(items: list[dict]) -> dict[int, dict]:
    """index → {size, weight, is_soft, is_priority}（PlacedItem/ItemSpec 構築用）。"""
    out: dict[int, dict] = {}
    for item in items:
        idx = int(item["index"])
        out[idx] = {
            "size": np.array(
                [float(item["length"]), float(item["width"]), float(item["height"])],
                dtype=np.float64,
            ),
            "weight": float(item["mass"]),
            "is_soft": bool(item.get("is_soft", False)),
            "is_priority": bool(item.get("is_prioritized", False)),
        }
    return out


# --- オフライン貪欲リプレイ（評価器 = fast_env 貪欲リプレイ, §4.14） --------------------

class _SimParams:
    """探索/確定リプレイで共有するパラメータ束。"""

    def __init__(self, *, support_only: bool, ems_cap: int, stage: StageParams) -> None:
        self.pp = PlacementParams()
        self.tp = TimeParams()
        self.score = ScoreParams()
        self.risk = ProvisionalRiskParams()
        self.stage = stage
        self.support_only = support_only
        self.ems_cap = ems_cap


def _decide_once(state: PackingState, sp: _SimParams):
    """1ステップの貪欲判定を再現する（agent._run_pipeline の候補列挙〜safe_decide と同経路）。

    Returns:
        (decided: Candidate | None, path_ok: bool)。path_ok = decided が L_PATH 検証済み
        （`pools.path_candidates` に含まれる）かどうか。
    """
    budget = StepBudget(t0=0.0, soft=_INF, hard=_INF, now_fn=lambda: 0.0)
    raw = enumerate_candidates(state, sp.pp, sp.tp, budget)
    if not raw:
        return None, False
    pools = filter_candidates(state, raw, sp.pp, sp.tp, budget)
    if not pools.geo_candidates:
        return None, False
    for cand in pools.geo_candidates:
        sr = support_ratio(state, cand)
        margin = 1.0 if sp.support_only else cg_margin(state, cand)
        png = provisional_p_ng(support_ratio=sr, cg_margin=margin, params=sp.risk)
        cand.p_success = float(np.clip(1.0 - png, 0.0, 1.0))
        cand.score = heuristic_score(state, cand, sp.score) - _W_LOW_SUPPORT * (1.0 - float(sr))
        cand.features["support_ratio"] = float(sr)
        cand.features["cg_margin"] = float(margin)
        cand.features["provisional_p_ng"] = float(png)
    layers = [
        functools.partial(layer1_main, pools=pools, pp=sp.pp, stage_params=sp.stage),
        functools.partial(layer2_dblf_strict, pools=pools, pp=sp.pp, stage_params=sp.stage, prefer_stable=True),
        functools.partial(layer3_first_fit, pools=pools, pp=sp.pp, stage_params=sp.stage, prefer_stable=True),
        functools.partial(layer4_max_p, pools=pools, pp=sp.pp, stage_params=sp.stage),
    ]
    decided = safe_decide(layers, state, budget, {})
    if decided is None:
        return None, False
    path_ok = any(decided is c for c in pools.path_candidates)
    return decided, path_ok


def _placed_aabbs(placed: list[PlacedItem]) -> list[tuple]:
    return [(it.aabb_min_rel, it.aabb_max_rel) for it in placed]


def _greedy_replay(order: list[int], containers, initial_placed, meta, specs, sp: _SimParams):
    """順序 `order` を貪欲配置でリプレイし、(placed_count, fill, cog) を返す（物理なし）。

    HF-011 v2: `ItemStreamManager` を模した **look_ahead ウィンドウ**でリプレイする。`order` は
    到着順で、可視窓（サイズ `k = meta["lookahead_k"]`）に先頭から補充する。各ステップは窓内の
    全荷物を pool として `_decide_once` に渡し（policy と同じく safe_decide が最良を選ぶ）、
    選ばれた pool 位置の荷物を消費する（`decided.item_idx`）。`k=1` なら到着順＝消費順（v1相当）。

    成功判定 = decided が L_PATH 検証済み（layer1/2）。layer3/4 の縮退選択は不成功扱い。最初の
    不成功（窓内のどれも置けない）で停止する。各ステップは `bake_placed`＋`update_ems` で状態を
    更新する（`build_state` と同義の全再構築セマンティクス）。
    """
    n_cont = len(containers)
    placed = {c: list(initial_placed.get(c, [])) for c in range(n_cont)}
    ems_full: dict[int, list[EMSBox]] = {}
    for c in range(n_cont):
        bake_placed(containers[c], placed[c])
        ems_full[c] = generate_ems(containers[c], _placed_aabbs(placed[c]))

    k = max(1, int(meta.get("lookahead_k", 1) or 1))
    arrival = list(order)
    pos = 0
    window: list[int] = []
    placed_count = 0
    while True:
        while len(window) < k and pos < len(arrival):
            window.append(arrival[pos])
            pos += 1
        if not window:
            break
        pool = []
        specs_ok = True
        for j, wi in enumerate(window):
            spec_j = specs.get(wi)
            if spec_j is None:
                specs_ok = False
                break
            pool.append(ItemSpec(
                idx=j, size=spec_j["size"], weight=spec_j["weight"], kind=None,
                is_soft=spec_j["is_soft"], is_priority=spec_j["is_priority"],
            ))
        if not specs_ok:
            break
        ems: dict[int, list[EMSBox]] = {}
        ems_trunc: dict[int, float] = {}
        for c in range(n_cont):
            sel, trunc = select_topn(ems_full[c], sp.stage.ems_top_n_per_container)
            if sp.ems_cap > 0 and len(sel) > sp.ems_cap:
                sel = sel[: sp.ems_cap]
            ems[c] = sel
            ems_trunc[c] = trunc
        state = PackingState(
            containers=containers, placed=placed, pool=pool,
            ems=ems, ems_truncation=ems_trunc, meta=meta,
        )
        decided, path_ok = _decide_once(state, sp)
        if decided is None or not path_ok:
            break
        j = int(decided.item_idx)
        if j < 0 or j >= len(window):  # 防御: pool 位置は 0..len(window)-1
            break
        item_index = window[j]
        spec = specs[item_index]
        cidx = int(decided.container_idx)
        settled = expected_settled_pos_rel(state, decided)
        amin, amax = aabb_from_center(settled, np.asarray(decided.osize, dtype=np.float64))
        placed_item = PlacedItem(
            pos_world=rel_to_world(settled, containers[cidx]),
            orn_quat=_IDENTITY_QUAT,
            size=spec["size"], weight=spec["weight"],
            is_soft=spec["is_soft"], is_priority=spec["is_priority"],
            aabb_min_rel=amin, aabb_max_rel=amax,
        )
        placed[cidx].append(placed_item)
        placed_count += 1
        bake_placed(containers[cidx], placed[cidx])
        ems_full[cidx] = update_ems(ems_full[cidx], (amin, amax), containers[cidx])
        window.pop(j)

    fill, cog = _fill_and_cog(placed, containers)
    return placed_count, fill, cog


def _fill_and_cog(placed: dict[int, list[PlacedItem]], containers) -> tuple[float, float]:
    """充填率（配置荷物体積 / 内寸箱体積合計）と重量加重 cog（相対Z）を返す（tie-break 用）。"""
    total_item_vol = 0.0
    num = 0.0
    den = 0.0
    for c, items in placed.items():
        for it in items:
            total_item_vol += float(np.prod(np.asarray(it.size, dtype=np.float64)))
            cz = 0.5 * (float(it.aabb_min_rel[2]) + float(it.aabb_max_rel[2]))
            num += float(it.weight) * cz
            den += float(it.weight)
    total_cont_vol = 0.0
    for space in containers:
        span = np.asarray(space.inner_max_rel, dtype=np.float64) - np.asarray(space.inner_min_rel, dtype=np.float64)
        total_cont_vol += float(np.prod(span))
    fill = total_item_vol / total_cont_vol if total_cont_vol > 0.0 else 0.0
    cog = num / den if den > 0.0 else 0.0
    return fill, cog


def _metric(result: tuple) -> tuple:
    """(placed, fill, cog) → 辞書式比較キー (placed, fill, -cog)。大きいほど良い。"""
    placed, fill, cog = result
    return (placed, fill, -cog)


# --- 近傍操作（swap / insert / reverse-segment） -------------------------------------

def _apply_move(order: list[int], move_type: int, rng) -> list[int]:
    n = len(order)
    new = list(order)
    if n < 2:
        return new
    if move_type == 0:  # swap
        i, j = int(rng.integers(n)), int(rng.integers(n))
        new[i], new[j] = new[j], new[i]
    elif move_type == 1:  # one-element insert
        i = int(rng.integers(n))
        j = int(rng.integers(n))
        v = new.pop(i)
        new.insert(j, v)
    else:  # short segment reverse (width 2..8)
        lo = int(rng.integers(n))
        max_w = min(8, n - lo)
        if max_w < 2:
            new[lo], new[max(0, lo - 1)] = new[max(0, lo - 1)], new[lo]
        else:
            width = int(rng.integers(2, max_w + 1))
            new[lo:lo + width] = list(reversed(new[lo:lo + width]))
    return new


# --- 公開API ------------------------------------------------------------------------

def _clone_containers(state0: PackingState):
    """height だけが可変（bake_placed が in-place 更新）なので、height のみコピーした
    ContainerSpace の浅いクローン一覧を返す（floor_z/ceil_z/shelf_boxes/path_* は read-only 共有）。"""
    work = []
    for space in state0.containers:
        c = copy.copy(space)
        c.height = np.array(space.height, dtype=np.float64)
        work.append(c)
    return work


def optimize_order(items: list[dict], state0: PackingState, budget_s: float, rng) -> list[int]:
    """荷物順序を探索し、公式 index の完全順列（builtin int）を返す（§4.14）。

    anytime：内部で常に有効な順列を保持し、例外は捕捉して複合キー順序へフォールバックする。
    探索は決定論的マルチスタート山登り（固定 rng）＋反復上限＋締切ゲート。評価器は
    `_greedy_replay`（オフライン貪欲リプレイ、探索は高速fidelity）。

    Args:
        items: 全荷物の情報辞書（`item["index"]` を含む）。
        state0: 初期状態（`build_state` で構築済み。容器形状＋初期 packed_items）。
        budget_s: 全体の時間予算 [s]（`ORDER_TIME_RESERVE_S` を差し引いた時刻で探索終了）。
        rng: `numpy.random.Generator`（固定シードで決定論）。

    Returns:
        list[int]: 全 index を過不足なく1回ずつ含む順列。
    """
    fallback = composite_key_order(items)
    try:
        if state0 is None or not state0.containers:
            return fallback
        metrics = _item_metrics(items)
        specs = _specs_by_index(items)
        seeds = _build_seed_orders(metrics)
        if not seeds:
            return fallback

        work_containers = _clone_containers(state0)
        meta = dict(state0.meta) if state0.meta else {}
        initial_placed = {c: list(state0.placed.get(c, [])) for c in range(len(work_containers))}

        search_stage = StageParams(l_path_top_m=SIM_LPATH_TOP_M)
        sp = _SimParams(support_only=True, ems_cap=SIM_EMS_TOPN, stage=search_stage)

        start = time.monotonic()
        deadline = start + max(0.0, float(budget_s) - ORDER_TIME_RESERVE_S)

        def _eval(order):
            return _metric(_greedy_replay(order, work_containers, initial_placed, meta, specs, sp))

        # 1) 全シード評価（少なくとも seed[0] は必ず評価）。締切超過で打ち切り。
        evaluated = []
        for i, order in enumerate(seeds):
            evaluated.append((order, _eval(order)))
            if i > 0 and time.monotonic() >= deadline:
                break
        evaluated.sort(key=lambda t: t[1], reverse=True)
        best_order, best_metric = evaluated[0]
        anchors = evaluated[: max(1, BEAM_W)]

        # 2) 決定論的マルチスタート best-improvement 山登り。
        active = 0
        current_order, current_metric = anchors[active]
        stagnant = 0
        for it in range(ORDER_MAX_ITERS):
            if time.monotonic() >= deadline:
                break
            neighbor = _apply_move(current_order, it % 3, rng)
            m = _eval(neighbor)
            if m > current_metric:
                current_order, current_metric = neighbor, m
                stagnant = 0
                if m > best_metric:
                    best_order, best_metric = neighbor, m
            else:
                stagnant += 1
                if stagnant >= ORDER_STAGNANT_RESTART and len(anchors) > 1:
                    active = (active + 1) % len(anchors)
                    current_order, current_metric = anchors[active]
                    stagnant = 0

        # 防御: 完全順列であることを保証（万一崩れていれば fallback）。
        if sorted(best_order) != sorted(fallback):
            return fallback
        return [int(x) for x in best_order]
    except Exception:
        return fallback
