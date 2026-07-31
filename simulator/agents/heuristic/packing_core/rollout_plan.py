"""課題A(optimize=true)向け 本物physics 先読みロールアウト計画器（HF-014）。

貪欲の近視眼を、REAL PyBullet で前方評価する先読みで回避する。各手で上位M候補を「候補適用→greedy
をH手前方シミュ」した num_placed で評価し最良を採用。pybullet saveState/restoreState で分岐評価を
高速化（前方の再生成なし）。物理なし surrogate は沈降を欠き本番で退行するため、必ず REAL env を使う。

agent が optimize() で持つデータ（init_states.container_list + item_list + look_ahead）だけから
task を再構築して入れ子 GroundHandlingEnv を建てる（validator/action/camera は評価基準の固定値、
README §性能要件/sample_config 準拠）。返すのは (order, plan)。plan[item_idx]=(container_idx,
pos_rel(list), orientation)。policy() はこの plan に従って配置し、無い/不成立なら貪欲へフォールバック。

安全: 例外時は None（呼び出し側=agent が貪欲順へフォールバック）。時間予算超過後は greedy に切替。
貪欲(=v14)をフロアに保持し、ロールアウトが貪欲を下回るなら貪欲 plan を採用（非退行）。
"""
from __future__ import annotations

import os
import time

import numpy as np

from . import heightmap as HM
from .constants import GridParams, HM_CASCADE, PlacementParams
from .container_space import build_container_space, contains_oriented_box
from .geometry import oriented_size
from .state import build_state, make_action

# 評価基準の固定設定（全課題共通。README 性能要件 / sample_config）。
_FIXED_VALIDATOR = {"inclusion_margin": -0.005, "start_z": 0.08, "safety_margin": 0.015,
                    "ceiling_margin": 0.018, "displacement_threshold": 0.3,
                    "angle_displacement_threshold": 45, "settle_wait_step": 300}
_FIXED_ACTION = {"keys": {"item_idx": "int", "container_idx": "int", "place_pos": "float", "orientation": "int"},
                 "pos_lim": {"low": -100, "high": 100}, "orientations": [0, 1, 2, 3, 4, 5]}
_FIXED_VIS = {"vis": False, "camera": {"yaw": 0, "pitch": -20}}


def _camera(n):
    return {"num_containers": int(n), "target_pos": [0, 0, 0], "distance": 3.0, "yaw": 0, "pitch": 0,
            "roll": 0, "img_width": 64, "img_height": 64, "fov": 60, "near_val": 0.1, "far_val": 10.0}


_ITEM_KEYS = ("index", "length", "width", "height", "mass", "is_prioritized", "is_soft", "pos", "orn")


def _filter_item(d):
    """container_list の packed_items 要素(item.get_info())を Item(**cfg) が受ける鍵だけに絞る。
    get_info は belongs_to/lateralFriction 等の余分な鍵を含み、そのまま渡すと TypeError になる。"""
    out = {k: d[k] for k in _ITEM_KEYS if k in d}
    if d.get("is_soft"):
        for k in ("contactStiffness", "contactDamping", "linearDamping"):
            if k in d:
                out[k] = d[k]
    return out


def reconstruct_task(container_list, item_list, lookahead_k, plan_lookahead=None):
    """agent 保有データ + 固定ブロックから env 構築用 task dict を組む。
    plan_lookahead を与えると planning 用 env の look_ahead を広げ、rollout に荷物選択(=順序)の自由度を
    与える（deployment の実 look_ahead とは独立。追従は placed_order 順で la 非依存）。
    HF-019b: 途中積付(container_list の packed_items 非空)を反映し、既存配置の上に積み増す。"""
    conts = []
    for c in container_list:
        packed = [_filter_item(pi) for pi in (c.get("packed_items") or [])]
        conts.append({"index": int(c["index"]), "length": float(c["length"]), "width": float(c["width"]),
                      "height": float(c["height"]), "thickness": float(c["thickness"]),
                      "buffer": float(c.get("buffer", 0.0)), "cut_x": float(c["cut_x"]), "cut_y": float(c["cut_y"]),
                      "packed_items": packed, "require_shelf": bool(c.get("shelf", c.get("require_shelf", False))),
                      "is_prioritized": bool(c.get("is_prioritized", False))})
    return {
        "containers": {"spacing": 2.5, "container_list": conts},
        "item_stream": {"item_list": item_list,
                        "look_ahead": int(plan_lookahead if plan_lookahead else lookahead_k),
                        "max_space": 1, "visible_pool": []},
        "validator": dict(_FIXED_VALIDATOR), "action": dict(_FIXED_ACTION),
        "camera": _camera(len(conts)), "visualizer": _FIXED_VIS,
        "agent": {"optimize": True, "init_timeout": 10.0, "optimization_timeout": 180.0,
                  "policy_timeout": 8.0, "allowed_methods": ["get_init_states", "optimize", "policy"], "max_mem": 12},
    }


def _budget():
    from .watchdog import StepBudget
    return StepBudget(t0=0.0, soft=1e18, hard=1e18, now_fn=lambda: 0.0)


class _RolloutEnv:
    """入れ子 GroundHandlingEnv を駆動し、候補生成 + saveState 分岐評価を提供する。"""

    def __init__(self, task, order=None):
        from src.ground_handling.env import GroundHandlingEnv
        self.env = GroundHandlingEnv(config=task, verbose=False, render_mode=None)
        self.env.reset_settings()
        self.init = self.env.get_init_states()
        # optimize=true では順序を確定させる。v14 の smart order(big-first) を与えると、以降の先読みは
        # 「位置のみ」を改善する → num_placed は v14 以上、fill は同一荷物集合で v14 同等（=strict superset）。
        # order 未指定時は自然順（診断用）。
        if getattr(self.env, "optimize", False):
            use = list(order) if order is not None else list(self.env.stream_manager.all_indices)
            self.env.set_item_order(use)
        self.env.reset_item_stream()
        obs, _ = self.env.reset(seed=42)
        self.last_obs = obs
        self.n_placed = 0
        self.volume = 0.0
        self.total = self.env.num_total_items

    def cands(self, topk):
        state = build_state(self.last_obs, self.init, compute_ems=False)
        models, ranked = HM._build_models_ranked(state, _budget())
        if not ranked:
            return []
        out = []
        n_stages = len(HM_CASCADE)
        for s_idx, (incl_m, safety_m, sr_min, sc_min, _pb, _ck) in enumerate(HM_CASCADE):
            is_desp = s_idx == n_stages - 1
            pp = PlacementParams(safety_margin=float(safety_m))
            for c in ranked:
                if len(out) >= topk:
                    return out
                if (not is_desp) and c.features.get("soft_nonflat"):
                    continue
                m = models[c.container_idx]
                if HM._feasible(state, m, c, incl_m, pp, sr_min, sc_min):
                    out.append(HM._densify(state, m, c, incl_m, pp, sr_min, sc_min, _budget()))
            if out:
                break
        return out

    def fixed_id(self, cand):
        """候補の pool 相対 item_idx を、その荷物の固定 index（config item_list の index）へ変換。
        plan は固定 index を鍵にする必要がある（pool 相対 index は la=1 では常に 0 で衝突するため）。"""
        return int(self.env.stream_manager.visible_pool[int(cand.item_idx)].index)

    def apply(self, cand):
        vol = float(getattr(self.env.stream_manager.visible_pool[int(cand.item_idx)], "volume", 0.0))
        act = make_action(item_idx=int(cand.item_idx), container_idx=int(cand.container_idx),
                          pos_rel=np.asarray(cand.pos_rel, dtype=np.float64), orientation=int(cand.orientation))
        self.last_obs, _, term, trunc, info = self.env.step(act)
        st = info.get("status", {})
        def ok(v):
            return (all(v.values()) if isinstance(v, dict) else bool(v)) if v is not None else False
        placed = ok(st.get("is_valid")) and ok(st.get("is_placed_safe"))
        if placed:
            self.n_placed += 1
            self.volume += vol
        done = term or trunc or (not placed)
        return placed, done

    def snapshot(self):
        cl = self.env.client
        sid = cl.saveState()
        bodies = frozenset(cl.getBodyUniqueId(i) for i in range(cl.getNumBodies()))
        sm = self.env.stream_manager; cm = self.env.container_manager
        return {"sid": sid, "bodies": bodies, "obs": self.last_obs, "n": self.n_placed, "vol": self.volume,
                "ci": sm.current_index, "vp": list(sm.visible_pool),
                "packed": [list(c.packed_items) for c in cm.containers]}

    def restore(self, snap):
        cl = self.env.client
        for b in [cl.getBodyUniqueId(i) for i in range(cl.getNumBodies())]:
            if b not in snap["bodies"]:
                try: cl.removeBody(b)
                except Exception: pass
        cl.restoreState(stateId=snap["sid"])
        sm = self.env.stream_manager; cm = self.env.container_manager
        sm.current_index = snap["ci"]; sm.visible_pool = list(snap["vp"])
        for c, pk in zip(cm.containers, snap["packed"]):
            c.packed_items = list(pk)
        self.last_obs = snap["obs"]; self.n_placed = snap["n"]; self.volume = snap["vol"]

    def free(self, snap):
        try: self.env.client.removeState(snap["sid"])
        except Exception: pass

    def close(self):
        try: self.env.close()
        except Exception: pass


def _score(e, obj):
    """rollout 目的関数。obj='vol' なら配置総体積（=fill に比例）、それ以外は num_placed。"""
    return e.volume if obj == "vol" else e.n_placed


def _forward_greedy(e, cap, obj):
    """greedy(候補先頭)で cap 手前方シミュ。戻り 目的値。"""
    done = False; g = 0
    while not done and g < cap:
        g += 1
        cs = e.cands(1)
        if not cs:
            break
        _, done = e.apply(cs[0])
    return _score(e, obj)


def _rollout(e, M, H, deadline_s, record=False, obj="count"):
    """先読みロールアウト本体。obj で num_placed / 総体積 を最大化。record=True で plan も返す。"""
    t0 = time.monotonic()
    plan = []
    done = False; guard = 0
    topk = max(M, 1)
    while not done and guard < e.total + 5:
        guard += 1
        cs = e.cands(topk)
        if not cs:
            break
        best_i = 0
        if (time.monotonic() - t0) < deadline_s and len(cs) > 1:
            snap = e.snapshot()
            best_fwd = -1.0
            for i in range(min(M, len(cs))):
                e.restore(snap)
                placed, d = e.apply(cs[i])
                fwd = _score(e, obj) if d else _forward_greedy(e, H, obj)
                if fwd > best_fwd:
                    best_fwd, best_i = fwd, i
            e.restore(snap); e.free(snap)
        c = cs[best_i]
        if record:
            # 固定 index を鍵に記録（適用前に visible_pool から取得）。
            plan.append((e.fixed_id(c), int(c.container_idx),
                         np.asarray(c.pos_rel, dtype=np.float64).tolist(), int(c.orientation)))
        _, done = e.apply(c)
    return e.n_placed, e.volume, plan


def _order_and_plan(best_plan, item_list):
    """best_plan(=[(fixed_id, cont, pos, orn), ...] rollout 順) から
    (order:全固定index列, plan_d:{fid->(cont,pos,orn)}, rank:{fid->placed_order内順位}) を作る。"""
    plan_d = {fid: (ci, pos, orn) for (fid, ci, pos, orn) in best_plan}
    placed_order = [fid for (fid, _c, _p, _o) in best_plan]
    rank = {fid: i for i, fid in enumerate(placed_order)}
    all_idx = [int(x["index"]) for x in item_list]
    rest = [i for i in all_idx if i not in plan_d]
    return placed_order + rest, plan_d, rank


def _construct(e, total, budget_s, rule="deeplow", record=False):
    """実env 上で選択規則に従い構成配置。各手を try/except で包み、config-data 起因の例外は
    その手で打切り（それまでの部分結果を保持）。rule='greedy'=候補先頭(=v14相当), 'deeplow'=最奥・最低。
    戻り (n_placed, plan)。plan は record=True 時のみ [(fid,cont,pos,orn),...]。"""
    t0 = time.monotonic()
    plan = []
    done = False
    guard = 0
    while not done and guard < total + 5:
        guard += 1
        if time.monotonic() - t0 > budget_s:
            break
        try:
            cs = e.cands(64)
            if not cs:
                break
            if rule == "greedy":
                idx = 0
            else:
                keys = [(float(np.asarray(c.pos_rel)[1]) + float(np.asarray(c.osize)[1]) / 2.0)
                        - float(np.asarray(c.pos_rel)[2]) for c in cs]
                idx = int(np.argmax(keys))
            c = cs[idx]
            if record:
                plan.append((e.fixed_id(c), int(c.container_idx),
                             np.asarray(c.pos_rel, dtype=np.float64).tolist(), int(c.orientation)))
            _, done = e.apply(c)
        except Exception:
            break        # config-data 起因の例外はその手で打切り（部分plan は floor 判定に回す）
    return e.n_placed, plan


def deeplow_optimize(container_list, item_list, lookahead_k, budget_s=120.0):
    """HF-019 課題A: 搬入路保存の構成的パッキング。実env で全荷物可視(plan_lookahead=total)にし、
    各手 feasible 候補から「最も奥・低い」(back_edge_y − z)を選んで実配置(沈降込み)する。深さ優先で
    置くため搬入路(door→奥のY-slide)が常に空き、v14(貪欲)が自ら塞ぐ奥空間を活用できる。
    実機検証済: c01 24→29 / fill 33.5→34.3、c03 fill +1.4、build==follow-plan 再生一致。

    HF-019c 堅牢化: (1) 各手 try/except で config-data 例外に耐える。(2) 非退行フロア＝**v14の実挙動**
    （plan_order_forward_sim 順序 + 貪欲デコードを native look_ahead で nested 再現）を基準に、deeplow が
    **num_placed と fill の両方で v14 以上（Pareto支配）**の時だけ採用。DR検証で判明した「+np/−fill の悪トレード
    （seed3001）」を、greedy基準・num_placed優先のフロアが誤って通し v20 が本番退行した問題への対処。
    途中積付は reconstruct_task が packed_items を反映。戻り (order, plan, rank) または None。"""
    try:
        t0 = time.monotonic()
        total = len(item_list)

        def _fill(env):
            try:
                return float(env.evaluate()["fill_score"])
            except Exception:
                return -1.0

        # --- 非退行フロア = v14 の実挙動を nested で再現（forward_sim order + 貪欲, native la）---
        try:
            from .order import plan_order_forward_sim
            v14_order = plan_order_forward_sim(item_list, container_list, max(5.0, min(30.0, budget_s * 0.25)))
        except Exception:
            v14_order = None
        task_native = reconstruct_task(container_list, item_list, lookahead_k)   # la=native
        ev = _RolloutEnv(task_native, order=v14_order)
        v14np, _ = _construct(ev, total, min(60.0, budget_s * 0.4), rule="greedy", record=False)
        v14fill = _fill(ev.env)
        ev.close()

        # --- deeplow 構成（全荷物可視, 残予算）---
        task_all = reconstruct_task(container_list, item_list, lookahead_k, plan_lookahead=total)
        remain = max(15.0, budget_s - (time.monotonic() - t0) - 5.0)
        ed = _RolloutEnv(task_all, order=None)
        dnp, plan = _construct(ed, total, remain, rule="deeplow", record=True)
        dfill = _fill(ed.env)
        ed.close()

        # Pareto フロア: deeplow が num_placed・fill の両方で v14 以上、かつ少なくとも一方で厳密に上。
        eps = 1e-6
        pareto = (dnp >= v14np - 1e-9 and dfill >= v14fill - eps) and (dnp > v14np or dfill > v14fill + eps)
        if not plan or not pareto:
            return None
        return _order_and_plan(plan, item_list)
    except Exception:
        return None


def plan_optimize(container_list, item_list, lookahead_k, budget_s=150.0, M=None, H=None):
    """課題A(la=1) 計画。戻り (order, plan:{fid->(cont,pos,orn)}, rank:{fid->順位}) または None。

    先読みロールアウト(REAL physics)で v14 を上回る配置計画を探す。base は v14 の smart order
    (plan_order_forward_sim)。planning look_ahead を広げると rollout が荷物選択(=順序)も探索できる
    (=surrogate ベースの v14 order を REAL physics で上書き)。目的は GH_OBJ で count/vol 切替。
    貪欲フロア(base order + 先読みなし)を測り、rollout が目的値でそれを上回った時のみ採用（さもなくば
    None → agent が v14 順序へ）。残予算を先読み deadline に割当て二重消費を避ける。パラメータは環境変数で調整可。
    """
    try:
        M = int(os.environ.get("GH_M", str(M if M else 4)))
        H = int(os.environ.get("GH_H", str(H if H else 8)))
        obj = os.environ.get("GH_OBJ", "count")           # 'count' | 'vol'
        pla = int(os.environ.get("GH_PLA", "0"))          # planning look_ahead（0=実 la と同じ）
        t0 = time.monotonic()
        task = reconstruct_task(container_list, item_list, lookahead_k, plan_lookahead=(pla or None))
        # v14 の smart order（forward_sim、~6s）。失敗時は自然順（order=None）。
        try:
            from .order import plan_order_forward_sim
            v14_order = plan_order_forward_sim(item_list, container_list, max(5.0, min(20.0, budget_s * 0.15)))
        except Exception:
            v14_order = None
        # 貪欲フロア（base order + 先読みなし ≈ v14）。目的値も測る。
        eg = _RolloutEnv(task, order=v14_order)
        ng, vg, _ = _rollout(eg, M=1, H=0, deadline_s=0.0, record=False, obj=obj)
        eg.close()
        # 残予算を先読みへ（12s の安全余裕）。
        remain = max(15.0, budget_s - (time.monotonic() - t0) - 12.0)
        er = _RolloutEnv(task, order=v14_order)
        nr, vr, rplan = _rollout(er, M=M, H=H, deadline_s=remain, record=True, obj=obj)
        er.close()
        # 採用条件: 目的値がフロア超 かつ num_placed 非悪化（vol 目的でも count を落とさない）。
        better = (vr > vg + 1e-9) if obj == "vol" else (nr > ng)
        if not (better and nr >= ng and rplan):
            return None
        return _order_and_plan(rplan, item_list)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# HF-024: 層積みプランナ ―― **No-Go**（既定 OFF、`GH_LAYERED=1` で実験可）
# ---------------------------------------------------------------------------
# 結論: **層積みは原理的に DBLF 貪欲に勝てない。** 実行できたタスクで ctr_mean 0.4074 に対し
# 貪欲(v35)は 0.3851。理由は一般的で、層積みは各層の高さをその層の**最厚**荷物に量子化する
# ので薄い荷物の上に無駄な空間が残る。一方 DBLF は各荷物を**個別の最低点**に置き、これが
# Σbottom_i に対する貪欲最適そのもの。層に揃えるのは個別最適を捨てて上へ丸める行為。
#
# 動機にした「理論下限 ctr_mean 0.2115」は **artifact だった**。あの値は容器ごとに
# 「v25 が実際に置いた約26個」だけを層積みした値で、3層しか積んでいない。荷物数を増やせば
# 必ず高くなるので、同じ配置数での比較になっていなかった。局所 fill と局所 ctr_mean の
# r=+0.93 は幾何的に不可避であり、「密に詰めて同時に低くする」構成は存在しない。
#
# 未解決の実装バグも残る: 9/10 タスクで初手が公式 `check_inclusion` に落ちる
# （`contains_oriented_box(margin=0)` は通るので幾何再構成が公式と一致していない）。
# 上記の原理的な結論により修正の価値がないため未修正のまま残す。
#
# 実装から得られた再利用可能な知見:
#   - **入口レーンの x 範囲でクランプが必須**（`path_lane_x_min/max_geom_rel`）。
#     平床は x=-0.540 から始まるがレーンは x=-0.520 からで、20mm 外れると搬入が通らない。
#   - **床層は内包マージン +5mm を原理的に満たせない**（底面が床そのもの）。
#     「構成で fill 内包を保証する」ことはできない。
#   - 棚詰めでは**行の先頭に深い荷物**を取る。浅い荷物を先に取ると行が浅くなり他が入らない。
#
# 以下、当初の設計意図（記録として保持）:
# 動機: cog_score は「相対中心高の**単純平均**」なので、その最小化は Σbottom_i の
# 最小化と厳密に等価。
# 動機: cog_score は「相対中心高の**単純平均**」なので、その最小化は Σbottom_i の
#
# 搬入の正当性は構成で保証される（公式 check_transport_path は単一高さの水平スライド）:
#   - 層を**下から**積むので、その高さ帯には上の層がまだ存在しない
#   - 層内は**奥→手前**に置くので、目標より手前は常に空
#   - 積み上げ層は底面が resting surface から離れるので 0.08m 持ち上げられる。
#     持ち上げ後の高さでも下の層（上面 = z）の上を通るので干渉しない。
#     ただし `z + LIFT + hz <= 天井` を満たす必要があるため層数の上限に反映する。
#
# fill の内包判定（8隅すべてが全境界面から 5mm 以上内側）も構成で満たす:
#   側壁から +INCL 余裕を取って詰めるので、床層以外は必ず fill に計上される。
#   床直置きは構造的に除外されるので（面0 の法線が z=thickness）、床層は薄物で埋める。

_LIFT = 0.08          # 公式の積み上げ搬入リフト量 [m]
# 幾何判定マージン。公式 check_inclusion は −0.005（緩和）なので 0.0 は安全側。
# 当初 +0.006（fill の 5mm 内包マージンより内側）にしたが、**床層は底面が床そのものなので
# z 方向で原理的に満たせない**（床直置きが fill から構造的に除外されるのと同じ理由）。
# よって「構成で fill 内包を保証する」ことはできない。床層は薄物で埋めて損失を最小化する。
_MARGIN = 0.0
_PAD = 0.008          # 矩形の余白。マージンと同値だと浮動小数で境界落ちする


def _flat_orientations(size):
    """扁平姿勢（最小辺が鉛直）になる orientation を、footprint の2通り分返す。

    戻り値は [(orn, wx, wy, hz), ...]。荷物は実測 489/489 が扁平で置かれるので
    鉛直方向は最小辺に固定し、床面上の向きだけを選択肢にする。
    """
    hmin = min(float(v) for v in size)
    out, seen = [], set()
    for orn in range(6):
        wx, wy, hz = (float(v) for v in oriented_size(size, orn))
        if abs(hz - hmin) > 1e-9:
            continue
        key = (round(wx, 6), round(wy, 6))
        if key in seen:
            continue
        seen.add(key)
        out.append((orn, wx, wy, hz))
    return out


def _flat_floor_x_range(space):
    """cut で持ち上がっていない「平らな床」の x 範囲を格子から求める。

    `space.floor_z` は cut plane を反映した床高。最小値と同じ列だけを使えば
    斜面（cut 楔）を避けられる。連続領域として最長の区間を返す。
    """
    fz = space.floor_z
    base = float(fz.min())
    ok = (fz <= base + 1e-6).all(axis=1)          # その x 列が全 y で平床か
    best = (0, -1)
    i = 0
    while i < ok.shape[0]:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < ok.shape[0] and ok[j + 1]:
            j += 1
        if j - i > best[1] - best[0]:
            best = (i, j)
        i = j + 1
    if best[1] < best[0]:
        return None
    x0 = float(space.inner_min_rel[0]) + best[0] * space.cell
    x1 = float(space.inner_min_rel[0]) + (best[1] + 1) * space.cell
    return base, x0, x1


def _pack_layer(cands, x0, x1, y0, y1, z, hz_cap, space, lift):
    """1層を棚詰めする。行は奥(y1)→手前(y0)、行内は x 昇順。

    `cands` は (fid, size, vol) のリスト（この層の候補）。置けたものを
    [(fid, orn, pos_rel, wy, hz), ...] で返し、`cands` から取り除く。

    重要: 配置が不可行でも `x_cur` を前進させる。cut 平面の境界など「その x では
    どの荷物も置けない」位置があり、前進しないと層全体が失敗する（初版のバグ）。
    可行性の最終判定は `contains_oriented_box` に委ね、探索側は保守的に刻む。
    """
    step = max(space.cell, 1e-3)
    placed = []
    y_cur = y1
    while y_cur > y0 + 1e-9 and cands:
        row_depth = 0.0
        x_cur = x0
        row_hit = False
        while x_cur < x1 - 1e-9 and cands:
            best = None
            for k, (fid, size, _v) in enumerate(cands):
                for orn, wx, wy, hz in _flat_orientations(size):
                    if hz > hz_cap + 1e-9:
                        continue
                    if x_cur + wx > x1 + 1e-9 or y_cur - wy < y0 - 1e-9:
                        continue
                    if row_depth > 0.0 and wy > row_depth + 1e-9:
                        continue          # 行の深さは先頭の荷物で決まる
                    if lift > 0.0 and (z + lift + hz) > float(space.inner_max_rel[2]) + 1e-9:
                        continue          # 積み上げは 8cm 持ち上げ後も天井下でなければ搬入不可
                    pos = np.array([x_cur + wx / 2.0, y_cur - wy / 2.0, z + hz / 2.0])
                    osz = np.array([wx, wy, hz])
                    if not contains_oriented_box(space, pos, osz, _MARGIN):
                        continue
                    # 層高を押し上げないため **薄い順** を最優先（理論下限 0.2115 を出した
                    # 「厚み昇順」構成の再現）。次に **深い順**: 棚詰めでは行の深さは先頭の
                    # 荷物で決まるので、浅い荷物を先に取ると行が浅くなり他が入らない
                    # （初版はこれで層あたり5個しか置けなかった）。同条件なら大体積を優先。
                    key = (round(hz, 4), -wy, -_v)
                    if best is None or key < best[0]:
                        best = (key, k, int(orn), pos, wx, wy, hz)
            if best is None:
                x_cur += step         # この x では何も置けない → 前進して再試行
                continue
            _key, k, orn, pos, wx, wy, hz = best
            fid = cands[k][0]
            placed.append((fid, orn, pos, wy, hz))
            cands.pop(k)
            x_cur += wx
            row_depth = max(row_depth, wy)
            row_hit = True
        if not row_hit:
            break                     # どの x でも置けない＝この層は終わり
        y_cur -= row_depth
    return placed


def layered_optimize(container_list, item_list, lookahead_k, thick_tol=0.03):
    """層積みで全荷物の配置計画を作る。`(order, plan, rank)` か None を返す。

    容器ごとに床から層を積む。各層は「厚みの近い荷物」だけで構成して上面を平坦に保つ
    （これが次の層の bottom を下げ、Σbottom = cog を直接下げる）。薄い順に使うので
    床層（fill から構造的に除外される）が最も薄く・最も小体積になる。
    """
    if not container_list or not item_list:
        return None
    cell = GridParams().cell
    spaces = []
    for i, cd in enumerate(container_list):
        try:
            spaces.append(build_container_space(cd, i, cell))
        except Exception:
            return None
    rem = []
    for it in item_list:
        size = (float(it["length"]), float(it["width"]), float(it["height"]))
        rem.append((int(it["index"]), size, size[0] * size[1] * size[2]))
    rem.sort(key=lambda t: (min(t[1]), -t[2]))       # 薄い順、同厚なら大体積優先

    plan_seq = []
    for cidx, space in enumerate(spaces):
        fr = _flat_floor_x_range(space)
        if fr is None:
            continue
        floor_z, fx0, fx1 = fr
        # 内包マージン分だけ内側へ寄せる。cut 側は斜面なので法線の x 成分で割って増やす。
        # cut 側は斜面なので法線の x 成分で割って余白を増やす
        nx_max = max((abs(float(n[0])) for n, _d in space.cut_planes), default=1.0)
        x0 = fx0 + _PAD / max(nx_max, 0.2)
        x1 = fx1 - _PAD
        # **入口レーンの x 範囲で必ずクランプする。** これを外すと搬入 (check_transport_path)
        # が通らず初手でエピソードが終了する（実測: 9/10 タスクで配置数ゼロ）。
        # 平床は x=-0.540 から始まるがレーンは x=-0.520 からで、20mm 外れていた。
        x0 = max(x0, float(space.path_lane_x_min_geom_rel) + _PAD)
        x1 = min(x1, float(space.path_lane_x_max_geom_rel) - _PAD)
        if x1 - x0 < 0.05:
            continue
        y0 = float(space.inner_min_rel[1]) + _PAD
        y1 = float(space.inner_max_rel[1]) - _PAD
        ceil_z = float(space.inner_max_rel[2])
        z = floor_z
        first = True
        guard = 0
        while z < ceil_z - 1e-6 and rem and guard < 60:
            guard += 1
            lift = 0.0 if first else _LIFT
            head = ceil_z - z - lift
            if head <= 1e-6:
                break
            t_min = min(min(t[1]) for t in rem)          # 残り荷物の最薄厚み
            # 厚みクラスで絞らない。**薄い順に取り、面積が埋まるまで**詰めるのが
            # 理論下限 0.2115 を出した構成。クラス制限は層を疎にして高さだけ消費する
            # （初版は t_min+3cm に絞って計画 ctr が 0.5123 まで悪化した）。
            hz_cap = head
            layer = [t for t in rem if min(t[1]) <= hz_cap + 1e-9]
            if not layer:
                break
            got = _pack_layer(layer, x0, x1, y0, y1, z, hz_cap, space, lift)
            if not got:
                z += max(t_min, cell)      # この厚みクラスが入らない → 層を閉じて上へ
                first = False
                continue
            done = {g[0] for g in got}
            rem = [t for t in rem if t[0] not in done]
            lay_h = max(g[4] for g in got)                # 実際に置いた最大厚み
            for fid, orn, pos, _wy, _hz in got:
                plan_seq.append((fid, cidx, pos, orn))
            z += max(lay_h, cell)
            first = False
    if not plan_seq:
        return None
    return _order_and_plan(plan_seq, item_list)
