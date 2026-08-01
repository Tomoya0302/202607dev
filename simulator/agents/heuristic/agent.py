"""heuristic Agent（T-023: 骨格／T-024: 手順2〜7／T-027: warmup・optimize）。

公式 `AgentFactory` からロードできる `Agent` を定義する。`policy()` は毎呼出しで
`build_state`→候補列挙→段階フィルタ→特徴計算→`heuristic_score`→`safe_decide`4層→
`make_action` を配線し（詳細仕様書 §4.12）、例外・候補ゼロ・全層None・action変換失敗の
いずれの場合も最外殻emergency actionへフォールバックしてプロセス外へ例外を漏らさない
（§4.11「最外殻emergency action」）。`__init__`は合成dummyで同じ手順2〜7を1回空回しし、
`optimize`は体積・重量で入力位置を並べて公式item index列を返す（§4.12 v1.23）。
"""
import json
import logging
import os
import time

import numpy as np

from .telemetry import format_row, make_writer_from_env
from .packing_core.constants import (
    PlacementParams,
    ProvisionalRiskParams,
    ScoreParams,
    StageParams,
    TimeParams,
)
from .packing_core import heightmap as _heightmap
from .packing_core.heightmap import decide_placement
from .packing_core.order import composite_key_order, plan_order_forward_sim
from .packing_core.state import build_state, make_action
from .packing_core.watchdog import StepBudget

logger = logging.getLogger("packing")

# HF-012 Phase H1: policy の配置決定は src/packing_core/heightmap.py::decide_placement へ委譲
# （EMS 列挙〜safe_decide を置換）。旧 EMS パイプライン用の定数・import（FEATURE_BUDGET_FRACTION,
# W_LOW_SUPPORT, layer*/safe_decide, enumerate/filter/support_ratio/cg_margin/heuristic_score,
# provisional_p_ng）は不要になったため撤去した。EMS モジュール自体はモジュール単体テストで維持。


def _proxy_log_state(state) -> None:
    """GH_PROXY_LOG 設定時、現在観測の settled 配置一覧を JSONL 1行追記する（既定 no-op）。

    HF-012 H2 の非fillサブスコア（cog/placement/soft/stability）プロキシ計測専用。各ステップで
    env が報告する packed 配置（`state.placed`）を記録し、最多配置の行を最終状態として使う。
    本番実行では GH_PROXY_LOG 未設定のため副作用ゼロ。
    """
    path = os.environ.get("GH_PROXY_LOG")
    if not path:
        return
    items = []
    for cidx, plist in state.placed.items():
        cont_prio = bool(state.containers[cidx].is_prioritized)
        for it in plist:
            amin = np.asarray(it.aabb_min_rel, dtype=float)
            amax = np.asarray(it.aabb_max_rel, dtype=float)
            items.append({
                "c": int(cidx), "cp": cont_prio, "w": float(it.weight),
                "soft": bool(it.is_soft), "prio": bool(it.is_priority),
                "amin": [float(v) for v in amin], "amax": [float(v) for v in amax],
            })
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"n": len(items), "items": items}) + "\n")
    except OSError:
        pass


def _new_timing() -> dict:
    """policy/warmup単位の段階時間accumulatorを返す。"""
    return {"t_state": 0.0, "t_enum": 0.0, "t_mask": 0.0, "t_lpath": 0.0, "t_total": 0.0}


def _new_local_telemetry() -> dict:
    """policy/warmup単位の局所telemetryを返す。"""
    return {
        "n_cand0": 0,
        "n_after_dims": 0,
        "n_after_geo": 0,
        "n_lpath_pass": 0,
        "reject_reason_counts": {},
        "decided_layer": 0,
        "layer_error": [],
    }


def _make_warmup_inputs() -> tuple[dict, dict]:
    """T-027の自己完結小型dummy init/observationを返す（HF-001でv1.27改訂：空コンテナ初手）。

    旧v1.23契約は配置済み荷物1件を含むdummyだったため、空コンテナ初手固有の不具合
    （HF-001：床置き候補がINCLUSION不合格になる問題）をウォームアップ自体では検出
    できなかった。本改訂で`packed_items=[]`の空コンテナ初手dummyへ変更する。
    """
    inner_min = np.asarray((-0.40, -0.40, 0.02), dtype=np.float64)
    inner_max = np.asarray((0.40, 0.40, 1.02), dtype=np.float64)
    mid = (inner_min + inner_max) / 2.0
    face_specs = [
        ((inner_min[0], mid[1], mid[2]), (-1.0, 0.0, 0.0)),
        ((inner_max[0], mid[1], mid[2]), (1.0, 0.0, 0.0)),
        ((mid[0], inner_min[1], mid[2]), (0.0, -1.0, 0.0)),
        ((mid[0], inner_max[1], mid[2]), (0.0, 1.0, 0.0)),
        ((mid[0], mid[1], inner_min[2]), (0.0, 0.0, -1.0)),
        ((mid[0], mid[1], inner_max[2]), (0.0, 0.0, 1.0)),
    ]
    thickness = 0.02
    buffer = 0.02
    height = float(inner_max[2]) + buffer
    container = {
        "index": 0,
        "length": float(inner_max[0] - inner_min[0]) + 2.0 * thickness,
        "width": float(inner_max[1] - inner_min[1]) + 2.0 * thickness,
        "height": height,
        "cut_x": 0.0,
        "cut_y": 0.0,
        "thickness": thickness,
        "center": (0.0, 0.0, height / 2.0 + buffer),
        "n_vecs": [normal for _, normal in face_specs],
        "points": [tuple(float(v) for v in point) for point, _ in face_specs],
        "volume": float(np.prod(inner_max - inner_min)),
        "shelf": False,
        "is_prioritized": False,
        "packed_items": [],
    }
    item = {
        "index": 0,
        "length": 0.10,
        "width": 0.10,
        "height": 0.10,
        "mass": 1.0,
        "is_prioritized": False,
        "is_soft": False,
        "belongs_to": None,
        "pos": None,
        "orn": None,
        "lateralFriction": 0.5,
        "rollingFriction": 0.01,
        "spinningFriction": 0.01,
        "restitution": 0.0,
        "angularDamping": 0.8,
    }
    observation_container = dict(container)
    observation_container["packed_items"] = []
    init = {"optimize": False, "lookahead_k": 1, "container_list": [container]}
    observation = {
        "optimize": False,
        "lookahead_k": 1,
        "depth_map": np.zeros((1, 4, 4), dtype=np.float32),
        "container_list": [observation_container],
        "pool_list": [item],
    }
    return init, observation


def _envf_local(name: str, default: float) -> float:
    """順序後処理の連続レバー用（HF-022）。未設定・数値化不能は既定値へフォールバック。"""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _emergency_action(observation: dict) -> dict:
    """最外殻emergency actionを返す（§4.11「最外殻emergency action」、HF-001でv1.27改訂）。

    `item_idx` は常に `0` とする。A11が定めるプール内index契約（現在のvisible pool先頭の
    action indexは常に0）に従い、`observation["pool_list"][0]["index"]`（荷物固有index）は
    使わない（旧v1.17契約はこの荷物固有indexをそのまま使っていたが、公式が要求するプール内
    位置と一致しないためHF-001で撤回した）。プールindexすら取得できない異常入力でも同じ
    `item_idx=0` を使う。

    Args:
        observation: `policy()` へ渡された観測（読み取りのみ、変更しない）。

    Returns:
        `item_idx=0, container_idx=0, pos_rel=(0.0,0.0,0.5), orientation=0` の
        action 辞書（`make_action()` 経由）。
    """
    pos_rel = np.asarray((0.0, 0.0, 0.5), dtype=np.float64)
    return make_action(item_idx=0, container_idx=0, pos_rel=pos_rel, orientation=0)


class Agent:
    """heuristic 積付 Agent（T-023骨格→T-024縦断実装）。"""

    def __init__(self, module_path: str) -> None:
        """パラメータとtelemetryを初期化し、合成dummyを1回ウォームアップする。

        ウォームアップは実init_statesを使わず、ファイル/telemetry出力や
        `_policy_step`の更新を行わない。パラメータ群は種別ごとに独立した
        属性として保持する（`PlacementParams`/`TimeParams`/`StageParams`/`ScoreParams`/
        `ProvisionalRiskParams` を1つにまとめない）。

        Args:
            module_path: 公式 `AgentFactory` から渡されるモジュールパス。
        """
        self.module_path = module_path
        self.placement_params = PlacementParams()
        self.time_params = TimeParams()
        self.stage_params = StageParams()
        self.score_params = ScoreParams()
        self.risk_params = ProvisionalRiskParams()
        # HF-005: 再起動耐性の既定値。`get_init_states` はエピソード開始時に1回だけ送られ、
        # ワーカーがタイムアウト/クラッシュで再起動されると二度と再送されない（app.py/runner.py）。
        # 既定値が無いと再起動後の policy() が self.optimize_enabled 参照で AttributeError を起こし、
        # 以降ずっと固定 _emergency_action に退行してしまう。ここで安全既定を用意し、さらに
        # _policy_impl は毎ステップ observation から init を再構築する（下記）。
        self.optimize_enabled = False
        self.lookahead_k = 1
        self.container_list = []
        # T-028: telemetry writer は起動時の環境変数（TELEMETRY_DIR/TELEMETRY_RUN_ID）から
        # 一度だけ生成する（§4.13）。policy呼出し通番はAgentインスタンスごとに0始まり。
        self._telemetry_writer = make_writer_from_env()
        self._policy_step = 0
        self._run_warmup()

    def _run_warmup(self) -> None:
        """T-027の合成dummyで手順2〜7を1回実行する。"""
        init, observation = _make_warmup_inputs()
        try:
            self._run_pipeline(
                observation,
                init,
                _new_timing(),
                _new_local_telemetry(),
                {"value": None},
                allow_emergency=False,
            )
        except Exception as exc:
            logger.warning("agent warmup failed; continuing without retry: %s", exc)

    def get_init_states(self, init_states: dict) -> None:
        """公式から渡される初期状態を保持する。

        `init_states` およびそのネスト要素は変更しない。子プロセス側で
        既にデシリアライズ済み・非共有のオブジェクトであるため、追加の
        コピーは行わず参照をそのまま保持する。

        保持先を `self.optimize` にすると、公式 I/F の `optimize()` メソッドを
        bool 属性が隠してしまい、Runner による `agent.optimize(item_list)` 呼出しが
        `TypeError: 'bool' object is not callable` で失敗する。そのため
        `self.optimize_enabled` という別名で保持する。

        Args:
            init_states: `optimize` / `lookahead_k` / `container_list` を持つ辞書。
        """
        self.optimize_enabled = bool(init_states["optimize"])
        self.lookahead_k = int(init_states["lookahead_k"])
        self.container_list = init_states["container_list"]

    def optimize(self, item_list: list[dict]) -> list[int]:
        """体積降順・同体積はmass降順で公式item index列を返す。

        indexは必須の一意な整数（bool禁止）。sort材料が1件でも不正な場合は、
        公式indexの入力順へ全体をフォールバックする（§4.12 v1.23）。

        Args:
            item_list: 全荷物の情報が格納された辞書のリスト。

        Returns:
            list[int]: 全荷物の公式 index を過不足なく1回ずつ含むリスト。

        Raises:
            ValueError: indexが欠落・非整数・bool・重複の場合。
        """
        indices: list[int] = []
        for position, item in enumerate(item_list):
            try:
                raw_index = item["index"]
            except (KeyError, TypeError) as exc:
                raise ValueError(f"item_list[{position}] index is required") from exc
            if isinstance(raw_index, (bool, np.bool_)) or not isinstance(
                raw_index, (int, np.integer)
            ):
                raise ValueError(f"item_list[{position}] index must be an integer")
            indices.append(int(raw_index))

        if len(set(indices)) != len(indices):
            raise ValueError("item index values must be unique")

        # HF-022 Phase 9: 優先品の総量を配置エンジンへ預ける（容量予約の分母）。
        # policy は look_ahead=1 で将来の荷物を見られないが、optimize() は全 item_list を受け取る。
        # 厚みは荷物が 100% 扁平姿勢で置かれる実測（489/489）に基づき min(3辺) を使う。
        try:
            _prios = [it for it in item_list if it.get("is_prioritized")]
            _heightmap.PRIO_TOTAL["n"] = float(len(_prios))
            _heightmap.PRIO_TOTAL["vol"] = float(sum(
                float(it["length"]) * float(it["width"]) * float(it["height"]) for it in _prios))
            _heightmap.PRIO_TOTAL["hmax"] = float(max(
                (min(float(it["length"]), float(it["width"]), float(it["height"]))
                 for it in _prios), default=0.0))
            _softs = [it for it in item_list if it.get("is_soft")]
            _heightmap.SOFT_TOTAL["n"] = float(len(_softs))
            _heightmap.SOFT_TOTAL["vol"] = float(sum(
                float(it["length"]) * float(it["width"]) * float(it["height"]) for it in _softs))
            _heightmap.SOFT_TOTAL["hmax"] = float(max(
                (min(float(it["length"]), float(it["width"]), float(it["height"]))
                 for it in _softs), default=0.0))
            # HF-025: 重量拒否のしきい値（全荷物の質量分位）。
            _ms = sorted(float(it.get("mass", 0.0)) for it in item_list)
            _q = float(os.environ.get("GH_HM_HEAVY_CEIL_Q", "0.5") or 0.5)
            _heightmap.MASS_THRESHOLD["v"] = (
                _ms[min(len(_ms) - 1, max(0, int(_q * len(_ms))))] if _ms else 0.0)
        except Exception:
            # 予約は改善レバーなので、材料が不正なら黙って無効化する（v29 恒等を保つ）。
            _heightmap.PRIO_TOTAL.update({"n": 0.0, "vol": 0.0, "hmax": 0.0})
            _heightmap.SOFT_TOTAL.update({"n": 0.0, "vol": 0.0, "hmax": 0.0})
            _heightmap.MASS_THRESHOLD["v"] = 0.0

        # HF-013 Phase H3: heightmap 貪欲の全 lookahead 前方シミュで fill 目的の配置順を計画する
        # （実配置エンジン自身の順に合わせる）。旧 EMS ベース順序探索(HF-011/v8)は本番退行のため不使用。
        # container 形状が無い/例外/未完了時は composite_key_order へ内部フォールバック。GH_ORDER_PLAN=0
        # で複合キー順へ明示的に戻せる（A/B・退行時の保険）。
        # HF-014: 課題A のみ、本物physics 先読みロールアウトで配置計画を作る（num_placed 向上）。
        # 入れ子 env を建てて上位M候補をH手前方シミュ評価。貪欲(=v14)をフロアに、rollout が上回る時のみ
        # 採用（非退行）。例外/未達/未設定は self._rollout_plan=None のまま従来の順序計画へフォールバック。
        self._rollout_plan = None
        self._rollout_rank = None
        # HF-019: 課題A 搬入路保存の構成的パッキング（deeplow）。実env で全荷物可視にし各手「最も奥・低い」
        # を実配置。深さ優先で搬入路を空け続けるため v14 が塞ぐ奥空間を使える（実機検証: c01 24→29,
        # fill 33.5→34.3, c03 fill +1.4, build==follow-plan 一致）。既定 ON（GH_DEEPLOW=0 で無効化）。
        # HF-019 結論: deeplow はローカル課題A(c01 24→29,fill+3)を上回るが、本番の隠し課題A構成では
        # v14(plan_order_forward_sim order)に敗け、v20提出で Public 53.24→52 に退行（local≠platform）。
        # フロアを greedy 比較にしていたため退行を防げず。既定 OFF（agent ≡ v14）。GH_DEEPLOW=1 で実験可。
        # HF-024: 層積みプランナ（既定 OFF）。全荷物既知なので容器を床から層で埋める。
        # cog = Σbottom_i（重みなし）の最小化が目的で、実荷物集合の完全層積み理論値
        # ctr_mean 0.2115 に対し現行貪欲は 0.3851。局所 fill と ctr_mean は r=+0.93 で
        # 結合しており既存レバーでは破れないが、層積みは上面を平坦に保つことで
        # fill を落とさず Σbottom を下げられる唯一の構成法。
        # 搬入は構成で保証（下から積む・層内は奥→手前・持ち上げ後も天井下）。
        # 判定条件: Δctr < 0 かつ Δfill ≥ 0 かつ Δnp ≥ 0 を3ベンチで満たすこと。
        if (os.environ.get("GH_LAYERED", "0") != "0" and getattr(self, "optimize_enabled", False)
                and getattr(self, "container_list", None)):
            try:
                from .packing_core.rollout_plan import layered_optimize
                res = layered_optimize(self.container_list, item_list, self.lookahead_k)
                if res is not None:
                    order, plan, rank = res
                    self._rollout_plan = plan
                    self._rollout_rank = rank
                    return order
            except Exception:
                self._rollout_plan = None
                self._rollout_rank = None

        if (os.environ.get("GH_DEEPLOW", "0") != "0" and getattr(self, "optimize_enabled", False)
                and getattr(self, "container_list", None)):
            try:
                from .packing_core.rollout_plan import deeplow_optimize
                bud = max(20.0, min(140.0, float(self.time_params.optimize_stop) - 30.0))
                res = deeplow_optimize(self.container_list, item_list, self.lookahead_k, budget_s=bud)
                if res is not None:
                    order, plan, rank = res
                    self._rollout_plan = plan
                    self._rollout_rank = rank
                    return order
            except Exception:
                self._rollout_plan = None
                self._rollout_rank = None
        # HF-014 結論: REAL physics 先読みは v14 の (plan_order_forward_sim order + greedy) を上回れず。
        # v14 order 上では位置先読みで num_placed/fill とも不変、自然順では num_placed+3 だが fill 低下
        # (gate 超では net negative)、体積目的の順序探索も v14 order 超えず。既定 OFF（agent ≡ v14）。
        # GH_ROLLOUT=1 で有効化（実験用。local では上振れ無し・platform 上振れは未検証の賭け）。
        if (os.environ.get("GH_ROLLOUT", "0") != "0" and getattr(self, "optimize_enabled", False)
                and int(getattr(self, "lookahead_k", 0)) == 1 and getattr(self, "container_list", None)):
            try:
                from .packing_core.rollout_plan import plan_optimize
                # 計画全体の壁時計予算（貪欲フロア込み）。optimize_stop から余裕を引く。
                bud = max(30.0, min(158.0, float(self.time_params.optimize_stop) - 12.0))
                res = plan_optimize(self.container_list, item_list, self.lookahead_k, budget_s=bud)
                if res is not None:
                    order, plan, rank = res
                    self._rollout_plan = plan
                    self._rollout_rank = rank
                    return order
            except Exception:
                self._rollout_plan = None
                self._rollout_rank = None

        # HF-015: 課題A 品質サブスコア狙いの順序後処理。soft/priority 荷物を後ろへ回すと、最終的に
        # 上側(=hard の上・取り出しやすい位置)へ載る → PDF §3(4)③(hardをsoftの上に載せない=soft_score)
        # ④(priority を上側・取り出しやすく=placement_score)。グループ内は base(fill最適)順を保持。
        # 既定 v14（両フラグOFF=恒等）。num_placed/fill を退行させないことを proxy_eval で要確認。
        def _reorder_quality(order):
            soft_late = os.environ.get("GH_SOFT_LATE", "0") != "0"
            # HF-022 v24: **識別用プローブ**として既定 ON にベイク（`GH_PRIO_LATE=0` で v14 へ戻る）。
            # 目的は改善ではなく採点式の同定。cog を上げた提出は歴史上 v18 だけで、v18 に固有なのは
            # 本フラグだけ（優先品配置率 0.629→0.100）。単独で出せば v18 の cog +3.49 / stability +3.78 /
            # placement −7.45 のどれが本フラグ由来か1回で確定する。本番は決定論的（v19 が v14 と16桁一致）
            # なので偏微分がノイズなしで取れる。ローカル23タスクで Δnp +1.52pt / Δfill −0.05。
            # HF-022 v31: 既定を OFF へ。v24 で PRIO_LATE は cog +7.40/stab +7.97 を稼いでいたが、
            # v29 の TALL_SHIFT=2.0 が**同じ機構**（厚い荷物を高所に置かない）で cog を取るため
            # 両者は代替物になった。実測: PRIO_LATE を外しても ctr_mean は +0.0032 しか動かない
            # （cog −0.28）＝ cog の見返り無しに placement を払っているだけ。
            # v29 の本番実測で placement の局所→本番転移比は 0.30、cog の傾きは ctr 1 単位あたり
            # 89（旧較正 500 は 5.6 倍過大）。この正しい傾きでは placement を取る方が有利。
            prio_late = os.environ.get("GH_PRIO_LATE", "1") != "0"
            if not (soft_late or prio_late):
                return order
            meta = {int(it["index"]): (bool(it.get("is_soft")), bool(it.get("is_prioritized")))
                    for it in item_list}

            def _cat(idx):
                soft, prio = meta.get(int(idx), (False, False))
                if prio_late and prio:
                    return 2            # 優先は最後（=上・取り出しやすい）
                if soft_late and soft and not prio:
                    return 1            # soft(非優先)は hard の後（=hard を soft の上に載せない）。優先は降格しない
                return 0
            return sorted(order, key=_cat)  # 安定ソート：カテゴリ内は base 順を保持

        # HF-022: soft を「最後」ではなく「途中まで」遅らせる連続レバー（GH_SOFT_SHIFT, 既定 0=v14）。
        # v18 の因子分解（bench_dr10, 状態ダンプ）で分かったこと:
        #   GH_PRIO_LATE=1 → 優先品の配置率 0.629→0.100。placement 崩壊（本番 50.70→43.25）の原因。
        #   GH_SOFT_LATE=1 → 重心 −0.0026・傾き 0.327→0.290（cog=stability に有利。COM を下げる唯一の因子）
        #                    だが soft 自由率 0.110→0.052。**最後に回した物がエピソード終了で未積載になる**。
        # 両者は「末尾に置くと積み残す」という同一の欠陥。ならば因子を捨てるのではなく欠陥を直せる:
        # 完全分離ではなく base 位置を n·shift だけ後ろへずらす。0.2 なら「2割ぶん遅らせる」で、
        # 上側へ載る利得は得つつ末尾の積み残しを避けられる。1.0 でほぼ GH_SOFT_LATE 相当。
        def _soft_shift_order(order):
            # v23 実測: GH_SOFT_SHIFT=0.35 は本番で cog −0.59 / stability −0.13（仮説は反証）。
            # soft の遅延は品質を上げないので既定 0（無効）のまま。
            #
            # HF-022 v25: **優先品側**の連続シフト（GH_PRIO_SHIFT）。v24 の本番実測で
            #   `GH_PRIO_LATE=1`（=完全に末尾へ）が cog +7.40 / stability +7.97 / soft +1.05 /
            #   fill +0.30 / np +1.48pt、placement のみ −7.45 で **Public 55（v14 初更新）**。
            # 優先品配置率 0.629→0.100 に対する傾きは cog +13.99 / stability +15.05 / placement −14.08
            # ＝利得が損失を2倍上回る。完全に末尾へ回すのが最適とは限らないので、連続量で最適点を探す。
            # v23 で連続シフトが「末尾の積み残し」を実際に半減させる（soft −5.65→−2.65）ことは実証済み。
            # HF-022 v25: 質量ランクによる連続遅延（GH_HEAVY_SHIFT）。v24 の機構は「優先品＝大きく重い
            # 荷物を積まないと cog/stability が上がる」だった（総質量ほぼ同じで置いた数が 47.3→49.2＝
            # 大物を小物に置換）。優先品は全体の15%しかないので、**質量で選べばより大きく振れる**。
            # 配置時の選択圧（GH_HM_W_BIG）で同じことを狙うのは23タスクで No-Go（Δnp −1.2〜−4.0pt）
            # だったので、PRIO_LATE と同じ「順序で除外する」機構を使う。
            # 重い順のランク r/n に比例して遅延: 最重量が n*heavy_shift、中央値がその半分。
            # v33 で **−0.6**（負値＝soft の前倒し）をベイク。v23 の +0.35（後回し）は本番で
            # cog −0.59 / stability −0.13 と回帰したが、**負側は未試行**だった。
            # 効果は2つ: (1) soft の積み残しが減る（soft_item_score は未積載も違反に数える）ので
            # 局所 soft 51.73 → 59.59、(2) **np を +1.9pt 押し上げる**ので PRIO_RESERVE を
            # 1.0 → 1.5 に上げても膝 0.588 を割らない（本番 np 換算 0.616）。
            # 3ベンチの予測 Δmean5（予約1.5 と併用）: dr10 +0.90 / holdout +0.41 / overload −0.28、
            # placement（転移が不確実な項）を除くと +0.38 / +0.23 / +0.45 で全ベンチ正。
            # −0.9 まで強めると soft はさらに伸びるが ctr_mean が holdout で +0.0104 悪化し、
            # 悪化側の傾き 350 で cog −3.62 を払って差引き負（No-Go）。
            # v35 で **−1.3**（soft を強く前倒し）。**num_placed が Public の支配変数**と判明した
            # ため（本番np↔Public r=+0.857 / R²=0.73、局所np↔本番np r=+0.938）、np を目的関数に
            # 取り直して走査した結果。局所 np: v32 の 0.642 → **0.686**（3ベンチで +4.4/+5.8/+4.6pt、
            # 測定した全設定で最大の np 利得）。fill も 24.84 → 26.37 でほぼ横ばい〜微増。
            # −2.5 まで強めると np 0.710 まで伸びるが ctr_mean が 0.4037 まで悪化し、重み付き
            # モデルが強く負を出す（両モデルが一致する範囲の上端が −1.3）。
            # 注意: 重み付きモデル（ctr_mean 経由の cog 換算）は本設定を holdout で −1.34 と評定し
            # 対立する。np モデルを採ったのは (1) Public 67 の実在が重み付きモデルの上限 60 を
            # 反証する (2) np は同一コード計算で連鎖が機構的に堅い一方 ctr_mean は v34 で破れた
            # (3) 上振れ +4.5 対 下振れ −1.0 の非対称、の3点による。
            shift = _envf_local("GH_SOFT_SHIFT", -1.3)
            # HF-022 v31: **負値＝優先品の前倒し**を −1.6 でベイク。placement は「正しく置けた
            # 優先品の割合」で**未積載も違反**に数えるので、前倒しで積み残しが減ると直接上がる。
            # dr10 でローカル placement 29.68 → 76.00。−0.8 と −1.6 を比較して 3ベンチすべてで
            # −1.6 が優位（過積載でも −0.14 対 +0.05）。さらに強めるのは np が 0.666→0.635 と
            # 削られており、ゲート膝 0.588 を割ると v16 のように 13 点失うので採らない。
            # 3ベンチの予測 Δmean5: dr10 +2.51 / holdout +1.73 / overload +0.05。
            prio_shift = _envf_local("GH_PRIO_SHIFT", 0.0)
            # HF-022 v25 で 0.3 へベイク（識別プローブ兼改善候補）。v24 は prio率と平均質量が同時に
            # 動いたため cog +7.40 の駆動因（質量か "優先"ラベルか）を分離できていない。本設定は
            # prio率を 0.100 に固定したまま平均質量だけを下げるので、駆動因が1回で判定できる。
            # 0.6 も測ったが 23タスクで Δnp −1.36pt。PRIO_LATE では局所 Δnp と本番 Δnp が
            # ほぼ 1:1（+1.52pt / +1.48pt）だったので本番 np ≈0.589 となりゲート膝(0.588)に接する。
            # v16 はそこを割って13点失ったため 0.3 を採る（23タスクで Δnp +0.05pt / Δfill −0.53、
            # 本番 np ≈0.603 で安全）。信号は 1/5 だが本番は決定論的でノイズが無いので判定は可能。
            heavy_shift = _envf_local("GH_HEAVY_SHIFT", 0.3)
            # HF-022 Phase 7: **厚み昇順シフト**（薄い荷物を先に＝層を下から薄く作る）。既定 0（v25 恒等）。
            # 導出: cog は「相対中心高の**単純平均**」なので ctr_mean = mean(bottom) + mean(h)/2 であり、
            # mean(h)/2 は集合で決まる ⇒ **cog の最小化は Σbottom_i の最小化と厳密に等価**。
            # 各層 m 個・層高 H_l の層積みでは Σbottom = m·Σ_l H_l·(k−l) なので、
            # **大きい H_l を上の層に置く＝薄い荷物から流す**のが最小。
            # 実測（dr10, 同一荷物・同一層積み機構で順序だけ変えた理想値）:
            #   昇順 ctr_mean 0.2115 / 降順 0.2974 → 順序だけで Δctr_mean −0.0859。
            # 現状の順序は ρ(配置順, 厚み) = −0.315＝**厚い方を先に**流しており理論と逆向き。
            # 加えて薄い荷物は体積が小さいので、fill から構造的に除外される床層の損失も最小になる
            # （床直置きは 5mm 内包マージンで必ず除外＝fill 10-12点が消える）。cog と fill の両方に正。
            # 荷物は 100% 扁平姿勢で置かれる（実測 489/489）ので、厚みは **min(3辺)** を使う。
            # 既知のリスクは「厚い荷物を末尾に回すと入らず np が落ちる」ことなので、
            # ハードソートではなく PRIO/HEAVY と同じ連続シフトで最適点を探す。負値も許す。
            # HF-022 v29 で 2.0 へベイク。3ベンチ（dr10 / holdout未見 / overload）で
            # Δctr_mean −0.054/−0.063/−0.053、Δplacement +27.8/+31.3/+26.7、Δsoft +38.1/+33.6/+20.9。
            # ρ(配置順,厚み) は −0.315 → +0.881 で飽和に近く、これ以上大きくしても伸びない。
            # コストは Δfill −3.6/−5.3/−4.5（本番換算 ×0.43、重み 1/5 なので mean5 −0.3〜−0.5）。
            # np は 0.670/0.651/0.675 でゲート膝 0.588 から十分上（overload のみ −2.5pt だが飽和域）。
            # 1.2 も同等に良いが全ベンチで 2.0 が優位。PRIO_LATE=0 との併用はローカル placement が
            # +49 まで伸びるが、v24 で**本番実測**した「prio率を落として cog/stab を取る」取引を
            # 逆向きに戻すことになるので採らない（local-win-platform-loss を5回踏んでいる）。
            tall_shift = _envf_local("GH_TALL_SHIFT", 2.0)
            # HF-022 Phase 11: **薄さ重み付き soft 前倒し**（既定 0 = 無効、v33 恒等）。
            # 一律の GH_SOFT_SHIFT<0 は soft の未積載（34.7%→3.8%）を消すが、**厚い soft も
            # 前へ出してしまう**ため TALL_SHIFT（薄物先行）を上書きし ctr_mean が悪化する
            # （配置済 soft の厚み 0.227m → 0.254m で非soft 0.242m を逆転、cog −2.87）。
            # 本レバーは soft の前倒し量を **(1 − 厚みランク)** で重み付けし、薄い soft だけを
            # 前へ出す。厚い soft は TALL_SHIFT に従って後段へ残る。
            soft_thin_shift = _envf_local("GH_SOFT_THIN_SHIFT", 0.0)
            if (shift <= 0.0 and prio_shift <= 0.0 and heavy_shift <= 0.0
                    and tall_shift == 0.0 and soft_thin_shift == 0.0):
                return order
            soft = {int(it["index"]) for it in item_list
                    if it.get("is_soft") and not it.get("is_prioritized")}
            prio = {int(it["index"]) for it in item_list if it.get("is_prioritized")}
            n = max(len(order), 1)
            trank: dict[int, float] = {}
            if tall_shift != 0.0 or soft_thin_shift != 0.0:
                def _thick(it) -> float:
                    return min(float(it["length"]), float(it["width"]), float(it["height"]))
                by_t = sorted(item_list, key=lambda it: -_thick(it))
                trank = {int(it["index"]): 1.0 - r / max(len(by_t) - 1, 1)
                         for r, it in enumerate(by_t)}   # 最厚=1.0 → 最薄=0.0
            hrank: dict[int, float] = {}
            if heavy_shift > 0.0:
                by_w = sorted(item_list, key=lambda it: -float(it.get("mass", 0.0)))
                hrank = {int(it["index"]): 1.0 - r / max(len(by_w) - 1, 1)
                         for r, it in enumerate(by_w)}   # 最重量=1.0 → 最軽量=0.0
            # 安定な連続シフト: soft だけ base 順位に +n*shift を加えて再ソート（同値は base 順維持）。
            # key に (ずらした位置, 元位置) を使うので、同カテゴリ内の base(fill最適)順は保たれる。
            def _delta(i):
                d = 0.0
                if int(i) in soft:
                    d += n * shift
                if int(i) in prio:
                    d += n * prio_shift
                if heavy_shift > 0.0:
                    d += n * heavy_shift * hrank.get(int(i), 0.0)
                if tall_shift != 0.0:
                    d += n * tall_shift * trank.get(int(i), 0.0)
                if soft_thin_shift != 0.0 and int(i) in soft:
                    # 薄い soft（trank≈0）だけ強く前へ、厚い soft（trank≈1）は動かさない。
                    d += n * soft_thin_shift * (1.0 - trank.get(int(i), 0.0))
                return d
            keyed = [(pos + _delta(i), pos, i) for pos, i in enumerate(order)]
            keyed.sort(key=lambda t: (t[0], t[1]))
            return [t[2] for t in keyed]

        # HF-017 (v19): cog/stability 狙いの「重い物を先(=下)へ」順序後処理。v18 本番信号の分解で、
        # 重量物を低く積むと cog +3.5 / stability +3.8（num_placed を維持した順序レバーのみ有効。位置cog項
        # =v16 は num_placed を落とし逆効果）。fill 順を重量バケツで安定ソートし、バケツ内は fill 順維持
        # （fill/num_placed 保護）。優先は前方バケツに固定（placement 保護）。soft フラグでは並べ替えない
        # （soft 変形回避）。GH_HEAVY_LOW=バケツ数（既定 0/1=無効＝v14）。
        def _heavy_low_order(order):
            nb = int(os.environ.get("GH_HEAVY_LOW", "0") or "0")
            if nb <= 1:
                return order
            meta = {int(it["index"]): (float(it.get("mass", 0.0)), bool(it.get("is_prioritized")))
                    for it in item_list}
            by_w = sorted(order, key=lambda i: -meta.get(int(i), (0.0, False))[0])  # 重い順
            n = max(len(by_w), 1)
            bucket = {int(i): min(nb - 1, rank * nb // n) for rank, i in enumerate(by_w)}

            def _key(i):
                _w, prio = meta.get(int(i), (0.0, False))
                return 0 if prio else bucket.get(int(i), nb - 1)  # 優先は前方固定
            return sorted(order, key=_key)  # 安定ソート：バケツ内は fill 順維持

        def _interleave_hard_order(order):
            # HF-034: 順序レベルの摩擦対策（findings §17 の続き）。SOFT_SHIFT はカテゴリ全体を
            # 前後にずらすだけなので、hard品の連続（=hard-on-hard接触の起きやすい構成）自体は
            # 減らせない（二値シフトはSOFT_SHIFTと数学的に重複するため試さなかった、§18）。
            # 本関数は「hard品が GH_HARD_MAX_RUN 個連続したら、残りの列から直近のsoft品を
            # 1個前へ引き出す」構造的に異なる操作。全index を過不足なく1回ずつ保つ
            # （pending から pop して result へ append するだけなので契約は自動的に保たれる）。
            # 既定 0（無効、v38恒等）。
            max_run = int(os.environ.get("GH_HARD_MAX_RUN", "2") or "2")
            if max_run <= 0:
                return order
            soft = {int(it["index"]) for it in item_list if it.get("is_soft")}
            pending = list(order)
            result = []
            run = 0
            while pending:
                idx = int(pending[0])
                if idx not in soft and run >= max_run:
                    found = None
                    for j in range(1, len(pending)):
                        if int(pending[j]) in soft:
                            found = j
                            break
                    if found is not None:
                        result.append(pending.pop(found))
                        run = 0
                        continue
                result.append(pending.pop(0))
                run = 0 if idx in soft else run + 1
            return result

        def _post(order):
            return _interleave_hard_order(_heavy_low_order(_soft_shift_order(_reorder_quality(order))))

        if os.environ.get("GH_ORDER_PLAN", "1") == "0" or not getattr(self, "container_list", None):
            # HF-022: 品質目的の並べ替え(_post)は**計画経路にだけ**適用する。この分岐は
            # 「コンテナ形状が無い/計画を明示的に切った」異常系の安全網で、`composite_key_order` は
            # 不正な並べ替え材料に対して入力順へフォールバックする契約（§4.12 v1.23、
            # tests/test_t027_optimize.py）を持つ。その出力を質量や優先度で並べ替えるのは設計として誤り。
            # 本番は常に container_list があるため提出時の挙動は変わらない。
            return composite_key_order(item_list)
        budget_s = max(10.0, float(self.time_params.optimize_stop) - 30.0)
        # 方策3（HF-030）: 忠実 window=1 リプレイ（実 decide_placement 使用）を評価器にした
        # GRASP＋山登り順序探索。既定 OFF（GH_ORDER_SEARCH=0）で v38 と挙動不変。
        # J は _post 適用後の順序で評価する（レバーが結果を大きく変えるため、post_fn 適用前を
        # 評価すると実際の提出順序とズレる）。非退行フロア: seed に plan_order_forward_sim の
        # 出力（v38 base）を必ず含むため、探索がこれを一度も上回れなくても劣化しない。
        if os.environ.get("GH_ORDER_SEARCH", "0") != "0":
            from .packing_core.order import search_order_composite
            return _post(search_order_composite(
                item_list, self.container_list, budget_s, post_fn=_post))
        if os.environ.get("GH_ORDER_BEAM", "0") != "0":
            from .packing_core.order import plan_order_beam
            return _post(plan_order_beam(item_list, self.container_list, budget_s))
        return _post(plan_order_forward_sim(item_list, self.container_list, budget_s))

    def policy(self, observation: dict) -> dict:
        """観測から実候補を選び action を返す（§4.12 手順2〜7）。

        `build_state`→候補列挙→段階フィルタ→特徴計算→`heuristic_score`→
        `safe_decide`4層→`make_action` の順に配線する。`StepBudget` は本メソッド
        呼出しごとに1個だけ生成し、全パイプライン段階（`enumerate_candidates`/
        `filter_candidates`/`safe_decide`）で共有する。例外・候補ゼロ・全層None・
        action変換失敗のいずれの場合も、プロセス外へ例外を漏らさず最外殻emergency
        actionへフォールバックする（§4.11）。

        T-028: 本メソッドが telemetry accumulator（段階時間・局所telemetry・選択済み
        `p_ng_chosen`）の最外殻オーナーであり、正常・emergency全return経路について
        最外殻 `finally` から行整形・書込みを1回だけ試行する（§4.12/§4.13）。書込みの
        成否に関わらず `self._policy_step` を1増加する。writer例外は`policy()`外へ
        漏らさない。

        Args:
            observation: 現在の観測情報（変更しない）。

        Returns:
            dict: `item_idx` / `container_idx` / `place_pos` / `orientation` の
            4キーのみを持つ action 辞書。
        """
        step = self._policy_step
        first_step = step == 0
        timing = _new_timing()
        local_telemetry = _new_local_telemetry()
        png_holder = {"value": None}
        t_policy_start = time.monotonic()
        try:
            try:
                return self._policy_impl(observation, timing, local_telemetry, png_holder)
            except Exception:
                return _emergency_action(observation)
        finally:
            # t_total: policy入口からaction生成完了後、telemetry serialize/write開始直前まで
            # （writer I/O時間は含めない、§4.13）。
            timing["t_total"] = time.monotonic() - t_policy_start
            try:
                row = format_row(
                    step=step, first_step=first_step, timing=timing,
                    local_telemetry=local_telemetry, p_ng_chosen=png_holder["value"],
                )
                self._telemetry_writer.write_row(row)
            except Exception:
                pass
            self._policy_step += 1

    def _policy_impl(
        self, observation: dict, timing: dict, local_telemetry: dict, png_holder: dict,
    ) -> dict:
        """`policy()` の本体実装（例外は呼び出し側 `policy()` が最終捕捉する）。

        Args:
            observation: 現在の観測情報（変更しない）。
            timing: 呼び出し側が用意した段階時間辞書。各段階を`try/finally`で計測し
                in-place更新する（§4.13）。
            local_telemetry: 呼び出し側が用意した局所telemetry辞書。
                `n_cand0`/`n_after_dims`/`n_after_geo`/`n_lpath_pass`/
                `reject_reason_counts`をin-place更新し、`decided_layer`/`layer_error`は
                `safe_decide`が直接書き込む（再計算しない）。
            png_holder: 選択済みCandidateの`provisional_p_ng`を`"value"`キーへ保持する
                1キー辞書（make_action失敗後もemergencyへフォールバックする前に確定済み）。
        """
        # HF-005: init を observation から毎ステップ再構築する（再起動耐性）。観測は
        # optimize/lookahead_k/container_list を含み（OBS_KEYS）、container_list は静的形状
        # （center/n_vecs/points 等）と現在の packed_items の両方を持つため build_state の
        # 形状取得元として init と等価に使える。get_init_states 由来の self.* はフォールバック。
        # これによりワーカー再起動で get_init_states が失われても emergency に固定化されない。
        init = {
            "optimize": observation.get("optimize", self.optimize_enabled),
            "lookahead_k": observation.get("lookahead_k", self.lookahead_k),
            "container_list": observation.get("container_list", self.container_list),
        }
        return self._run_pipeline(
            observation,
            init,
            timing,
            local_telemetry,
            png_holder,
            allow_emergency=True,
        )

    def _run_pipeline(
        self,
        observation: dict,
        init: dict,
        timing: dict,
        local_telemetry: dict,
        png_holder: dict,
        *,
        allow_emergency: bool,
    ) -> dict:
        """実policyとT-027 warmupが共有する手順2〜7パイプライン。

        `allow_emergency=False`のwarmupでは未到達・変換失敗を例外として呼び出し側へ返し、
        `_run_warmup()`がwarningを残す。実policyは従来どおりemergency actionへ退避する。
        """
        t0 = time.monotonic()
        budget = StepBudget(
            t0=t0, soft=self.time_params.policy_soft, hard=self.time_params.policy_hard,
        )

        t_state_start = time.monotonic()
        try:
            # heightmap 配置エンジンは state.ems を使わない。EMS 全再構築（O(n_placed²)）を
            # 省いて late-game の policy 時間（pmax）を抑える（HF-012）。
            state = build_state(observation, init, compute_ems=False)
        except Exception:
            if not allow_emergency:
                raise
            return _emergency_action(observation)
        finally:
            timing["t_state"] = time.monotonic() - t_state_start

        _proxy_log_state(state)  # GH_PROXY_LOG 設定時のみ（H2 非fillプロキシ計測用、既定 no-op）

        # HF-012 Phase H1: heightmap 配置エンジンで1手を決定（EMS 列挙〜safe_decide を置換）。
        # build_state / StepBudget / telemetry 契約 / make_action / emergency は維持する。
        timing["t_enum"] = 0.0
        timing["t_mask"] = 0.0
        t_decide = time.monotonic()
        try:
            # HF-014: 課題A で optimize() が本物physics先読みロールアウトの plan を用意していれば追従。
            # plan は固定 index を鍵に持つ。observation の pool_list から可視荷物の固定 index を得て、
            # placed_order で最先の計画荷物を選び、その pool 位置を action の item_idx にする。
            # 計画に無い荷物しか見えない場合は従来の貪欲 decide にフォールバック（≥v14 を保証）。
            decided_tuple = None
            plan = getattr(self, "_rollout_plan", None)
            rank = getattr(self, "_rollout_rank", None)
            if plan:
                pool_list = observation.get("pool_list", []) or []
                visible = [(j, int(po["index"])) for j, po in enumerate(pool_list)
                           if int(po["index"]) in plan]
                if visible:
                    j, fid = min(visible, key=lambda t: rank.get(t[1], 1 << 30))
                    ci, pos, orn = plan[fid]
                    decided_tuple = (int(j), int(ci), np.asarray(pos, dtype=np.float64), int(orn))
            if decided_tuple is None:
                decided_tuple = decide_placement(state, budget)
        finally:
            timing["t_lpath"] = time.monotonic() - t_decide

        # telemetry（EMS 固有の候補数は heightmap では持たないため近似値を記録）。
        placed_ok = decided_tuple is not None
        local_telemetry["n_after_geo"] = 0
        local_telemetry["n_lpath_pass"] = 1 if placed_ok else 0
        local_telemetry["decided_layer"] = 1 if placed_ok else 0

        if decided_tuple is None:
            if not allow_emergency:
                raise RuntimeError("warmup heightmap engine returned no candidate")
            return _emergency_action(observation)

        item_idx, container_idx, pos_rel, orientation = decided_tuple
        png_holder["value"] = None  # H1: heightmap エンジンは provisional_p_ng を出さない

        try:
            return make_action(
                item_idx=item_idx,
                container_idx=container_idx,
                pos_rel=pos_rel,
                orientation=orientation,
            )
        except Exception:
            if not allow_emergency:
                raise
            return _emergency_action(observation)
