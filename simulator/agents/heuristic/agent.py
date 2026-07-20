"""heuristic Agent（T-023: 骨格／T-024: 手順2〜7／T-027: warmup・optimize）。

公式 `AgentFactory` からロードできる `Agent` を定義する。`policy()` は毎呼出しで
`build_state`→候補列挙→段階フィルタ→特徴計算→`heuristic_score`→`safe_decide`4層→
`make_action` を配線し（詳細仕様書 §4.12）、例外・候補ゼロ・全層None・action変換失敗の
いずれの場合も最外殻emergency actionへフォールバックしてプロセス外へ例外を漏らさない
（§4.11「最外殻emergency action」）。`__init__`は合成dummyで同じ手順2〜7を1回空回しし、
`optimize`は体積・重量で入力位置を並べて公式item index列を返す（§4.12 v1.23）。
"""
import functools
import logging
import math
import time

import numpy as np

from agents.heuristic.telemetry import format_row, make_writer_from_env
from src.packing_core.candidates import enumerate_candidates, filter_candidates
from src.packing_core.constants import (
    PlacementParams,
    ProvisionalRiskParams,
    ScoreParams,
    StageParams,
    TimeParams,
)
from src.packing_core.risk import provisional_p_ng
from src.packing_core.score import heuristic_score
from src.packing_core.stability import cg_margin, support_ratio
from src.packing_core.state import build_state, make_action
from src.packing_core.watchdog import (
    StepBudget,
    layer1_main,
    layer2_dblf_strict,
    layer3_first_fit,
    layer4_max_p,
    safe_decide,
)

logger = logging.getLogger("packing")


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
    """T-027の自己完結小型dummy init/observationを返す。"""
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
    placed_item = dict(item)
    placed_item.update(
        {
            "index": 99,
            "length": 0.20,
            "width": 0.20,
            "height": 0.20,
            "mass": 2.0,
            "belongs_to": 0,
            "pos": (0.0, 0.0, 0.12),
            "orn": (0.0, 0.0, 0.0, 1.0),
        }
    )
    observation_container = dict(container)
    observation_container["packed_items"] = [placed_item]
    init = {"optimize": False, "lookahead_k": 1, "container_list": [container]}
    observation = {
        "optimize": False,
        "lookahead_k": 1,
        "depth_map": np.zeros((1, 4, 4), dtype=np.float32),
        "container_list": [observation_container],
        "pool_list": [item],
    }
    return init, observation


def _emergency_action(observation: dict) -> dict:
    """最外殻emergency actionを返す（§4.11「最外殻emergency action」）。

    正常な `observation` から現在プール先頭の公式indexを読める場合はそれを使い、
    プールindexすら取得できない異常入力の場合に限りT-023の固定プレースホルダー
    `item_idx=0` を使う。

    Args:
        observation: `policy()` へ渡された観測（読み取りのみ、変更しない）。

    Returns:
        `item_idx=<pool先頭index or 0>, container_idx=0, pos_rel=(0.0,0.0,0.5),
        orientation=0` の action 辞書（`make_action()` 経由）。
    """
    item_idx = 0
    try:
        pool_list = observation["pool_list"]
        if len(pool_list) > 0:
            item_idx = int(pool_list[0]["index"])
    except Exception:
        item_idx = 0

    pos_rel = np.asarray((0.0, 0.0, 0.5), dtype=np.float64)
    return make_action(item_idx=item_idx, container_idx=0, pos_rel=pos_rel, orientation=0)


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

        numeric_types = (int, float, np.integer, np.floating)
        try:
            metrics: list[tuple[float, float]] = []
            for item in item_list:
                values = []
                for key in ("length", "width", "height", "mass"):
                    raw_value = item[key]
                    if isinstance(raw_value, (bool, np.bool_)) or not isinstance(
                        raw_value, numeric_types
                    ):
                        raise ValueError(f"{key} must be numeric")
                    value = float(raw_value)
                    if not math.isfinite(value):
                        raise ValueError(f"{key} must be finite")
                    values.append(value)

                length, width, height, mass = values
                if length <= 0.0 or width <= 0.0 or height <= 0.0 or mass < 0.0:
                    raise ValueError("dimensions must be positive and mass must be nonnegative")
                volume = length * width * height
                if not math.isfinite(volume):
                    raise ValueError("volume must be finite")
                metrics.append((volume, mass))

            sorted_positions = sorted(
                range(len(item_list)),
                key=lambda i: (-metrics[i][0], -metrics[i][1], i),
            )
        except Exception:
            return list(indices)

        return [indices[position] for position in sorted_positions]

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
        init = {
            "optimize": self.optimize_enabled,
            "lookahead_k": self.lookahead_k,
            "container_list": self.container_list,
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
            state = build_state(observation, init)
        except Exception:
            if not allow_emergency:
                raise
            return _emergency_action(observation)
        finally:
            timing["t_state"] = time.monotonic() - t_state_start

        t_enum_start = time.monotonic()
        try:
            raw_candidates = enumerate_candidates(
                state, self.placement_params, self.time_params, budget
            )
        finally:
            timing["t_enum"] = time.monotonic() - t_enum_start
        if len(raw_candidates) == 0:
            if not allow_emergency:
                raise RuntimeError("warmup generated no candidates")
            return _emergency_action(observation)

        t_mask_start = time.monotonic()
        try:
            pools = filter_candidates(
                state, raw_candidates, self.placement_params, self.time_params, budget
            )
        finally:
            timing["t_mask"] = time.monotonic() - t_mask_start
        if len(pools.dims_candidates) == 0:
            if not allow_emergency:
                raise RuntimeError("warmup generated no DIMS candidates")
            return _emergency_action(observation)

        # Agent配線契約（§4.12「Agent配線契約」）: dims_candidatesの各候補へ
        # support_ratio→cg_margin→provisional_p_ng→p_success→heuristic_score→features の順で設定。
        for cand in pools.dims_candidates:
            sr = support_ratio(state, cand)
            margin = cg_margin(state, cand)
            p_ng = provisional_p_ng(support_ratio=sr, cg_margin=margin, params=self.risk_params)
            cand.p_success = float(np.clip(1.0 - p_ng, 0.0, 1.0))
            cand.score = heuristic_score(state, cand, self.score_params)
            cand.features["support_ratio"] = float(sr)
            cand.features["cg_margin"] = float(margin)
            cand.features["provisional_p_ng"] = float(p_ng)

        # telemetryローカル辞書契約（§4.12）: filter_candidates完了後・safe_decide呼出し前に、
        # 呼び出し側から渡された辞書をin-place更新する。n_lpath_passはこの時点で0のまま
        # （path_candidatesはまだ空）。
        local_telemetry["n_cand0"] = len(pools.raw_candidates)
        local_telemetry["n_after_dims"] = len(pools.dims_candidates)
        local_telemetry["n_after_geo"] = len(pools.geo_candidates)
        local_telemetry["reject_reason_counts"] = dict(pools.reject_counts)

        layers = [
            functools.partial(
                layer1_main, pools=pools, pp=self.placement_params, stage_params=self.stage_params,
            ),
            functools.partial(
                layer2_dblf_strict,
                pools=pools, pp=self.placement_params, stage_params=self.stage_params,
            ),
            functools.partial(
                layer3_first_fit,
                pools=pools, pp=self.placement_params, stage_params=self.stage_params,
            ),
            functools.partial(
                layer4_max_p, pools=pools, pp=self.placement_params, stage_params=self.stage_params,
            ),
        ]

        t_lpath_start = time.monotonic()
        try:
            decided = safe_decide(layers, state, budget, local_telemetry)
        finally:
            # safe_decide呼出し後は戻り値の如何によらず、同一local_telemetry辞書のn_lpath_passを
            # 必ずlen(pools.path_candidates)で更新する（§4.12「telemetryローカル辞書契約」）。
            local_telemetry["n_lpath_pass"] = len(pools.path_candidates)
            timing["t_lpath"] = time.monotonic() - t_lpath_start

        if decided is None:
            if not allow_emergency:
                raise RuntimeError("warmup safe_decide returned no candidate")
            return _emergency_action(observation)

        # T-028 D5: 選択済みCandidateのprovisional_p_ngは、この後のaction変換
        # （make_action）が失敗しemergencyへフォールバックしても保持する。
        png_holder["value"] = decided.features.get("provisional_p_ng")

        try:
            return make_action(
                item_idx=decided.item_idx,
                container_idx=decided.container_idx,
                pos_rel=decided.pos_rel,
                orientation=decided.orientation,
            )
        except Exception:
            if not allow_emergency:
                raise
            return _emergency_action(observation)
