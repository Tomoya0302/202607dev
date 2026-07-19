"""heuristic Agent（T-023: 骨格／T-024: 手順2〜7縦断実装）。

公式 `AgentFactory` からロードできる `Agent` を定義する。`policy()` は毎呼出しで
`build_state`→候補列挙→段階フィルタ→特徴計算→`heuristic_score`→`safe_decide`4層→
`make_action` を配線し（詳細仕様書 §4.12）、例外・候補ゼロ・全層None・action変換失敗の
いずれの場合も最外殻emergency actionへフォールバックしてプロセス外へ例外を漏らさない
（§4.11「最外殻emergency action」）。T-027以降が担う `__init__` ウォームアップ・`optimize`
本格化・JSONL出力・学習済み `RiskModel` はここでは実装しない。
"""
import functools
import time

import numpy as np

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
        """最小限の初期化のみ行う。

        モデル・設定ファイルの読込、ウォームアップ（T-027 責務）、状態構築、
        ファイル出力、telemetry 生成は行わない。パラメータ群は種別ごとに独立した
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
        """全荷物の積み込み順を決定する（T-023 は入力順を維持）。

        体積・重量等による並べ替えは T-027 の責務。本メソッドは `item_list` を
        変更せず、各荷物の公式 `item["index"]` を入力順のまま返す。

        Args:
            item_list: 全荷物の情報が格納された辞書のリスト。

        Returns:
            list[int]: 全荷物の公式 index を過不足なく1回ずつ含む、入力順のリスト。
        """
        return [int(item["index"]) for item in item_list]

    def policy(self, observation: dict) -> dict:
        """観測から実候補を選び action を返す（§4.12 手順2〜7）。

        `build_state`→候補列挙→段階フィルタ→特徴計算→`heuristic_score`→
        `safe_decide`4層→`make_action` の順に配線する。`StepBudget` は本メソッド
        呼出しごとに1個だけ生成し、全パイプライン段階（`enumerate_candidates`/
        `filter_candidates`/`safe_decide`）で共有する。例外・候補ゼロ・全層None・
        action変換失敗のいずれの場合も、プロセス外へ例外を漏らさず最外殻emergency
        actionへフォールバックする（§4.11）。

        Args:
            observation: 現在の観測情報（変更しない）。

        Returns:
            dict: `item_idx` / `container_idx` / `place_pos` / `orientation` の
            4キーのみを持つ action 辞書。
        """
        try:
            return self._policy_impl(observation)
        except Exception:
            return _emergency_action(observation)

    def _policy_impl(self, observation: dict) -> dict:
        """`policy()` の本体実装（例外は呼び出し側 `policy()` が最終捕捉する）。"""
        t0 = time.monotonic()
        budget = StepBudget(
            t0=t0, soft=self.time_params.policy_soft, hard=self.time_params.policy_hard,
        )

        try:
            init = {
                "optimize": self.optimize_enabled,
                "lookahead_k": self.lookahead_k,
                "container_list": self.container_list,
            }
            state = build_state(observation, init)
        except Exception:
            return _emergency_action(observation)

        raw_candidates = enumerate_candidates(
            state, self.placement_params, self.time_params, budget
        )
        if len(raw_candidates) == 0:
            return _emergency_action(observation)

        pools = filter_candidates(
            state, raw_candidates, self.placement_params, self.time_params, budget
        )
        if len(pools.dims_candidates) == 0:
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

        # telemetryローカル辞書契約（§4.12）: filter_candidates完了後・safe_decide呼出し前に
        # 新規辞書を生成する。n_lpath_passはこの時点で0初期化（path_candidatesはまだ空）。
        telemetry = {
            "n_cand0": len(pools.raw_candidates),
            "n_after_dims": len(pools.dims_candidates),
            "n_after_geo": len(pools.geo_candidates),
            "n_lpath_pass": 0,
            "reject_reason_counts": dict(pools.reject_counts),
            "decided_layer": 0,
            "layer_error": [],
        }

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

        try:
            decided = safe_decide(layers, state, budget, telemetry)
        finally:
            # safe_decide呼出し後は戻り値の如何によらず、同一telemetry辞書のn_lpath_passを
            # 必ずlen(pools.path_candidates)で更新する（§4.12「telemetryローカル辞書契約」）。
            telemetry["n_lpath_pass"] = len(pools.path_candidates)

        if decided is None:
            return _emergency_action(observation)

        try:
            return make_action(
                item_idx=decided.item_idx,
                container_idx=decided.container_idx,
                pos_rel=decided.pos_rel,
                orientation=decided.orientation,
            )
        except Exception:
            return _emergency_action(observation)
