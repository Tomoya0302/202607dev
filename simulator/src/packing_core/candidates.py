"""候補列挙・段階フィルタの構築（詳細仕様書 §4.11「候補集合の共有」・§4.12「候補列挙・
候補生成契約」「candidates.py 公開API契約」、T-024所有）。

`CandidatePools`/`CandidateKey` は本モジュールが所有する（`types.py` には置かない、
§4.11「循環import回避契約」）。`enumerate_candidates`/`filter_candidates` が `StepBudget` を
参照するため、`watchdog.py` を通常の実行時importとして行う。逆方向
（`watchdog.py` → 本モジュール）は `typing.TYPE_CHECKING` 経由のみであり、実行時循環importは
発生しない（`tests/test_t024_import_contract.py`）。
"""
from dataclasses import dataclass, field

import numpy as np
from typing import TYPE_CHECKING

from src.packing_core.constants import PlacementParams, TimeParams
from src.packing_core.geometry import oriented_size
from src.packing_core.masks import MaskStage, evaluate_stage
from src.packing_core.types import Candidate, EMSBox, ItemSpec
from src.packing_core.watchdog import StepBudget

if TYPE_CHECKING:
    # 循環import回避（state.py は本モジュールをimportしないが、逆方向の実行時依存を
    # 作らないよう型チェック専用importに留める。masks.py/stability.py と同方針）。
    from src.packing_core.state import PackingState

CandidateKey = tuple[int, int, int, int]  # (item_idx, container_idx, orientation, ems_id)


@dataclass
class CandidatePools:
    """1回の `policy` 呼出しで全層が共有する候補集合（§4.11「候補集合の共有」）。

    Attributes:
        raw_candidates: `enumerate_candidates` が生成した全候補。
        dims_candidates: `raw_candidates` のうち DIMS 通過（`provisional_p_ng` 計算対象、§4.7）。
        geo_candidates: `dims_candidates` のうち INCLUSION+OVERLAP+CEILING も通過。
        path_candidates: `geo_candidates` のうち L_PATH 通過（層をまたいで随時追加される）。
        l_path_cache: `CandidateKey` → L_PATH 合否。`True`=合格、`False`=不合格、キー不在=未評価。
        reject_counts: 不合格理由（最初の不合格段階）ごとの件数。
    """

    raw_candidates: list[Candidate]
    dims_candidates: list[Candidate]
    geo_candidates: list[Candidate]
    path_candidates: list[Candidate] = field(default_factory=list)
    l_path_cache: dict[CandidateKey, bool] = field(default_factory=dict)
    reject_counts: dict[str, int] = field(default_factory=dict)


def candidate_key(cand: Candidate) -> CandidateKey:
    """`Candidate` から `(item_idx, container_idx, orientation, ems_id)` を射影する（§4.12）。

    Args:
        cand: 対象の配置候補。

    Returns:
        全要素が組込み `int` の4要素タプル。
    """
    return (
        int(cand.item_idx),
        int(cand.container_idx),
        int(cand.orientation),
        int(cand.ems_id),
    )


def candidate_from_ems(
    item: ItemSpec,
    container_idx: int,
    orientation: int,
    ems_id: int,
    ems: EMSBox,
    pp: PlacementParams,
) -> Candidate | None:
    """1つの (item, container, orientation, EMS) 組合せから候補を生成する（§4.12、HF-001でv1.27改訂）。

    DBLF最小角（X=若い側／Y=奥(+Y)／Z=EMS支持面から`z_generation_clearance`だけ浮いた
    action位置）で `pos_rel` を固定する。想定沈降後位置（EMS支持面へ底面一致する位置）は
    `pos_rel` とは別に `stability.expected_settled_pos_rel` が導出する（§4.6）。
    `prefilter_dims` 等のmask判定はここでは呼ばない。別EMSへの再割当は行わない（案A）。
    入力の `ItemSpec`/`EMSBox`/配列は変更しない。

    Args:
        item: 対象荷物の `ItemSpec`。
        container_idx: 対象コンテナ index。
        orientation: §3.2 の orientation コード（0..5）。
        ems_id: 対象 EMS の index（`state.ems[container_idx]` 内の0始まり位置）。
        ems: 対象 EMS。
        pp: 配置判定パラメータ。

    Returns:
        寸法確認（§4.12「候補生成前の寸法確認」）を満たせば新規 `Candidate`、満たさなければ `None`。
    """
    osize = np.array(oriented_size(item.size, orientation), dtype=np.float64)
    ems_size = ems.size()

    xy_required_clearance = max(-pp.inclusion_margin + pp.internal_extra, pp.internal_extra)
    xy_generation_clearance = xy_required_clearance + pp.candidate_generation_slack
    z_required_clearance = max(-pp.inclusion_margin + pp.internal_extra, pp.internal_extra)
    z_generation_clearance = z_required_clearance + pp.candidate_generation_slack

    if not (
        float(ems_size[0]) >= float(osize[0]) + xy_generation_clearance
        and float(ems_size[1]) >= float(osize[1]) + xy_generation_clearance
        and float(ems_size[2]) >= float(osize[2]) + z_generation_clearance
    ):
        return None

    settled_z = float(ems.min_rel[2]) + float(osize[2]) / 2.0
    pos_rel = np.array(
        [
            float(ems.min_rel[0]) + float(osize[0]) / 2.0 + xy_generation_clearance,
            float(ems.max_rel[1]) - float(osize[1]) / 2.0 - xy_generation_clearance,
            settled_z + z_generation_clearance,
        ],
        dtype=np.float64,
    )

    return Candidate(
        item_idx=int(item.idx),
        container_idx=int(container_idx),
        ems_id=int(ems_id),
        orientation=int(orientation),
        pos_rel=pos_rel,
        osize=osize,
    )


def enumerate_candidates(
    state: "PackingState", pp: PlacementParams, tp: TimeParams, budget: StepBudget,
) -> list[Candidate]:
    """基本順序（決定論）で全候補を列挙する（§4.11「基本順序」・§4.12「候補列挙・候補生成契約」）。

    `pool現在順 → container_idx昇順 → orientation昇順(0..5) → state.ems[container_idx]の
    ems_id昇順（0始まり、select_topn出力順と同義）` の完全直積を、`candidate_from_ems` が
    `None` を返さないものだけ順序維持のまま返す。mask判定・score・L_PATHは行わない。新しい
    `StepBudget` は生成しない。

    Args:
        state: 現在の `PackingState`。
        pp: 配置判定パラメータ。
        tp: 時間予算パラメータ（`budget_poll_every` を使用）。
        budget: 呼び出し側が1回だけ生成した `StepBudget`（全層で共有）。

    Returns:
        生成された `Candidate` のリスト（基本順序）。時間切れの場合は部分結果（先頭からの接頭辞）。

    Raises:
        ValueError: `tp.budget_poll_every <= 0` の場合。
    """
    if tp.budget_poll_every <= 0:
        raise ValueError(
            f"budget_poll_every は正の整数である必要があります: {tp.budget_poll_every}"
        )
    if budget.over_soft():
        return []

    result: list[Candidate] = []
    n_processed = 0
    for item in state.pool:
        for container_idx in range(len(state.containers)):
            for orientation in range(6):
                for ems_id, ems in enumerate(state.ems[container_idx]):
                    cand = candidate_from_ems(item, container_idx, orientation, ems_id, ems, pp)
                    if cand is not None:
                        result.append(cand)
                    n_processed += 1
                    if n_processed % tp.budget_poll_every == 0 and budget.over_soft():
                        return result
    return result


def _evaluate_candidate_stages(
    state: "PackingState", cand: Candidate, pp: PlacementParams, pools: CandidatePools
) -> None:
    """1候補をDIMS→INCLUSION→OVERLAP→CEILINGの順に短絡評価し `pools` を更新する（私有ヘルパ）。"""
    for stage in (MaskStage.DIMS, MaskStage.INCLUSION, MaskStage.OVERLAP, MaskStage.CEILING):
        evaluate_stage(state, cand, pp, stage)
        if not cand.feasible:
            pools.reject_counts[cand.reject_reason] = (
                pools.reject_counts.get(cand.reject_reason, 0) + 1
            )
            return
        if stage is MaskStage.DIMS:
            pools.dims_candidates.append(cand)
    pools.geo_candidates.append(cand)


def filter_candidates(
    state: "PackingState",
    raw_candidates: list[Candidate],
    pp: PlacementParams,
    tp: TimeParams,
    budget: StepBudget,
) -> CandidatePools:
    """`raw_candidates` へ DIMS→INCLUSION→OVERLAP→CEILING の短絡評価を適用する（§4.12）。

    最初の不合格段階だけを `reject_counts` へ計上し、不合格候補へ後続段階を適用しない。
    `path_candidates`/`l_path_cache` は空で初期化する（L_PATH・score・risk計算はここでは
    行わない）。

    Args:
        state: 現在の `PackingState`。
        raw_candidates: `enumerate_candidates` が返した全候補（基本順序）。
        pp: 配置判定パラメータ。
        tp: 時間予算パラメータ（`budget_poll_every` を使用）。
        budget: 呼び出し側が1回だけ生成した `StepBudget`（全層で共有）。

    Returns:
        `CandidatePools`。`raw_candidates` は入力全体をそのまま保持する（縮小しない）。
        時間切れの場合、`dims_candidates`/`geo_candidates`/`reject_counts` は処理済み
        prefixのみを反映し、未処理の候補は既定値のまま変更しない。

    Raises:
        ValueError: `tp.budget_poll_every <= 0` の場合。
    """
    if tp.budget_poll_every <= 0:
        raise ValueError(
            f"budget_poll_every は正の整数である必要があります: {tp.budget_poll_every}"
        )

    pools = CandidatePools(raw_candidates=raw_candidates, dims_candidates=[], geo_candidates=[])
    if budget.over_soft():
        return pools

    n_processed = 0
    for cand in raw_candidates:
        _evaluate_candidate_stages(state, cand, pp, pools)
        n_processed += 1
        if n_processed % tp.budget_poll_every == 0 and budget.over_soft():
            break
    return pools
