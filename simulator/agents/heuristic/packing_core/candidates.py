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

from .constants import PlacementParams, TimeParams
from .geometry import oriented_size
from .masks import MaskStage, evaluate_stage
from .types import Candidate, EMSBox, ItemSpec
from .watchdog import StepBudget

if TYPE_CHECKING:
    # 循環import回避（state.py は本モジュールをimportしないが、逆方向の実行時依存を
    # 作らないよう型チェック専用importに留める。masks.py/stability.py と同方針）。
    from .state import PackingState

CandidateKey = tuple[int, int, int, int, int]  # (item_idx, container_idx, orientation, ems_id, anchor)

# HF-003: 候補上限。多アンカー化で raw 候補が (コンテナ×EMS×向き×アンカー) と積算され
# 数千件に達し、下流の特徴量計算（cg_margin=凸包）が policy 時間予算(8s)を圧迫するため、
# enumerate 段で決定論的接頭辞として上限を課す。EMSは低z・大容量優先でソート済み
# （select_topn）なので、接頭辞は良質なEMSを保持する。
# HF-010: 1500→5000 に引上げ（ems_top_n 220 では 220×6向き×4アンカー≈5280 と積算されるため、
# 1500 では搬入可能候補を打ち切っていた）。2vCPU で最大 policy ≤2.6s と予算内、積載数 +44%。
MAX_RAW_CANDIDATES = 5000


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
    """`Candidate` から `(item_idx, container_idx, orientation, ems_id, anchor)` を射影する（§4.12）。

    HF-003: `anchor` を含める。同一(item,container,orientation,ems_id)でもアンカー違いで
    pos_rel が異なる複数候補が存在しうるため、L_PATHキャッシュキーの衝突（別位置候補の
    合否を誤って再利用する）を防ぐ。

    Args:
        cand: 対象の配置候補。

    Returns:
        全要素が組込み `int` の5要素タプル。
    """
    return (
        int(cand.item_idx),
        int(cand.container_idx),
        int(cand.orientation),
        int(cand.ems_id),
        int(cand.anchor),
    )


# HF-003/HF-007: side-load 挿入性のためのアンカー集合（決定順）。各要素は (x, y) で
# "min"=軸最小面寄せ / "max"=軸最大面寄せ / "mid"=軸中央。EMS床面上の複数位置に荷物を置く
# ことで、既配置荷物で搬入経路(L字, safety_margin=15mm)が塞がれても別位置で挿入できる確率を
# 上げる（多アンカーは v1→v2 で積載16%→28%を実証した唯一のレバー）。
# 先頭4隅は現行DBLF/壁寄せ（実証済み）で、予算打切り時も接頭辞として残す。HF-007 grid3 は
# さらに辺中点・中央を加え、幅広EMSで搬入経路の空いた中間X列へ置ける候補を増やす。
# HF-003: side-load 挿入性のためのアンカー集合（4隅、決定順）。DBLF(X若い側・Y奥)＋壁寄せ隅。
# 注記: HF-007 の 3×3グリッド(9点)はローカル2vCPUでは+6%だったが**プラットフォームで-9%回帰**した
# （ローカルA/Bはプラットフォームを予測しない）ため v2 実証の4隅へ戻した。
_ANCHORS: tuple[tuple[str, str], ...] = (
    ("min", "max"), ("max", "max"), ("min", "min"), ("max", "min"),
)


def _candidate_at_anchor(
    item: ItemSpec,
    container_idx: int,
    orientation: int,
    ems_id: int,
    ems: EMSBox,
    pp: PlacementParams,
    x_side: str,
    y_side: str,
    anchor: int = 0,
) -> Candidate | None:
    """指定アンカー（X/Y各 min|max 面寄せ / mid 中央）で1候補を生成する（§4.12、HF-003/HF-007）。

    Z は常にEMS支持面（`settled_z + z_generation_clearance`）。X/Y は `x_side`/`y_side` が
    `"min"`→軸最小面へ、`"max"`→軸最大面へ生成クリアランスだけ内側に寄せ、`"mid"`→軸中央。
    寸法確認を満たさなければ `None`。入力オブジェクト・配列は変更しない。
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

    def _axis(lo: float, hi: float, half: float, which: str) -> float:
        if which == "min":
            return lo + half + xy_generation_clearance
        if which == "max":
            return hi - half - xy_generation_clearance
        return (lo + hi) / 2.0  # "mid": 荷物をEMS中央へ（寸法確認済みなので範囲内）

    settled_z = float(ems.min_rel[2]) + float(osize[2]) / 2.0
    pos_x = _axis(float(ems.min_rel[0]), float(ems.max_rel[0]), float(osize[0]) / 2.0, x_side)
    pos_y = _axis(float(ems.min_rel[1]), float(ems.max_rel[1]), float(osize[1]) / 2.0, y_side)

    pos_rel = np.array([pos_x, pos_y, settled_z + z_generation_clearance], dtype=np.float64)

    return Candidate(
        item_idx=int(item.idx),
        container_idx=int(container_idx),
        ems_id=int(ems_id),
        orientation=int(orientation),
        pos_rel=pos_rel,
        osize=osize,
        anchor=int(anchor),
    )


def candidate_from_ems(
    item: ItemSpec,
    container_idx: int,
    orientation: int,
    ems_id: int,
    ems: EMSBox,
    pp: PlacementParams,
) -> Candidate | None:
    """1つの (item, container, orientation, EMS) 組合せからDBLF最小角候補を生成する（§4.12）。

    現行DBLF最小角（X=若い側／Y=奥(+Y)／Z=EMS支持面から`z_generation_clearance`だけ浮いた
    action位置）。想定沈降後位置は `stability.expected_settled_pos_rel` が別途導出する（§4.6）。
    単一候補を返す既存契約を維持する（多様なアンカーは `candidate_variants_from_ems`）。
    """
    return _candidate_at_anchor(
        item, container_idx, orientation, ems_id, ems, pp, x_side="min", y_side="max"
    )


def candidate_variants_from_ems(
    item: ItemSpec,
    container_idx: int,
    orientation: int,
    ems_id: int,
    ems: EMSBox,
    pp: PlacementParams,
) -> list[Candidate]:
    """1つの (item, container, orientation, EMS) 組合せから複数アンカー候補を生成する（§4.12、HF-003）。

    `_ANCHORS` の決定順で各隅の候補を生成し、`None`（寸法不足）と、XY位置が既出と一致する
    重複（EMSが荷物とほぼ同寸で min/max 隅が一致する場合）を除いた順序保存リストを返す。
    先頭は現行DBLF最小角であり、既存の単一候補挙動を接頭辞として含む。
    """
    variants: list[Candidate] = []
    seen: set[tuple[float, float]] = set()
    for anchor_idx, (x_side, y_side) in enumerate(_ANCHORS):
        cand = _candidate_at_anchor(
            item, container_idx, orientation, ems_id, ems, pp,
            x_side=x_side, y_side=y_side, anchor=anchor_idx,
        )
        if cand is None:
            continue
        key = (round(float(cand.pos_rel[0]), 6), round(float(cand.pos_rel[1]), 6))
        if key in seen:
            continue
        seen.add(key)
        variants.append(cand)
    return variants


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

    # 基本順序（pool → container → orientation → ems）を維持しつつ、HF-003で1EMSごとに
    # 壁寄せ隅を含む複数アンカーを展開する。多アンカーで候補数が (コンテナ×EMS×向き×アンカー)
    # と積算されるため、コンテナ単位の上限 `per_container_cap` を課して policy 時間予算(8s)を
    # 守る。上限をコンテナ単位にすることで、グローバル上限が後続コンテナを丸ごと落とす問題を
    # 避ける（単一コンテナでは実質 MAX_RAW_CANDIDATES）。
    n_containers = len(state.containers)
    per_container_cap = MAX_RAW_CANDIDATES
    if n_containers > 1:
        per_container_cap = max(1, MAX_RAW_CANDIDATES // n_containers)

    result: list[Candidate] = []
    n_processed = 0
    for item in state.pool:
        for container_idx in range(n_containers):
            container_start = len(result)
            stop_container = False
            for orientation in range(6):
                if stop_container:
                    break
                for ems_id, ems in enumerate(state.ems[container_idx]):
                    result.extend(
                        candidate_variants_from_ems(
                            item, container_idx, orientation, ems_id, ems, pp
                        )
                    )
                    n_processed += 1
                    if len(result) - container_start >= per_container_cap:
                        stop_container = True
                        break
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
