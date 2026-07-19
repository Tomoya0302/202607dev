"""実行可能性判定（段階フィルタ）（詳細仕様書 §4.5、T-014〜T-017）。

T-014 時点では `check_inclusion`／`check_ceiling` のみを実装した。T-015 で `MaskStage`・
`prefilter_dims`・`check_overlap`・`evaluate_stage`（DIMS〜CEILINGの単一段階ディスパッチ）を
追加した。T-016B で `l_path_sweep_boxes`・`check_l_path`（純NumPy保守プロキシ、P4確定式、
出典: `validator.py::PlacementValidator.check_transport_path`／`_move_item`、転記元HEAD
`abf630f`）を実装した。`evaluate_stage` への `MaskStage.L_PATH` 接続は T-017 の対象外であり、
本ファイルでは変更しない（T-016B時点でも `MaskStage.L_PATH` 指定は `NotImplementedError`）。
"""
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from src.packing_core import geometry
from src.packing_core.constants import (
    CEILING_CLIP_SAFETY,
    EPS_GEOM,
    RESTING_SNAP_BAND,
    PlacementParams,
)
from src.packing_core.container_space import ContainerSpace, contains_oriented_box
from src.packing_core.types import Candidate, EMSBox, Vec3

if TYPE_CHECKING:
    # 循環import回避（state.py はこのモジュールをimportしないが、逆方向の実行時依存を
    # 作らないよう型チェック専用importに留める）。
    from src.packing_core.state import PackingState


def check_inclusion(space: ContainerSpace, cand: Candidate, pp: PlacementParams) -> bool:
    """候補が有効空間（内壁AABB・cut半空間・棚非交差）に収まるかを判定する（§4.5）。

    `contains_oriented_box` への委譲。margin は A14 の符号規約
    （`inclusion_margin` は負で厳格・正で緩和）を `contains_oriented_box` の margin 規約
    （正で厳格・負で緩和）へ変換するため `-pp.inclusion_margin + pp.internal_extra` とする
    （更新履歴 v1.9）。

    Args:
        space: 対象コンテナの `ContainerSpace`。
        cand: 判定対象の配置候補。`pos_rel`/`osize` を使用する。
        pp: 配置判定パラメータ。

    Returns:
        有効空間に収まっていれば True。
    """
    margin = -pp.inclusion_margin + pp.internal_extra
    return contains_oriented_box(space, cand.pos_rel, cand.osize, margin=margin)


def _xy_overlaps(a_min: Vec3, a_max: Vec3, b_min: Vec3, b_max: Vec3) -> bool:
    """2つのAABBのXY投影が有意に重なるかを判定する（§4.5、更新履歴 v1.10）。

    各軸の重なり幅を独立に `EPS_GEOM` と比較する。辺・角の接触（重なり幅=0）および
    数値誤差レベル（0〜EPS_GEOM）の重なりは重なりとみなさない。

    Args:
        a_min: 箱Aの最小点。shape (3,), float64。
        a_max: 箱Aの最大点。shape (3,), float64。
        b_min: 箱Bの最小点。shape (3,), float64。
        b_max: 箱Bの最大点。shape (3,), float64。

    Returns:
        両軸で重なり幅が `EPS_GEOM` を超えていれば True。
    """
    overlap_x = min(float(a_max[0]), float(b_max[0])) - max(float(a_min[0]), float(b_min[0]))
    overlap_y = min(float(a_max[1]), float(b_max[1])) - max(float(a_min[1]), float(b_min[1]))
    return overlap_x > EPS_GEOM and overlap_y > EPS_GEOM


def check_ceiling(space: ContainerSpace, cand: Candidate, pp: PlacementParams) -> bool:
    """候補上端と局所天井のクリアランスを判定する（§4.5）。

    局所天井は内壁上面と「候補とXY投影が有意に重なり、かつ棚下面が候補上端以上（＝候補より
    上方にある）棚」の下面の最小値。`cut_planes` は対象外（`check_inclusion` に委譲）、
    格子 `ceil_z` は絞り込み専用であり最終判定には使わない。候補より下方の棚は天井として
    扱わない。

    Args:
        space: 対象コンテナの `ContainerSpace`。
        cand: 判定対象の配置候補。`pos_rel`/`osize` を使用する。
        pp: 配置判定パラメータ。

    Returns:
        `box_top + (ceiling_margin + internal_extra) <= local_ceiling_z + EPS_GEOM` なら True。
    """
    box_min, box_max = geometry.aabb_from_center(cand.pos_rel, cand.osize)
    box_top = float(box_max[2])
    required_clearance = pp.ceiling_margin + pp.internal_extra

    local_ceiling_z = float(space.inner_max_rel[2])
    for shelf_min, shelf_max in space.shelf_boxes:
        shelf_min_z = float(shelf_min[2])
        if (
            _xy_overlaps(box_min, box_max, shelf_min, shelf_max)
            and shelf_min_z >= box_top - EPS_GEOM
        ):
            local_ceiling_z = min(local_ceiling_z, shelf_min_z)

    return bool(box_top + required_clearance <= local_ceiling_z + EPS_GEOM)


class MaskStage(IntEnum):
    """段階フィルタの段階識別子（§4.5、T-015確定）。

    物理定数・調整パラメータではなく `masks.py` 固有の処理識別子であるため `constants.py`
    には置かない。`IntEnum` は int 互換であり、`evaluate_stage(..., stage: int)` の
    シグネチャと両立する。

    Attributes:
        DIMS: EMS寸法プレフィルタ段階。
        INCLUSION: 内壁包含判定段階。
        OVERLAP: 既配置との重なり判定段階。
        CEILING: 天井クリアランス判定段階。
        L_PATH: L字経路判定段階（T-016で実装、T-017で `evaluate_stage` へ接続）。
    """

    DIMS = 0
    INCLUSION = 1
    OVERLAP = 2
    CEILING = 3
    L_PATH = 4


def prefilter_dims(cand: Candidate, ems: EMSBox) -> bool:
    """候補寸法がEMS寸法に収まるかを判定する（§4.5、最速の絞り込み段階）。

    Args:
        cand: 判定対象の配置候補。`osize` を使用する。
        ems: 対象EMS。`size()`（= `max_rel - min_rel`）を使用する。

    Returns:
        `cand.osize` の各軸が `ems.size()` 以下（境界一致を含む）なら True。
    """
    return bool(np.all(cand.osize <= ems.size()))


def check_overlap(state: "PackingState", cand: Candidate, tol: float) -> bool:
    """候補が同一コンテナの既配置荷物と重ならないかを判定する（§4.5）。

    候補AABBは `cand.pos_rel`/`cand.osize` から `geometry.aabb_from_center` で生成する。
    比較対象は `state.placed[cand.container_idx]` のみであり、他コンテナの既配置は参照
    しない。各既配置AABBと `geometry.aabb_intersects(..., tol=tol)` で判定し、1件でも
    交差すれば不合格とする。

    Args:
        state: 現在の `PackingState`。
        cand: 判定対象の配置候補。`container_idx`/`pos_rel`/`osize` を使用する。
        tol: `aabb_intersects` へ渡す許容量。呼び出し側（`evaluate_stage`）は
            `-pp.internal_extra` を渡し、隙間 `internal_extra` 未満を交差扱いとする。

    Returns:
        既配置と1件も交差しなければ True（既配置が空でも True）。
    """
    cand_min, cand_max = geometry.aabb_from_center(cand.pos_rel, cand.osize)
    for item in state.placed[cand.container_idx]:
        if geometry.aabb_intersects(
            cand_min, cand_max, item.aabb_min_rel, item.aabb_max_rel, tol=tol
        ):
            return False
    return True


def l_path_sweep_boxes(
    space: ContainerSpace, cand: Candidate, pp: PlacementParams
) -> tuple[tuple[Vec3, Vec3], tuple[Vec3, Vec3]]:
    """L字経路のYレグ・Xレグの連続スイープAABB（相対座標）を返す（§4.5、T-016B確定、A15）。

    出典: `src/ground_handling/validator.py::PlacementValidator.check_transport_path`
    L96-99,101-137（転記元HEAD `abf630f`）。公式は `lane_x`／`effective_start_z`／`rel_z` を
    算出したうえでYレグ（`y: entry -> target`, x=lane_x, z=rel_z 固定）→Xレグ
    （`x: lane_x -> target`, y=target, z=rel_z 固定）の順で候補を運ぶ（下降レグなし、z固定）。
    ORNS は全て90度倍数の軸整列回転のため、スイープ包絡は候補の実移動と厳密一致する
    （近似ではない）。

    Args:
        space: 対象コンテナの `ContainerSpace`（`path_*` 6フィールドを使用）。
        cand: 判定対象の配置候補。`pos_rel`（目標中心）／`osize`（回転後寸法）を使用する。
        pp: 配置判定パラメータ。`start_margin`／`start_z`／`ceiling_margin` を使用する
            （L字経路では `internal_extra` は使用しない）。

    Returns:
        `(y_leg, x_leg)`。各要素は `(min, max)` のAABB（コンテナ相対）。要素0=Yレグ、
        要素1=Xレグ。Xレグはスイープ長ゼロ（`lane_x==target_x`）でも必ず2件目として返す
        （候補自身の半幅のみを持つ非退化AABBになる）。
    """
    half = cand.osize / 2.0
    target = cand.pos_rel

    # 入口レーンx（A12/A15）: 公式 x_min/x_max は candidate（half_lwh[0]）・validator設定
    # （start_margin）依存のため、space の幾何基底へ使用時に合成する。
    lane_x_min = space.path_lane_x_min_geom_rel + half[0] + pp.start_margin
    lane_x_max = space.path_lane_x_max_geom_rel - half[0] - pp.start_margin
    lane_x = min(max(float(target[0]), lane_x_min), lane_x_max)  # validator.py L99 と同じ順序

    resting_surfaces = (float(space.inner_min_rel[2]), space.path_mid_resting_z_rel)
    ceiling_surfaces = (space.path_mid_ceiling_z_rel, float(space.inner_max_rel[2]))

    # 2. 直置き判定（validator.py L117-122）。
    bottom_z = float(target[2]) - half[2]
    effective_start_z = pp.start_z
    for r_z in resting_surfaces:
        if 0.0 <= (bottom_z - r_z) <= RESTING_SNAP_BAND:
            effective_start_z = 0.0
            break

    # 3. 頭打ち回避クリップ（validator.py L124-133）。
    top_z = float(target[2]) + half[2]
    if effective_start_z > 0.0:
        for c_z in ceiling_surfaces:
            clearance = c_z - top_z
            if 0.0 <= clearance < (effective_start_z + pp.ceiling_margin):
                effective_start_z = max(0.0, clearance - pp.ceiling_margin - CEILING_CLIP_SAFETY)
                break

    # validator.py L135: rel_z = min(height+buffer-thickness-half_z-start_margin, target_z+eff)。
    # height+buffer-thickness は inner_max_rel[2] と恒等的に一致する（A15、§4.2）。
    rel_z = min(float(space.inner_max_rel[2]) - half[2] - pp.start_margin, float(target[2]) + effective_start_z)

    entry_y = space.path_entry_y_rel
    target_y = float(target[1])
    target_x = float(target[0])

    y_lo, y_hi = (entry_y, target_y) if entry_y <= target_y else (target_y, entry_y)
    y_leg_min = np.array([lane_x - half[0], y_lo - half[1], rel_z - half[2]], dtype=np.float64)
    y_leg_max = np.array([lane_x + half[0], y_hi + half[1], rel_z + half[2]], dtype=np.float64)

    x_lo, x_hi = (lane_x, target_x) if lane_x <= target_x else (target_x, lane_x)
    x_leg_min = np.array([x_lo - half[0], target_y - half[1], rel_z - half[2]], dtype=np.float64)
    x_leg_max = np.array([x_hi + half[0], target_y + half[1], rel_z + half[2]], dtype=np.float64)

    return (y_leg_min, y_leg_max), (x_leg_min, x_leg_max)


def check_l_path(state: "PackingState", cand: Candidate, pp: PlacementParams) -> bool:
    """L字経路の純NumPy保守プロキシ（P4確定式）で判定する（§4.5、T-016B確定、v1.13）。

    プロキシ = P4：全レグ×全障害物（`state.placed[cand.container_idx]` の既配置AABB ∪
    `state.containers[cand.container_idx].path_obstacle_boxes_rel`）について、軸ごとの隙間
    `gap_i`（重なる軸は0）から `distance_sq = Σgap_i²` を求め、`distance_sq > (pp.safety_margin
    + EPS_GEOM) ** 2` を要求する。L字経路では `pp.internal_extra` を加算しない。片側含意
    `check_l_path()==True ⇒ 公式 PlacementValidator.check_transport_path()==True` を満たす
    （証明: docs/実装詳細仕様書.md §4.5「L字経路：プロキシ確定仕様（v1.13）」）。格子は使わない。

    Args:
        state: 現在の `PackingState`。
        cand: 判定対象の配置候補。
        pp: 配置判定パラメータ。`safety_margin` を使用する。

    Returns:
        全レグ・全障害物ペアが `safety_margin` を超えて分離していれば True。
    """
    space = state.containers[cand.container_idx]
    leg_boxes = l_path_sweep_boxes(space, cand, pp)

    obstacles: list[tuple[Vec3, Vec3]] = [
        (item.aabb_min_rel, item.aabb_max_rel) for item in state.placed[cand.container_idx]
    ]
    obstacles.extend(space.path_obstacle_boxes_rel)

    threshold_sq = (pp.safety_margin + EPS_GEOM) ** 2

    for leg_min, leg_max in leg_boxes:
        for obs_min, obs_max in obstacles:
            gap = np.maximum(0.0, np.maximum(leg_min - obs_max, obs_min - leg_max))
            distance_sq = float(np.dot(gap, gap))
            if not (distance_sq > threshold_sq):
                return False
    return True


def evaluate_stage(
    state: "PackingState", cand: Candidate, pp: PlacementParams, stage: int
) -> Candidate:
    """指定された単一段階の判定だけを実行し、結果を `cand` へ反映して返す（§4.5）。

    累積短絡パイプラインではない単一段階ディスパッチである。`feasible=True` は
    「指定された今回の段階を通過した」ことのみを意味し、全段階通過の判定は呼び出し側
    （§4.12）が既定順（DIMS→INCLUSION→OVERLAP→CEILING→（上位Mのみ）L_PATH）で本関数を
    繰り返し呼び出すことで構成する。

    Args:
        state: 現在の `PackingState`。
        cand: 判定対象の配置候補。判定結果に応じて `feasible`/`reject_reason` を
            書き換える（同一オブジェクトを変更する）。
        pp: 配置判定パラメータ。
        stage: `MaskStage` の値（int互換）。

    Returns:
        更新後の `cand`（引数と同一オブジェクト）。

    Raises:
        ValueError: `stage` が `MaskStage` に未定義の値の場合。
        IndexError: `MaskStage.DIMS` において `cand.ems_id` が
            `state.ems[cand.container_idx]` の範囲外の場合（自然な例外を握りつぶさない）。
        NotImplementedError: `stage` が `MaskStage.L_PATH` の場合（T-015時点では
            L字経路判定を実装しない。`cand` を変更する前に送出する）。
    """
    stage_enum = MaskStage(stage)

    if stage_enum is MaskStage.DIMS:
        ems = state.ems[cand.container_idx][cand.ems_id]
        passed = prefilter_dims(cand, ems)
        reason = "dims"
    elif stage_enum is MaskStage.INCLUSION:
        space = state.containers[cand.container_idx]
        passed = check_inclusion(space, cand, pp)
        reason = "inclusion"
    elif stage_enum is MaskStage.OVERLAP:
        passed = check_overlap(state, cand, tol=-pp.internal_extra)
        reason = "overlap"
    elif stage_enum is MaskStage.CEILING:
        space = state.containers[cand.container_idx]
        passed = check_ceiling(space, cand, pp)
        reason = "ceiling"
    else:
        # MaskStage.L_PATH: T-016で l_path_sweep_boxes/check_l_path を実装し、T-017で
        # ここへ接続する。T-015時点では cand を変更する前に例外を送出する。
        raise NotImplementedError

    if passed:
        cand.feasible = True
        cand.reject_reason = ""
    else:
        cand.feasible = False
        cand.reject_reason = reason
    return cand
