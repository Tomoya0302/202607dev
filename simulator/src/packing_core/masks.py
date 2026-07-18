"""実行可能性判定（段階フィルタ）（詳細仕様書 §4.5、T-014〜T-017）。

T-014 時点では `check_inclusion`／`check_ceiling` のみを実装した。T-015 で `MaskStage`・
`prefilter_dims`・`check_overlap`・`evaluate_stage`（DIMS〜CEILINGの単一段階ディスパッチ）を
追加する。`l_path_sweep_boxes`・`check_l_path`、および `evaluate_stage` への `MaskStage.L_PATH`
接続は T-016／T-017 の対象外であり、本ファイルではスタブも含め追加しない
（T-015時点の `MaskStage.L_PATH` 指定は `NotImplementedError`）。
"""
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from src.packing_core import geometry
from src.packing_core.constants import EPS_GEOM, PlacementParams
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
