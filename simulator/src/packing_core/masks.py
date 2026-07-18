"""実行可能性判定（段階フィルタ）（詳細仕様書 §4.5、T-014〜T-017）。

T-014 時点では `check_inclusion`／`check_ceiling` のみを実装する。`prefilter_dims`・
`check_overlap`・`evaluate_stage`・`l_path_sweep_boxes`・`check_l_path`（T-015以降）は
本ファイルの対象外であり、スタブも含め追加しない。
"""
from src.packing_core import geometry
from src.packing_core.constants import EPS_GEOM, PlacementParams
from src.packing_core.container_space import ContainerSpace, contains_oriented_box
from src.packing_core.types import Candidate, Vec3


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
