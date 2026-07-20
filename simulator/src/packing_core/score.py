"""スコアリング（詳細仕様書 §4.8、T-024所有: `heuristic_score`）。

`regime`／`effective_score`（T-036所有）は本チケットでは実装しない（§4.8「v1.16 追記
（責務境界）」）。`cg_margin` はスコア式に含めない（`provisional_p_ng`・`layer4_max_p` 側でのみ
評価する）。HF-001（v1.27）：`z_top_norm`/`z_center_norm`は`cand.pos_rel[2]`ではなく
`stability.expected_settled_pos_rel`が返す想定沈降後Zを使う（§4.8「v1.27追記」）。
"""
import math

from src.packing_core import constants
from src.packing_core.constants import ScoreParams
from src.packing_core.stability import expected_settled_pos_rel, soft_below_ratio, support_ratio
from src.packing_core.types import Candidate


def _require_finite(value: float, label: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{label} は有限である必要があります: {value}")


def heuristic_score(state, cand: Candidate, sp: ScoreParams) -> float:
    """低さ・奥(+Y)・若いX・支持率を優先する線形結合スコアを返す（詳細仕様書 §4.8）。

    `cand.item_idx` は `state.pool` を `ItemSpec.idx` で一意検索して解決する
    （`state.pool[cand.item_idx]` という list 位置添字は行わない、v1.18 item解決契約）。
    `Candidate`/`PackingState`/`ItemSpec` を変更しない。`cand.score`/`cand.features` への
    代入は呼び出し側（agent.py、§4.12）の責務である。

    Args:
        state: 現在の `PackingState`。
        cand: 対象の配置候補。
        sp: スコア重みパラメータ。

    Returns:
        組込み `float`。有効入力では常に有限。

    Raises:
        ValueError: `cand.pos_rel`/`cand.osize` にNaN/Inf、`item.weight` が非有限、
            `space.inner_min_rel`/`inner_max_rel` が非有限、`span` のいずれかが0以下、
            `POOL_MAX_WEIGHT` が非有限または0以下、`support_ratio`/`soft_below_ratio` の
            戻り値が非有限、`sp` の重みが非有限、`cand.item_idx` に一致する `ItemSpec` が
            `state.pool` に一意に存在しない（0件または重複）場合、`cand.container_idx`/
            `cand.ems_id` が範囲外の場合（`expected_settled_pos_rel` 由来）。
    """
    matches = [item for item in state.pool if int(item.idx) == int(cand.item_idx)]
    if len(matches) != 1:
        raise ValueError(
            "cand.item_idx must match exactly one item in state.pool"
            f"（一致件数={len(matches)}, item_idx={cand.item_idx}）"
        )
    item = matches[0]

    for component, label in zip(cand.pos_rel, ("pos_rel.x", "pos_rel.y", "pos_rel.z")):
        _require_finite(float(component), label)
    for component, label in zip(cand.osize, ("osize.x", "osize.y", "osize.z")):
        _require_finite(float(component), label)
    _require_finite(float(item.weight), "item.weight")

    space = state.containers[cand.container_idx]
    inner_min = space.inner_min_rel
    inner_max = space.inner_max_rel
    for component, label in zip(inner_min, ("inner_min_rel.x", "inner_min_rel.y", "inner_min_rel.z")):
        _require_finite(float(component), label)
    for component, label in zip(inner_max, ("inner_max_rel.x", "inner_max_rel.y", "inner_max_rel.z")):
        _require_finite(float(component), label)

    span = inner_max - inner_min
    if not (float(span[0]) > 0.0 and float(span[1]) > 0.0 and float(span[2]) > 0.0):
        raise ValueError(f"span の全軸は正である必要があります: {span}")

    pool_max_weight = constants.POOL_MAX_WEIGHT
    _require_finite(float(pool_max_weight), "POOL_MAX_WEIGHT")
    if not float(pool_max_weight) > 0.0:
        raise ValueError(f"POOL_MAX_WEIGHT は正である必要があります: {pool_max_weight}")

    for attr in ("w_z", "w_y", "w_x", "w_support", "w_cg_h", "w_soft", "w_prio"):
        _require_finite(float(getattr(sp, attr)), attr)

    settled = expected_settled_pos_rel(state, cand)
    z_top_norm = _clip01(
        (float(settled[2]) + float(cand.osize[2]) / 2.0 - float(inner_min[2])) / float(span[2])
    )
    z_center_norm = _clip01((float(settled[2]) - float(inner_min[2])) / float(span[2]))
    y_center_norm = _clip01((float(cand.pos_rel[1]) - float(inner_min[1])) / float(span[1]))
    x_center_norm = _clip01((float(cand.pos_rel[0]) - float(inner_min[0])) / float(span[0]))
    weight_norm = _clip01(float(item.weight) / float(pool_max_weight))
    cg_height_norm = z_center_norm * weight_norm

    support_raw = float(support_ratio(state, cand))
    _require_finite(support_raw, "support_ratio")
    support = _clip01(support_raw)

    soft_below_raw = float(soft_below_ratio(state, cand))
    _require_finite(soft_below_raw, "soft_below_ratio")
    hard_on_soft_flag = float((not item.is_soft) and soft_below_raw > 0.0)

    priority_ok_flag = 0.0  # T-024固定

    score = (
        -sp.w_z * z_top_norm
        + sp.w_y * y_center_norm
        - sp.w_x * x_center_norm
        + sp.w_support * support
        - sp.w_cg_h * cg_height_norm
        - sp.w_soft * hard_on_soft_flag
        + sp.w_prio * priority_ok_flag
    )
    return float(score)


def _clip01(value: float) -> float:
    return min(max(value, 0.0), 1.0)
