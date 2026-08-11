"""HF-045 Sequence Triple: 実コンテナ統合（`decode_container`）の非重なり・非浮遊検証。

design_sequence_triple_2026-08-10.md §7-1「decodeの正しさの単体検証（最優先）」。
軸整列の単純な直方体コンテナ（`test_container_space.py::_box_cdict` と同型、cutなし・
棚なし）上で、ランダムな荷物集合をSequence Triple decodeし、
  (a) 確定配置された荷物どうしに3D AABB重なりが無いこと
  (b) 全荷物が「支持面に接している」こと（浮いていない）
をproperty testで確認する。
"""
import random

import numpy as np
import pytest

from agents.heuristic.packing_core import constants
from agents.heuristic.packing_core.container_space import build_container_space
from agents.heuristic.packing_core.geometry import aabb_intersects
from agents.heuristic.packing_core.seq_triple import _try_place_lower, decode_container
from agents.heuristic.packing_core.state import PackingState, rel_to_world
from agents.heuristic.packing_core.types import PlacedItem

CELL = constants.GridParams().cell
INNER_MIN_REL = np.array([-0.5, -0.5, 0.02], dtype=np.float64)
INNER_MAX_REL = np.array([0.5, 0.5, 1.5], dtype=np.float64)


def _box_cdict(index: int = 0) -> dict:
    """`tests/test_container_space.py::_box_cdict` と同型の、cutなし・棚なし単純直方体。"""
    imin, imax = INNER_MIN_REL, INNER_MAX_REL
    mid = (imin + imax) / 2.0
    face_specs = [
        ((imin[0], mid[1], mid[2]), (-1.0, 0.0, 0.0)),
        ((imax[0], mid[1], mid[2]), (1.0, 0.0, 0.0)),
        ((mid[0], imin[1], mid[2]), (0.0, -1.0, 0.0)),
        ((mid[0], imax[1], mid[2]), (0.0, 1.0, 0.0)),
        ((mid[0], mid[1], imin[2]), (0.0, 0.0, -1.0)),
        ((mid[0], mid[1], imax[2]), (0.0, 0.0, 1.0)),
    ]
    offset_x = 0.0
    points = [(px + offset_x, py, pz) for (px, py, pz), _ in face_specs]
    n_vecs = [n for _, n in face_specs]
    thickness = 0.02
    buffer = 0.02
    length = float(imax[0] - imin[0]) + 2 * thickness
    width = float(imax[1] - imin[1]) + 2 * thickness
    height = float(imax[2]) + buffer
    center = (offset_x, 0.0, height / 2.0 + buffer)
    return {
        "index": index, "length": length, "width": width, "height": height,
        "cut_x": 0.0, "cut_y": 0.0, "thickness": thickness, "center": center,
        "n_vecs": n_vecs, "points": points,
        "volume": float(np.prod(imax - imin)), "shelf": False,
        "is_prioritized": False, "packed_items": [],
    }


def _random_items(rng, n):
    """コンテナ(1m x 1m x 1.48m)に収まる程度の小さい荷物をランダム生成する。"""
    specs = {}
    for i in range(n):
        size = np.array([rng.uniform(0.08, 0.35), rng.uniform(0.08, 0.35), rng.uniform(0.08, 0.35)])
        specs[i] = {"size": size, "mass": rng.uniform(1.0, 10.0),
                    "is_soft": rng.random() < 0.3, "is_priority": rng.random() < 0.2}
    return specs


def _aabbs_from_placed(placed):
    return [(np.asarray(p.aabb_min_rel), np.asarray(p.aabb_max_rel)) for p in placed]


@pytest.mark.parametrize("seed", range(30))
def test_decode_container_no_3d_overlap(seed):
    rng = random.Random(seed)
    n = rng.randint(2, 15)
    space = build_container_space(_box_cdict(), 0, CELL)
    state = PackingState(containers=[space], placed={0: []}, pool=[], ems={0: []}, ems_truncation={0: 0.0}, meta={})

    specs = _random_items(rng, n)
    ids = list(specs.keys())
    o1 = ids[:]
    rng.shuffle(o1)
    o2 = ids[:]
    rng.shuffle(o2)
    o3 = ids[:]
    rng.shuffle(o3)
    orn_map = {i: rng.randrange(6) for i in ids}

    committed, plan_entries, failed = decode_container(space, state, 0, specs, o1, o2, o3, orn_map)

    assert set(committed) | failed == set(ids)
    assert set(plan_entries.keys()) == set(committed)

    boxes = _aabbs_from_placed(state.placed[0])
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            amin, amax = boxes[a]
            bmin, bmax = boxes[b]
            assert not aabb_intersects(amin, amax, bmin, bmax, tol=1e-6), (seed, a, b)


@pytest.mark.parametrize("seed", range(30))
def test_decode_container_no_floating_items(seed):
    """全ての確定配置は、底面が床基準線 or 他荷物の天面に接している（浮いていない）。"""
    rng = random.Random(1000 + seed)
    n = rng.randint(2, 15)
    space = build_container_space(_box_cdict(), 0, CELL)
    state = PackingState(containers=[space], placed={0: []}, pool=[], ems={0: []}, ems_truncation={0: 0.0}, meta={})

    specs = _random_items(rng, n)
    ids = list(specs.keys())
    o1 = ids[:]
    rng.shuffle(o1)
    o2 = ids[:]
    rng.shuffle(o2)
    o3 = ids[:]
    rng.shuffle(o3)
    orn_map = {i: rng.randrange(6) for i in ids}

    committed, _plan, _failed = decode_container(space, state, 0, specs, o1, o2, o3, orn_map)
    placed = state.placed[0]

    floor_z = float(np.asarray(space.floor_z).max())  # 床は cut なしなので一定
    for idx, target in enumerate(placed):
        bottom = float(target.aabb_min_rel[2])
        if bottom <= floor_z + constants.HM_WALL_CLEARANCE + 1e-3:
            continue  # 床置き
        # 直下（XY重なり）に、天面がこの荷物の底面と一致する荷物が無ければ浮いている。
        supported = False
        for other in placed[:idx] + placed[idx + 1:]:
            if other is target:
                continue
            ox0, oy0 = float(other.aabb_min_rel[0]), float(other.aabb_min_rel[1])
            ox1, oy1 = float(other.aabb_max_rel[0]), float(other.aabb_max_rel[1])
            tx0, ty0 = float(target.aabb_min_rel[0]), float(target.aabb_min_rel[1])
            tx1, ty1 = float(target.aabb_max_rel[0]), float(target.aabb_max_rel[1])
            x_overlap = min(ox1, tx1) - max(ox0, tx0)
            y_overlap = min(oy1, ty1) - max(oy0, ty0)
            if x_overlap <= 1e-6 or y_overlap <= 1e-6:
                continue
            if abs(float(other.aabb_max_rel[2]) - bottom) < 1e-3:
                supported = True
                break
        assert supported, (seed, idx, bottom)


def test_try_place_lower_refuses_hard_on_soft_when_no_other_room():
    """design文書§11.8: 本番`HM_SOFT_VETO`（既定1、v50ベイク）と同型のsoft veto回帰guard。

    床のほぼ全面（0.9m x 0.9m、コンテナ内寸1m x 1m）を覆うsoft品を先に置き、非soft品
    （0.5m x 0.5m）を側方の残り0.05mの隙間には収まらない大きさにすることで、
    「softの真上に着地する以外に幾何学的な選択肢が無い」局面を作る。veto無しなら
    真上への着地が採用されてしまうが、本番同型のvetoが効いていれば`None`
    （置き場所なし）を返すはずである。
    """
    space = build_container_space(_box_cdict(), 0, CELL)
    floor_z = float(np.asarray(space.floor_z).max())
    soft_size = np.array([0.9, 0.9, 0.1], dtype=np.float64)
    soft_half = soft_size / 2.0
    soft_center_rel = np.array([0.0, 0.0, floor_z + soft_half[2]], dtype=np.float64)
    soft_item = PlacedItem(
        pos_world=rel_to_world(soft_center_rel, space), orn_quat=np.array([0.0, 0.0, 0.0, 1.0]),
        size=soft_size, weight=5.0, is_soft=True, is_priority=False,
        aabb_min_rel=soft_center_rel - soft_half, aabb_max_rel=soft_center_rel + soft_half,
    )
    state = PackingState(containers=[space], placed={0: [soft_item]}, pool=[], ems={0: []},
                         ems_truncation={0: 0.0}, meta={})

    cand = _try_place_lower(
        space, state, 0, id_=1, center_xy=(0.0, 0.0), osize=(0.5, 0.5, 0.2), orn=0,
        margins=constants.HM_CASCADE, is_soft=False,
    )
    assert cand is None, "非soft品がsoft最上面へ着地する候補しか無い局面でvetoされていない"

    # 対照: soft品自身なら同じ位置へ着地できてよい（veto対象外）ことを確認する。
    cand_soft = _try_place_lower(
        space, state, 0, id_=2, center_xy=(0.0, 0.0), osize=(0.5, 0.5, 0.2), orn=0,
        margins=constants.HM_CASCADE, is_soft=True,
    )
    assert cand_soft is not None
