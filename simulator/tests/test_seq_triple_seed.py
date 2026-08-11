"""HF-045 Sequence Triple: 棚/列パッキングシード（`shelf_seed_genes`）の検証。

design_sequence_triple_2026-08-10.md §10.4(1)「多層シードの層間整合」実装の
regression guard。実装中に3件のバグ（重複id混入によるGamma1/2/3の不正な順列化、
`_GAP`未考慮によるfounderの壁超過、headroom_probe拒否による荷物の消失）を
発見・修正しており、再発防止のためproperty testとして固定する。
"""
import random

import numpy as np
import pytest

from agents.heuristic.packing_core import constants
from agents.heuristic.packing_core.container_space import build_container_space
from agents.heuristic.packing_core.geometry import aabb_intersects
from agents.heuristic.packing_core.seq_triple import (
    _flat_orientation, _shelf_rows, decode_all, fill_and_cog, item_footprint, shelf_seed_genes,
)

from test_seq_triple_container import CELL, _box_cdict


def _random_specs(rng, n):
    specs = {}
    for i in range(n):
        size = np.array([rng.uniform(0.08, 0.35), rng.uniform(0.08, 0.35), rng.uniform(0.08, 0.35)])
        specs[i] = {"size": size, "mass": rng.uniform(1.0, 10.0),
                    "is_soft": rng.random() < 0.3, "is_priority": rng.random() < 0.2}
    return specs


def _build_seed(space, specs, all_ids):
    orn_map = {i: _flat_orientation(specs[i]["size"]) for i in all_ids}
    order = sorted(all_ids, key=lambda i: -float(np.prod(specs[i]["size"])))
    footprint0 = {i: item_footprint(specs[i]["size"], orn_map[i])[:2] for i in all_ids}
    layer_height = {i: item_footprint(specs[i]["size"], orn_map[i])[2] for i in all_ids}
    width = float(space.inner_max_rel[0] - space.inner_min_rel[0])
    depth = float(space.inner_max_rel[1] - space.inner_min_rel[1])
    height = float(space.inner_max_rel[2] - space.inner_min_rel[2])
    g1, g2, g3, slot_fp = shelf_seed_genes(
        {0: order}, footprint0, {0: width}, {0: depth}, {0: height}, layer_height,
    )
    return g1, g2, g3, slot_fp, orn_map


@pytest.mark.parametrize("seed", range(20))
def test_shelf_seed_genes_is_a_valid_permutation(seed):
    """重複id混入バグ（実装時に確定率2/41への急落として発覚）の再発防止。"""
    rng = random.Random(seed)
    n = rng.randint(5, 60)  # 60はコンテナに収まらない量を含め、overflow経路も踏む
    space = build_container_space(_box_cdict(), 0, CELL)
    specs = _random_specs(rng, n)
    all_ids = list(specs.keys())
    g1, g2, g3, _slot_fp, _orn = _build_seed(space, specs, all_ids)
    for name, g in (("g1", g1), ("g2", g2), ("g3", g3)):
        assert len(g) == n, name
        assert sorted(g) == sorted(all_ids), name
        assert len(set(g)) == n, f"{name} has duplicate ids"


@pytest.mark.parametrize("seed", range(20))
def test_shelf_seed_genes_decode_no_overlap_no_float(seed):
    """列構造シード（slot_footprint込み）でdecodeしても非重なり・非浮遊が保たれる。"""
    rng = random.Random(1000 + seed)
    n = rng.randint(5, 60)
    space = build_container_space(_box_cdict(), 0, CELL)
    specs = _random_specs(rng, n)
    all_ids = list(specs.keys())
    g1, g2, g3, slot_fp, orn_map = _build_seed(space, specs, all_ids)

    committed_all, _plan, failed_all, state = decode_all(
        [space], {0: []}, specs, g1, g2, g3, orn_map, {i: 0 for i in all_ids}, slot_footprint=slot_fp,
    )
    assert set(committed_all[0]) | failed_all == set(all_ids)

    placed = state.placed[0]
    boxes = [(np.asarray(p.aabb_min_rel), np.asarray(p.aabb_max_rel)) for p in placed]
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            amin, amax = boxes[a]
            bmin, bmax = boxes[b]
            assert not aabb_intersects(amin, amax, bmin, bmax, tol=1e-6), (seed, a, b)

    floor_z = float(np.asarray(space.floor_z).max())
    for idx, target in enumerate(placed):
        bottom = float(target.aabb_min_rel[2])
        if bottom <= floor_z + constants.HM_WALL_CLEARANCE + 1e-3:
            continue
        supported = False
        for other in placed[:idx] + placed[idx + 1:]:
            ox0, oy0 = float(other.aabb_min_rel[0]), float(other.aabb_min_rel[1])
            ox1, oy1 = float(other.aabb_max_rel[0]), float(other.aabb_max_rel[1])
            tx0, ty0 = float(target.aabb_min_rel[0]), float(target.aabb_min_rel[1])
            tx1, ty1 = float(target.aabb_max_rel[0]), float(target.aabb_max_rel[1])
            if min(ox1, tx1) - max(ox0, tx0) <= 1e-6 or min(oy1, ty1) - max(oy0, ty0) <= 1e-6:
                continue
            if abs(float(other.aabb_max_rel[2]) - bottom) < 1e-3:
                supported = True
                break
        assert supported, (seed, idx, bottom)


def test_shelf_seed_genes_fill_and_cog_finite():
    rng = random.Random(42)
    space = build_container_space(_box_cdict(), 0, CELL)
    specs = _random_specs(rng, 40)
    all_ids = list(specs.keys())
    g1, g2, g3, slot_fp, orn_map = _build_seed(space, specs, all_ids)
    committed_all, _plan, _failed, state = decode_all(
        [space], {0: []}, specs, g1, g2, g3, orn_map, {i: 0 for i in all_ids}, slot_footprint=slot_fp,
    )
    fill, cog = fill_and_cog([space], state.placed)
    assert 0.0 <= fill <= 1.0
    assert cog >= 0.0
    assert sum(len(v) for v in committed_all.values()) > 0  # シードは何かしら置けること


@pytest.mark.parametrize("seed", range(10))
def test_shelf_rows_hostile_headroom_probe_never_loses_ids(seed):
    """headroom_probeが常に拒否しても、全idが必ずrowsかoverflowのどちらかに残る
    （実装時に発見: real_000相当タスクで41件中16件が消える不具合の再発防止。
    `_shelf_columns`の最終救済passがこの`_shelf_rows`にheadroom_probeを渡さない
    設計にしたため直接は踏まないが、`_shelf_rows`自体の契約として固定する）。
    """
    rng = random.Random(seed)
    n = rng.randint(1, 30)
    ids = list(range(n))
    footprint = {i: (rng.uniform(0.1, 0.5), rng.uniform(0.1, 0.5)) for i in ids}
    layer_height = {i: rng.uniform(0.05, 0.3) for i in ids}
    rows, overflow = _shelf_rows(
        ids, footprint, 2.0, float("inf"),
        headroom_probe=lambda cx, cy, wx, wy: 0.0, item_height=layer_height,
    )
    accounted = [i for row in rows for i in row] + overflow
    assert sorted(accounted) == sorted(ids)
    assert rows == []  # 頭上余裕ゼロなので全てoverflowへ


def test_shelf_columns_valid_permutation_with_cut_shelf_container():
    """cut/棚構造を持つ実コンテナ相当（`shelf=True`）でも順列が壊れないことを確認する
    （実装時に発見した3件目のバグ: headroom_probeが最終救済passにも適用され、
    拒否された荷物がGamma1/2/3から完全に消えていた）。
    """
    from test_seq_triple_container import CELL as _CELL

    cdict = _box_cdict()
    cdict = dict(cdict)
    cdict["cut_x"] = 0.15
    cdict["cut_y"] = 0.15
    cdict["shelf"] = True
    space = build_container_space(cdict, 0, _CELL)

    rng = random.Random(7)
    specs = _random_specs(rng, 45)  # コンテナ容量を明らかに超える数、overflow経路を強制する
    all_ids = list(specs.keys())
    g1, g2, g3, slot_fp = shelf_seed_genes(
        {0: sorted(all_ids, key=lambda i: -float(np.prod(specs[i]["size"])))},
        {i: item_footprint(specs[i]["size"], _flat_orientation(specs[i]["size"]))[:2] for i in all_ids},
        {0: float(space.inner_max_rel[0] - space.inner_min_rel[0])},
        {0: float(space.inner_max_rel[1] - space.inner_min_rel[1])},
        {0: float(space.inner_max_rel[2] - space.inner_min_rel[2])},
        {i: item_footprint(specs[i]["size"], _flat_orientation(specs[i]["size"]))[2] for i in all_ids},
        spaces={0: space},
    )
    for name, g in (("g1", g1), ("g2", g2), ("g3", g3)):
        assert sorted(g) == sorted(all_ids), name
        assert len(set(g)) == len(all_ids), f"{name} has duplicate/missing ids"
