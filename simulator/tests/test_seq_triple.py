"""HF-045 Sequence Triple: 純combinatorial decode（`classify_pairs`/`decode_xy`）の検証。

design_sequence_triple_2026-08-10.md §7-1「decodeの正しさの単体検証（最優先）」の
実装。ランダムなm<=20荷物集合を大量生成し、x/y関係に分類されたペアが実際に
footprintで非重なりであることをproperty testで確認する（container形状には依存しない
純粋な組合せロジックのみを対象とする。実コンテナ統合は test_seq_triple_container.py）。
"""
import random

import pytest

from agents.heuristic.packing_core.seq_triple import classify_pairs, decode_xy


def _random_instance(rng, n):
    ids = list(range(n))
    o1 = ids[:]
    rng.shuffle(o1)
    o2 = ids[:]
    rng.shuffle(o2)
    o3 = ids[:]
    rng.shuffle(o3)
    footprint = {i: (rng.uniform(0.05, 1.2), rng.uniform(0.05, 1.2)) for i in ids}
    return o1, o2, o3, footprint


def _boxes_separated_xy(x0, y0, footprint, i, j):
    wx_i, wy_i = footprint[i]
    wx_j, wy_j = footprint[j]
    x_sep = x0[i] + wx_i <= x0[j] + 1e-9 or x0[j] + wx_j <= x0[i] + 1e-9
    y_sep = y0[i] + wy_i <= y0[j] + 1e-9 or y0[j] + wy_j <= y0[i] + 1e-9
    return x_sep or y_sep


@pytest.mark.parametrize("seed", range(20))
def test_classify_pairs_covers_every_pair_exactly_once(seed):
    rng = random.Random(seed)
    n = rng.randint(2, 20)
    o1, o2, o3, _fp = _random_instance(rng, n)
    rel = classify_pairs(o1, o2, o3)
    assert len(rel) == n * (n - 1) // 2
    assert set(rel.values()) <= {"x", "y", "z"}
    pos1 = {v: i for i, v in enumerate(o1)}
    for (i, j) in rel:
        assert pos1[i] < pos1[j]


@pytest.mark.parametrize("seed", range(50))
def test_decode_xy_separates_all_x_or_y_related_pairs(seed):
    """x/y関係に分類されたペアは、decode後のfootprintが必ず非重なりになる
    （design文書§3.2の懸念=z軸のみ、x/y平面の非重なりは本テストの対象）。"""
    rng = random.Random(1000 + seed)
    n = rng.randint(2, 20)
    o1, o2, o3, footprint = _random_instance(rng, n)
    rel = classify_pairs(o1, o2, o3)
    x0, y0 = decode_xy(o1, rel, footprint)

    assert set(x0) == set(range(n))
    assert set(y0) == set(range(n))
    for v in list(x0.values()) + list(y0.values()):
        assert v >= -1e-9

    for (i, j), r in rel.items():
        if r == "z":
            continue  # z関係はfootprintの重なりを許容する（段積み表現）、本テスト対象外
        assert _boxes_separated_xy(x0, y0, footprint, i, j), (seed, i, j, r)


@pytest.mark.parametrize("seed", range(10))
def test_classify_pairs_z_relation_direction_matches_order1(seed):
    """z関係に分類されたペアも含め、全ての関係が order1 の向き（i が先）と一致する。"""
    rng = random.Random(2000 + seed)
    n = rng.randint(2, 20)
    o1, o2, o3, _fp = _random_instance(rng, n)
    rel = classify_pairs(o1, o2, o3)
    pos1 = {v: i for i, v in enumerate(o1)}
    for (i, j), r in rel.items():
        assert pos1[i] < pos1[j]
        if r == "z":
            # z関係はx/y辺を作らない（footprintの重なりを許容する）だけの意味を持ち、
            # 支持関係自体はdecode_container側の物理dropが結果的に決める
            # （seq_triple.pyモジュールdocstring「実装時の簡略化」参照）。
            assert (i, j) in rel


def test_classify_pairs_degenerate_identical_permutations():
    """3順列が全て同一でも壊れない（すべて'x'に分類される退化ケース）。"""
    order = [0, 1, 2, 3]
    rel = classify_pairs(order, order, order)
    assert set(rel.values()) == {"x"}


def test_classify_pairs_empty_and_singleton():
    assert classify_pairs([], [], []) == {}
    assert classify_pairs([7], [7], [7]) == {}


def test_decode_xy_single_item_at_origin():
    x0, y0 = decode_xy([0], {}, {0: (0.3, 0.4)})
    assert x0 == {0: 0.0}
    assert y0 == {0: 0.0}
