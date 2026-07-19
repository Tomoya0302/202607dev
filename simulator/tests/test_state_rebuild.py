"""T-013a: packing_core.state の増分キャッシュ契約テスト（実装詳細仕様書 §4.4「増分キャッシュ
（T-013／`build_state_cached`）」節、§4.2 `cache_bake_placed`、§4.3 EMS丸め比較、§6 T-013a/T-013b、
`docs/初期検討_実装手順書.md` Phase 1）。

`build_state_cached`/`StateCache` は本チケット時点で未実装（本体実装は T-013b で別途行う）。
`--collect-only` を成功させるため、対象シンボルへの参照は各テスト関数の内部で行う
（`tests/test_state.py` と同方針）。skip/xfail/仮実装は禁止。実行時は `state.StateCache`/
`state.build_state_cached` が存在しないため `AttributeError` で失敗する想定。

このファイルは `tests/test_state.py` の入力ビルダー（`_item`/`_container`/`_container_lists`/
`_init`/`_observation`/`_grid_positions`/`_deep_equal`）と同一仕様のヘルパを自己完結に複製する
（本リポジトリには `tests/__init__.py`・`conftest.py` が無く、テストモジュール間の import は
既存ファイルでも行われていないため、それに倣う）。

契約の要点（詳細は §4.4）:
    * `build_state` は全再構築の唯一の正として不変。
    * `build_state_cached(observation, init, cache=None) -> (PackingState, StateCache)` は
      キャッシュヒット時のみ増分更新し、無効化時は該当コンテナだけ全再構築へフォールバックする。
      戻り値の `PackingState` はヒット/無効化の別によらず、常に `build_state` と同値
      （height は `np.array_equal` 完全一致、EMSは1e-6丸め6タプルのソート済み列一致）。
    * `StateCache`（`frozen=True`）の `placed_signatures[idx]` はAABB署名の**多重集合**
      （重複個数を保持するソート済み tuple。`frozenset` は使わない）。
    * `heights[idx]` は返却する `PackingState` とメモリを共有しないコピーで `writeable=False`。

各テストの docstring に、そのテストが踏むべき経路（ヒット/無効化のどちらを検証するか）を
1行で明記する（人間による目視確認用）。
"""
import copy
import math

import numpy as np
import pytest

from src.packing_core import constants

CELL = constants.GridParams().cell
IDENTITY_QUAT: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

# 汎用テスト用の内壁範囲（cut/棚なし直方体）。tests/test_state.py と同じ値。
DEFAULT_INNER_MIN = np.array([-0.4, -0.5, 0.02], dtype=np.float64)
DEFAULT_INNER_MAX = np.array([0.4, 0.5, 1.0], dtype=np.float64)


# --- 入力ビルダー（tests/test_state.py の同名ヘルパと同一仕様の複製） -----------------------

def _item(
    index: int,
    size: tuple[float, float, float],
    mass: float,
    is_soft: bool = False,
    is_priority: bool = False,
    pos: np.ndarray | tuple[float, float, float] | None = None,
    orn: np.ndarray | tuple[float, float, float, float] | None = None,
    belongs_to: int | None = None,
) -> dict:
    """§E item/pool_list 要素の共通キーを持つ辞書を組み立てる（世界座標の中心 pos を使う）。"""
    length, width, height = (float(v) for v in size)
    item = {
        "index": index,
        "length": length,
        "width": width,
        "height": height,
        "mass": float(mass),
        "is_prioritized": bool(is_priority),
        "is_soft": bool(is_soft),
        "belongs_to": belongs_to,
        "pos": None if pos is None else np.array(pos, dtype=np.float64),
        "orn": None if orn is None else np.array(orn, dtype=np.float64),
        "lateralFriction": 0.5,
        "rollingFriction": 0.01,
        "spinningFriction": 0.01,
        "restitution": 0.1,
        "angularDamping": 0.05,
    }
    if is_soft:
        item["contactStiffness"] = 1000.0
        item["contactDamping"] = 10.0
        item["linearDamping"] = 0.02
    return item


def _container(
    index: int,
    offset_x: float,
    inner_min_rel: np.ndarray,
    inner_max_rel: np.ndarray,
    packed_items: list[dict] | None = None,
    thickness: float = 0.02,
    buffer: float = 0.02,
) -> dict:
    """軸整列直方体の container_list 要素（cdict）を組み立てる（cut_x=cut_y=0、shelf=False）。"""
    imin = np.asarray(inner_min_rel, dtype=np.float64)
    imax = np.asarray(inner_max_rel, dtype=np.float64)
    mid = (imin + imax) / 2.0

    face_specs = [
        ((imin[0], mid[1], mid[2]), (-1.0, 0.0, 0.0)),
        ((imax[0], mid[1], mid[2]), (1.0, 0.0, 0.0)),
        ((mid[0], imin[1], mid[2]), (0.0, -1.0, 0.0)),
        ((mid[0], imax[1], mid[2]), (0.0, 1.0, 0.0)),
        ((mid[0], mid[1], imin[2]), (0.0, 0.0, -1.0)),
        ((mid[0], mid[1], imax[2]), (0.0, 0.0, 1.0)),
    ]
    points = [(px + offset_x, py, pz) for (px, py, pz), _ in face_specs]
    n_vecs = [n for _, n in face_specs]

    length = float(imax[0] - imin[0]) + 2 * thickness
    width = float(imax[1] - imin[1]) + 2 * thickness
    height = float(imax[2]) + buffer

    assert abs(imin[2] - thickness) < 1e-9, "inner_min_rel[2] は thickness と自己無矛盾である必要がある"

    center = (float(offset_x), 0.0, height / 2.0 + buffer)

    return {
        "index": index,
        "length": length,
        "width": width,
        "height": height,
        "cut_x": 0.0,
        "cut_y": 0.0,
        "thickness": thickness,
        "center": center,
        "n_vecs": n_vecs,
        "points": points,
        "volume": float(np.prod(imax - imin)),
        "shelf": False,
        "is_prioritized": False,
        "packed_items": list(packed_items) if packed_items else [],
    }


def _container_lists(
    specs: list[tuple[int, float, np.ndarray, np.ndarray]],
    packed_by_index: dict[int, list[dict]] | None = None,
) -> tuple[list[dict], list[dict]]:
    """init 用・observation 用の container_list を別オブジェクトとして2つ生成する
    （init 側は常に packed_items=[]、observation 側にのみ packed_items を格納する）。"""
    packed_by_index = packed_by_index or {}
    init_list = [_container(idx, off, imin, imax, packed_items=[]) for idx, off, imin, imax in specs]
    obs_list = [
        _container(idx, off, imin, imax, packed_items=packed_by_index.get(idx, []))
        for idx, off, imin, imax in specs
    ]
    return init_list, obs_list


def _init(container_list: list[dict], optimize: bool = True, lookahead_k: int = 5) -> dict:
    """§E init_states のキー（optimize, lookahead_k, container_list）を持つ辞書を返す。"""
    return {
        "optimize": optimize,
        "lookahead_k": lookahead_k,
        "container_list": container_list,
    }


def _observation(
    container_list: list[dict],
    pool_list: list[dict],
    optimize: bool = True,
    lookahead_k: int = 5,
) -> dict:
    """§E observation のキー（optimize, lookahead_k, depth_map, container_list, pool_list）を持つ辞書を返す。"""
    n = len(container_list)
    return {
        "optimize": optimize,
        "lookahead_k": lookahead_k,
        "depth_map": np.zeros((n, 64, 64), dtype=np.float32),
        "container_list": container_list,
        "pool_list": pool_list,
    }


def _deep_equal(a, b) -> bool:
    """dict/list/tuple/np.ndarray を含む入れ子構造の深い等価判定（入力辞書の非改変検証用）。"""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_deep_equal(x, y) for x, y in zip(a, b))
    return a == b


def _grid_positions(
    n: int,
    inner_min_rel: np.ndarray,
    inner_max_rel: np.ndarray,
    item_size: float,
    pitch: float,
) -> list[np.ndarray]:
    """内壁範囲内に収まる、相互非重複なグリッド中心（コンテナ相対）を n 個返す。"""
    assert pitch >= item_size, "pitch は item_size 以上である必要がある（非重複を保証するため）"
    half = item_size / 2.0
    x0 = float(inner_min_rel[0]) + half
    y0 = float(inner_min_rel[1]) + half
    z0 = float(inner_min_rel[2]) + half
    nx = int((float(inner_max_rel[0]) - float(inner_min_rel[0]) - item_size) // pitch) + 1
    ny = int((float(inner_max_rel[1]) - float(inner_min_rel[1]) - item_size) // pitch) + 1

    positions: list[np.ndarray] = []
    for iy in range(ny):
        for ix in range(nx):
            if len(positions) >= n:
                return positions
            positions.append(np.array([x0 + ix * pitch, y0 + iy * pitch, z0], dtype=np.float64))
    raise ValueError(f"グリッド容量不足: {n}個の非重複位置を生成できません（容量={nx * ny}）")


def _scattered_disjoint_aabbs(
    rng: np.random.Generator, n: int, inner_min: np.ndarray, inner_max: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """内壁範囲内・相互XY非重複な n 個のAABB（コンテナ相対、全て floor から起立）を、
    固定シードの `rng` から決定的に構築する（test_state_rebuild_ems_truncation_...専用）。

    全AABBが同じ床 `inner_min[2]` から起立するため、XY footprint が非重複なら3D非重複が
    保証される。サイズ・位置をばらつかせることで `generate_ems` の候補数を素早く増やし、
    `StageParams().ems_top_n_per_container`（=80）を超える決定論的fixtureを作る
    （作成時に実際に `generate_ems` を実行し、n=9で72件・n=10で84件になることを確認済み）。
    """
    aabbs: list[tuple[np.ndarray, np.ndarray]] = []
    tries = 0
    while len(aabbs) < n and tries < n * 50:
        tries += 1
        size = rng.uniform(0.03, 0.08, size=3)
        size[2] = rng.uniform(0.05, 0.3)
        cx = rng.uniform(inner_min[0] + size[0] / 2, inner_max[0] - size[0] / 2)
        cy = rng.uniform(inner_min[1] + size[1] / 2, inner_max[1] - size[1] / 2)
        bmin = np.array([cx - size[0] / 2, cy - size[1] / 2, inner_min[2]])
        bmax = np.array([cx + size[0] / 2, cy + size[1] / 2, inner_min[2] + size[2]])
        overlaps = any(
            np.all(bmin < amax) and np.all(amin < bmax) for amin, amax in aabbs
        )
        if not overlaps:
            aabbs.append((bmin, bmax))
    if len(aabbs) < n:
        raise ValueError(f"disjoint AABB生成に失敗: {len(aabbs)}/{n}")
    return aabbs


# --- 比較ヘルパ（§4.3/§4.4と同じ丸め比較） -------------------------------------------

def _ems_signature(ems_list) -> tuple:
    """EMSBox一覧を、各座標1e-6丸めの6要素tupleのソート済み列へ変換する（§4.3/§4.4）。"""
    return tuple(sorted(
        tuple(round(float(v), 6) for v in (*box.min_rel, *box.max_rel))
        for box in ems_list
    ))


def _placed_signature(placed_list) -> tuple:
    """PlacedItem一覧の既配置AABB署名（1e-6丸め6要素tuple）を、重複を保持したまま
    ソートした多重集合表現へ変換する。"""
    return tuple(sorted(
        tuple(round(float(v), 6) for v in (*item.aabb_min_rel, *item.aabb_max_rel))
        for item in placed_list
    ))


def _assert_state_equivalent(state_a, state_b, n_containers: int) -> None:
    """`build_state_cached` と `build_state` の戻り値が同値であることを検証する（T-013契約）。

    height は `np.array_equal` による完全一致、EMS は1e-6丸め6要素tupleのソート済み列一致、
    `ems_truncation` は `math.isclose`、placed/pool/meta は内容一致を確認する。
    """
    assert len(state_a.containers) == len(state_b.containers) == n_containers

    for idx in range(n_containers):
        np.testing.assert_array_equal(
            state_a.containers[idx].height,
            state_b.containers[idx].height,
            err_msg=f"container {idx}: height が全再構築と一致しない",
        )
        assert _ems_signature(state_a.ems[idx]) == _ems_signature(state_b.ems[idx]), (
            f"container {idx}: EMS集合が全再構築と一致しない"
        )
        assert math.isclose(
            state_a.ems_truncation[idx], state_b.ems_truncation[idx], abs_tol=1e-9
        ), f"container {idx}: ems_truncation が全再構築と一致しない"
        assert _placed_signature(state_a.placed[idx]) == _placed_signature(state_b.placed[idx]), (
            f"container {idx}: placed のAABB多重集合が全再構築と一致しない"
        )

    assert len(state_a.pool) == len(state_b.pool)
    for spec_a, spec_b in zip(state_a.pool, state_b.pool):
        assert spec_a.idx == spec_b.idx
        np.testing.assert_allclose(spec_a.size, spec_b.size, atol=1e-9)
        assert spec_a.weight == pytest.approx(spec_b.weight)
        assert spec_a.is_soft == spec_b.is_soft
        assert spec_a.is_priority == spec_b.is_priority

    assert state_a.meta == state_b.meta


# --- 1) cache=None ラウンド --------------------------------------------------------

def test_state_rebuild_cache_none_first_step_matches_full():
    """経路: cache=None での全再構築フォールバック。

    2容器・既配置0件で `build_state_cached(obs, init, cache=None)` を呼び、
    戻り値が `build_state` と同値であること、返る `StateCache` が
    `geometry_key`/`placed_signatures`/`heights`/`ems_full` を全容器indexキーで保持し、
    `heights[idx]` が `build_state` の height と一致し `writeable=False` であること、
    `ems_full[idx]` が select_topn 前の `generate_ems` 相当（tuple）と一致することを検証する。
    """
    from src.packing_core import ems as ems_module
    from src.packing_core import state

    offsets = [0.0, 2.0]
    specs = [(i, off, DEFAULT_INNER_MIN, DEFAULT_INNER_MAX) for i, off in enumerate(offsets)]
    init_containers, obs_containers = _container_lists(specs)

    init = _init(init_containers)
    observation = _observation(obs_containers, pool_list=[
        _item(index=0, size=(0.3, 0.2, 0.2), mass=5.0),
        _item(index=1, size=(0.4, 0.3, 0.2), mass=6.0),
    ])

    state_full = state.build_state(observation, init)
    state_c, cache = state.build_state_cached(observation, init, cache=None)

    _assert_state_equivalent(state_c, state_full, 2)

    assert isinstance(cache, state.StateCache)
    assert isinstance(cache.geometry_key, tuple)
    assert set(cache.placed_signatures.keys()) == {0, 1}
    assert set(cache.heights.keys()) == {0, 1}
    assert set(cache.ems_full.keys()) == {0, 1}

    for idx in (0, 1):
        assert cache.placed_signatures[idx] == (), "既配置0件のコンテナは空の多重集合になる"
        np.testing.assert_array_equal(cache.heights[idx], state_full.containers[idx].height)
        assert cache.heights[idx].flags.writeable is False

        assert isinstance(cache.ems_full[idx], tuple)
        expected_ems_full = ems_module.generate_ems(state_full.containers[idx], [])
        assert _ems_signature(cache.ems_full[idx]) == _ems_signature(expected_ems_full)


# --- 2) 単一追加ヒット --------------------------------------------------------------

def test_state_rebuild_hit_single_add_empty_to_one():
    """経路: 単一コンテナへの単一追加ヒット（空→1個）。

    step1（既配置0件）で作った cache を使い、step2（1個追加）が `build_state` と
    同値になることを検証する（`cache_bake_placed` + 既存 `update_ems` の増分適用）。
    """
    from src.packing_core import state

    specs = [(0, 0.0, DEFAULT_INNER_MIN, DEFAULT_INNER_MAX)]

    init_containers1, obs_containers1 = _container_lists(specs, packed_by_index={0: []})
    init1 = _init(init_containers1)
    obs1 = _observation(obs_containers1, pool_list=[])
    _, cache1 = state.build_state_cached(obs1, init1, cache=None)

    item = _item(index=0, size=(0.3, 0.2, 0.2), mass=4.0, pos=(0.0, 0.0, 0.12), orn=IDENTITY_QUAT, belongs_to=0)
    init_containers2, obs_containers2 = _container_lists(specs, packed_by_index={0: [item]})
    init2 = _init(init_containers2)
    obs2 = _observation(obs_containers2, pool_list=[])

    state2, _ = state.build_state_cached(obs2, init2, cache=cache1)
    state_full2 = state.build_state(obs2, init2)
    _assert_state_equivalent(state2, state_full2, 1)


# --- 3) 逐次単一追加（複数ステップ） -----------------------------------------------

def test_state_rebuild_hit_sequential_adds_each_step_matches_full():
    """経路: 5ステップにわたり毎回1個ずつ追加する単一追加ヒットの連続適用。

    各ステップで cache を引き回し、そのステップの `build_state_cached` 結果が
    同ステップの `build_state` と一致することを毎回検証する（古い height/EMSが
    残らないことの確認）。
    """
    from src.packing_core import state

    positions = _grid_positions(5, DEFAULT_INNER_MIN, DEFAULT_INNER_MAX, item_size=0.15, pitch=0.2)
    specs = [(0, 0.0, DEFAULT_INNER_MIN, DEFAULT_INNER_MAX)]

    def _mk_item(i: int, pos: np.ndarray) -> dict:
        return _item(index=i, size=(0.15, 0.15, 0.15), mass=2.0, pos=pos, orn=IDENTITY_QUAT, belongs_to=0)

    cache = None
    placed_so_far: list[dict] = []
    for step in range(5):
        placed_so_far = placed_so_far + [_mk_item(step, positions[step])]
        init_containers, obs_containers = _container_lists(specs, packed_by_index={0: placed_so_far})
        init_d = _init(init_containers)
        obs_d = _observation(obs_containers, pool_list=[])

        state_step, cache = state.build_state_cached(obs_d, init_d, cache=cache)
        state_full_step = state.build_state(obs_d, init_d)
        _assert_state_equivalent(state_step, state_full_step, 1)


# --- 4) 複数追加（1回のステップで3個） ---------------------------------------------

def test_state_rebuild_multi_add_matches_full_rebuild():
    """経路: 単一コンテナへの複数追加ヒット（2個の cache から一度に3個追加）。"""
    from src.packing_core import state

    positions = _grid_positions(5, DEFAULT_INNER_MIN, DEFAULT_INNER_MAX, item_size=0.15, pitch=0.2)
    specs = [(0, 0.0, DEFAULT_INNER_MIN, DEFAULT_INNER_MAX)]

    def _mk_item(i: int, pos: np.ndarray) -> dict:
        return _item(index=i, size=(0.15, 0.15, 0.15), mass=2.0, pos=pos, orn=IDENTITY_QUAT, belongs_to=0)

    baseline_items = [_mk_item(0, positions[0]), _mk_item(1, positions[1])]
    init_containers1, obs_containers1 = _container_lists(specs, packed_by_index={0: baseline_items})
    init1 = _init(init_containers1)
    obs1 = _observation(obs_containers1, pool_list=[])
    _, cache1 = state.build_state_cached(obs1, init1, cache=None)

    all_items = baseline_items + [
        _mk_item(2, positions[2]), _mk_item(3, positions[3]), _mk_item(4, positions[4]),
    ]
    init_containers2, obs_containers2 = _container_lists(specs, packed_by_index={0: all_items})
    init2 = _init(init_containers2)
    obs2 = _observation(obs_containers2, pool_list=[])

    state2, _ = state.build_state_cached(obs2, init2, cache=cache1)
    state_full2 = state.build_state(obs2, init2)
    _assert_state_equivalent(state2, state_full2, 1)


# --- 5) 複数追加の順序非依存 ---------------------------------------------------------

def test_state_rebuild_multi_add_order_independent():
    """経路: 複数追加ヒットが `packed_items` の並び順に依存しないことの検証。

    同一の cache（2個既配置）から同じ3個を異なる並び順の2つの observation で増分適用し、
    2つの増分結果が一致し、かつどちらも全再構築と一致することを確認する。T-013b実装時に
    本テストを緑にできない場合は、§4.4の方針どおり増分ヒットを「1回の呼び出しにつき
    追加は最大1件」へ縮退させ、本テストは単一追加限定へ置き換える（本書と対で改定）。
    """
    from src.packing_core import state

    positions = _grid_positions(5, DEFAULT_INNER_MIN, DEFAULT_INNER_MAX, item_size=0.15, pitch=0.2)
    specs = [(0, 0.0, DEFAULT_INNER_MIN, DEFAULT_INNER_MAX)]

    def _mk_item(i: int, pos: np.ndarray) -> dict:
        return _item(index=i, size=(0.15, 0.15, 0.15), mass=2.0, pos=pos, orn=IDENTITY_QUAT, belongs_to=0)

    baseline_items = [_mk_item(0, positions[0]), _mk_item(1, positions[1])]
    init_containers1, obs_containers1 = _container_lists(specs, packed_by_index={0: baseline_items})
    init1 = _init(init_containers1)
    obs1 = _observation(obs_containers1, pool_list=[])
    _, cache1 = state.build_state_cached(obs1, init1, cache=None)

    all_items = baseline_items + [
        _mk_item(2, positions[2]), _mk_item(3, positions[3]), _mk_item(4, positions[4]),
    ]
    reversed_items = list(reversed(all_items))

    init_a_containers, obs_a_containers = _container_lists(specs, packed_by_index={0: all_items})
    init_b_containers, obs_b_containers = _container_lists(specs, packed_by_index={0: reversed_items})

    init_a = _init(init_a_containers)
    obs_a = _observation(obs_a_containers, pool_list=[])
    init_b = _init(init_b_containers)
    obs_b = _observation(obs_b_containers, pool_list=[])

    state_a, _ = state.build_state_cached(obs_a, init_a, cache=cache1)
    state_b, _ = state.build_state_cached(obs_b, init_b, cache=cache1)

    _assert_state_equivalent(state_a, state_b, 1)

    state_full_a = state.build_state(obs_a, init_a)
    _assert_state_equivalent(state_a, state_full_a, 1)


# --- 6) 物理沈降ドリフトによる無効化（核テスト） -------------------------------------

def test_state_rebuild_invalidate_on_drift_uses_full_rebuild():
    """経路: 既配置の物理沈降ドリフトによる無効化→全再構築フォールバック（核テスト）。

    step1で1個の既配置アイテムを cache し、step2ではそのアイテムが同一個数のまま
    別位置へ移動した状態（物理沈降を模した位置変化）を与える。素朴に「件数が同じだから
    差分なし」と誤判定する増分実装は、旧位置に残った stale な height/EMSと新位置の
    未反映のずれを見逃すため、本アサーション（全再構築との height/EMS 一致）で検出される。
    """
    from src.packing_core import state

    specs = [(0, 0.0, DEFAULT_INNER_MIN, DEFAULT_INNER_MAX)]

    item_before = _item(
        index=0, size=(0.2, 0.2, 0.2), mass=3.0, pos=(-0.2, 0.0, 0.12), orn=IDENTITY_QUAT, belongs_to=0,
    )
    init_containers1, obs_containers1 = _container_lists(specs, packed_by_index={0: [item_before]})
    init1 = _init(init_containers1)
    obs1 = _observation(obs_containers1, pool_list=[])
    _, cache1 = state.build_state_cached(obs1, init1, cache=None)

    item_after = _item(
        index=0, size=(0.2, 0.2, 0.2), mass=3.0, pos=(0.2, 0.0, 0.12), orn=IDENTITY_QUAT, belongs_to=0,
    )
    init_containers2, obs_containers2 = _container_lists(specs, packed_by_index={0: [item_after]})
    init2 = _init(init_containers2)
    obs2 = _observation(obs_containers2, pool_list=[])

    state2, _ = state.build_state_cached(obs2, init2, cache=cache1)
    state_full2 = state.build_state(obs2, init2)
    _assert_state_equivalent(state2, state_full2, 1)


# --- 7) 配置削除による無効化 ---------------------------------------------------------

def test_state_rebuild_invalidate_on_removal():
    """経路: 既配置の削除（件数減少）による無効化→全再構築フォールバック。"""
    from src.packing_core import state

    specs = [(0, 0.0, DEFAULT_INNER_MIN, DEFAULT_INNER_MAX)]

    item_lo = _item(index=0, size=(0.2, 0.2, 0.2), mass=3.0, pos=(-0.2, 0.0, 0.12), orn=IDENTITY_QUAT, belongs_to=0)
    item_hi = _item(index=1, size=(0.2, 0.2, 0.2), mass=3.0, pos=(0.1, 0.0, 0.12), orn=IDENTITY_QUAT, belongs_to=0)

    init_containers1, obs_containers1 = _container_lists(specs, packed_by_index={0: [item_lo, item_hi]})
    init1 = _init(init_containers1)
    obs1 = _observation(obs_containers1, pool_list=[])
    _, cache1 = state.build_state_cached(obs1, init1, cache=None)

    init_containers2, obs_containers2 = _container_lists(specs, packed_by_index={0: [item_lo]})
    init2 = _init(init_containers2)
    obs2 = _observation(obs_containers2, pool_list=[])

    state2, _ = state.build_state_cached(obs2, init2, cache=cache1)
    state_full2 = state.build_state(obs2, init2)
    _assert_state_equivalent(state2, state_full2, 1)


# --- 8) コンテナ形状変更による無効化 -------------------------------------------------

def test_state_rebuild_invalidate_on_geometry_change():
    """経路: コンテナ形状変更（`geometry_key` 不一致）による無効化→全再構築フォールバック。"""
    from src.packing_core import state

    item = _item(index=0, size=(0.2, 0.2, 0.2), mass=4.0, pos=(0.0, 0.0, 0.12), orn=IDENTITY_QUAT, belongs_to=0)

    specs1 = [(0, 0.0, DEFAULT_INNER_MIN, DEFAULT_INNER_MAX)]
    init_containers1, obs_containers1 = _container_lists(specs1, packed_by_index={0: [item]})
    init1 = _init(init_containers1)
    obs1 = _observation(obs_containers1, pool_list=[])
    _, cache1 = state.build_state_cached(obs1, init1, cache=None)

    inner_max2 = DEFAULT_INNER_MAX + np.array([0.0, 0.3, 0.0])
    specs2 = [(0, 0.0, DEFAULT_INNER_MIN, inner_max2)]
    init_containers2, obs_containers2 = _container_lists(specs2, packed_by_index={0: [item]})
    init2 = _init(init_containers2)
    obs2 = _observation(obs_containers2, pool_list=[])

    state2, _ = state.build_state_cached(obs2, init2, cache=cache1)
    state_full2 = state.build_state(obs2, init2)
    _assert_state_equivalent(state2, state_full2, 1)


# --- 8b) T-016B: geometry_key への path_* 6フィールド追加 ------------------------------
#
# interface_notes.md §L.5: height/buffer の個別値は既存キー要素（inner_min_rel/
# inner_max_rel/クリップ後shelf_boxes/cut_planes）から一般に一意復元できることを証明
# できなかった（クリップ後shelf_boxesが退化・空集合になり得るedge caseで分離情報が失わ
# れるため）。安全側の最小修正として path_* 6フィールドを _container_geometry_key へ
# 直接追加した。本節はこの拡張の必要性そのものを回帰確認する：cut_x=0（小棚もクリップ後
# 空集合）の2コンテナで inner_min_rel/inner_max_rel/shelf_boxes/cut_planes が完全一致
# しつつ height/buffer の組だけが異なる（height+buffer の和は inner_max_rel[2] 経由で
# 一致させたまま、height と buffer の内訳だけを変える）fixtureを用いる。


def test_container_geometry_key_distinguishes_height_buffer_split_with_identical_legacy_key():
    """§L.5の動機となったedge case: 既存キー要素が完全一致しつつ height/buffer の内訳だけが
    異なる2コンテナで `_container_geometry_key` が異なる値を返すこと（path_mid_resting_z_rel/
    path_mid_ceiling_z_rel の差がキーに反映される）。"""
    from src.packing_core import state

    inner_min = DEFAULT_INNER_MIN
    inner_max = DEFAULT_INNER_MAX

    cdict_a = _container(0, 0.0, inner_min, inner_max, thickness=0.02, buffer=0.02)
    cdict_b = _container(0, 0.0, inner_min, inner_max, thickness=0.02, buffer=0.05)
    assert cdict_a["height"] != cdict_b["height"]  # height+buffer の和は inner_max_rel[2] 経由で一致

    space_a = _build_container_space_from_cdict(cdict_a)
    space_b = _build_container_space_from_cdict(cdict_b)

    # 既存キー要素（拡張前の legacy 部分）は完全一致することを前提として確認する。
    np.testing.assert_array_equal(space_a.inner_min_rel, space_b.inner_min_rel)
    np.testing.assert_array_equal(space_a.inner_max_rel, space_b.inner_max_rel)
    assert space_a.shelf_boxes == [] == space_b.shelf_boxes
    assert space_a.cut_planes == [] == space_b.cut_planes

    # path_mid_resting_z_rel/path_mid_ceiling_z_rel は height/buffer の内訳に依存するため異なる。
    assert space_a.path_mid_resting_z_rel != pytest.approx(space_b.path_mid_resting_z_rel)
    assert space_a.path_mid_ceiling_z_rel != pytest.approx(space_b.path_mid_ceiling_z_rel)

    key_a = state._container_geometry_key(space_a)
    key_b = state._container_geometry_key(space_b)
    assert key_a != key_b


def test_state_rebuild_invalidate_on_height_buffer_only_geometry_change():
    """経路: 既存キー要素が不変で height/buffer の内訳だけが変わる`geometry_key`不一致に
    よる無効化→全再構築フォールバック（§L.5、path_*拡張が無ければ誤ヒットし得たケース）。"""
    from src.packing_core import state

    item = _item(index=0, size=(0.2, 0.2, 0.2), mass=4.0, pos=(0.0, 0.0, 0.12), orn=IDENTITY_QUAT, belongs_to=0)

    cdict1 = _container(0, 0.0, DEFAULT_INNER_MIN, DEFAULT_INNER_MAX, thickness=0.02, buffer=0.02)
    init1 = _init([{**cdict1, "packed_items": []}])
    obs1 = _observation([{**cdict1, "packed_items": [item]}], pool_list=[])
    _, cache1 = state.build_state_cached(obs1, init1, cache=None)

    cdict2 = _container(0, 0.0, DEFAULT_INNER_MIN, DEFAULT_INNER_MAX, thickness=0.02, buffer=0.05)
    init2 = _init([{**cdict2, "packed_items": []}])
    obs2 = _observation([{**cdict2, "packed_items": [item]}], pool_list=[])

    state2, _ = state.build_state_cached(obs2, init2, cache=cache1)
    state_full2 = state.build_state(obs2, init2)
    _assert_state_equivalent(state2, state_full2, 1)

    # path_* も全再構築結果と一致することを直接確認する（_assert_state_equivalent は
    # height/EMS/placed/pool/meta のみを比較し ContainerSpace.path_* を見ないため）。
    assert state2.containers[0].path_mid_resting_z_rel == pytest.approx(
        state_full2.containers[0].path_mid_resting_z_rel
    )
    assert state2.containers[0].path_mid_ceiling_z_rel == pytest.approx(
        state_full2.containers[0].path_mid_ceiling_z_rel
    )


def _build_container_space_from_cdict(cdict: dict):
    from src.packing_core.container_space import build_container_space

    return build_container_space(cdict, index=cdict["index"], cell=CELL)


# --- 9) 並べ替えのみ（追加差分なし） -------------------------------------------------

def test_state_rebuild_reorder_without_change_matches_full():
    """経路: 差分ゼロ（既配置の内容は同一・`packed_items` の並び順だけが変わる）でのヒット。

    追加も削除もない場合、`placed_signatures` の多重集合比較により差分ゼロと判定され、
    cache の再利用結果が全再構築と一致することを検証する。
    """
    from src.packing_core import state

    specs = [(0, 0.0, DEFAULT_INNER_MIN, DEFAULT_INNER_MAX)]
    item_a = _item(index=0, size=(0.15, 0.15, 0.15), mass=2.0, pos=(-0.25, 0.0, 0.095), orn=IDENTITY_QUAT, belongs_to=0)
    item_b = _item(index=1, size=(0.15, 0.15, 0.15), mass=2.0, pos=(0.0, 0.0, 0.095), orn=IDENTITY_QUAT, belongs_to=0)
    item_c = _item(index=2, size=(0.15, 0.15, 0.15), mass=2.0, pos=(0.25, 0.0, 0.095), orn=IDENTITY_QUAT, belongs_to=0)

    init_containers1, obs_containers1 = _container_lists(specs, packed_by_index={0: [item_a, item_b, item_c]})
    init1 = _init(init_containers1)
    obs1 = _observation(obs_containers1, pool_list=[])
    _, cache1 = state.build_state_cached(obs1, init1, cache=None)

    init_containers2, obs_containers2 = _container_lists(specs, packed_by_index={0: [item_c, item_a, item_b]})
    init2 = _init(init_containers2)
    obs2 = _observation(obs_containers2, pool_list=[])

    state2, _ = state.build_state_cached(obs2, init2, cache=cache1)
    state_full2 = state.build_state(obs2, init2)
    _assert_state_equivalent(state2, state_full2, 1)


# --- 10) 同一AABBの多重集合（重複個数の保持） ----------------------------------------

def test_state_rebuild_duplicate_aabbs_multiset():
    """`StateCache.placed_signatures` が多重集合（重複個数を保持するソート済みtuple）で
    あり、`frozenset` 等の集合ではないことを直接検証する（AABBの値自体は完全に重複する
    非物理的な合成であり、tests/test_state.py::test_state_packed_item_routed_by_belongs_to_not_position
    と同様に物理的妥当性は検証しない）。

    height/EMSは既配置AABBの**和集合**（footprint）にしか依存しないため、同一footprintの
    荷物が1個か2個かで最終的な height/EMS 自体は変化しない。そのため多重集合であることの
    discriminating な検証は、最終出力の比較ではなく `cache.placed_signatures` の要素数を
    直接確認する形で行う（`frozenset` 実装だと重複が失われ長さ1になってしまう）。
    追加・削除後の `build_state` との出力一致は、コード経路が正しく実行されることの
    副次的な確認として合わせて行う。
    """
    from src.packing_core import state

    specs = [(0, 0.0, DEFAULT_INNER_MIN, DEFAULT_INNER_MAX)]

    def _dup(i: int) -> dict:
        return _item(index=i, size=(0.2, 0.2, 0.2), mass=3.0, pos=(0.0, 0.0, 0.12), orn=IDENTITY_QUAT, belongs_to=0)

    init_containers, obs_containers = _container_lists(
        specs, packed_by_index={0: [_dup(0), _dup(1), _dup(2)]}
    )
    init0 = _init(init_containers)
    observation0 = _observation(obs_containers, pool_list=[])
    _, cache0 = state.build_state_cached(observation0, init0, cache=None)

    sigs = cache0.placed_signatures[0]
    assert len(sigs) == 3, "同一AABBの3個が多重集合として保持されず、重複個数が失われている"
    assert len(set(sigs)) == 1, "3個とも同一署名であるはず"

    # 追加（3→4個、いずれも同一AABB）: 全再構築との一致を副次確認する。
    init_add, obs_add = _container_lists(specs, packed_by_index={0: [_dup(0), _dup(1), _dup(2), _dup(3)]})
    init_add_d = _init(init_add)
    obs_add_d = _observation(obs_add, pool_list=[])
    state_add, cache_add = state.build_state_cached(obs_add_d, init_add_d, cache=cache0)
    state_full_add = state.build_state(obs_add_d, init_add_d)
    _assert_state_equivalent(state_add, state_full_add, 1)
    assert len(cache_add.placed_signatures[0]) == 4

    # 削除（3→2個、いずれも同一AABB）: 件数減少は無効化条件に該当する。
    init_rm, obs_rm = _container_lists(specs, packed_by_index={0: [_dup(0), _dup(1)]})
    init_rm_d = _init(init_rm)
    obs_rm_d = _observation(obs_rm, pool_list=[])
    state_rm, cache_rm = state.build_state_cached(obs_rm_d, init_rm_d, cache=cache0)
    state_full_rm = state.build_state(obs_rm_d, init_rm_d)
    _assert_state_equivalent(state_rm, state_full_rm, 1)
    assert len(cache_rm.placed_signatures[0]) == 2


# --- 11) 複数コンテナでのヒット/無効化の混在 -----------------------------------------

def test_state_rebuild_per_container_mixed_hit_and_fallback():
    """経路: 3容器でヒット種別が独立に混在するケース（コンテナ単位の独立性）。

    c0: 空→1個追加（単一追加ヒット）／c1: 既配置1個が別位置へドリフト（無効化）／
    c2: 既配置1個が変化なし（差分ゼロ・ヒット）。全体の `PackingState` が `build_state`
    と一致することを確認し、あるコンテナの無効化が他コンテナのヒット判定に影響しないことを
    検証する。
    """
    from src.packing_core import state

    offsets = [0.0, 2.0, 4.0]
    specs = [(i, off, DEFAULT_INNER_MIN, DEFAULT_INNER_MAX) for i, off in enumerate(offsets)]

    item_c1_before = _item(
        index=0, size=(0.2, 0.2, 0.2), mass=3.0, pos=(1.8, 0.0, 0.12), orn=IDENTITY_QUAT, belongs_to=1,
    )
    item_c2 = _item(
        index=0, size=(0.2, 0.2, 0.2), mass=3.0, pos=(4.0, 0.0, 0.12), orn=IDENTITY_QUAT, belongs_to=2,
    )

    init_containers1, obs_containers1 = _container_lists(
        specs, packed_by_index={0: [], 1: [item_c1_before], 2: [item_c2]}
    )
    init1 = _init(init_containers1)
    obs1 = _observation(obs_containers1, pool_list=[])
    _, cache1 = state.build_state_cached(obs1, init1, cache=None)

    item_c0 = _item(
        index=0, size=(0.2, 0.2, 0.2), mass=3.0, pos=(0.0, 0.0, 0.12), orn=IDENTITY_QUAT, belongs_to=0,
    )
    item_c1_after = _item(
        index=0, size=(0.2, 0.2, 0.2), mass=3.0, pos=(2.2, 0.0, 0.12), orn=IDENTITY_QUAT, belongs_to=1,
    )
    init_containers2, obs_containers2 = _container_lists(
        specs, packed_by_index={0: [item_c0], 1: [item_c1_after], 2: [item_c2]}
    )
    init2 = _init(init_containers2)
    obs2 = _observation(obs_containers2, pool_list=[])

    state2, _ = state.build_state_cached(obs2, init2, cache=cache1)
    state_full2 = state.build_state(obs2, init2)
    _assert_state_equivalent(state2, state_full2, 3)


# --- 12) cut/棚ありコンテナ ---------------------------------------------------------

def test_state_rebuild_cut_shelf_container_matches_full():
    """経路: cut・棚ありコンテナでの単一追加ヒット。棚は静的障害物として扱われる。

    `tests/fixtures/container_space_golden.py`（読み取り専用フィクスチャ、T-007由来）の
    `build_fixture_ab_cdict(shelf=True)` を用いる。
    """
    from fixtures import container_space_golden as golden
    from src.packing_core import state
    from src.packing_core.container_space import build_container_space

    init_cdict1 = golden.build_fixture_ab_cdict(index=0, spacing=0.0, shelf=True)
    obs_cdict1 = golden.build_fixture_ab_cdict(index=0, spacing=0.0, shelf=True)

    init1 = _init([init_cdict1])
    obs1 = _observation([obs_cdict1], pool_list=[])
    _, cache1 = state.build_state_cached(obs1, init1, cache=None)

    # 有効空間内の安全な設置点を production の build_container_space から直接得る
    # （このコンテナ固有の inner_min/max を推測しない）。
    space0 = build_container_space(init_cdict1, 0, CELL)
    assert len(space0.shelf_boxes) >= 1, "shelf=True のfixtureは棚AABBを持つはず"

    mid_xy = (space0.inner_min_rel[:2] + space0.inner_max_rel[:2]) / 2.0
    item_size = np.array([0.15, 0.15, 0.15])
    pos_rel = np.array([mid_xy[0], mid_xy[1], space0.inner_min_rel[2] + item_size[2] / 2.0])
    packed_item = _item(
        index=0, size=tuple(item_size), mass=5.0, pos=pos_rel, orn=IDENTITY_QUAT, belongs_to=0,
    )

    init_cdict2 = golden.build_fixture_ab_cdict(index=0, spacing=0.0, shelf=True)
    obs_cdict2 = golden.build_fixture_ab_cdict(index=0, spacing=0.0, shelf=True)
    obs_cdict2["packed_items"] = [packed_item]

    init2 = _init([init_cdict2])
    obs2 = _observation([obs_cdict2], pool_list=[])

    state2, _ = state.build_state_cached(obs2, init2, cache=cache1)
    state_full2 = state.build_state(obs2, init2)
    _assert_state_equivalent(state2, state_full2, 1)


# --- 13) 予算超過による打切り ---------------------------------------------------------

def test_state_rebuild_ems_truncation_over_budget_matches_full():
    """経路: 単一追加ヒットで候補数が `StageParams().ems_top_n_per_container`（=80）を
    超え、`select_topn` の打切りが発生するケース。

    固定シード（`np.random.default_rng(42)`）による決定論的fixtureで、9個時点では
    EMS候補72件（打切りなし）、10個目を追加すると84件（80件選択・打切り率4/84）になる
    ことを作成時に確認済み。増分側の選択EMS列・打切り率が全再構築と一致することを検証する。
    """
    from src.packing_core import state

    rng = np.random.default_rng(42)
    aabbs = _scattered_disjoint_aabbs(rng, 10, DEFAULT_INNER_MIN, DEFAULT_INNER_MAX)
    specs = [(0, 0.0, DEFAULT_INNER_MIN, DEFAULT_INNER_MAX)]

    def _items_from_aabbs(aabb_list) -> list[dict]:
        items = []
        for i, (bmin, bmax) in enumerate(aabb_list):
            center = (bmin + bmax) / 2.0
            size = bmax - bmin
            items.append(_item(
                index=i, size=tuple(size), mass=2.0, pos=center, orn=IDENTITY_QUAT, belongs_to=0,
            ))
        return items

    items_9 = _items_from_aabbs(aabbs[:9])
    init_containers1, obs_containers1 = _container_lists(specs, packed_by_index={0: items_9})
    init1 = _init(init_containers1)
    obs1 = _observation(obs_containers1, pool_list=[])
    state1, cache1 = state.build_state_cached(obs1, init1, cache=None)
    assert state1.ems_truncation[0] == pytest.approx(0.0), "9個時点では打切りが発生しない想定"

    items_10 = _items_from_aabbs(aabbs[:10])
    init_containers2, obs_containers2 = _container_lists(specs, packed_by_index={0: items_10})
    init2 = _init(init_containers2)
    obs2 = _observation(obs_containers2, pool_list=[])

    state2, _ = state.build_state_cached(obs2, init2, cache=cache1)
    state_full2 = state.build_state(obs2, init2)
    _assert_state_equivalent(state2, state_full2, 1)

    n_budget = constants.StageParams().ems_top_n_per_container
    assert len(state_full2.ems[0]) == n_budget
    assert state_full2.ems_truncation[0] == pytest.approx(4.0 / 84.0)


# --- 14) キャッシュ配列の非共有・書き込み禁止 -----------------------------------------

def test_state_rebuild_cache_arrays_not_shared_and_readonly():
    """`StateCache.heights[idx]` は返却する `PackingState` の `ContainerSpace.height` と
    メモリを共有しない独立コピーであり、`writeable=False` であることを検証する。
    返却側の height を書き換えても、cache 側の値が変化しないことを確認する。
    """
    from src.packing_core import state

    specs = [(0, 0.0, DEFAULT_INNER_MIN, DEFAULT_INNER_MAX)]
    item = _item(index=0, size=(0.2, 0.2, 0.2), mass=3.0, pos=(0.0, 0.0, 0.12), orn=IDENTITY_QUAT, belongs_to=0)
    init_containers, obs_containers = _container_lists(specs, packed_by_index={0: [item]})
    init = _init(init_containers)
    observation = _observation(obs_containers, pool_list=[])

    result_state, cache = state.build_state_cached(observation, init, cache=None)

    assert cache.heights[0].flags.writeable is False
    assert not np.shares_memory(result_state.containers[0].height, cache.heights[0])
    assert isinstance(cache.ems_full[0], tuple)

    original_cached_height = cache.heights[0].copy()
    result_state.containers[0].height[...] = 12345.0

    np.testing.assert_array_equal(cache.heights[0], original_cached_height)


# --- 15) 入力辞書の非改変 -----------------------------------------------------------

def test_state_rebuild_input_dicts_not_mutated():
    """`build_state_cached` は `build_state` と同じ入力非改変契約に従う（T-012契約の
    増分パスへの継承）。cache を渡した2回目の呼び出しでも observation/init が
    変更されないことを確認する。"""
    from src.packing_core import state

    specs = [(0, 0.0, DEFAULT_INNER_MIN, DEFAULT_INNER_MAX)]
    item = _item(index=0, size=(0.2, 0.2, 0.2), mass=3.0, pos=(0.0, 0.0, 0.12), orn=IDENTITY_QUAT, belongs_to=0)
    init_containers, obs_containers = _container_lists(specs, packed_by_index={0: [item]})
    init = _init(init_containers)
    observation = _observation(obs_containers, pool_list=[])

    init_snapshot = copy.deepcopy(init)
    observation_snapshot = copy.deepcopy(observation)

    _, cache1 = state.build_state_cached(observation, init, cache=None)
    assert _deep_equal(init, init_snapshot), "build_state_cached が init を変更してはならない"
    assert _deep_equal(observation, observation_snapshot), "build_state_cached が observation を変更してはならない"

    _, cache2 = state.build_state_cached(observation, init, cache=cache1)
    assert _deep_equal(init, init_snapshot), "cache渡しの2回目呼び出しでも init を変更してはならない"
    assert _deep_equal(observation, observation_snapshot), "cache渡しの2回目呼び出しでも observation を変更してはならない"
