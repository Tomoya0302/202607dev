"""T-012: packing_core.state のテスト（先行作成、実装詳細仕様書 §3.1/§3.3/§3.4/§4.4、
§6 T-012、interface_notes.md §E/§F/§H/§I-8）。

state.py は本チケット時点で未実装（本体実装は別セッション）。--collect-only を成功させるため、
対象モジュール src.packing_core.state の import は各テスト関数の内部で行う
（tests/test_geometry.py・tests/test_container_space.py と同方針）。skip/xfail/仮実装は禁止。

init/observation の container_list について:
    build_state(observation, init) の呼び出しでは、init と observation に**同一の container_list
    オブジェクトを渡さない**。形状・center は同一にしつつ、init 側は常に packed_items=[]、
    observation 側にのみテスト対象の packed_items を格納する（_container_lists ヘルパ参照）。
    これにより「build_state が現在の既配置荷物を observation から取得する」ことを検証する
    （init 側の packed_items を誤って参照する実装は失敗する）。optimize/lookahead_k は
    取得元の優先順位が未定義のため、init/observation で同値にする。
"""
import copy
import statistics
import time

import numpy as np
import pytest

from src.packing_core import constants

CELL = constants.GridParams().cell
EPS_GEOM = constants.EPS_GEOM
IDENTITY_QUAT: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

# 汎用テスト用の内壁範囲（cut/棚なし直方体）。
DEFAULT_INNER_MIN = np.array([-0.4, -0.5, 0.02], dtype=np.float64)
DEFAULT_INNER_MAX = np.array([0.4, 0.5, 1.0], dtype=np.float64)

# 回転AABBテスト用に余裕を持たせた内壁範囲（合成回転後の箱を完全に内包するため）。
WIDE_INNER_MIN = np.array([-1.0, -1.0, 0.02], dtype=np.float64)
WIDE_INNER_MAX = np.array([1.0, 1.0, 1.5], dtype=np.float64)


# --- フィクスチャ・ヘルパ ------------------------------------------------------------

def _make_space(
    offset_x: float,
    inner_min_rel: np.ndarray,
    inner_max_rel: np.ndarray,
    cell: float = CELL,
):
    """world_to_rel/rel_to_world 単体テスト用の最小 ContainerSpace を直接構築する。

    cut/棚・格子内容はこれらのテストで検証しないため、floor_z/ceil_z/height は
    1x1セルのダミー配列で代用する（container_space.py は実装済みのためコンストラクタを
    直接呼べるが、対象モジュール state.py には依存しないようこの関数内で import する）。
    """
    from src.packing_core.container_space import ContainerSpace

    imin = np.asarray(inner_min_rel, dtype=np.float64)
    imax = np.asarray(inner_max_rel, dtype=np.float64)
    floor = np.full((1, 1), imin[2], dtype=np.float64)
    ceil = np.full((1, 1), imax[2], dtype=np.float64)
    return ContainerSpace(
        index=0,
        offset_x=float(offset_x),
        inner_min_rel=imin,
        inner_max_rel=imax,
        cut_planes=[],
        shelf_boxes=[],
        cell=float(cell),
        floor_z=floor,
        ceil_z=ceil,
        height=floor.copy(),
    )


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
    """§E item/pool_list 要素の共通キーを持つ辞書を組み立てる。

    pool 要素は pos=orn=belongs_to=None（interface_notes.md §E: 未配置時は None）。
    packed 要素は世界座標の中心 pos・姿勢 orn・所属コンテナ index belongs_to を指定する。
    is_soft=True のときのみ item_soft_extra（contactStiffness 等）を付与する。
    """
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
        # np.array(...) は既定でコピーする（呼び出し元の配列とエイリアスさせず、
        # build_state 側の変更検知テストが偽陽性/偽陰性にならないようにするため）。
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
    """軸整列直方体の container_list 要素（cdict）を組み立てる。

    6面（±X/±Y/±Z）の代表点（世界座標）と外向き法線から points/n_vecs を構成するため
    cut_planes は常に空になり、cut_x=cut_y=0.0・shelf=False のため shelf_boxes も常に空になる
    （tests/test_container_space.py の _box_cdict と同方針、interface_notes.md §K.5）。
    offset_x は center[0] にそのまま反映し、index*spacing の再計算はしない（§I-8）。
    """
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
    height = float(imax[2]) + buffer  # ceil_z(=inner_max.z) = height - buffer と自己無矛盾

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
    """init 用・observation 用の container_list を**別オブジェクトとして**2つ生成する。

    形状・center は specs の内容で共通にしつつ、init 側は常に packed_items=[]、observation 側は
    `packed_by_index` に指定した index にのみ packed_items を格納する（本ファイル冒頭のポリシー参照）。
    """
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
    """内壁範囲内に収まる、相互非重複なグリッド中心（コンテナ相対）を n 個返す。

    `pitch >= item_size` により隣接セル間の非重複を保証し、各セル中心は half-size 分だけ
    内壁から内側にオフセットするため、返す全ての箱が内壁AABB内に収まる。
    """
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


def _assert_perf_fixture_valid(
    obs_containers: list[dict], spec_by_index: dict[int, tuple[np.ndarray, np.ndarray]]
) -> None:
    """性能fixtureが要件（有効空間内・相互非重複・belongs_to一致・identity quat）を満たすことを
    計測前に検証する。"""
    identity = np.array(IDENTITY_QUAT, dtype=np.float64)
    for cdict in obs_containers:
        idx = cdict["index"]
        imin, imax = spec_by_index[idx]
        offset_x = float(cdict["center"][0])
        aabbs: list[tuple[np.ndarray, np.ndarray]] = []
        for item in cdict["packed_items"]:
            assert item["belongs_to"] == idx, "belongs_to は配置先コンテナindexと一致する必要がある"
            np.testing.assert_allclose(item["orn"], identity, atol=EPS_GEOM)

            size = np.array([item["length"], item["width"], item["height"]], dtype=np.float64)
            pos_rel = np.asarray(item["pos"], dtype=np.float64) - np.array([offset_x, 0.0, 0.0])
            half = size / 2.0
            bmin, bmax = pos_rel - half, pos_rel + half
            assert np.all(bmin >= imin - EPS_GEOM) and np.all(bmax <= imax + EPS_GEOM), (
                f"container {idx}: item AABB は有効空間内に収まる必要がある"
            )
            aabbs.append((bmin, bmax))

        for i in range(len(aabbs)):
            for j in range(i + 1, len(aabbs)):
                amin, amax = aabbs[i]
                bmin, bmax = aabbs[j]
                overlap = bool(np.all(amin < bmax - EPS_GEOM) and np.all(bmin < amax - EPS_GEOM))
                assert not overlap, f"container {idx}: items {i},{j} が重複している"


# --- world_to_rel / rel_to_world ----------------------------------------------------

def test_state_world_to_rel_and_rel_to_world_direction_and_roundtrip():
    """§3.1: pos_rel = pos_world - (offset_x,0,0)。X方向のみ offset_x を加減し、Y/Z は不変。
    往復性 rel_to_world(world_to_rel(p)) == p（atol=1e-12）を確認する。"""
    from src.packing_core import state

    space = _make_space(offset_x=1.8, inner_min_rel=DEFAULT_INNER_MIN, inner_max_rel=DEFAULT_INNER_MAX)
    p_world = np.array([2.3, -0.5, 0.9], dtype=np.float64)

    p_rel = state.world_to_rel(p_world, space)
    np.testing.assert_allclose(p_rel[0], p_world[0] - 1.8, atol=1e-12)
    np.testing.assert_allclose(p_rel[1], p_world[1], atol=1e-12)
    np.testing.assert_allclose(p_rel[2], p_world[2], atol=1e-12)

    p_world_back = state.rel_to_world(p_rel, space)
    np.testing.assert_allclose(p_world_back, p_world, atol=1e-12)

    # 逆方向（rel -> world -> rel）の往復性・方向性も独立に確認する。
    p_rel2 = np.array([0.1, 0.2, 0.5], dtype=np.float64)
    p_world2 = state.rel_to_world(p_rel2, space)
    np.testing.assert_allclose(p_world2[0], p_rel2[0] + 1.8, atol=1e-12)
    np.testing.assert_allclose(p_world2[1], p_rel2[1], atol=1e-12)
    np.testing.assert_allclose(p_world2[2], p_rel2[2], atol=1e-12)
    np.testing.assert_allclose(state.world_to_rel(p_world2, space), p_rel2, atol=1e-12)


def test_state_offset_x_nonuniform_multi_container():
    """§I-8/§4.4: offset_x は cdict["center"][0] を直接使用する（index*spacing ではない）。
    非等間隔配置（index*spacing で表現できない）の3容器で、各 ContainerSpace.offset_x が
    対応する center[0] と一致し、world_to_rel が自容器の offset のみを使うことを確認する。"""
    from src.packing_core import state

    offsets = [0.0, 1.8, 5.93]
    specs = [(i, off, DEFAULT_INNER_MIN, DEFAULT_INNER_MAX) for i, off in enumerate(offsets)]
    init_containers, obs_containers = _container_lists(specs)

    init = _init(init_containers)
    observation = _observation(obs_containers, pool_list=[])

    result = state.build_state(observation, init)

    assert len(result.containers) == 3
    for i, off in enumerate(offsets):
        space_i = result.containers[i]
        assert space_i.offset_x == pytest.approx(off)
        assert space_i.offset_x == pytest.approx(obs_containers[i]["center"][0])

    p_world = np.array([offsets[2] + 0.1, 0.05, 0.5], dtype=np.float64)
    p_rel = state.world_to_rel(p_world, result.containers[2])
    np.testing.assert_allclose(p_rel, np.array([0.1, 0.05, 0.5]), atol=1e-9)


# --- make_action ----------------------------------------------------------------------

def test_state_make_action_keys_types_dtype():
    """§3.4: make_action の戻り値は item_idx/container_idx/place_pos/orientation の4キーのみ。
    place_pos は np.float32 の ndarray、他は int（np.int64 を int() に変換済み）。"""
    from src.packing_core import state

    action = state.make_action(
        np.int64(2), np.int64(1), np.array([0.25, -0.5, 0.15], dtype=np.float64), np.int64(3)
    )

    assert set(action.keys()) == {"item_idx", "container_idx", "place_pos", "orientation"}

    assert type(action["item_idx"]) is int
    assert action["item_idx"] == 2
    assert type(action["container_idx"]) is int
    assert action["container_idx"] == 1
    assert type(action["orientation"]) is int
    assert action["orientation"] == 3

    assert isinstance(action["place_pos"], np.ndarray)
    assert action["place_pos"].dtype == np.float32
    assert action["place_pos"].shape == (3,)
    np.testing.assert_allclose(
        action["place_pos"], np.array([0.25, -0.5, 0.15], dtype=np.float32), atol=1e-6
    )


# --- PackingState 構造・meta --------------------------------------------------------

def test_state_packing_state_structure_and_meta():
    """§4.4: PackingState の型・辞書契約（placed/ems/ems_truncation は全コンテナindexキー、
    pool は ItemSpec のリスト）と meta 契約（optimize/lookahead_k必須、spacing禁止）を検証する。"""
    from src.packing_core import state
    from src.packing_core.container_space import ContainerSpace
    from src.packing_core.types import EMSBox, ItemSpec

    offsets = [0.0, 2.0]
    specs = [(i, off, DEFAULT_INNER_MIN, DEFAULT_INNER_MAX) for i, off in enumerate(offsets)]

    packed_item = _item(
        index=0,
        size=(0.3, 0.2, 0.2),
        mass=5.0,
        is_soft=False,
        is_priority=False,
        pos=(offsets[1], 0.0, 0.12),
        orn=IDENTITY_QUAT,
        belongs_to=1,
    )
    init_containers, obs_containers = _container_lists(specs, packed_by_index={1: [packed_item]})

    pool_items = [
        _item(index=0, size=(0.5, 0.4, 0.3), mass=10.0, is_soft=False, is_priority=True),
        _item(index=1, size=(0.6, 0.3, 0.25), mass=7.0, is_soft=True, is_priority=False),
    ]

    init = _init(init_containers, optimize=True, lookahead_k=10)
    observation = _observation(obs_containers, pool_items, optimize=True, lookahead_k=10)

    result = state.build_state(observation, init)

    assert isinstance(result.containers, list)
    assert len(result.containers) == 2
    assert all(isinstance(c, ContainerSpace) for c in result.containers)

    assert set(result.placed.keys()) == {0, 1}
    assert set(result.ems.keys()) == {0, 1}
    assert set(result.ems_truncation.keys()) == {0, 1}

    assert isinstance(result.pool, list)
    assert len(result.pool) == 2
    assert all(isinstance(p, ItemSpec) for p in result.pool)

    for placed_list in result.placed.values():
        assert isinstance(placed_list, list)
    for ems_list in result.ems.values():
        assert isinstance(ems_list, list)
        assert all(isinstance(e, EMSBox) for e in ems_list)

    assert result.meta["optimize"] == observation["optimize"]
    assert result.meta["lookahead_k"] == observation["lookahead_k"]
    assert "spacing" not in result.meta


# --- belongs_to による振り分け -------------------------------------------------------

def test_state_packed_item_routed_by_belongs_to_not_position():
    """belongs_to による振り分けを明示的に検証する。packed item の world 位置は container0 の
    内壁世界範囲（x∈[-0.4,0.4]）内に置きつつ、container_list の入れ子（container1側の
    packed_items）と belongs_to=1 の両方が container1 を指す構成にする。位置だけから所属
    コンテナを推測する実装は container0 に誤って割り当てるため、このテストで不合格になる
    （AABB の値自体はこのテストでは非物理的な合成であり検証しない）。"""
    from src.packing_core import state

    offsets = [0.0, 2.0]
    specs = [(i, off, DEFAULT_INNER_MIN, DEFAULT_INNER_MAX) for i, off in enumerate(offsets)]

    packed_item = _item(
        index=0,
        size=(0.2, 0.2, 0.2),
        mass=3.0,
        pos=(0.1, 0.0, 0.12),  # container0の内壁世界範囲内（位置ベース推測の罠）
        orn=IDENTITY_QUAT,
        belongs_to=1,
    )
    init_containers, obs_containers = _container_lists(specs, packed_by_index={1: [packed_item]})

    init = _init(init_containers)
    observation = _observation(obs_containers, pool_list=[])

    result = state.build_state(observation, init)

    assert result.placed[0] == []
    assert len(result.placed[1]) == 1


# --- packed item -> PlacedItem -------------------------------------------------------

def test_state_packed_item_to_placeditem_identity_quat():
    """§4.4手順3: pos_world→pos_rel、identity quat での回転後AABBは aabb_from_center と一致する。
    PlacedItem の全フィールドが入力と一致することを検証する。"""
    from src.packing_core import state

    offset_x = 1.8
    specs = [(0, offset_x, WIDE_INNER_MIN, WIDE_INNER_MAX)]

    pos_world = np.array([2.3, 0.6, 0.7], dtype=np.float64)
    orn = np.array(IDENTITY_QUAT, dtype=np.float64)
    size = np.array([0.5, 0.4, 0.3], dtype=np.float64)
    packed_item = _item(
        index=0, size=size, mass=12.0, is_soft=True, is_priority=False,
        pos=pos_world, orn=orn, belongs_to=0,
    )
    init_containers, obs_containers = _container_lists(specs, packed_by_index={0: [packed_item]})

    init = _init(init_containers)
    observation = _observation(obs_containers, pool_list=[])

    result = state.build_state(observation, init)

    assert len(result.placed[0]) == 1
    placed = result.placed[0][0]

    np.testing.assert_allclose(placed.pos_world, pos_world, atol=1e-9)
    np.testing.assert_allclose(placed.orn_quat, orn, atol=1e-9)
    np.testing.assert_allclose(placed.size, size, atol=1e-9)
    assert placed.weight == pytest.approx(12.0)
    assert placed.is_soft is True
    assert placed.is_priority is False

    pos_rel = pos_world - np.array([offset_x, 0.0, 0.0])
    half = size / 2.0
    np.testing.assert_allclose(placed.aabb_min_rel, pos_rel - half, atol=1e-9)
    np.testing.assert_allclose(placed.aabb_max_rel, pos_rel + half, atol=1e-9)


def test_state_placeditem_rotated_aabb_composite_quaternion():
    """§4.4テスト節・T-012確定契約: 合成クォータニオン（q_z90 ⊗ q_x90 の標準Hamilton積）による
    回転後AABBが、geometry.rotated_aabb を経由せず解析的に導出した期待値と一致することを
    検証する（atol=1e-6）。geometry.rotated_aabb は本テストの期待値算出・受入条件には使用しない
    （実装後の任意のscratch突合にのみ用いてよい）。

    導出根拠:
        q_z90=(0,0,0.7071068,0.7071068)（Z軸90°、test_geometry.py::test_quat_to_matrix_z90 と同値、
        R_z90=[[0,-1,0],[1,0,0],[0,0,1]]）。
        q_x90=(0.7071068,0,0,0.7071068)（X軸90°、R_x90=[[1,0,0],[0,0,-1],[0,1,0]]、手計算突合済み）。

        標準Hamilton積 q_z90 ⊗ q_x90（q1=q_z90, q2=q_x90として
        w=w1w2-x1x2-y1y2-z1z2, x=w1x2+x1w2+y1z2-z1y2,
        y=w1y2-x1z2+y1w2+z1x2, z=w1z2+x1y2-y1x2+z1w2 を適用）は
        expected_quat = (0.5, 0.5, 0.5, 0.5) に一致する（手計算で確定）。

        対応する回転行列 R = R_z90 @ R_x90 = [[0,0,1],[1,0,0],[0,1,0]]（行列積は手計算）。

        軸整列直方体を回転させたAABBの半寸法は、幾何学的恒等式
        extent_i = sum_j |R_ij| * half_j（各世界軸方向への射影の最大値）で独立に求まる。
        half=(0.25,0.2,0.15) に対し extent=(0.15,0.25,0.20)。
    """
    from src.packing_core import state

    offset_x = 1.8
    specs = [(0, offset_x, WIDE_INNER_MIN, WIDE_INNER_MAX)]

    pos_world = np.array([2.3, 0.6, 0.7], dtype=np.float64)  # pos_rel = (0.5, 0.6, 0.7)
    size = np.array([0.5, 0.4, 0.3], dtype=np.float64)
    expected_quat = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)  # q_z90 ⊗ q_x90（上記docstring参照）

    packed_item = _item(
        index=0, size=size, mass=9.0, pos=pos_world, orn=expected_quat, belongs_to=0,
    )
    init_containers, obs_containers = _container_lists(specs, packed_by_index={0: [packed_item]})

    init = _init(init_containers)
    observation = _observation(obs_containers, pool_list=[])

    result = state.build_state(observation, init)
    placed = result.placed[0][0]

    np.testing.assert_allclose(placed.orn_quat, expected_quat, atol=1e-9)

    pos_rel = pos_world - np.array([offset_x, 0.0, 0.0])  # (0.5, 0.6, 0.7)
    expected_extent = np.array([0.15, 0.25, 0.20], dtype=np.float64)  # R=[[0,0,1],[1,0,0],[0,1,0]] の解析値
    expected_min = pos_rel - expected_extent
    expected_max = pos_rel + expected_extent

    np.testing.assert_allclose(placed.aabb_min_rel, expected_min, atol=1e-6)
    np.testing.assert_allclose(placed.aabb_max_rel, expected_max, atol=1e-6)


# --- pool -> ItemSpec -----------------------------------------------------------------

def test_state_pool_to_itemspec():
    """§4.4手順6・A11: ItemSpec.idx はプール内 index（列挙位置）であり、item["index"] ではない。
    size/weight/is_soft/is_priority の写像も検証する。"""
    from src.packing_core import state

    specs = [(0, 0.0, DEFAULT_INNER_MIN, DEFAULT_INNER_MAX)]
    init_containers, obs_containers = _container_lists(specs)

    # item["index"] をプール内列挙位置とは異なる値にして、idx が列挙位置由来であることを検証する。
    pool_items = [
        _item(index=7, size=(0.55, 0.40, 0.24), mass=8.0, is_soft=False, is_priority=True),
        _item(index=3, size=(0.60, 0.30, 0.25), mass=7.0, is_soft=True, is_priority=False),
        _item(index=9, size=(0.50, 0.40, 0.40), mass=10.0, is_soft=True, is_priority=False),
    ]

    init = _init(init_containers)
    observation = _observation(obs_containers, pool_items)

    result = state.build_state(observation, init)

    assert len(result.pool) == 3
    for pool_i, raw in enumerate(pool_items):
        spec = result.pool[pool_i]
        assert spec.idx == pool_i
        np.testing.assert_allclose(
            spec.size, np.array([raw["length"], raw["width"], raw["height"]]), atol=1e-9
        )
        assert spec.weight == pytest.approx(raw["mass"])
        assert spec.is_soft == raw["is_soft"]
        assert spec.is_priority == raw["is_prioritized"]


def test_state_itemspec_kind_is_none():
    """§3.3/§H/§4.4手順6（T-012確定）: 公式observationにkind相当キーは無いため、build_state経由の
    ItemSpec.kind は常に None になる。"""
    from src.packing_core import state

    specs = [(0, 0.0, DEFAULT_INNER_MIN, DEFAULT_INNER_MAX)]
    init_containers, obs_containers = _container_lists(specs)

    pool_items = [
        _item(index=0, size=(0.5, 0.4, 0.3), mass=10.0, is_soft=False, is_priority=True),
        _item(index=1, size=(0.6, 0.3, 0.25), mass=7.0, is_soft=True, is_priority=False),
    ]

    init = _init(init_containers)
    observation = _observation(obs_containers, pool_items)

    result = state.build_state(observation, init)

    assert len(result.pool) == 2
    for spec in result.pool:
        assert spec.kind is None


# --- 空コンテナの placed/ems/ems_truncation ------------------------------------------

def test_state_empty_container_placed_ems_truncation():
    """§4.4辞書契約: 空・cut/棚なし直方体で全容器indexがキーとして存在し、placed[idx]==[]、
    ems[idx] は内壁AABBに一致する EMS 1個、ems_truncation[idx]==0.0（valid_count=1<=80で打切りなし）。"""
    from src.packing_core import state

    offsets = [0.0, 3.0, 6.5]
    specs = [(i, off, DEFAULT_INNER_MIN, DEFAULT_INNER_MAX) for i, off in enumerate(offsets)]
    init_containers, obs_containers = _container_lists(specs)

    init = _init(init_containers)
    observation = _observation(obs_containers, pool_list=[])

    result = state.build_state(observation, init)

    assert set(result.placed.keys()) == {0, 1, 2}
    assert set(result.ems.keys()) == {0, 1, 2}
    assert set(result.ems_truncation.keys()) == {0, 1, 2}

    for idx in (0, 1, 2):
        assert result.placed[idx] == []
        ems_list = result.ems[idx]
        assert len(ems_list) == 1
        box = ems_list[0]
        np.testing.assert_allclose(box.min_rel, DEFAULT_INNER_MIN, atol=1e-9)
        np.testing.assert_allclose(box.max_rel, DEFAULT_INNER_MAX, atol=1e-9)
        assert result.ems_truncation[idx] == pytest.approx(0.0)


# --- 性能契約テスト ---------------------------------------------------------------

def test_state_build_state_performance_median_under_500ms():
    """§4.4性能予算: 6台・既配置合計約80個で build_state 全体の実行時間中央値が500ms未満
    （1回ウォームアップ、5回計測、time.monotonic()、§2.4）。実装後に実行される契約テストとして
    作成する（state.py 未実装のため現時点では import 失敗で赤になる想定）。

    fixture要件（ユーザー承認条件）:
        各荷物が (a) 対応コンテナの有効空間内、(b) 相互に非重複、(c) belongs_toが配置先
        コンテナindexと一致、(d) identity quat、を満たすことを計測前に _assert_perf_fixture_valid
        で検証する。また、同一observation/initを反復利用しても入力辞書が変更されないことを
        _deep_equal によるスナップショット比較で毎回確認する。
    """
    from src.packing_core import state

    imin = np.array([-0.9, -1.3, 0.02], dtype=np.float64)
    imax = np.array([0.9, 1.3, 1.6], dtype=np.float64)
    offsets = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    counts = [14, 14, 13, 13, 13, 13]  # 合計80個
    assert sum(counts) == 80

    item_size = 0.15
    pitch = 0.2  # > item_size のため相互非重複を保証する

    specs = [(i, off, imin, imax) for i, off in enumerate(offsets)]
    spec_by_index = {i: (imin, imax) for i in range(6)}

    packed_by_index: dict[int, list[dict]] = {}
    for idx, (off, n) in enumerate(zip(offsets, counts)):
        rel_positions = _grid_positions(n, imin, imax, item_size, pitch)
        items = []
        for item_i, rel_pos in enumerate(rel_positions):
            world_pos = rel_pos + np.array([off, 0.0, 0.0])
            items.append(
                _item(
                    index=item_i,
                    size=(item_size, item_size, item_size),
                    mass=4.0,
                    pos=world_pos,
                    orn=IDENTITY_QUAT,
                    belongs_to=idx,
                )
            )
        packed_by_index[idx] = items

    init_containers, obs_containers = _container_lists(specs, packed_by_index=packed_by_index)
    _assert_perf_fixture_valid(obs_containers, spec_by_index)

    init = _init(init_containers, optimize=True, lookahead_k=5)
    observation = _observation(obs_containers, pool_list=[], optimize=True, lookahead_k=5)

    # 反復呼び出しによる入力辞書の非改変を確認するためのスナップショット。
    init_snapshot = copy.deepcopy(init)
    observation_snapshot = copy.deepcopy(observation)

    # 1回ウォームアップ。
    state.build_state(observation, init)
    assert _deep_equal(init, init_snapshot), "build_state が init を変更してはならない"
    assert _deep_equal(observation, observation_snapshot), "build_state が observation を変更してはならない"

    durations = []
    for _ in range(5):
        start = time.monotonic()
        state.build_state(observation, init)
        durations.append(time.monotonic() - start)
        assert _deep_equal(init, init_snapshot), "build_state が init を変更してはならない"
        assert _deep_equal(observation, observation_snapshot), "build_state が observation を変更してはならない"

    median = statistics.median(durations)
    assert median < 0.5, f"build_state median duration {median:.3f}s exceeds 500ms budget"
