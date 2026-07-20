"""統合T-024: float32往復の境界検証（詳細仕様書 v1.18 §4.12「float32往復の境界検証
（candidate_generation_slackの決定試験）」、契約計画 §3.E）。

**分類 I（訂正）**：本ファイルは `candidates.candidate_from_ems`（未実装）で境界Candidateを
生成し、`PlacementParams.candidate_generation_slack`（未実装）を用いる。両者が未実装のため、
authoring時点では実際の数値往復検証に到達する前に `ModuleNotFoundError`/`AttributeError` で
失敗する（他領域と同じ未実装module起因のRED）。**T-024実装後**にこのテストを再実行し、
slack=1e-6での往復不合格が判明した場合に限り、値を自動調整せず**STOP・付録D形式で照会**する
（本ファイル・テストコード自体は変更しない）。

手順（§4.12）: `float64 Candidate.pos_rel → make_action() → float32 place_pos →
float64へ復元 → DIMS/INCLUSION/OVERLAP/CEILING 再判定`。

境界fixtureの設計方針（`candidate_from_ems` のアンカー式・各maskの既存margin公式から
symbolic に導出、ハードコードしない）（HF-001でv1.27改訂：Zもaction位置は想定沈降後位置から
`z_generation_clearance`だけ浮く）:
    xy_required = max(-pp.inclusion_margin + pp.internal_extra, pp.internal_extra)
    xy_clearance = xy_required + pp.candidate_generation_slack
    z_required = max(-pp.inclusion_margin + pp.internal_extra, pp.internal_extra)
    z_clearance = z_required + pp.candidate_generation_slack  # xy_clearanceと同一式
    X: アンカー側=min（近傍、slackぶんの余裕のみ）／far側=max（EMSを2*clearanceぶん
       大きくして遠い側もslackぶんの余裕だけにする）
    Y: アンカー側=max（近傍）／far側=min
    Z: 支持面(想定沈降後位置)から`z_clearance`だけ浮いたaction位置（旧v1.16の
       「支持面へ厳密一致・slack非加算」契約はHF-001で撤回）
    CEILING/OVERLAP: 既存マージン（ceiling_margin+internal_extra／internal_extra）自体の
       往復健全性を検証（こちらはslackに依存しない。ただしCEILINGはZ位置が
       `z_clearance`ぶん上昇した影響を受けるため、fixtureのEMS Z範囲を再設計している）
"""
import numpy as np
import pytest

from src.packing_core import constants
from src.packing_core.container_space import build_container_space
from src.packing_core.state import PackingState
from src.packing_core.types import Candidate, EMSBox, ItemSpec, PlacedItem

INNER_MIN_REL = np.array([-0.40, -0.40, 0.02], dtype=np.float64)
INNER_MAX_REL = np.array([0.40, 0.40, 1.02], dtype=np.float64)
CELL = 0.02
PP0 = constants.PlacementParams()
OSIZE = np.array([0.1, 0.1, 0.1], dtype=np.float64)
# HF-001（v1.27）でZ方向にもz_generation_clearanceを適用するようになったため、コンテナの
# 文字通りの床(INNER_MIN_REL[2])へEMSをフラッシュさせても、action位置は自動的にz_clearance
# だけ浮きINCLUSION合格になる。ただしX/Y境界検証が主目的のfixtureでは、Z自体の境界的な
# 挙動（BND-ZBOT/BND-CEILが担当）と混ざらないよう、引き続きZを床から十分離した安全値を使う。
Z_SAFE_MIN = float(INNER_MIN_REL[2]) + 0.05


def _box_cdict():
    imin, imax = INNER_MIN_REL, INNER_MAX_REL
    mid = (imin + imax) / 2.0
    thickness = 0.02
    buffer = 0.02
    length = float(imax[0] - imin[0]) + 2 * thickness
    width = float(imax[1] - imin[1]) + 2 * thickness
    height = float(imax[2]) + buffer

    face_specs = [
        ((imin[0], mid[1], mid[2]), (-1.0, 0.0, 0.0)),
        ((imax[0], mid[1], mid[2]), (1.0, 0.0, 0.0)),
        ((mid[0], imin[1], mid[2]), (0.0, -1.0, 0.0)),
        ((mid[0], imax[1], mid[2]), (0.0, 1.0, 0.0)),
        ((mid[0], mid[1], imin[2]), (0.0, 0.0, -1.0)),
        ((mid[0], mid[1], imax[2]), (0.0, 0.0, 1.0)),
    ]
    points = [(px, py, pz) for (px, py, pz), _ in face_specs]
    n_vecs = [n for _, n in face_specs]

    return {
        "index": 0, "length": length, "width": width, "height": height,
        "cut_x": 0.0, "cut_y": 0.0, "thickness": thickness,
        "center": (0.0, 0.0, height / 2.0 + buffer),
        "n_vecs": n_vecs, "points": points,
        "volume": float(np.prod(imax - imin)),
        "shelf": False, "is_prioritized": False, "packed_items": [],
    }


def _flat_space():
    return build_container_space(_box_cdict(), index=0, cell=CELL)


def _xy_clearance(pp):
    required = max(-pp.inclusion_margin + pp.internal_extra, pp.internal_extra)
    return required + pp.candidate_generation_slack


def _item(size=OSIZE, idx=0, weight=1.0):
    return ItemSpec(idx=idx, size=np.array(size, dtype=np.float64), weight=weight, kind=None, is_soft=False, is_priority=False)


def _state_for(ems_list, placed=()):
    space = _flat_space()
    return PackingState(
        containers=[space], placed={0: list(placed)}, pool=[_item()],
        ems={0: list(ems_list)}, ems_truncation={0: 0.0}, meta={},
    )


def _generate_and_roundtrip(state, item, ems, pp):
    """candidate_from_emsで生成→make_action→float32→float64復元→4段階再判定。

    Returns: (original_cand, roundtripped_cand)
    """
    from src.packing_core.candidates import candidate_from_ems
    from src.packing_core.masks import MaskStage, evaluate_stage
    from src.packing_core.state import make_action

    cand = candidate_from_ems(item, container_idx=0, orientation=0, ems_id=0, ems=ems, pp=pp)
    assert cand is not None, "fixtureの寸法設計が不正でCandidateが生成されなかった"

    action = make_action(
        item_idx=cand.item_idx, container_idx=cand.container_idx,
        pos_rel=cand.pos_rel, orientation=cand.orientation,
    )
    assert action["place_pos"].dtype == np.float32
    roundtripped_pos = np.asarray(action["place_pos"], dtype=np.float64)

    rt_cand = Candidate(
        item_idx=cand.item_idx, container_idx=cand.container_idx,
        ems_id=cand.ems_id, orientation=cand.orientation,
        pos_rel=roundtripped_pos, osize=cand.osize,
    )
    for stage in (MaskStage.DIMS, MaskStage.INCLUSION, MaskStage.OVERLAP, MaskStage.CEILING):
        evaluate_stage(state, rt_cand, pp, stage)
        assert rt_cand.feasible, (
            f"往復後 stage={stage!r} で不合格（reject_reason={rt_cand.reject_reason!r}）。"
            "STOP: 値を自動調整せず付録D形式で照会すること。"
        )
    return cand, rt_cand


# --- F32-001: 決定論的境界fixture（param×7） -----------------------------------------------


def _fixture_bnd_xmin():
    """Xアンカー側（近傍）: slackぶんの余裕のみで境界ぎりぎり合格。

    EMSのX方向"遠い側"（アンカーに関与しない側）はDIMS事前チェックの境界ちょうど
    （浮動小数点の丸めで偶発的にDIMS不合格になり得る）を避けるため、余裕を持たせて
    大きく取る（DIMSのちょうど境界検証はCAND-008が別途担当）。
    """
    item = _item()
    clr = _xy_clearance(PP0)
    ems = EMSBox(
        min_rel=np.array([INNER_MIN_REL[0], -0.3, Z_SAFE_MIN], dtype=np.float64),
        max_rel=np.array([INNER_MIN_REL[0] + 0.3, 0.3, Z_SAFE_MIN + 0.5], dtype=np.float64),
    )
    return item, ems


def _fixture_bnd_xmax():
    """X遠い側: EMSをclearance*2ぶん大きく取り、far側もslackぶんの余裕だけにする。"""
    item = _item()
    clr = _xy_clearance(PP0)
    ems = EMSBox(
        min_rel=np.array([INNER_MAX_REL[0] - OSIZE[0] - 2 * clr, -0.3, Z_SAFE_MIN], dtype=np.float64),
        max_rel=np.array([INNER_MAX_REL[0], 0.3, Z_SAFE_MIN + 0.5], dtype=np.float64),
    )
    return item, ems


def _fixture_bnd_ymin():
    """Y近い側（Yのfar側、アンカーはY_MAX）: far側もslackぶんの余裕だけにする。"""
    item = _item()
    clr = _xy_clearance(PP0)
    ems = EMSBox(
        min_rel=np.array([-0.3, INNER_MIN_REL[1], Z_SAFE_MIN], dtype=np.float64),
        max_rel=np.array([0.3, INNER_MIN_REL[1] + OSIZE[1] + 2 * clr, Z_SAFE_MIN + 0.5], dtype=np.float64),
    )
    return item, ems


def _fixture_bnd_ymax():
    """Yアンカー側（奥+Y、近傍）: slackぶんの余裕のみで境界ぎりぎり合格。

    Y方向"遠い側"（アンカーに関与しない側）は浮動小数点丸めによる偶発的なDIMS不合格を
    避けるため余裕を持たせる（_fixture_bnd_xminと同方針）。
    """
    item = _item()
    clr = _xy_clearance(PP0)
    ems = EMSBox(
        min_rel=np.array([-0.3, INNER_MAX_REL[1] - 0.3, Z_SAFE_MIN], dtype=np.float64),
        max_rel=np.array([0.3, INNER_MAX_REL[1], Z_SAFE_MIN + 0.5], dtype=np.float64),
    )
    return item, ems


def _fixture_bnd_zbot():
    """Z支持面から`z_clearance`だけ浮いたaction位置（HF-001でv1.27改訂）のsanity往復検証。"""
    item = _item()
    ems = EMSBox(
        min_rel=np.array([-0.3, -0.3, Z_SAFE_MIN], dtype=np.float64),
        max_rel=np.array([0.3, 0.3, Z_SAFE_MIN + 0.5], dtype=np.float64),
    )
    return item, ems


def _fixture_bnd_ceil():
    """CEILING自身のmargin（ceiling_margin+internal_extra）付近の境界（slack非依存）。

    CEILINGは候補生成時に新規余裕を追加しないが、HF-001（v1.27）でaction位置のZが
    想定沈降後位置から`z_clearance`だけ底上げされるため、box_topもそのぶん押し上げられる。
    fixtureのems_min_zはこの押し上げを織り込んで、往復後のbox_topが
    ちょうどCEILING境界のごくわずか内側（tiny_buffer）に来るよう逆算する。
    厳密なゼロ余裕の等号境界ではなく、必要余裕へごくわずかな実余裕（1e-6、X/Yのslackと
    同オーダー）を加えた"ほぼ境界"の位置を使う。等号ちょうどはstrict比較の性質上、
    どちらのfloat32丸め方向が起きても符号が確定せずtest自体が不安定になるため使わない
    （X/Yのslack付きケースと同じ「ごく僅かでも実余裕がある」設計に統一する）。
    EMSのXYは十分大きく取り、DIMS事前チェックの境界ちょうどを偶発的に踏まないようにする。
    """
    item = _item()
    required_clearance = PP0.ceiling_margin + PP0.internal_extra
    z_gen_clearance = _xy_clearance(PP0)  # xy/z共通のclearance式（HF-001）
    tiny_buffer = 1e-6
    # box_top = ems.min_z + osize[2] + z_gen_clearance とし（HF-001でZ位置がz_gen_clearance
    # だけ底上げされるため）、box_top + required_clearance == inner_max_rel[2] - tiny_buffer
    # となるよう ems.min_z を選ぶ（ちょうど境界より tiny_buffer だけ内側＝実余裕を持たせる）。
    ems_min_z = (
        float(INNER_MAX_REL[2]) - OSIZE[2] - z_gen_clearance - required_clearance - tiny_buffer
    )
    # ems.max_rel[2] は新しいZ方向寸法確認（ems.size()[2]>=osize[2]+z_gen_clearance）を
    # 満たしつつ、浮動小数点の減算誤差で偶発的に不合格化しないよう、ems側のZサイズにだけ
    # 十分な余裕（1e-4）を追加する。
    ems = EMSBox(
        min_rel=np.array([-0.3, -0.3, ems_min_z], dtype=np.float64),
        max_rel=np.array(
            [0.3, 0.3, ems_min_z + OSIZE[2] + z_gen_clearance + 1e-4], dtype=np.float64
        ),
    )
    return item, ems


def _fixture_bnd_overlap():
    """OVERLAP自身のtol（-internal_extra、必要隙間=internal_extra）付近の境界（slack非依存）。

    OVERLAPはcandidate_generation_slackの対象外のため、`_fixture_bnd_ceil` と同じ理由
    （strict不等式の等号ちょうどは丸め方向次第で不安定）で、ちょうど`internal_extra`ではなく
    ごくわずかな実余裕（1e-6）を加えた隙間を使う。

    candidate_from_ems を呼ばず、アンカー式を手計算でそのまま適用してbox位置を求め、
    その位置から隙間を空けて隣接する PlacedItem を配置する（production関数を期待値oracleに
    使わず、既知の固定アンカー式から直接算出する）。
    """
    item = _item()
    clr = _xy_clearance(PP0)
    ems = EMSBox(
        min_rel=np.array([-0.3, -0.3, Z_SAFE_MIN], dtype=np.float64),
        max_rel=np.array([0.3, 0.3, Z_SAFE_MIN + 0.5], dtype=np.float64),
    )
    # アンカー式（§4.12、HF-001でv1.27改訂）を手計算で適用:
    # box_min_x = ems.min_x + clr、box_max_x = box_min_x+osize[0]。
    # Zもaction位置は想定沈降後位置(ems.min_z+osize[2]/2)からclr(=z_clearance)だけ浮く。
    box_min_x = float(ems.min_rel[0]) + clr
    box_max_x = box_min_x + OSIZE[0]
    box_min_y = float(ems.max_rel[1]) - clr - OSIZE[1]
    box_max_y = box_min_y + OSIZE[1]
    box_min_z = float(ems.min_rel[2]) + clr
    box_max_z = box_min_z + OSIZE[2]

    gap = PP0.internal_extra + 1e-6  # ちょうど境界ではなく、ごくわずかな実余裕を持たせる
    neighbor_min = np.array([box_max_x + gap, box_min_y, box_min_z], dtype=np.float64)
    neighbor_max = np.array([box_max_x + gap + 0.1, box_max_y, box_max_z], dtype=np.float64)
    neighbor = PlacedItem(
        pos_world=(neighbor_min + neighbor_max) / 2.0,
        orn_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        size=neighbor_max - neighbor_min, weight=1.0, is_soft=False, is_priority=False,
        aabb_min_rel=neighbor_min, aabb_max_rel=neighbor_max,
    )
    return item, ems, [neighbor]


_F32_001_CASES = {
    "T024-F32-001-BND-XMIN": lambda: (*_fixture_bnd_xmin(), []),
    "T024-F32-001-BND-XMAX": lambda: (*_fixture_bnd_xmax(), []),
    "T024-F32-001-BND-YMIN": lambda: (*_fixture_bnd_ymin(), []),
    "T024-F32-001-BND-YMAX": lambda: (*_fixture_bnd_ymax(), []),
    "T024-F32-001-BND-ZBOT": lambda: (*_fixture_bnd_zbot(), []),
    "T024-F32-001-BND-CEIL": lambda: (*_fixture_bnd_ceil(), []),
    "T024-F32-001-BND-OVERLAP": _fixture_bnd_overlap,
}


@pytest.mark.parametrize(
    "case_id",
    [
        pytest.param("T024-F32-001-BND-XMIN", id="T024-F32-001-BND-XMIN"),
        pytest.param("T024-F32-001-BND-XMAX", id="T024-F32-001-BND-XMAX"),
        pytest.param("T024-F32-001-BND-YMIN", id="T024-F32-001-BND-YMIN"),
        pytest.param("T024-F32-001-BND-YMAX", id="T024-F32-001-BND-YMAX"),
        pytest.param("T024-F32-001-BND-ZBOT", id="T024-F32-001-BND-ZBOT"),
        pytest.param("T024-F32-001-BND-CEIL", id="T024-F32-001-BND-CEIL"),
        pytest.param("T024-F32-001-BND-OVERLAP", id="T024-F32-001-BND-OVERLAP"),
    ],
)
def test_f32_001_boundary_fixture_survives_roundtrip(case_id):
    item, ems, placed = _F32_001_CASES[case_id]()
    state = _state_for([ems], placed=placed)
    _generate_and_roundtrip(state, item, ems, PP0)


# --- F32-002: Z支持面側もz_clearanceぶん浮いた位置であることが往復後も維持される ----------


def test_f32_002_z_bottom_clearance_survives_roundtrip():
    item, ems = _fixture_bnd_zbot()
    state = _state_for([ems])
    cand, rt_cand = _generate_and_roundtrip(state, item, ems, PP0)

    # 元候補のZは想定沈降後位置(ems.min_z + osize/2)からz_clearanceだけ浮く（HF-001, v1.27）。
    clr = _xy_clearance(PP0)
    assert cand.pos_rel[2] == pytest.approx(float(ems.min_rel[2]) + cand.osize[2] / 2.0 + clr)
    # 往復後もfloat32丸め誤差の範囲でしか変化しない（XYのslack量1e-6より十分小さいずれのみ許容）。
    assert abs(float(rt_cand.pos_rel[2]) - float(cand.pos_rel[2])) < 1e-5


# --- F32-003: 天井・L_PATHは候補生成時に新規余裕を加えない（Zの支持面側はHF-001で加える） ---


def test_f32_003_ceiling_side_receives_no_extra_clearance_from_slack():
    """Z支持面側はXYと同じ`z_clearance`を加える（HF-001, v1.27）が、ceiling_marginは
    既存判定へ委譲し候補生成時に新規余裕を追加しない（生成直後の pos_rel を直接検証）。"""
    from src.packing_core.candidates import candidate_from_ems

    item, ems = _fixture_bnd_zbot()
    clr = _xy_clearance(PP0)
    cand = candidate_from_ems(item, container_idx=0, orientation=0, ems_id=0, ems=ems, pp=PP0)
    assert cand is not None
    # Z: ems.min_rel[2]+osize[2]/2+clr（HF-001でXYと同式のclearanceを加える）。
    assert cand.pos_rel[2] == pytest.approx(float(ems.min_rel[2]) + cand.osize[2] / 2.0 + clr)

    # slackを大きく変えるとZもXYと同量だけシフトする（支持面側はもはや無関係方向ではない）。
    pp_big_slack = constants.PlacementParams(
        inclusion_margin=PP0.inclusion_margin, safety_margin=PP0.safety_margin,
        start_z=PP0.start_z, ceiling_margin=PP0.ceiling_margin,
        internal_extra=PP0.internal_extra, start_margin=PP0.start_margin,
        candidate_generation_slack=1e-2,
    )
    clr_big = _xy_clearance(pp_big_slack)
    cand_big_slack = candidate_from_ems(item, container_idx=0, orientation=0, ems_id=0, ems=ems, pp=pp_big_slack)
    assert cand_big_slack is not None
    assert cand_big_slack.pos_rel[2] == pytest.approx(float(ems.min_rel[2]) + cand.osize[2] / 2.0 + clr_big)
    assert cand_big_slack.pos_rel[2] - cand.pos_rel[2] == pytest.approx(clr_big - clr)
