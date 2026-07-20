"""統合T-024: candidates.py 候補生成契約テスト（詳細仕様書 v1.18 §4.12「候補列挙・候補生成
契約」「candidates.py 公開API契約」、契約計画 §3.B）。

`src/packing_core/candidates.py` は本チケット時点で未実装のため、対象モジュールの import は
各テスト関数の内部で行う（collection error 回避、`tests/test_masks.py`・
`tests/test_stability.py` と同方針）。全19家族が分類 I（未実装 module）。

候補位置生成式（§4.12、`candidate_from_ems`、HF-001でv1.27改訂：action位置と想定沈降後位置を分離）:
    xy_required_clearance = max(-pp.inclusion_margin + pp.internal_extra, pp.internal_extra)
    xy_generation_clearance = xy_required_clearance + pp.candidate_generation_slack
    z_required_clearance = max(-pp.inclusion_margin + pp.internal_extra, pp.internal_extra)
    z_generation_clearance = z_required_clearance + pp.candidate_generation_slack

    pos_rel[0] = ems.min_rel[0] + osize[0]/2 + xy_generation_clearance   # X: 若い側
    pos_rel[1] = ems.max_rel[1] - osize[1]/2 - xy_generation_clearance   # Y: 奥(+Y)
    pos_rel[2] = ems.min_rel[2] + osize[2]/2 + z_generation_clearance    # Z: 支持面から浮かせたaction位置

候補生成前の寸法確認（全て満たす場合のみCandidateを生成）:
    ems.size()[0] >= osize[0] + xy_generation_clearance
    ems.size()[1] >= osize[1] + xy_generation_clearance
    ems.size()[2] >= osize[2] + z_generation_clearance
"""
import numpy as np
import pytest

from src.packing_core import constants
from src.packing_core.geometry import oriented_size
from src.packing_core.types import EMSBox, ItemSpec

PP0 = constants.PlacementParams()


def _item(idx=0, size=(0.3, 0.2, 0.1), weight=5.0, is_soft=False, is_priority=False):
    return ItemSpec(
        idx=idx,
        size=np.array(size, dtype=np.float64),
        weight=weight,
        kind=None,
        is_soft=is_soft,
        is_priority=is_priority,
    )


def _ems(min_rel, max_rel):
    return EMSBox(
        min_rel=np.array(min_rel, dtype=np.float64),
        max_rel=np.array(max_rel, dtype=np.float64),
    )


def _xy_gen_clearance(pp) -> float:
    """§4.12 の xy_generation_clearance を pp フィールドから手計算する（テスト側oracle）。"""
    xy_required = max(-pp.inclusion_margin + pp.internal_extra, pp.internal_extra)
    return xy_required + pp.candidate_generation_slack


# --- CAND-001/002/003: CandidateKey -------------------------------------------------------


def test_cand_001_candidate_key_type_and_field_order():
    """`CandidateKey = tuple[int,int,int,int]`、順序は
    (item_idx, container_idx, orientation, ems_id)。"""
    from src.packing_core.candidates import candidate_key
    from src.packing_core.types import Candidate

    cand = Candidate(
        item_idx=3, container_idx=1, ems_id=2, orientation=4,
        pos_rel=np.zeros(3), osize=np.ones(3),
    )
    key = candidate_key(cand)
    assert key == (3, 1, 4, 2)


def test_cand_002_candidate_key_all_builtin_int():
    """`candidate_key` の戻り値は全て組込み int（numpy int を漏らさない）。"""
    from src.packing_core.candidates import candidate_key
    from src.packing_core.types import Candidate

    cand = Candidate(
        item_idx=np.int64(3), container_idx=np.int64(1),
        ems_id=np.int64(2), orientation=np.int64(4),
        pos_rel=np.zeros(3), osize=np.ones(3),
    )
    key = candidate_key(cand)
    assert all(type(v) is int for v in key)


def test_cand_003_candidate_key_projects_from_candidate_fields():
    """`candidate_key` の各要素が cand の該当属性の値と一致する（別集計値でない）。"""
    from src.packing_core.candidates import candidate_key
    from src.packing_core.types import Candidate

    cand = Candidate(
        item_idx=7, container_idx=2, ems_id=9, orientation=5,
        pos_rel=np.zeros(3), osize=np.ones(3),
    )
    key = candidate_key(cand)
    assert key[0] == cand.item_idx
    assert key[1] == cand.container_idx
    assert key[2] == cand.orientation
    assert key[3] == cand.ems_id


# --- CAND-004: pos_rel アンカー式（param×6） ----------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        pytest.param("x_anchor", id="T024-CAND-004-X-ANCHOR"),
        pytest.param("y_anchor", id="T024-CAND-004-Y-ANCHOR"),
        pytest.param("z_bottom", id="T024-CAND-004-Z-BOTTOM"),
        pytest.param("xyclr_derived", id="T024-CAND-004-XYCLR-DERIVED"),
        pytest.param("no_hardcode", id="T024-CAND-004-NO-HARDCODE"),
        pytest.param("slack_applied", id="T024-CAND-004-SLACK-APPLIED"),
    ],
)
def test_cand_004_pos_rel_anchor_formula(case):
    from src.packing_core.candidates import candidate_from_ems

    item = _item(size=(0.3, 0.2, 0.1))
    ems = _ems([0.1, 0.2, 0.3], [1.1, 1.2, 1.3])  # 非ゼロオフセットEMS（ハードコード0の検出用）
    osize = oriented_size(item.size, orientation=0)

    if case in ("x_anchor", "y_anchor", "z_bottom"):
        pp = PP0
        clr = _xy_gen_clearance(pp)
        cand = candidate_from_ems(item, container_idx=0, orientation=0, ems_id=0, ems=ems, pp=pp)
        assert cand is not None
        if case == "x_anchor":
            assert cand.pos_rel[0] == pytest.approx(ems.min_rel[0] + osize[0] / 2.0 + clr)
        elif case == "y_anchor":
            assert cand.pos_rel[1] == pytest.approx(ems.max_rel[1] - osize[1] / 2.0 - clr)
        else:  # z_bottom
            assert cand.pos_rel[2] == pytest.approx(ems.min_rel[2] + osize[2] / 2.0 + clr)

    elif case == "xyclr_derived":
        # inclusion_margin/internal_extra/slack を既定から変え、式が定数から導出されている
        # ことを確認する（0.010001直書きなら以下は一致しない）。
        pp = constants.PlacementParams(
            inclusion_margin=-0.02, safety_margin=PP0.safety_margin, start_z=PP0.start_z,
            ceiling_margin=PP0.ceiling_margin, internal_extra=0.01,
            start_margin=PP0.start_margin, candidate_generation_slack=3e-6,
        )
        clr = _xy_gen_clearance(pp)
        cand = candidate_from_ems(item, container_idx=0, orientation=0, ems_id=0, ems=ems, pp=pp)
        assert cand is not None
        assert cand.pos_rel[0] == pytest.approx(ems.min_rel[0] + osize[0] / 2.0 + clr)
        assert cand.pos_rel[1] == pytest.approx(ems.max_rel[1] - osize[1] / 2.0 - clr)

    elif case == "no_hardcode":
        # anchor formula に無関係な pp フィールド（safety_margin/start_margin/start_z/
        # ceiling_margin）を変えても pos_rel は変化しない（無関係な定数へ依存しない）。
        pp_a = PP0
        pp_b = constants.PlacementParams(
            inclusion_margin=PP0.inclusion_margin, safety_margin=0.5, start_z=5.0,
            ceiling_margin=0.5, internal_extra=PP0.internal_extra,
            start_margin=0.5, candidate_generation_slack=PP0.candidate_generation_slack,
        )
        cand_a = candidate_from_ems(item, container_idx=0, orientation=0, ems_id=0, ems=ems, pp=pp_a)
        cand_b = candidate_from_ems(item, container_idx=0, orientation=0, ems_id=0, ems=ems, pp=pp_b)
        assert cand_a is not None and cand_b is not None
        np.testing.assert_allclose(cand_a.pos_rel, cand_b.pos_rel)

    else:  # slack_applied
        pp_a = PP0
        pp_b = constants.PlacementParams(
            inclusion_margin=PP0.inclusion_margin, safety_margin=PP0.safety_margin,
            start_z=PP0.start_z, ceiling_margin=PP0.ceiling_margin,
            internal_extra=PP0.internal_extra, start_margin=PP0.start_margin,
            candidate_generation_slack=PP0.candidate_generation_slack + 5e-6,
        )
        cand_a = candidate_from_ems(item, container_idx=0, orientation=0, ems_id=0, ems=ems, pp=pp_a)
        cand_b = candidate_from_ems(item, container_idx=0, orientation=0, ems_id=0, ems=ems, pp=pp_b)
        assert cand_a is not None and cand_b is not None
        # X/Yはslack差分だけシフトする。
        assert cand_b.pos_rel[0] - cand_a.pos_rel[0] == pytest.approx(5e-6)
        assert cand_a.pos_rel[1] - cand_b.pos_rel[1] == pytest.approx(5e-6)


# --- CAND-005: osize = oriented_size(item.size, orientation) --------------------------


def test_cand_005_osize_matches_oriented_size():
    from src.packing_core.candidates import candidate_from_ems

    item = _item(size=(0.3, 0.2, 0.1))
    ems = _ems([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    orientation = 1  # perm (0,2,1) -> (0.3, 0.1, 0.2)
    expected_osize = oriented_size(item.size, orientation)

    cand = candidate_from_ems(item, container_idx=0, orientation=orientation, ems_id=0, ems=ems, pp=PP0)
    assert cand is not None
    np.testing.assert_allclose(cand.osize, expected_osize)


# --- CAND-006: XYクリアランスはアンカー側1面のみ ------------------------------------------


def test_cand_006_xy_clearance_only_on_anchor_faces():
    """余裕のある大きなEMSで、アンカー側（X若い側・Y奥側）のみ厳密クリアランス、対辺
    （X遠い側・Y近い側）は余りの空間がそのまま残る（対辺にも同じクリアランスを課さない）。"""
    from src.packing_core.candidates import candidate_from_ems

    item = _item(size=(0.3, 0.2, 0.1))
    ems = _ems([0.0, 0.0, 0.0], [10.0, 10.0, 10.0])  # 十分に大きいEMS
    clr = _xy_gen_clearance(PP0)

    cand = candidate_from_ems(item, container_idx=0, orientation=0, ems_id=0, ems=ems, pp=PP0)
    assert cand is not None
    osize = cand.osize

    box_min_x = cand.pos_rel[0] - osize[0] / 2.0
    box_max_x = cand.pos_rel[0] + osize[0] / 2.0
    box_min_y = cand.pos_rel[1] - osize[1] / 2.0
    box_max_y = cand.pos_rel[1] + osize[1] / 2.0

    # アンカー側（X若い側=min、Y奥側=max）は厳密にクリアランス一致。
    assert box_min_x - ems.min_rel[0] == pytest.approx(clr)
    assert ems.max_rel[1] - box_max_y == pytest.approx(clr)

    # 対辺（X遠い側=max、Y近い側=min）はクリアランスと一致しない（余剰空間がそのまま残る）。
    assert ems.max_rel[0] - box_max_x != pytest.approx(clr)
    assert box_min_y - ems.min_rel[1] != pytest.approx(clr)
    assert ems.max_rel[0] - box_max_x > 1.0  # 明確に大きな余剰
    assert box_min_y - ems.min_rel[1] > 1.0


# --- CAND-007: Z支持面側もXYと同じclearance式でslackに応じてシフトする（HF-001, v1.27） ---


def test_cand_007_z_action_position_scales_with_slack():
    """Z方向のaction位置は想定沈降後位置(`ems.min_rel[2]+osize[2]/2`)から
    `z_generation_clearance`だけ浮く。`candidate_generation_slack`を変えるとXYと同じ量だけ
    Zもシフトする（HF-001でのZクリアランス追加、旧v1.16の「Zはslack非加算」契約を撤回）。"""
    from src.packing_core.candidates import candidate_from_ems

    item = _item(size=(0.3, 0.2, 0.1))
    ems = _ems([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    settled_z = float(ems.min_rel[2]) + item.size[2] / 2.0

    pp_a = PP0
    pp_b = constants.PlacementParams(
        inclusion_margin=PP0.inclusion_margin, safety_margin=PP0.safety_margin,
        start_z=PP0.start_z, ceiling_margin=PP0.ceiling_margin,
        internal_extra=PP0.internal_extra, start_margin=PP0.start_margin,
        candidate_generation_slack=PP0.candidate_generation_slack + 1e-3,  # 大きく変える
    )
    cand_a = candidate_from_ems(item, container_idx=0, orientation=0, ems_id=0, ems=ems, pp=pp_a)
    cand_b = candidate_from_ems(item, container_idx=0, orientation=0, ems_id=0, ems=ems, pp=pp_b)
    assert cand_a is not None and cand_b is not None
    # Zは想定沈降後位置(settled_z)からxy/z共通のclearance式ぶんだけ浮く。
    assert cand_a.pos_rel[2] == pytest.approx(settled_z + _xy_gen_clearance(pp_a))
    assert cand_b.pos_rel[2] == pytest.approx(settled_z + _xy_gen_clearance(pp_b))
    # slackを1e-3増やした分、ZもXYと同量だけシフトする（非加算ではなくなった）。
    assert cand_b.pos_rel[2] - cand_a.pos_rel[2] == pytest.approx(1e-3)


# --- CAND-008/009/010: 寸法不足（X/Y/Z）でNone -------------------------------------------


def test_cand_008_insufficient_x_dimension_returns_none():
    from src.packing_core.candidates import candidate_from_ems

    item = _item(size=(0.3, 0.2, 0.1))
    clr = _xy_gen_clearance(PP0)
    # ems.size()[0] = 0.3+clr-1e-4 < osize[0]+clr
    ems = _ems([0.0, 0.0, 0.0], [0.3 + clr - 1e-4, 1.0, 1.0])
    cand = candidate_from_ems(item, container_idx=0, orientation=0, ems_id=0, ems=ems, pp=PP0)
    assert cand is None


def test_cand_009_insufficient_y_dimension_returns_none():
    from src.packing_core.candidates import candidate_from_ems

    item = _item(size=(0.3, 0.2, 0.1))
    clr = _xy_gen_clearance(PP0)
    ems = _ems([0.0, 0.0, 0.0], [1.0, 0.2 + clr - 1e-4, 1.0])
    cand = candidate_from_ems(item, container_idx=0, orientation=0, ems_id=0, ems=ems, pp=PP0)
    assert cand is None


def test_cand_010_insufficient_z_dimension_returns_none():
    from src.packing_core.candidates import candidate_from_ems

    item = _item(size=(0.3, 0.2, 0.1))
    ems = _ems([0.0, 0.0, 0.0], [1.0, 1.0, 0.1 - 1e-4])  # z不足
    cand = candidate_from_ems(item, container_idx=0, orientation=0, ems_id=0, ems=ems, pp=PP0)
    assert cand is None


# --- CAND-011: 寸法充足でCandidateを返す ---------------------------------------------------


def test_cand_011_sufficient_dimensions_returns_candidate():
    from src.packing_core.candidates import candidate_from_ems
    from src.packing_core.types import Candidate

    item = _item(size=(0.3, 0.2, 0.1))
    ems = _ems([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    cand = candidate_from_ems(item, container_idx=0, orientation=0, ems_id=0, ems=ems, pp=PP0)
    assert isinstance(cand, Candidate)


# --- CAND-012: item_idx = int(item.idx)（list位置ではない） ------------------------------


def test_cand_012_item_idx_is_item_spec_idx_not_position():
    from src.packing_core.candidates import candidate_from_ems

    item = _item(idx=42, size=(0.3, 0.2, 0.1))
    ems = _ems([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    cand = candidate_from_ems(item, container_idx=0, orientation=0, ems_id=0, ems=ems, pp=PP0)
    assert cand is not None
    assert cand.item_idx == 42
    assert type(cand.item_idx) is int


# --- CAND-013: container_idx/orientation/ems_id は組込みint ------------------------------


def test_cand_013_index_fields_are_builtin_int():
    from src.packing_core.candidates import candidate_from_ems

    item = _item(size=(0.3, 0.2, 0.1))
    ems = _ems([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    cand = candidate_from_ems(
        item, container_idx=np.int64(2), orientation=np.int64(0), ems_id=np.int64(3),
        ems=ems, pp=PP0,
    )
    assert cand is not None
    assert type(cand.container_idx) is int
    assert type(cand.orientation) is int
    assert type(cand.ems_id) is int
    assert cand.container_idx == 2
    assert cand.ems_id == 3


# --- CAND-014: 入力ItemSpec/EMSBox/配列を変更しない ---------------------------------------


def test_cand_014_does_not_mutate_inputs():
    from src.packing_core.candidates import candidate_from_ems

    item = _item(idx=5, size=(0.3, 0.2, 0.1))
    ems = _ems([0.1, 0.2, 0.3], [1.1, 1.2, 1.3])
    size_before = item.size.copy()
    ems_min_before = ems.min_rel.copy()
    ems_max_before = ems.max_rel.copy()

    candidate_from_ems(item, container_idx=0, orientation=0, ems_id=0, ems=ems, pp=PP0)

    np.testing.assert_array_equal(item.size, size_before)
    np.testing.assert_array_equal(ems.min_rel, ems_min_before)
    np.testing.assert_array_equal(ems.max_rel, ems_max_before)


# --- CAND-015: mask判定を呼ばない ----------------------------------------------------------


def test_cand_015_does_not_call_mask_functions(monkeypatch):
    from src.packing_core import candidates, masks

    calls = {"dims": 0, "inclusion": 0, "overlap": 0, "ceiling": 0, "l_path": 0}

    def _spy(name):
        def _fn(*args, **kwargs):
            calls[name] += 1
            return True
        return _fn

    for mod in (masks, candidates):
        monkeypatch.setattr(mod, "prefilter_dims", _spy("dims"), raising=False)
        monkeypatch.setattr(mod, "check_inclusion", _spy("inclusion"), raising=False)
        monkeypatch.setattr(mod, "check_overlap", _spy("overlap"), raising=False)
        monkeypatch.setattr(mod, "check_ceiling", _spy("ceiling"), raising=False)
        monkeypatch.setattr(mod, "check_l_path", _spy("l_path"), raising=False)

    item = _item(size=(0.3, 0.2, 0.1))
    ems = _ems([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    candidates.candidate_from_ems(item, container_idx=0, orientation=0, ems_id=0, ems=ems, pp=PP0)

    assert calls == {"dims": 0, "inclusion": 0, "overlap": 0, "ceiling": 0, "l_path": 0}


# --- CAND-016: 別EMSへの再割当を行わない（ems_idはそのまま透過） -------------------------


def test_cand_016_no_ems_reassignment_ems_id_passthrough():
    from src.packing_core.candidates import candidate_from_ems

    item = _item(size=(0.3, 0.2, 0.1))
    ems = _ems([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    cand = candidate_from_ems(item, container_idx=0, orientation=0, ems_id=7, ems=ems, pp=PP0)
    assert cand is not None
    assert cand.ems_id == 7  # 別EMSへの付替え・再計算をせず、そのまま透過する


# --- CAND-017: 返すCandidateの初期状態 -----------------------------------------------------


def test_cand_017_returned_candidate_initial_state():
    from src.packing_core.candidates import candidate_from_ems

    item = _item(size=(0.3, 0.2, 0.1))
    ems = _ems([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    cand = candidate_from_ems(item, container_idx=0, orientation=0, ems_id=0, ems=ems, pp=PP0)
    assert cand is not None
    assert cand.feasible is False
    assert cand.reject_reason == ""
    assert cand.score == pytest.approx(0.0)
    assert cand.p_success == pytest.approx(1.0)
    assert cand.features == {}


# --- CAND-018: xy_required_clearance = max(-inclusion_margin+internal_extra, internal_extra) ---


def test_cand_018_xy_required_clearance_floor_branch():
    """inclusion_margin>0（symp的に -inclusion_margin+internal_extra が internal_extra 未満）
    のとき、xy_required_clearance は internal_extra の床値を使う（max()のfloor分岐）。"""
    from src.packing_core.candidates import candidate_from_ems

    pp = constants.PlacementParams(
        inclusion_margin=0.02, safety_margin=PP0.safety_margin, start_z=PP0.start_z,
        ceiling_margin=PP0.ceiling_margin, internal_extra=0.005,
        start_margin=PP0.start_margin, candidate_generation_slack=1e-6,
    )
    # -0.02+0.005 = -0.015 < 0.005 → floor分岐（internal_extra=0.005）が採用される。
    expected_required = 0.005
    assert max(-pp.inclusion_margin + pp.internal_extra, pp.internal_extra) == pytest.approx(
        expected_required
    )

    item = _item(size=(0.3, 0.2, 0.1))
    ems = _ems([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    cand = candidate_from_ems(item, container_idx=0, orientation=0, ems_id=0, ems=ems, pp=pp)
    assert cand is not None
    expected_clr = expected_required + pp.candidate_generation_slack
    osize = cand.osize
    assert cand.pos_rel[0] == pytest.approx(ems.min_rel[0] + osize[0] / 2.0 + expected_clr)


# --- CAND-019: xy_generation_clearance = xy_required_clearance + slack -------------------


def test_cand_019_xy_generation_clearance_adds_slack_exactly():
    from src.packing_core.candidates import candidate_from_ems

    pp = constants.PlacementParams(
        inclusion_margin=0.02, safety_margin=PP0.safety_margin, start_z=PP0.start_z,
        ceiling_margin=PP0.ceiling_margin, internal_extra=0.005,
        start_margin=PP0.start_margin, candidate_generation_slack=7e-6,
    )
    expected_required = 0.005  # floor分岐（CAND-018と同条件）
    expected_generation = expected_required + 7e-6

    item = _item(size=(0.3, 0.2, 0.1))
    ems = _ems([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    cand = candidate_from_ems(item, container_idx=0, orientation=0, ems_id=0, ems=ems, pp=pp)
    assert cand is not None
    osize = cand.osize
    assert cand.pos_rel[1] == pytest.approx(ems.max_rel[1] - osize[1] / 2.0 - expected_generation)
