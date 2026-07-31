"""T-019: パリティ検証（分布(a) = T-018ランダム候補）レポータ（詳細仕様書 §5.3、§6 T-019）。

`datasets/fixtures/geo_verdicts.jsonl`（T-018が公式validatorから生成し凍結したラベル）を
全件再生し、自作 `evaluate_stage`（内部厳格化ON、`PlacementParams()` 既定値）と、保存済みの
公式判定（`official_ok`/`ng_type`）の混同行列を出力する。

対象は3分布のうち **(a) T-018のランダム候補のみ**。(b) 実運転ログの全採用手・(c) その境界摂動
はT-039の責務であり、本スクリプトは実装しない（テレメトリ連携・境界摂動生成は行わない）。

プロジェクト定義のFN（危険側の不一致）:
    project_fn = (self_ok is True) and (official_ok is False)

`self_ok` は呼び出し側（本スクリプト）が `MaskStage.DIMS -> INCLUSION -> OVERLAP -> CEILING ->
L_PATH` の規定順で `evaluate_stage` を1段階ずつ呼び、不合格になった時点で後続段階を呼ばない
（短絡評価）ことで定義する（`evaluate_stage` 自体は単一段階ディスパッチであり、全段階通過の
判定は呼び出し側の責務、詳細仕様書 §4.5）。

公式コード（`src/ground_handling/**`）・PyBullet は一切実行しない。`official_ok`/`ng_type` は
凍結ラベルとして読み取るのみで再計算しない。fixture（`datasets/fixtures/**`）は一切変更しない。

入力不正・schema不整合・state再構築不能・計算例外・未処理レコードは黙ってskipせず、対象の
line番号・state_ref・candidate_indexをログへ出したうえで exit code 2 とする。

exit code:
    0: 入力整合性OK・全件処理・project_fn=0
    1: 入力整合性OK・全件処理・project_fn>0（診断レポートは原子的に保存済み）
    2: 入力不正・state再構築不能・schema不整合・計算例外・未処理レコード・report書込み失敗

実行（`simulator/` で）:
    python -m scripts.parity_report \
        --input datasets/fixtures/geo_verdicts.jsonl \
        --report ../docs/parity/2026-07-19.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from agents.heuristic.packing_core import geometry
from agents.heuristic.packing_core.constants import EPS_GEOM, PlacementParams
from agents.heuristic.packing_core.masks import MaskStage, evaluate_stage
from agents.heuristic.packing_core.state import PackingState, build_state
from agents.heuristic.packing_core.types import Candidate

logger = logging.getLogger("packing")

# ---------------------------------------------------------------------------
# 定数（順序はすべて明示。dictの挿入順やソートの暗黙依存を避けるための固定順）
# ---------------------------------------------------------------------------
STAGE_ORDER: tuple[MaskStage, ...] = (
    MaskStage.DIMS, MaskStage.INCLUSION, MaskStage.OVERLAP, MaskStage.CEILING, MaskStage.L_PATH,
)
STAGE_LABEL: dict[MaskStage, str] = {
    MaskStage.DIMS: "DIMS",
    MaskStage.INCLUSION: "INCLUSION",
    MaskStage.OVERLAP: "OVERLAP",
    MaskStage.CEILING: "CEILING",
    MaskStage.L_PATH: "L_PATH",
}
STAGE_REASON: dict[MaskStage, str] = {
    MaskStage.DIMS: "dims",
    MaskStage.INCLUSION: "inclusion",
    MaskStage.OVERLAP: "overlap",
    MaskStage.CEILING: "ceiling",
    MaskStage.L_PATH: "path",
}
STAGE_LABELS_ORDER: list[str] = [STAGE_LABEL[s] for s in STAGE_ORDER]
FIRST_REJECT_LABELS_ORDER: list[str] = STAGE_LABELS_ORDER + ["(pass)"]
REJECT_REASON_ORDER: list[str] = ["dims", "inclusion", "overlap", "ceiling", "path", ""]
NG_TYPE_ORDER: list[str] = ["", "inclusion", "path"]
ORIENTATION_ORDER: list[int] = [0, 1, 2, 3, 4, 5]
BOOL_ORDER: list[bool] = [True, False]

REQUIRED_TOP_KEYS = {"state_ref", "cand", "official_ok", "ng_type"}
REQUIRED_CAND_KEYS = {
    "candidate_index", "item_idx", "container_idx", "ems_id",
    "ems_min_rel", "ems_max_rel", "orientation", "size", "osize", "pos_rel",
}
ALLOWED_NG_TYPES = {"", "inclusion", "path"}

REQUIRED_STATE_TOP_KEYS = {"init", "observation", "meta"}
REQUIRED_META_KEYS = {
    "schema_version", "run", "step", "task_id", "base_seed",
    "episode_env_seed", "source_config", "source_config_sha256",
    "effective_validator_config",
}
REQUIRED_VALIDATOR_CFG_KEYS = {
    "inclusion_margin", "safety_margin", "start_z", "ceiling_margin", "start_margin",
}

GEN_DIR_PATTERN = re.compile(r"^t018_[0-9a-f]+$")
SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ParityInputError(Exception):
    """入力不正・schema不整合・再構築不能・計算例外・report書込み失敗を表す（exit code 2）。"""


# ---------------------------------------------------------------------------
# 厳密型検証ヘルパ（黙った補正・型変換は一切行わない）
# ---------------------------------------------------------------------------
def _require_strict_int(value: object, field_name: str, ctx: str) -> int:
    """値が厳密なint（bool除外）であることを検証する。

    Args:
        value: 検証対象。
        field_name: エラーメッセージ用のフィールド名。
        ctx: エラーメッセージ用の文脈（line番号・state_ref等）。

    Returns:
        検証済みのint値。

    Raises:
        ParityInputError: `bool`型、または`int`型でない場合。
    """
    if isinstance(value, bool) or type(value) is not int:
        raise ParityInputError(
            f"{ctx}: {field_name} は厳密なint型である必要があります: "
            f"value={value!r} type={type(value).__name__}"
        )
    return value


def _require_strict_bool(value: object, field_name: str, ctx: str) -> bool:
    """値が厳密なbool型であることを検証する。"""
    if type(value) is not bool:
        raise ParityInputError(
            f"{ctx}: {field_name} は厳密なbool型である必要があります: "
            f"value={value!r} type={type(value).__name__}"
        )
    return value


def _require_finite_number(value: object, field_name: str, ctx: str) -> float:
    """値がbool以外の有限実数（int/float）であることを検証しfloatへ変換する。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParityInputError(
            f"{ctx}: {field_name} は数値である必要があります: "
            f"value={value!r} type={type(value).__name__}"
        )
    fval = float(value)
    if not math.isfinite(fval):
        raise ParityInputError(f"{ctx}: {field_name} は有限の実数である必要があります: value={value!r}")
    return fval


def _require_vec3(value: object, field_name: str, ctx: str) -> np.ndarray:
    """値が長さ3のJSON配列（全要素bool以外の有限実数）であることを検証しfloat64配列化する。"""
    if not isinstance(value, list) or len(value) != 3:
        raise ParityInputError(f"{ctx}: {field_name} は長さ3の配列である必要があります: value={value!r}")
    comps = [_require_finite_number(v, f"{field_name}[{i}]", ctx) for i, v in enumerate(value)]
    return np.array(comps, dtype=np.float64)


def _validate_state_ref(state_ref: str, ctx: str) -> None:
    """`state_ref` が空文字・絶対パス・`..`/`.`/空要素を含まないことを検証する。"""
    if state_ref == "":
        raise ParityInputError(f"{ctx}: state_ref が空文字です")
    if state_ref.startswith("/") or state_ref.startswith("\\"):
        raise ParityInputError(f"{ctx}: state_ref が絶対パスです: {state_ref!r}")
    if len(state_ref) > 1 and state_ref[1] == ":":
        raise ParityInputError(f"{ctx}: state_ref がドライブレター付き絶対パスです: {state_ref!r}")
    parts = state_ref.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise ParityInputError(f"{ctx}: state_ref に不正なパス要素が含まれます: {state_ref!r}")


# ---------------------------------------------------------------------------
# geo_verdicts.jsonl レコード
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CandRecord:
    """`geo_verdicts.jsonl` の1行を厳密検証・型付けしたレコード。"""

    line_no: int
    state_ref: str
    candidate_index: int
    item_idx: int
    container_idx: int
    ems_id: int
    ems_min_rel: np.ndarray
    ems_max_rel: np.ndarray
    orientation: int
    size: np.ndarray
    osize: np.ndarray
    pos_rel: np.ndarray
    official_ok: bool
    ng_type: str


def parse_record(line_no: int, raw: dict) -> CandRecord:
    """JSONLの1行分の生dictを検証済み `CandRecord` へ変換する。

    Args:
        line_no: 1始まりの行番号（エラーメッセージ用）。
        raw: `json.loads` 済みの1行分の辞書。

    Returns:
        検証済みの `CandRecord`。

    Raises:
        ParityInputError: schema・型・値域・契約のいずれかに違反する場合。
    """
    ctx = f"line={line_no}"
    missing = REQUIRED_TOP_KEYS - raw.keys()
    if missing:
        raise ParityInputError(f"{ctx}: レコードに必須キーがありません: {sorted(missing)}")

    state_ref = raw["state_ref"]
    if type(state_ref) is not str:
        raise ParityInputError(f"{ctx}: state_ref は文字列である必要があります: {state_ref!r}")
    _validate_state_ref(state_ref, ctx)

    official_ok = _require_strict_bool(raw["official_ok"], "official_ok", ctx)
    ng_type = raw["ng_type"]
    if type(ng_type) is not str or ng_type not in ALLOWED_NG_TYPES:
        raise ParityInputError(f"{ctx}: ng_type が不正です（\"\"|\"inclusion\"|\"path\"のみ）: {ng_type!r}")
    if official_ok and ng_type != "":
        raise ParityInputError(f"{ctx}: official_ok=True なのに ng_type={ng_type!r}（\"\"である必要）")
    if not official_ok and ng_type not in ("inclusion", "path"):
        raise ParityInputError(f"{ctx}: official_ok=False なのに ng_type={ng_type!r}（inclusion|pathである必要）")

    cand = raw["cand"]
    if not isinstance(cand, dict):
        raise ParityInputError(f"{ctx}: cand がobjectではありません")
    missing_c = REQUIRED_CAND_KEYS - cand.keys()
    if missing_c:
        raise ParityInputError(f"{ctx}: cand に必須キーがありません: {sorted(missing_c)}")

    cctx = f"{ctx} state_ref={state_ref}"
    candidate_index = _require_strict_int(cand["candidate_index"], "cand.candidate_index", cctx)
    if candidate_index < 0:
        raise ParityInputError(f"{cctx}: candidate_index が負です: {candidate_index}")
    item_idx = _require_strict_int(cand["item_idx"], "cand.item_idx", cctx)
    if item_idx < 0:
        raise ParityInputError(f"{cctx}: item_idx が負です: {item_idx}")
    container_idx = _require_strict_int(cand["container_idx"], "cand.container_idx", cctx)
    if container_idx < 0:
        raise ParityInputError(f"{cctx}: container_idx が負です: {container_idx}")
    ems_id = _require_strict_int(cand["ems_id"], "cand.ems_id", cctx)
    if ems_id < 0:
        raise ParityInputError(f"{cctx}: ems_id が負です: {ems_id}")
    orientation = _require_strict_int(cand["orientation"], "cand.orientation", cctx)
    if not (0 <= orientation <= 5):
        raise ParityInputError(f"{cctx}: orientation は0..5である必要があります: {orientation}")

    ems_min_rel = _require_vec3(cand["ems_min_rel"], "cand.ems_min_rel", cctx)
    ems_max_rel = _require_vec3(cand["ems_max_rel"], "cand.ems_max_rel", cctx)
    size = _require_vec3(cand["size"], "cand.size", cctx)
    osize = _require_vec3(cand["osize"], "cand.osize", cctx)
    pos_rel = _require_vec3(cand["pos_rel"], "cand.pos_rel", cctx)

    if not np.all(ems_min_rel < ems_max_rel):
        raise ParityInputError(
            f"{cctx}: ems_min_rel < ems_max_rel を全軸で満たしません: min={ems_min_rel} max={ems_max_rel}"
        )
    if not np.all(size > 0):
        raise ParityInputError(f"{cctx}: size は全軸正である必要があります: {size}")
    if not np.all(osize > 0):
        raise ParityInputError(f"{cctx}: osize は全軸正である必要があります: {osize}")

    return CandRecord(
        line_no=line_no, state_ref=state_ref, candidate_index=candidate_index,
        item_idx=item_idx, container_idx=container_idx, ems_id=ems_id,
        ems_min_rel=ems_min_rel, ems_max_rel=ems_max_rel,
        orientation=orientation, size=size, osize=osize, pos_rel=pos_rel,
        official_ok=official_ok, ng_type=ng_type,
    )


def load_and_validate_jsonl(input_path: Path) -> list[CandRecord]:
    """`geo_verdicts.jsonl` を全行読み込み・厳密検証する（黙ったskip禁止）。

    Args:
        input_path: `geo_verdicts.jsonl` への絶対パス。

    Returns:
        行順の検証済み `CandRecord` リスト。

    Raises:
        ParityInputError: decode失敗・空行・重複・candidate_index契約違反等。
    """
    records: list[CandRecord] = []
    seen_keys: set[tuple[str, int]] = set()
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if line.strip() == "":
                raise ParityInputError(f"line {line_no}: 空行が含まれています")
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ParityInputError(f"line {line_no}: JSON decode error: {exc}") from exc
            if not isinstance(raw, dict):
                raise ParityInputError(f"line {line_no}: レコードがobjectではありません")
            rec = parse_record(line_no, raw)
            key = (rec.state_ref, rec.candidate_index)
            if key in seen_keys:
                raise ParityInputError(
                    f"line {line_no}: 重複candidate (state_ref={rec.state_ref}, "
                    f"candidate_index={rec.candidate_index})"
                )
            seen_keys.add(key)
            records.append(rec)

    if not records:
        raise ParityInputError("JSONLにレコードが1件もありません")

    # 各stateのcandidate_indexが 0..count-1 の連続範囲であること（Kの値はハードコードしない）。
    by_state: dict[str, list[int]] = defaultdict(list)
    for rec in records:
        by_state[rec.state_ref].append(rec.candidate_index)
    for state_ref, indices in by_state.items():
        expected = list(range(len(indices)))
        if sorted(indices) != expected:
            raise ParityInputError(
                f"state_ref={state_ref}: candidate_indexが0..{len(indices) - 1}の連続範囲では"
                f"ありません: {sorted(indices)}"
            )

    return records


# ---------------------------------------------------------------------------
# state_ref集合 <-> ディスク上世代ディレクトリの完全一致検証
# ---------------------------------------------------------------------------
def validate_state_directory(fixtures_root: Path, state_refs: list[str]) -> tuple[str, dict[str, Path]]:
    """state_ref集合が単一世代ディレクトリを参照し、そのディレクトリの`*.json`全件と完全一致することを検証する。

    Args:
        fixtures_root: `--input` の親ディレクトリ（`state_ref` の解決基準）。
        state_refs: JSONLに現れた一意な `state_ref` の一覧。

    Returns:
        `(世代ディレクトリ名, {state_ref: 絶対パス})`。

    Raises:
        ParityInputError: 複数世代参照・世代外参照・孤立state・参照先不在のいずれか。
    """
    gen_dirs: set[str] = set()
    referenced: dict[str, Path] = {}
    states_root = (fixtures_root / "states").resolve()

    for ref in state_refs:
        parts = ref.split("/")
        if len(parts) != 3 or parts[0] != "states":
            raise ParityInputError(
                f"state_ref の形式が不正です（states/<gen>/<file>.json の3階層のみ許可）: {ref}"
            )
        gen_name, filename = parts[1], parts[2]
        if not filename.endswith(".json"):
            raise ParityInputError(f"state_ref の拡張子が不正です（.json以外）: {ref}")
        gen_dirs.add(gen_name)
        abs_path = (fixtures_root / ref).resolve()
        expected_dir = (states_root / gen_name).resolve()
        try:
            abs_path.relative_to(expected_dir)
        except ValueError as exc:
            raise ParityInputError(f"state_ref が世代ディレクトリ外を指しています: {ref}") from exc
        referenced[ref] = abs_path

    if not gen_dirs:
        raise ParityInputError("JSONLにstate_refが1件もありません")
    if len(gen_dirs) > 1:
        raise ParityInputError(f"複数世代ディレクトリが参照されています: {sorted(gen_dirs)}")
    gen_name = next(iter(gen_dirs))
    if not GEN_DIR_PATTERN.match(gen_name):
        raise ParityInputError(f"世代ディレクトリ名が規約(t018_<hex>)に一致しません: {gen_name}")

    gen_dir = states_root / gen_name
    if not gen_dir.is_dir():
        raise ParityInputError(f"世代ディレクトリが存在しません: {gen_dir}")

    disk_files = sorted(p for p in gen_dir.glob("*.json") if p.is_file())
    disk_refs = {f"states/{gen_name}/{p.name}" for p in disk_files}
    referenced_set = set(referenced.keys())

    orphan_disk = disk_refs - referenced_set
    missing_disk = referenced_set - disk_refs
    if orphan_disk:
        raise ParityInputError(f"孤立state（JSONLから参照されない*.json）: {sorted(orphan_disk)}")
    if missing_disk:
        raise ParityInputError(f"state_refがディスクに存在しません: {sorted(missing_disk)}")

    return gen_name, referenced


def compute_state_set_hash(state_files_map: dict[str, Path]) -> str:
    """state集合ハッシュを算出する。

    `"<state_ref>\\t<file_sha256>\\n"` を state_ref 昇順で連結したバイト列のSHA-256。
    ファイル名（state_ref）と内容（file_sha256）の両方を拘束する。

    Args:
        state_files_map: `{state_ref: 絶対パス}`。

    Returns:
        16進SHA-256文字列。
    """
    hasher = hashlib.sha256()
    for state_ref in sorted(state_files_map.keys()):
        file_sha256 = hashlib.sha256(state_files_map[state_ref].read_bytes()).hexdigest()
        hasher.update(f"{state_ref}\t{file_sha256}\n".encode("utf-8"))
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# state再構築
# ---------------------------------------------------------------------------
@dataclass
class StateInfo:
    """1 state分の再構築結果とmeta（検証済み）。"""

    state_ref: str
    abs_path: Path
    state: PackingState
    schema_version: str
    run: int
    step: int
    task_id: str
    source_config: str
    source_config_sha256: str
    effective_validator_config: dict
    packed_items_present: bool
    packed_items_count: int


def load_state_info(state_ref: str, abs_path: Path) -> StateInfo:
    """1 state JSONを読み込み・schema検証し、`build_state` で再構築する。

    Args:
        state_ref: `geo_verdicts.jsonl` に現れた相対パス。
        abs_path: 解決済み絶対パス。

    Returns:
        検証済みの `StateInfo`。

    Raises:
        ParityInputError: JSON decode失敗・schema不整合・`build_state` 例外のいずれか。
    """
    ctx = f"state_ref={state_ref}"
    try:
        raw_text = abs_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ParityInputError(f"{ctx}: stateファイル読込失敗: {exc}") from exc
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ParityInputError(f"{ctx}: stateファイルJSON decode失敗: {exc}") from exc
    if not isinstance(payload, dict):
        raise ParityInputError(f"{ctx}: stateファイルのトップレベルがobjectではありません")
    missing = REQUIRED_STATE_TOP_KEYS - payload.keys()
    if missing:
        raise ParityInputError(f"{ctx}: stateファイルに必須キーがありません: {sorted(missing)}")

    init = payload["init"]
    observation = payload["observation"]
    meta = payload["meta"]
    if not isinstance(init, dict) or not isinstance(observation, dict):
        raise ParityInputError(f"{ctx}: init/observation がobjectではありません")
    if not isinstance(meta, dict):
        raise ParityInputError(f"{ctx}: meta がobjectではありません")
    missing_meta = REQUIRED_META_KEYS - meta.keys()
    if missing_meta:
        raise ParityInputError(f"{ctx}: metaに必須キーがありません: {sorted(missing_meta)}")

    schema_version = meta["schema_version"]
    if type(schema_version) is not str:
        raise ParityInputError(f"{ctx}: meta.schema_version は文字列である必要があります")
    run = _require_strict_int(meta["run"], "meta.run", ctx)
    step = _require_strict_int(meta["step"], "meta.step", ctx)
    task_id = meta["task_id"]
    if type(task_id) is not str:
        raise ParityInputError(f"{ctx}: meta.task_id は文字列である必要があります")
    source_config = meta["source_config"]
    if type(source_config) is not str:
        raise ParityInputError(f"{ctx}: meta.source_config は文字列である必要があります")
    source_config_sha256 = meta["source_config_sha256"]
    if type(source_config_sha256) is not str or not SHA256_HEX_PATTERN.match(source_config_sha256):
        raise ParityInputError(f"{ctx}: meta.source_config_sha256 が64桁16進文字列ではありません")
    evc = meta["effective_validator_config"]
    if not isinstance(evc, dict):
        raise ParityInputError(f"{ctx}: meta.effective_validator_config がobjectではありません")
    missing_evc = REQUIRED_VALIDATOR_CFG_KEYS - evc.keys()
    if missing_evc:
        raise ParityInputError(
            f"{ctx}: meta.effective_validator_configに必須キーがありません: {sorted(missing_evc)}"
        )
    for key in sorted(REQUIRED_VALIDATOR_CFG_KEYS):
        _require_finite_number(evc[key], f"meta.effective_validator_config.{key}", ctx)

    try:
        state = build_state(observation, init)
    except Exception as exc:
        raise ParityInputError(f"{ctx}: build_state失敗: {type(exc).__name__}: {exc}") from exc

    container_list = observation.get("container_list")
    if not isinstance(container_list, list):
        raise ParityInputError(f"{ctx}: observation.container_list がlistではありません")
    packed_items_count = 0
    for cdict in container_list:
        if not isinstance(cdict, dict) or not isinstance(cdict.get("packed_items"), list):
            raise ParityInputError(f"{ctx}: container_list要素のpacked_itemsがlistではありません")
        packed_items_count += len(cdict["packed_items"])

    return StateInfo(
        state_ref=state_ref, abs_path=abs_path, state=state,
        schema_version=schema_version, run=run, step=step, task_id=task_id,
        source_config=source_config, source_config_sha256=source_config_sha256,
        effective_validator_config=evc,
        packed_items_present=packed_items_count > 0,
        packed_items_count=packed_items_count,
    )


def validate_meta_consistency(state_infos: dict[str, StateInfo], repo_root: Path) -> None:
    """全stateで `schema_version`/`source_config`/`source_config_sha256`/`effective_validator_config`
    が単一値であること、および `source_config` の実ファイルSHA-256が一致することを検証する。

    Args:
        state_infos: `{state_ref: StateInfo}`。
        repo_root: `source_config`（リポジトリルート基準の相対パス）の解決基準。

    Raises:
        ParityInputError: 複数値混在・実ファイル不在・SHA-256不一致のいずれか。
    """
    schema_versions = {si.schema_version for si in state_infos.values()}
    if len(schema_versions) != 1:
        raise ParityInputError(f"schema_versionが複数存在します: {sorted(schema_versions)}")
    source_configs = {si.source_config for si in state_infos.values()}
    if len(source_configs) != 1:
        raise ParityInputError(f"source_configが複数存在します: {sorted(source_configs)}")
    source_config_shas = {si.source_config_sha256 for si in state_infos.values()}
    if len(source_config_shas) != 1:
        raise ParityInputError(f"source_config_sha256が複数存在します: {sorted(source_config_shas)}")
    evc_json = {json.dumps(si.effective_validator_config, sort_keys=True) for si in state_infos.values()}
    if len(evc_json) != 1:
        raise ParityInputError(f"effective_validator_configが複数存在します: {sorted(evc_json)}")

    source_config_rel = next(iter(source_configs))
    source_config_sha_expected = next(iter(source_config_shas))
    resolved_path = (repo_root / source_config_rel).resolve()
    if not resolved_path.is_file():
        raise ParityInputError(f"source_configが実ファイルとして存在しません: {resolved_path}")
    actual_sha = hashlib.sha256(resolved_path.read_bytes()).hexdigest()
    if actual_sha != source_config_sha_expected:
        raise ParityInputError(
            f"source_configのSHA-256が不一致です: fixture記録={source_config_sha_expected} "
            f"実ファイル={actual_sha} path={resolved_path}"
        )


def validate_placement_params(state_infos: dict[str, StateInfo]) -> PlacementParams:
    """fixtureの `effective_validator_config` と `PlacementParams()` 既定値の5項目一致を検証する。

    `internal_extra` は公式設定に存在しない自作側の安全余裕であるため比較対象外（レポートへは
    別途明記する）。

    Args:
        state_infos: `{state_ref: StateInfo}`（`validate_meta_consistency` 通過後、単一値前提）。

    Returns:
        検証済みの `PlacementParams()`（既定値）。

    Raises:
        ParityInputError: いずれかの項目が不一致の場合（勝手な補正はしない）。
    """
    pp = PlacementParams()
    evc = next(iter(state_infos.values())).effective_validator_config
    mapping = [
        ("inclusion_margin", pp.inclusion_margin),
        ("safety_margin", pp.safety_margin),
        ("start_z", pp.start_z),
        ("ceiling_margin", pp.ceiling_margin),
        ("start_margin", pp.start_margin),
    ]
    mismatches = [
        f"{key}: fixture={float(evc[key])!r} PlacementParams={expected!r}"
        for key, expected in mapping
        if float(evc[key]) != expected
    ]
    if mismatches:
        raise ParityInputError(
            "fixtureのeffective_validator_configとPlacementParams()既定値が不一致: "
            + "; ".join(mismatches)
        )
    return pp


# ---------------------------------------------------------------------------
# Candidate再構築・EMS/pool整合性検証
# ---------------------------------------------------------------------------
def validate_and_build_candidate(rec: CandRecord, state_info: StateInfo) -> Candidate:
    """1候補について state.pool/state.ems との整合性を検証し、新規 `Candidate` を構築する。

    Args:
        rec: 検証済みJSONLレコード。
        state_info: 対応するstateの再構築結果。

    Returns:
        新規構築された `Candidate`（前レコードの可変状態は再利用しない）。

    Raises:
        ParityInputError: item_idx/container_idx/ems_id範囲外、EMS幾何不一致、
            item size不一致、osize不一致のいずれか。
    """
    state = state_info.state
    ctx = f"state_ref={rec.state_ref} candidate_index={rec.candidate_index} line={rec.line_no}"

    if not (0 <= rec.item_idx < len(state.pool)):
        raise ParityInputError(
            f"{ctx}: item_idx範囲外: item_idx={rec.item_idx} pool_len={len(state.pool)}"
        )
    if state.pool[rec.item_idx].idx != rec.item_idx:
        raise ParityInputError(
            f"{ctx}: pool idx不整合: pool[{rec.item_idx}].idx={state.pool[rec.item_idx].idx}"
        )
    if not (0 <= rec.container_idx < len(state.containers)):
        raise ParityInputError(
            f"{ctx}: container_idx範囲外: container_idx={rec.container_idx} "
            f"containers_len={len(state.containers)}"
        )
    ems_list = state.ems[rec.container_idx]
    if not (0 <= rec.ems_id < len(ems_list)):
        raise ParityInputError(
            f"{ctx}: ems_id範囲外: ems_id={rec.ems_id} ems_len={len(ems_list)}"
        )
    ems = ems_list[rec.ems_id]
    if not np.allclose(ems.min_rel, rec.ems_min_rel, rtol=0.0, atol=EPS_GEOM):
        raise ParityInputError(
            f"{ctx}: EMS min_rel不一致: state={ems.min_rel} fixture={rec.ems_min_rel}"
        )
    if not np.allclose(ems.max_rel, rec.ems_max_rel, rtol=0.0, atol=EPS_GEOM):
        raise ParityInputError(
            f"{ctx}: EMS max_rel不一致: state={ems.max_rel} fixture={rec.ems_max_rel}"
        )

    pool_size = state.pool[rec.item_idx].size
    if not np.allclose(pool_size, rec.size, rtol=0.0, atol=EPS_GEOM):
        raise ParityInputError(f"{ctx}: item size不一致: pool={pool_size} fixture={rec.size}")

    try:
        expected_osize = geometry.oriented_size(rec.size, rec.orientation)
    except Exception as exc:
        raise ParityInputError(f"{ctx}: oriented_size計算失敗: {exc}") from exc
    if not np.allclose(expected_osize, rec.osize, rtol=0.0, atol=EPS_GEOM):
        raise ParityInputError(
            f"{ctx}: osize不一致: 期待={expected_osize} fixture={rec.osize}"
        )

    # pos_rel はJSON->float64化のみ（float32往復・座標調整は行わない）。size はCandidate引数に
    # 使わず整合性検証専用（概念上のconstructorに準拠、実際のtypes.pyシグネチャを正とする）。
    return Candidate(
        item_idx=rec.item_idx,
        container_idx=rec.container_idx,
        ems_id=rec.ems_id,
        orientation=rec.orientation,
        pos_rel=np.array(rec.pos_rel, dtype=np.float64),
        osize=np.array(rec.osize, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# 5段階短絡評価（呼び出し側責務、§4.12）
# ---------------------------------------------------------------------------
def run_five_stage_pipeline(
    state: PackingState, cand: Candidate, pp: PlacementParams, ctx: str,
) -> tuple[bool, MaskStage | None, str, dict[str, str]]:
    """DIMS->INCLUSION->OVERLAP->CEILING->L_PATHの規定順で `evaluate_stage` を短絡評価する。

    各呼出し直後に契約（戻り値が同一Candidateオブジェクト・pass時feasible=True/reject_reason=""・
    fail時feasible=False/当該stage規定reject_reason）を検証し、違反は計算例外として送出する。

    Args:
        state: 対象 `PackingState`。
        cand: 判定対象の `Candidate`（本関数呼び出しの間、同一オブジェクトを維持する）。
        pp: 使用する `PlacementParams`。
        ctx: エラーメッセージ用の文脈。

    Returns:
        `(self_ok, first_reject_stage_or_None, final_reject_reason, per_stage_status)`。
        `per_stage_status` は `STAGE_LABEL` 値をキーとする `"pass"|"fail"|"not_run"`。

    Raises:
        ParityInputError: `evaluate_stage` 自体の例外、または戻り値/`feasible`/`reject_reason`
            契約違反。
    """
    per_stage_status: dict[str, str] = {}
    for stage in STAGE_ORDER:
        try:
            result = evaluate_stage(state, cand, pp, stage)
        except Exception as exc:
            raise ParityInputError(
                f"{ctx}: evaluate_stage例外 (stage={stage.name}): {type(exc).__name__}: {exc}"
            ) from exc
        if result is not cand:
            raise ParityInputError(
                f"{ctx}: evaluate_stageが同一Candidateを返さない契約違反 (stage={stage.name})"
            )
        if cand.feasible:
            if cand.reject_reason != "":
                raise ParityInputError(
                    f"{ctx}: 契約違反 feasible=True だが reject_reason={cand.reject_reason!r} "
                    f"(stage={stage.name})"
                )
            per_stage_status[STAGE_LABEL[stage]] = "pass"
        else:
            expected_reason = STAGE_REASON[stage]
            if cand.reject_reason != expected_reason:
                raise ParityInputError(
                    f"{ctx}: 契約違反 feasible=False だが reject_reason={cand.reject_reason!r} "
                    f"（期待={expected_reason!r}） (stage={stage.name})"
                )
            per_stage_status[STAGE_LABEL[stage]] = "fail"
            for later in STAGE_ORDER[STAGE_ORDER.index(stage) + 1:]:
                per_stage_status[STAGE_LABEL[later]] = "not_run"
            return False, stage, expected_reason, per_stage_status
    return True, None, "", per_stage_status


# ---------------------------------------------------------------------------
# 集計
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProjectFnDetail:
    """project_fn（危険側の不一致）1件分の診断情報。"""

    state_ref: str
    candidate_index: int
    run: int
    step: int
    task_id: str
    item_idx: int
    container_idx: int
    ems_id: int
    orientation: int
    pos_rel: tuple[float, float, float]
    size: tuple[float, float, float]
    osize: tuple[float, float, float]
    official_ng_type: str
    stage_results: dict[str, str]
    final_reject_reason: str
    packed_items_count: int
    ems_min_rel: tuple[float, float, float]
    ems_max_rel: tuple[float, float, float]


@dataclass
class Stats:
    """全レコードの集計結果。"""

    total_records: int = 0
    unique_states: int = 0
    unique_runs: int = 0
    self_ok_count: int = 0
    official_ok_count: int = 0
    confusion: Counter = field(default_factory=Counter)
    official_ng_type_counts: Counter = field(default_factory=Counter)
    task_official_counts: Counter = field(default_factory=Counter)
    packed_official_counts: Counter = field(default_factory=Counter)
    first_reject_stage_counts: Counter = field(default_factory=Counter)
    reject_reason_counts: Counter = field(default_factory=Counter)
    orientation_self_counts: Counter = field(default_factory=Counter)
    task_self_counts: Counter = field(default_factory=Counter)
    packed_self_counts: Counter = field(default_factory=Counter)
    orientation_project_fn_counts: Counter = field(default_factory=Counter)
    task_project_fn_counts: Counter = field(default_factory=Counter)
    container_project_fn_counts: Counter = field(default_factory=Counter)
    official_ng_type_project_fn_counts: Counter = field(default_factory=Counter)
    conservative_miss_count: int = 0
    first_reject_stage_conservative_counts: Counter = field(default_factory=Counter)
    project_fn_records: list[ProjectFnDetail] = field(default_factory=list)


def evaluate_all(
    records: list[CandRecord], state_infos: dict[str, StateInfo], pp: PlacementParams,
) -> Stats:
    """全レコードについてCandidate再構築・5段階評価・集計を行う。

    Args:
        records: 検証済みJSONLレコード（行順）。
        state_infos: `{state_ref: StateInfo}`。
        pp: 使用する `PlacementParams`。

    Returns:
        集計結果 `Stats`。

    Raises:
        ParityInputError: いずれかのレコードでCandidate再構築・5段階評価が失敗した場合。
    """
    stats = Stats()
    stats.total_records = len(records)
    stats.unique_states = len(state_infos)
    stats.unique_runs = len({si.run for si in state_infos.values()})

    for rec in records:
        state_info = state_infos[rec.state_ref]
        ctx = f"state_ref={rec.state_ref} candidate_index={rec.candidate_index} line={rec.line_no}"
        cand = validate_and_build_candidate(rec, state_info)
        self_ok, first_reject_stage, final_reason, stage_status = run_five_stage_pipeline(
            state_info.state, cand, pp, ctx
        )
        official_ok = rec.official_ok

        stats.confusion[(self_ok, official_ok)] += 1
        if self_ok:
            stats.self_ok_count += 1
        if official_ok:
            stats.official_ok_count += 1
        stats.official_ng_type_counts[rec.ng_type] += 1
        stats.task_official_counts[(state_info.task_id, official_ok)] += 1
        stats.packed_official_counts[(state_info.packed_items_present, official_ok)] += 1

        first_reject_label = STAGE_LABEL[first_reject_stage] if first_reject_stage is not None else "(pass)"
        stats.first_reject_stage_counts[first_reject_label] += 1
        stats.reject_reason_counts[final_reason] += 1
        stats.orientation_self_counts[(rec.orientation, self_ok)] += 1
        stats.task_self_counts[(state_info.task_id, self_ok)] += 1
        stats.packed_self_counts[(state_info.packed_items_present, self_ok)] += 1

        is_project_fn = self_ok and not official_ok
        is_conservative_miss = (not self_ok) and official_ok

        if is_project_fn:
            stats.project_fn_records.append(ProjectFnDetail(
                state_ref=rec.state_ref, candidate_index=rec.candidate_index,
                run=state_info.run, step=state_info.step, task_id=state_info.task_id,
                item_idx=rec.item_idx, container_idx=rec.container_idx, ems_id=rec.ems_id,
                orientation=rec.orientation,
                pos_rel=tuple(float(v) for v in rec.pos_rel),
                size=tuple(float(v) for v in rec.size),
                osize=tuple(float(v) for v in rec.osize),
                official_ng_type=rec.ng_type,
                stage_results=dict(stage_status),
                final_reject_reason=final_reason,
                packed_items_count=state_info.packed_items_count,
                ems_min_rel=tuple(float(v) for v in rec.ems_min_rel),
                ems_max_rel=tuple(float(v) for v in rec.ems_max_rel),
            ))
            stats.orientation_project_fn_counts[rec.orientation] += 1
            stats.task_project_fn_counts[state_info.task_id] += 1
            stats.container_project_fn_counts[rec.container_idx] += 1
            stats.official_ng_type_project_fn_counts[rec.ng_type] += 1

        if is_conservative_miss:
            stats.conservative_miss_count += 1
            stats.first_reject_stage_conservative_counts[first_reject_label] += 1

    return stats


# ---------------------------------------------------------------------------
# ハッシュ・Git情報
# ---------------------------------------------------------------------------
def get_git_head(repo_root: Path) -> str:
    """`.git/HEAD` を読み解いて現在のコミットハッシュを取得する（git CLI不使用、読み取り専用）。

    Args:
        repo_root: リポジトリルート（`.git` を直下に持つディレクトリ）。

    Returns:
        40桁16進のコミットハッシュ文字列。

    Raises:
        ParityInputError: `.git/HEAD` が解決できない場合。
    """
    git_dir = repo_root / ".git"
    head_file = git_dir / "HEAD"
    if not head_file.is_file():
        raise ParityInputError(f".git/HEAD が見つかりません: {head_file}")
    content = head_file.read_text(encoding="utf-8").strip()
    if not content.startswith("ref:"):
        return content
    ref_path = content.split(" ", 1)[1].strip()
    ref_file = git_dir / ref_path
    if ref_file.is_file():
        return ref_file.read_text(encoding="utf-8").strip()
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for pline in packed.read_text(encoding="utf-8").splitlines():
            if pline.endswith(" " + ref_path):
                return pline.split(" ", 1)[0]
    raise ParityInputError(f"git HEAD ref を解決できません: {ref_path}")


def git_blob_sha1(content: bytes) -> str:
    """git blobオブジェクトのSHA-1（`git hash-object` 相当）を算出する。"""
    header = f"blob {len(content)}\0".encode("utf-8")
    return hashlib.sha1(header + content).hexdigest()


# ---------------------------------------------------------------------------
# レポート生成
# ---------------------------------------------------------------------------
def _bool_str(b: bool) -> str:
    return "True" if b else "False"


def _pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return f"{numerator}/0 (N/A)"
    return f"{numerator}/{denominator} ({100.0 * numerator / denominator:.4f}%)"


def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def build_report(
    *,
    git_head: str,
    script_sha256: str,
    masks_blob_sha1: str,
    masks_sha256: str,
    constants_sha256: str,
    input_path_display: str,
    input_sha256: str,
    state_set_hash: str,
    gen_dir_name: str,
    schema_version: str,
    source_config: str,
    source_config_sha256: str,
    pp: PlacementParams,
    all_task_ids: list[str],
    all_container_idxs: list[int],
    task_id_state_counts: Counter,
    task_id_record_counts: Counter,
    stats: Stats,
) -> str:
    """全集計結果からMarkdownレポート本文を組み立てる（順序は完全に決定論的）。"""
    lines: list[str] = []

    # 1. タイトルと対象分布
    lines.append("# T-019 パリティレポート（分布(a): T-018ランダム候補）")
    lines.append("")
    lines.append(
        "対象分布は詳細仕様書 §5.3 の3分布のうち **(a) T-018のランダム候補のみ**。"
        "(b) 実運転ログの全採用手・(c) その境界摂動はT-039の責務であり本レポートの対象外。"
    )
    lines.append("")
    lines.append(
        "project_fn定義: `self_ok is True and official_ok is False`"
        "（自作マスクは配置可能と判定したが、公式validatorラベルは配置不可＝危険側の不一致）。"
    )
    lines.append("")

    # 2. 実行対象Git HEAD
    lines.append("## 実行対象")
    lines.append("")
    lines.append(f"- Git HEAD: `{git_head}`")
    # 3-5. ハッシュ
    lines.append(f"- `scripts/parity_report.py` SHA-256: `{script_sha256}`")
    lines.append(f"- `agents/heuristic/packing_core/masks.py` git blob SHA-1: `{masks_blob_sha1}`")
    lines.append(f"- `agents/heuristic/packing_core/masks.py` SHA-256: `{masks_sha256}`")
    lines.append(f"- `agents/heuristic/packing_core/constants.py` SHA-256: `{constants_sha256}`")
    # 6. 入力
    lines.append(f"- 入力 `geo_verdicts.jsonl` パス: `{input_path_display}`")
    lines.append(f"- 入力 `geo_verdicts.jsonl` SHA-256: `{input_sha256}`")
    # 7. state集合ハッシュ
    lines.append(
        f"- state集合ハッシュ（相対パス昇順で `<state_ref>\\t<file_sha256>\\n` を連結した"
        f"バイト列のSHA-256）: `{state_set_hash}`"
    )
    # 8-9. schema/dataset世代
    lines.append(f"- fixture schema_version: `{schema_version}`")
    lines.append(f"- dataset世代ディレクトリ: `{gen_dir_name}`")
    # 10. source_config
    lines.append(f"- source_config: `{source_config}`")
    lines.append(f"- source_config SHA-256: `{source_config_sha256}`（実ファイルと一致確認済み）")
    lines.append("")

    # 11. 使用したPlacementParams全値
    lines.append("## 使用したPlacementParams")
    lines.append("")
    lines.append(
        "`PlacementParams()` 既定値を使用（内部厳格化ON）。fixtureの `effective_validator_config` "
        "5項目と一致確認済み。`internal_extra` は公式設定に存在しない自作側の安全余裕であり"
        "比較対象外（値のみ以下に明記）。"
    )
    lines.append("")
    lines.extend(_md_table(
        ["field", "value"],
        [
            ["inclusion_margin", repr(pp.inclusion_margin)],
            ["safety_margin", repr(pp.safety_margin)],
            ["start_z", repr(pp.start_z)],
            ["ceiling_margin", repr(pp.ceiling_margin)],
            ["internal_extra", repr(pp.internal_extra) + "（公式設定外・自作側余裕）"],
            ["start_margin", repr(pp.start_margin)],
        ],
    ))
    lines.append("")

    # 12. 入力整合性検査結果
    lines.append("## 入力整合性検査結果")
    lines.append("")
    lines.append("以下の項目はすべてPASS（1件でも違反があればこのレポートは生成されず exit code 2）。")
    lines.append("")
    checklist = [
        "JSONL全行のJSON decode・必須キー・型（strict int/bool・有限float64長さ3配列）",
        "official_ok/ng_type契約（bool厳密・ng_type∈{\"\",\"inclusion\",\"path\"}・True⇔\"\"・False⇔{inclusion,path}）",
        "(state_ref, candidate_index)の一意性、各stateのcandidate_indexが0..count-1の連続範囲",
        "ems_min_rel<ems_max_rel、size>0、osize>0（全軸）",
        "state_ref集合と単一世代ディレクトリ上の*.json全件の完全一致（孤立state・世代外参照・複数世代参照なし）",
        "全stateのschema_version/source_config/source_config_sha256/effective_validator_configが単一値",
        "source_configの実ファイルSHA-256とmeta記録値の一致",
        "fixtureのeffective_validator_config 5項目とPlacementParams()既定値の一致",
        "全stateのbuild_state再構築成功",
        "全候補のitem_idx/container_idx/ems_id範囲・EMS幾何・item size・osizeの再構築整合性（EPS_GEOM許容）",
        "全候補の5段階evaluate_stage呼出し契約（同一Candidate返却・feasible/reject_reason整合）",
    ]
    for item in checklist:
        lines.append(f"- [x] {item}")
    lines.append("")

    # 13. 総レコード数・state数・run数
    lines.append("## 総数")
    lines.append("")
    lines.append(f"- 総レコード数: {stats.total_records}")
    lines.append(f"- 一意state数: {stats.unique_states}")
    lines.append(f"- run数: {stats.unique_runs}")
    lines.append("")
    lines.append("### task_id分布")
    lines.append("")
    lines.extend(_md_table(
        ["task_id", "state数", "レコード数"],
        [[tid, str(task_id_state_counts.get(tid, 0)), str(task_id_record_counts.get(tid, 0))]
         for tid in all_task_ids],
    ))
    lines.append("")

    # 14. 4区分の混同行列
    lines.append("## 混同行列（4区分）")
    lines.append("")
    lines.extend(_md_table(
        ["self_ok", "official_ok", "boolean条件", "件数"],
        [
            ["True", "True", "self_ok=True / official_ok=True",
             str(stats.confusion.get((True, True), 0))],
            ["True", "False", "self_ok=True / official_ok=False  ← project_fn",
             str(stats.confusion.get((True, False), 0))],
            ["False", "True", "self_ok=False / official_ok=True  ← 保守的取りこぼし",
             str(stats.confusion.get((False, True), 0))],
            ["False", "False", "self_ok=False / official_ok=False",
             str(stats.confusion.get((False, False), 0))],
        ],
    ))
    lines.append("")
    match_count = stats.confusion.get((True, True), 0) + stats.confusion.get((False, False), 0)
    lines.append(f"- self_ok件数: {stats.self_ok_count}")
    lines.append(f"- official_ok件数: {stats.official_ok_count}")
    lines.append(f"- 一致率: {_pct(match_count, stats.total_records)}")
    lines.append("")

    # 15. project_fn件数・率
    project_fn_count = len(stats.project_fn_records)
    lines.append("## project_fn（危険側の不一致）")
    lines.append("")
    lines.append(f"- project_fn件数: {project_fn_count}")
    lines.append(f"- project_fn率（分母=全レコード）: {_pct(project_fn_count, stats.total_records)}")
    lines.append("")

    # 16. 保守的取りこぼし件数・率
    lines.append("## 保守的取りこぼし（self_ok=False かつ official_ok=True）")
    lines.append("")
    lines.append(f"- 件数: {stats.conservative_miss_count}")
    lines.append(
        f"- 率（分母=official_ok=True件数={stats.official_ok_count}）: "
        f"{_pct(stats.conservative_miss_count, stats.official_ok_count)}"
    )
    lines.append("")
    lines.append("### 保守的取りこぼしの first reject stage 別件数")
    lines.append("")
    lines.extend(_md_table(
        ["first reject stage", "件数"],
        [[label, str(stats.first_reject_stage_conservative_counts.get(label, 0))]
         for label in FIRST_REJECT_LABELS_ORDER],
    ))
    lines.append("")

    # 17. official ng_type分布
    lines.append("## official側集計")
    lines.append("")
    lines.append("### official ng_type別件数")
    lines.append("")
    lines.extend(_md_table(
        ["ng_type", "件数"],
        [[repr(nt) if nt != "" else '""', str(stats.official_ng_type_counts.get(nt, 0))]
         for nt in NG_TYPE_ORDER],
    ))
    lines.append("")
    lines.append("### task_id別official_ok件数")
    lines.append("")
    lines.extend(_md_table(
        ["task_id", "official_ok=True", "official_ok=False"],
        [[tid,
          str(stats.task_official_counts.get((tid, True), 0)),
          str(stats.task_official_counts.get((tid, False), 0))]
         for tid in all_task_ids],
    ))
    lines.append("")
    lines.append("### packed_items有無別official_ok件数")
    lines.append("")
    lines.extend(_md_table(
        ["packed_items", "official_ok=True", "official_ok=False"],
        [[_bool_str(p),
          str(stats.packed_official_counts.get((p, True), 0)),
          str(stats.packed_official_counts.get((p, False), 0))]
         for p in BOOL_ORDER],
    ))
    lines.append("")

    # 18. self reject stage／reason分布
    lines.append("## self側集計")
    lines.append("")
    lines.append("### first reject stage別件数（\"(pass)\"=全5段階通過）")
    lines.append("")
    lines.extend(_md_table(
        ["first reject stage", "件数"],
        [[label, str(stats.first_reject_stage_counts.get(label, 0))] for label in FIRST_REJECT_LABELS_ORDER],
    ))
    lines.append("")
    lines.append("### reject_reason別件数（\"\"=全段階通過）")
    lines.append("")
    lines.extend(_md_table(
        ["reject_reason", "件数"],
        [[repr(r) if r != "" else '""', str(stats.reject_reason_counts.get(r, 0))]
         for r in REJECT_REASON_ORDER],
    ))
    lines.append("")
    lines.append("### orientation別self_ok件数")
    lines.append("")
    lines.extend(_md_table(
        ["orientation", "self_ok=True", "self_ok=False"],
        [[str(o),
          str(stats.orientation_self_counts.get((o, True), 0)),
          str(stats.orientation_self_counts.get((o, False), 0))]
         for o in ORIENTATION_ORDER],
    ))
    lines.append("")
    lines.append("### task_id別self_ok件数")
    lines.append("")
    lines.extend(_md_table(
        ["task_id", "self_ok=True", "self_ok=False"],
        [[tid,
          str(stats.task_self_counts.get((tid, True), 0)),
          str(stats.task_self_counts.get((tid, False), 0))]
         for tid in all_task_ids],
    ))
    lines.append("")
    lines.append("### packed_items有無別self_ok件数")
    lines.append("")
    lines.extend(_md_table(
        ["packed_items", "self_ok=True", "self_ok=False"],
        [[_bool_str(p),
          str(stats.packed_self_counts.get((p, True), 0)),
          str(stats.packed_self_counts.get((p, False), 0))]
         for p in BOOL_ORDER],
    ))
    lines.append("")

    # 19 (不一致側の残り): orientation/task/container別project_fn、official ng_type別project_fn
    lines.append("## 不一致側集計（project_fn内訳）")
    lines.append("")
    lines.append("### orientation別project_fn件数")
    lines.append("")
    lines.extend(_md_table(
        ["orientation", "project_fn件数"],
        [[str(o), str(stats.orientation_project_fn_counts.get(o, 0))] for o in ORIENTATION_ORDER],
    ))
    lines.append("")
    lines.append("### task_id別project_fn件数")
    lines.append("")
    lines.extend(_md_table(
        ["task_id", "project_fn件数"],
        [[tid, str(stats.task_project_fn_counts.get(tid, 0))] for tid in all_task_ids],
    ))
    lines.append("")
    lines.append("### container_idx別project_fn件数")
    lines.append("")
    lines.extend(_md_table(
        ["container_idx", "project_fn件数"],
        [[str(ci), str(stats.container_project_fn_counts.get(ci, 0))] for ci in all_container_idxs],
    ))
    lines.append("")
    lines.append("### 公式ng_type別project_fn件数")
    lines.append("")
    lines.extend(_md_table(
        ["official ng_type", "project_fn件数"],
        [[repr(nt), str(stats.official_ng_type_project_fn_counts.get(nt, 0))]
         for nt in ("inclusion", "path")],
    ))
    lines.append("")

    # 20. project_fnの詳細
    lines.append("## project_fn詳細")
    lines.append("")
    if project_fn_count == 0:
        lines.append("project_fn は0件でした（DoD達成）。詳細テーブルなし。")
        lines.append("")
    else:
        sorted_fns = sorted(stats.project_fn_records, key=lambda d: (d.state_ref, d.candidate_index))
        lines.append(f"project_fn {project_fn_count}件の全識別子と診断情報を以下に示す。")
        lines.append("")
        for i, d in enumerate(sorted_fns, start=1):
            lines.append(f"### project_fn #{i}: state_ref=`{d.state_ref}` candidate_index={d.candidate_index}")
            lines.append("")
            lines.extend(_md_table(
                ["field", "value"],
                [
                    ["run", str(d.run)],
                    ["step", str(d.step)],
                    ["task_id", d.task_id],
                    ["item_idx", str(d.item_idx)],
                    ["container_idx", str(d.container_idx)],
                    ["ems_id", str(d.ems_id)],
                    ["orientation", str(d.orientation)],
                    ["pos_rel", repr(d.pos_rel)],
                    ["size", repr(d.size)],
                    ["osize", repr(d.osize)],
                    ["ems_min_rel", repr(d.ems_min_rel)],
                    ["ems_max_rel", repr(d.ems_max_rel)],
                    ["official ng_type", repr(d.official_ng_type)],
                    ["final self reject_reason", repr(d.final_reject_reason)],
                    ["packed_items数", str(d.packed_items_count)],
                ],
            ))
            lines.append("")
            lines.append("MaskStage別結果:")
            lines.append("")
            lines.extend(_md_table(
                ["stage", "result"],
                [[label, d.stage_results.get(label, "not_run")] for label in STAGE_LABELS_ORDER],
            ))
            lines.append("")

    # 21. 未処理・skip・例外件数
    lines.append("## 未処理・skip・例外")
    lines.append("")
    lines.append(
        "本スクリプトは不正入力・schema不整合・再構築不能・計算例外を検出した時点で即座に "
        "exit code 2 とし、レポートを生成しない（黙ったskip・退化的なFalse変換は行わない）。"
        "本レポートが生成されている時点で以下はすべて0件。"
    )
    lines.append("")
    lines.append("- skip件数: 0")
    lines.append("- 未処理レコード数: 0")
    lines.append("- 処理例外件数: 0")
    lines.append("")

    # 22-23. 最終判定・想定exit code
    lines.append("## 最終判定")
    lines.append("")
    if project_fn_count == 0:
        lines.append("**PASS**（project_fn=0、DoD達成）")
        lines.append("")
        lines.append("想定exit code: **0**")
    else:
        lines.append(f"**FAIL**（project_fn={project_fn_count}件、DoD未達。要修正・本セッションでは修正しない）")
        lines.append("")
        lines.append("想定exit code: **1**")
    lines.append("")

    return "\n".join(lines) + "\n"


def atomic_write(report_path: Path, content: str) -> None:
    """レポートを一時ファイルへ書いたのち `os.replace` で原子的に公開する。

    Args:
        report_path: 最終出力パス。
        content: 書き込む本文（改行込み）。

    Raises:
        ParityInputError: 親ディレクトリ不在、または書込み失敗の場合（親ディレクトリの作成は行わない）。
    """
    parent = report_path.parent
    if not parent.is_dir():
        raise ParityInputError(f"--reportの親ディレクトリが存在しません（作成しません）: {parent}")
    fd, tmp_name = tempfile.mkstemp(prefix=".parity_report_", suffix=".tmp", dir=str(parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, report_path)
    except Exception as exc:
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        raise ParityInputError(f"reportの書込みに失敗しました: {report_path}: {exc}") from exc


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """CLI引数を解析する（argparseの引数エラーはexit code 2、許容契約）。"""
    parser = argparse.ArgumentParser(
        description=(
            "T-019: T-018幾何判定フィクスチャ（ランダム候補分布）の自作/公式パリティレポート生成"
        ),
    )
    parser.add_argument("--input", required=True, type=str, help="geo_verdicts.jsonl へのパス")
    parser.add_argument("--report", required=True, type=str, help="出力Markdownレポートへのパス")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    """検証・パリティ集計・レポート生成の全体フローを実行する。

    Args:
        args: `parse_args()` の戻り値。

    Returns:
        0（project_fn=0）または1（project_fn>0）。

    Raises:
        ParityInputError: 入力不正・再構築不能・計算例外・report書込み失敗のいずれか
            （exit code 2として `main()` が処理する）。
    """
    script_path = Path(__file__).resolve()
    simulator_root = script_path.parents[1]
    repo_root = simulator_root.parent

    input_path = Path(args.input)
    input_path = input_path.resolve() if input_path.is_absolute() else (Path.cwd() / input_path).resolve()
    if not input_path.is_file():
        raise ParityInputError(f"--input が存在しません: {input_path}")
    fixtures_root = input_path.parent

    report_path = Path(args.report)
    report_path = report_path.resolve() if report_path.is_absolute() else (Path.cwd() / report_path).resolve()

    logger.info("input=%s report=%s", input_path, report_path)

    records = load_and_validate_jsonl(input_path)
    logger.info("JSONLレコード読込・スキーマ検証完了: %d件", len(records))

    state_refs = sorted({r.state_ref for r in records})
    gen_dir_name, state_files_map = validate_state_directory(fixtures_root, state_refs)
    logger.info(
        "state_ref集合とディスク上世代ディレクトリの完全一致を確認: gen=%s states=%d",
        gen_dir_name, len(state_files_map),
    )

    state_set_hash = compute_state_set_hash(state_files_map)

    state_infos: dict[str, StateInfo] = {
        state_ref: load_state_info(state_ref, state_files_map[state_ref])
        for state_ref in sorted(state_files_map.keys())
    }
    logger.info("全stateのbuild_state再構築完了: %d件", len(state_infos))

    validate_meta_consistency(state_infos, repo_root)
    pp = validate_placement_params(state_infos)
    logger.info(
        "PlacementParams検証OK: inclusion_margin=%r safety_margin=%r start_z=%r "
        "ceiling_margin=%r internal_extra=%r start_margin=%r",
        pp.inclusion_margin, pp.safety_margin, pp.start_z, pp.ceiling_margin,
        pp.internal_extra, pp.start_margin,
    )

    stats = evaluate_all(records, state_infos, pp)
    logger.info(
        "全%d件処理完了: self_ok=%d official_ok=%d project_fn=%d conservative_miss=%d",
        stats.total_records, stats.self_ok_count, stats.official_ok_count,
        len(stats.project_fn_records), stats.conservative_miss_count,
    )

    git_head = get_git_head(repo_root)
    script_sha256 = hashlib.sha256(script_path.read_bytes()).hexdigest()
    masks_path = simulator_root / "src" / "packing_core" / "masks.py"
    constants_path = simulator_root / "src" / "packing_core" / "constants.py"
    masks_bytes = masks_path.read_bytes()
    masks_sha256 = hashlib.sha256(masks_bytes).hexdigest()
    masks_blob_sha1 = git_blob_sha1(masks_bytes)
    constants_sha256 = hashlib.sha256(constants_path.read_bytes()).hexdigest()
    input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()

    try:
        input_path_display = str(input_path.relative_to(repo_root))
    except ValueError:
        input_path_display = str(input_path)

    first_state = next(iter(state_infos.values()))
    all_task_ids = sorted({si.task_id for si in state_infos.values()})
    all_container_idxs = sorted({rec.container_idx for rec in records})
    task_id_state_counts = Counter(si.task_id for si in state_infos.values())
    task_id_record_counts: Counter = Counter()
    for rec in records:
        task_id_record_counts[state_infos[rec.state_ref].task_id] += 1

    report_text = build_report(
        git_head=git_head,
        script_sha256=script_sha256,
        masks_blob_sha1=masks_blob_sha1,
        masks_sha256=masks_sha256,
        constants_sha256=constants_sha256,
        input_path_display=input_path_display,
        input_sha256=input_sha256,
        state_set_hash=state_set_hash,
        gen_dir_name=gen_dir_name,
        schema_version=first_state.schema_version,
        source_config=first_state.source_config,
        source_config_sha256=first_state.source_config_sha256,
        pp=pp,
        all_task_ids=all_task_ids,
        all_container_idxs=all_container_idxs,
        task_id_state_counts=task_id_state_counts,
        task_id_record_counts=task_id_record_counts,
        stats=stats,
    )

    atomic_write(report_path, report_text)
    logger.info("レポートを書き込みました: %s", report_path)

    project_fn_count = len(stats.project_fn_records)
    if project_fn_count > 0:
        logger.error("project_fn=%d件検出（DoD未達）。exit code 1。", project_fn_count)
        return 1
    logger.info("project_fn=0件（DoD達成）。exit code 0。")
    return 0


def main() -> int:
    """CLIエントリポイント。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    try:
        return run(args)
    except ParityInputError as exc:
        logger.error("入力/整合性エラー: %s", exc)
        return 2
    except Exception:
        logger.exception("予期しない例外が発生しました")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
