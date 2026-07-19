"""統合T-024: 循環import回避契約のテスト（詳細仕様書 v1.18 §4.11「循環import回避契約」、
契約計画 §3.M）。

`candidates.py`/`watchdog.py`（4層追加分）は未実装のため、対象importは各テスト関数内・
subprocessの中で行う。4家族の分類: I=3（IMPORT-001,002,003）、III=1（IMPORT-004、現行
watchdog.pyはcandidatesを一切importしないためimport順序に関わらず既にGREEN、実測確認済み）。

契約（§4.11）: `watchdog.py`（layer1_main〜layer4_max_pの型注釈でCandidatePools/CandidateKey
を参照する）は `typing.TYPE_CHECKING` ブロック内でのみ candidates.py を参照する（実行時import
はしない）。一方 `candidates.py`（enumerate_candidates/filter_candidatesがStepBudgetを参照する）
は `from src.packing_core.watchdog import StepBudget` を通常の実行時importとして行ってよい。
実行時依存は candidates.py → watchdog.py の一方向のみ。

source文字列（`TYPE_CHECKING`等）は検証しない。`sys.modules`を用いた新規プロセスでの
実行時import有無の検証のみを正とする。
"""
import subprocess
import sys
from pathlib import Path

import pytest

SIMULATOR_ROOT = Path(__file__).resolve().parent.parent


def _run_snippet(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(SIMULATOR_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )


# --- IMPORT-001: 両順序importが成功する（param×2） -------------------------------------------


@pytest.mark.parametrize(
    "first,second",
    [
        pytest.param("candidates", "watchdog", id="T024-IMPORT-001-CAND-FIRST"),
        pytest.param("watchdog", "candidates", id="T024-IMPORT-001-WATCH-FIRST"),
    ],
)
def test_import_001_both_orders_succeed(first, second):
    code = f"import src.packing_core.{first}; import src.packing_core.{second}"
    result = _run_snippet(code)
    assert result.returncode == 0, result.stderr


# --- IMPORT-002: candidates.pyはStepBudgetを実行時importする -----------------------------------


def test_import_002_candidates_imports_watchdog_at_runtime():
    """candidates.py単独importで、watchdog.pyがsys.modulesへ実際に読み込まれる
    （StepBudgetの実行時import、§4.11「候補集合の共有」で使用する契約）。"""
    code = (
        "import src.packing_core.candidates, sys; "
        "sys.exit(0 if 'src.packing_core.watchdog' in sys.modules else 1)"
    )
    result = _run_snippet(code)
    assert result.returncode == 0, (
        f"candidates.py単独importでwatchdog.pyがsys.modulesに現れない "
        f"(stdout={result.stdout!r}, stderr={result.stderr!r})"
    )


# --- IMPORT-003: CandidatePools/CandidateKeyはcandidates.py所有、types.pyには無い ----------------


def test_import_003_candidate_pools_and_key_owned_by_candidates_not_types():
    from src.packing_core.candidates import CandidateKey, CandidatePools
    from src.packing_core import types as types_module

    assert CandidatePools is not None
    assert CandidateKey is not None
    assert not hasattr(types_module, "CandidatePools")
    assert not hasattr(types_module, "CandidateKey")


# --- IMPORT-004（訂正）: watchdog単独import後、candidatesがsys.modulesに無い（III） --------------


def test_import_004_watchdog_alone_does_not_pull_in_candidates_at_runtime():
    code = (
        "import src.packing_core.watchdog, sys; "
        "sys.exit(0 if 'src.packing_core.candidates' not in sys.modules else 1)"
    )
    result = _run_snippet(code)
    assert result.returncode == 0, (
        f"watchdog.py単独importでcandidates.pyがsys.modulesに現れてしまっている（実行時循環import混入）"
        f" (stdout={result.stdout!r}, stderr={result.stderr!r})"
    )
