"""T-029 ``scripts.package_submit`` public contract tests (specification v1.26 §5.8).

The production module is intentionally absent while this Phase B test commit is prepared.
Production imports are delayed until each test executes, so collection succeeds while execution
is RED with an uncaught ``ModuleNotFoundError`` until Phase C implements T-029.

The tests observe only ``main(argv) -> int``, the real ``python -m`` boundary, source-tree side
effects, and the generated archive.  They do not constrain private helper names or inspect the
production module's source text.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import os
import shutil
import stat
import subprocess
import sys
import textwrap
import typing
import warnings
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pytest


SIMULATOR_ROOT = Path(__file__).resolve().parent.parent
REAL_AGENT_DIR = SIMULATOR_ROOT / "agents" / "heuristic"
REAL_CORE_DIR = SIMULATOR_ROOT / "src" / "packing_core"

CORE_FILES = (
    "__init__.py",
    "candidates.py",
    "constants.py",
    "container_space.py",
    "ems.py",
    "geometry.py",
    "masks.py",
    "risk.py",
    "score.py",
    "stability.py",
    "state.py",
    "types.py",
    "watchdog.py",
)

EXPECTED_ENTRIES = (
    "heuristic/",
    "heuristic/agent.py",
    "heuristic/telemetry.py",
    "heuristic/packing_core/",
    "heuristic/packing_core/__init__.py",
    "heuristic/packing_core/candidates.py",
    "heuristic/packing_core/constants.py",
    "heuristic/packing_core/container_space.py",
    "heuristic/packing_core/ems.py",
    "heuristic/packing_core/geometry.py",
    "heuristic/packing_core/masks.py",
    "heuristic/packing_core/risk.py",
    "heuristic/packing_core/score.py",
    "heuristic/packing_core/stability.py",
    "heuristic/packing_core/state.py",
    "heuristic/packing_core/types.py",
    "heuristic/packing_core/watchdog.py",
)

OLD_PREFIXES = (b"agents.heuristic", b"src.packing_core")
SENTINEL = b"existing-output-must-survive"


def _write_source(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = textwrap.dedent(source).lstrip("\n")
    path.write_bytes(normalized.encode("utf-8"))


def _synthetic_agent_source() -> str:
    """A self-checking Agent that validates the documented isolated subprocess boundary."""
    modules = ["heuristic.telemetry", "heuristic.packing_core"]
    modules.extend(f"heuristic.packing_core.{Path(name).stem}" for name in CORE_FILES[1:])
    rendered_modules = repr(tuple(modules))
    creator_pid = os.getpid()
    repository_root = repr(str(SIMULATOR_ROOT.parent.resolve()))
    simulator_root = repr(str(SIMULATOR_ROOT.resolve()))
    return f'''\
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from agents.heuristic.telemetry import format_row
from src.packing_core import geometry
from src.packing_core.types import TOKEN

if TYPE_CHECKING:
    from src.packing_core.state import PackingState

CREATOR_PID = {creator_pid}
REPOSITORY_ROOT = Path({repository_root})
SIMULATOR_ROOT = Path({simulator_root})


class Agent:
    def __init__(self, module_path):
        if module_path != "heuristic/":
            raise RuntimeError("wrong module path")
        if os.getpid() == CREATOR_PID:
            raise RuntimeError("inspection did not use a subprocess")
        root = Path(__file__).resolve().parent.parent
        if Path.cwd().resolve() != root:
            raise RuntimeError("inspection cwd is not extraction root")
        if root == REPOSITORY_ROOT or REPOSITORY_ROOT in root.parents:
            raise RuntimeError("inspection ran inside the development tree")
        for entry in sys.path:
            if entry and Path(entry).resolve() in (REPOSITORY_ROOT, SIMULATOR_ROOT):
                raise RuntimeError("development tree is present on sys.path")
        if os.environ.get("PYTHONPATH") != "":
            raise RuntimeError("PYTHONPATH is not empty")
        if os.environ.get("PYTHONNOUSERSITE") != "1":
            raise RuntimeError("PYTHONNOUSERSITE is not enabled")
        if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
            raise RuntimeError("PYTHONDONTWRITEBYTECODE is not enabled")
        if "TELEMETRY_DIR" in os.environ or "TELEMETRY_RUN_ID" in os.environ:
            raise RuntimeError("telemetry environment leaked")
        old_prefixes = ("agents" + ".heuristic", "src" + ".packing_core")
        if any(
            name == prefix or name.startswith(prefix + ".")
            for name in sys.modules
            for prefix in old_prefixes
        ):
            raise RuntimeError("development prefix loaded")
        expected = {rendered_modules}
        missing = set(expected).difference(sys.modules)
        if missing:
            raise RuntimeError(f"modules not imported: {{sorted(missing)}}")
        payload_root = root / "heuristic"
        for name in expected:
            origin = Path(sys.modules[name].__file__).resolve()
            if origin != payload_root and payload_root not in origin.parents:
                raise RuntimeError(f"module outside payload: {{name}}={{origin}}")
        if TOKEN != "ready" or geometry.TOKEN != "geometry" or format_row() != {{}}:
            raise RuntimeError("packaged dependencies are wrong")
'''


def _materialize_synthetic(agent_dir: Path, core_dir: Path) -> None:
    _write_source(agent_dir / "agent.py", _synthetic_agent_source())
    _write_source(
        agent_dir / "telemetry.py",
        """
        def format_row():
            return {}
        """,
    )
    _write_source(core_dir / "__init__.py", "from src.packing_core.types import TOKEN\n")
    _write_source(core_dir / "types.py", 'TOKEN = "ready"\n')
    _write_source(core_dir / "geometry.py", 'TOKEN = "geometry"\n')
    _write_source(core_dir / "ems.py", "from src.packing_core import geometry\n")
    _write_source(
        core_dir / "stability.py",
        """
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            from src.packing_core.state import PackingState

        VALUE = "stability"
        """,
    )
    for name in CORE_FILES:
        path = core_dir / name
        if not path.exists():
            _write_source(path, f'VALUE = "{path.stem}"\n')


def _copy_real_sources(agent_dir: Path, core_dir: Path) -> None:
    agent_dir.mkdir(parents=True)
    core_dir.mkdir(parents=True)
    for name in ("agent.py", "telemetry.py"):
        shutil.copyfile(REAL_AGENT_DIR / name, agent_dir / name)
    for name in CORE_FILES:
        shutil.copyfile(REAL_CORE_DIR / name, core_dir / name)


def _tree_snapshot(*roots: Path) -> tuple[tuple[object, ...], ...]:
    """Capture lstat identity, times, types, modes, links, and bytes without following."""
    rows: list[tuple[object, ...]] = []
    for root in roots:
        label = root.name
        pending = [root]
        while pending:
            current = pending.pop()
            relative = "." if current == root else current.relative_to(root).as_posix()
            info = current.lstat()
            kind = stat.S_IFMT(info.st_mode)
            mode = stat.S_IMODE(info.st_mode)
            identity = (info.st_ino, info.st_mtime_ns)
            if stat.S_ISDIR(info.st_mode):
                rows.append((label, relative, kind, mode, *identity, None))
                pending.extend(sorted(current.iterdir(), reverse=True))
            elif stat.S_ISREG(info.st_mode):
                rows.append((label, relative, kind, mode, *identity, current.read_bytes()))
            elif stat.S_ISLNK(info.st_mode):
                rows.append((label, relative, kind, mode, *identity, os.readlink(current)))
            else:
                rows.append((label, relative, kind, mode, *identity, None))
    return tuple(sorted(rows, key=lambda row: (str(row[0]), str(row[1]))))


@dataclass
class PackageEnvironment:
    root: Path
    agent_dir: Path
    core_dir: Path
    output: Path

    @classmethod
    def create(cls, root: Path, *, real_sources: bool = False) -> "PackageEnvironment":
        agent_dir = root / "inputs" / "heuristic"
        core_dir = root / "inputs" / "packing_core"
        if real_sources:
            _copy_real_sources(agent_dir, core_dir)
        else:
            _materialize_synthetic(agent_dir, core_dir)
        return cls(root, agent_dir, core_dir, root / "publish" / "heuristic.zip")

    def argv(self, *, force: bool = False, output: Path | None = None) -> list[str]:
        values = [
            "--agent-dir",
            str(self.agent_dir),
            "--core-dir",
            str(self.core_dir),
            "--output",
            str(self.output if output is None else output),
        ]
        if force:
            values.append("--force")
        return values

    def snapshot(self) -> tuple[tuple[object, ...], ...]:
        return _tree_snapshot(self.agent_dir, self.core_dir)


def _environment(tmp_path: Path, name: str = "case", *, real_sources: bool = False):
    return PackageEnvironment.create(tmp_path / name, real_sources=real_sources)


def _import_package_submit():
    """Per-test delayed import; ModuleNotFoundError deliberately remains uncaught."""
    importlib.invalidate_caches()
    sys.modules.pop("scripts.package_submit", None)
    return importlib.import_module("scripts.package_submit")


def _run_main(package_submit, argv: list[str] | None) -> int:
    result = package_submit.main(argv)
    assert type(result) is int
    return result


def _assert_absent(path: Path) -> None:
    assert not os.path.lexists(path)


def _zip_bytes(path: Path, name: str) -> bytes:
    with zipfile.ZipFile(path) as archive:
        return archive.read(name)


def _append_source(path: Path, text: str) -> None:
    path.write_bytes(path.read_bytes() + text.encode("utf-8"))


def _expected_staged_source(source: bytes, archive_name: str) -> bytes:
    """Apply only the documented module-span substitutions to a known synthetic source."""
    text = source.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    if archive_name == "heuristic/agent.py":
        text = text.replace(
            "from agents.heuristic.telemetry import ", "from .telemetry import "
        )
        text = text.replace("from src.packing_core import ", "from .packing_core import ")
        text = text.replace("from src.packing_core.", "from .packing_core.")
    elif archive_name.startswith("heuristic/packing_core/"):
        text = text.replace("from src.packing_core import ", "from . import ")
        text = text.replace("from src.packing_core.", "from .")
    return text.encode("utf-8")


def _run_cli(
    argv: list[str],
    *,
    timeout: float = 120,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the real module boundary with an operational hang guard, not a runtime SLA."""
    return subprocess.run(
        [sys.executable, "-m", "scripts.package_submit", *argv],
        cwd=SIMULATOR_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )


def _interrupt_publish_environment(
    root: Path, output: Path
) -> tuple[dict[str, str], Path]:
    injector = root / "interrupt-injector"
    injector.mkdir()
    _write_source(
        injector / "sitecustomize.py",
        """
        import os
        from pathlib import Path

        _real_replace = os.replace
        _target = os.path.abspath(
            os.fsdecode(os.fspath(os.environ["T029_INTERRUPT_OUTPUT"]))
        )
        _trigger = Path(os.environ["T029_INTERRUPT_TRIGGER"])

        def _absolute_destination(destination, dst_dir_fd):
            rendered = os.fsdecode(os.fspath(destination))
            if os.path.isabs(rendered):
                return os.path.abspath(rendered)
            if dst_dir_fd is not None:
                directory = os.readlink(f"/proc/self/fd/{dst_dir_fd}")
                return os.path.abspath(os.path.join(directory, rendered))
            return os.path.abspath(rendered)

        def _interrupt_publish(source, destination, *args, **kwargs):
            if _absolute_destination(destination, kwargs.get("dst_dir_fd")) == _target:
                _trigger.write_text("os.replace")
                raise KeyboardInterrupt
            return _real_replace(source, destination, *args, **kwargs)

        os.replace = _interrupt_publish
        """,
    )
    trigger = root / "interrupt-trigger.txt"
    process_env = os.environ.copy()
    process_env["PYTHONPATH"] = str(injector)
    process_env["T029_INTERRUPT_OUTPUT"] = str(output)
    process_env["T029_INTERRUPT_TRIGGER"] = str(trigger)
    return process_env, trigger


def _assert_output_absent_and_no_temp(path: Path) -> None:
    _assert_absent(path)
    if path.parent.exists():
        assert list(path.parent.iterdir()) == []


# =============================================================================
# API/CLI — public main and the real module entry point
# =============================================================================


def test_api_main_has_exact_public_signature():
    package_submit = _import_package_submit()
    signature = inspect.signature(package_submit.main)
    assert tuple(signature.parameters) == ("argv",)
    argv = signature.parameters["argv"]
    assert argv.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert argv.default is None
    hints = typing.get_type_hints(package_submit.main)
    assert set(hints) == {"argv", "return"}
    assert hints["argv"] == (list[str] | None)
    assert hints["return"] is int


def test_api_absolute_and_relative_paths_and_main_none_return_strict_zero(
    tmp_path, monkeypatch
):
    package_submit = _import_package_submit()

    absolute = _environment(tmp_path, "absolute")
    before = absolute.snapshot()
    assert _run_main(package_submit, absolute.argv()) == 0
    assert absolute.output.is_file()
    assert absolute.snapshot() == before

    relative = _environment(tmp_path, "relative")
    monkeypatch.chdir(relative.root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "package_submit",
            "--agent-dir",
            "inputs/heuristic",
            "--core-dir",
            "inputs/packing_core",
            "--output",
            "publish/heuristic.zip",
        ],
    )
    assert _run_main(package_submit, None) == 0
    assert relative.output.is_file()


@pytest.mark.parametrize(
    "missing",
    ["--agent-dir", "--core-dir", "--output"],
    ids=["agent-dir", "core-dir", "output"],
)
def test_api_required_argument_errors_are_normalized_to_exit_2(tmp_path, missing):
    package_submit = _import_package_submit()
    env = _environment(tmp_path)
    argv = env.argv()
    index = argv.index(missing)
    del argv[index : index + 2]
    assert _run_main(package_submit, argv) == 2
    _assert_output_absent_and_no_temp(env.output)


@pytest.mark.parametrize("case", ["help", "unknown"], ids=["help", "unknown"])
def test_api_all_argparse_system_exit_paths_are_normalized_to_exit_2(tmp_path, case):
    package_submit = _import_package_submit()
    env = _environment(tmp_path)
    argv = ["--help"] if case == "help" else [*env.argv(), "--unknown-option"]
    assert _run_main(package_submit, argv) == 2
    _assert_output_absent_and_no_temp(env.output)


@pytest.mark.parametrize(
    "case",
    [
        "agent-basename",
        "core-basename",
        "output-suffix",
        "inside-agent",
        "inside-core",
        "parent-is-file",
        "parent-is-symlink",
    ],
)
def test_api_path_constraints_fail_closed_with_exit_2(tmp_path, case):
    package_submit = _import_package_submit()
    env = _environment(tmp_path)
    output = env.output
    if case == "agent-basename":
        renamed = env.agent_dir.with_name("renamed-agent")
        env.agent_dir.rename(renamed)
        env.agent_dir = renamed
    elif case == "core-basename":
        renamed = env.core_dir.with_name("renamed-core")
        env.core_dir.rename(renamed)
        env.core_dir = renamed
    elif case == "output-suffix":
        output = env.root / "publish" / "heuristic.tar"
    elif case == "inside-agent":
        output = env.agent_dir / "payload.zip"
    elif case == "inside-core":
        output = env.core_dir / "payload.zip"
    elif case == "parent-is-file":
        parent = env.root / "blocked-parent"
        parent.write_text("not a directory")
        output = parent / "payload.zip"
    elif case == "parent-is-symlink":
        target = env.root / "real-parent"
        target.mkdir()
        parent = env.root / "linked-parent"
        parent.symlink_to(target, target_is_directory=True)
        output = parent / "payload.zip"
    assert _run_main(package_submit, env.argv(output=output)) == 2
    _assert_absent(output)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("success", 0), ("content", 1), ("arguments", 2)],
    ids=["exit-0", "exit-1", "exit-2"],
)
def test_cli_python_m_reports_documented_process_exit_codes(tmp_path, mode, expected):
    _import_package_submit()  # Keep Phase B RED on the one intended missing-module cause.
    env = _environment(tmp_path)
    argv = env.argv()
    if mode == "content":
        _append_source(env.agent_dir / "telemetry.py", '\nMARKER = "src.packing_core"\n')
    elif mode == "arguments":
        index = argv.index("--output")
        del argv[index : index + 2]
    completed = _run_cli(argv)
    assert completed.returncode == expected, completed.stderr
    if expected == 0:
        assert env.output.is_file()
    else:
        _assert_output_absent_and_no_temp(env.output)


# =============================================================================
# INPUT/WHITELIST — fixed source set, junk policy, and lstat safety
# =============================================================================


@pytest.mark.parametrize(
    "case",
    [
        "agent-directory",
        "core-directory",
        "agent-not-directory",
        "core-not-directory",
        "agent-file",
        "telemetry-file",
        "core-file",
    ],
)
def test_input_missing_directory_or_required_file_is_exit_2(tmp_path, case):
    package_submit = _import_package_submit()
    env = _environment(tmp_path)
    if case == "agent-directory":
        shutil.rmtree(env.agent_dir)
    elif case == "core-directory":
        shutil.rmtree(env.core_dir)
    elif case == "agent-not-directory":
        shutil.rmtree(env.agent_dir)
        env.agent_dir.write_text("not a directory")
    elif case == "core-not-directory":
        shutil.rmtree(env.core_dir)
        env.core_dir.write_text("not a directory")
    elif case == "agent-file":
        (env.agent_dir / "agent.py").unlink()
    elif case == "telemetry-file":
        (env.agent_dir / "telemetry.py").unlink()
    else:
        (env.core_dir / "watchdog.py").unlink()
    assert _run_main(package_submit, env.argv()) == 2
    _assert_output_absent_and_no_temp(env.output)


def test_whitelist_ignores_only_pycache_pyc_and_pyo(tmp_path):
    package_submit = _import_package_submit()
    env = _environment(tmp_path)
    cache = env.agent_dir / "__pycache__"
    cache.mkdir()
    (cache / "agent.cpython-311.pyc").write_bytes(b"junk")
    (env.agent_dir / "loose.pyc").write_bytes(b"junk")
    (env.core_dir / "loose.pyo").write_bytes(b"junk")
    before = env.snapshot()
    assert _run_main(package_submit, env.argv()) == 0
    assert env.snapshot() == before
    with zipfile.ZipFile(env.output) as archive:
        assert tuple(info.filename for info in archive.infolist()) == EXPECTED_ENTRIES


@pytest.mark.parametrize("case", ["unknown-file", "unknown-directory", "pyd"])
def test_whitelist_unknown_regular_content_including_pyd_is_exit_1(tmp_path, case):
    package_submit = _import_package_submit()
    env = _environment(tmp_path)
    if case == "unknown-file":
        (env.agent_dir / "notes.txt").write_text("unknown")
    elif case == "unknown-directory":
        (env.core_dir / "extra").mkdir()
    else:
        (env.agent_dir / "native.pyd").write_bytes(b"unknown")
    before = env.snapshot()
    assert _run_main(package_submit, env.argv()) == 1
    assert env.snapshot() == before
    _assert_output_absent_and_no_temp(env.output)


@pytest.mark.parametrize(
    "case",
    [
        "required-file-symlink",
        "unknown-symlink",
        "source-directory-symlink",
        "pycache-symlink",
        "pyc-symlink",
    ],
)
def test_lstat_rejects_symlinks_and_nonregular_files_with_exit_2(tmp_path, case):
    package_submit = _import_package_submit()
    env = _environment(tmp_path)
    protected_roots: list[Path] = []
    if case == "required-file-symlink":
        required = env.agent_dir / "telemetry.py"
        target = env.root / "telemetry-target.py"
        target.write_bytes(required.read_bytes())
        required.unlink()
        required.symlink_to(target)
        protected_roots.append(target)
    elif case == "unknown-symlink":
        target = env.root / "outside.txt"
        target.write_text("outside")
        (env.core_dir / "linked.txt").symlink_to(target)
        protected_roots.append(target)
    elif case == "source-directory-symlink":
        real_core = env.core_dir.with_name("real-packing-core")
        env.core_dir.rename(real_core)
        env.core_dir.symlink_to(real_core, target_is_directory=True)
        protected_roots.append(real_core)
    elif case == "pycache-symlink":
        target = env.root / "outside-cache"
        target.mkdir()
        (env.agent_dir / "__pycache__").symlink_to(target, target_is_directory=True)
        protected_roots.append(target)
    else:
        target = env.root / "outside.pyc"
        target.write_bytes(b"outside")
        (env.agent_dir / "ignored.pyc").symlink_to(target)
        protected_roots.append(target)
    snapshot_roots = (env.agent_dir, env.core_dir, *protected_roots)
    before = _tree_snapshot(*snapshot_roots)
    assert _run_main(package_submit, env.argv()) == 2
    assert _tree_snapshot(*snapshot_roots) == before
    _assert_output_absent_and_no_temp(env.output)


@pytest.mark.parametrize("name", ["channel", "ignored.pyo"], ids=["unknown", "junk-name"])
def test_lstat_source_fifo_fails_without_hanging_the_test_process(tmp_path, name):
    _import_package_submit()
    env = _environment(tmp_path)
    os.mkfifo(env.agent_dir / name)
    before = env.snapshot()
    completed = _run_cli(env.argv(), timeout=10)
    assert completed.returncode == 2, completed.stderr
    assert env.snapshot() == before
    _assert_output_absent_and_no_temp(env.output)


@pytest.mark.parametrize(
    ("directory_kind", "node_kind"),
    [
        ("pycache", "symlink"),
        ("pycache", "fifo"),
        ("unknown-directory", "symlink"),
        ("unknown-directory", "fifo"),
    ],
)
def test_lstat_scans_ignored_and_unknown_subtrees_before_exit_classification(
    tmp_path, directory_kind, node_kind
):
    _import_package_submit()
    env = _environment(tmp_path, f"{directory_kind}-{node_kind}")
    if directory_kind == "pycache":
        nested = env.agent_dir / "__pycache__" / "nested"
    else:
        nested = env.core_dir / "extra" / "nested"
    nested.mkdir(parents=True)

    protected: list[Path] = []
    if node_kind == "symlink":
        target = env.root / f"{directory_kind}-outside.txt"
        target.write_bytes(SENTINEL)
        (nested / "ignored.pyc").symlink_to(target)
        protected.append(target)
    else:
        os.mkfifo(nested / "ignored.pyo")

    roots = (env.agent_dir, env.core_dir, *protected)
    before = _tree_snapshot(*roots)
    completed = _run_cli(env.argv(), timeout=10)
    assert completed.returncode == 2, completed.stderr
    assert _tree_snapshot(*roots) == before
    _assert_output_absent_and_no_temp(env.output)


@pytest.mark.parametrize("case", ["invalid-utf8", "invalid-python"])
def test_readable_but_invalid_source_is_exit_1(tmp_path, case):
    package_submit = _import_package_submit()
    env = _environment(tmp_path)
    target = env.agent_dir / "telemetry.py"
    if case == "invalid-utf8":
        target.write_bytes(b"value = '\xff'\n")
    else:
        target.write_text("def broken(:\n")
    before = env.snapshot()
    assert _run_main(package_submit, env.argv()) == 1
    assert env.snapshot() == before
    _assert_output_absent_and_no_temp(env.output)


# =============================================================================
# TRANSFORM — staging-only import rewrite and fail-closed residual checks
# =============================================================================


def test_transform_rewrites_only_module_spans_preserves_indent_and_source_bytes(tmp_path):
    package_submit = _import_package_submit()
    env = _environment(tmp_path)
    sources = (
        env.agent_dir / "agent.py",
        env.agent_dir / "telemetry.py",
        *[env.core_dir / name for name in CORE_FILES],
    )
    for source in sources:
        source.write_bytes(source.read_bytes().replace(b"\n", b"\r\n"))
    before = env.snapshot()

    assert _run_main(package_submit, env.argv()) == 0
    assert env.snapshot() == before

    agent = _zip_bytes(env.output, "heuristic/agent.py").decode("utf-8")
    init = _zip_bytes(env.output, "heuristic/packing_core/__init__.py").decode("utf-8")
    ems = _zip_bytes(env.output, "heuristic/packing_core/ems.py").decode("utf-8")
    stability = _zip_bytes(env.output, "heuristic/packing_core/stability.py").decode("utf-8")
    assert "from .telemetry import format_row" in agent
    assert "from .packing_core import geometry" in agent
    assert "from .packing_core.types import TOKEN" in agent
    assert "    from .packing_core.state import PackingState" in agent
    assert "from .types import TOKEN" in init
    assert "from . import geometry" in ems
    assert "    from .state import PackingState" in stability
    source_by_archive_name = {
        "heuristic/agent.py": env.agent_dir / "agent.py",
        "heuristic/telemetry.py": env.agent_dir / "telemetry.py",
        **{
            f"heuristic/packing_core/{name}": env.core_dir / name
            for name in CORE_FILES
        },
    }
    for name, source_path in source_by_archive_name.items():
        payload = _zip_bytes(env.output, name)
        assert payload == _expected_staged_source(source_path.read_bytes(), name)
        assert b"\r" not in payload
        ast.parse(payload.decode("utf-8"), filename=name)
        assert all(prefix not in payload for prefix in OLD_PREFIXES)


@pytest.mark.parametrize(
    ("case", "addition"),
    [
        ("absolute-import", "\nimport src.packing_core.types\n"),
        ("dynamic-import", '\nimport importlib\nVALUE = importlib.import_module("src.packing_core.types")\n'),
        ("string", '\nVALUE = "agents.heuristic"\n'),
        ("comment", "\n# src.packing_core must not remain\n"),
    ],
    ids=["absolute-import", "dynamic-import", "string", "comment"],
)
def test_transform_rejects_old_prefix_references_in_every_source_location(
    tmp_path, case, addition
):
    package_submit = _import_package_submit()
    env = _environment(tmp_path, case)
    _append_source(env.agent_dir / "telemetry.py", addition)
    before = env.snapshot()
    assert _run_main(package_submit, env.argv()) == 1
    assert env.snapshot() == before
    _assert_output_absent_and_no_temp(env.output)


@pytest.mark.parametrize(
    ("location", "addition"),
    [
        ("agent-comment", "\n# from src.packing_core import geometry\n"),
        ("agent-string", '\nTEXT = "from src.packing_core.types import TOKEN"\n'),
        ("core-comment", "\n# from src.packing_core.types import TOKEN\n"),
        ("core-string", '\nTEXT = "from src.packing_core import geometry"\n'),
        ("telemetry-import", "\nfrom src.packing_core import geometry\n"),
    ],
    ids=[
        "agent-comment",
        "agent-string",
        "core-comment",
        "core-string",
        "telemetry-import",
    ],
)
def test_transform_only_rewrites_allowed_from_statement_module_spans(
    tmp_path, location, addition
):
    package_submit = _import_package_submit()
    env = _environment(tmp_path, location)
    if location.startswith("agent-"):
        target = env.agent_dir / "agent.py"
    elif location.startswith("core-"):
        target = env.core_dir / "types.py"
    else:
        target = env.agent_dir / "telemetry.py"
    _append_source(target, addition)
    before = env.snapshot()
    assert _run_main(package_submit, env.argv()) == 1
    assert env.snapshot() == before
    _assert_output_absent_and_no_temp(env.output)


def test_transform_preserves_unrelated_dynamic_import_without_rejecting_it(tmp_path):
    package_submit = _import_package_submit()
    env = _environment(tmp_path)
    addition = '\nimport importlib\nDYNAMIC_VALUE = importlib.import_module("math").sqrt(4)\n'
    _append_source(env.agent_dir / "telemetry.py", addition)
    before = env.snapshot()
    assert _run_main(package_submit, env.argv()) == 0
    assert env.snapshot() == before
    packaged = _zip_bytes(env.output, "heuristic/telemetry.py")
    assert packaged == _expected_staged_source(
        env.agent_dir.joinpath("telemetry.py").read_bytes(), "heuristic/telemetry.py"
    )
    assert addition.strip() in packaged.decode("utf-8")


@pytest.mark.parametrize("case", ["sys-path", "sys-modules-alias", "dual-import"])
def test_transform_rejects_forbidden_payload_import_mechanisms(tmp_path, case):
    package_submit = _import_package_submit()
    env = _environment(tmp_path)
    agent = env.agent_dir / "agent.py"
    if case == "sys-path":
        _append_source(agent, '\nimport sys\nsys.path.append("vendor")\n')
    elif case == "sys-modules-alias":
        _append_source(
            agent,
            '\nimport sys\nsys.modules["legacy.alias"] = sys.modules[__name__]\n',
        )
    else:
        _append_source(
            agent,
            """

try:
    from src.packing_core.types import TOKEN as DUAL_TOKEN
except ImportError:
    from .packing_core.types import TOKEN as DUAL_TOKEN
""",
        )
    before = env.snapshot()
    assert _run_main(package_submit, env.argv()) == 1
    assert env.snapshot() == before
    _assert_output_absent_and_no_temp(env.output)


# =============================================================================
# ZIP — exact payload, metadata, integrity, and reproducibility
# =============================================================================


def test_zip_has_exact_order_safe_paths_metadata_crc_and_utf8_lf(tmp_path):
    package_submit = _import_package_submit()
    env = _environment(tmp_path)
    assert _run_main(package_submit, env.argv()) == 0

    with zipfile.ZipFile(env.output) as archive:
        infos = archive.infolist()
        names = tuple(info.filename for info in infos)
        assert names == EXPECTED_ENTRIES
        assert len(names) == len(set(names)) == 17
        assert archive.comment == b""
        assert archive.testzip() is None
        for info in infos:
            name = info.filename
            assert not name.startswith("/")
            assert "\\" not in name
            assert ".." not in PurePosixPath(name).parts
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.create_system == 3
            assert info.extra == b""
            assert info.compress_type == zipfile.ZIP_STORED
            expected_mode = 0o040755 if name.endswith("/") else 0o100644
            assert info.external_attr >> 16 == expected_mode
            payload = archive.read(info)
            assert info.CRC == (zlib.crc32(payload) & 0xFFFFFFFF)
            assert info.compress_size == info.file_size == len(payload)
            if name.endswith(".py"):
                source = payload.decode("utf-8")
                assert "\r" not in source
                ast.parse(source, filename=name)


@pytest.mark.parametrize(
    "bad_name",
    [
        "/absolute.py",
        "heuristic/../escape.py",
        r"heuristic\escape.py",
        "heuristic/agent.py",
    ],
    ids=["absolute", "parent", "backslash", "duplicate"],
)
def test_zip_unsafe_or_duplicate_entry_is_rejected_before_publish(
    tmp_path, monkeypatch, bad_name
):
    env = _environment(tmp_path)
    env.output.parent.mkdir(parents=True)
    env.output.write_bytes(SENTINEL)
    output_before = env.output.stat()
    source_before = env.snapshot()
    output_parent = env.output.parent.resolve()
    real_init = zipfile.ZipFile.__init__
    real_close = zipfile.ZipFile.close
    real_replace = os.replace
    injected: list[Path] = []
    publish_calls: list[tuple[Path, Path]] = []

    def candidate_path(value):
        try:
            rendered = os.fsdecode(os.fspath(value))
        except TypeError:
            name = getattr(value, "name", None)
            if name is None:
                return None
            try:
                rendered = os.fsdecode(os.fspath(name))
            except TypeError:
                return None
        return Path(os.path.abspath(rendered))

    def invalid_info():
        info = zipfile.ZipInfo(bad_name, date_time=(1980, 1, 1, 0, 0, 0))
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        info.compress_type = zipfile.ZIP_STORED
        info.extra = b""
        return info

    def append_invalid_entry(candidate):
        writer = object.__new__(zipfile.ZipFile)
        real_init(writer, candidate, mode="a", compression=zipfile.ZIP_STORED)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                writer.writestr(invalid_info(), b"injected-invalid-entry")
        finally:
            real_close(writer)
        injected.append(candidate)

    def injecting_close(archive):
        filename = getattr(archive, "filename", None)
        candidate = candidate_path(filename) if filename is not None else None
        if (
            archive.fp is not None
            and archive.mode in {"w", "x", "a"}
            and candidate is not None
            and candidate.parent == output_parent
            and not getattr(archive, "_t029_bad_entry_injected", False)
        ):
            archive._t029_bad_entry_injected = True
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr(invalid_info(), b"injected-invalid-entry")
            injected.append(candidate)
        return real_close(archive)

    def injecting_init(archive, file, mode="r", *args, **kwargs):
        candidate = candidate_path(file)
        if (
            mode == "r"
            and candidate is not None
            and candidate != env.output
            and candidate.parent == output_parent
            and candidate not in injected
        ):
            append_invalid_entry(candidate)
        return real_init(archive, file, mode, *args, **kwargs)

    def observed_replace(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path == env.output:
            publish_calls.append((source_path, destination_path))
        return real_replace(source, destination)

    monkeypatch.setattr(zipfile.ZipFile, "__init__", injecting_init)
    monkeypatch.setattr(zipfile.ZipFile, "close", injecting_close)
    monkeypatch.setattr(os, "replace", observed_replace)
    package_submit = _import_package_submit()
    assert _run_main(package_submit, env.argv(force=True)) == 1
    assert injected
    assert publish_calls == []
    output_after = env.output.stat()
    assert env.output.read_bytes() == SENTINEL
    assert (output_after.st_ino, output_after.st_mtime_ns) == (
        output_before.st_ino,
        output_before.st_mtime_ns,
    )
    assert env.snapshot() == source_before
    assert list(env.output.parent.iterdir()) == [env.output]


def test_zip_generation_is_byte_for_byte_and_sha256_reproducible(tmp_path):
    package_submit = _import_package_submit()
    env = _environment(tmp_path)
    first = env.root / "first" / "heuristic.zip"
    second = env.root / "second" / "heuristic.zip"
    assert _run_main(package_submit, env.argv(output=first)) == 0
    assert _run_main(package_submit, env.argv(output=second)) == 0
    first_bytes = first.read_bytes()
    second_bytes = second.read_bytes()
    assert first_bytes == second_bytes
    assert hashlib.sha256(first_bytes).digest() == hashlib.sha256(second_bytes).digest()


# =============================================================================
# PUBLISH — output protection, atomic replace, cleanup, and interrupt mapping
# =============================================================================


def test_publish_existing_regular_output_without_force_is_exit_2_and_untouched(tmp_path):
    package_submit = _import_package_submit()
    env = _environment(tmp_path)
    env.output.parent.mkdir(parents=True)
    env.output.write_bytes(SENTINEL)
    before = env.output.stat()
    assert _run_main(package_submit, env.argv()) == 2
    after = env.output.stat()
    assert env.output.read_bytes() == SENTINEL
    assert after.st_ino == before.st_ino
    assert after.st_mtime_ns == before.st_mtime_ns
    assert list(env.output.parent.iterdir()) == [env.output]


def test_publish_force_uses_one_final_same_parent_replace_after_full_validation(
    tmp_path, monkeypatch
):
    env = _environment(tmp_path)
    env.output.parent.mkdir(parents=True)
    env.output.write_bytes(SENTINEL)
    warmup_marker = tmp_path / "warmup-complete.txt"
    _append_source(
        env.agent_dir / "agent.py",
        f'\n        Path({str(warmup_marker)!r}).write_text("complete")\n',
    )
    real_replace = os.replace
    calls: list[tuple[Path, Path]] = []

    def observed_replace(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        calls.append((source_path, destination_path))
        if destination_path == env.output:
            assert warmup_marker.read_text() == "complete"
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", observed_replace)
    package_submit = _import_package_submit()
    assert _run_main(package_submit, env.argv(force=True)) == 0
    assert env.output.read_bytes() != SENTINEL
    publish_calls = [call for call in calls if call[1] == env.output]
    assert len(publish_calls) == 1
    source, destination = publish_calls[0]
    assert destination == env.output
    assert source.parent == destination.parent
    assert not source.exists()
    assert list(env.output.parent.iterdir()) == [env.output]


@pytest.mark.parametrize(
    ("exception", "expected"),
    [
        (OSError("replace failed"), 2),
        (RuntimeError("unexpected internal failure"), 2),
    ],
    ids=["io-error", "internal-error"],
)
def test_publish_replace_failure_preserves_existing_output_and_removes_temp(
    tmp_path, monkeypatch, exception, expected
):
    env = _environment(tmp_path)
    env.output.parent.mkdir(parents=True)
    env.output.write_bytes(SENTINEL)
    before = env.output.stat()
    real_replace = os.replace

    def failing_replace(source, destination):
        if Path(destination) == env.output:
            raise exception
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", failing_replace)
    package_submit = _import_package_submit()
    assert _run_main(package_submit, env.argv(force=True)) == expected
    after = env.output.stat()
    assert env.output.read_bytes() == SENTINEL
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert list(env.output.parent.iterdir()) == [env.output]


def test_api_keyboard_interrupt_returns_strict_130_and_preserves_output(tmp_path):
    _import_package_submit()
    env = _environment(tmp_path)
    env.output.parent.mkdir(parents=True)
    env.output.write_bytes(SENTINEL)
    before = env.output.stat()
    source_before = env.snapshot()
    process_env, trigger = _interrupt_publish_environment(tmp_path, env.output)
    result_marker = tmp_path / "interrupt-result.txt"
    process_env["T029_INTERRUPT_RESULT"] = str(result_marker)
    probe = textwrap.dedent(
        """
        import os
        import sys
        from pathlib import Path

        import scripts.package_submit as module

        marker = Path(os.environ["T029_INTERRUPT_RESULT"])
        try:
            result = module.main(sys.argv[1:])
        except KeyboardInterrupt:
            marker.write_text("escaped-keyboard-interrupt")
            raise SystemExit(90)
        except SystemExit as exc:
            marker.write_text(f"escaped-system-exit:{exc.code!r}")
            raise SystemExit(91)
        marker.write_text(f"{type(result).__module__}.{type(result).__qualname__}:{result!r}")
        raise SystemExit(0)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe, *env.argv(force=True)],
        cwd=SIMULATOR_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=process_env,
    )
    assert completed.returncode == 0, completed.stderr
    assert result_marker.read_text() == "builtins.int:130"
    assert trigger.read_text() == "os.replace"
    after = env.output.stat()
    assert env.output.read_bytes() == SENTINEL
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert env.snapshot() == source_before
    assert list(env.output.parent.iterdir()) == [env.output]


def test_cli_python_m_maps_keyboard_interrupt_return_to_process_exit_130(tmp_path):
    _import_package_submit()
    env = _environment(tmp_path)
    env.output.parent.mkdir(parents=True)
    env.output.write_bytes(SENTINEL)
    before = env.output.stat()
    source_before = env.snapshot()
    process_env, trigger = _interrupt_publish_environment(tmp_path, env.output)
    completed = _run_cli(env.argv(force=True), timeout=30, env=process_env)
    assert completed.returncode == 130, completed.stderr
    assert trigger.read_text() == "os.replace"
    after = env.output.stat()
    assert env.output.read_bytes() == SENTINEL
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert env.snapshot() == source_before
    assert list(env.output.parent.iterdir()) == [env.output]


def test_publish_content_failure_with_force_preserves_existing_output_and_source(tmp_path):
    package_submit = _import_package_submit()
    env = _environment(tmp_path)
    env.output.parent.mkdir(parents=True)
    env.output.write_bytes(SENTINEL)
    output_before = env.output.stat()
    _append_source(env.agent_dir / "telemetry.py", '\nVALUE = "src.packing_core"\n')
    before = env.snapshot()
    assert _run_main(package_submit, env.argv(force=True)) == 1
    output_after = env.output.stat()
    assert env.output.read_bytes() == SENTINEL
    assert (output_after.st_ino, output_after.st_mtime_ns) == (
        output_before.st_ino,
        output_before.st_mtime_ns,
    )
    assert env.snapshot() == before
    assert list(env.output.parent.iterdir()) == [env.output]


@pytest.mark.parametrize("kind", ["symlink", "broken-symlink", "directory"])
@pytest.mark.parametrize("force", [False, True], ids=["no-force", "force"])
def test_publish_nonregular_or_symlink_output_is_always_exit_2(tmp_path, kind, force):
    package_submit = _import_package_submit()
    env = _environment(tmp_path)
    env.output.parent.mkdir(parents=True)
    protected_roots: list[Path] = []
    if kind == "symlink":
        target = env.root / "existing-target.zip"
        target.write_bytes(SENTINEL)
        env.output.symlink_to(target)
        protected_roots.append(target)
    elif kind == "broken-symlink":
        missing_target = env.root / "missing-target.zip"
        env.output.symlink_to(missing_target)
    else:
        env.output.mkdir()
        marker = env.output / "keep.txt"
        marker.write_bytes(SENTINEL)
        protected_roots.append(marker)
    before_link = os.readlink(env.output) if env.output.is_symlink() else None
    protected_before = _tree_snapshot(*protected_roots) if protected_roots else ()
    protected_stats = {
        path: (path.stat().st_ino, path.stat().st_mtime_ns) for path in protected_roots
    }
    before = env.output.lstat()
    assert _run_main(package_submit, env.argv(force=force)) == 2
    after = env.output.lstat()
    assert stat.S_IFMT(after.st_mode) == stat.S_IFMT(before.st_mode)
    assert after.st_ino == before.st_ino
    if before_link is not None:
        assert os.readlink(env.output) == before_link
    if kind == "broken-symlink":
        assert not os.path.lexists(missing_target)
    if protected_roots:
        assert _tree_snapshot(*protected_roots) == protected_before
        assert {
            path: (path.stat().st_ino, path.stat().st_mtime_ns) for path in protected_roots
        } == protected_stats


@pytest.mark.parametrize("force", [False, True], ids=["no-force", "force"])
def test_publish_fifo_output_fails_without_hanging_the_test_process(tmp_path, force):
    _import_package_submit()
    env = _environment(tmp_path)
    env.output.parent.mkdir(parents=True)
    os.mkfifo(env.output)
    before = env.output.lstat()
    completed = _run_cli(env.argv(force=force), timeout=10)
    assert completed.returncode == 2, completed.stderr
    after = env.output.lstat()
    assert stat.S_IFMT(after.st_mode) == stat.S_IFMT(before.st_mode)
    assert after.st_ino == before.st_ino


# =============================================================================
# ISOLATION — all modules, origins, environment, CWD, and Agent warmup
# =============================================================================


def test_isolation_imports_all_modules_from_payload_and_unsets_telemetry_env(
    tmp_path, monkeypatch
):
    package_submit = _import_package_submit()
    env = _environment(tmp_path)
    telemetry_target = tmp_path / "must-not-exist"
    monkeypatch.setenv("TELEMETRY_DIR", str(telemetry_target))
    monkeypatch.setenv("TELEMETRY_RUN_ID", "must-not-leak")
    before = env.snapshot()
    assert _run_main(package_submit, env.argv()) == 0
    assert env.snapshot() == before
    assert env.output.is_file()
    assert not telemetry_target.exists()
    assert not list(env.agent_dir.rglob("__pycache__"))
    assert not list(env.core_dir.rglob("__pycache__"))


@pytest.mark.parametrize("case", ["module-import", "agent-warmup"])
def test_isolation_import_or_agent_warmup_failure_is_exit_1_without_publish(tmp_path, case):
    package_submit = _import_package_submit()
    env = _environment(tmp_path)
    if case == "module-import":
        _append_source(env.core_dir / "watchdog.py", '\nraise RuntimeError("import failed")\n')
    else:
        agent = env.agent_dir / "agent.py"
        source = agent.read_text()
        source = source.replace(
            '    def __init__(self, module_path):\n',
            '    def __init__(self, module_path):\n        raise RuntimeError("warmup failed")\n',
        )
        agent.write_text(source)
    env.output.parent.mkdir(parents=True)
    env.output.write_bytes(SENTINEL)
    output_before = env.output.stat()
    before = env.snapshot()
    assert _run_main(package_submit, env.argv(force=True)) == 1
    output_after = env.output.stat()
    assert env.snapshot() == before
    assert env.output.read_bytes() == SENTINEL
    assert (output_after.st_ino, output_after.st_mtime_ns) == (
        output_before.st_ino,
        output_before.st_mtime_ns,
    )
    assert list(env.output.parent.iterdir()) == [env.output]


def test_real_agent_and_core_package_without_changing_development_sources(tmp_path):
    package_submit = _import_package_submit()
    env = _environment(tmp_path, real_sources=True)
    before = env.snapshot()
    assert _run_main(package_submit, env.argv()) == 0
    assert env.snapshot() == before
    with zipfile.ZipFile(env.output) as archive:
        assert tuple(info.filename for info in archive.infolist()) == EXPECTED_ENTRIES
        for name in EXPECTED_ENTRIES:
            if name.endswith(".py"):
                payload = archive.read(name)
                assert all(prefix not in payload for prefix in OLD_PREFIXES)
