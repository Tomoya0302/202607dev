"""Deterministic black-box fixtures for the T-031 benchmark contract.

The production module does not exist while these tests are authored.  This module therefore
contains only inputs and public-boundary fakes: config files, ``run_suite.run_suite``,
``check_telemetry.main``, Linux proc/cgroup files, Git subprocess results, and report IO.  It
does not import ``scripts.bench_policy`` and does not name any helper owned by that module.
"""
from __future__ import annotations

import builtins
import importlib
import io
import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable


RSS_LIMIT_BYTES = 10 * 1024**3
MEMORY_LIMIT_BYTES = 12 * 1024**3
POLICY_LIMIT_SECONDS = 5.0
FIRST_STEP_LIMIT_SECONDS = 6.0
OPTIMIZE_LIMIT_SECONDS = 170.0

def telemetry_row(
    step: int,
    t_total: Any,
    *,
    first_step: bool | None = None,
    emergency: bool = False,
) -> dict[str, Any]:
    """Return one T-028-compatible telemetry row."""
    if first_step is None:
        first_step = step == 0
    return {
        "step": step,
        "first_step": first_step,
        "t_total": t_total,
        "t_state": 0.01,
        "t_enum": 0.02,
        "t_mask": 0.03,
        "t_lpath": 0.04,
        "n_cand0": 4,
        "n_after_dims": 3,
        "n_after_geo": 2,
        "n_lpath_pass": 1,
        "decided_layer": 1,
        "regime": "conservative" if emergency else None,
        "p_ng_chosen": None,
        "reject_top3": [],
        "layer_error": ([{"layer": 1, "error_type": "emergency"}] if emergency else []),
    }


def task_config(
    task_id: str = "000",
    *,
    optimize: bool = True,
    policy_timeout: float = 8.0,
    optimization_timeout: float = 180.0,
) -> tuple[str, dict[str, Any]]:
    """Return the minimum config data used by bench-policy input validation."""
    return task_id, {
        "agent": {
            "optimize": optimize,
            "init_timeout": 10.0,
            "optimization_timeout": optimization_timeout,
            "policy_timeout": policy_timeout,
            "allowed_methods": ["get_init_states", "optimize", "policy"]
            if optimize
            else ["get_init_states", "policy"],
            "max_mem": 12,
        },
        "containers": {"container_list": []},
        "item_stream": {"item_list": []},
    }


def suite_row(
    config: str,
    task_id: str = "000",
    *,
    status: str = "success",
    optimization_time: Any = 12.5,
    policy_time: Any = 4.0,
    validator_ng: int | None = 0,
) -> dict[str, Any]:
    """Return one row matching the committed T-030B public JSONL schema."""
    return {
        "config": config,
        "agent": "agents/heuristic",
        "task_id": task_id,
        "status": status,
        "fill_score": 0.75 if status == "success" else None,
        "num_placed_items": 3.0 if status == "success" else None,
        "n_validator_ng": validator_ng if status == "success" else None,
        "format_error": status == "format_error",
        "exec_exception": status == "exec_exception",
        "policy_time": policy_time,
        "optimization_time": optimization_time,
        "wall_time": 0.5,
    }


@dataclass(frozen=True)
class ProcProcess:
    pid: int
    ppid: int
    start_time: int
    rss_bytes: int | float
    name: str = "worker"
    cmdline: str = "python\x00-m\x00scripts.run_local\x00"


def _proc_stat(process: ProcProcess) -> str:
    # proc(5): PPID is field 4 and starttime is field 22.  Keep enough trailing fields for
    # parsers that validate a normal-looking record.
    fields = [str(process.pid), f"({process.name})", "S", str(process.ppid)]
    fields.extend("0" for _ in range(17))
    fields.append(str(process.start_time))
    fields.extend("0" for _ in range(30))
    return " ".join(fields) + "\n"


class _ScandirIterator:
    def __init__(self, iterator):
        self._iterator = iterator

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self):
        self._iterator.close()


class FakeLinuxFS:
    """Route only Linux observation paths to a deterministic directory tree."""

    def __init__(self, root: Path):
        self.root = root
        self.proc_root = root / "proc"
        self.cgroup_root = root / "sys" / "fs" / "cgroup"
        self.proc_root.mkdir(parents=True)
        self.cgroup_root.mkdir(parents=True)
        self.proc_observed = threading.Event()
        self.descendant_status_observed = threading.Event()
        self._scan_condition = threading.Condition()
        self.scan_count = 0
        self.status_read_count = 0
        self._orig_open = builtins.open
        self._orig_io_open = io.open
        self._orig_os_open = os.open
        self._orig_stat = os.stat
        self._orig_lstat = os.lstat
        self._orig_scandir = os.scandir
        self._orig_listdir = os.listdir
        self._orig_readlink = os.readlink
        self.errors: dict[str, BaseException] = {}

    def _translate(self, path: Any) -> Any:
        if isinstance(path, int):
            return path
        try:
            raw = os.fspath(path)
        except TypeError:
            return path
        is_bytes = isinstance(raw, bytes)
        text = os.fsdecode(raw)
        replacement: Path | None = None
        if text == "/proc" or text.startswith("/proc/"):
            replacement = self.proc_root / text.removeprefix("/proc/") if text != "/proc" else self.proc_root
        elif text == "/sys/fs/cgroup" or text.startswith("/sys/fs/cgroup/"):
            suffix = text.removeprefix("/sys/fs/cgroup/")
            replacement = self.cgroup_root / suffix if text != "/sys/fs/cgroup" else self.cgroup_root
        if replacement is None:
            return path
        rendered = os.fspath(replacement)
        return os.fsencode(rendered) if is_bytes else rendered

    def _note_proc_access(self, path: Any) -> None:
        try:
            text = os.fsdecode(os.fspath(path))
        except (TypeError, ValueError):
            return
        relative = text.removeprefix("/proc/") if text.startswith("/proc/") else ""
        if text == "/proc" or (relative and relative.split("/", 1)[0].isdigit()):
            self.proc_observed.set()
        parts = relative.split("/") if relative else []
        if (
            len(parts) >= 2
            and parts[0].isdigit()
            and int(parts[0]) != os.getpid()
            and parts[1] == "status"
        ):
            self.descendant_status_observed.set()
            with self._scan_condition:
                self.status_read_count += 1
                self._scan_condition.notify_all()
        if text == "/proc":
            with self._scan_condition:
                self.scan_count += 1
                self._scan_condition.notify_all()

    def _raise_configured_error(self, path: Any) -> None:
        try:
            text = os.fsdecode(os.fspath(path))
        except (TypeError, ValueError):
            return
        for prefix, error in self.errors.items():
            if text == prefix or text.startswith(prefix.rstrip("/") + "/"):
                raise error

    def install(self, monkeypatch) -> None:
        def routed_open(file, *args, **kwargs):
            self._note_proc_access(file)
            self._raise_configured_error(file)
            return self._orig_open(self._translate(file), *args, **kwargs)

        def routed_io_open(file, *args, **kwargs):
            self._note_proc_access(file)
            self._raise_configured_error(file)
            return self._orig_io_open(self._translate(file), *args, **kwargs)

        def routed_os_open(file, *args, **kwargs):
            self._note_proc_access(file)
            self._raise_configured_error(file)
            return self._orig_os_open(self._translate(file), *args, **kwargs)

        def routed_stat(path, *args, **kwargs):
            self._note_proc_access(path)
            self._raise_configured_error(path)
            return self._orig_stat(self._translate(path), *args, **kwargs)

        def routed_lstat(path, *args, **kwargs):
            self._note_proc_access(path)
            self._raise_configured_error(path)
            return self._orig_lstat(self._translate(path), *args, **kwargs)

        def routed_scandir(path="."):
            self._note_proc_access(path)
            self._raise_configured_error(path)
            return _ScandirIterator(self._orig_scandir(self._translate(path)))

        def routed_listdir(path="."):
            self._note_proc_access(path)
            return self._orig_listdir(self._translate(path))

        def routed_readlink(path, *args, **kwargs):
            return self._orig_readlink(self._translate(path), *args, **kwargs)

        monkeypatch.setattr(builtins, "open", routed_open)
        monkeypatch.setattr(io, "open", routed_io_open)
        monkeypatch.setattr(os, "open", routed_os_open)
        monkeypatch.setattr(os, "stat", routed_stat)
        monkeypatch.setattr(os, "lstat", routed_lstat)
        monkeypatch.setattr(os, "scandir", routed_scandir)
        monkeypatch.setattr(os, "listdir", routed_listdir)
        monkeypatch.setattr(os, "readlink", routed_readlink)

    def write_processes(self, processes: Iterable[ProcProcess]) -> None:
        for entry in list(self.proc_root.iterdir()):
            if entry.name.isdigit():
                shutil.rmtree(entry)
        for process in processes:
            proc_dir = self.proc_root / str(process.pid)
            proc_dir.mkdir()
            (proc_dir / "stat").write_text(_proc_stat(process))
            # Fractional kB is deliberately supported by the fake so the byte-exact comparator
            # can be tested at LIMIT+1 even though a real kernel reports integral kB.
            rss_kib = float(process.rss_bytes) / 1024.0
            (proc_dir / "status").write_text(
                f"Name:\t{process.name}\nPid:\t{process.pid}\nPPid:\t{process.ppid}\n"
                f"VmRSS:\t{rss_kib:.12f} kB\n"
            )
            (proc_dir / "cmdline").write_bytes(process.cmdline.encode())

    def wait_for_scans(self, minimum: int, timeout: float = 3.0) -> bool:
        with self._scan_condition:
            return self._scan_condition.wait_for(lambda: self.scan_count >= minimum, timeout)

    def wait_for_status_reads(self, minimum: int, timeout: float = 3.0) -> bool:
        with self._scan_condition:
            return self._scan_condition.wait_for(
                lambda: self.status_read_count >= minimum, timeout
            )

    def write_cgroup_v2(
        self,
        *,
        quota: str = "200000 100000",
        cpuset: str = "0-1",
        memory_max: str = str(MEMORY_LIMIT_BYTES),
        memory_current: str = "4096",
        memory_peak: str = "8192",
    ) -> None:
        (self.proc_root / "self").mkdir(exist_ok=True)
        (self.proc_root / "self" / "cgroup").write_text("0::/bench\n")
        (self.proc_root / "self" / "mountinfo").write_text(
            "29 23 0:26 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime - cgroup2 cgroup rw\n"
        )
        (self.proc_root / "mounts").write_text("cgroup2 /sys/fs/cgroup cgroup2 rw 0 0\n")
        target = self.cgroup_root / "bench"
        target.mkdir(parents=True, exist_ok=True)
        (target / "cpu.max").write_text(quota + "\n")
        (target / "cpuset.cpus.effective").write_text(cpuset + "\n")
        (target / "memory.max").write_text(memory_max + "\n")
        (target / "memory.current").write_text(memory_current + "\n")
        (target / "memory.peak").write_text(memory_peak + "\n")

    def write_cgroup_v1(
        self,
        *,
        quota: str = "200000",
        period: str = "100000",
        cpuset: str = "0-1",
        memory_limit: str = str(MEMORY_LIMIT_BYTES),
        memory_current: str = "4096",
        memory_peak: str = "8192",
    ) -> None:
        (self.proc_root / "self").mkdir(exist_ok=True)
        (self.proc_root / "self" / "cgroup").write_text(
            "2:cpu,cpuacct:/bench\n3:cpuset:/bench\n4:memory:/bench\n"
        )
        (self.proc_root / "self" / "mountinfo").write_text(
            "30 23 0:27 / /sys/fs/cgroup/cpu rw - cgroup cgroup rw,cpu,cpuacct\n"
            "31 23 0:28 / /sys/fs/cgroup/cpuset rw - cgroup cgroup rw,cpuset\n"
            "32 23 0:29 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n"
        )
        (self.proc_root / "mounts").write_text(
            "cgroup /sys/fs/cgroup/cpu cgroup rw,cpu,cpuacct 0 0\n"
            "cgroup /sys/fs/cgroup/cpuset cgroup rw,cpuset 0 0\n"
            "cgroup /sys/fs/cgroup/memory cgroup rw,memory 0 0\n"
        )
        cpu = self.cgroup_root / "cpu" / "bench"
        cpu.mkdir(parents=True, exist_ok=True)
        (cpu / "cpu.cfs_quota_us").write_text(quota + "\n")
        (cpu / "cpu.cfs_period_us").write_text(period + "\n")
        cpus = self.cgroup_root / "cpuset" / "bench"
        cpus.mkdir(parents=True, exist_ok=True)
        (cpus / "cpuset.cpus").write_text(cpuset + "\n")
        memory = self.cgroup_root / "memory" / "bench"
        memory.mkdir(parents=True, exist_ok=True)
        (memory / "memory.limit_in_bytes").write_text(memory_limit + "\n")
        (memory / "memory.usage_in_bytes").write_text(memory_current + "\n")
        (memory / "memory.max_usage_in_bytes").write_text(memory_peak + "\n")


@dataclass
class FakeGit:
    real_run: Callable[..., subprocess.CompletedProcess]
    commit: str = "0123456789abcdef0123456789abcdef01234567"
    status: str = ""
    fail: str | None = None
    calls: list[list[str]] = field(default_factory=list)

    def __call__(self, args, *pargs, **kwargs):
        argv = [os.fspath(a) for a in args]
        if not argv or Path(argv[0]).name != "git":
            return self.real_run(args, *pargs, **kwargs)
        self.calls.append(argv)
        if self.fail == "missing":
            raise FileNotFoundError("git executable not found")
        if argv[1:] == ["rev-parse", "HEAD"]:
            if self.fail == "outside":
                return subprocess.CompletedProcess(
                    argv, 128, "", "fatal: not a git repository (or any parent directory)"
                )
            return self._result(argv, "rev-parse", self.commit + "\n")
        if argv[1:] == ["status", "--porcelain=v1", "--untracked-files=all"]:
            return self._result(argv, "status", self.status)
        return subprocess.CompletedProcess(argv, 2, "", "unexpected git argv")

    def _result(self, argv: list[str], operation: str, stdout: str):
        if self.fail == operation:
            return subprocess.CompletedProcess(argv, 128, "", f"{operation} failed")
        return subprocess.CompletedProcess(argv, 0, stdout, "")


@dataclass
class AtomicReplaceSpy:
    """Record report replacement or inject its filesystem failure deterministically."""

    real_replace: Callable[[Any, Any], None]
    fail: bool = False
    calls: list[tuple[Path, Path]] = field(default_factory=list)

    def __call__(self, source, destination) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        self.calls.append((source_path, destination_path))
        if self.fail:
            raise OSError("atomic replace failed")
        self.real_replace(source, destination)


@dataclass
class BenchEnvironment:
    tmp_path: Path
    monkeypatch: Any
    config_stems: list[str] = field(default_factory=lambda: ["custom-alpha"])
    telemetry: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    configs: dict[str, dict[str, Any]] = field(default_factory=dict)
    suite_rows: list[dict[str, Any]] = field(default_factory=list)
    suite_code: int = 0
    checker_code: int = 0
    checker_codes: dict[str, int] = field(default_factory=dict)
    run_suite_exception: BaseException | None = None
    run_suite_calls: list[dict[str, Any]] = field(default_factory=list)
    checker_calls: list[list[str]] = field(default_factory=list)
    before_suite_return: Callable[["BenchEnvironment"], None] | None = None
    wait_for_descendant_status: bool = True

    def __post_init__(self):
        self.config_dir = self.tmp_path / "configs"
        self.result_dir = self.tmp_path / "results"
        self.report_path = self.tmp_path / "bench-report.json"
        self.module_path = "agents/heuristic/"
        self.linux = FakeLinuxFS(self.tmp_path / "linux")
        bench_pid = os.getpid()
        self.linux.write_processes(
            [
                ProcProcess(bench_pid, os.getppid(), 10, 99 * 1024, name="bench"),
                ProcProcess(41001, bench_pid, 11, 256 * 1024**2, name="run_local"),
                ProcProcess(41002, 41001, 12, 128 * 1024**2, name="Runner"),
                ProcProcess(41003, 41002, 13, 64 * 1024**2, name="Agent"),
                ProcProcess(41004, 41001, 14, 8 * 1024**2, name="resource_tracker"),
            ]
        )
        self.linux.write_cgroup_v2()
        self.real_subprocess_run = subprocess.run
        self.git = FakeGit(self.real_subprocess_run)
        self._materialize_inputs()

    def _materialize_inputs(self) -> None:
        self.config_dir.mkdir(exist_ok=True)
        if not self.configs:
            for stem in self.config_stems:
                task_id, task = task_config()
                self.configs[stem] = {task_id: task}
        for stem, data in self.configs.items():
            (self.config_dir / f"{stem}.json").write_text(json.dumps(data))
        if not self.telemetry:
            self.telemetry = {
                stem: [telemetry_row(0, 5.5), telemetry_row(1, 1.0)]
                for stem in self.config_stems
            }
        if not self.suite_rows:
            self.suite_rows = [suite_row(stem) for stem in self.config_stems]

    @property
    def argv(self) -> list[str]:
        return [
            "--config-dir",
            str(self.config_dir),
            "--module-path",
            self.module_path,
            "--result-dir",
            str(self.result_dir),
            "--report",
            str(self.report_path),
        ]

    def _write_suite_artifacts(self, result_dir: Path) -> None:
        result_dir.mkdir(parents=True, exist_ok=True)
        with (result_dir / "suite_results.jsonl").open("w") as handle:
            for row in self.suite_rows:
                handle.write(json.dumps(row, allow_nan=True) + "\n")
        completed = {
            stem
            for stem in self.config_stems
            if any(row["config"] == stem for row in self.suite_rows)
            and all(
                row["status"] == "success"
                for row in self.suite_rows
                if row["config"] == stem
            )
        }
        validator_ng = sum(
            row["n_validator_ng"] or 0 for row in self.suite_rows if row["status"] == "success"
        )
        summary = {
            "agent": "agents/heuristic",
            "num_discovered": len(self.config_stems),
            "num_completed": len(completed),
            "completion_rate": len(completed) / len(self.config_stems) if self.config_stems else 0.0,
            "n_validator_ng_total": validator_ng,
            "n_format_error_total": sum(row["status"] == "format_error" for row in self.suite_rows),
            "n_exec_exception_total": sum(
                row["status"] == "exec_exception" for row in self.suite_rows
            ),
            "mean_fill": 0.75 if completed else 0.0,
            "suite_score": 0.75 if completed else 0.0,
            "compared_config_stems": sorted(self.config_stems),
        }
        (result_dir / "suite_summary.json").write_text(json.dumps(summary))
        (result_dir / "suite_report.md").write_text("# fake suite report\n")
        for stem, rows in self.telemetry.items():
            telemetry_dir = result_dir / stem / "telemetry"
            telemetry_dir.mkdir(parents=True, exist_ok=True)
            with (telemetry_dir / f"{stem}-run.jsonl").open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row, allow_nan=True) + "\n")

    def fake_run_suite(self, *args, **kwargs) -> int:
        self.run_suite_calls.append({"args": args, "kwargs": dict(kwargs)})
        if self.run_suite_exception is not None:
            raise self.run_suite_exception
        # The sampler and suite fake meet at an event; wall-clock duration never supplies an RSS
        # value or determines the scenario result.
        observed = (
            self.linux.descendant_status_observed
            if self.wait_for_descendant_status
            else self.linux.proc_observed
        )
        if not observed.wait(3.0):
            raise RuntimeError("RSS sampler did not inspect fake /proc while suite was active")
        result_arg = kwargs.get("result_dir")
        if result_arg is None and len(args) >= 3:
            result_arg = args[2]
        result_dir = Path(result_arg)
        self._write_suite_artifacts(result_dir)
        if self.before_suite_return is not None:
            self.before_suite_return(self)
        return self.suite_code

    def fake_check_telemetry(self, argv: list[str] | None = None) -> int:
        values = list(argv or [])
        self.checker_calls.append(values)
        if not values:
            return 2
        return self.checker_codes.get(Path(values[0]).name, self.checker_code)

    def import_bench_policy(self):
        """Install boundary fakes and perform the per-test delayed import."""
        import scripts.check_telemetry as check_telemetry
        import scripts.run_suite as run_suite

        self.linux.install(self.monkeypatch)
        self.monkeypatch.setattr(run_suite, "run_suite", self.fake_run_suite)
        self.monkeypatch.setattr(check_telemetry, "main", self.fake_check_telemetry)
        self.monkeypatch.setattr(subprocess, "run", self.git)
        sys.modules.pop("scripts.bench_policy", None)
        return importlib.import_module("scripts.bench_policy")

    def run_main(self, *, argv: list[str] | None = None):
        bench_policy = self.import_bench_policy()
        return bench_policy.main(self.argv if argv is None else argv)

    def read_report(self) -> dict[str, Any]:
        def reject_constant(token: str):
            raise ValueError(f"non-standard JSON constant: {token}")

        return json.loads(self.report_path.read_text(), parse_constant=reject_constant)


def normalize_report(report: dict[str, Any]) -> dict[str, Any]:
    """Remove only fields declared variable by the v1.25 reproducibility contract."""
    volatile_keys = {
        "generated_at_utc",
        "argv",
        "config_dir",
        "module_path",
        "result_dir",
        "report_path",
        "python_version",
        "platform",
        "telemetry_file",
    }

    def visit(value):
        if isinstance(value, dict):
            return {
                key: visit(child)
                for key, child in value.items()
                if key not in volatile_keys
            }
        if isinstance(value, list):
            return [visit(child) for child in value]
        if isinstance(value, str) and value.startswith("/"):
            return "<absolute-path>"
        return value

    return visit(json.loads(json.dumps(report)))
