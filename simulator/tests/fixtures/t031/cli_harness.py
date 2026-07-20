"""True ``python -m scripts.bench_policy`` subprocess harness for T-031.

The generated first ``PYTHONPATH`` entry is a namespace-style ``scripts`` package.  It supplies
only public dependency fakes; ``scripts.bench_policy`` itself is resolved from the repository.
``sitecustomize`` redirects Linux observation paths and Git before module execution.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from tests.fixtures.t031.harness import MEMORY_LIMIT_BYTES, suite_row, task_config, telemetry_row


_SITECUSTOMIZE = r'''
import builtins
import io
import os
import subprocess
import threading
from pathlib import Path

ROOT = Path(os.environ["T031_CLI_FAKE_ROOT"])
PROC_ROOT = ROOT / "proc"
CGROUP_ROOT = ROOT / "sys" / "fs" / "cgroup"
PROC_SCANNED = threading.Event()
DESCENDANT_STATUS_OBSERVED = threading.Event()

def _stat_line(pid, ppid, start, name):
    values = [str(pid), f"({name})", "S", str(ppid)] + ["0"] * 17 + [str(start)] + ["0"] * 30
    return " ".join(values) + "\n"

# Bind the synthetic tree to this actual module-CLI process, not to the parent pytest PID.
pid = os.getpid()
bench = PROC_ROOT / str(pid)
bench.mkdir(parents=True, exist_ok=True)
(bench / "stat").write_text(_stat_line(pid, os.getppid(), 1, "bench"))
(bench / "status").write_text("Name:\tbench\nVmRSS:\t1 kB\n")
(bench / "cmdline").write_bytes(b"python\0-m\0scripts.bench_policy\0")
worker = PROC_ROOT / "51001"
worker.mkdir(parents=True, exist_ok=True)
(worker / "stat").write_text(_stat_line(51001, pid, 2, "run_local"))
(worker / "status").write_text("Name:\trun_local\nVmRSS:\t65536 kB\n")
(worker / "cmdline").write_bytes(b"python\0-m\0scripts.run_local\0")

_open = builtins.open
_io_open = io.open
_os_open = os.open
_stat = os.stat
_lstat = os.lstat
_scandir = os.scandir
_listdir = os.listdir
_readlink = os.readlink
_run = subprocess.run

def _translate(path):
    if isinstance(path, int):
        return path
    try:
        raw = os.fspath(path)
    except TypeError:
        return path
    text = os.fsdecode(raw)
    replacement = None
    if text == "/proc" or text.startswith("/proc/"):
        replacement = PROC_ROOT if text == "/proc" else PROC_ROOT / text.removeprefix("/proc/")
    elif text == "/sys/fs/cgroup" or text.startswith("/sys/fs/cgroup/"):
        replacement = CGROUP_ROOT if text == "/sys/fs/cgroup" else CGROUP_ROOT / text.removeprefix("/sys/fs/cgroup/")
    if replacement is None:
        return path
    rendered = os.fspath(replacement)
    return os.fsencode(rendered) if isinstance(raw, bytes) else rendered

def _note(path):
    try:
        text = os.fsdecode(os.fspath(path))
    except (TypeError, ValueError):
        return
    if text == "/proc":
        PROC_SCANNED.set()
    relative = text.removeprefix("/proc/") if text.startswith("/proc/") else ""
    parts = relative.split("/") if relative else []
    if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) != os.getpid() and parts[1] == "status":
        DESCENDANT_STATUS_OBSERVED.set()

class _Scan:
    def __init__(self, value): self.value = value
    def __iter__(self): return self
    def __next__(self): return next(self.value)
    def __enter__(self): return self
    def __exit__(self, *args): self.value.close(); return False
    def close(self): self.value.close()

def routed_open(path, *args, **kwargs):
    _note(path)
    return _open(_translate(path), *args, **kwargs)
def routed_io_open(path, *args, **kwargs):
    _note(path)
    return _io_open(_translate(path), *args, **kwargs)
def routed_os_open(path, *args, **kwargs):
    _note(path)
    return _os_open(_translate(path), *args, **kwargs)
def routed_stat(path, *args, **kwargs):
    _note(path)
    return _stat(_translate(path), *args, **kwargs)
def routed_lstat(path, *args, **kwargs):
    _note(path)
    return _lstat(_translate(path), *args, **kwargs)
def routed_scandir(path="."):
    _note(path)
    return _Scan(_scandir(_translate(path)))
def routed_listdir(path="."):
    _note(path)
    return _listdir(_translate(path))
def routed_readlink(path, *args, **kwargs):
    return _readlink(_translate(path), *args, **kwargs)
def routed_run(args, *pargs, **kwargs):
    argv = [os.fspath(value) for value in args]
    if argv and Path(argv[0]).name == "git":
        if argv[1:] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(argv, 0, "0123456789abcdef0123456789abcdef01234567\n", "")
        if argv[1:] == ["status", "--porcelain=v1", "--untracked-files=all"]:
            return subprocess.CompletedProcess(argv, 0, "", "")
    return _run(args, *pargs, **kwargs)

builtins.open = routed_open
io.open = routed_io_open
os.open = routed_os_open
os.stat = routed_stat
os.lstat = routed_lstat
os.scandir = routed_scandir
os.listdir = routed_listdir
os.readlink = routed_readlink
subprocess.run = routed_run
'''


_RUN_SUITE_STUB = r'''
import json
import os
from pathlib import Path

def run_suite(config_dir, module_path, result_dir, baseline=None, timeout_sec=600.0):
    import sitecustomize
    if not sitecustomize.DESCENDANT_STATUS_OBSERVED.wait(3.0):
        raise RuntimeError("fake proc was not sampled")
    mode = os.environ["T031_CLI_MODE"]
    if mode == "interrupt":
        raise KeyboardInterrupt
    root = Path(os.environ["T031_CLI_DATA"])
    output = Path(result_dir)
    output.mkdir(parents=True, exist_ok=True)
    row = json.loads((root / "suite-row.json").read_text())
    (output / "suite_results.jsonl").write_text(json.dumps(row) + "\n")
    summary = {
        "agent": "agents/heuristic",
        "num_discovered": 1,
        "num_completed": 1,
        "completion_rate": 1.0,
        "n_validator_ng_total": 0,
        "n_format_error_total": 0,
        "n_exec_exception_total": 0,
        "mean_fill": 0.75,
        "suite_score": 0.75,
        "compared_config_stems": ["custom-alpha"],
    }
    (output / "suite_summary.json").write_text(json.dumps(summary))
    (output / "suite_report.md").write_text("# fake\n")
    telemetry_dir = output / "custom-alpha" / "telemetry"
    telemetry_dir.mkdir(parents=True)
    rows = json.loads((root / "telemetry.json").read_text())
    with (telemetry_dir / "run.jsonl").open("w") as handle:
        for value in rows:
            handle.write(json.dumps(value) + "\n")
    return 0
'''


_CHECKER_STUB = r'''
def main(argv=None):
    return 0
'''


def _stat_line(pid: int, ppid: int, start_time: int, name: str) -> str:
    values = [str(pid), f"({name})", "S", str(ppid)]
    values.extend("0" for _ in range(17))
    values.append(str(start_time))
    values.extend("0" for _ in range(30))
    return " ".join(values) + "\n"


@dataclass
class CliHarness:
    root: Path
    simulator_root: Path

    def __post_init__(self):
        self.inject = self.root / "inject"
        scripts = self.inject / "scripts"
        scripts.mkdir(parents=True)
        (scripts / "__init__.py").write_text(
            "from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n"
        )
        (scripts / "run_suite.py").write_text(_RUN_SUITE_STUB)
        (scripts / "check_telemetry.py").write_text(_CHECKER_STUB)
        (self.inject / "sitecustomize.py").write_text(_SITECUSTOMIZE)
        self.data = self.root / "data"
        self.data.mkdir()
        config_dir = self.root / "configs"
        config_dir.mkdir()
        task_id, task = task_config()
        (config_dir / "custom-alpha.json").write_text(json.dumps({task_id: task}))
        self.config_dir = config_dir
        self.result_dir = self.root / "results"
        self.report_path = self.root / "report.json"

    def _write_linux_files(self) -> Path:
        fake = self.root / "linux"
        proc = fake / "proc"
        cgroup = fake / "sys" / "fs" / "cgroup" / "bench"
        proc.mkdir(parents=True, exist_ok=True)
        cgroup.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()  # overwritten by the child bootstrap below through an env marker file
        # Include every plausible parent as a bench record; the child adds its own record before
        # execution via the launcher command's sitecustomize import.
        child_pid = int(os.environ.get("T031_PLACEHOLDER_CHILD_PID", pid))
        for value, ppid, start, rss, name in (
            (child_pid, os.getppid(), 1, 1, "bench"),
            (51001, child_pid, 2, 64 * 1024**2, "run_local"),
        ):
            target = proc / str(value)
            target.mkdir(exist_ok=True)
            (target / "stat").write_text(_stat_line(value, ppid, start, name))
            (target / "status").write_text(f"Name:\t{name}\nVmRSS:\t{rss / 1024:.12f} kB\n")
            (target / "cmdline").write_bytes(b"python\0")
        (proc / "self").mkdir(exist_ok=True)
        (proc / "self" / "cgroup").write_text("0::/bench\n")
        (proc / "self" / "mountinfo").write_text(
            "29 23 0:26 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n"
        )
        (proc / "mounts").write_text("cgroup2 /sys/fs/cgroup cgroup2 rw 0 0\n")
        (cgroup / "cpu.max").write_text("200000 100000\n")
        (cgroup / "cpuset.cpus.effective").write_text("0-1\n")
        (cgroup / "memory.max").write_text(f"{MEMORY_LIMIT_BYTES}\n")
        (cgroup / "memory.current").write_text("4096\n")
        (cgroup / "memory.peak").write_text("8192\n")
        return fake

    def run(self, mode: str) -> subprocess.CompletedProcess[str]:
        steady = 5.000001 if mode == "threshold" else 1.0
        (self.data / "telemetry.json").write_text(
            json.dumps([telemetry_row(0, 5.5), telemetry_row(1, steady)])
        )
        (self.data / "suite-row.json").write_text(json.dumps(suite_row("custom-alpha")))
        fake_linux = self._write_linux_files()
        env = dict(os.environ)
        old_path = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(
            [str(self.inject), str(self.simulator_root)] + ([old_path] if old_path else [])
        )
        env["T031_CLI_FAKE_ROOT"] = str(fake_linux)
        env["T031_CLI_DATA"] = str(self.data)
        env["T031_CLI_MODE"] = mode
        args = [
            sys.executable,
            "-m",
            "scripts.bench_policy",
            "--config-dir",
            str(self.config_dir),
            "--module-path",
            "agents/heuristic/",
            "--result-dir",
            str(self.result_dir),
            "--report",
            str(self.report_path),
        ]
        if mode == "input":
            args = [sys.executable, "-m", "scripts.bench_policy"]
        return subprocess.run(
            args,
            cwd=self.simulator_root,
            env=env,
            capture_output=True,
            text=True,
            shell=False,
        )
