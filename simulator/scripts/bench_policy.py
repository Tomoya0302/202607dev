"""T-031: 正式性能ゲート `scripts.bench_policy`（詳細仕様書 v1.25 §5.4）。

``scripts.run_suite.run_suite()`` をsuite全体につき厳密に1回呼び、2 CPU／12GiB以下のcgroup制約下で
次を判定する：定常policy p99 <= 5.0s／全初手最大 <= 6.0s／process tree RSS peak <= 10GiB／
``optimize=true`` taskの ``optimization_time`` <= 170.0s。

公開API： ``main(argv: list[str] | None = None) -> int``。argparseの ``SystemExit`` は ``main()`` 外へ
漏らさない。exit code：0=成功、1=閾値超過、2=入力・実行・計測・report不成立、130=KeyboardInterrupt
（優先順位 2 > 1 > 0）。

outcome文字列（人間確定の公開契約）：0→"pass"／1→"threshold_exceeded"／2→"error"／130→"interrupted"。
"fail"・"threshold_failed" は使用しない。標準ライブラリのみを使用する。
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
import platform
import shutil
import signal
import stat as stat_module
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# --- 固定契約値 -----------------------------------------------------------------------------

SCHEMA_VERSION = "t031.bench_policy.v1"
POLICY_LIMIT_SECONDS = 5.0
FIRST_STEP_LIMIT_SECONDS = 6.0
RSS_LIMIT_BYTES = 10 * 1024**3
OPTIMIZE_LIMIT_SECONDS = 170.0
MEMORY_LIMIT_BYTES = 12 * 1024**3
REQUIRED_CPU_LIMIT = 2.0
RSS_SAMPLE_INTERVAL_SECONDS = 0.02
DEFAULT_TIMEOUT_SECONDS = 600.0

_EXIT_OK = 0
_EXIT_THRESHOLD = 1
_EXIT_INPUT_ERROR = 2
_EXIT_INTERRUPTED = 130

_OUTCOME_PASS = "pass"
_OUTCOME_THRESHOLD_EXCEEDED = "threshold_exceeded"
_OUTCOME_ERROR = "error"
_OUTCOME_INTERRUPTED = "interrupted"


class _InputError(Exception):
    """事前検証・実行成立性・計測不成立を表す内部例外（exit 2 に正規化される）。"""


# --- 引数解析 --------------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="T-031 performance gate for a policy agent module (formal bench_policy CLI)."
    )
    parser.add_argument("--config-dir", required=True, type=str)
    parser.add_argument("--module-path", required=True, type=str)
    parser.add_argument("--result-dir", default=None, type=str)
    parser.add_argument("--report", required=True, type=str)
    parser.add_argument("--timeout-sec", default=None, type=str)
    return parser


def _parse_timeout(raw: str | None) -> float:
    """``--timeout-sec`` を検証する：finite かつ ``>0`` の数値文字列のみ許容する。"""
    if raw is None:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise _InputError(f"--timeout-sec must be a finite positive number: {raw!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise _InputError(f"--timeout-sec must be finite and > 0: {raw!r}")
    return value


# --- Git provenance --------------------------------------------------------------------------


def _collect_git_provenance() -> tuple[str | None, bool | None, str | None]:
    """``(git_commit, git_dirty, error)`` を返す。取得不能時は前2者を ``None`` にする。"""
    try:
        rev_parse = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
    except (OSError, FileNotFoundError) as exc:
        return None, None, f"git rev-parse HEAD failed to launch: {exc}"
    if rev_parse.returncode != 0:
        return None, None, "git rev-parse HEAD returned a non-zero exit code"
    commit = rev_parse.stdout.strip()
    if not commit:
        return None, None, "git rev-parse HEAD returned an empty commit id"
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, FileNotFoundError) as exc:
        return None, None, f"git status failed to launch: {exc}"
    if status.returncode != 0:
        return None, None, "git status --porcelain=v1 returned a non-zero exit code"
    dirty = len(status.stdout.strip("\n")) > 0 and status.stdout.strip() != ""
    return commit, bool(dirty), None


# --- cgroup ----------------------------------------------------------------------------------


def _read_mounts() -> list[tuple[str, str, str, list[str]]]:
    """``/proc/mounts`` を ``(device, mountpoint, fstype, options)`` の行群として返す。"""
    rows: list[tuple[str, str, str, list[str]]] = []
    with open("/proc/mounts") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 4:
                continue
            rows.append((fields[0], fields[1], fields[2], fields[3].split(",")))
    return rows


def _parse_cpu_range(text: str) -> int:
    """``cpuset.cpus`` 形式（例 ``0-3,7``）のCPU数を返す。不正形式は ``ValueError``。"""
    total = 0
    text = text.strip()
    if not text:
        raise ValueError("empty cpuset range")
    for part in text.split(","):
        part = part.strip()
        if not part:
            raise ValueError("empty cpuset range segment")
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError("invalid cpuset range order")
            total += hi - lo + 1
        else:
            int(part)  # validate
            total += 1
    if total <= 0:
        raise ValueError("cpuset range resolves to zero CPUs")
    return total


def _read_required_file(directory: str, name: str) -> str:
    path = os.path.join(directory, name)
    with open(path) as handle:
        return handle.read().strip()


def _detect_cgroup_v2_dir() -> str | None:
    try:
        with open("/proc/self/cgroup") as handle:
            self_lines = handle.readlines()
    except OSError:
        return None
    cgroup_path = None
    for line in self_lines:
        parts = line.strip().split(":", 2)
        if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
            cgroup_path = parts[2]
            break
    if cgroup_path is None:
        return None
    try:
        mounts = _read_mounts()
    except OSError:
        return None
    mountpoint = None
    for _device, mp, fstype, _opts in mounts:
        if fstype == "cgroup2":
            mountpoint = mp
            break
    if mountpoint is None:
        return None
    if cgroup_path in ("", "/"):
        return mountpoint.rstrip("/") or "/"
    return mountpoint.rstrip("/") + cgroup_path


def _detect_cgroup_v1_dir(controller: str) -> str | None:
    try:
        with open("/proc/self/cgroup") as handle:
            self_lines = handle.readlines()
    except OSError:
        return None
    controller_path = None
    for line in self_lines:
        parts = line.strip().split(":", 2)
        if len(parts) != 3:
            continue
        controllers = [c for c in parts[1].split(",") if c]
        if controller in controllers:
            controller_path = parts[2]
            break
    if controller_path is None:
        return None
    try:
        mounts = _read_mounts()
    except OSError:
        return None
    mountpoint = None
    for _device, mp, fstype, opts in mounts:
        if fstype == "cgroup" and controller in opts:
            mountpoint = mp
            break
    if mountpoint is None:
        return None
    if controller_path in ("", "/"):
        return mountpoint.rstrip("/") or "/"
    return mountpoint.rstrip("/") + controller_path


def _sched_affinity_count() -> int | None:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None


@dataclass
class _ContainerLimits:
    verified: bool = False
    limits_source: str | None = None
    effective_cpu_limit: float | None = None
    cpu_quota_limit: float | None = None
    cpuset_cpu_limit: int | None = None
    memory_limit_bytes: int | None = None
    memory_current_bytes: int | None = None
    memory_peak_bytes: int | None = None
    sched_affinity_cpu_count: int | None = None
    error: str | None = None

    def as_report_dict(self) -> dict[str, Any]:
        return {
            "required_cpu_limit": REQUIRED_CPU_LIMIT,
            "required_memory_limit_bytes": MEMORY_LIMIT_BYTES,
            "limits_source": self.limits_source,
            "effective_cpu_limit": self.effective_cpu_limit,
            "cpu_quota_limit": self.cpu_quota_limit,
            "cpuset_cpu_limit": self.cpuset_cpu_limit,
            "memory_limit_bytes": self.memory_limit_bytes,
            "memory_current_bytes": self.memory_current_bytes,
            "memory_peak_bytes": self.memory_peak_bytes,
            "sched_affinity_cpu_count": self.sched_affinity_cpu_count,
            "container_limits_verified": self.verified,
        }


def _read_container_limits() -> _ContainerLimits:
    result = _ContainerLimits(sched_affinity_cpu_count=_sched_affinity_count())
    v2_dir = _detect_cgroup_v2_dir()
    if v2_dir is not None and os.path.isdir(v2_dir):
        result.limits_source = "cgroup_v2"
        try:
            quota_text = _read_required_file(v2_dir, "cpu.max")
            cpuset_text = _read_required_file(v2_dir, "cpuset.cpus.effective")
            memory_text = _read_required_file(v2_dir, "memory.max")
        except OSError as exc:
            result.error = f"cgroup v2 required file unavailable: {exc}"
            return result
        try:
            quota_parts = quota_text.split()
            if len(quota_parts) != 2:
                raise ValueError("cpu.max must have 2 fields")
            quota_raw, period_raw = quota_parts
            if quota_raw == "max":
                result.cpu_quota_limit = None
            else:
                quota_val = float(quota_raw)
                period_val = float(period_raw)
                if period_val <= 0:
                    raise ValueError("cpu.max period must be positive")
                result.cpu_quota_limit = quota_val / period_val
            result.cpuset_cpu_limit = _parse_cpu_range(cpuset_text)
            if memory_text == "max":
                result.memory_limit_bytes = None
            else:
                mem_val = int(memory_text)
                result.memory_limit_bytes = mem_val
        except (ValueError, TypeError) as exc:
            result.error = f"cgroup v2 malformed value: {exc}"
            return result
        try:
            result.memory_current_bytes = int(_read_required_file(v2_dir, "memory.current"))
        except (OSError, ValueError):
            pass
        try:
            peak_text = _read_required_file(v2_dir, "memory.peak")
            result.memory_peak_bytes = int(peak_text)
        except (OSError, ValueError):
            pass
    else:
        cpu_dir = _detect_cgroup_v1_dir("cpu")
        cpuset_dir = _detect_cgroup_v1_dir("cpuset")
        memory_dir = _detect_cgroup_v1_dir("memory")
        if cpu_dir is None or cpuset_dir is None or memory_dir is None:
            result.error = "cgroup v1/v2 controllers could not be located"
            return result
        result.limits_source = "cgroup_v1"
        try:
            quota_raw = _read_required_file(cpu_dir, "cpu.cfs_quota_us")
            period_raw = _read_required_file(cpu_dir, "cpu.cfs_period_us")
            cpuset_text = _read_required_file(cpuset_dir, "cpuset.cpus")
            memory_raw = _read_required_file(memory_dir, "memory.limit_in_bytes")
        except OSError as exc:
            result.error = f"cgroup v1 required file unavailable: {exc}"
            return result
        try:
            quota_val = int(quota_raw)
            period_val = int(period_raw)
            if period_val <= 0:
                raise ValueError("cpu.cfs_period_us must be positive")
            result.cpu_quota_limit = None if quota_val <= 0 else quota_val / period_val
            result.cpuset_cpu_limit = _parse_cpu_range(cpuset_text)
            mem_val = int(memory_raw)
            # cgroup v1 の "unlimited" は極めて大きな値で表現される（実質無制限）。
            result.memory_limit_bytes = None if mem_val >= (1 << 62) else mem_val
        except (ValueError, TypeError) as exc:
            result.error = f"cgroup v1 malformed value: {exc}"
            return result
        try:
            result.memory_current_bytes = int(
                _read_required_file(memory_dir, "memory.usage_in_bytes")
            )
        except (OSError, ValueError):
            pass
        try:
            result.memory_peak_bytes = int(
                _read_required_file(memory_dir, "memory.max_usage_in_bytes")
            )
        except (OSError, ValueError):
            pass

    # CPUの有効上限は、quota/cpusetの優先順位ではなく、利用可能な有限・正のhard limitの最小値。
    # quota=unlimitedとcpusetの空・欠落・parse不能は、それぞれ「有限・正の候補を提供しない」
    # として扱い、他方に有効候補があればその値で判定する（malformed値そのものは上流のparse
    # ブロックで既に計測不成立として弾かれている）。
    finite_positive_limits: list[float] = []
    if (
        result.cpu_quota_limit is not None
        and math.isfinite(result.cpu_quota_limit)
        and result.cpu_quota_limit > 0
    ):
        finite_positive_limits.append(float(result.cpu_quota_limit))
    if (
        result.cpuset_cpu_limit is not None
        and math.isfinite(result.cpuset_cpu_limit)
        and result.cpuset_cpu_limit > 0
    ):
        finite_positive_limits.append(float(result.cpuset_cpu_limit))
    if not finite_positive_limits:
        result.error = "no finite positive CPU limit could be determined (unlimited)"
        return result
    result.effective_cpu_limit = min(finite_positive_limits)
    if result.effective_cpu_limit > REQUIRED_CPU_LIMIT:
        result.error = f"CPU limit exceeds {REQUIRED_CPU_LIMIT}: {result.effective_cpu_limit}"
        return result
    if result.memory_limit_bytes is None:
        result.error = "memory limit is unlimited"
        return result
    if not math.isfinite(result.memory_limit_bytes) or result.memory_limit_bytes <= 0:
        result.error = "memory limit is not finite/positive"
        return result
    if result.memory_limit_bytes > MEMORY_LIMIT_BYTES:
        result.error = f"memory limit exceeds {MEMORY_LIMIT_BYTES}: {result.memory_limit_bytes}"
        return result
    result.verified = True
    return result


# --- filesystem 事前検証 helper --------------------------------------------------------------


def _lstat_kind(path: Path) -> str:
    """``"missing"``／``"symlink"``／``"file"``／``"dir"``／``"other"`` を返す。"""
    try:
        st = os.lstat(path)
    except OSError:
        return "missing"
    if stat_module.S_ISLNK(st.st_mode):
        return "symlink"
    if stat_module.S_ISDIR(st.st_mode):
        return "dir"
    if stat_module.S_ISREG(st.st_mode):
        return "file"
    return "other"


def _validate_config_dir(config_dir: Path) -> None:
    kind = _lstat_kind(config_dir)
    if kind != "dir":
        raise _InputError(f"--config-dir is not a regular directory ({kind}): {config_dir}")


def _validate_report_path(report_path: Path) -> None:
    kind = _lstat_kind(report_path)
    if kind in ("dir", "symlink", "other"):
        raise _InputError(f"--report path is not writable as a regular file ({kind}): {report_path}")
    if not report_path.parent.is_dir():
        raise _InputError(f"--report parent directory does not exist: {report_path.parent}")


def _validate_result_dir(result_dir: Path) -> None:
    kind = _lstat_kind(result_dir)
    if kind == "symlink":
        raise _InputError(f"--result-dir must not be a symlink: {result_dir}")
    if kind == "file" or kind == "other":
        raise _InputError(f"--result-dir is not a directory: {result_dir}")
    if kind == "dir":
        if any(result_dir.iterdir()):
            raise _InputError(f"--result-dir must be empty when it already exists: {result_dir}")


def _discover_configs(config_dir: Path) -> list[Path]:
    """``config_dir`` 直下の ``c*.json`` をファイル名昇順で発見する。"""
    try:
        entries = list(os.scandir(config_dir))
    except OSError as exc:
        raise _InputError(f"failed to list --config-dir: {exc}") from exc
    matched: list[str] = [entry.name for entry in entries if fnmatch.fnmatch(entry.name, "c*.json")]
    matched.sort()
    found: list[Path] = []
    for name in matched:
        candidate = config_dir / name
        kind = _lstat_kind(candidate)
        if kind != "file":
            raise _InputError(f"config entry is not a regular file ({kind}): {candidate}")
        found.append(candidate)
    if not found:
        raise _InputError(f"no c*.json configs discovered in {config_dir}")
    stems = [p.stem for p in found]
    if len(set(stems)) != len(stems):
        raise _InputError("duplicate config stem discovered")
    return found


def _load_config_json(path: Path) -> dict[str, Any]:
    try:
        with open(path) as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise _InputError(f"failed to parse config JSON {path}: {exc}") from exc
    if not isinstance(data, dict) or not data:
        raise _InputError(f"config JSON must be a non-empty object: {path}")
    return data


# --- RSS sampler -------------------------------------------------------------------------------


@dataclass
class _ProcInfo:
    pid: int
    ppid: int
    start_time: int


def _parse_stat_line(text: str) -> tuple[int, int, int]:
    """``/proc/<pid>/stat`` の内容から ``(pid, ppid, start_time)`` を返す。"""
    close = text.rfind(")")
    if close == -1:
        raise ValueError("malformed /proc/<pid>/stat: no comm field")
    pid_field = text[:text.find("(")].strip()
    rest = text[close + 1:].split()
    if len(rest) < 20:
        raise ValueError("malformed /proc/<pid>/stat: too few fields")
    pid = int(pid_field)
    ppid = int(rest[1])
    start_time = int(rest[19])
    return pid, ppid, start_time


def _read_stat(pid: int) -> _ProcInfo:
    with open(f"/proc/{pid}/stat") as handle:
        text = handle.read()
    read_pid, ppid, start_time = _parse_stat_line(text)
    return _ProcInfo(pid=read_pid, ppid=ppid, start_time=start_time)


def _read_vmrss_bytes(pid: int) -> int:
    with open(f"/proc/{pid}/status") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"malformed VmRSS line: {line!r}")
                kib = float(parts[1])
                return round(kib * 1024.0)
    return 0


def _read_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            raw = handle.read()
    except OSError:
        return ""
    return raw.decode("utf-8", errors="replace")


def _attributed_stem(cmdline: str, discovered_stems: set[str]) -> str | None:
    if "scripts.run_local" not in cmdline:
        return None
    tokens = cmdline.split("\x00")
    if "--config-path" not in tokens:
        return None
    idx = tokens.index("--config-path")
    if idx + 1 >= len(tokens):
        return None
    stem = Path(tokens[idx + 1]).stem
    return stem if stem in discovered_stems else None


class _RssSampler:
    """bench自身を除く全子孫processのRSS合計peakを ``0.02`` 秒間隔でsamplingする。"""

    def __init__(self, root_pid: int, discovered_stems: set[str]) -> None:
        self._root_pid = root_pid
        self._discovered_stems = discovered_stems
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_bytes = 0
        self.config_peak_bytes: dict[str, int] = {stem: 0 for stem in discovered_stems}
        self.config_attributed: dict[str, bool] = {stem: False for stem in discovered_stems}
        self.observed = False
        self.unattributed_observed = False
        self.error: str | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop_join(self, timeout: float = 10.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        try:
            while True:
                self._sample_once()
                if self._stop_event.wait(RSS_SAMPLE_INTERVAL_SECONDS):
                    break
        except BaseException as exc:  # noqa: BLE001 - sampler異常は全てreport化しexit2にする
            if self.error is None:
                self.error = f"RSS sampler failed: {exc}"

    def _sample_once(self) -> None:
        with os.scandir("/proc") as it:
            pids = [entry.name for entry in it if entry.name.isdigit()]
        proc_map: dict[int, _ProcInfo] = {}
        for name in pids:
            pid = int(name)
            try:
                proc_map[pid] = _read_stat(pid)
            except FileNotFoundError:
                continue  # 正常終了とのrace
        # bench(root)の子孫集合をPPIDグラフで算出する。
        children: dict[int, list[int]] = {}
        for pid, info in proc_map.items():
            children.setdefault(info.ppid, []).append(pid)
        descendants: list[int] = []
        frontier = [self._root_pid]
        seen = {self._root_pid}
        while frontier:
            next_frontier: list[int] = []
            for parent in frontier:
                for child in children.get(parent, []):
                    if child in seen:
                        continue
                    seen.add(child)
                    descendants.append(child)
                    next_frontier.append(child)
            frontier = next_frontier
        if not descendants:
            return
        self.observed = True

        attribution_root: dict[int, str] = {}
        for pid in descendants:
            cmdline = _read_cmdline(pid)
            stem = _attributed_stem(cmdline, self._discovered_stems)
            if stem is not None:
                attribution_root[pid] = stem

        sample_total = 0
        per_stem_total: dict[str, int] = {stem: 0 for stem in self._discovered_stems}
        any_attributed_this_sample: set[str] = set()
        for pid in descendants:
            info = proc_map[pid]
            try:
                start_before = info.start_time
                rss_bytes = _read_vmrss_bytes(pid)
                confirm = _read_stat(pid)
            except FileNotFoundError:
                continue  # 正常終了とのrace
            if confirm.start_time != start_before:
                continue  # PID再利用: 読取り前後のstart time不一致は破棄
            sample_total += rss_bytes
            stem = self._nearest_attributed_stem(pid, proc_map, attribution_root)
            if stem is not None:
                per_stem_total[stem] += rss_bytes
                any_attributed_this_sample.add(stem)
            else:
                self.unattributed_observed = True

        self.peak_bytes = max(self.peak_bytes, sample_total)
        for stem, total in per_stem_total.items():
            if stem in any_attributed_this_sample:
                self.config_attributed[stem] = True
                self.config_peak_bytes[stem] = max(self.config_peak_bytes[stem], total)

    def _nearest_attributed_stem(
        self, pid: int, proc_map: dict[int, _ProcInfo], attribution_root: dict[int, str]
    ) -> str | None:
        current = pid
        visited = 0
        while current in proc_map and visited < len(proc_map) + 1:
            if current in attribution_root:
                return attribution_root[current]
            if current == self._root_pid:
                return None
            current = proc_map[current].ppid
            visited += 1
        return None


def _reap_descendants(root_pid: int, timeout: float = 2.0) -> None:
    """KeyboardInterrupt時、bench自身の子孫processだけを明示的に終了・回収する。"""

    def _collect() -> dict[int, _ProcInfo]:
        try:
            with os.scandir("/proc") as it:
                names = [entry.name for entry in it if entry.name.isdigit()]
        except OSError:
            return {}
        info_by_pid: dict[int, _ProcInfo] = {}
        for name in names:
            pid = int(name)
            try:
                info_by_pid[pid] = _read_stat(pid)
            except (FileNotFoundError, ValueError, OSError):
                continue
        return info_by_pid

    def _descendants_of(info_by_pid: dict[int, _ProcInfo]) -> list[_ProcInfo]:
        children: dict[int, list[int]] = {}
        for pid, info in info_by_pid.items():
            children.setdefault(info.ppid, []).append(pid)
        result: list[_ProcInfo] = []
        frontier = [root_pid]
        seen = {root_pid}
        while frontier:
            next_frontier = []
            for parent in frontier:
                for child in children.get(parent, []):
                    if child in seen:
                        continue
                    seen.add(child)
                    result.append(info_by_pid[child])
                    next_frontier.append(child)
            frontier = next_frontier
        return result

    targets = _descendants_of(_collect())
    if not targets:
        return
    for info in sorted(targets, key=lambda i: i.pid):
        try:
            os.kill(info.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            pass

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remaining = _descendants_of(_collect())
        if not remaining:
            return
        time.sleep(0.05)

    for info in _descendants_of(_collect()):
        try:
            os.kill(info.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            pass


# --- percentile --------------------------------------------------------------------------------


def _percentile(sorted_values: list[float], p: float) -> float:
    """linear interpolation（Hyndman-Fan type 7）で百分位数を算出する。"""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    q = p / 100.0
    h = (n - 1) * q
    j = math.floor(h)
    g = h - j
    lo = sorted_values[int(j)]
    hi = sorted_values[min(int(j) + 1, n - 1)]
    return lo + g * (hi - lo)


def _percentiles_dict(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "p99": None}
    ordered = sorted(values)
    return {
        "p50": _percentile(ordered, 50),
        "p95": _percentile(ordered, 95),
        "p99": _percentile(ordered, 99),
    }


def _validate_t_total(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _InputError(f"t_total must be a numeric, non-bool value: {value!r}")
    fvalue = float(value)
    if not math.isfinite(fvalue) or fvalue < 0:
        raise _InputError(f"t_total must be finite and >= 0: {value!r}")
    return fvalue


# --- telemetry ---------------------------------------------------------------------------------


@dataclass
class _ConfigTelemetry:
    telemetry_file: str | None
    telemetry_check_exit_code: int | None
    policy_count: int
    first_step_count: int
    steady_step_count: int
    first_step_values: list[float]
    steady_values: list[float]


class _TelemetryCheckFailed(Exception):
    """telemetry検証の失敗。computed済みの部分結果を ``telemetry`` に保持し report化を可能にする。"""

    def __init__(self, telemetry: "_ConfigTelemetry", message: str) -> None:
        super().__init__(message)
        self.telemetry = telemetry


def _find_single_telemetry_file(telemetry_dir: Path) -> Path:
    if not telemetry_dir.is_dir():
        raise _InputError(f"telemetry directory missing: {telemetry_dir}")
    matched = sorted(p.name for p in telemetry_dir.iterdir() if fnmatch.fnmatch(p.name, "*.jsonl"))
    if len(matched) != 1:
        raise _InputError(f"expected exactly one telemetry JSONL file, found {len(matched)}: {telemetry_dir}")
    candidate = telemetry_dir / matched[0]
    kind = _lstat_kind(candidate)
    if kind != "file":
        raise _InputError(f"telemetry file is not a regular file ({kind}): {candidate}")
    if candidate.stat().st_size == 0:
        raise _InputError(f"telemetry file is empty: {candidate}")
    return candidate


def _parse_telemetry_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = path.read_text()
    for line in text.splitlines():
        if line.strip() == "":
            raise _InputError(f"telemetry file has a blank line: {path}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _InputError(f"telemetry file has an unparseable line: {path}: {exc}") from exc
        rows.append(row)
    return rows


def _validate_segments_and_populate(rows: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    first_values: list[float] = []
    steady_values: list[float] = []
    expected_next: int | None = None
    for row in rows:
        first_step = row.get("first_step")
        step = row.get("step")
        if not isinstance(first_step, bool):
            raise _InputError(f"telemetry row first_step must be bool: {row!r}")
        if not isinstance(step, int) or isinstance(step, bool):
            raise _InputError(f"telemetry row step must be int: {row!r}")
        t_total = _validate_t_total(row.get("t_total"))
        if first_step:
            if step != 0:
                raise _InputError("telemetry first_step row must have step == 0")
            expected_next = 1
            first_values.append(t_total)
        else:
            if expected_next is None or step != expected_next:
                raise _InputError("telemetry steady row breaks segment continuity")
            expected_next += 1
            steady_values.append(t_total)
    return first_values, steady_values


def _check_config_telemetry(
    checker_main, result_dir: Path, stem: str
) -> _ConfigTelemetry:
    """configのtelemetryを検証する。失敗時も部分結果を ``_TelemetryCheckFailed.telemetry`` へ載せる。"""
    partial = _ConfigTelemetry(
        telemetry_file=None,
        telemetry_check_exit_code=None,
        policy_count=0,
        first_step_count=0,
        steady_step_count=0,
        first_step_values=[],
        steady_values=[],
    )
    telemetry_dir = result_dir / stem / "telemetry"
    try:
        path = _find_single_telemetry_file(telemetry_dir)
    except _InputError as exc:
        raise _TelemetryCheckFailed(partial, str(exc)) from exc
    partial.telemetry_file = str(path)
    exit_code = checker_main([str(path)])
    partial.telemetry_check_exit_code = exit_code
    if exit_code != 0:
        raise _TelemetryCheckFailed(
            partial, f"check_telemetry.main reported exit {exit_code} for {path}"
        )
    try:
        rows = _parse_telemetry_rows(path)
        first_values, steady_values = _validate_segments_and_populate(rows)
    except _InputError as exc:
        raise _TelemetryCheckFailed(partial, str(exc)) from exc
    partial.policy_count = len(rows)
    partial.first_step_count = len(first_values)
    partial.steady_step_count = len(steady_values)
    partial.first_step_values = first_values
    partial.steady_values = steady_values
    if not first_values:
        raise _TelemetryCheckFailed(partial, f"config {stem} has zero first-step telemetry rows")
    return partial


# --- suite artifacts / identity ------------------------------------------------------------


def _load_suite_results(result_dir: Path) -> list[dict[str, Any]]:
    path = result_dir / "suite_results.jsonl"
    try:
        text = path.read_text()
    except OSError as exc:
        raise _InputError(f"suite_results.jsonl missing/unreadable: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.strip() == "":
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise _InputError(f"suite_results.jsonl has an unparseable line: {exc}") from exc
    if not rows:
        raise _InputError("suite_results.jsonl has zero rows")
    return rows


def _verify_suite_artifacts_present(result_dir: Path) -> None:
    summary_path = result_dir / "suite_summary.json"
    try:
        with open(summary_path) as handle:
            json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise _InputError(f"suite_summary.json missing/unreadable: {exc}") from exc
    report_path = result_dir / "suite_report.md"
    if not report_path.is_file():
        raise _InputError(f"suite_report.md missing: {report_path}")


def _verify_identity(
    discovered_stems: list[str], suite_rows: list[dict[str, Any]], expected_task_count: int
) -> None:
    if len(suite_rows) != expected_task_count:
        raise _InputError(
            f"suite_results.jsonl row count {len(suite_rows)} != expected task count {expected_task_count}"
        )
    seen_pairs: set[tuple[str, Any]] = set()
    first_appearance: list[str] = []
    seen_configs: set[str] = set()
    for row in suite_rows:
        config = row.get("config")
        task_id = row.get("task_id")
        pair = (config, task_id)
        if pair in seen_pairs:
            raise _InputError(f"duplicate (config, task_id) row: {pair}")
        seen_pairs.add(pair)
        if config not in seen_configs:
            seen_configs.add(config)
            first_appearance.append(config)
    if set(first_appearance) != set(discovered_stems):
        raise _InputError("suite_results.jsonl config set does not match discovered configs")
    if first_appearance != discovered_stems:
        raise _InputError("suite_results.jsonl config order does not match discovered order")
    for row in suite_rows:
        status = row.get("status")
        if status not in ("success", "format_error", "exec_exception"):
            raise _InputError(f"unexpected status value: {status!r}")
        if status in ("format_error", "exec_exception"):
            raise _InputError(f"config {row.get('config')!r} has status {status!r} (not a valid measurement)")


# --- optimize / timeout suspicion -------------------------------------------------------------


def _finite_nonneg_nonbool(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _InputError(f"expected a finite non-bool numeric value: {value!r}")
    fvalue = float(value)
    if not math.isfinite(fvalue) or fvalue < 0:
        raise _InputError(f"expected a finite value >= 0: {value!r}")
    return fvalue


def _evaluate_task(
    task_config: dict[str, Any], suite_row: dict[str, Any] | None
) -> tuple[dict[str, Any], str | None]:
    """taskを評価し ``(entry, error)`` を返す。可能な限り ``entry`` へ値を残してから ``error`` を返す。"""
    agent_cfg = task_config.get("agent", {}) if isinstance(task_config, dict) else {}
    optimize = bool(agent_cfg.get("optimize", False))
    entry: dict[str, Any] = {
        "task_id": suite_row.get("task_id") if suite_row else None,
        "optimization_time_seconds": None,
        "optimization_threshold_pass": None,
    }
    if suite_row is None:
        if optimize:
            return entry, "optimize=true task missing a suite result row"
        return entry, None

    error: str | None = None
    policy_time = suite_row.get("policy_time")
    policy_timeout = agent_cfg.get("policy_timeout")
    if policy_time is not None and isinstance(policy_timeout, (int, float)) and not isinstance(
        policy_timeout, bool
    ):
        try:
            policy_time_f = _finite_nonneg_nonbool(policy_time)
        except _InputError:
            policy_time_f = None
        if policy_time_f is not None and policy_time_f >= float(policy_timeout):
            error = "policy_time >= agent.policy_timeout: external timeout not excludable"

    if not optimize:
        return entry, error

    raw_value = suite_row.get("optimization_time")
    try:
        value = _finite_nonneg_nonbool(raw_value)
    except _InputError as exc:
        return entry, (error or str(exc))
    entry["optimization_time_seconds"] = value
    entry["optimization_threshold_pass"] = value <= OPTIMIZE_LIMIT_SECONDS
    optimization_timeout = agent_cfg.get("optimization_timeout")
    if isinstance(optimization_timeout, (int, float)) and not isinstance(optimization_timeout, bool):
        if value >= float(optimization_timeout):
            error = error or "optimization_time >= agent.optimization_timeout: external timeout not excludable"
    return entry, error


# --- report構築・atomic write ------------------------------------------------------------------


def _now_utc_iso() -> str:
    now = datetime.now(timezone.utc)
    return now.isoformat(timespec="microseconds").replace("+00:00", "") + "Z"


def _write_report_atomic(report: dict[str, Any], report_path: Path) -> None:
    directory = report_path.parent
    fd, tmp_name = tempfile.mkstemp(
        prefix=".bench_policy_report_", suffix=".tmp", dir=str(directory)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(report, handle, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, report_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _finalize(
    report: dict[str, Any], report_path: Path | None, outcome: str, overall_pass: bool, code: int
) -> int:
    report["outcome"] = outcome
    report["overall_pass"] = overall_pass
    report["generated_at_utc"] = _now_utc_iso()
    if report_path is not None:
        try:
            _write_report_atomic(report, report_path)
        except OSError:
            return _EXIT_INPUT_ERROR
    return code


# --- main ----------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    resolved_argv = list(sys.argv[1:] if argv is None else argv)
    report: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    errors: list[str] = []
    warnings: list[str] = []
    report_path: Path | None = None
    sampler: _RssSampler | None = None
    temp_result_dir: str | None = None

    try:
        parser = _build_parser()
        try:
            args = parser.parse_args(resolved_argv)
        except SystemExit:
            return _EXIT_INPUT_ERROR

        report["config_dir"] = str(Path(args.config_dir).resolve())
        report["module_path"] = args.module_path
        report["result_dir"] = str(Path(args.result_dir).resolve()) if args.result_dir else None
        # symlink検出のため resolve() は行わない（resolve() はsymlinkを辿ってしまう）。
        report_path = Path(args.report)
        report["report_path"] = str(report_path)
        report["argv"] = resolved_argv
        report["python_version"] = platform.python_version()
        report["platform"] = platform.platform()
        report["thresholds"] = {
            "steady_policy_p99_seconds": POLICY_LIMIT_SECONDS,
            "first_step_max_seconds": FIRST_STEP_LIMIT_SECONDS,
            "process_tree_rss_peak_bytes": RSS_LIMIT_BYTES,
            "optimization_time_max_seconds": OPTIMIZE_LIMIT_SECONDS,
        }
        report["measurement_contract"] = {
            "telemetry_completeness": "best_effort",
            "strict_completed_policy_count_available": False,
            "tail_loss_detectable": False,
            "writer_warning_observable": False,
            "percentile_method": "linear_type_7",
            "rss_measurement": "process_tree_sum_rss",
            "rss_sample_interval_seconds": RSS_SAMPLE_INTERVAL_SECONDS,
        }
        report["git_commit"] = None
        report["git_dirty"] = None
        report["environment"] = {}
        report["configs"] = []
        report["overall"] = {}

        try:
            timeout_sec = _parse_timeout(args.timeout_sec)
            if not isinstance(args.module_path, str) or not os.path.exists(args.module_path):
                raise _InputError(f"--module-path does not exist: {args.module_path!r}")

            config_dir = Path(args.config_dir)
            _validate_config_dir(config_dir)
            _validate_report_path(report_path)

            result_dir_given = args.result_dir is not None
            if result_dir_given:
                # symlink検出のため resolve() 前に検証する。
                result_dir = Path(args.result_dir)
                _validate_result_dir(result_dir)
            else:
                result_dir = None  # type: ignore[assignment]

            commit, dirty, git_error = _collect_git_provenance()
            report["git_commit"] = commit
            report["git_dirty"] = dirty
            if git_error is not None:
                raise _InputError(f"git provenance unavailable: {git_error}")

            limits = _read_container_limits()
            report["environment"] = limits.as_report_dict()
            if not limits.verified:
                raise _InputError(f"container limits not verified: {limits.error}")

            discovered = _discover_configs(config_dir)
            discovered_stems = [p.stem for p in discovered]
            report["overall"]["config_stems"] = discovered_stems

            configs_by_stem: dict[str, dict[str, Any]] = {}
            for path in discovered:
                configs_by_stem[path.stem] = _load_config_json(path)

            if not result_dir_given:
                temp_result_dir = tempfile.mkdtemp(prefix="bench_policy_result_")
                result_dir = Path(temp_result_dir)
            report["result_dir"] = str(result_dir)

            sampler = _RssSampler(root_pid=os.getpid(), discovered_stems=set(discovered_stems))
            sampler.start()
            try:
                import scripts.check_telemetry as check_telemetry
                import scripts.run_suite as run_suite

                suite_code = run_suite.run_suite(
                    str(config_dir),
                    args.module_path,
                    result_dir=str(result_dir),
                    timeout_sec=timeout_sec,
                )
            finally:
                sampler.stop_join()

            if suite_code != 0:
                raise _InputError(f"run_suite.run_suite returned non-zero exit code: {suite_code}")

            report["overall"]["process_tree_observed"] = sampler.observed
            report["overall"]["process_tree_rss_peak_bytes"] = sampler.peak_bytes
            report["overall"]["rss_unattributed_observed"] = sampler.unattributed_observed
            if sampler.error is not None:
                raise _InputError(sampler.error)
            if not sampler.observed:
                raise _InputError("process tree was never observed during the suite run")

            _verify_suite_artifacts_present(result_dir)
            suite_rows = _load_suite_results(result_dir)
            expected_task_count = sum(len(tasks) for tasks in configs_by_stem.values())
            _verify_identity(discovered_stems, suite_rows, expected_task_count)

            rows_by_stem: dict[str, list[dict[str, Any]]] = {stem: [] for stem in discovered_stems}
            for row in suite_rows:
                rows_by_stem[row["config"]].append(row)

            all_first_values: list[float] = []
            all_steady_values: list[float] = []
            optimize_task_values: list[float] = []
            config_reports: list[dict[str, Any]] = []

            stem_errors: list[str] = []
            for stem in discovered_stems:
                try:
                    telemetry = _check_config_telemetry(check_telemetry.main, result_dir, stem)
                except _TelemetryCheckFailed as exc:
                    telemetry = exc.telemetry
                    stem_errors.append(str(exc))
                all_first_values.extend(telemetry.first_step_values)
                all_steady_values.extend(telemetry.steady_values)

                tasks_json = configs_by_stem[stem]
                rows_for_stem = {row.get("task_id"): row for row in rows_by_stem[stem]}
                task_entries: list[dict[str, Any]] = []
                for task_id, task_cfg in tasks_json.items():
                    entry, task_error = _evaluate_task(task_cfg, rows_for_stem.get(task_id))
                    task_entries.append(entry)
                    if task_error is not None:
                        stem_errors.append(task_error)
                    if entry["optimization_time_seconds"] is not None:
                        optimize_task_values.append(entry["optimization_time_seconds"])

                n_validator_ng = sum(
                    (row.get("n_validator_ng") or 0)
                    for row in rows_by_stem[stem]
                    if row.get("status") == "success"
                )
                steady_percentiles = _percentiles_dict(telemetry.steady_values)
                config_reports.append(
                    {
                        "config": stem,
                        "telemetry_file": telemetry.telemetry_file,
                        "telemetry_check_exit_code": telemetry.telemetry_check_exit_code,
                        "policy_count": telemetry.policy_count,
                        "first_step_count": telemetry.first_step_count,
                        "steady_step_count": telemetry.steady_step_count,
                        "steady_percentiles_seconds": steady_percentiles,
                        "threshold_results": {
                            "steady_policy_p99": (
                                None
                                if steady_percentiles["p99"] is None
                                else steady_percentiles["p99"] <= POLICY_LIMIT_SECONDS
                            ),
                        },
                        "rss_peak_bytes": sampler.config_peak_bytes.get(stem, 0),
                        "rss_attribution": (
                            "attributed" if sampler.config_attributed.get(stem) else "unattributed"
                        ),
                        "n_validator_ng": n_validator_ng,
                        "tasks": task_entries,
                    }
                )

            report["configs"] = config_reports
            for message in stem_errors:
                if message not in errors:
                    errors.append(message)
            if not all_steady_values:
                errors.append("suite-wide steady telemetry population is empty")
            if not optimize_task_values:
                report["overall"]["optimize_task_count"] = 0
                report["overall"]["optimization_time_max_seconds"] = None
                errors.append("no optimize=true tasks were found in the suite")
            if errors:
                raise _InputError(errors[0])
            steady_percentiles_overall = _percentiles_dict(all_steady_values)
            first_step_max = max(all_first_values) if all_first_values else None
            rss_pass = sampler.peak_bytes <= RSS_LIMIT_BYTES
            first_step_pass = (
                first_step_max is not None and first_step_max <= FIRST_STEP_LIMIT_SECONDS
            )
            p99_pass = (
                steady_percentiles_overall["p99"] is not None
                and steady_percentiles_overall["p99"] <= POLICY_LIMIT_SECONDS
            )
            optimize_max = max(optimize_task_values)
            optimize_pass = optimize_max <= OPTIMIZE_LIMIT_SECONDS

            report["overall"]["first_step_max_seconds"] = first_step_max
            report["overall"]["steady_percentiles_seconds"] = steady_percentiles_overall
            report["overall"]["optimize_task_count"] = len(optimize_task_values)
            report["overall"]["optimization_time_max_seconds"] = optimize_max
            report["overall"]["threshold_results"] = {
                "first_step_max": first_step_pass,
                "steady_policy_p99": p99_pass,
                "process_tree_rss_peak": rss_pass,
                "optimization_time_max": optimize_pass,
            }
            report["diagnostics"] = {"errors": errors, "warnings": warnings}

            all_pass = rss_pass and first_step_pass and p99_pass and optimize_pass
            if all_pass:
                return _finalize(report, report_path, _OUTCOME_PASS, True, _EXIT_OK)
            return _finalize(report, report_path, _OUTCOME_THRESHOLD_EXCEEDED, False, _EXIT_THRESHOLD)

        except _InputError as exc:
            if str(exc) not in errors:
                errors.append(str(exc))
            report["diagnostics"] = {"errors": errors, "warnings": warnings}
            return _finalize(report, report_path, _OUTCOME_ERROR, False, _EXIT_INPUT_ERROR)
        except Exception as exc:  # noqa: BLE001 - 未想定の例外もexit 2へ正規化する（安全網）
            message = f"unexpected error: {exc}"
            if message not in errors:
                errors.append(message)
            report["diagnostics"] = {"errors": errors, "warnings": warnings}
            return _finalize(report, report_path, _OUTCOME_ERROR, False, _EXIT_INPUT_ERROR)
        finally:
            if temp_result_dir is not None:
                shutil.rmtree(temp_result_dir, ignore_errors=True)

    except KeyboardInterrupt:
        try:
            if sampler is not None:
                sampler.stop_join()
        except Exception:  # noqa: BLE001 - 割込み時cleanupは最善努力
            pass
        try:
            _reap_descendants(os.getpid())
        except Exception:  # noqa: BLE001
            pass
        if temp_result_dir is not None:
            shutil.rmtree(temp_result_dir, ignore_errors=True)
        report["diagnostics"] = {"errors": errors, "warnings": warnings}
        _finalize(report, report_path, _OUTCOME_INTERRUPTED, False, _EXIT_INTERRUPTED)
        return _EXIT_INTERRUPTED


if __name__ == "__main__":
    raise SystemExit(main())
