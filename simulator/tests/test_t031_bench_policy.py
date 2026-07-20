"""T-031 ``scripts.bench_policy`` public contract tests (specification v1.25 §5.4).

The production module is intentionally absent while this test commit is prepared.  Every test
performs a delayed import through :class:`BenchEnvironment`; collection must therefore succeed,
while execution is RED with an uncaught ``ModuleNotFoundError`` until T-031 is implemented.

Only the public ``main(argv) -> int`` API, the real ``python -m`` boundary, documented report
data, generated artifacts, and public dependency calls are asserted.  No production helper,
private name, or source text is inspected.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path

import pytest

from tests.fixtures.t031.cli_harness import CliHarness
from tests.fixtures.t031.harness import (
    FIRST_STEP_LIMIT_SECONDS,
    MEMORY_LIMIT_BYTES,
    OPTIMIZE_LIMIT_SECONDS,
    POLICY_LIMIT_SECONDS,
    RSS_LIMIT_BYTES,
    AtomicReplaceSpy,
    BenchEnvironment,
    ProcProcess,
    normalize_report,
    suite_row,
    task_config,
    telemetry_row,
)


SIMULATOR_ROOT = Path(__file__).resolve().parent.parent
LOCAL_SUITE = SIMULATOR_ROOT / "configs" / "local_suite"
VALID_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _environment(tmp_path: Path, monkeypatch, name: str = "case") -> BenchEnvironment:
    root = tmp_path / name
    root.mkdir()
    return BenchEnvironment(root, monkeypatch)


def _import_bench_policy(env: BenchEnvironment):
    """Per-test delayed import; ModuleNotFoundError deliberately remains uncaught."""
    return env.import_bench_policy()


def _call_main(env: BenchEnvironment, argv: list[str] | None = None) -> int:
    bench_policy = _import_bench_policy(env)
    return bench_policy.main(env.argv if argv is None else argv)


def _run(env: BenchEnvironment, argv: list[str] | None = None) -> tuple[int, dict]:
    code = _call_main(env, argv)
    report = env.read_report() if env.report_path.is_file() else {}
    return code, report


def _config_report(report: dict, stem: str = "custom-alpha") -> dict:
    matches = [entry for entry in report["configs"] if entry["config"] == stem]
    assert len(matches) == 1
    return matches[0]


def _diagnostic_text(report: dict) -> str:
    return json.dumps(report.get("diagnostics", {}), ensure_ascii=False).lower()


def _replace_option(argv: list[str], option: str, value: str) -> list[str]:
    changed = list(argv)
    if option in changed:
        changed[changed.index(option) + 1] = value
    else:
        changed.extend([option, value])
    return changed


def _without_option(argv: list[str], option: str) -> list[str]:
    changed = list(argv)
    index = changed.index(option)
    del changed[index : index + 2]
    return changed


def _set_stems(env: BenchEnvironment, stems: list[str]) -> None:
    for path in env.config_dir.iterdir():
        if path.suffix == ".json":
            path.unlink()
    env.config_stems = list(stems)
    env.configs = {}
    env.telemetry = {}
    env.suite_rows = []
    for index, stem in enumerate(stems):
        task_id, task = task_config(task_id=f"{index:03d}", optimize=index == 0)
        env.configs[stem] = {task_id: task}
        (env.config_dir / f"{stem}.json").write_text(json.dumps({task_id: task}))
        env.telemetry[stem] = [telemetry_row(0, 5.0 + index / 10), telemetry_row(1, 1.0 + index)]
        env.suite_rows.append(
            suite_row(
                stem,
                task_id,
                optimization_time=12.5 if index == 0 else None,
            )
        )
    env.suite_rows.sort(key=lambda row: row["config"])


def _bench_process(rss_bytes: int = 1024) -> ProcProcess:
    return ProcProcess(os.getpid(), os.getppid(), 10, rss_bytes, name="bench")


# =============================================================================
# API/CLI — main and module entry point are separate public boundaries (15 items)
# =============================================================================


@pytest.mark.parametrize(
    "missing",
    ["--config-dir", "--module-path", "--report"],
    ids=["config-dir", "module-path", "report"],
)
def test_api_required_argument_error_is_builtin_int_without_system_exit(tmp_path, monkeypatch, missing):
    """Each required flag is normalized by main itself to the built-in integer exit code 2."""
    env = _environment(tmp_path, monkeypatch)
    code = _call_main(env, _without_option(env.argv, missing))
    assert type(code) is int
    assert code == 2


@pytest.mark.parametrize(
    "bad_value",
    ["not-a-number", "True", "nan", "inf", "0", "-0.001"],
    ids=["nonnumeric", "bool-like", "nan", "infinity", "zero", "negative"],
)
def test_api_timeout_rejects_nonfinite_nonpositive_and_bool_like_values(
    tmp_path, monkeypatch, bad_value
):
    """The timeout parser accepts only finite numeric seconds strictly greater than zero."""
    env = _environment(tmp_path, monkeypatch)
    code = _call_main(env, env.argv + ["--timeout-sec", bad_value])
    assert type(code) is int
    assert code == 2
    assert env.run_suite_calls == []


def test_api_timeout_default_is_600_seconds_at_run_suite_boundary(tmp_path, monkeypatch):
    """The default is 600.0 and a valid explicit timeout reaches the one suite call unchanged."""
    observed = []
    for name, extra, expected in (
        ("default", [], 600.0),
        ("explicit", ["--timeout-sec", "123.25"], 123.25),
    ):
        with monkeypatch.context() as scoped:
            env = _environment(tmp_path, scoped, name)
            code, _ = _run(env, env.argv + extra)
            assert code == 0
            assert len(env.run_suite_calls) == 1
            call = env.run_suite_calls[0]
            timeout = call["kwargs"].get(
                "timeout_sec", call["args"][4] if len(call["args"]) >= 5 else 600.0
            )
            observed.append(timeout)
    assert observed == [600.0, 123.25]


def test_api_main_returns_strict_int_for_success_threshold_failure_and_interrupt(tmp_path, monkeypatch):
    """main returns strict ints; steady/first-step limits and interrupt map to 0/1/130."""
    observed = []
    cases = (
        ("ok", "ok"),
        ("first-equal", "first-equal"),
        ("steady-slow", "steady-slow"),
        ("first-over", "first-over"),
        ("interrupt", "interrupt"),
    )
    for name, mode in cases:
        with monkeypatch.context() as scoped:
            env = _environment(tmp_path, scoped, name)
            if mode == "first-equal":
                env.telemetry["custom-alpha"] = [
                    telemetry_row(0, FIRST_STEP_LIMIT_SECONDS),
                    telemetry_row(1, 1.0),
                ]
            elif mode == "steady-slow":
                env.telemetry["custom-alpha"] = [
                    telemetry_row(0, 5.5),
                    telemetry_row(1, POLICY_LIMIT_SECONDS + 0.001),
                ]
            elif mode == "first-over":
                env.telemetry["custom-alpha"] = [
                    telemetry_row(0, math.nextafter(FIRST_STEP_LIMIT_SECONDS, math.inf)),
                    telemetry_row(1, 1.0),
                ]
            elif mode == "interrupt":
                env.run_suite_exception = KeyboardInterrupt()
            value = _call_main(env)
            observed.append(value)
            assert type(value) is int
            if mode in {"first-equal", "first-over"}:
                report = env.read_report()
                expected_pass = mode == "first-equal"
                assert report["overall"]["threshold_results"]["first_step_max"] is expected_pass
                if expected_pass:
                    assert report["overall"]["first_step_max_seconds"] == FIRST_STEP_LIMIT_SECONDS
                else:
                    assert report["overall"]["first_step_max_seconds"] > FIRST_STEP_LIMIT_SECONDS
    assert observed == [0, 0, 1, 1, 130]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("ok", 0), ("threshold", 1), ("input", 2), ("interrupt", 130)],
    ids=["exit-0", "exit-1", "exit-2", "exit-130"],
)
def test_cli_python_m_reports_process_exit_codes(tmp_path, monkeypatch, mode, expected):
    """The real ``python -m scripts.bench_policy`` boundary maps outcomes to process status."""
    # This explicit delayed import makes the item RED for the intended common cause today; once
    # implemented, the separate subprocess assertion below exercises the actual module entry.
    env = _environment(tmp_path, monkeypatch, "parent-import")
    _import_bench_policy(env)
    cli_root = tmp_path / f"cli-{mode}"
    cli_root.mkdir()
    completed = CliHarness(cli_root, SIMULATOR_ROOT).run(mode)
    assert completed.returncode == expected, completed.stderr


# =============================================================================
# DISCOVERY/RUN_SUITE — general suite logic and committed local-suite integration (6 items)
# =============================================================================


def test_discovery_sorts_all_c_json_ignores_unrelated_and_calls_suite_once(tmp_path, monkeypatch):
    """Arbitrary c*.json names are reported once in filename order through one suite invocation."""
    env = _environment(tmp_path, monkeypatch)
    _set_stems(env, ["czulu", "calpha", "cmiddle"])
    (env.config_dir / "README.txt").write_text("ignore")
    (env.config_dir / "manifest.json").write_text("{}")
    (env.config_dir / "x-config.json").write_text("{}")
    code, report = _run(env)
    assert code == 0
    assert len(env.run_suite_calls) == 1
    assert report["overall"]["config_stems"] == ["calpha", "cmiddle", "czulu"]
    assert [entry["config"] for entry in report["configs"]] == ["calpha", "cmiddle", "czulu"]


def test_discovery_zero_and_invalid_suite_execution_are_input_errors(tmp_path, monkeypatch):
    """Zero configs and every documented suite execution/artifact failure return exit 2."""
    with monkeypatch.context() as scoped:
        env = _environment(tmp_path, scoped, "zero-config")
        for path in env.config_dir.glob("c*.json"):
            path.unlink()
        (env.config_dir / "manifest.json").write_text("{}")
        code = _call_main(env)
        assert code == 2
        assert env.run_suite_calls == []

    invalid_modes = (
        "run-suite-nonzero",
        "suite-results-missing",
        "suite-results-unparseable",
        "task-zero",
        "format-error",
        "exec-exception",
        "policy-timeout-not-excluded",
    )
    outcomes = []
    for index, mode in enumerate(invalid_modes):
        with monkeypatch.context() as scoped:
            env = _environment(tmp_path, scoped, f"invalid-suite-{index}")
            if mode == "run-suite-nonzero":
                env.suite_code = 1
            elif mode in {"suite-results-missing", "suite-results-unparseable"}:

                def corrupt(current, selected=mode):
                    path = current.result_dir / "suite_results.jsonl"
                    if selected == "suite-results-missing":
                        path.unlink()
                    else:
                        path.write_text("{not-json}\n")

                env.before_suite_return = corrupt
            elif mode == "task-zero":
                env.suite_rows = []
            elif mode == "format-error":
                env.suite_rows = [suite_row("custom-alpha", status="format_error")]
            elif mode == "exec-exception":
                env.suite_rows = [suite_row("custom-alpha", status="exec_exception")]
            else:
                env.suite_rows[0]["policy_time"] = 8.0  # config policy_timeout is also 8.0
            code, report = _run(env)
            outcomes.append(code)
            if report:
                assert report["overall_pass"] is False
            assert len(env.run_suite_calls) == 1
    assert outcomes == [2] * len(invalid_modes)


@pytest.mark.parametrize(
    "mismatch",
    ["duplicate", "set", "order"],
    ids=["duplicate", "set-mismatch", "order-mismatch"],
)
def test_discovery_rejects_suite_artifact_identity_mismatches(tmp_path, monkeypatch, mismatch):
    """Duplicate, wrong-set, and wrong-order result rows make the measurement invalid."""
    env = _environment(tmp_path, monkeypatch)
    _set_stems(env, ["c-any-a", "c-any-b"])
    if mismatch == "duplicate":
        env.suite_rows.append(dict(env.suite_rows[0]))
    elif mismatch == "set":
        env.suite_rows[1]["config"] = "c-not-discovered"
    else:
        env.suite_rows.reverse()
    code, report = _run(env)
    assert code == 2
    assert len(env.run_suite_calls) == 1
    assert report.get("overall_pass") is False


def test_discovery_real_local_suite_is_the_only_c01_to_c05_specific_contract(tmp_path, monkeypatch):
    """The committed integration directory contains and reports c01..c05 exactly once."""
    env = _environment(tmp_path, monkeypatch)
    expected = ["c01", "c02", "c03", "c04", "c05"]
    env.config_dir = LOCAL_SUITE
    env.config_stems = expected
    env.telemetry = {
        stem: [telemetry_row(0, 5.0), telemetry_row(1, 1.0)] for stem in expected
    }
    env.suite_rows = [
        suite_row(stem, optimization_time=12.0 if stem in {"c01", "c03"} else None)
        for stem in expected
    ]
    code, report = _run(env)
    assert code == 0
    assert report["overall"]["config_stems"] == expected
    assert [entry["config"] for entry in report["configs"]] == expected
    assert len(env.run_suite_calls) == 1


# =============================================================================
# TELEMETRY — files, checker, population, and Agent segment continuity (12 items)
# =============================================================================


@pytest.mark.parametrize(
    "invalid_kind",
    ["missing", "multiple", "empty", "directory", "symlink"],
    ids=["zero-files", "multiple-files", "empty", "nonregular", "symlink"],
)
def test_telemetry_requires_exactly_one_nonempty_regular_nonsymlink_jsonl(
    tmp_path, monkeypatch, invalid_kind
):
    """Every config telemetry directory has one and only one valid physical JSONL file."""
    env = _environment(tmp_path, monkeypatch)

    def corrupt(current: BenchEnvironment) -> None:
        directory = current.result_dir / "custom-alpha" / "telemetry"
        path = directory / "custom-alpha-run.jsonl"
        if invalid_kind == "missing":
            path.unlink()
        elif invalid_kind == "multiple":
            (directory / "second.jsonl").write_text(path.read_text())
        elif invalid_kind == "empty":
            path.write_text("")
        elif invalid_kind == "directory":
            path.unlink()
            path.mkdir()
        else:
            target = directory / "target.txt"
            target.write_text(path.read_text())
            path.unlink()
            path.symlink_to(target)

    env.before_suite_return = corrupt
    code, report = _run(env)
    assert code == 2
    assert report.get("overall_pass") is False


def test_telemetry_calls_public_checker_with_single_path_list(tmp_path, monkeypatch):
    """Each telemetry file is accepted only through check_telemetry.main([path])."""
    env = _environment(tmp_path, monkeypatch)
    code, report = _run(env)
    assert code == 0
    assert env.checker_calls == [
        [str(env.result_dir / "custom-alpha" / "telemetry" / "custom-alpha-run.jsonl")]
    ]
    assert _config_report(report)["telemetry_check_exit_code"] == 0


def test_telemetry_checker_nonzero_is_measurement_error(tmp_path, monkeypatch):
    """A checker error outranks an otherwise valid threshold failure (2 > 1 > 0)."""
    env = _environment(tmp_path, monkeypatch)
    env.checker_code = 1
    env.telemetry["custom-alpha"] = [
        telemetry_row(0, 5.5),
        telemetry_row(1, POLICY_LIMIT_SECONDS + 1.0),
    ]
    code, report = _run(env)
    assert code == 2
    assert len(env.checker_calls) == 1
    assert _config_report(report)["telemetry_check_exit_code"] == 1


def test_telemetry_includes_normal_emergency_and_every_agent_first_step(tmp_path, monkeypatch):
    """Emergency rows remain in the population and repeated Agent step-zero starts are valid."""
    env = _environment(tmp_path, monkeypatch)
    env.telemetry["custom-alpha"] = [
        telemetry_row(0, 5.0),
        telemetry_row(1, 1.0),
        telemetry_row(0, 5.5, emergency=True),
        telemetry_row(1, 2.0, emergency=True),
        telemetry_row(2, 3.0),
    ]
    code, report = _run(env)
    config = _config_report(report)
    assert code == 0
    assert (config["policy_count"], config["first_step_count"], config["steady_step_count"]) == (
        5,
        2,
        3,
    )
    assert report["overall"]["first_step_max_seconds"] == 5.5
    assert report["overall"]["steady_percentiles_seconds"]["p50"] == 2.0


@pytest.mark.parametrize(
    "steps",
    [[(0, True), (2, False)], [(0, True), (1, False), (1, False)], [(0, True), (2, False), (1, False)]],
    ids=["missing", "duplicate", "reverse"],
)
def test_telemetry_rejects_gap_duplicate_and_reverse_within_agent_segment(
    tmp_path, monkeypatch, steps
):
    """Within each Agent segment, recorded steps must be exactly 0,1,2,... in JSONL order."""
    env = _environment(tmp_path, monkeypatch)
    env.telemetry["custom-alpha"] = [
        telemetry_row(step, 1.0 + index, first_step=first)
        for index, (step, first) in enumerate(steps)
    ]
    code, report = _run(env)
    assert code == 2
    assert report.get("overall_pass") is False


def test_telemetry_null_config_steady_stats_but_requires_suite_steady_and_each_config_first(
    tmp_path, monkeypatch
):
    """Config-level N/A is null; suite steady zero and per-config first-step zero are exit 2."""
    with monkeypatch.context() as scoped:
        env = _environment(tmp_path, scoped, "config-na")
        _set_stems(env, ["c-no-steady", "c-with-steady"])
        env.telemetry["c-no-steady"] = [telemetry_row(0, 5.0)]
        code, report = _run(env)
        config = _config_report(report, "c-no-steady")
        assert code == 0
        assert config["steady_percentiles_seconds"] == {"p50": None, "p95": None, "p99": None}
        assert config["threshold_results"]["steady_policy_p99"] is None

    for name, rows in (
        ("suite-zero", [telemetry_row(0, 5.0)]),
        ("first-zero", [telemetry_row(1, 1.0, first_step=False)]),
        ("bad-first-step", [telemetry_row(0, 5.0), telemetry_row(1, 1.0, first_step=True)]),
    ):
        with monkeypatch.context() as scoped:
            env = _environment(tmp_path, scoped, name)
            env.telemetry["custom-alpha"] = rows
            code, report = _run(env)
            assert code == 2
            assert report.get("overall_pass") is False


# =============================================================================
# PERCENTILE — external-dependency-free linear type 7 and exact thresholds (7 items)
# =============================================================================


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([2.5], {"p50": 2.5, "p95": 2.5, "p99": 2.5}),
        ([1.0, 3.0], {"p50": 2.0, "p95": 2.9, "p99": 2.98}),
        ([1.0, 2.0, 4.0, 8.0], {"p50": 3.0, "p95": 7.399999999999999, "p99": 7.879999999999999}),
        ([8.0, 1.0, 4.0, 2.0], {"p50": 3.0, "p95": 7.399999999999999, "p99": 7.879999999999999}),
    ],
    ids=["n-1", "n-2", "many", "input-order-independent"],
)
def test_percentile_linear_type7_exact_values(tmp_path, monkeypatch, values, expected):
    """p50/p95/p99 use sorted binary64 linear interpolation without an external package."""
    env = _environment(tmp_path, monkeypatch)
    rows = [telemetry_row(0, 5.0)]
    rows.extend(telemetry_row(index + 1, value, first_step=False) for index, value in enumerate(values))
    env.telemetry["custom-alpha"] = rows
    code, report = _run(env)
    assert code in {0, 1}  # datasets above 5s legitimately exercise the threshold independently
    actual = _config_report(report)["steady_percentiles_seconds"]
    assert actual == pytest.approx(expected, rel=0.0, abs=1e-15)


def test_percentile_rejects_bool_nonnumeric_nonfinite_and_negative_samples(tmp_path, monkeypatch):
    """Every invalid t_total type/value makes the whole measurement exit 2."""
    invalid = [True, "1.0", float("nan"), float("inf"), float("-inf"), -0.001]
    observed = []
    for index, value in enumerate(invalid):
        with monkeypatch.context() as scoped:
            env = _environment(tmp_path, scoped, f"invalid-{index}")
            env.telemetry["custom-alpha"] = [telemetry_row(0, 5.0), telemetry_row(1, value)]
            code, report = _run(env)
            observed.append(code)
            assert report.get("overall_pass") is False
    assert observed == [2] * len(invalid)


def test_percentile_exact_five_seconds_passes_unrounded(tmp_path, monkeypatch):
    """A suite p99 exactly equal to 5.0 seconds satisfies the inclusive threshold."""
    env = _environment(tmp_path, monkeypatch)
    env.telemetry["custom-alpha"] = [telemetry_row(0, 6.0), telemetry_row(1, 5.0)]
    code, report = _run(env)
    assert code == 0
    assert report["overall"]["steady_percentiles_seconds"]["p99"] == 5.0
    assert report["overall"]["threshold_results"]["steady_policy_p99"] is True


def test_percentile_value_above_five_fails_even_if_display_rounding_would_hide_it(tmp_path, monkeypatch):
    """Report and gate both retain the unrounded binary64 value above 5.0."""
    env = _environment(tmp_path, monkeypatch)
    value = math.nextafter(5.0, math.inf)
    env.telemetry["custom-alpha"] = [telemetry_row(0, 5.0), telemetry_row(1, value)]
    code, report = _run(env)
    stored = report["overall"]["steady_percentiles_seconds"]["p99"]
    assert code == 1
    assert stored == value and stored > 5.0
    assert report["overall"]["threshold_results"]["steady_policy_p99"] is False


# =============================================================================
# RSS — synchronized fake process tree sampling (8 items)
# =============================================================================


def test_rss_sums_all_descendants_excludes_bench_and_records_contract_interval(tmp_path, monkeypatch):
    """One sample includes run_local/Runner/Agent/tracker and an unattributed descendant."""
    env = _environment(tmp_path, monkeypatch)
    direct = 32 * 1024**2
    env.linux.write_processes(
        [
            _bench_process(99 * 1024**2),
            ProcProcess(
                42001,
                os.getpid(),
                11,
                256 * 1024**2,
                name="run_local",
                cmdline=(
                    "python\x00-m\x00scripts.run_local\x00--config-path\x00"
                    f"{env.config_dir / 'custom-alpha.json'}\x00"
                ),
            ),
            ProcProcess(42002, 42001, 12, 128 * 1024**2, name="Runner"),
            ProcProcess(42003, 42002, 13, 64 * 1024**2, name="Agent"),
            ProcProcess(42004, 42001, 14, 8 * 1024**2, name="resource_tracker"),
            ProcProcess(42005, os.getpid(), 15, direct, name="unattributed"),
        ]
    )
    code, report = _run(env)
    expected = (256 + 128 + 64 + 8 + 32) * 1024**2
    assert code == 0
    assert report["overall"]["process_tree_rss_peak_bytes"] == expected
    config = _config_report(report)
    assert config["rss_peak_bytes"] == (256 + 128 + 64 + 8) * 1024**2
    assert config["rss_attribution"] == "attributed"
    assert report["overall"]["rss_unattributed_observed"] is True
    assert report["measurement_contract"]["rss_sample_interval_seconds"] == 0.02
    assert report["measurement_contract"]["rss_measurement"] == "process_tree_sum_rss"


def test_rss_exact_limit_passes_and_one_byte_over_fails(tmp_path, monkeypatch):
    """RSS compares exact byte totals inclusively at 10 GiB with no display rounding."""
    observed = []
    for name, rss, expected in (
        ("equal", RSS_LIMIT_BYTES, 0),
        ("over", RSS_LIMIT_BYTES + 1, 1),
    ):
        with monkeypatch.context() as scoped:
            env = _environment(tmp_path, scoped, name)
            env.linux.write_processes(
                [_bench_process(), ProcProcess(43001, os.getpid(), 20, rss, name="run_local")]
            )
            code, report = _run(env)
            observed.append(code)
            assert report["overall"]["process_tree_rss_peak_bytes"] == rss
            assert report["overall"]["threshold_results"]["process_tree_rss_peak"] is (expected == 0)
    assert observed == [0, 1]


def test_rss_rediscovers_dynamic_children_and_agent_restart_each_sample(tmp_path, monkeypatch):
    """A child created after sampling starts and a restarted Agent both affect the suite peak."""
    env = _environment(tmp_path, monkeypatch)
    env.linux.write_processes(
        [_bench_process(), ProcProcess(44001, os.getpid(), 30, 100 * 1024**2, name="run_local")]
    )
    synchronization = {"second_sample": False}

    def restart(current: BenchEnvironment) -> None:
        status_reads = current.linux.status_read_count
        current.linux.write_processes(
            [
                _bench_process(),
                ProcProcess(44001, os.getpid(), 30, 100 * 1024**2, name="run_local"),
                ProcProcess(44002, 44001, 31, 200 * 1024**2, name="Agent"),
                ProcProcess(44003, 44001, 32, 50 * 1024**2, name="dynamic-child"),
            ]
        )
        synchronization["second_sample"] = current.linux.wait_for_status_reads(status_reads + 1)

    env.before_suite_return = restart
    code, report = _run(env)
    assert synchronization["second_sample"] is True
    assert code == 0
    assert report["overall"]["process_tree_rss_peak_bytes"] == 350 * 1024**2


def test_rss_pid_reuse_uses_start_time_identity_without_double_counting(tmp_path, monkeypatch):
    """The same PID with a changed start time is a new process, never two live RSS entries."""
    env = _environment(tmp_path, monkeypatch)
    env.linux.write_processes(
        [_bench_process(), ProcProcess(45001, os.getpid(), 40, 100 * 1024**2, name="Agent")]
    )

    def reuse(current: BenchEnvironment) -> None:
        status_reads = current.linux.status_read_count
        current.linux.write_processes(
            [_bench_process(), ProcProcess(45001, os.getpid(), 41, 225 * 1024**2, name="Agent")]
        )
        assert current.linux.wait_for_status_reads(status_reads + 1)

    env.before_suite_return = reuse
    code, report = _run(env)
    assert code == 0
    assert report["overall"]["process_tree_rss_peak_bytes"] == 225 * 1024**2


def test_rss_normal_exit_enoent_race_is_ignored_after_a_valid_observation(tmp_path, monkeypatch):
    """A child disappearing between discovery and read does not invalidate earlier samples."""
    env = _environment(tmp_path, monkeypatch)
    env.linux.write_processes(
        [
            _bench_process(),
            ProcProcess(46001, os.getpid(), 50, 75 * 1024**2, name="run_local"),
            ProcProcess(46002, 46001, 51, 20 * 1024**2, name="finished-Agent"),
        ]
    )
    env.linux.errors["/proc/46002/status"] = FileNotFoundError("normal process-exit race")
    code, report = _run(env)
    assert code == 0
    assert report["overall"]["process_tree_rss_peak_bytes"] == 75 * 1024**2


def test_rss_permission_error_is_measurement_error_not_zero(tmp_path, monkeypatch):
    """Permission failure reading a live descendant RSS produces exit 2."""
    env = _environment(tmp_path, monkeypatch)
    env.linux.errors["/proc/41001/status"] = PermissionError("denied")
    code, report = _run(env)
    assert code == 2
    assert report.get("overall_pass") is False
    assert "rss" in _diagnostic_text(report) or "permission" in _diagnostic_text(report)


def test_rss_parse_and_sampler_exceptions_are_measurement_errors(tmp_path, monkeypatch):
    """Malformed proc records and unexpected sampler exceptions both return exit 2."""
    observed = []
    for name in ("parse", "sampler"):
        with monkeypatch.context() as scoped:
            env = _environment(tmp_path, scoped, name)
            if name == "parse":
                (env.linux.proc_root / "41001" / "stat").write_text("not proc stat\n")
            else:
                env.linux.errors["/proc"] = RuntimeError("sampler exploded")
            env.wait_for_descendant_status = False
            code, report = _run(env)
            observed.append(code)
            assert report.get("overall_pass") is False
    assert observed == [2, 2]


def test_rss_no_descendant_tree_observed_is_error_not_zero_peak(tmp_path, monkeypatch):
    """Seeing only the benchmark process never establishes a valid zero-byte measurement."""
    env = _environment(tmp_path, monkeypatch)
    env.linux.write_processes([_bench_process()])
    env.wait_for_descendant_status = False
    code, report = _run(env)
    assert code == 2
    assert report.get("overall_pass") is False
    assert report.get("overall", {}).get("process_tree_observed") is False


# =============================================================================
# CGROUP — deterministic v1/v2 limits and strict validity (7 items)
# =============================================================================


def test_cgroup_v2_records_quota_cpuset_memory_and_auxiliary_usage(tmp_path, monkeypatch):
    """A valid v2 hierarchy verifies limits and records current/peak as auxiliary metrics."""
    env = _environment(tmp_path, monkeypatch)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3, 4, 5})
    env.linux.write_cgroup_v2(
        quota="200000 100000", cpuset="0-3", memory_current="111", memory_peak="222"
    )
    code, report = _run(env)
    limits = report["environment"]
    assert code == 0
    assert limits["limits_source"] == "cgroup_v2"
    assert limits["effective_cpu_limit"] == 2.0
    assert limits["memory_limit_bytes"] == MEMORY_LIMIT_BYTES
    assert (limits["memory_current_bytes"], limits["memory_peak_bytes"]) == (111, 222)
    assert limits["sched_affinity_cpu_count"] == 6


def test_cgroup_v1_records_quota_cpuset_memory_and_auxiliary_usage(tmp_path, monkeypatch):
    """Equivalent cgroup v1 controller files satisfy the same public environment contract."""
    env = _environment(tmp_path, monkeypatch)
    env.linux.write_cgroup_v1(
        quota="150000",
        period="100000",
        cpuset="0-1",
        memory_limit=str(11 * 1024**3),
        memory_current="333",
        memory_peak="444",
    )
    code, report = _run(env)
    limits = report["environment"]
    assert code == 0
    assert limits["limits_source"] == "cgroup_v1"
    assert limits["effective_cpu_limit"] == 1.5
    assert limits["memory_limit_bytes"] == 11 * 1024**3
    assert (limits["memory_current_bytes"], limits["memory_peak_bytes"]) == (333, 444)


def test_cgroup_effective_cpu_is_minimum_positive_limit_and_stricter_limits_pass(
    tmp_path, monkeypatch
):
    """Quota 1.5 and cpuset 1 yield effective 1; limits stricter than recommendations pass."""
    env = _environment(tmp_path, monkeypatch)
    env.linux.write_cgroup_v2(
        quota="150000 100000", cpuset="7", memory_max=str(4 * 1024**3)
    )
    code, report = _run(env)
    limits = report["environment"]
    assert code == 0
    assert limits["cpu_quota_limit"] == 1.5
    assert limits["cpuset_cpu_limit"] == 1
    assert limits["effective_cpu_limit"] == 1.0
    assert limits["container_limits_verified"] is True


@pytest.mark.parametrize(
    ("quota", "cpuset", "expected_effective"),
    [
        ("400000 100000", "0-1", 2.0),  # quota=4, cpuset=2 -> min=2.0
        ("200000 100000", "0-15", 2.0),  # quota=2, cpuset=16 -> min=2.0
        ("max 100000", "0-1", 2.0),  # quota=unlimited, cpuset=2 -> cpusetのみ有効、2.0
    ],
    ids=["quota4-cpuset2", "quota2-cpuset16", "quota-unlimited-cpuset2"],
)
def test_cgroup_effective_cpu_is_min_of_available_finite_positive_axes(
    tmp_path, monkeypatch, quota, cpuset, expected_effective
):
    """有効CPU上限は利用可能な有限・正のhard limitの最小値。片方が緩くても他方が救済しない。"""
    env = _environment(tmp_path, monkeypatch)
    env.linux.write_cgroup_v2(quota=quota, cpuset=cpuset)
    code, report = _run(env)
    limits = report["environment"]
    assert code == 0
    assert limits["effective_cpu_limit"] == expected_effective
    assert limits["container_limits_verified"] is True


@pytest.mark.parametrize(
    "invalid_kind",
    [
        "cpu-over",
        "cpu-quota-and-cpuset-both-over",
        "cpu-no-positive-candidate",
        "memory-over",
        "unlimited",
        "invalid-or-unavailable",
    ],
    ids=[
        "cpu-over",
        "cpu-quota-and-cpuset-both-over",
        "cpu-no-positive-candidate",
        "memory-over",
        "unlimited",
        "invalid-and-unavailable",
    ],
)
def test_cgroup_invalid_or_unverifiable_limits_exit_2(tmp_path, monkeypatch, invalid_kind):
    """Over-limit, unlimited, malformed, and unavailable controller data are input errors."""
    modes = [invalid_kind]
    if invalid_kind == "invalid-or-unavailable":
        modes = ["invalid", "unavailable"]
    outcomes = []
    for index, mode in enumerate(modes):
        with monkeypatch.context() as scoped:
            env = _environment(tmp_path, scoped, f"{invalid_kind}-{index}")
            if mode == "cpu-over":
                # 有限・正のhard limitの最小値が2.0を超える：quota=2.00001, cpuset=4（両軸とも超過）。
                env.linux.write_cgroup_v2(quota="200001 100000", cpuset="0-3")
            elif mode == "cpu-quota-and-cpuset-both-over":
                # 有限・正の候補2件（quota=4, cpuset=4）の最小値も4で2.0を超える。
                env.linux.write_cgroup_v2(quota="400000 100000", cpuset="0-3")
            elif mode == "cpu-no-positive-candidate":
                # quota=unlimitedかつcpusetが空文字列＝有効な有限・正の候補が0件。
                # "0-63"のような有限値は「64 CPU制約」であり「制約なし」の表現として不適切なため使わない。
                env.linux.write_cgroup_v2(quota="max 100000", cpuset="")
            elif mode == "memory-over":
                env.linux.write_cgroup_v2(memory_max=str(MEMORY_LIMIT_BYTES + 1))
            elif mode == "unlimited":
                env.linux.write_cgroup_v2(quota="max 100000", memory_max="max")
            elif mode == "invalid":
                env.linux.write_cgroup_v2(quota="garbage", cpuset="bad-range", memory_max="NaN")
            else:
                env.linux.write_cgroup_v2()
                (env.linux.cgroup_root / "bench" / "cpu.max").unlink()
                scoped.setattr(os, "sched_getaffinity", lambda _pid: {0, 1})
            code, report = _run(env)
            outcomes.append(code)
            assert report.get("environment", {}).get("container_limits_verified") is False
    assert outcomes == [2] * len(modes)


# =============================================================================
# OPTIMIZE — target selection, values, nullability, and maximum (5 items)
# =============================================================================


def test_optimize_exact_170_overage_and_external_timeout_have_distinct_outcomes(tmp_path, monkeypatch):
    """170 passes, an ordinary overage is 1, and timeout-not-excludable is priority exit 2."""
    observed = []
    for name, value, expected in (
        ("equal", OPTIMIZE_LIMIT_SECONDS, 0),
        ("over", math.nextafter(OPTIMIZE_LIMIT_SECONDS, math.inf), 1),
        ("external-timeout", 180.0, 2),
    ):
        with monkeypatch.context() as scoped:
            env = _environment(tmp_path, scoped, name)
            env.suite_rows[0]["optimization_time"] = value
            code, report = _run(env)
            task = _config_report(report)["tasks"][0]
            observed.append(code)
            assert task["optimization_time_seconds"] == value
            assert task["optimization_threshold_pass"] is (expected == 0)
    assert observed == [0, 1, 2]


def test_optimize_rejects_missing_bool_nonnumeric_nonfinite_and_negative(tmp_path, monkeypatch):
    """Every optimize=true task needs a finite nonnegative non-bool numeric result."""
    invalid = [None, True, "12.0", float("nan"), float("inf"), -0.1]
    observed = []
    for index, value in enumerate(invalid):
        with monkeypatch.context() as scoped:
            env = _environment(tmp_path, scoped, f"invalid-{index}")
            env.suite_rows[0]["optimization_time"] = value
            code, report = _run(env)
            observed.append(code)
            assert report.get("overall_pass") is False
    assert observed == [2] * len(invalid)


def test_optimize_false_and_uninvoked_tasks_are_null_and_not_gated(tmp_path, monkeypatch):
    """Non-target task report entries retain null optimization value and threshold result."""
    env = _environment(tmp_path, monkeypatch)
    task_id, task = task_config(optimize=False)
    env.configs["custom-alpha"] = {task_id: task}
    (env.config_dir / "custom-alpha.json").write_text(json.dumps({task_id: task}))
    env.suite_rows[0]["optimization_time"] = None
    # Retain one target in a second config so the suite-level target count remains valid.
    _task_id, target = task_config()
    env.configs["c-target"] = {_task_id: target}
    (env.config_dir / "c-target.json").write_text(json.dumps({_task_id: target}))
    env.config_stems.append("c-target")
    env.telemetry["c-target"] = [telemetry_row(0, 5.0), telemetry_row(1, 1.0)]
    env.suite_rows.append(suite_row("c-target", optimization_time=10.0))
    env.suite_rows.sort(key=lambda row: row["config"])
    code, report = _run(env)
    task_report = _config_report(report)["tasks"][0]
    assert code == 0
    assert task_report["optimization_time_seconds"] is None
    assert task_report["optimization_threshold_pass"] is None


def test_optimize_suite_with_zero_target_tasks_is_input_error(tmp_path, monkeypatch):
    """A formal benchmark suite must contain at least one optimize=true task."""
    env = _environment(tmp_path, monkeypatch)
    task_id, task = task_config(optimize=False)
    (env.config_dir / "custom-alpha.json").write_text(json.dumps({task_id: task}))
    env.configs["custom-alpha"] = {task_id: task}
    env.suite_rows[0]["optimization_time"] = None
    code, report = _run(env)
    assert code == 2
    assert report["overall"]["optimize_task_count"] == 0
    assert report["overall"]["optimization_time_max_seconds"] is None


def test_optimize_records_each_task_and_global_maximum(tmp_path, monkeypatch):
    """Task-level optimize values and the maximum across all targets are both reported."""
    env = _environment(tmp_path, monkeypatch)
    first_id, first = task_config("000", optimize=True)
    second_id, second = task_config("001", optimize=True)
    env.configs["custom-alpha"] = {first_id: first, second_id: second}
    (env.config_dir / "custom-alpha.json").write_text(json.dumps(env.configs["custom-alpha"]))
    env.suite_rows = [
        suite_row("custom-alpha", "000", optimization_time=10.0, validator_ng=2),
        suite_row("custom-alpha", "001", optimization_time=25.0),
    ]
    code, report = _run(env)
    config = _config_report(report)
    assert code == 0
    assert [task["optimization_time_seconds"] for task in config["tasks"]] == [10.0, 25.0]
    assert report["overall"]["optimize_task_count"] == 2
    assert report["overall"]["optimization_time_max_seconds"] == 25.0
    assert config["n_validator_ng"] == 2  # validator NG is still a completed measurement


# =============================================================================
# GIT PROVENANCE — exact commands, dirty semantics, and fail-closed behavior (5 items)
# =============================================================================


def test_git_clean_or_detached_head_records_full_commit_and_strict_bool(tmp_path, monkeypatch):
    """A full rev-parse result works without a branch lookup and clean is built-in False."""
    env = _environment(tmp_path, monkeypatch)
    env.git.commit = VALID_COMMIT
    env.git.status = ""
    code, report = _run(env)
    assert code == 0
    assert report["git_commit"] == VALID_COMMIT
    assert type(report["git_dirty"]) is bool and report["git_dirty"] is False
    assert [call[1:] for call in env.git.calls] == [
        ["rev-parse", "HEAD"],
        ["status", "--porcelain=v1", "--untracked-files=all"],
    ]


@pytest.mark.parametrize(
    ("status", "expected"),
    [(" M tracked.py\n", True), ("?? untracked.txt\n", True), ("", False)],
    ids=["tracked-dirty", "untracked-dirty", "ignored-not-reported"],
)
def test_git_status_porcelain_controls_dirty_without_failing_benchmark(
    tmp_path, monkeypatch, status, expected
):
    """Tracked/untracked output is dirty; ignored files absent from porcelain remain clean."""
    env = _environment(tmp_path, monkeypatch)
    env.git.status = status
    code, report = _run(env)
    assert code == 0
    assert type(report["git_dirty"]) is bool
    assert report["git_dirty"] is expected
    assert report["overall_pass"] is True


def test_git_any_provenance_failure_nulls_both_fields_reports_error_and_exits_2(
    tmp_path, monkeypatch
):
    """Repository/git/rev-parse/status failures fail closed while preserving a report."""
    observed = []
    failures = ("outside", "missing", "rev-parse", "status")
    for index, failure in enumerate(failures):
        with monkeypatch.context() as scoped:
            env = _environment(tmp_path, scoped, f"git-{index}")
            env.git.fail = failure
            code, report = _run(env)
            observed.append(code)
            assert report["git_commit"] is None
            assert report["git_dirty"] is None
            assert report["outcome"] == "error"
            assert report["overall_pass"] is False
            assert "git" in _diagnostic_text(report)
            assert "unknown" not in env.report_path.read_text().lower()
    assert observed == [2] * len(failures)


# =============================================================================
# REPORT/DIRECTORY/HYGIENE — complete schema, atomic IO, cleanup (3 items)
# =============================================================================


def test_report_schema_measurement_contract_thresholds_and_deterministic_content(
    tmp_path, monkeypatch
):
    """The strict JSON report contains every fixed contract and stable ordering/value data."""
    reports = []
    report_allow_nan_flags = []
    for name in ("first", "second"):
        with monkeypatch.context() as scoped:
            env = _environment(tmp_path, scoped, name)
            _set_stems(env, ["cz", "ca"])
            original_dump = json.dump
            original_dumps = json.dumps

            def spy_dump(value, *args, **kwargs):
                if isinstance(value, dict) and value.get("schema_version") == "t031.bench_policy.v1":
                    report_allow_nan_flags.append(kwargs.get("allow_nan"))
                return original_dump(value, *args, **kwargs)

            def spy_dumps(value, *args, **kwargs):
                if isinstance(value, dict) and value.get("schema_version") == "t031.bench_policy.v1":
                    report_allow_nan_flags.append(kwargs.get("allow_nan"))
                return original_dumps(value, *args, **kwargs)

            scoped.setattr(json, "dump", spy_dump)
            scoped.setattr(json, "dumps", spy_dumps)
            code, report = _run(env)
            assert code == 0
            reports.append(report)
    first = reports[0]
    assert first["schema_version"] == "t031.bench_policy.v1"
    assert isinstance(first["outcome"], str) and first["outcome"] not in {"error", "interrupted"}
    assert first["overall_pass"] is True
    assert first["generated_at_utc"].endswith("Z")
    assert datetime.fromisoformat(first["generated_at_utc"].removesuffix("Z") + "+00:00").utcoffset() is not None
    assert first["config_dir"].endswith("/configs")
    assert first["module_path"] == "agents/heuristic/"
    assert isinstance(first["argv"], list)
    assert isinstance(first["python_version"], str) and isinstance(first["platform"], str)
    assert first["thresholds"] == {
        "steady_policy_p99_seconds": POLICY_LIMIT_SECONDS,
        "first_step_max_seconds": FIRST_STEP_LIMIT_SECONDS,
        "process_tree_rss_peak_bytes": RSS_LIMIT_BYTES,
        "optimization_time_max_seconds": OPTIMIZE_LIMIT_SECONDS,
    }
    assert first["measurement_contract"] | {
        "telemetry_completeness": "best_effort",
        "strict_completed_policy_count_available": False,
        "tail_loss_detectable": False,
        "writer_warning_observable": False,
        "percentile_method": "linear_type_7",
        "rss_measurement": "process_tree_sum_rss",
        "rss_sample_interval_seconds": 0.02,
    } == first["measurement_contract"]
    assert first["environment"]["required_cpu_limit"] == 2.0
    assert first["environment"]["required_memory_limit_bytes"] == MEMORY_LIMIT_BYTES
    assert [entry["config"] for entry in first["configs"]] == ["ca", "cz"]
    assert first["diagnostics"].keys() == {"errors", "warnings"}
    assert report_allow_nan_flags == [False, False]
    serialized = json.dumps(first, allow_nan=False)
    assert all(token not in serialized for token in ("NaN", "Infinity", "-Infinity"))
    assert normalize_report(reports[0]) == normalize_report(reports[1])

    error_diagnostics = []
    for name in ("error-order-a", "error-order-b"):
        with monkeypatch.context() as scoped:
            env = _environment(tmp_path, scoped, name)
            _set_stems(env, ["c-error-b", "c-error-a"])
            env.suite_rows[0].update(
                {"status": "format_error", "format_error": True, "exec_exception": False}
            )
            env.suite_rows[1].update(
                {"status": "exec_exception", "format_error": False, "exec_exception": True}
            )
            code, report = _run(env)
            assert code == 2
            error_diagnostics.append(report["diagnostics"])
    assert error_diagnostics[0] == error_diagnostics[1]


def test_report_uses_atomic_replace_write_failure_is_2_and_interrupt_report_is_best_effort(
    tmp_path, monkeypatch
):
    """Successful writes replace atomically; replace failure and KeyboardInterrupt have fixed outcomes."""
    with monkeypatch.context() as scoped:
        env = _environment(tmp_path, scoped, "atomic")
        env.report_path.write_text("old report must be replaced")
        replacement = AtomicReplaceSpy(os.replace)
        scoped.setattr(os, "replace", replacement)
        code, _ = _run(env)
        assert code == 0
        assert len(replacement.calls) == 1
        source, destination = replacement.calls[0]
        assert destination == env.report_path
        assert source.parent == destination.parent
        assert not source.exists()
        assert env.read_report()["schema_version"] == "t031.bench_policy.v1"

    with monkeypatch.context() as scoped:
        env = _environment(tmp_path, scoped, "write-failure")
        replacement = AtomicReplaceSpy(os.replace, fail=True)
        scoped.setattr(os, "replace", replacement)
        code = _call_main(env)
        assert code == 2
        assert len(replacement.calls) == 1

    with monkeypatch.context() as scoped:
        env = _environment(tmp_path, scoped, "interrupted")
        env.run_suite_exception = KeyboardInterrupt()
        code, report = _run(env)
        assert code == 130
        if report:
            assert report["outcome"] == "interrupted"
            assert report["overall_pass"] is False


def test_directory_rules_temporary_cleanup_symlinks_and_sentinels_preserve_hygiene(
    tmp_path, monkeypatch
):
    """Scratch paths are isolated, validated, retained/cleaned as specified, and never clobber sentinels."""
    sentinel_a = tmp_path / "existing-suite-results.jsonl"
    sentinel_b = tmp_path / "existing-report.json"
    sentinel_a.write_text("alpha sentinel\n")
    sentinel_b.write_text("beta sentinel\n")
    fixed_ns = 1_600_000_000_123_456_789
    os.utime(sentinel_a, ns=(fixed_ns, fixed_ns))
    os.utime(sentinel_b, ns=(fixed_ns + 1, fixed_ns + 1))
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (sentinel_a, sentinel_b)
    }

    with monkeypatch.context() as scoped:
        env = _environment(tmp_path, scoped, "specified")
        env.result_dir.mkdir()  # an explicitly supplied existing empty directory is valid
        code, _ = _run(env)
        assert code == 0
        assert env.result_dir.is_dir()
        assert (env.result_dir / "suite_results.jsonl").is_file()

    with monkeypatch.context() as scoped:
        env = _environment(tmp_path, scoped, "temporary")
        code, _ = _run(env, argv=_without_option(env.argv, "--result-dir"))
        temporary_result = Path(env.run_suite_calls[0]["kwargs"]["result_dir"])
        assert code == 0
        assert not temporary_result.exists()
        assert env.report_path.is_file()

    invalid_results = []
    invalid_kinds = (
        "nonempty-result",
        "result-file",
        "result-symlink",
        "config-dir-file",
        "config-dir-symlink",
        "config-file-directory",
        "config-symlink",
        "report-directory",
        "report-symlink",
    )
    for index, kind in enumerate(invalid_kinds):
        with monkeypatch.context() as scoped:
            env = _environment(tmp_path, scoped, f"invalid-path-{index}")
            if kind == "nonempty-result":
                env.result_dir.mkdir()
                (env.result_dir / "do-not-touch.txt").write_text("preserve")
            elif kind == "result-file":
                env.result_dir.write_text("not a directory")
            elif kind == "result-symlink":
                target = env.tmp_path / "real-result"
                target.mkdir()
                env.result_dir.symlink_to(target, target_is_directory=True)
            elif kind == "config-dir-file":
                real = env.tmp_path / "real-configs"
                env.config_dir.rename(real)
                env.config_dir.write_text("not a directory")
            elif kind == "config-dir-symlink":
                real = env.tmp_path / "real-configs"
                env.config_dir.rename(real)
                env.config_dir.symlink_to(real, target_is_directory=True)
            elif kind == "config-file-directory":
                config = env.config_dir / "custom-alpha.json"
                config.unlink()
                config.mkdir()
            elif kind == "config-symlink":
                config = env.config_dir / "custom-alpha.json"
                target = env.config_dir / "target.txt"
                target.write_text(config.read_text())
                config.unlink()
                config.symlink_to(target)
            elif kind == "report-directory":
                env.report_path.mkdir()
            else:  # report-symlink
                target = env.tmp_path / "real-report.json"
                target.write_text("preserve")
                env.report_path.symlink_to(target)
            code = _call_main(env)
            invalid_results.append(code)
            if kind == "nonempty-result":
                assert (env.result_dir / "do-not-touch.txt").read_text() == "preserve"
    assert invalid_results == [2] * len(invalid_kinds)
    after = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before}
    assert after == before
