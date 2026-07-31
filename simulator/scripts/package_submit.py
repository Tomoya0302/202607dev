"""T-029: reproducible, self-contained submission package builder (specification v1.26 §5.8).

The public API is :func:`main`.  Source trees are read without following links, transformed only
in an external staging directory, archived deterministically, and inspected from a safely
extracted copy in an isolated subprocess before one final atomic publish.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import io
import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import tokenize
import zipfile
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


CORE_FILES = (
    "__init__.py",
    "candidates.py",
    "constants.py",
    "container_space.py",
    "ems.py",
    "geometry.py",
    "heightmap.py",
    "masks.py",
    "order.py",
    "risk.py",
    "rollout_plan.py",
    "score.py",
    "stability.py",
    "state.py",
    "types.py",
    "watchdog.py",
)

ARCHIVE_ENTRIES = (
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
    "heuristic/packing_core/heightmap.py",
    "heuristic/packing_core/masks.py",
    "heuristic/packing_core/order.py",
    "heuristic/packing_core/risk.py",
    "heuristic/packing_core/rollout_plan.py",
    "heuristic/packing_core/score.py",
    "heuristic/packing_core/stability.py",
    "heuristic/packing_core/state.py",
    "heuristic/packing_core/types.py",
    "heuristic/packing_core/watchdog.py",
)

_AGENT_FILES = ("agent.py", "telemetry.py")
_OLD_PREFIXES = ("agents.heuristic", "src.packing_core")
_FIXED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_DIRECTORY_MODE = 0o040755
_REGULAR_MODE = 0o100644

_EXIT_OK = 0
_EXIT_CONTENT = 1
_EXIT_INPUT = 2
_EXIT_INTERRUPTED = 130


class _ContentError(Exception):
    """A readable source or generated archive violates the submission contract."""


class _InputError(Exception):
    """Arguments, paths, file types, permissions, or required inputs are invalid."""


@dataclass(frozen=True)
class _Paths:
    agent_dir: Path
    core_dir: Path
    output: Path
    force: bool
    output_parent_device: int
    output_parent_inode: int


@dataclass(frozen=True)
class _SourceRecord:
    path: Path
    device: int
    inode: int
    mode: int
    mtime_ns: int
    size: int


@dataclass(frozen=True)
class _AbstractValue:
    """Possible protected objects held directly or as literal iterable elements."""

    direct: frozenset[str] = frozenset()
    elements: frozenset[str] = frozenset()


class _PathNamedFile:
    """Delegate a binary file while exposing its path to ``ZipFile`` and test instrumentation."""

    def __init__(self, handle: object, path: Path) -> None:
        self._handle = handle
        self.name = str(path)

    def __getattr__(self, name: str) -> object:
        return getattr(self._handle, name)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and inspect the fixed T-029 heuristic submission archive.",
        allow_abbrev=False,
    )
    parser.add_argument("--agent-dir", required=True)
    parser.add_argument("--core-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def _absolute_path(value: str) -> Path:
    return Path(os.path.abspath(os.fsdecode(os.fspath(value))))


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _require_real_directory(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise _InputError(f"{label} does not exist: {path}") from exc
    except OSError as exc:
        raise _InputError(f"cannot inspect {label}: {path}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise _InputError(f"{label} must be a non-symlink directory: {path}")


def _ensure_output_parent(output: Path) -> None:
    parent = output.parent
    try:
        if os.path.lexists(parent):
            info = parent.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise _InputError(
                    f"--output parent must be a non-symlink directory: {parent}"
                )
        else:
            parent.mkdir(parents=True, exist_ok=False)
            info = parent.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise _InputError(
                    f"--output parent must be a non-symlink directory: {parent}"
                )
    except _InputError:
        raise
    except OSError as exc:
        raise _InputError(f"cannot prepare --output parent {parent}: {exc}") from exc


def _validate_output_state(output: Path, *, force: bool) -> None:
    try:
        exists = os.path.lexists(output)
    except OSError as exc:
        raise _InputError(f"cannot inspect --output {output}: {exc}") from exc
    if not exists:
        return
    try:
        info = output.lstat()
    except OSError as exc:
        raise _InputError(f"cannot inspect --output {output}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise _InputError(f"existing --output must be a non-symlink regular file: {output}")
    if not force:
        raise _InputError(f"--output already exists; use --force to replace it: {output}")


def _validate_paths(args: argparse.Namespace) -> _Paths:
    try:
        agent_dir = _absolute_path(args.agent_dir)
        core_dir = _absolute_path(args.core_dir)
        output = _absolute_path(args.output)
    except (TypeError, ValueError, OSError) as exc:
        raise _InputError(f"invalid path argument: {exc}") from exc

    if agent_dir.name != "heuristic":
        raise _InputError("--agent-dir basename must be 'heuristic'")
    if core_dir.name != "packing_core":
        raise _InputError("--core-dir basename must be 'packing_core'")
    if output.suffix != ".zip":
        raise _InputError("--output must have the .zip suffix")

    _require_real_directory(agent_dir, "--agent-dir")
    _require_real_directory(core_dir, "--core-dir")

    if _is_within(output, agent_dir) or _is_within(output, core_dir):
        raise _InputError("--output must not be a source directory or one of its descendants")

    real_output = Path(os.path.realpath(output))
    real_agent = Path(os.path.realpath(agent_dir))
    real_core = Path(os.path.realpath(core_dir))
    if _is_within(real_output, real_agent) or _is_within(real_output, real_core):
        raise _InputError("--output resolves inside a source directory")

    _ensure_output_parent(output)

    # Recheck after creation as well, so a concurrent parent change fails closed.
    real_output = Path(os.path.realpath(output))
    if _is_within(real_output, real_agent) or _is_within(real_output, real_core):
        raise _InputError("--output resolves inside a source directory")

    force = bool(args.force)
    _validate_output_state(output, force=force)
    parent_info = output.parent.lstat()
    return _Paths(
        agent_dir=agent_dir,
        core_dir=core_dir,
        output=output,
        force=force,
        output_parent_device=parent_info.st_dev,
        output_parent_inode=parent_info.st_ino,
    )


def _validate_output_parent(paths: _Paths) -> None:
    try:
        info = paths.output.parent.lstat()
    except OSError as exc:
        raise _InputError(f"cannot inspect --output parent {paths.output.parent}: {exc}") from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or info.st_dev != paths.output_parent_device
        or info.st_ino != paths.output_parent_inode
    ):
        raise _InputError("--output parent changed during packaging")


def _record(path: Path, info: os.stat_result) -> _SourceRecord:
    return _SourceRecord(
        path=path,
        device=info.st_dev,
        inode=info.st_ino,
        mode=info.st_mode,
        mtime_ns=info.st_mtime_ns,
        size=info.st_size,
    )


def _scan_source_tree(
    root: Path,
    required_names: tuple[str, ...],
    skip_dirs: frozenset[str] = frozenset(),
) -> dict[str, _SourceRecord]:
    """Inventory an entire tree with lstat, without following even ignored entries.

    ``skip_dirs`` names top-level subdirectories that are packaged separately (e.g. the
    nested ``packing_core`` tree when the agent and core share one directory); they are
    neither recursed into nor flagged as non-whitelisted content.
    """
    required = set(required_names)
    found: dict[str, _SourceRecord] = {}
    unknown: list[str] = []
    special: list[str] = []

    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC

    def visit(directory_fd: int, relative_directory: str, inside_cache: bool) -> None:
        try:
            with os.scandir(directory_fd) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name, reverse=True)
        except OSError as exc:
            label = root / relative_directory if relative_directory else root
            raise _InputError(f"cannot scan source directory {label}: {exc}") from exc

        for entry in entries:
            relative = (
                f"{relative_directory}/{entry.name}" if relative_directory else entry.name
            )
            path = root.joinpath(*PurePosixPath(relative).parts)
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise _InputError(f"cannot lstat source entry {path}: {exc}") from exc

            if stat.S_ISLNK(info.st_mode):
                special.append(relative)
                continue
            if stat.S_ISDIR(info.st_mode):
                # Top-level subtrees packaged separately (e.g. nested packing_core) are
                # skipped outright: not recursed into, not treated as unknown content.
                if relative_directory == "" and entry.name in skip_dirs:
                    continue
                child_inside_cache = inside_cache or entry.name == "__pycache__"
                if not child_inside_cache:
                    unknown.append(relative)
                child_fd = -1
                try:
                    child_fd = os.open(entry.name, directory_flags, dir_fd=directory_fd)
                    opened = os.fstat(child_fd)
                    if (
                        not stat.S_ISDIR(opened.st_mode)
                        or opened.st_dev != info.st_dev
                        or opened.st_ino != info.st_ino
                        or opened.st_mtime_ns != info.st_mtime_ns
                    ):
                        raise _InputError(f"source directory changed during scan: {path}")
                    visit(child_fd, relative, child_inside_cache)
                    finished = os.fstat(child_fd)
                    if (
                        not stat.S_ISDIR(finished.st_mode)
                        or finished.st_dev != info.st_dev
                        or finished.st_ino != info.st_ino
                        or finished.st_mtime_ns != info.st_mtime_ns
                    ):
                        raise _InputError(f"source directory changed during scan: {path}")
                except _InputError:
                    raise
                except OSError as exc:
                    raise _InputError(
                        f"cannot open source directory without following links {path}: {exc}"
                    ) from exc
                finally:
                    if child_fd >= 0:
                        os.close(child_fd)
                continue
            if stat.S_ISREG(info.st_mode):
                if relative in required and not inside_cache:
                    found[relative] = _record(path, info)
                elif inside_cache or entry.name.endswith((".pyc", ".pyo")):
                    continue
                else:
                    unknown.append(relative)
                continue
            special.append(relative)

    root_fd = -1
    try:
        root_before = root.lstat()
        root_fd = os.open(root, directory_flags)
        root_opened = os.fstat(root_fd)
        if (
            not stat.S_ISDIR(root_opened.st_mode)
            or root_opened.st_dev != root_before.st_dev
            or root_opened.st_ino != root_before.st_ino
            or root_opened.st_mtime_ns != root_before.st_mtime_ns
        ):
            raise _InputError(f"source directory changed during scan: {root}")
        visit(root_fd, "", False)
        root_finished = os.fstat(root_fd)
        if (
            not stat.S_ISDIR(root_finished.st_mode)
            or root_finished.st_dev != root_before.st_dev
            or root_finished.st_ino != root_before.st_ino
            or root_finished.st_mtime_ns != root_before.st_mtime_ns
        ):
            raise _InputError(f"source directory changed during scan: {root}")
    except _InputError:
        raise
    except OSError as exc:
        raise _InputError(f"cannot scan source directory {root}: {exc}") from exc
    finally:
        if root_fd >= 0:
            os.close(root_fd)

    missing = sorted(required.difference(found))
    if special:
        raise _InputError(f"source tree contains symlink or non-regular entry: {special[0]}")
    if missing:
        raise _InputError(f"required source file is missing or non-regular: {missing[0]}")
    if unknown:
        raise _ContentError(f"source tree contains non-whitelisted content: {unknown[0]}")
    return found


def _same_record(info: os.stat_result, record: _SourceRecord) -> bool:
    return (
        stat.S_ISREG(info.st_mode)
        and info.st_dev == record.device
        and info.st_ino == record.inode
        and info.st_mode == record.mode
        and info.st_mtime_ns == record.mtime_ns
        and info.st_size == record.size
    )


def _require_candidate(candidate: Path, record: _SourceRecord) -> None:
    try:
        info = candidate.lstat()
    except OSError as exc:
        raise _InputError(f"cannot inspect candidate ZIP {candidate}: {exc}") from exc
    if not _same_record(info, record):
        raise _InputError("candidate ZIP changed during packaging")


@contextmanager
def _open_candidate(
    candidate: Path, record: _SourceRecord
) -> Iterator[_PathNamedFile]:
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(candidate, flags)
        opened = os.fstat(descriptor)
        if not _same_record(opened, record):
            raise _InputError("candidate ZIP changed before inspection")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            yield _PathNamedFile(handle, candidate)
    except _InputError:
        raise
    except OSError as exc:
        raise _InputError(f"cannot open candidate ZIP {candidate}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _require_candidate(candidate, record)


def _read_source(record: _SourceRecord) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        before = record.path.lstat()
        if not _same_record(before, record):
            raise _InputError(f"source changed during packaging: {record.path}")
        descriptor = os.open(record.path, flags)
        try:
            opened = os.fstat(descriptor)
            if not _same_record(opened, record):
                raise _InputError(f"source changed during packaging: {record.path}")
            with os.fdopen(descriptor, "rb", closefd=True) as handle:
                descriptor = -1
                payload = handle.read()
                finished = os.fstat(handle.fileno())
            after = record.path.lstat()
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    except _InputError:
        raise
    except OSError as exc:
        raise _InputError(f"cannot read source file {record.path}: {exc}") from exc
    if not _same_record(finished, record) or not _same_record(after, record):
        raise _InputError(f"source changed during packaging: {record.path}")
    return payload


def _attribute_chain(node: ast.AST) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _attribute_chain(node.value)
        if parent is not None:
            return (*parent, node.attr)
    return None


def _branch_contains_import(statements: list[ast.stmt]) -> bool:
    return any(
        isinstance(descendant, (ast.Import, ast.ImportFrom))
        for statement in statements
        for descendant in ast.walk(statement)
    )


def _validate_forbidden_import_mechanisms(tree: ast.AST) -> None:
    mutating_methods = {
        "__delitem__",
        "__iadd__",
        "__imul__",
        "__ior__",
        "__setitem__",
        "append",
        "clear",
        "extend",
        "insert",
        "pop",
        "popitem",
        "remove",
        "reverse",
        "sort",
        "update",
        "setdefault",
    }
    protected = {"path", "modules"}
    mutators = {"path-mutator", "modules-mutator"}
    safe = _AbstractValue()
    function_registry: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    active_calls: set[str] = set()
    class_function_outers: list[dict[str, _AbstractValue]] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.Try, ast.TryStar)) and _branch_contains_import(
            node.body
        ) and any(_branch_contains_import(handler.body) for handler in node.handlers):
            raise _ContentError("try/except import fallback is forbidden")

    def merged_value(*values: _AbstractValue) -> _AbstractValue:
        return _AbstractValue(
            frozenset().union(*(value.direct for value in values)),
            frozenset().union(*(value.elements for value in values)),
        )

    def merged_environment(
        *environments: dict[str, _AbstractValue],
    ) -> dict[str, _AbstractValue]:
        result: dict[str, _AbstractValue] = {}
        for name in set().union(*(environment.keys() for environment in environments)):
            value = merged_value(
                *(environment.get(name, safe) for environment in environments)
            )
            if value != safe:
                result[name] = value
        return result

    def merged_optional_environment(
        *environments: dict[str, _AbstractValue] | None,
    ) -> dict[str, _AbstractValue] | None:
        live = [environment for environment in environments if environment is not None]
        return merged_environment(*live) if live else None

    def modules_escape(value: _AbstractValue) -> bool:
        return "modules" in value.direct or "modules" in value.elements

    def import_bindings(
        statements: list[ast.stmt],
    ) -> dict[str, _AbstractValue]:
        bindings: dict[str, _AbstractValue] = {}

        class ScopeImports(ast.NodeVisitor):
            def visit_Import(self, node: ast.Import) -> None:
                for alias in node.names:
                    bound = alias.asname or alias.name.split(".", 1)[0]
                    if alias.name == "sys":
                        bindings[bound] = _AbstractValue(frozenset({"sys"}))

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                if node.module != "sys":
                    return
                for alias in node.names:
                    if alias.name in {"*", "modules"}:
                        raise _ContentError("aliasing sys.modules is forbidden")
                    if alias.name == "path":
                        bindings[alias.asname or alias.name] = _AbstractValue(
                            frozenset({"path"})
                        )

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                return

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                return

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                return

            def visit_Lambda(self, node: ast.Lambda) -> None:
                return

        visitor = ScopeImports()
        for statement in statements:
            visitor.visit(statement)
        return bindings

    def local_bindings(statements: list[ast.stmt]) -> set[str]:
        names: set[str] = set()
        declared_global: set[str] = set()
        declared_nonlocal: set[str] = set()

        def add_target(target: ast.AST) -> None:
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for element in target.elts:
                    add_target(element)
            elif isinstance(target, ast.Starred):
                add_target(target.value)

        class Locals(ast.NodeVisitor):
            def visit_Assign(self, node: ast.Assign) -> None:
                for target in node.targets:
                    add_target(target)
                self.visit(node.value)

            def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
                add_target(node.target)
                if node.value is not None:
                    self.visit(node.value)

            def visit_AugAssign(self, node: ast.AugAssign) -> None:
                add_target(node.target)
                self.visit(node.value)

            def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
                add_target(node.target)
                self.visit(node.value)

            def visit_For(self, node: ast.For) -> None:
                add_target(node.target)
                self.generic_visit(node)

            def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
                add_target(node.target)
                self.generic_visit(node)

            def visit_With(self, node: ast.With) -> None:
                for item in node.items:
                    if item.optional_vars is not None:
                        add_target(item.optional_vars)
                self.generic_visit(node)

            def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
                self.visit_With(node)

            def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
                if node.name is not None:
                    names.add(node.name)
                self.generic_visit(node)

            def visit_Import(self, node: ast.Import) -> None:
                for alias in node.names:
                    names.add(alias.asname or alias.name.split(".", 1)[0])

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                for alias in node.names:
                    if alias.name != "*":
                        names.add(alias.asname or alias.name)

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                names.add(node.name)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                names.add(node.name)

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                names.add(node.name)

            def visit_Lambda(self, node: ast.Lambda) -> None:
                return

            def visit_Global(self, node: ast.Global) -> None:
                declared_global.update(node.names)

            def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
                declared_nonlocal.update(node.names)

        visitor = Locals()
        for statement in statements:
            visitor.visit(statement)
        return names.difference(declared_global, declared_nonlocal)

    def bind_name(
        name: str, value: _AbstractValue, environment: dict[str, _AbstractValue]
    ) -> None:
        if modules_escape(value):
            raise _ContentError("aliasing sys.modules is forbidden")
        if value == safe:
            environment.pop(name, None)
        else:
            environment[name] = value

    def value_of(node: ast.AST | None, environment: dict[str, _AbstractValue]) -> _AbstractValue:
        if node is None:
            return safe
        if isinstance(node, ast.Name):
            return environment.get(node.id, safe)
        if isinstance(node, ast.Starred):
            return value_of(node.value, environment)
        if isinstance(node, ast.Attribute):
            owner = value_of(node.value, environment)
            direct: set[str] = set()
            if "sys" in owner.direct and node.attr in protected:
                direct.add(node.attr)
            if owner.direct.intersection(protected) and node.attr in mutating_methods:
                direct.update(f"{kind}-mutator" for kind in owner.direct & protected)
            return _AbstractValue(frozenset(direct))
        if isinstance(node, ast.NamedExpr):
            value = value_of(node.value, environment)
            bind_target(node.target, value, environment, node.value)
            return value
        if isinstance(node, ast.IfExp):
            value_of(node.test, environment)
            body_environment = environment.copy()
            otherwise_environment = environment.copy()
            body_value = value_of(node.body, body_environment)
            otherwise_value = value_of(node.orelse, otherwise_environment)
            environment.clear()
            environment.update(
                merged_environment(body_environment, otherwise_environment)
            )
            return merged_value(body_value, otherwise_value)
        if isinstance(node, ast.BoolOp):
            continuing = environment.copy()
            values: list[_AbstractValue] = []
            branches: list[dict[str, _AbstractValue]] = []
            for value_node in node.values:
                values.append(value_of(value_node, continuing))
                branches.append(continuing.copy())
            environment.clear()
            environment.update(merged_environment(*branches))
            return merged_value(*values)
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            elements: set[str] = set()
            for element in node.elts:
                value = value_of(element, environment)
                if isinstance(element, ast.Starred):
                    elements.update(value.elements)
                else:
                    elements.update(value.direct)
                    if "modules" in value.elements:
                        elements.add("modules")
            return _AbstractValue(elements=frozenset(elements))
        if isinstance(node, ast.Dict):
            elements: set[str] = set()
            for key, item in zip(node.keys, node.values):
                if key is not None:
                    key_value = value_of(key, environment)
                    elements.update(key_value.direct)
                    if "modules" in key_value.elements:
                        elements.add("modules")
                item_value = value_of(item, environment)
                elements.update(item_value.direct)
                if "modules" in item_value.elements:
                    elements.add("modules")
            return _AbstractValue(elements=frozenset(elements))
        if isinstance(node, ast.Subscript):
            owner = value_of(node.value, environment)
            value_of(node.slice, environment)
            return _AbstractValue(owner.elements)
        if isinstance(node, ast.Call):
            function = value_of(node.func, environment)
            positional_arguments = [
                value_of(argument, environment) for argument in node.args
            ]
            keyword_arguments = {
                keyword.arg: value_of(keyword.value, environment)
                for keyword in node.keywords
                if keyword.arg is not None
            }
            arguments = [*positional_arguments, *keyword_arguments.values()]
            for keyword in node.keywords:
                if keyword.arg is None:
                    arguments.append(value_of(keyword.value, environment))
            if function.direct.intersection(mutators):
                raise _ContentError("sys.path/sys.modules mutation is forbidden")
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in {"setattr", "delattr"}
                and len(node.args) >= 2
                and "sys" in arguments[0].direct
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in protected
            ):
                raise _ContentError("sys.path/sys.modules mutation is forbidden")
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                attribute = node.args[1].value
                owner = arguments[0]
                if "sys" in owner.direct and attribute in protected:
                    return _AbstractValue(frozenset({attribute}))
                if owner.direct.intersection(protected) and attribute in mutating_methods:
                    return _AbstractValue(
                        frozenset(
                            f"{kind}-mutator" for kind in owner.direct & protected
                        )
                    )
            for marker in function.direct.intersection(function_registry):
                if marker in active_calls:
                    continue
                active_calls.add(marker)
                try:
                    analyze_function(
                        function_registry[marker],
                        environment,
                        positional_arguments,
                        keyword_arguments,
                        inspect_header=False,
                    )
                finally:
                    active_calls.remove(marker)
            return safe
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            nested = environment.copy()
            for generator in node.generators:
                iterable = value_of(generator.iter, nested)
                bind_target(
                    generator.target,
                    _AbstractValue(iterable.elements),
                    nested,
                    None,
                )
                for condition in generator.ifs:
                    value_of(condition, nested)
            element = value_of(node.elt, nested)
            elements = set(element.direct)
            if "modules" in element.elements:
                elements.add("modules")
            return _AbstractValue(elements=frozenset(elements))
        if isinstance(node, ast.DictComp):
            nested = environment.copy()
            for generator in node.generators:
                iterable = value_of(generator.iter, nested)
                bind_target(
                    generator.target,
                    _AbstractValue(iterable.elements),
                    nested,
                    None,
                )
                for condition in generator.ifs:
                    value_of(condition, nested)
            key = value_of(node.key, nested)
            item = value_of(node.value, nested)
            elements = set(key.direct | item.direct)
            if "modules" in key.elements or "modules" in item.elements:
                elements.add("modules")
            return _AbstractValue(elements=frozenset(elements))
        if isinstance(node, ast.Lambda):
            positional_defaults = [
                value_of(default, environment) for default in node.args.defaults
            ]
            keyword_defaults = [
                value_of(default, environment) if default is not None else safe
                for default in node.args.kw_defaults
            ]
            if any(
                modules_escape(default)
                for default in (*positional_defaults, *keyword_defaults)
            ):
                raise _ContentError("aliasing sys.modules is forbidden")
            nested = (
                class_function_outers[-1].copy()
                if class_function_outers
                else environment.copy()
            )
            argument_names = {
                argument.arg
                for argument in (
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                )
            }
            if node.args.vararg is not None:
                argument_names.add(node.args.vararg.arg)
            if node.args.kwarg is not None:
                argument_names.add(node.args.kwarg.arg)
            for name in argument_names:
                nested.pop(name, None)
            positional = [*node.args.posonlyargs, *node.args.args]
            if positional_defaults:
                for argument, default in zip(
                    positional[-len(positional_defaults) :], positional_defaults
                ):
                    bind_name(argument.arg, default, nested)
            for argument, default in zip(node.args.kwonlyargs, keyword_defaults):
                bind_name(argument.arg, default, nested)
            result = value_of(node.body, nested)
            if modules_escape(result):
                raise _ContentError("aliasing sys.modules is forbidden")
            return safe
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            value = value_of(node.value, environment)
            if modules_escape(value):
                raise _ContentError("aliasing sys.modules is forbidden")
            return safe
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                value_of(child, environment)
        return safe

    def target_mutates(
        target: ast.AST, environment: dict[str, _AbstractValue]
    ) -> bool:
        if isinstance(target, (ast.Tuple, ast.List)):
            return any(target_mutates(element, environment) for element in target.elts)
        if isinstance(target, ast.Starred):
            return target_mutates(target.value, environment)
        if isinstance(target, ast.Subscript):
            owner = value_of(target.value, environment)
            value_of(target.slice, environment)
            return bool(owner.direct.intersection(protected))
        if isinstance(target, ast.Attribute):
            owner = value_of(target.value, environment)
            return "sys" in owner.direct and target.attr in protected
        return False

    def bind_target(
        target: ast.AST,
        value: _AbstractValue,
        environment: dict[str, _AbstractValue],
        value_node: ast.AST | None,
    ) -> None:
        if target_mutates(target, environment):
            raise _ContentError("sys.path/sys.modules mutation is forbidden")
        if isinstance(target, ast.Name):
            bind_name(target.id, value, environment)
            return
        if isinstance(target, ast.Starred):
            bind_target(
                target.value,
                _AbstractValue(elements=value.direct | value.elements),
                environment,
                None,
            )
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            if (
                isinstance(value_node, (ast.Tuple, ast.List))
                and len(target.elts) == len(value_node.elts)
            ):
                for child_target, child_value in zip(target.elts, value_node.elts):
                    bind_target(
                        child_target,
                        value_of(child_value, environment),
                        environment,
                        child_value,
                    )
            else:
                for child_target in target.elts:
                    bind_target(
                        child_target,
                        _AbstractValue(value.elements),
                        environment,
                        None,
                    )

    def bind_import(node: ast.stmt, environment: dict[str, _AbstractValue]) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".", 1)[0]
                bind_name(
                    name,
                    _AbstractValue(frozenset({"sys"}))
                    if alias.name == "sys"
                    else safe,
                    environment,
                )
            return
        if not isinstance(node, ast.ImportFrom):
            return
        for alias in node.names:
            if node.module == "sys" and alias.name in {"*", "modules"}:
                raise _ContentError("aliasing sys.modules is forbidden")
            if alias.name == "*":
                continue
            name = alias.asname or alias.name
            bind_name(
                name,
                _AbstractValue(frozenset({"path"}))
                if node.module == "sys" and alias.name == "path"
                else safe,
                environment,
            )

    def analyze_function(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        environment: dict[str, _AbstractValue],
        positional_arguments: list[_AbstractValue] | None = None,
        keyword_arguments: dict[str, _AbstractValue] | None = None,
        *,
        inspect_header: bool = True,
    ) -> None:
        if inspect_header:
            for decorator in node.decorator_list:
                value_of(decorator, environment)
            positional_defaults = [
                value_of(default, environment) for default in node.args.defaults
            ]
            keyword_defaults = [
                value_of(default, environment) if default is not None else safe
                for default in node.args.kw_defaults
            ]
        else:
            positional_defaults = []
            keyword_defaults = [safe for _ in node.args.kw_defaults]
        if any(
            modules_escape(default)
            for default in (*positional_defaults, *keyword_defaults)
        ):
            raise _ContentError("aliasing sys.modules is forbidden")
        nested = (
            class_function_outers[-1].copy()
            if class_function_outers
            else environment.copy()
        )
        local_names = local_bindings(node.body)
        argument_names = {
            argument.arg
            for argument in (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            )
        }
        if node.args.vararg is not None:
            argument_names.add(node.args.vararg.arg)
        if node.args.kwarg is not None:
            argument_names.add(node.args.kwarg.arg)
        for name in local_names | argument_names:
            nested.pop(name, None)
        nested.update(import_bindings(node.body))
        positional = [*node.args.posonlyargs, *node.args.args]
        if positional_defaults:
            for argument, default in zip(
                positional[-len(positional_defaults) :], positional_defaults
            ):
                bind_name(argument.arg, default, nested)
        for argument, default in zip(node.args.kwonlyargs, keyword_defaults):
            bind_name(argument.arg, default, nested)
        if positional_arguments is not None:
            for argument, value in zip(positional, positional_arguments):
                bind_name(argument.arg, value, nested)
            if node.args.vararg is not None and len(positional_arguments) > len(
                positional
            ):
                extras = positional_arguments[len(positional) :]
                bind_name(
                    node.args.vararg.arg,
                    _AbstractValue(
                        elements=frozenset().union(
                            *(value.direct for value in extras)
                        )
                    ),
                    nested,
                )
        if keyword_arguments is not None:
            valid_arguments = {
                argument.arg: argument
                for argument in (
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                )
            }
            for name, value in keyword_arguments.items():
                if name in valid_arguments:
                    bind_name(name, value, nested)
                elif node.args.kwarg is not None:
                    if modules_escape(value):
                        raise _ContentError("aliasing sys.modules is forbidden")
        saved_class_outers = class_function_outers.copy()
        class_function_outers.clear()
        try:
            analyze_block(node.body, nested)
        finally:
            class_function_outers.extend(saved_class_outers)

    def loop_contains_break(statements: list[ast.stmt]) -> bool:
        class BreakFinder(ast.NodeVisitor):
            found = False

            def visit_Break(self, node: ast.Break) -> None:
                self.found = True

            def visit_For(self, node: ast.For) -> None:
                return

            def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
                return

            def visit_While(self, node: ast.While) -> None:
                return

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                return

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                return

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                return

            def visit_Lambda(self, node: ast.Lambda) -> None:
                return

        finder = BreakFinder()
        for statement in statements:
            finder.visit(statement)
        return finder.found

    def match_bound_names(pattern: ast.pattern) -> set[str]:
        names: set[str] = set()
        for node in ast.walk(pattern):
            if isinstance(node, ast.MatchAs) and node.name is not None:
                names.add(node.name)
            elif isinstance(node, ast.MatchStar) and node.name is not None:
                names.add(node.name)
            elif isinstance(node, ast.MatchMapping) and node.rest is not None:
                names.add(node.rest)
        return names

    def pattern_is_irrefutable(pattern: ast.pattern) -> bool:
        if isinstance(pattern, ast.MatchAs):
            return pattern.pattern is None or pattern_is_irrefutable(pattern.pattern)
        if isinstance(pattern, ast.MatchOr):
            return any(pattern_is_irrefutable(child) for child in pattern.patterns)
        return False

    def analyze_class(
        node: ast.ClassDef, environment: dict[str, _AbstractValue]
    ) -> None:
        for expression in (*node.decorator_list, *node.bases):
            value_of(expression, environment)
        for keyword in node.keywords:
            value_of(keyword.value, environment)
        lexical_outer = (
            class_function_outers[-1].copy()
            if class_function_outers
            else environment.copy()
        )
        nested = lexical_outer.copy()
        nested.update(import_bindings(node.body))
        class_function_outers.append(lexical_outer)
        try:
            analyze_block(node.body, nested)
        finally:
            class_function_outers.pop()

    def analyze_block(
        statements: list[ast.stmt], environment: dict[str, _AbstractValue]
    ) -> dict[str, _AbstractValue] | None:
        deferred_functions: list[
            tuple[ast.FunctionDef | ast.AsyncFunctionDef, dict[str, _AbstractValue]]
        ] = []
        deferred_classes: list[tuple[ast.ClassDef, dict[str, _AbstractValue]]] = []
        terminated = False
        for statement in statements:
            if isinstance(statement, ast.Expr):
                value_of(statement.value, environment)
            elif isinstance(statement, ast.Assign):
                value = value_of(statement.value, environment)
                for target in statement.targets:
                    bind_target(target, value, environment, statement.value)
            elif isinstance(statement, ast.AnnAssign):
                value_of(statement.annotation, environment)
                value = value_of(statement.value, environment)
                bind_target(statement.target, value, environment, statement.value)
            elif isinstance(statement, ast.AugAssign):
                value_of(statement.value, environment)
                if target_mutates(statement.target, environment) or (
                    isinstance(statement.target, ast.Name)
                    and environment.get(statement.target.id, safe).direct.intersection(
                        protected
                    )
                ):
                    raise _ContentError("sys.path/sys.modules mutation is forbidden")
                if isinstance(statement.target, ast.Name):
                    environment.pop(statement.target.id, None)
            elif isinstance(statement, (ast.Import, ast.ImportFrom)):
                bind_import(statement, environment)
            elif isinstance(statement, ast.Delete):
                if any(target_mutates(target, environment) for target in statement.targets):
                    raise _ContentError("sys.path/sys.modules mutation is forbidden")
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        environment.pop(target.id, None)
            elif isinstance(statement, ast.If):
                value_of(statement.test, environment)
                body = analyze_block(statement.body, environment.copy())
                otherwise = analyze_block(statement.orelse, environment.copy())
                fallthrough = merged_optional_environment(body, otherwise)
                if fallthrough is None:
                    terminated = True
                    break
                environment = fallthrough
            elif isinstance(statement, (ast.For, ast.AsyncFor)):
                iterable = value_of(statement.iter, environment)
                before = environment.copy()
                loop_head = before.copy()
                while True:
                    body_entry = loop_head.copy()
                    bind_target(
                        statement.target,
                        _AbstractValue(iterable.elements),
                        body_entry,
                        None,
                    )
                    body_environment = analyze_block(
                        statement.body, body_entry.copy()
                    )
                    if body_environment is None:
                        break
                    next_head = merged_environment(before, body_environment)
                    if next_head == loop_head:
                        break
                    loop_head = next_head
                definitely_nonempty = bool(
                    isinstance(statement.iter, (ast.Tuple, ast.List, ast.Set))
                    and statement.iter.elts
                )
                if body_environment is None:
                    normal_exit = None if definitely_nonempty else before
                else:
                    normal_exit = (
                        body_environment
                        if definitely_nonempty
                        else merged_environment(before, body_environment)
                    )
                exhausted = (
                    analyze_block(statement.orelse, normal_exit)
                    if normal_exit is not None
                    else None
                )
                if loop_contains_break(statement.body):
                    fallthrough = merged_optional_environment(
                        exhausted, body_entry, body_environment
                    )
                else:
                    fallthrough = exhausted
                if fallthrough is None:
                    terminated = True
                    break
                environment = fallthrough
            elif isinstance(statement, ast.While):
                value_of(statement.test, environment)
                before = environment.copy()
                loop_head = before.copy()
                while True:
                    body = analyze_block(statement.body, loop_head.copy())
                    if body is None:
                        break
                    next_head = merged_environment(before, body)
                    if next_head == loop_head:
                        break
                    loop_head = next_head
                normal_exit = (
                    before if body is None else merged_environment(before, body)
                )
                exhausted = analyze_block(statement.orelse, normal_exit)
                if loop_contains_break(statement.body):
                    fallthrough = merged_optional_environment(
                        exhausted, loop_head, body
                    )
                else:
                    fallthrough = exhausted
                if fallthrough is None:
                    terminated = True
                    break
                environment = fallthrough
            elif isinstance(statement, (ast.Try, ast.TryStar)):
                body_state = environment.copy()
                body = analyze_block(statement.body, body_state)
                normal = (
                    analyze_block(statement.orelse, body.copy())
                    if body is not None
                    else None
                )
                branches = [normal]
                final_inspection_states = [body_state]
                for handler in statement.handlers:
                    handled = environment.copy()
                    if handler.type is not None:
                        value_of(handler.type, handled)
                    if handler.name is not None:
                        handled.pop(handler.name, None)
                    branches.append(analyze_block(handler.body, handled))
                    final_inspection_states.append(handled)
                branch_fallthrough = merged_optional_environment(*branches)
                for inspection_state in final_inspection_states:
                    analyze_block(statement.finalbody, inspection_state.copy())
                if branch_fallthrough is None:
                    terminated = True
                    break
                final_environment = analyze_block(
                    statement.finalbody, branch_fallthrough
                )
                if final_environment is None:
                    terminated = True
                    break
                environment = final_environment
            elif isinstance(statement, (ast.With, ast.AsyncWith)):
                for item in statement.items:
                    value_of(item.context_expr, environment)
                    if item.optional_vars is not None:
                        bind_target(item.optional_vars, safe, environment, None)
                fallthrough = analyze_block(statement.body, environment)
                if fallthrough is None:
                    terminated = True
                    break
                environment = fallthrough
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                analyze_function(statement, environment)
                marker = f"function:{id(statement)}"
                function_registry[marker] = statement
                bind_name(
                    statement.name,
                    _AbstractValue(frozenset({marker})),
                    environment,
                )
                closure = (
                    class_function_outers[-1].copy()
                    if class_function_outers
                    else environment.copy()
                )
                deferred_functions.append((statement, closure))
            elif isinstance(statement, ast.ClassDef):
                analyze_class(statement, environment)
                deferred_classes.append((statement, environment.copy()))
                bind_name(statement.name, safe, environment)
            elif isinstance(statement, ast.Match):
                subject = value_of(statement.subject, environment)
                exhaustive = bool(
                    statement.cases
                    and statement.cases[-1].guard is None
                    and pattern_is_irrefutable(statement.cases[-1].pattern)
                )
                branches = [] if exhaustive else [environment.copy()]
                for case in statement.cases:
                    branch = environment.copy()
                    for child in ast.walk(case.pattern):
                        if isinstance(child, ast.expr):
                            value_of(child, branch)
                    for name in match_bound_names(case.pattern):
                        bind_name(name, subject, branch)
                    if case.guard is not None:
                        value_of(case.guard, branch)
                    branches.append(analyze_block(case.body, branch))
                fallthrough = merged_optional_environment(*branches)
                if fallthrough is None:
                    terminated = True
                    break
                environment = fallthrough
            elif isinstance(statement, ast.Return):
                value = value_of(statement.value, environment)
                if modules_escape(value):
                    raise _ContentError("aliasing sys.modules is forbidden")
                terminated = True
                break
            elif isinstance(statement, ast.Raise):
                value_of(statement.exc, environment)
                value_of(statement.cause, environment)
                terminated = True
                break
            elif isinstance(statement, ast.Assert):
                value_of(statement.test, environment)
                value_of(statement.msg, environment)
            else:
                for child in ast.iter_child_nodes(statement):
                    if isinstance(child, ast.expr):
                        value_of(child, environment)
        for node, closure in deferred_functions:
            if class_function_outers:
                analyze_function(node, closure, inspect_header=False)
            else:
                analyze_function(
                    node,
                    merged_environment(closure, environment),
                    inspect_header=False,
                )
        for node, outer in deferred_classes:
            analyze_class(node, merged_environment(outer, environment))
        return None if terminated else environment

    if not isinstance(tree, ast.Module):
        raise _ContentError("source is not a Python module")
    initial = import_bindings(tree.body)
    analyze_block(tree.body, initial)


def _module_replacement(role: str, module: str) -> str | None:
    if role == "agent":
        if module == "agents.heuristic.telemetry":
            return ".telemetry"
        if module == "src.packing_core":
            return ".packing_core"
        prefix = "src.packing_core."
        if module.startswith(prefix):
            suffix = module[len(prefix) :]
            if suffix.isidentifier():
                return f".packing_core.{suffix}"
    elif role == "core":
        if module == "src.packing_core":
            return "."
        prefix = "src.packing_core."
        if module.startswith(prefix):
            suffix = module[len(prefix) :]
            if suffix.isidentifier():
                return f".{suffix}"
    return None


def _rewrite_import_modules(text: str, role: str) -> str:
    lines = text.splitlines(keepends=True)
    replacements: dict[int, list[tuple[int, int, str]]] = {}
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (IndentationError, tokenize.TokenError) as exc:
        raise _ContentError(f"cannot tokenize source: {exc}") from exc

    for index, token in enumerate(tokens):
        if token.type != tokenize.NAME or token.string != "from":
            continue
        line_index = token.start[0] - 1
        if line_index < 0 or line_index >= len(lines):
            continue
        if lines[line_index][: token.start[1]].strip(" \t"):
            continue

        module_tokens: list[tokenize.TokenInfo] = []
        cursor = index + 1
        while cursor < len(tokens):
            current = tokens[cursor]
            if current.type == tokenize.NAME and current.string == "import":
                break
            if current.type == tokenize.NAME or (
                current.type == tokenize.OP and current.string == "."
            ):
                module_tokens.append(current)
                cursor += 1
                continue
            module_tokens = []
            break
        if not module_tokens or cursor >= len(tokens):
            continue
        if any(part.start[0] != token.start[0] for part in module_tokens):
            continue

        module = "".join(part.string for part in module_tokens)
        replacement = _module_replacement(role, module)
        if replacement is None:
            continue
        start = module_tokens[0].start[1]
        end = module_tokens[-1].end[1]
        replacements.setdefault(line_index, []).append((start, end, replacement))

    for line_index, edits in replacements.items():
        line = lines[line_index]
        for start, end, replacement in sorted(edits, reverse=True):
            line = f"{line[:start]}{replacement}{line[end:]}"
        lines[line_index] = line
    return "".join(lines)


def _transform_source(payload: bytes, *, archive_name: str, role: str) -> bytes:
    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _ContentError(f"source is not UTF-8: {archive_name}") from exc
    try:
        tree = ast.parse(decoded, filename=archive_name)
    except SyntaxError as exc:
        raise _ContentError(f"source is not valid Python: {archive_name}: {exc}") from exc

    _validate_forbidden_import_mechanisms(tree)
    normalized = decoded.replace("\r\n", "\n").replace("\r", "\n")
    transformed = _rewrite_import_modules(normalized, role)

    try:
        ast.parse(transformed, filename=archive_name)
    except SyntaxError as exc:
        raise _ContentError(
            f"transformed source is not valid Python: {archive_name}: {exc}"
        ) from exc
    if any(prefix in transformed for prefix in _OLD_PREFIXES):
        raise _ContentError(f"source retains a forbidden development prefix: {archive_name}")
    return transformed.encode("utf-8")


def _prepare_staging(
    agent_sources: dict[str, _SourceRecord],
    core_sources: dict[str, _SourceRecord],
    staging_root: Path,
) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    source_specs = [
        ("heuristic/agent.py", agent_sources["agent.py"], "agent"),
        ("heuristic/telemetry.py", agent_sources["telemetry.py"], "telemetry"),
    ]
    source_specs.extend(
        (f"heuristic/packing_core/{name}", core_sources[name], "core")
        for name in CORE_FILES
    )

    for archive_name, record, role in source_specs:
        transformed = _transform_source(
            _read_source(record), archive_name=archive_name, role=role
        )
        destination = staging_root.joinpath(*PurePosixPath(archive_name).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(transformed)
        payloads[archive_name] = transformed
    return payloads


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=_FIXED_TIMESTAMP)
    info.create_system = 3
    info.external_attr = (
        _DIRECTORY_MODE if name.endswith("/") else _REGULAR_MODE
    ) << 16
    info.compress_type = zipfile.ZIP_STORED
    info.extra = b""
    info.comment = b""
    return info


def _write_archive(candidate_file: object, payloads: dict[str, bytes]) -> None:
    with zipfile.ZipFile(
        candidate_file, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        archive.comment = b""
        for name in ARCHIVE_ENTRIES:
            payload = b"" if name.endswith("/") else payloads[name]
            archive.writestr(_zip_info(name), payload)


def _unsafe_archive_name(name: str) -> bool:
    path = PurePosixPath(name)
    return name.startswith("/") or path.is_absolute() or "\\" in name or ".." in path.parts


def _validate_archive(
    candidate: Path, record: _SourceRecord, payloads: dict[str, bytes]
) -> None:
    try:
        with _open_candidate(candidate, record) as candidate_file:
            with zipfile.ZipFile(candidate_file, mode="r") as archive:
                infos = archive.infolist()
                names = tuple(info.filename for info in infos)
                if any(_unsafe_archive_name(name) for name in names):
                    raise _ContentError("generated ZIP contains an unsafe entry path")
                if len(names) != len(set(names)):
                    raise _ContentError("generated ZIP contains duplicate entries")
                if names != ARCHIVE_ENTRIES:
                    raise _ContentError("generated ZIP entry set or order is incorrect")
                if archive.comment != b"":
                    raise _ContentError("generated ZIP archive comment is not empty")

                for info in infos:
                    name = info.filename
                    expected_mode = (
                        _DIRECTORY_MODE if name.endswith("/") else _REGULAR_MODE
                    )
                    if info.date_time != _FIXED_TIMESTAMP:
                        raise _ContentError(f"generated ZIP timestamp is invalid: {name}")
                    if info.create_system != 3:
                        raise _ContentError(
                            f"generated ZIP platform metadata is invalid: {name}"
                        )
                    if info.extra != b"" or info.comment != b"":
                        raise _ContentError(
                            f"generated ZIP extra metadata is not empty: {name}"
                        )
                    if info.compress_type != zipfile.ZIP_STORED:
                        raise _ContentError(f"generated ZIP compression is invalid: {name}")
                    if info.external_attr >> 16 != expected_mode:
                        raise _ContentError(f"generated ZIP mode is invalid: {name}")

                    payload = archive.read(info)
                    expected_payload = b"" if name.endswith("/") else payloads[name]
                    if payload != expected_payload:
                        raise _ContentError(
                            f"generated ZIP payload differs from staging: {name}"
                        )
                    if info.CRC != (zlib.crc32(payload) & 0xFFFFFFFF):
                        raise _ContentError(f"generated ZIP CRC is invalid: {name}")
                    if info.compress_size != len(payload) or info.file_size != len(payload):
                        raise _ContentError(
                            f"generated ZIP size metadata is invalid: {name}"
                        )
                    if name.endswith(".py"):
                        try:
                            source = payload.decode("utf-8")
                            if "\r" in source:
                                raise _ContentError(
                                    f"generated source is not LF-only: {name}"
                                )
                            ast.parse(source, filename=name)
                        except UnicodeDecodeError as exc:
                            raise _ContentError(
                                f"generated source is not UTF-8: {name}"
                            ) from exc
                        except SyntaxError as exc:
                            raise _ContentError(f"generated source is invalid: {name}") from exc

                if archive.testzip() is not None:
                    raise _ContentError("generated ZIP failed ZipFile.testzip()")
    except _ContentError:
        raise
    except (zipfile.BadZipFile, zipfile.LargeZipFile, EOFError, KeyError, RuntimeError) as exc:
        raise _ContentError(f"generated ZIP cannot be validated: {exc}") from exc


def _repository_root() -> Path:
    module = Path(__file__).resolve()
    for candidate in module.parents:
        if candidate.joinpath(".git").exists():
            return candidate
    # The clean CPU image mounts only ``simulator/`` at ``/workspace`` and has no
    # repository metadata.  In that layout the package containing ``scripts`` is
    # the narrowest trustworthy boundary available.
    return module.parents[1]


def _require_external_temp(path: Path) -> None:
    resolved = path.resolve()
    repository = _repository_root()
    if resolved == repository or repository in resolved.parents:
        raise _InputError(f"temporary directory is inside the repository: {resolved}")


def _extract_archive(
    candidate: Path, record: _SourceRecord, extraction_root: Path
) -> None:
    root = extraction_root.resolve()
    try:
        with _open_candidate(candidate, record) as candidate_file:
            with zipfile.ZipFile(candidate_file, mode="r") as archive:
                infos = archive.infolist()
                names = tuple(info.filename for info in infos)
                if names != ARCHIVE_ENTRIES or len(names) != len(set(names)):
                    raise _ContentError("archive changed before isolated inspection")
                for info in infos:
                    name = info.filename
                    if _unsafe_archive_name(name):
                        raise _ContentError("archive became unsafe before extraction")
                    expected_mode = (
                        _DIRECTORY_MODE if name.endswith("/") else _REGULAR_MODE
                    )
                    if info.external_attr >> 16 != expected_mode:
                        raise _ContentError(
                            f"archive entry mode changed before extraction: {name}"
                        )
                    destination = extraction_root.joinpath(*PurePosixPath(name).parts)
                    resolved = destination.resolve(strict=False)
                    if not _is_within(resolved, root):
                        raise _ContentError(f"archive entry escapes extraction root: {name}")
                    if name.endswith("/"):
                        destination.mkdir(parents=True, exist_ok=True)
                        destination.chmod(0o755)
                    else:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with destination.open("xb") as handle:
                            handle.write(archive.read(info))
                        destination.chmod(0o644)
    except _ContentError:
        raise
    except (zipfile.BadZipFile, zipfile.LargeZipFile, EOFError, KeyError, RuntimeError) as exc:
        raise _ContentError(f"generated ZIP cannot be extracted: {exc}") from exc


def _inspection_probe() -> str:
    modules = ("heuristic.telemetry", "heuristic.packing_core") + tuple(
        f"heuristic.packing_core.{Path(name).stem}" for name in CORE_FILES[1:]
    )
    return textwrap.dedent(
        f"""
        import importlib
        import sys
        from pathlib import Path

        names = {modules!r}
        loaded = {{name: importlib.import_module(name) for name in names}}
        loaded["heuristic.agent"] = importlib.import_module("heuristic.agent")
        payload_root = (Path.cwd() / "heuristic").resolve()

        def check_modules():
            old_prefixes = ("agents.heuristic", "src.packing_core")
            for existing in sys.modules:
                if any(
                    existing == prefix or existing.startswith(prefix + ".")
                    for prefix in old_prefixes
                ):
                    raise RuntimeError(f"development module loaded: {{existing}}")
            for name, module in loaded.items():
                origin_value = getattr(module, "__file__", None)
                if origin_value is None:
                    raise RuntimeError(f"module has no file origin: {{name}}")
                origin = Path(origin_value).resolve()
                if origin != payload_root and payload_root not in origin.parents:
                    raise RuntimeError(f"module outside payload: {{name}}={{origin}}")

        check_modules()
        loaded["heuristic.agent"].Agent("heuristic/")
        check_modules()
        """
    )


def _inspect_isolated(extraction_root: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = ""
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("TELEMETRY_DIR", None)
    environment.pop("TELEMETRY_RUN_ID", None)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _inspection_probe()],
            cwd=extraction_root,
            env=environment,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise _InputError(f"cannot start isolated package inspection: {exc}") from exc
    if completed.returncode != 0:
        detail_bytes = completed.stderr.strip() or completed.stdout.strip()
        detail = detail_bytes.decode("utf-8", errors="replace")
        raise _ContentError(f"isolated package inspection failed: {detail[-4000:]}")


def _unlink_candidate(candidate: Path | None, *, suppress_errors: bool = False) -> None:
    if candidate is None:
        return
    try:
        candidate.unlink(missing_ok=True)
    except OSError as exc:
        if suppress_errors:
            return
        raise _InputError(f"cannot remove candidate ZIP {candidate}: {exc}") from exc


def _path_matches_record(path: Path, record: _SourceRecord) -> bool:
    try:
        return _same_record(path.lstat(), record)
    except OSError:
        return False


def _publication_committed(paths: _Paths, candidate: Path, record: _SourceRecord) -> bool:
    return not os.path.lexists(candidate) and _path_matches_record(paths.output, record)


def _create_candidate(
    paths: _Paths, payloads: dict[str, bytes]
) -> tuple[Path, _SourceRecord]:
    _validate_output_parent(paths)
    descriptor = -1
    candidate: Path | None = None
    try:
        descriptor, candidate_name = tempfile.mkstemp(
            prefix=".package-submit.", suffix=".zip", dir=paths.output.parent
        )
        candidate = Path(candidate_name)
        with os.fdopen(descriptor, "w+b", closefd=True) as handle:
            descriptor = -1
            _write_archive(_PathNamedFile(handle, candidate), payloads)
            handle.flush()
            record = _record(candidate, os.fstat(handle.fileno()))
        _require_candidate(candidate, record)
        return candidate, record
    except BaseException as active:
        close_error: OSError | None = None
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError as exc:
                close_error = exc
        try:
            _unlink_candidate(
                candidate, suppress_errors=isinstance(active, KeyboardInterrupt)
            )
        except _InputError as cleanup_error:
            raise cleanup_error from active
        if close_error is not None and not isinstance(active, KeyboardInterrupt):
            raise _InputError(f"cannot close candidate ZIP: {close_error}") from active
        raise


def _publish_force(paths: _Paths, candidate: Path, record: _SourceRecord) -> None:
    try:
        os.replace(candidate, paths.output)
    except BaseException:
        # A signal can be delivered after the atomic syscall committed but before Python returns.
        if _publication_committed(paths, candidate, record):
            return
        raise


def _publish_no_force(paths: _Paths, candidate: Path, record: _SourceRecord) -> None:
    """Atomically create a previously absent output without a check/replace clobber race."""
    try:
        os.link(candidate, paths.output, follow_symlinks=False)
        candidate.unlink()
    except KeyboardInterrupt:
        if _path_matches_record(paths.output, record):
            try:
                candidate.unlink(missing_ok=True)
            except OSError as exc:
                if _path_matches_record(paths.output, record):
                    paths.output.unlink(missing_ok=True)
                raise _InputError(
                    f"cannot finish no-clobber publication cleanup: {exc}"
                ) from exc
            return
        raise
    except FileExistsError as exc:
        raise _InputError(f"--output appeared during packaging: {paths.output}") from exc
    except OSError as exc:
        if _path_matches_record(paths.output, record) and not os.path.lexists(candidate):
            return
        if _path_matches_record(paths.output, record):
            try:
                paths.output.unlink()
            except OSError as rollback_exc:
                raise _InputError(
                    f"cannot roll back failed no-clobber publication: {rollback_exc}"
                ) from exc
        raise _InputError(f"cannot publish candidate ZIP: {exc}") from exc


def _publish_candidate(paths: _Paths, candidate: Path, record: _SourceRecord) -> None:
    _validate_output_parent(paths)
    _validate_output_state(paths.output, force=paths.force)
    _require_candidate(candidate, record)
    if paths.force:
        _publish_force(paths, candidate, record)
    else:
        _publish_no_force(paths, candidate, record)


def _execute(paths: _Paths) -> None:
    content_errors: list[_ContentError] = []
    agent_sources: dict[str, _SourceRecord] | None = None
    core_sources: dict[str, _SourceRecord] | None = None
    try:
        agent_sources = _scan_source_tree(
            paths.agent_dir, _AGENT_FILES, skip_dirs=frozenset({"packing_core"})
        )
    except _ContentError as exc:
        content_errors.append(exc)
    try:
        core_sources = _scan_source_tree(paths.core_dir, CORE_FILES)
    except _ContentError as exc:
        content_errors.append(exc)
    if content_errors:
        raise content_errors[0]
    if agent_sources is None or core_sources is None:
        raise _InputError("source inventory did not complete")

    candidate: Path | None = None
    record: _SourceRecord | None = None
    try:
        with tempfile.TemporaryDirectory(
            prefix="t029-package-staging.", dir="/tmp"
        ) as staging_name:
            staging_root = Path(staging_name)
            _require_external_temp(staging_root)
            payloads = _prepare_staging(
                agent_sources, core_sources, staging_root
            )
            candidate, record = _create_candidate(paths, payloads)

        _validate_archive(candidate, record, payloads)

        with tempfile.TemporaryDirectory(
            prefix="t029-package-inspection.", dir="/tmp"
        ) as extraction_name:
            extraction_root = Path(extraction_name)
            _require_external_temp(extraction_root)
            _extract_archive(candidate, record, extraction_root)
            _inspect_isolated(extraction_root)

        # The inspected payload runs as the same user; revalidate the candidate afterwards.
        _validate_archive(candidate, record, payloads)
        try:
            _publish_candidate(paths, candidate, record)
            candidate = None
        except KeyboardInterrupt:
            if _publication_committed(paths, candidate, record):
                candidate = None
                return
            raise
    finally:
        active = sys.exc_info()[1]
        _unlink_candidate(
            candidate, suppress_errors=isinstance(active, KeyboardInterrupt)
        )


def main(argv: list[str] | None = None) -> int:
    """Build, validate, and atomically publish the fixed submission archive."""
    resolved_argv = sys.argv[1:] if argv is None else argv
    try:
        try:
            args = _build_parser().parse_args(resolved_argv)
        except SystemExit:
            return _EXIT_INPUT
        paths = _validate_paths(args)
        _execute(paths)
        return _EXIT_OK
    except KeyboardInterrupt:
        return _EXIT_INTERRUPTED
    except _ContentError as exc:
        print(f"package_submit: {exc}", file=sys.stderr)
        return _EXIT_CONTENT
    except Exception as exc:
        print(f"package_submit: {exc}", file=sys.stderr)
        return _EXIT_INPUT


if __name__ == "__main__":
    raise SystemExit(main())
