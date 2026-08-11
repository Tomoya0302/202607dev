"""`stair_mass_diag.py` を複数タスクへ並列適用する（`drift_batch.py`/`tilt_batch.py` と同じ設計）。"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed


def _run_one(config_path: str, module_path: str, env_overrides: dict) -> dict:
    env = dict(os.environ)
    env.update(env_overrides)
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.stair_mass_diag",
         "--config-path", config_path, "--module-path", module_path],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=env, capture_output=True, text=True, timeout=300,
    )
    result = None
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith('{"config"'):
            try:
                result = json.loads(line)
            except Exception:
                result = None
            break
    if result is None:
        result = {"config": os.path.basename(config_path), "error": True,
                  "stderr_tail": proc.stderr[-1000:]}
    result["task_path"] = os.path.basename(config_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--module-path", default="agents/heuristic/")
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--extra-env", action="append", default=[],
                        help="KEY=VALUE 形式（複数指定可、GH_* レバー上書き用）")
    args = parser.parse_args()

    env_overrides = {}
    for kv in args.extra_env:
        k, _, v = kv.partition("=")
        env_overrides[k] = v

    task_paths = sorted(
        os.path.join(args.task_dir, f) for f in os.listdir(args.task_dir) if f.endswith(".json")
    )
    print(f"running stair_mass_diag on {len(task_paths)} tasks, extra_env={env_overrides}, "
          f"workers={args.workers}", file=sys.stderr)
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    n_done = 0
    with open(args.out_path, "w") as out_f, ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, p, args.module_path, env_overrides): p for p in task_paths}
        for fut in as_completed(futures):
            try:
                result = fut.result()
            except Exception as exc:
                result = {"error": True, "message": str(exc)}
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            n_done += 1
            print(f"  {n_done}/{len(task_paths)} done", file=sys.stderr)
    print(f"wrote {args.out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
