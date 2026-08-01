"""ベンチタスク群を複数 agent 設定（v38既定 / GH_ORDER_SEARCH=1）で並列評価する
（方策3判定用 Phase 0、findings_2026-07.md §9.1「14並列」を踏襲）。

各タスク・各設定を `scripts/bench_run.py` の別プロセス（`subprocess`）として実行する。
プロセス分離により、Agent モジュール側のグローバル状態（`_heightmap.PRIO_TOTAL` 等）が
タスク間で漏れないことを保証する（`bench_run.py` 自体も `TimedAgentRunner` で二重に
プロセス分離するが、この外側の並列化はタスク単位のスループット向上が目的）。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed


def _run_one(config_path: str, module_path: str, env_overrides: dict, shake_test: bool,
             kick_speed: float) -> dict:
    env = dict(os.environ)
    env.update(env_overrides)
    cmd = [sys.executable, "-m", "scripts.bench_run",
           "--config-path", config_path, "--module-path", module_path]
    if shake_test:
        cmd.extend(["--shake-test", "--kick-speed", str(kick_speed)])
    proc = subprocess.run(
        cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=env, capture_output=True, text=True, timeout=300,
    )
    # bench_run.py の JSON 出力後に、PyBullet/インタプリタ終了時の診断行（例: "argv[0]="）が
    # 追加で出ることがあるため、末尾行ではなく `{"config"` で始まる行を探す（末尾優先）。
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
        result = {"config": os.path.basename(config_path), "status": "exec_exception",
                  "message": f"unparseable output; stdout_tail={proc.stdout[-500:]!r}; "
                              f"stderr_tail={proc.stderr[-2000:]!r}"}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True, help="scripts.gen_tasks の出力ディレクトリ")
    parser.add_argument("--module-path", default="agents/heuristic/")
    parser.add_argument("--variant", required=True,
                        help="出力タグ用の自由ラベル。'search' は GH_ORDER_SEARCH=1 を自動付与（後方互換）")
    parser.add_argument("--extra-env", action="append", default=[],
                        help="KEY=VALUE 形式（複数指定可）。任意のレバーをサブプロセスへ渡す")
    parser.add_argument("--shake-test", action="store_true",
                        help="bench_run.py --shake-test を各タスクへ渡す")
    parser.add_argument("--kick-speed", type=float, default=0.4,
                        help="bench_run.py --kick-speed を各タスクへ渡す")
    parser.add_argument("--out-path", required=True, help="結果 JSONL 出力先")
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()

    env_overrides = {"GH_ORDER_SEARCH": "1"} if args.variant == "search" else {}
    for kv in args.extra_env:
        k, _, v = kv.partition("=")
        env_overrides[k] = v
    task_paths = sorted(
        os.path.join(args.task_dir, f) for f in os.listdir(args.task_dir) if f.endswith(".json")
    )
    print(f"evaluating {len(task_paths)} tasks, variant={args.variant}, "
          f"extra_env={env_overrides}, shake_test={args.shake_test}, workers={args.workers}",
          file=sys.stderr)

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    n_done = 0
    with open(args.out_path, "w") as out_f, ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_one, p, args.module_path, env_overrides, args.shake_test,
                       args.kick_speed): p
            for p in task_paths
        }
        for fut in as_completed(futures):
            task_path = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                result = {"config": os.path.basename(task_path), "status": "exec_exception",
                          "message": str(exc)}
            result["variant"] = args.variant
            result["task_path"] = os.path.basename(task_path)
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            n_done += 1
            if n_done % 5 == 0 or n_done == len(task_paths):
                print(f"  {n_done}/{len(task_paths)} done", file=sys.stderr)
    print(f"wrote {args.out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
