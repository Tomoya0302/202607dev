"""到達可能性の上限を測る診断ランナー（findings §HF-035）。

公式 `PlacementValidator.check_transport_path`（搬入経路の水平スイープ判定）を
常時 True に差し替えて 1 課題を実行し、`bench_run.run_one` と同じ指標を出す。

目的: 「壁は幾何ではなく到達可能性」（`packing_core/constants.py:312-314` が
v25 期に記録した 97.2% / fill 63.30）が現行ベースラインでも成立するかを確認する。
本スクリプトは**診断専用**で、提出物・本番挙動には一切影響しない
（公式コードのファイル自体は変更せず、実行時にメソッドを差し替えるだけ）。

使い方:
    python -m scripts.reach_ceiling --config-path artifacts/bench_tasks/family1/f1_0000.json
    python -m scripts.reach_ceiling --config-path <task> --no-patch   # 対照（v43既定）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())


def _patch_transport_path() -> None:
    """搬入経路の衝突判定だけを無効化する。

    `check_transport_path` 自体を潰すと、その中で行われる `item.spawn()` の副作用まで
    消えてしまい（後段の `place_item` が None 姿勢で落ちる）、実験にならない。
    そこで衝突判定を行う `_move_item` だけを「常に到達成功」へ差し替え、
    spawn / saveState / restoreState の流れは公式のまま残す。
    """
    from src.ground_handling.validator import PlacementValidator

    def _always_reach(self, container, item, item_id, start_pos, target_pos, target_orn, steps):
        return True, tuple(target_pos)

    PlacementValidator._move_item = _always_reach  # type: ignore[method-assign]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--module-path", default="agents/heuristic/")
    parser.add_argument("--agent-class", default="Agent")
    parser.add_argument("--no-patch", action="store_true",
                        help="差し替えを行わない対照実行（現行ベースライン）")
    args = parser.parse_args()

    if not args.no_patch:
        # 到達可能性を系から完全に取り除くには両側を外す必要がある:
        #   (1) 公式 validator の衝突判定（打ち切りの原因）
        #   (2) エージェント自身の保守プロキシ `check_l_path`（候補段階で先に弾いている）
        # (1) だけ外すと、エージェントは同じ手で投了し続けるので上限は測れない。
        _patch_transport_path()
        os.environ["GH_DIAG_NO_LPATH"] = "1"

    from scripts.bench_run import run_one

    agent_module = ".".join(args.module_path.split("/")) + "agent"
    result = run_one(args.config_path, args.module_path, agent_module, args.agent_class)
    result["variant"] = "baseline" if args.no_patch else "no_transport_check"
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
