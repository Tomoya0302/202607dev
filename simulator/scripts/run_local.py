"""最小ローカル実行基盤（T-030A）。

既存の公式実行経路（``src.ground_handling.app.EvaluationApp``、
``scripts/run_test.py`` と同一クラス）を薄いアダプタとして再利用し、指定 Agent・
単一 config を実行する。ステップ実行・Agent ロード・結果集計などの実行ロジックは
本スクリプトで重複実装しない（詳細仕様書 §5.7）。

実行後は公式の結果 JSON（``EvaluationApp`` が書き出すスキーマそのまま）を読み，
(a) validator NG（Runner が正常応答として処理した配置失敗。
``status == 'success'`` かつ ``place_states`` のいずれかが False）と
(b) Agent／Runner 側の未捕捉例外（``status == 'format_error'``）を区別してログ報告する。
独自の結果スキーマ・例外タクソノミ・exit code 体系は追加しない。

想定実行形式::

    python -m scripts.run_local \\
        --module-path agents/base/ \\
        --config-path configs/local_suite/c01.json \\
        --result-dir /tmp/t030a-result
"""

from __future__ import annotations

import argparse
import json
import logging
import os

from src.ground_handling.app import EvaluationApp

_RESULT_FNAME = "evaluation_results.json"  # run_test.py の既定出力名と一致させる

logger = logging.getLogger("packing")


def parse_args() -> argparse.Namespace:
    """CLI 引数を解析する。

    ``run_test.py`` と引数名・意味を揃える（``--config-path`` / ``--module-path``）。
    ``--result-dir`` は既定値を持たせず必須とし、追跡済み ``results/`` を
    誤って汚さないようにする。

    Returns:
        argparse.Namespace: 解析済み引数（``config_path`` / ``module_path`` /
        ``result_dir``）。
    """
    parser = argparse.ArgumentParser(
        description="T-030A minimal local execution adapter (thin wrapper over EvaluationApp)."
    )
    parser.add_argument(
        "--config-path",
        default="configs/sample_config.json",
        help="config path (run_test.py と同じ既定値)",
        type=str,
    )
    parser.add_argument(
        "--module-path",
        default="agents/base/",
        help="agent module path（このディレクトリに対する相対パス。末尾 / 必須）",
        type=str,
    )
    parser.add_argument(
        "--result-dir",
        required=True,
        help="結果出力先ディレクトリ（必須。追跡済み results/ を汚さないため既定値なし）",
        type=str,
    )
    return parser.parse_args()


def summarize_results(result_path: str) -> tuple[int, int]:
    """公式結果 JSON を読み、validator NG と Agent 例外の件数をログ報告する。

    ``EvaluationApp`` が書き出す結果スキーマをそのまま読み取るだけで、
    validator 判定や評価値の再計算は行わない。

    Args:
        result_path: ``EvaluationApp`` が書き出した結果 JSON のパス。

    Returns:
        tuple[int, int]: ``(validator_ng_count, agent_exception_count)``。
        ``validator_ng_count`` は ``status == 'success'`` かつ
        ``place_states`` のいずれかが False であるタスク数。
        ``agent_exception_count`` は ``status == 'format_error'``
        （Agent／Runner 側の未捕捉例外、または optimize 不正応答）のタスク数。
    """
    with open(result_path) as f:
        results = json.load(f)

    validator_ng_count = 0
    agent_exception_count = 0

    for task_id, task_result in results.items():
        status = task_result.get("status")
        place_states = task_result.get("place_states") or {}
        is_ng = bool(place_states) and not all(place_states.values())

        if status == "format_error":
            agent_exception_count += 1
            logger.info(
                "[task %s] AGENT/RUNNER EXCEPTION: status=%s message=%s",
                task_id, status, task_result.get("message"),
            )
        elif status == "success" and is_ng:
            validator_ng_count += 1
            logger.info(
                "[task %s] VALIDATOR NG: place_states=%s", task_id, place_states,
            )
        else:
            logger.info(
                "[task %s] status=%s place_states=%s", task_id, status, place_states,
            )

    logger.info(
        "summary: validator_ng=%d, agent_exceptions=%d, total_tasks=%d",
        validator_ng_count, agent_exception_count, len(results),
    )
    return validator_ng_count, agent_exception_count


def main() -> None:
    """CLI エントリポイント。

    ``EvaluationApp`` を構築して単一 config・単一 Agent を実行し、
    公式結果 JSON から validator NG と Agent 例外の件数を要約報告する。
    実行ロジック自体はすべて ``EvaluationApp`` に委譲する。
    """
    logging.basicConfig(level=logging.INFO)

    args = parse_args()
    module_path = args.module_path  # 相対パス
    agent_module_path = ".".join(module_path.split("/")) + "agent"  # run_test.py と同一導出

    app = EvaluationApp(
        config_path=args.config_path,
        module_path=module_path,
        agent_module=agent_module_path,
        agent_class="Agent",
        result_dir=args.result_dir,
        result_fname=_RESULT_FNAME,
    )
    app.run(render_mode=None, verbose=False)

    result_path = os.path.join(args.result_dir, _RESULT_FNAME)
    summarize_results(result_path)


if __name__ == "__main__":
    main()
