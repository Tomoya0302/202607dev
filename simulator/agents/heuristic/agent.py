"""heuristic Agent 骨格（T-023）。

公式 `AgentFactory` からロードできる最小の `Agent` を定義する。積付アルゴリズムは
実装せず、公式 Runner から4メソッド（`__init__` / `get_init_states` / `optimize` /
`policy`）を正常に呼び出せ、`policy` が常に正式な4キー action 辞書を返すことのみを
満たす（詳細仕様書 §4.12・§6 T-023 行）。候補列挙・段階フィルタ・スコアリング・
L字判定・フォールバックは T-024 以降で配線する。
"""

import numpy as np

from src.packing_core.state import make_action


class Agent:
    """heuristic 積付 Agent の骨格実装（T-023）。"""

    def __init__(self, module_path: str) -> None:
        """最小限の初期化のみ行う。

        モデル・設定ファイルの読込、ウォームアップ（T-027 責務）、状態構築、
        ファイル出力、telemetry 生成は行わない。

        Args:
            module_path: 公式 `AgentFactory` から渡されるモジュールパス。
        """
        self.module_path = module_path

    def get_init_states(self, init_states: dict) -> None:
        """公式から渡される初期状態を保持する。

        `init_states` およびそのネスト要素は変更しない。子プロセス側で
        既にデシリアライズ済み・非共有のオブジェクトであるため、追加の
        コピーは行わず参照をそのまま保持する。

        保持先を `self.optimize` にすると、公式 I/F の `optimize()` メソッドを
        bool 属性が隠してしまい、Runner による `agent.optimize(item_list)` 呼出しが
        `TypeError: 'bool' object is not callable` で失敗する。そのため
        `self.optimize_enabled` という別名で保持する。

        Args:
            init_states: `optimize` / `lookahead_k` / `container_list` を持つ辞書。
        """
        self.optimize_enabled = bool(init_states["optimize"])
        self.lookahead_k = int(init_states["lookahead_k"])
        self.container_list = init_states["container_list"]

    def optimize(self, item_list: list[dict]) -> list[int]:
        """全荷物の積み込み順を決定する（T-023 は入力順を維持）。

        体積・重量等による並べ替えは T-027 の責務。本メソッドは `item_list` を
        変更せず、各荷物の公式 `item["index"]` を入力順のまま返す。

        Args:
            item_list: 全荷物の情報が格納された辞書のリスト。

        Returns:
            list[int]: 全荷物の公式 index を過不足なく1回ずつ含む、入力順のリスト。
        """
        return [int(item["index"]) for item in item_list]

    def policy(self, observation: dict) -> dict:
        """固定 placeholder action を返す（T-023 では探索を行わない）。

        仕様書 v1.15 で確定した固定引数（`item_idx=0, container_idx=0,
        pos_rel=(0.0, 0.0, 0.5), orientation=0`）を `make_action()` 経由で返す。
        `observation` は読み取らず、変更もしない。乱数は使用しない。

        Args:
            observation: 現在の観測情報（本メソッドでは使用しない）。

        Returns:
            dict: `item_idx` / `container_idx` / `place_pos` / `orientation` の
            4キーのみを持つ action 辞書。
        """
        pos_rel = np.asarray((0.0, 0.0, 0.5), dtype=np.float64)
        return make_action(
            item_idx=0,
            container_idx=0,
            pos_rel=pos_rel,
            orientation=0,
        )
