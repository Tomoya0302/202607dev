"""packing_core.watchdog: 1ステップの時間予算管理（T-022）。

本モジュールは `StepBudget` のみを提供する。後続チケット（T-026）が実装する
多層フォールバック機能はここには含まれない。
"""

import time
from collections.abc import Callable


class StepBudget:
    """1ステップの時間予算（soft/hard 締切）を表す。

    経過時間は既定で `time.monotonic()` から計算するが、決定論的なテストの
    ために `now_fn` へゼロ引数の時計関数を注入できる。

    Attributes:
        t0: 計測開始時刻 [s]（既定時計と同じ基準の値）。
        soft: ソフト締切までの経過時間 [s]。
        hard: ハード締切までの経過時間 [s]。
    """

    def __init__(
        self,
        t0: float,
        soft: float,
        hard: float,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        """StepBudget を生成する。

        Args:
            t0: 計測開始時刻 [s]。通常は `time.monotonic()` の呼び出し結果。
            soft: ソフト締切までの経過時間 [s]。
            hard: ハード締切までの経過時間 [s]。
            now_fn: 現在時刻 [s] を返すゼロ引数の呼び出し可能オブジェクト。
                既定は `time.monotonic`。テストでは fake clock を注入する。
        """
        self.t0 = t0
        self.soft = soft
        self.hard = hard
        self._now_fn = now_fn

    def elapsed(self) -> float:
        """`t0` からの経過時間を返す。

        Returns:
            `now_fn()` と `t0` の差 [s]。
        """
        return self._now_fn() - self.t0

    def over_soft(self) -> bool:
        """経過時間がソフト締切に達したかを返す。

        Returns:
            `elapsed() >= soft` なら True。
        """
        return self.elapsed() >= self.soft

    def over_hard(self) -> bool:
        """経過時間がハード締切に達したかを返す。

        Returns:
            `elapsed() >= hard` なら True。
        """
        return self.elapsed() >= self.hard
