"""packing_core.watchdog: 1ステップの時間予算管理（T-022）と多層フォールバック（T-024）。

`StepBudget` に加え、`safe_decide`・4層（`layer1_main`〜`layer4_max_p`）を提供する
（詳細仕様書 §4.11）。循環import回避契約（§4.11）により、`CandidatePools`/`CandidateKey`
（`candidates.py` 所有）は `typing.TYPE_CHECKING` ブロック内でのみ参照する（実行時import
しない）。`candidates.py` 側は本モジュールの `StepBudget` を通常の実行時importとして行うため、
実行時依存は `candidates.py` → `watchdog.py` の一方向のみとなる。
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from .masks import check_l_path
from .types import Candidate

if TYPE_CHECKING:
    from .candidates import CandidateKey, CandidatePools
    from .constants import PlacementParams, StageParams
    from .state import PackingState


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


# 4層フォールバックの公開シグネチャ（§4.11）。`PackingState`/`StepBudget` は文字列の前方参照
# として渡す（typing の Callable subscript は文字列を ForwardRef として遅延解決するため、
# TYPE_CHECKING 専用importの PackingState を実行時に評価しない）。
Layer = Callable[["PackingState", "StepBudget"], "Candidate | None"]


def _cache_key(cand: Candidate) -> "CandidateKey":
    """`Candidate` から L_PATHキャッシュキーを射影する（`candidates.candidate_key` と同一の
    式・順序。実行時循環importを避けるため `candidates.py` の関数はimportせずここで独立に
    再定義する）。HF-003: `anchor` を含め、同一(item,container,orientation,ems_id)でも位置の
    異なるアンカー候補を区別する（別位置の合否を誤再利用しない）。"""
    return (
        int(cand.item_idx), int(cand.container_idx), int(cand.orientation),
        int(cand.ems_id), int(cand.anchor),
    )


def layer1_main(
    state: "PackingState", budget: StepBudget, *,
    pools: "CandidatePools", pp: "PlacementParams", stage_params: "StageParams",
) -> Candidate | None:
    """`geo_candidates` の score上位 `l_path_top_m` 件のみへ `check_l_path` を実行する層（§4.11）。

    `pools.geo_candidates` は in-place sort せず、`sorted()` の新規コピーを評価対象に使う
    （元の基本順序は `pools.geo_candidates` に温存し、layer2/layer3 が参照できるようにする）。

    Args:
        state: 現在の `PackingState`。
        budget: 全層で共有する `StepBudget`。
        pools: 候補集合（`geo_candidates`/`l_path_cache`/`path_candidates` を使用・更新）。
        pp: 配置判定パラメータ（`check_l_path` へ渡す）。
        stage_params: `l_path_top_m` を使用。

    Returns:
        L_PATH合格候補のうち `cand.score` 最大（同点は基本順序）。合格候補が無い、
        `l_path_top_m<=0`、または層内で例外が発生した場合は `None`。
    """
    try:
        if stage_params.l_path_top_m <= 0:
            return None

        # score降順・安定sort（同点はsort前の基本順序を保つ）。元のlistは変更しない。
        ranked = sorted(pools.geo_candidates, key=lambda c: -c.score)
        top = ranked[: stage_params.l_path_top_m]

        # layer1のゲートはsoft締切のみ（§4.11「soft超過時はその時点までに確認済みの最良
        # L_PATH合格候補を返す（未評価候補へ進まない）」）。over_hard()の独立確認は追加せず、
        # StepBudgetの単一確認（1候補あたり1回のover_soft()呼出し）で決定論的に打ち切る。
        best: Candidate | None = None
        for cand in top:
            if budget.over_soft():
                break
            key = _cache_key(cand)
            if key in pools.l_path_cache:
                passed = pools.l_path_cache[key]
            else:
                passed = bool(check_l_path(state, cand, pp))
                pools.l_path_cache[key] = passed
            if passed:
                if all(cand is not c for c in pools.path_candidates):
                    pools.path_candidates.append(cand)
                if best is None or cand.score > best.score:
                    best = cand
        return best
    except Exception:
        return None


def _best_by_stability(cands: list[Candidate]) -> Candidate | None:
    """最も安定な候補を返す（cg_margin 最大、同点は score 最大、さらに基本順序を保持）。

    HF-006: フォールバック層(layer2/3)がスコア/安定性を無視して基本順先頭を返すと、重心が
    支持外(cg_margin<0)の候補を選び物理沈降で転倒する。合格候補の中から cg_margin（features、
    未設定は -inf 扱い）が最大＝最も転倒しにくい候補を選ぶことで、候補集合を縮小せずに
    （layer1・pool は不変のまま）フォールバック時のみ安定側へ寄せる。空なら None。
    """
    best: Candidate | None = None
    best_key: tuple[float, float] | None = None
    for cand in cands:
        cg = float(cand.features.get("cg_margin", float("-inf")))
        key = (cg, float(cand.score))
        if best is None or key > best_key:  # type: ignore[operator]
            best, best_key = cand, key
    return best


def layer2_dblf_strict(
    state: "PackingState", budget: StepBudget, *,
    pools: "CandidatePools", pp: "PlacementParams", stage_params: "StageParams",
    prefer_stable: bool = False,
) -> Candidate | None:
    """`geo_candidates` を基本順序で走査し、L_PATH合格候補を返す層（§4.11）。

    layer1がまだ評価していない候補を、hard期限内に限り追加で `check_l_path` を評価する
    （既評価分はキャッシュを再利用し重複呼出ししない）。`prefer_stable=False`（既定）は最初の
    合格候補を返す（従来契約）。`prefer_stable=True`（HF-006）は hard 期限内に評価した合格候補の
    うち最も安定なもの（`_best_by_stability`）を返す＝転倒しやすい先頭候補を避ける。

    Args:
        state: 現在の `PackingState`。
        budget: 全層で共有する `StepBudget`。
        pools: 候補集合（`geo_candidates`/`l_path_cache`/`path_candidates` を使用・更新）。
        pp: 配置判定パラメータ（`check_l_path` へ渡す）。
        stage_params: 本層では未使用（シグネチャ統一のため受け取る）。
        prefer_stable: True で合格候補中の最安定を返す（HF-006）。

    Returns:
        合格候補（既定=基本順序先頭 / prefer_stable=最安定）。候補なし・時間切れ・例外なら `None`。
    """
    try:
        # layer2のゲートはhard締切のみ（§4.11）。
        passers: list[Candidate] = []
        for cand in pools.geo_candidates:
            if budget.over_hard():
                break
            key = _cache_key(cand)
            if key in pools.l_path_cache:
                passed = pools.l_path_cache[key]
            else:
                passed = bool(check_l_path(state, cand, pp))
                pools.l_path_cache[key] = passed
            if passed:
                if all(cand is not c for c in pools.path_candidates):
                    pools.path_candidates.append(cand)
                if not prefer_stable:
                    return cand
                passers.append(cand)
        return _best_by_stability(passers) if prefer_stable else None
    except Exception:
        return None


def layer3_first_fit(
    state: "PackingState", budget: StepBudget, *,
    pools: "CandidatePools", pp: "PlacementParams", stage_params: "StageParams",
    prefer_stable: bool = False,
) -> Candidate | None:
    """新しい `check_l_path` 評価を行わず、既知のL_PATH状態で縮退選択する層
    （§4.11、HF-001でv1.27改訂）。

    `pools.geo_candidates` を (1) `l_path_cache` が `True` (2) 未評価 の2群へ安定分類し、
    各群内は基本順序を維持したまま、群優先順位(1)→(2)で先頭のCandidateを返す。
    `l_path_cache` が `False`（既知のL_PATH不合格）と判明した候補は選択対象から完全に
    除外する（旧v1.17の3群「(1)True→(2)未評価→(3)False」契約は撤回。既知不合格を最後の
    手段として返すと公式validatorで確実にNGとなる縮退を選んでしまうため禁止する）。

    Args:
        state: 現在の `PackingState`（本層では未使用、シグネチャ統一のため受け取る）。
        budget: 全層で共有する `StepBudget`（本層では未使用）。
        pools: 候補集合（`geo_candidates`/`l_path_cache` を参照。更新しない）。
        pp: 配置判定パラメータ（本層では未使用）。
        stage_params: 本層では未使用。

    Returns:
        群優先順位(1)→(2)の先頭Candidate。両群とも空、`geo_candidates` が空、
        または層内例外なら `None`。
    """
    try:
        group_true: list[Candidate] = []
        group_unknown: list[Candidate] = []
        for cand in pools.geo_candidates:
            key = _cache_key(cand)
            if key not in pools.l_path_cache:
                group_unknown.append(cand)
            elif pools.l_path_cache[key]:
                group_true.append(cand)
            # l_path_cache が False の候補はどちらの群にも含めない（選択対象から除外）。

        for group in (group_true, group_unknown):
            if group:
                # HF-006: prefer_stable のとき、各群内で最も安定な候補を返す（従来は先頭）。
                return _best_by_stability(group) if prefer_stable else group[0]
        return None
    except Exception:
        return None


def layer4_max_p(
    state: "PackingState", budget: StepBudget, *,
    pools: "CandidatePools", pp: "PlacementParams", stage_params: "StageParams",
) -> Candidate | None:
    """`geo_candidates` のうち有効な `p_success` が最大の候補を返す最終層
    （HF-001でv1.27改訂、§4.11）。

    入力は `pools.geo_candidates`（旧v1.18の `dims_candidates` 契約は、INCLUSION/OVERLAP/
    CEILING未確認の候補を最終選択し得たため撤回した。空コンテナ初手のようにDIMSのみ通過し
    INCLUSION不合格の候補しか無い場合に、その不合格候補を選んでしまうことがHF-001の
    直接原因だった）。追加のマスク・L_PATH・特徴計算は開始しない。`l_path_cache` が
    `False`（既知のL_PATH不合格）と判明した候補も選択対象から除外する。DIMS〜CEILINGの
    4段階合格は保証するが、L_PATH合格は保証しない最終Candidate層である。

    Args:
        state: 現在の `PackingState`（本層では未使用、シグネチャ統一のため受け取る）。
        budget: 全層で共有する `StepBudget`（本層では未使用）。
        pools: 候補集合（`geo_candidates`/`l_path_cache` を使用）。
        pp: 配置判定パラメータ（本層では未使用）。
        stage_params: 本層では未使用。

    Returns:
        有効候補（`math.isfinite(cand.p_success) and 0.0<=cand.p_success<=1.0` かつ
        `l_path_cache` が `False` でない）のうち `p_success` 最大（同点は基本順序）。
        有効候補が0件、または層内例外なら `None`。
    """
    try:
        best: Candidate | None = None
        for cand in pools.geo_candidates:
            p = cand.p_success
            if not (math.isfinite(p) and 0.0 <= p <= 1.0):
                continue
            if pools.l_path_cache.get(_cache_key(cand)) is False:
                continue
            if best is None or p > best.p_success:
                best = cand
        return best
    except Exception:
        return None


def safe_decide(
    layers: Sequence[Layer], state: "PackingState", budget: StepBudget, telemetry: dict,
) -> Candidate | None:
    """4層を順に呼び、最初にCandidateを返した層の結果を確定する（§4.11）。

    各層の例外はここで捕捉し `telemetry["layer_error"]` へ記録して次層へ進む
    （`safe_decide` 自身は例外を外へ送出しない）。`make_action` は呼ばず、Candidateも
    変更しない。全層が同一 `state`／同一 `budget` オブジェクトを共有する。

    Args:
        layers: `Callable[[state, budget], Candidate | None]` の層列（通常は
            `[layer1_main, layer2_dblf_strict, layer3_first_fit, layer4_max_p]` を
            `functools.partial` で `pools`/`pp`/`stage_params` を束縛したもの）。
        state: 現在の `PackingState`。
        budget: 全層で共有する `StepBudget`。
        telemetry: 呼び出し側が用意した辞書。`decided_layer`/`layer_error` を
            `setdefault` で初期化し、実行結果に応じて更新する。

    Returns:
        最初にCandidateを返した層の結果。全層 `None` または全層例外なら `None`。
    """
    telemetry.setdefault("decided_layer", 0)
    telemetry.setdefault("layer_error", [])

    for layer_index, layer in enumerate(layers, start=1):
        try:
            result = layer(state, budget)
        except Exception as exc:  # noqa: BLE001（各層の例外を記録し次層へ進む契約そのもの）
            telemetry["layer_error"].append(
                {"layer": layer_index, "error_type": type(exc).__name__}
            )
            continue
        if result is not None:
            telemetry["decided_layer"] = layer_index
            return result
    return None
