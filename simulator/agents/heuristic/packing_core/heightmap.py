"""Heightmap 配置エンジン（Phase H1、HF-012）。

2.5D heightmap（per-cell 上面高さ）上で候補を全 footprint 走査し、fill 寄りスコアで並べ、
inclusion（`contains_oriented_box`）＋support（heightmap 由来）＋transport（`check_l_path`）の
実行可能性をマージン緩和カスケード（strict→relaxed→desperate）で確認して1手を選ぶ。EMS を
使わず、`ContainerSpace` の幾何（`floor_z`/`ceil_z`/`shelf_boxes`）と既配置荷物から heightmap を
毎ステップ構築する。`agents/heuristic/agent.py::_run_pipeline` から呼ばれ、EMS 列挙〜
`safe_decide` を置換する。搬入プロキシ・inclusion・geometry は既存の検証済み関数を再利用する。

公開: `decide_placement(state, budget) -> (item_idx, container_idx, pos_rel, orientation) | None`。
"""
from __future__ import annotations

import os

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .constants import (
    HM_CASCADE,
    HM_CEIL_CLEARANCE,
    HM_COARSE_TOPK,
    HM_DENSIFY_MAX,
    HM_DENSIFY_STEP,
    HM_POOL_CAP,
    HM_SHELF_STANDOFF,
    HM_SOFT_FLAT_TOL,
    HM_STRAT_BINS_X,
    HM_STRAT_BINS_Y,
    HM_STRAT_PER_BIN,
    HM_FLOOR_FIRST,
    HM_FRONTIER_SLACK,
    HM_W_VOL_HIGH,
    HM_PRIO_RESERVE,
    HM_SOFT_RESERVE,
    HM_SOFT_VETO,
    HM_HEAVY_CEIL,
    HM_HEAVY_CEIL_Q,
    HM_PRIO_RESERVE_GLOBAL,
    HM_BREADTH_MUL,
    HM_WIDE_BREADTH,
    HM_FLOOR_VOL_SKIP_PRIO,
    HM_MAX_ROUGH,
    HM_SUPPORT_TOL,
    HM_WALL_CLEARANCE,
    HM_W_BIG,
    HM_W_BLOCK,
    HM_LOOKAHEAD_K,
    HM_SLAB_H,
    HM_SLAB_REF_Q,
    HM_LOOKAHEAD_PROBE,
    HM_W_ZBLOCK,
    HM_W_COG,
    HM_W_FLOOR_VOL,
    HM_W_DEPTH,
    HM_W_FLAT,
    HM_W_HEIGHT,
    HM_W_LEFT,
    HM_W_NONPRIO_IN_PRIO_CONTAINER_PENALTY,
    HM_W_PRIO_CONTAINER_BONUS,
    HM_W_PRIO_CONTAINER_PENALTY,
    HM_W_PRIO_HIGH,
    HM_W_PRIO_VIOL,
    HM_W_HARDHARD,
    HM_W_SLAB,
    HM_W_TIPRISK,
    HM_W_SOFT_CAP,
    HM_W_SOFT_VIOL,
    HM_W_TALL,
    PlacementParams,
)
from .container_space import (
    ContainerSpace,
    _cell_centers,
    cells_of_aabb,
    contains_oriented_box,
)
from .geometry import oriented_size
from .masks import check_l_path
from .types import Candidate

_NEG_INF = -1.0e18

# HF-012 Phase H2: 非fillサブスコア（cog/placement/soft）向けスコア項の一括トグル（既定 ON）。
# 環境変数 GH_HM_H2=0 で H1 挙動へ戻す（A/B・非退行計測用。提出時は既定 ON）。
_H2_ENABLED = os.environ.get("GH_HM_H2", "1") != "0"
# 項別トグルの既定値（本番提出で較正）。教訓: 品質サブスコア ≈ raw_quality(配置) × gate(num_placed)、
# gate は num_placed≈0.5-0.6 で急峻。num_placed を下げる/配置を乱す変更は全サブスコアを暴落させる。
#   soft-cap(v15): 本番 soft_score 43.9→22.95 に暴落（"soft を蓋に"プロキシは逆符号）→ 既定 OFF。
#   cog(v16): 本番 cog 60→45・全品質暴落。cog-on が num_placed 0.588→0.552 に下げ gate を滑り落ちた
#     （ローカル COG は改善したが本番では逆）→ 既定 OFF。num_placed を下げる変更は不可。
#   prio(優先容器 bonus/penalty): ローカルは inert（優先容器なし）、本番 placement_score 用 → 既定 ON。
# いずれも GH_HM_COG/SOFT/PRIO=1/0 で明示切替でき、A/B・回帰計測に使う。
_H2_COG = _H2_ENABLED and os.environ.get("GH_HM_COG", "0") != "0"
_H2_SOFT = _H2_ENABLED and os.environ.get("GH_HM_SOFT", "0") != "0"
_H2_PRIO = _H2_ENABLED and os.environ.get("GH_HM_PRIO", "1") != "0"
# soft を flat 制限（塔化防止, fill を強く食う）と soft-cap（低頭上ポケット誘導, 安価）に分離。
# 既定は _H2_SOFT に従うが GH_HM_SOFT_FLAT / GH_HM_SOFT_CAP で個別に切替可（soft_item_score の A/B 用）。
_H2_SOFT_FLAT = os.environ.get("GH_HM_SOFT_FLAT", "1" if _H2_SOFT else "0") != "0"
# HF-022 v22: soft-cap は **既定 ON にベイク**（重みは constants.HM_W_SOFT_CAP = −0.5 ＝符号反転）。
# v15 の本番実測が「np を動かさず品質4項を大きく動かす」偏微分を与えたので、その逆向きを測る。
# `GH_HM_SOFT_CAP=0` で v14 挙動へ戻せる。v15 の再現は `GH_HM_SOFT=1 GH_HM_W_SOFT_CAP=4.0`。
# v36: **既定 ON にベイク**（重みは constants.HM_W_SOFT_CAP = 4.0 の正の向き）。
# それまでコードは `"1" if _H2_SOFT else "0"` で _H2_SOFT が既定 False のため**完全に不活性**
# だった（コメントは「既定 ON」と書いてあったが実体は OFF。v22 が符号反転版で回帰した後に
# 戻され、コメントだけ残っていた）。つまり**正の向きは一度も本番で試されていない**。
# soft_item_score の違反内訳（v35, dr10 239個）は 正常 72.4% / 未積載 3.8% / **下敷き 23.8%** で
# 下敷きが律速。本項は「頭上余裕が大きい場所ほど減点」＝soft を蓋の位置へ誘導するもので、
# 実測で **下敷き 23.8% → 13.8%、局所 soft 70.45 → 80.95**。fill も 26.37 → 27.87 と上がる。
# 3ベンチの ΔPublic +0.65/+0.72/+0.99（soft の転移比 0.39 実測、符号 3/3 一致で計算）。
# W=7 では fill が 22.96 へ崖落ちするので既定の 4.0 が山頂。`GH_HM_SOFT_CAP=0` で v35 へ戻る。
# **v36 で ON にして Public 60.00 → 54.00（−6.00）と大退行。既定 OFF に戻した。**
# 局所 soft は 70.45 → 80.95（+10.50）と設計どおり上がり下敷きも 23.8% → 13.8% に減ったのに、
# **本番 soft は 61.15 → 45.40（−15.75）で符号が反転**（実効転移比 −1.50）。同時に
# cog −9.28 / stability −7.47 / placement −5.70。soft を低頭上ポケット（＝高い位置）へ
# 誘導することが本番では逆効果になる。`GH_HM_SOFT_CAP=1` で再現できる。
_H2_SOFT_CAP = os.environ.get("GH_HM_SOFT_CAP", "0") != "0"

# HF-022 Phase 0.1: 投了（decide_placement→None）の原因内訳を採るための診断フック。
# 既定 OFF。本番経路には None を返す直前の `if _DIAG:` 1個しか増えない（採用経路は不変）。
# 全提出が「候補ゼロ→_emergency_action→搬入路失敗」で終わっているため、その瞬間に
# inclusion / support / l_path のどれが何件落としたかを数え、次に緩める場所を決める。
_DIAG = os.environ.get("GH_DIAG", "0") != "0"
# HF-022 Phase 4: **診断専用**。搬入路（l_path）チェックを外して「搬入制約が律速か、幾何が律速か」
# を切り分ける。外して配置率が跳ね上がるなら搬入制約が壁＝方策改善の余地あり、変わらないなら
# 幾何（体積・形状）が壁＝方策では届かない。提出物では絶対に有効化しない（公式判定を無視するため
# 実機では is_valid=False で即終了する）。
_DIAG_NO_LPATH = os.environ.get("GH_DIAG_NO_LPATH", "0") != "0"
_DIAG_LOG: list[dict] = []   # 診断時のみ追記（scripts/diag_termination.py が読む）


# --- sliding-window 縮約（全 footprint 窓を一括評価） -------------------------------

def _sliding_max2d(grid: np.ndarray, wx: int, wy: int) -> np.ndarray:
    """(nx,ny) 格子の wx×wy 窓ごと最大値。戻り (nx-wx+1, ny-wy+1)。

    max は分離可能（2D窓の最大 = 各軸1Dスライディング最大の合成）なので、O(wx·wy) の
    2D reduce ではなく O(wx+wy) の2回の1D reduce で計算する（大窓で ~10x 高速、純numpy）。
    """
    a = grid
    if wx > 1:
        a = sliding_window_view(a, wx, axis=0).max(axis=-1)
    if wy > 1:
        a = sliding_window_view(a, wy, axis=1).max(axis=-1)
    return a


def _sliding_min2d(grid: np.ndarray, wx: int, wy: int) -> np.ndarray:
    a = grid
    if wx > 1:
        a = sliding_window_view(a, wx, axis=0).min(axis=-1)
    if wy > 1:
        a = sliding_window_view(a, wy, axis=1).min(axis=-1)
    return a


# --- コンテナ heightmap モデル ------------------------------------------------------

class ContainerHeightModel:
    """1コンテナの 2.5D heightmap（lower＝床/床積み、upper＝棚上）を既配置から構築する。

    lower[i,j] = max(cut反映の床, 床側既配置荷物の上面)。upper[i,j] = 棚上面(+standoff) と
    棚上既配置の上面（棚被覆セルのみ、非被覆は -inf）。lower_ceiling[i,j] = 棚下面 or 天井
    （-CEIL_CLEARANCE）。soft_top/prio_top は各セル上面を成す荷物の種別フラグ（違反減点用）。
    """

    def __init__(self, space: ContainerSpace, placed: list) -> None:
        self.space = space
        self.cell = float(space.cell)
        inner_min = np.asarray(space.inner_min_rel, dtype=np.float64)
        inner_max = np.asarray(space.inner_max_rel, dtype=np.float64)
        self.inner_min = inner_min
        self.inner_max = inner_max
        nx, ny, xs, ys = _cell_centers(inner_min, inner_max, self.cell)
        self.nx, self.ny, self.xs, self.ys = nx, ny, xs, ys

        # 棚上面グリッド（被覆セルの棚上面 max、非被覆 -inf）と被覆マスク。
        shelf_top = np.full((nx, ny), _NEG_INF, dtype=np.float64)
        shelf_cover = np.zeros((nx, ny), dtype=bool)
        for bmin, bmax in space.shelf_boxes:
            sx, sy = cells_of_aabb(space, np.asarray(bmin), np.asarray(bmax))
            if sx.stop > sx.start and sy.stop > sy.start:
                shelf_top[sx, sy] = np.maximum(shelf_top[sx, sy], float(bmax[2]))
                shelf_cover[sx, sy] = True
        self.shelf_cover = shelf_cover
        shelf_top_for_route = np.where(shelf_cover, shelf_top, np.inf)  # 非被覆は +inf（下層扱い）

        # lower = cut床（+WALL_CLEARANCE）、upper = 棚上面+standoff（被覆のみ）。
        # 床を WALL_CLEARANCE 持ち上げる：床直置きが inclusion の底面クリアランス（正マージン）を
        # 満たせるようにする（物理沈降でこの隙間は無害に閉じる。sample の lower baseline 同型）。
        lower = np.array(space.floor_z, dtype=np.float64) + HM_WALL_CLEARANCE
        upper = np.where(shelf_cover, shelf_top + HM_SHELF_STANDOFF, _NEG_INF)
        soft_low = np.zeros((nx, ny), dtype=bool)
        prio_low = np.zeros((nx, ny), dtype=bool)
        soft_up = np.zeros((nx, ny), dtype=bool)
        prio_up = np.zeros((nx, ny), dtype=bool)

        for item in placed:
            amin = np.asarray(item.aabb_min_rel, dtype=np.float64)
            amax = np.asarray(item.aabb_max_rel, dtype=np.float64)
            ix, iy = cells_of_aabb(space, amin, amax)
            if ix.stop <= ix.start or iy.stop <= iy.start:
                continue
            top = float(amax[2])
            # 棚上（底面が棚上面近傍以上）なら upper、それ以外は lower へ振り分け。
            bottom = float(amin[2])
            on_shelf = np.any(shelf_cover[ix, iy]) and bottom >= (
                float(np.min(shelf_top_for_route[ix, iy])) - 0.02
            )
            if on_shelf:
                cur = upper[ix, iy]
                win = top > cur
                upper[ix, iy] = np.where(win, top, cur)
                if item.is_soft:
                    soft_up[ix, iy] = np.where(win, True, soft_up[ix, iy])
                if item.is_priority:
                    prio_up[ix, iy] = np.where(win, True, prio_up[ix, iy])
            else:
                cur = lower[ix, iy]
                win = top > cur
                lower[ix, iy] = np.where(win, top, cur)
                if item.is_soft:
                    soft_low[ix, iy] = np.where(win, True, soft_low[ix, iy])
                if item.is_priority:
                    prio_low[ix, iy] = np.where(win, True, prio_low[ix, iy])

        self.lower = lower
        self.upper = upper
        self.soft_low = soft_low
        self.prio_low = prio_low
        self.soft_up = soft_up
        self.prio_up = prio_up
        # 頭上上限（候補生成用、構造物より CEIL_CLEARANCE 内側）。
        self.lower_ceiling = np.asarray(space.ceil_z, dtype=np.float64) - HM_CEIL_CLEARANCE
        self.upper_ceiling = float(inner_max[2]) - HM_CEIL_CLEARANCE
        # HF-022 Phase 4: 床被覆率（素の床ベースラインより高いセルの割合）。層規律の判定に使う。
        base = np.array(space.floor_z, dtype=np.float64) + HM_WALL_CLEARANCE
        self.floor_covered = float(np.mean(lower > base + 1e-6))
        # HF-022 Phase 7: Xレーンごとの「フロンティア」= そのレーンで**最も手前**の占有 j。
        # 奥→手前の一方向構成では、新しい配置はここより手前（j が小さい側）に限る。
        # 何も置かれていないレーンは ny-1（最奥まで自由）。
        occ = lower > (base + 1e-6)                     # (nx, ny) 占有マスク
        jj = np.arange(ny)[None, :]
        masked = np.where(occ, jj, ny)                  # 占有セルの j、非占有は ny
        self.lane_frontier = masked.min(axis=1)         # (nx,) 各レーンの最小占有 j
        self.lane_frontier = np.where(self.lane_frontier >= ny, ny - 1, self.lane_frontier)


# --- 候補生成（向きごとに全 footprint 窓を評価） -----------------------------------

def _select_cells(score: np.ndarray, valid: np.ndarray, per_bin: int, coarse_topk: int) -> list[tuple[int, int]]:
    """スコア配列上で「全体上位 coarse_topk ＋ Y×X ビンごと上位 per_bin」を選び (i,j) 群を返す。

    候補を全 valid セル分 materialize すると数万個になり遅い。ここで配列演算だけで上位に
    絞り、呼び出し側は選抜セルのみ `Candidate` 化する（sample の _topk_layer 同型）。
    """
    ni, nj = score.shape
    flat = np.where(valid, score, _NEG_INF).ravel()
    good = flat > (_NEG_INF / 2.0)
    if not np.any(good):
        return []
    keep: set[int] = set()
    n_good = int(good.sum())
    k = min(coarse_topk, n_good)
    if k > 0:
        top = np.argpartition(-flat, k - 1)[:k]
        keep.update(int(t) for t in top if good[t])
    ii, jj = np.divmod(np.arange(flat.size), nj)      # i=x-anchor, j=y-anchor
    by = np.clip(jj * HM_STRAT_BINS_Y // max(nj, 1), 0, HM_STRAT_BINS_Y - 1)
    bx = np.clip(ii * HM_STRAT_BINS_X // max(ni, 1), 0, HM_STRAT_BINS_X - 1)
    bid = bx * HM_STRAT_BINS_Y + by
    for b in np.unique(bid[good]):
        m = np.where((bid == b) & good)[0]
        if m.size:
            t = m[np.argsort(-flat[m])[:per_bin]]
            keep.update(int(x) for x in t)
    return [(int(idx // nj), int(idx % nj)) for idx in keep]


def _blocked_fraction(layer: np.ndarray, ceiling: np.ndarray, wx: int, wy: int) -> np.ndarray:
    """各アンカー窓が「奥側に作る影」の体積比 (0-1) を返す。形状は land と同じ (ni, nj)。

    搬入は側扉 y=-W/2（= j=0 側）から +Y へスライドするので、窓 [i:i+wx)×[j:j+wy) に物を
    置くと、同じ X レーンのより奥（j' >= j+wy）にある自由空間へは到達しにくくなる。
    その自由体積を「コンテナ全体の残り自由体積」で割った比を返す。

    計算量は O(nx·ny)：j 方向の suffix 和 → x 方向の窓和（どちらも cumsum）。
    セル面積は分母分子で相殺するため、自由“高さ”の和だけで比が出る。
    """
    lay = np.where(layer > (_NEG_INF / 2.0), layer, ceiling)   # 無効セルは自由高 0 扱い
    free = np.maximum(ceiling - lay, 0.0)                      # (nx, ny) セルごとの自由高 [m]
    nx, ny = free.shape
    total = float(free.sum())
    ni, nj = nx - wx + 1, ny - wy + 1
    if total <= 1e-9 or ni <= 0 or nj <= 0:
        return np.zeros((max(ni, 0), max(nj, 0)), dtype=np.float64)

    suf = np.zeros((nx, ny + 1), dtype=np.float64)             # suf[:, j] = Σ_{j'>=j} free[:, j']
    suf[:, :ny] = np.cumsum(free[:, ::-1], axis=1)[:, ::-1]
    behind = suf[:, wy:wy + nj]                                # アンカー j の「奥」は j+wy から
    cs = np.zeros((nx + 1, nj), dtype=np.float64)              # x 窓和
    np.cumsum(behind, axis=0, out=cs[1:])
    return (cs[wx:wx + ni] - cs[:ni]) / total


def _zblocked_fraction(layer: np.ndarray, ceiling: np.ndarray, land: np.ndarray,
                       top_z: np.ndarray, wx: int, wy: int, nb: int = 8) -> np.ndarray:
    """**z帯を考慮した**「奥側に作る影」の体積比 (0-1)。形状は land と同じ (ni, nj)。

    HF-022 Phase 4。搬入路チェックを外すと同じ貪欲が 97.2% を配置できる（現状 61.8%）ので、
    壁は幾何ではなく **到達可能性** だと確定した。公式の搬入は
      底面が resting surface から 0.05m 以内 → 目標高のまま水平搬入
      それ以外 → +0.08m 持ち上げて水平搬入
    なので、ある配置が塞ぐのは「同じXレーンでより奥」かつ「自分が占める z 帯 + 0.08m」に限られる。

    先に入れた `_blocked_fraction` は z を無視して奥の自由体積すべてを罰したため、低い荷物にも
    過大な罰を与え W_DEPTH と競合して単調に悪化した（v-）。本関数は z を帯に離散化し、
    候補が占める帯だけを数える。

    計算量: 帯数 nb について O(nb·nx·ny)。帯ごとに j 方向 suffix 和と x 方向窓和を取り、
    帯方向の累積で候補ごとの区間和を O(1) 参照する。
    """
    lay = np.where(layer > (_NEG_INF / 2.0), layer, ceiling)
    nx, ny = lay.shape
    ni, nj = nx - wx + 1, ny - wy + 1
    if ni <= 0 or nj <= 0:
        return np.zeros((max(ni, 0), max(nj, 0)), dtype=np.float64)
    zlo = float(np.min(lay))
    zhi = float(np.max(ceiling))
    if not (zhi > zlo):
        return np.zeros((ni, nj), dtype=np.float64)
    edges = np.linspace(zlo, zhi, nb + 1)

    # 帯ごとの自由高 → j 方向 suffix → x 方向窓和。cs_b[b] は (ni, nj)。
    cum = np.zeros((nb + 1, ni, nj), dtype=np.float64)   # 帯 0..b-1 の累積
    total = 0.0
    for b in range(nb):
        lo, hi = edges[b], edges[b + 1]
        free_b = np.clip(np.minimum(hi, ceiling) - np.maximum(lo, lay), 0.0, None)
        total += float(free_b.sum())
        suf = np.zeros((nx, ny + 1), dtype=np.float64)
        suf[:, :ny] = np.cumsum(free_b[:, ::-1], axis=1)[:, ::-1]
        behind = suf[:, wy:wy + nj]
        cs = np.zeros((nx + 1, nj), dtype=np.float64)
        np.cumsum(behind, axis=0, out=cs[1:])
        cum[b + 1] = cum[b] + (cs[wx:wx + ni] - cs[:ni])
    if total <= 1e-9:
        return np.zeros((ni, nj), dtype=np.float64)

    # 候補が占める z 区間 [land, top_z + 0.08]（持ち上げ分を足す）に重なる帯を選ぶ
    dz = (zhi - zlo) / nb
    b_lo = np.clip(((land - zlo) / dz).astype(np.int64), 0, nb)
    b_hi = np.clip(np.ceil((top_z + 0.08 - zlo) / dz).astype(np.int64), 0, nb)
    b_hi = np.maximum(b_hi, b_lo)
    ii = np.arange(ni)[:, None]
    jj = np.arange(nj)[None, :]
    return (cum[b_hi, ii, jj] - cum[b_lo, ii, jj]) / total


# HF-022 Phase 9: 優先品の総量。`optimize()` が全荷物から設定する（policy は look_ahead=1 で
# 将来を見られないため、ここに預ける）。未設定なら予約は無効。
PRIO_TOTAL: dict[str, float] = {"n": 0.0, "vol": 0.0, "hmax": 0.0}
# HF-025: 重量拒否の質量しきい値。`optimize()` が全荷物の分位から設定する。
MASS_THRESHOLD: dict[str, float] = {"v": 0.0}
SOFT_TOTAL: dict[str, float] = {"n": 0.0, "vol": 0.0, "hmax": 0.0}


def _layer_candidates(
    model: ContainerHeightModel, item, container_idx: int, orn: int, osize: np.ndarray,
    *, upper: bool, per_bin: int, coarse_topk: int,
    cont_prio: bool = False, prio_exists: bool = False, reserve_h: float = 0.0,
    reserve_soft: float = 0.0,
) -> list[Candidate]:
    """1コンテナ・1向き・1層の全 footprint 窓から候補を生成（スコア付き）。

    fill 寄りの基本スコア（depth/height/left/flat/tall/big）に加え、HF-012 H2 の非fill項:
    cog（`-W_COG·mass·land`）、soft-cap（softのみ `-W_SOFT_CAP·max(headroom,0)`）、優先容器
    （item/container の優先度に応じた bonus/penalty スカラー）を反映する。soft の非flat姿勢は
    `soft_nonflat` フラグを立て、呼び出し側が desperate 段以外で除外する。
    """
    cell = model.cell
    nx, ny, xs, ys = model.nx, model.ny, model.xs, model.ys
    hx, hy, hz = float(osize[0]) / 2.0, float(osize[1]) / 2.0, float(osize[2]) / 2.0
    wx = max(1, int(np.ceil(osize[0] / cell - 1e-6)))
    wy = max(1, int(np.ceil(osize[1] / cell - 1e-6)))
    if wx > nx or wy > ny:
        return []

    layer = model.upper if upper else model.lower
    land = _sliding_max2d(layer, wx, wy)                       # 着地面（窓内max）
    raw_min = _sliding_min2d(layer, wx, wy)                    # 窓内min（平坦度）
    soft_grid = model.soft_up if upper else model.soft_low
    prio_grid = model.prio_up if upper else model.prio_low
    soft_hit = _sliding_max2d(soft_grid.astype(np.float64), wx, wy) > 0.5
    prio_hit = _sliding_max2d(prio_grid.astype(np.float64), wx, wy) > 0.5

    # 有効性：天井・（upper は棚完全被覆）。ceil_arr は soft-cap の headroom 算出にも使う。
    top_z = land + osize[2]
    if upper:
        ceil_arr = np.full_like(land, float(model.upper_ceiling))
        ceil_ok = top_z <= (ceil_arr + 1e-6)
        cover = _sliding_min2d(model.shelf_cover.astype(np.float64), wx, wy) >= (1.0 - 1e-6)
        finite = land > (_NEG_INF / 2.0)
        valid = ceil_ok & cover & finite
    else:
        ceil_arr = _sliding_min2d(model.lower_ceiling, wx, wy)
        valid = top_z <= (ceil_arr + 1e-6)

    # HF-022 Phase 9/10: 上部の予約帯。優先品予約（優先容器のみ）と soft 予約（全容器）を
    # 独立に持ち、その荷物が該当しない帯の**最大**を禁止幅とする。
    #   優先品でも soft でもない → 両方の帯を避ける
    #   優先品だが soft でない   → soft 帯だけ避ける（優先帯は自分のもの）
    #   soft だが優先品でない    → 優先帯だけ避ける
    #   両方                     → 制約なし
    if HM_SOFT_VETO and not bool(item.is_soft):
        # HF-022 Phase 11: soft 上面への非soft 着地を無効化（下敷き違反の根絶）。
        # soft_hit は「窓内のどこかのセル上面が soft」を意味する（既に計算済み）。
        valid = valid & ~soft_hit

    _band = 0.0
    if reserve_h > 0.0 and cont_prio and not bool(item.is_priority):
        _band = max(_band, reserve_h)
    if reserve_soft > 0.0 and not bool(item.is_soft):
        _band = max(_band, reserve_soft)
    if _band > 0.0:
        valid = valid & (top_z <= (ceil_arr - _band + 1e-6))

    if HM_MAX_ROUGH < 1.0e8:
        # HF-022 安定性ゲート: 接地面の高低差が大きい窓は、支持率を満たしていても
        # 深い谷にせり出したカンチレバーになり沈降で崩れる（崩れ15件の最小 roughness 0.266m）。
        # `land - raw_min` は上で計算済みなので実質無コスト。
        valid = valid & ((land - raw_min) <= HM_MAX_ROUGH)

    if not np.any(valid):
        return []

    # 窓アンカー(i,j) → 中心座標。窓は cells [i:i+wx),[j:j+wy) を覆う。
    ni, nj = land.shape
    ci = xs[np.arange(ni)] + (wx - 1) * cell / 2.0                # (ni,)
    cj = ys[np.arange(nj)] + (wy - 1) * cell / 2.0                # (nj,)
    cx = np.broadcast_to(ci[:, None], (ni, nj))
    cy = np.broadcast_to(cj[None, :], (ni, nj))
    y_back = cy + hy

    is_soft = bool(item.is_soft)
    is_prio = bool(item.is_priority)
    vol = float(np.prod(np.asarray(item.size, dtype=np.float64)))
    mass = float(item.weight)

    if HM_HEAVY_CEIL >= 0.0 and mass >= MASS_THRESHOLD["v"] > 0.0:
        # HF-025: 質量上位の荷物は内高の HM_HEAVY_CEIL より上に着地させない（硬い拒否）。
        # 公式 cog は質量加重の重心で「置いた荷物だけ」を数えるので、高所へ行く重量物は
        # 置かない方がスコアが上がる。減点では貪欲が押し切るため候補段階で無効化する。
        _zf = float(model.space.inner_min_rel[2])
        _zt = float(model.space.inner_max_rel[2])
        _lim = _zf + HM_HEAVY_CEIL * (_zt - _zf)
        valid = valid & (land <= _lim)

    # soft の非flat姿勢（最小半寸 + 許容 を超える高さ）は塔化＝不安定。フラグを立て、
    # 呼び出し側の cascade が desperate 段以外で除外する（H2 有効時のみ）。
    min_half = float(np.min(np.asarray(item.size, dtype=np.float64))) / 2.0
    soft_nonflat = _H2_SOFT_FLAT and is_soft and (hz > min_half + HM_SOFT_FLAT_TOL)

    score = (
        HM_W_DEPTH * y_back
        - HM_W_HEIGHT * land
        - HM_W_LEFT * cx
        - HM_W_FLAT * (land - raw_min)
        - HM_W_TALL * hz
        + HM_W_BIG * vol
    )
    if not is_soft:
        score = score - HM_W_SOFT_VIOL * soft_hit.astype(np.float64)
    if not is_prio:
        score = score - HM_W_PRIO_VIOL * prio_hit.astype(np.float64)
    if HM_W_HARDHARD != 0.0 and not upper and not is_soft:
        # HF-031: 摩擦仮説の検証用（docs/findings_2026-07.md §15）。7SKU閉集合は
        # is_soft=False の全SKUが lateralFriction=0.4 固定（is_soft=True は0.6-0.8）で、
        # hard-on-hard 接触が系内最低の組合せ摩擦になる（コンテナ内壁は0.8固定）。
        # 床（bare floor）は摩擦0.8で対象外、soft上面は既に HM_W_SOFT_VIOL が扱うので、
        # 「非soft品が既存の非soft品の上面に着地する」窓だけを減点する。既定 0（無効）。
        floor_base = np.asarray(model.space.floor_z, dtype=np.float64) + HM_WALL_CLEARANCE
        on_bare_floor = land <= (_sliding_max2d(floor_base, wx, wy) + 1e-6)
        hard_surface_hit = (~soft_hit) & (~on_bare_floor)
        score = score - HM_W_HARDHARD * hard_surface_hit.astype(np.float64)
    if HM_W_TIPRISK != 0.0 and not upper:
        # HF-032: HM_W_HARDHARD の後継（findings §15.4で棄却）。荷物の形状（倒れやすさ）を
        # 表面摩擦と掛け合わせる。tip_ratio = 高さ / 底面最小辺（大きいほど倒れやすい）。
        # 低摩擦面（非soft品の露出上面。床0.8・soft0.6-0.8より低い0.4）へ着地する窓を、
        # is_soft を問わず tip_ratio に比例して減点する。既定 0（無効）。
        floor_base2 = np.asarray(model.space.floor_z, dtype=np.float64) + HM_WALL_CLEARANCE
        on_bare_floor2 = land <= (_sliding_max2d(floor_base2, wx, wy) + 1e-6)
        hard_surface_hit2 = (~soft_hit) & (~on_bare_floor2)
        base_min = max(min(float(osize[0]), float(osize[1])), 1e-6)
        tip_ratio = float(osize[2]) / base_min
        score = score - HM_W_TIPRISK * tip_ratio * hard_surface_hit2.astype(np.float64)
    if _H2_COG:
        # cog_score: 重量物ほど低く着地させる。
        score = score - HM_W_COG * mass * land
    if _H2_SOFT_CAP and is_soft:
        # soft_item_score: soft は列の蓋。頭上余裕が大きい場所ほど減点し低頭上ポケットへ誘導。
        score = score - HM_W_SOFT_CAP * np.maximum(ceil_arr - top_z, 0.0)
    if HM_W_VOL_HIGH != 0.0:
        # HF-022 Phase 7: cog = Σbottom_i の**重みなし**和なので、高所コストは体積に依らず一定。
        # ならば高所は体積の大きい荷物に使うのが最適（同じ cog 支出で fill 収入が大きい）。
        score = score + HM_W_VOL_HIGH * vol * land

    if HM_W_FLOOR_VOL > 0.0 and not upper and not (HM_FLOOR_VOL_SKIP_PRIO and is_prio):
        # HF-022: 床直置き（= 素の床ベースラインに着地する窓）は沈降後に fill から除外される。
        # 体積に比例して減点し、床には小物・上段には大物を誘導する。棚上(upper)は対象外。
        floor_base = np.asarray(model.space.floor_z, dtype=np.float64) + HM_WALL_CLEARANCE
        on_bare_floor = land <= (_sliding_max2d(floor_base, wx, wy) + 1e-6)
        score = score - HM_W_FLOOR_VOL * vol * on_bare_floor.astype(np.float64)
    if HM_W_SLAB > 0.0 and not upper:
        # HF-022 Phase 6: 現在の作業面（下層サーフェスの分位点）からの飛び出しを罰する。
        # 診断で 44% の手が高さ帯を跨いでおり、どの帯の回廊も完成していなかった。
        _lv = layer[layer > (_NEG_INF / 2.0)]
        if _lv.size:
            _ref = float(np.percentile(_lv, HM_SLAB_REF_Q))
            score = score - HM_W_SLAB * np.maximum(land - _ref, 0.0) / max(HM_SLAB_H, 1e-6)
    if HM_W_ZBLOCK > 0.0:
        # HF-022 Phase 4: z帯を考慮した搬入路シャドウ。ceil_arr は窓適用後なので生グリッドを渡す。
        _cf = (np.full(layer.shape, float(model.upper_ceiling)) if upper else model.lower_ceiling)
        score = score - HM_W_ZBLOCK * _zblocked_fraction(layer, _cf, land, top_z, wx, wy)
    if HM_W_BLOCK > 0.0:
        # HF-022: 奥を塞ぐ配置を減点し、搬入路（l_path）で候補が全滅するのを遅らせる。
        # ceil_arr は窓適用後の (ni,nj) なので、ここでは生グリッド (nx,ny) を渡す。
        ceil_full = (np.full(layer.shape, float(model.upper_ceiling)) if upper
                     else model.lower_ceiling)
        score = score - HM_W_BLOCK * _blocked_fraction(layer, ceil_full, wx, wy)
    if is_prio and HM_W_PRIO_HIGH != 0.0:
        # HF-022 v26: 優先品だけ「高く置く」を加点（placement_score）。v24/v25 では置かれる優先品が
        # 平均1個で z_rel 0.036（ほぼ床＝取り出しにくい）だった。1荷物にしか効かないので
        # np/fill への副作用はほぼ無く、cog（駆動因は item selection）も動かさない。
        score = score + HM_W_PRIO_HIGH * land
    if _H2_PRIO:
        # placement_score: 優先度の一致でスカラー加減点（層内一律）。
        if is_prio and cont_prio:
            score = score + HM_W_PRIO_CONTAINER_BONUS
        elif is_prio and (not cont_prio) and prio_exists:
            score = score - HM_W_PRIO_CONTAINER_PENALTY
        elif (not is_prio) and cont_prio:
            score = score - HM_W_NONPRIO_IN_PRIO_CONTAINER_PENALTY
    score = np.where(valid, score, _NEG_INF)

    # HF-022 Phase 4: 層規律のため「素の床に直置きか」を候補に印す。
    # 公式 check_transport_path は底面が resting surface から 0.05m 以内なら effective_start_z=0
    # ＝目標高のまま水平搬入。それ以外は 8cm 持ち上げるので、扉から目標まで target_z+0.08 の
    # 高さで回廊が空いている必要がある。つまり床層は順序自由だが積み上げは階段を要求する。
    if HM_FRONTIER_SLACK >= 0 and not upper:
        # HF-022 Phase 7: 奥→手前の一方向フロンティア制約。窓 [i,i+wx)×[j,j+wy) の
        # **最奥端 j+wy-1** が、その窓が覆う全レーンのフロンティア + slack 以下であることを要求する。
        # これにより「既に手前へ進んだレーンで、また奥に戻る」配置を候補段階で排除し、
        # 搬入路が塞がれる構造を作らない（搬入チェックに弾かれるのを待たない）。
        fr = model.lane_frontier
        nxw = fr.shape[0] - wx + 1
        if nxw > 0:
            # 窓が覆うレーンのフロンティア最小値（最も手前まで進んでいるレーンに合わせる）
            lane_min = np.empty(nxw, dtype=fr.dtype)
            for a in range(nxw):
                lane_min[a] = fr[a:a + wx].min()
            j_far = np.arange(nj) + wy - 1                  # 窓の最奥端 j
            allowed = j_far[None, :] <= (lane_min[:, None] + HM_FRONTIER_SLACK)
            valid = valid & allowed

    if HM_FLOOR_FIRST > 0.0 and not upper:
        _fb = np.asarray(model.space.floor_z, dtype=np.float64) + HM_WALL_CLEARANCE
        is_floor_cell = land <= (_sliding_max2d(_fb, wx, wy) + 1e-6)
    else:
        is_floor_cell = None

    cands: list[Candidate] = []
    for i, j in _select_cells(score, valid, per_bin, coarse_topk):
        center = np.array([cx[i, j], cy[i, j], land[i, j] + hz], dtype=np.float64)
        c = Candidate(
            item_idx=int(item.idx), container_idx=int(container_idx), ems_id=0,
            orientation=int(orn), pos_rel=center, osize=np.asarray(osize, dtype=np.float64),
            anchor=1 if upper else 0,
        )
        c.score = float(score[i, j])
        c.features["hm_upper"] = bool(upper)  # 層情報（support 計算・densify で使用）
        c.features["soft_nonflat"] = bool(soft_nonflat)  # desperate 段のみ許容
        c.features["floor"] = bool(is_floor_cell[i, j]) if is_floor_cell is not None else False
        cands.append(c)
    return cands


# --- 安定性モデル用の特徴量 ---------------------------------------------------------

# 実測（HF-022）: 支持率を 0.80→0.60 に一律で緩めると、たまに不安定な配置が通って沈降で崩れ、
# `is_placed_safe` 単独失敗でエピソードが4割地点で即終了する（−32pt np / −15 fill）。
# 一律閾値は「本当は安全な配置」も捨て「本当は危険な配置」も通す。そこで沈降結果（変位・傾き）を
# 候補の特徴量から予測し、危険なものだけ弾く方向へ進む。まずは教師データ収集用の特徴量抽出。
STAB_FEATURE_NAMES = (
    "support_ratio",     # 接触セル率（既存判定と同じ）
    "center_ratio",      # 中央1/4 の接触セル率
    "roughness",         # 窓内の高さレンジ max-min [m]（凸凹の上に載るほど不安定）
    "std",               # 窓内の高さ標準偏差 [m]
    "half_h",            # 荷物の半高 hz [m]（高いほど転びやすい）
    "tippiness",         # hz / 最小半寸（1超で縦長＝転倒しやすい）
    "mass",              # 質量 [kg]
    "volume",            # 体積 [m^3]
    "is_soft",           # ソフト荷物か
    "soft_below",        # 支持面がソフト荷物か（沈み込む）
    "land",              # 着地高 [m]（高いほど揺れの腕が長い）
    "footprint_cells",   # 接地セル数（小さいほど支持が乏しい）
)

# GH_DIAG=1 のとき、採用された候補の特徴量をここへ置く（scripts/collect_stability_data.py が読む）。
LAST_ACCEPTED: dict = {}

# HF-022 Phase 4: 先読みが実際に選択を変えた回数（診断。無言の no-op を検出するため）。
LOOKAHEAD_STATS: dict = {}


def support_features(model: ContainerHeightModel, cand: Candidate, item_meta: dict) -> dict:
    """採用候補の安定性特徴量。**採用された1手だけ**に対して呼ぶ（ホットパスには入れない）。"""
    osize = np.asarray(cand.osize, dtype=np.float64)
    amin, amax = cand.pos_rel - osize / 2.0, cand.pos_rel + osize / 2.0
    sx, sy = cells_of_aabb(model.space, amin, amax)
    layer = model.upper if cand.features.get("hm_upper") else model.lower
    softg = model.soft_up if cand.features.get("hm_upper") else model.soft_low
    sub = layer[sx, sy]
    if sub.size == 0:
        return {}
    surface = float(sub.max())
    supported = sub >= (surface - HM_SUPPORT_TOL)
    ax, ay = sub.shape
    cq = supported[ax // 4: max(ax - ax // 4, ax // 4 + 1), ay // 4: max(ay - ay // 4, ay // 4 + 1)]
    hz = float(osize[2]) / 2.0
    min_half = float(np.min(osize)) / 2.0
    return {
        "support_ratio": float(supported.mean()),
        "center_ratio": float(cq.mean()) if cq.size else float(supported.mean()),
        "roughness": surface - float(sub.min()),
        "std": float(sub.std()),
        "half_h": hz,
        "tippiness": hz / max(min_half, 1e-6),
        "mass": float(item_meta.get("mass", 0.0)),
        "volume": float(np.prod(osize)),
        "is_soft": float(bool(item_meta.get("is_soft"))),
        "soft_below": float(bool(np.any(softg[sx, sy]))),
        "land": surface,
        "footprint_cells": float(sub.size),
    }


# --- 実行可能性（inclusion / support / transport） ---------------------------------

def _support_ok(model: ContainerHeightModel, cand: Candidate, sr_min: float, sc_min: float) -> bool:
    """heightmap から接触セル率（＋中央1/4）で支持を判定。"""
    osize = np.asarray(cand.osize, dtype=np.float64)
    amin = cand.pos_rel - osize / 2.0
    amax = cand.pos_rel + osize / 2.0
    sx, sy = cells_of_aabb(model.space, amin, amax)
    if sx.stop <= sx.start or sy.stop <= sy.start:
        return False
    layer = model.upper if cand.features.get("hm_upper") else model.lower
    sub = layer[sx, sy]
    if sub.size == 0:
        return False
    surface = float(sub.max())
    supported = sub >= (surface - HM_SUPPORT_TOL)
    ratio = float(supported.mean())
    ax, ay = sub.shape
    cq = supported[ax // 4: max(ax - ax // 4, ax // 4 + 1), ay // 4: max(ay - ay // 4, ay // 4 + 1)]
    center_ratio = float(cq.mean()) if cq.size else ratio
    return ratio >= sr_min and center_ratio >= sc_min


def _feasible(state, model, cand, incl_m, pp_stage, sr_min, sc_min) -> bool:
    if not contains_oriented_box(model.space, cand.pos_rel, cand.osize, incl_m):
        return False
    if not _support_ok(model, cand, sr_min, sc_min):
        return False
    if not _DIAG_NO_LPATH and not check_l_path(state, cand, pp_stage):
        return False
    return True


def _densify(state, model, cand, incl_m, pp_stage, sr_min, sc_min, budget) -> Candidate:
    """採用候補を +y（奥）→ -x（左壁）へ 4mm 刻みで詰める（実行可能な限り）。z は維持。"""
    for axis, delta in ((1, +HM_DENSIFY_STEP), (0, -HM_DENSIFY_STEP)):
        for _ in range(HM_DENSIFY_MAX):
            if budget.over_soft():
                return cand
            trial_pos = cand.pos_rel.copy()
            trial_pos[axis] += delta
            trial = Candidate(
                item_idx=cand.item_idx, container_idx=cand.container_idx, ems_id=0,
                orientation=cand.orientation, pos_rel=trial_pos, osize=cand.osize,
                anchor=cand.anchor,
            )
            trial.features["hm_upper"] = cand.features.get("hm_upper", False)
            if _feasible(state, model, trial, incl_m, pp_stage, sr_min, sc_min):
                cand = trial
            else:
                break
    return cand


# --- 1手決定（カスケード） ----------------------------------------------------------

def _build_models_ranked(state, budget):
    """コンテナ heightmap モデル群と、スコア降順の候補列 (models, ranked) を構築する。

    decide_placement / decide_topk が共有する候補生成。候補は一度だけ生成し（スコアはカスケードの
    マージンに依存しない）、体積降順上位 HM_POOL_CAP の pool × コンテナ × 向き × 層を走査する。
    生成は soft 予算で荷物ごとに打ち切る。候補ゼロ・空 pool は (models, []) を返す。
    """
    n_cont = len(state.containers)
    if n_cont == 0 or not state.pool:
        return [], []
    models = [ContainerHeightModel(state.containers[c], state.placed.get(c, [])) for c in range(n_cont)]
    # HF-022 Phase 10: soft 予約は容器を問わないので、残り soft の体積を**全容器の合計床面積**で
    # 割った高さを全容器に適用する。優先品予約と同じく、置かれるたび縮む。
    reserve_soft = 0.0
    if HM_SOFT_RESERVE > 0.0 and SOFT_TOTAL["n"] > 0:
        n_done = sum(1 for lst in state.placed.values() for it in lst
                     if getattr(it, "is_soft", False))
        remain = max(0.0, SOFT_TOTAL["n"] - n_done)
        if remain > 0.0:
            area_all = max(sum(m.nx * m.ny * m.cell * m.cell for m in models), 1e-9)
            h_vol = SOFT_TOTAL["vol"] * remain / SOFT_TOTAL["n"] / area_all
            reserve_soft = HM_SOFT_RESERVE * max(h_vol, SOFT_TOTAL["hmax"])
    # HF-027: 既定は strict 幅のみ（従来どおり）。GH_HM_WIDE_BREADTH=1 で
    # HM_CASCADE 第3段の幅（14 / 250）で生成し、全段に広い候補集合を渡す。
    # HF-033: GH_HM_BREADTH_MUL は desperate 幅を上限に strict 幅を連続倍率で刻む
    # （WIDE_BREADTH の二値と異なり中間の幅を作れる、findings §4.5.1/§12.1）。
    if HM_WIDE_BREADTH:
        per_bin, coarse = HM_CASCADE[-1][4], HM_CASCADE[-1][5]
    elif HM_BREADTH_MUL != 1.0:
        per_bin = min(HM_CASCADE[-1][4],
                      max(HM_STRAT_PER_BIN, int(round(HM_STRAT_PER_BIN * HM_BREADTH_MUL))))
        coarse = min(HM_CASCADE[-1][5],
                     max(HM_COARSE_TOPK, int(round(HM_COARSE_TOPK * HM_BREADTH_MUL))))
    else:
        per_bin, coarse = HM_STRAT_PER_BIN, HM_COARSE_TOPK
    prio_exists = any(bool(c.is_prioritized) for c in state.containers)
    pool = sorted(state.pool, key=lambda it: -float(np.prod(np.asarray(it.size, dtype=np.float64))))
    pool = pool[:HM_POOL_CAP]
    raw: list[Candidate] = []
    for cidx in range(n_cont):
        model = models[cidx]
        has_shelf = bool(np.any(model.shelf_cover))
        cont_prio = bool(model.space.is_prioritized)
        # HF-022 Phase 9: 残り優先品のための予約高さ。天井から下へ reserve_h を空け、
        # 非優先品の上面がその帯に入る配置を禁止する。優先品が積まれるたび縮む。
        # 体積由来の高さ（残り優先品がぴったり詰まる高さ）と最厚優先品の厚みの大きい方を採る
        # ―― 帯が1個の厚みより薄いと優先品が入らず予約の意味がない。
        reserve_h = 0.0
        if HM_PRIO_RESERVE > 0.0 and cont_prio and PRIO_TOTAL["n"] > 0:
            # HF-027: 既定は従来どおりコンテナ内だけを数える（後方互換）。
            # GH_HM_PRIO_RESERVE_GLOBAL=1 で全コンテナ合算に直す（過剰予約バグの修正）。
            if HM_PRIO_RESERVE_GLOBAL:
                n_done = sum(1 for lst in state.placed.values() for it in lst
                             if getattr(it, "is_priority", False))
            else:
                n_done = sum(1 for it in state.placed.get(cidx, [])
                             if getattr(it, "is_priority", False))
            remain = max(0.0, PRIO_TOTAL["n"] - n_done)
            if remain > 0.0:
                area = max(model.nx * model.ny * model.cell * model.cell, 1e-9)
                h_vol = PRIO_TOTAL["vol"] * remain / PRIO_TOTAL["n"] / area
                reserve_h = HM_PRIO_RESERVE * max(h_vol, PRIO_TOTAL["hmax"])
        for item in pool:
            if raw and budget.over_soft():
                break
            size = np.asarray(item.size, dtype=np.float64)
            seen_shapes: set[tuple] = set()
            for orn in range(6):
                osize = np.asarray(oriented_size(size, orn), dtype=np.float64)
                key = (round(float(osize[0]), 6), round(float(osize[1]), 6), round(float(osize[2]), 6))
                if key in seen_shapes:
                    continue
                seen_shapes.add(key)
                raw.extend(_layer_candidates(model, item, cidx, orn, osize,
                                             upper=False, per_bin=per_bin, coarse_topk=coarse,
                                             cont_prio=cont_prio, prio_exists=prio_exists,
                                             reserve_h=reserve_h, reserve_soft=reserve_soft))
                if has_shelf:
                    raw.extend(_layer_candidates(model, item, cidx, orn, osize,
                                                 upper=True, per_bin=per_bin, coarse_topk=coarse,
                                                 cont_prio=cont_prio, prio_exists=prio_exists,
                                                 reserve_h=reserve_h, reserve_soft=reserve_soft))
        if raw and budget.over_soft():
            break
    ranked = sorted(raw, key=lambda c: -c.score) if raw else []
    return models, ranked


def decide_topk(state, budget, k):
    """スコア上位から feasible な配置を「異なる item ごとに1つ」最大 k 件返す（beam 探索用）。

    decide_placement と同じ候補生成・カスケードを使い、各 item の最良 feasible 配置を高スコア順に
    集める（densify 済み）。戻り: [(item_idx, container_idx, pos_rel, orientation), ...]（最大 k 件）。
    """
    models, ranked = _build_models_ranked(state, budget)
    if not ranked:
        return []
    out: list[tuple] = []
    seen_items: set[int] = set()
    n_stages = len(HM_CASCADE)
    for s_idx, (incl_m, safety_m, sr_min, sc_min, _pb, _ck) in enumerate(HM_CASCADE):
        is_desperate = s_idx == n_stages - 1
        pp_stage = PlacementParams(safety_margin=float(safety_m))
        for cand in ranked:
            if len(out) >= k or budget.over_soft():
                return out
            if int(cand.item_idx) in seen_items:
                continue
            if (not is_desperate) and cand.features.get("soft_nonflat"):
                continue
            model = models[cand.container_idx]
            if _feasible(state, model, cand, incl_m, pp_stage, sr_min, sc_min):
                cand = _densify(state, model, cand, incl_m, pp_stage, sr_min, sc_min, budget)
                seen_items.add(int(cand.item_idx))
                out.append((int(cand.item_idx), int(cand.container_idx),
                            np.asarray(cand.pos_rel, dtype=np.float64), int(cand.orientation)))
        if len(out) >= k:
            break
    return out


def _explain_none(state, models, ranked, reason: str) -> dict:
    """投了時の内訳を作る（診断専用・`_DIAG` のときだけ呼ばれる）。

    各カスケード段について、候補が inclusion / support / l_path のどれで最初に落ちたかを数える。
    `_feasible` は短絡するため、ここでは3判定を個別に回して初回失敗を帰属させる。
    """
    out = {"reason": reason, "n_candidates": len(ranked),
           "n_pool": len(state.pool), "n_containers": len(state.containers), "stages": []}
    for s_idx, (incl_m, safety_m, sr_min, sc_min, _pb, _ck) in enumerate(HM_CASCADE):
        pp_stage = PlacementParams(safety_margin=float(safety_m))
        c = {"stage": s_idx, "incl_fail": 0, "support_fail": 0, "l_path_fail": 0,
             "soft_nonflat_skip": 0, "pass": 0}
        is_desperate = s_idx == len(HM_CASCADE) - 1
        for cand in ranked:
            if (not is_desperate) and cand.features.get("soft_nonflat"):
                c["soft_nonflat_skip"] += 1
                continue
            model = models[cand.container_idx]
            if not contains_oriented_box(model.space, cand.pos_rel, cand.osize, incl_m):
                c["incl_fail"] += 1
            elif not _support_ok(model, cand, sr_min, sc_min):
                c["support_fail"] += 1
            elif not check_l_path(state, cand, pp_stage):
                c["l_path_fail"] += 1
            else:
                c["pass"] += 1
        out["stages"].append(c)
    return out


class _TentativeBox:
    """`check_l_path` は state.placed から aabb_min_rel/aabb_max_rel だけを読むので、
    先読みの仮置きはこの2属性だけ持つダミーで足りる（HF-022 Phase 4）。"""

    __slots__ = ("aabb_min_rel", "aabb_max_rel")

    def __init__(self, lo, hi):
        self.aabb_min_rel = lo
        self.aabb_max_rel = hi


def _survivors(state, cand, probe, pp_stage) -> int:
    """`cand` を仮置きしたとき、`probe`（別荷物の候補）のうち何個が l_path を通り続けるか。"""
    half = np.asarray(cand.osize, dtype=np.float64) / 2.0
    box = _TentativeBox(cand.pos_rel - half, cand.pos_rel + half)
    plist = state.placed.setdefault(int(cand.container_idx), [])
    plist.append(box)
    try:
        return sum(1 for p in probe if check_l_path(state, p, pp_stage))
    finally:
        plist.pop()


def decide_placement(state, budget):
    """heightmap エンジンで1手を選び (item_idx, container_idx, pos_rel, orientation) を返す。

    候補ゼロ／全段不合格／時間切れは None。`state.pool`（ItemSpec 一覧）× コンテナ × 向き ×
    層(下/棚上)を走査し、fill 寄りスコア順に inclusion+support+transport をカスケードで確認する。

    Args:
        state: 現在の `PackingState`（containers/placed/pool を使用）。
        budget: 共有 `StepBudget`（hard 締切で打ち切り）。

    Returns:
        (item_idx:int, container_idx:int, pos_rel:np.ndarray(3,), orientation:int) または None。
    """
    models, ranked = _build_models_ranked(state, budget)
    if not ranked:
        if _DIAG:
            _DIAG_LOG.append(_explain_none(state, models, ranked, "no_candidates"))
        return None

    n_stages = len(HM_CASCADE)
    for s_idx, (incl_m, safety_m, sr_min, sc_min, _pb, _ck) in enumerate(HM_CASCADE):
        is_desperate = s_idx == n_stages - 1  # 最終段のみ soft の非flat姿勢を許容
        pp_stage = PlacementParams(safety_margin=float(safety_m))
        # HF-022 Phase 4 層規律: 段ごとに「床候補のみ」→「全候補」の2巡。床被覆が閾値未満の
        # コンテナでは積み上げ候補を後回しにする（診断で、床が32%しか埋まらないうちに積み上げが
        # 始まり扉側に壁ができて後続の8cm持ち上げ搬入を塞いでいた）。
        passes = (True, False) if HM_FLOOR_FIRST > 0.0 else (False,)
        for floor_only in passes:
          for cand in ranked:
            if budget.over_soft():
                if _DIAG:
                    _DIAG_LOG.append(_explain_none(state, models, ranked, "over_soft"))
                return None  # soft 予算で全体を打ち切り（policy 時間ガード。max policy ≲5s に収める）
            if (not is_desperate) and cand.features.get("soft_nonflat"):
                continue
            model = models[cand.container_idx]
            if floor_only and not cand.features.get("floor"):
                continue
            if floor_only and model.floor_covered >= HM_FLOOR_FIRST:
                continue     # このコンテナは既に床が十分埋まっている＝床限定にしない
            if _feasible(state, model, cand, incl_m, pp_stage, sr_min, sc_min):
                if HM_LOOKAHEAD_K > 1:
                    # HF-022 Phase 4: この段で実行可能な上位 K 個を集め、到達可能性の生存数で選ぶ。
                    feas = [cand]
                    for c2 in ranked:
                        if len(feas) >= HM_LOOKAHEAD_K or budget.over_soft():
                            break
                        if c2 is cand or ((not is_desperate) and c2.features.get("soft_nonflat")):
                            continue
                        if _feasible(state, models[c2.container_idx], c2, incl_m, pp_stage,
                                     sr_min, sc_min):
                            feas.append(c2)
                    if len(feas) > 1:
                        # probe は「その候補の荷物以外」の候補。K 個ぶん一括除外すると
                        # HM_POOL_CAP=8 のプールでは probe が空になる（最初の実装のバグ）。
                        best, best_n = cand, -1
                        for c2 in feas:
                            probe = [p for p in ranked[:HM_LOOKAHEAD_PROBE]
                                     if int(p.item_idx) != int(c2.item_idx)]
                            if not probe:
                                continue
                            n = _survivors(state, c2, probe, pp_stage)
                            # 生存率で比較（probe 数が候補間で違うため）
                            r = n / len(probe)
                            if r > best_n:          # 同率なら先（=高スコア）を保つ
                                best, best_n = c2, r
                        if best is not cand:
                            LOOKAHEAD_STATS["changed"] = LOOKAHEAD_STATS.get("changed", 0) + 1
                            cand = best
                            model = models[cand.container_idx]
                        LOOKAHEAD_STATS["calls"] = LOOKAHEAD_STATS.get("calls", 0) + 1
                cand = _densify(state, model, cand, incl_m, pp_stage, sr_min, sc_min, budget)
                if _DIAG:
                    # 採用手だけ特徴量を記録（安定性モデルの教師データ）。ホットパス外。
                    it = next((p for p in state.pool if p.idx == cand.item_idx), None)
                    LAST_ACCEPTED.clear()
                    LAST_ACCEPTED.update(support_features(
                        model, cand,
                        {"mass": getattr(it, "weight", 0.0), "is_soft": getattr(it, "is_soft", False)}))
                    LAST_ACCEPTED["stage"] = s_idx
                return (int(cand.item_idx), int(cand.container_idx),
                        np.asarray(cand.pos_rel, dtype=np.float64), int(cand.orientation))
    if _DIAG:
        _DIAG_LOG.append(_explain_none(state, models, ranked, "all_stages_failed"))
    return None
