"""Sequence Triple 事前最適化（HF-045、既定OFF、GH_SEQ_TRIPLE=1で有効）。

設計: `docs/design_sequence_triple_2026-08-10.md`（案C）。3順列(Gamma1,Gamma2,Gamma3) +
向き + コンテナ割当から、全荷物の配置を同時に決めるcombinatorial decodeを実装する。
x/y は2D Sequence Pair式のLCS最長路で確定し、z は物理的な支持面dropで確定する
（教科書的な3軸対称decodeは支持面と無関係な「浮いた配置」を生みうるため採用しない、
design文書 §3.2/§3.4参照）。

**関係分類の出典に関する注記**: 文献（Yamazaki et al. 2000ほか3D floorplanning系）は
web検索で存在は確認できたが、関係分類の正確な数式は参照できなかった（design文書
§8.1で「文献どおりに転記する」ことを推奨していたが実施できなかった）。そのため
以下の `classify_pairs` は自前で正しさを証明した構成（本docstring参照）を使う。
非重なり・非浮遊はproperty testで経験的に確認する（design文書 §7-1の方針どおり）。

**実装時の簡略化（設計文書からの差分）**:
1. Gamma1が関係分類の基準順序であると同時に、x/y両方の制約グラフに対する単一の
   トポロジカル順序にもなる（両関係とも「Gamma1で先」の荷物から辺が伸びるため）。
   これは物理drop順序としても有効なので、設計文書が想定していた独立の「Gammaz」は
   不要と判明し、Gamma1が兼務する（3遺伝子構成 Gamma1/Gamma2/Gamma3のまま、役割が
   単純化された）。設計文書が想定していた明示的な支持DAGの構築（z関係ペアの列挙・
   footprint重なりでの枝刈り）も同じ理由で不要と判明した——z関係は「x/y辺を作らない」
   というだけの意味しか持たず、実際の支持関係は`decode_container`が`order1`順に
   1品ずつ物理drop（現在のheightmapへ実際に着地）する過程で結果的に決まる。これは
   分類ラベルに依存しないため、万一分類にバグがあっても3D非重なりは物理drop側の
   「landは現在の実占有grid由来」という不変条件だけで独立に保たれる（二重の安全網）。
2. v1は棚上面（upper層）を対象外とし、床/床積み（lower層）のみを decode する。
   shelf層は既存貪欲(`decide_placement`)が引き続き担当する仕組みは無い——
   Sequence Triple経由の配置はlower層に限定されるという明示的なスコープ制限。
   実装・検証の複雑さ（upper/lower二重層の同時最適化）を初回実装から除外するための
   意図的な判断（design文書 §7の「まずdecode本体の正しさ」を優先）。
"""
from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import numpy as np

from .geometry import oriented_size
from .types import Vec3

if TYPE_CHECKING:
    from .types import Candidate

# --- §3.4 手順1: 3順列 -> 全ペア関係分類（純関数、コンテナ非依存） -------------------


def classify_pairs(order1: list[int], order2: list[int], order3: list[int]) -> dict[tuple[int, int], str]:
    """3順列から全ペアの関係を返す。

    `order1`/`order2`/`order3` は同じ荷物id集合の3順列（並べ替えた `list[int]`）。

    戻り値: `dict[(i, j)] -> 'x' | 'y' | 'z'`。キー `(i, j)` は常に
    `pos1[i] < pos1[j]`（`order1` で i が先）となるよう向きを揃えてある。

    構成（自前で正しさを検証済み、モジュールdocstring参照）:
        1. `order1` を全ペア共通の基準順序に使う（`pos1[i] < pos1[j]` となるペアだけを扱う）。
        2. `pos2[i] < pos2[j]`（Gamma1・Gamma2が一致）なら 'x'
           （＝古典2D Sequence Pairの「両順列で同じ順」＝水平関係、という規則そのもの）。
        3. 不一致なら、`pos3[i] < pos3[j]`（Gamma1・Gamma3が一致）で 'y'、
           不一致なら 'z'（Gamma1に対しGamma2・Gamma3の両方が不一致）。
           Gamma1を固定リファレンスにして同じ一致/不一致規則をGamma3にも適用する
           入れ子構成であり、8通りの(b1,b2,b3)パターンのうち「pos1[i]<pos1[j]」の
           4通り（b2,b3の2x2）を過不足なくx/y/zへ割り付ける。

    この構成により、全ペアの関係付けは一意（missing/重複なし）であり、かつ
    全ての関係が「`order1` で先の荷物が前（x-before/y-before/z-below）」という
    共通の向きを持つ。したがって `order1` 自体が x関係グラフ・y関係グラフ・z関係の
    いずれに対しても有効なトポロジカル順序になる（`decode_xy` はこれを利用し
    `order1` の1パスで最長路を計算する）。

    Args:
        order1: 基準順列（荷物idの並び）。
        order2: 第2順列。
        order3: 第3順列。

    Returns:
        `(i, j)`（`pos1[i] < pos1[j]`）をキーに `'x'|'y'|'z'` を値に持つ辞書。
    """
    pos2 = {v: i for i, v in enumerate(order2)}
    pos3 = {v: i for i, v in enumerate(order3)}
    rel: dict[tuple[int, int], str] = {}
    n = len(order1)
    for a in range(n):
        i = order1[a]
        for b in range(a + 1, n):
            j = order1[b]
            if pos2[i] < pos2[j]:
                rel[(i, j)] = "x"
            elif pos3[i] < pos3[j]:
                rel[(i, j)] = "y"
            else:
                rel[(i, j)] = "z"
    return rel


# --- §3.4 手順2: x,y footprint decode（最長路） --------------------------------------


def decode_xy(
    order1: list[int], rel: dict[tuple[int, int], str], footprint: dict[int, tuple[float, float]],
) -> tuple[dict[int, float], dict[int, float]]:
    """x関係・y関係のペアから、`order1` の1パスで (x0, y0)（footprint最小角）を決める。

    z関係に分類されたペアは辺を作らない（footprintの重なりを許容する＝段積みの表現）。
    計算量は `O(n^2)`（`n<=80` 程度では愚直実装で十分、design文書 §3.4）。

    Args:
        order1: 荷物idの基準順列（`classify_pairs` と同じもの）。
        rel: `classify_pairs` の戻り値。
        footprint: `id -> (wx, wy)`（回転後のfootprint寸法 [m]）。

    Returns:
        `(x0, y0)`: それぞれ `id -> float`（footprint最小角の座標 [m]）。
    """
    x0: dict[int, float] = {}
    y0: dict[int, float] = {}
    n = len(order1)
    for a in range(n):
        i = order1[a]
        best_x = 0.0
        best_y = 0.0
        for b in range(a):
            j = order1[b]
            r = rel[(j, i)]
            if r == "x":
                wx_j, _wy_j = footprint[j]
                cand = x0[j] + wx_j
                if cand > best_x:
                    best_x = cand
            elif r == "y":
                _wx_j, wy_j = footprint[j]
                cand = y0[j] + wy_j
                if cand > best_y:
                    best_y = cand
            # r == "z": 辺を作らない（footprintの重なりを許容）。
        x0[i] = best_x
        y0[i] = best_y
    return x0, y0


def item_footprint(size: Vec3, orientation: int) -> tuple[float, float, float]:
    """回転後の (wx, wy, hz) を返す（`geometry.oriented_size` の薄いラッパ）。"""
    osz = oriented_size(size, orientation)
    return float(osz[0]), float(osz[1]), float(osz[2])


# --- §3.4 手順4: 実コンテナへの物理drop（lower層のみ、v1のスコープ制限） ----------------

_IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
# x/y方向の隙間パディング [m]（decode_xyへ渡すfootprintに加算する。壁・隣接荷物からの
# 実クリアランスを確保する目的、`decode_container`のコメント参照）。strict段の
# inclusion_margin=12mmより大きく、複数の候補が排他的に競合しない程度の値として
# 30mmを選んだ（経験的、n=99×3族検証で確定率が低ければ調整対象）。
_GAP = 0.03


def _try_place_lower(space, state, container_idx: int, id_: int, center_xy: tuple[float, float],
                      osize: tuple[float, float, float], orn: int, margins,
                      is_soft: bool = False) -> "Candidate | None":
    """(x,y中心)・回転後寸法を固定し、lower層（床/床積み）へ着地させて実行可能性を判定する。

    `land`（現在の天面高の最大値、windowはSPで確定済みで探索しない）だけを自前で計算し、
    inclusion/support_ratio/搬入経路（`check_l_path`）は既存 `heightmap._feasible` に委譲する
    （design文書 §8.1の方針どおり、独自の幾何判定を増やさず既存の検証済みロジックを再利用する）。
    `margins` は `constants.HM_CASCADE` 型のタプル列（strict→relaxed→desperateの順に試す）。

    `is_soft`: 置こうとしている荷物自身がsoftか。design文書§11.8で判明したギャップの修正
    ——本番既定エンジン（`heightmap._layer_candidates`）は`HM_SOFT_VETO`（既定1、v50で
    ベイク済み）でsoft最上面への非soft着地をcandidate生成段階で無効化するが、
    `_feasible`はこれを検査しない（inclusion/support_ratio/l_pathのみ）。ここで同型の
    veto（`model.soft_low`基準、HM_SOFT_VETO==2ならdesperate段のみ解禁）を追加し、
    本番エンジンに存在しない「soft上へのhard着地」がseq_triple候補にだけ紛れ込む
    非対称を無くす。

    Returns:
        feasibleな最初のtierで構築した `Candidate`（`pos_rel`は着地後の中心座標）。
        全tierで不可行なら `None`。
    """
    from .constants import HM_SOFT_VETO, PlacementParams
    from .container_space import cells_of_aabb
    from .heightmap import ContainerHeightModel, _feasible
    from .types import Candidate

    model = ContainerHeightModel(space, state.placed[container_idx])
    wx, wy, hz_full = osize
    amin_xy = np.array([center_xy[0] - wx / 2.0, center_xy[1] - wy / 2.0, 0.0], dtype=np.float64)
    amax_xy = np.array([center_xy[0] + wx / 2.0, center_xy[1] + wy / 2.0, 0.0], dtype=np.float64)
    sx, sy = cells_of_aabb(space, amin_xy, amax_xy)
    if sx.stop <= sx.start or sy.stop <= sy.start:
        return None
    sub = model.lower[sx, sy]
    if sub.size == 0:
        return None
    soft_hit = (not is_soft) and bool(np.any(model.soft_low[sx, sy]))
    land = float(sub.max())
    ceiling = float(model.lower_ceiling[sx, sy].min())
    hz = hz_full / 2.0
    if (land + hz_full) > (ceiling + 1e-6):
        return None
    center = np.array([center_xy[0], center_xy[1], land + hz], dtype=np.float64)
    osize_arr = np.array([wx, wy, hz_full], dtype=np.float64)
    n_tiers = len(margins)
    for tier_idx, (incl_m, safety_m, sr_min, sc_min, _pb, _ck) in enumerate(margins):
        if soft_hit and HM_SOFT_VETO != 0:
            is_desperate = tier_idx == n_tiers - 1
            if HM_SOFT_VETO != 2 or not is_desperate:
                continue  # soft veto: HM_SOFT_VETO==1は全tier、==2はdesperate段のみ解禁
        pp = PlacementParams(safety_margin=float(safety_m))
        cand = Candidate(item_idx=int(id_), container_idx=int(container_idx), ems_id=0,
                          orientation=int(orn), pos_rel=center.copy(), osize=osize_arr.copy())
        cand.features["hm_upper"] = False
        if _feasible(state, model, cand, incl_m, pp, sr_min, sc_min):
            return cand
    return None


def decode_container(space, state, container_idx: int, item_specs: dict[int, dict],
                      order1: list[int], order2: list[int], order3: list[int],
                      orn_map: dict[int, int],
                      slot_footprint: dict[int, tuple[float, float]] | None = None,
                      ) -> tuple[list[int], dict[int, tuple], set[int]]:
    """1コンテナぶんのdecodeを実行し、`state.placed[container_idx]` へ確定分を追記する。

    `item_specs`: このコンテナへ割り当てられた**未配置**荷物のみ（`size`/`mass`/`is_soft`/
    `is_priority`を持つ辞書）。`state.placed[container_idx]`には既に初期積載済み荷物
    （design文書§2.6）が入っている前提（本関数はそれを変更せず、末尾に追記するのみ）。
    `order1/2/3`: `item_specs`のキー集合の3順列。`orn_map`: `id -> 0..5`。

    `slot_footprint`（既定None＝各荷物は自分自身の実寸footprintで位置決めする）:
    段積みシード（`shelf_seed_genes`のスロット/スタック構成、design文書§10.4(1)）が
    使う整合用オーバーライド。同じ「列」に積まれる複数層の荷物は寸法が異なりうるが、
    それぞれの実寸footprintで独立に`decode_xy`の間隔計算をすると、層ごとに列境界が
    微妙にズレて支持率判定が落ちやすい（実装時に11/41止まりの原因として発見）。
    `slot_footprint[id]`を指定すると、その荷物の**位置決め**（`decode_xy`への
    footprint・実座標中心の算出）だけを列の基準（founder、通常は最下層の荷物）の
    footprintに合わせ、実際の当たり判定・着地面計算（`_try_place_lower`のosize）は
    引き続きその荷物自身の実寸を使う——「大きめの仮想スロットの中央に実寸の荷物を
    センタリングする」構成。同じ列の全層が同じ(x,y)中心を共有するため、層間の
    支持整合（真上に正確に載ること）が構造的に保証される。

    **座標決定順序と物理drop順序を分離する（実装時に発見・修正した設計文書からの
    差分）**: `x0,y0`（footprint位置）は`order1`をトポロジカル順序とした最長路で
    決める（§3.4手順2どおり）。しかし**物理drop（実際に`_try_place_lower`へ渡す順序）
    は`order1`をそのまま使わず、`y0`降順（コンテナ奥側から手前側へ）で並べ替える**。
    理由: `order1`は「早い＝小さいy」というSP decodeの構造的性質を持つ（y0は
    predecessorからの累積なので、`order1`で早い荷物ほどy0が小さくなりやすい）。
    これをそのまま物理drop順に使うと「扉に近い(小y)荷物を先に置き、奥(大y)の荷物を
    後で搬入する」構成になり、後から来る荷物の搬入経路（`check_l_path`のYスイープ
    レグ）が先に置いた手前の荷物にブロックされる——§19（findings §19.4）が示す
    「奥から手前へ」という搬入安全の鉄則と正反対になってしまう（実装時、単純な
    棚パッキング初期解で確定率が2/12まで落ちる不具合として発覚し、原因を特定した）。
    `y0`降順（大きいyつまり奥から先に処理）へ並べ替えることで、物理drop順序を
    §19の鉄則に合わせる。3D非重なり・非浮遊の保証（`land`は常に「現在の実占有grid」
    から計算される）は物理drop順序に依存しないため、この並べ替えで安全性は損なわれない
    （`test_seq_triple_container.py`のproperty testは本並べ替え後の実装で再確認済み）。
    空間が無い等で`_try_place_lower`が全tier不可行を返した荷物は**スキップして
    次へ進む**（design文書§3.4手順5、候補自体は棄却しない）。

    Returns:
        `(committed_ids, plan_entries, failed_ids)`。
        `committed_ids`: 実際に置けたidを確定順（=物理drop順）で並べたもの。
        `plan_entries`: `id -> (container_idx, pos_rel:list[float,3], orientation:int)`
            （`rollout_plan.py`の`plan`と同一契約）。
        `failed_ids`: この呼び出しでは置けなかったid集合。
    """
    from .constants import HM_CASCADE
    from .state import rel_to_world
    from .types import PlacedItem

    ids = list(order1)
    # decode_xy は自然に「隙間ゼロの密着パッキング」を生成する（LCS最長路の定義そのもの）。
    # しかし公式validatorは壁・隣接荷物からの正のクリアランスを要求する
    # （inclusion_margin=5mm、QA#51）。密着させたまま実座標へ変換すると、壁沿い・
    # 隣接荷物のほぼ全候補がinclusion判定で落ちる（実装時に発見：確定率が2〜5/12〜30
    # まで落ちる不具合として顕在化、原因を`_feasible`の内訳を1件ずつ調べて特定した）。
    # 対策: decode_xyには実寸+`_GAP`を渡して「隙間ぶん大きい仮想footprint」で
    # 最長路を計算し、実際の荷物はその仮想footprint内の中央（前後`_GAP/2`ずつの
    # 余白）に置く。壁からの余白も自動的に`_GAP/2`確保される（最初の荷物のx0=0が
    # 仮想footprintの先頭になるため）。
    footprint: dict[int, tuple[float, float]] = {}
    pos_footprint: dict[int, tuple[float, float]] = {}  # 位置決め専用（slot_footprintがあればそれ）
    osize_map: dict[int, tuple[float, float, float]] = {}
    for id_ in ids:
        wx, wy, hz = item_footprint(item_specs[id_]["size"], orn_map[id_])
        pwx, pwy = (slot_footprint.get(id_, (wx, wy)) if slot_footprint else (wx, wy))
        footprint[id_] = (pwx + _GAP, pwy + _GAP)
        pos_footprint[id_] = (pwx, pwy)
        osize_map[id_] = (wx, wy, hz)

    rel = classify_pairs(order1, order2, order3)
    x0, y0 = decode_xy(order1, rel, footprint)
    # decode_xy は footprint 最小角を原点(0,0)基準で返す（コンテナ座標系を知らない純関数、
    # §3.4手順2）。コンテナの実座標系（`inner_min_rel`、非対称な場合が大半）へオフセットする
    # ——これを怠ると非対称コンテナで負側の面積が一切使われない（実装時に発見・修正）。
    ox = float(space.inner_min_rel[0]) + _GAP / 2.0
    oy = float(space.inner_min_rel[1]) + _GAP / 2.0

    # 物理drop順序 = y0降順（奥から手前へ、§19の鉄則）。order1内の相対順位を
    # タイブレークに使う（z関係ペアで「支持側(=order1で先)が先に置かれる」ことを
    # y0が同点の場合に維持する）。
    order1_rank = {v: i for i, v in enumerate(order1)}
    drop_order = sorted(ids, key=lambda i: (-y0[i], order1_rank[i]))

    committed: list[int] = []
    plan_entries: dict[int, tuple] = {}
    failed: set[int] = set()
    for id_ in drop_order:
        wx, wy, hz_full = osize_map[id_]
        pwx, pwy = pos_footprint[id_]
        # 中心座標はスロット（位置決め用footprint、既定は自分自身の実寸）基準で計算する。
        # slot_footprintで大きめのスロットを渡された荷物は、実寸(wx,wy)より広いスロットの
        # 中央に自動的にセンタリングされる（列の他の層と同じ中心を共有するため）。
        cx = ox + x0[id_] + pwx / 2.0
        cy = oy + y0[id_] + pwy / 2.0
        cand = _try_place_lower(
            space, state, container_idx, id_, (cx, cy), (wx, wy, hz_full), orn_map[id_], HM_CASCADE,
            is_soft=bool(item_specs[id_].get("is_soft")),
        )
        if cand is None:
            failed.add(id_)
            continue
        spec = item_specs[id_]
        half = cand.osize / 2.0
        pi = PlacedItem(
            pos_world=rel_to_world(cand.pos_rel, space), orn_quat=_IDENTITY_QUAT,
            size=np.asarray(spec["size"], dtype=np.float64), weight=float(spec.get("mass", 0.0)),
            is_soft=bool(spec.get("is_soft")), is_priority=bool(spec.get("is_priority")),
            aabb_min_rel=cand.pos_rel - half, aabb_max_rel=cand.pos_rel + half,
        )
        state.placed[container_idx].append(pi)
        committed.append(id_)
        plan_entries[id_] = (container_idx, cand.pos_rel.tolist(), int(cand.orientation))
    return committed, plan_entries, failed


# --- §4.2: 搬入経路制約（階段）罰則 ---------------------------------------------------


def stair_violation(space, placed_items: list) -> float:
    """decode後の集約heightmapへ`_stair_deep_min`（HF-035の階段判定式）を一括適用し、
    §19の不変条件（各xレーンで天面が扉へ向かって単調非増加）からの逸脱量を返す。

    design文書§4.2のとおり、既存 `heightmap._stair_deep_min` をそのまま再利用する
    （1候補だけでなく、コンテナ内の全placed_itemsそれぞれについて、自分の窓を除いた
    「奥の使用可能セルの最小天面高」を求め、`top_z - deep_min - HM_STAIR_SLACK` の
    正の超過分を合計する）。0であれば全荷物が階段制約を（既定スラック内で）満たす。
    lower層（床/床積み）の荷物のみを対象とする（v1のスコープ、モジュールdocstring参照）。
    """
    from .constants import HM_STAIR_SLACK
    from .container_space import cells_of_aabb
    from .heightmap import ContainerHeightModel, _stair_deep_min

    if not placed_items:
        return 0.0
    model = ContainerHeightModel(space, placed_items)
    cell = model.cell
    cache: dict[tuple, np.ndarray] = {}
    violation = 0.0
    for item in placed_items:
        amin = np.asarray(item.aabb_min_rel, dtype=np.float64)
        amax = np.asarray(item.aabb_max_rel, dtype=np.float64)
        osize = amax - amin
        wx = max(1, int(np.ceil(osize[0] / cell - 1e-6)))
        wy = max(1, int(np.ceil(osize[1] / cell - 1e-6)))
        h = float(osize[2])
        key = (wx, wy, round(h, 6))
        deep_min_grid = cache.get(key)
        if deep_min_grid is None:
            deep_min_grid = _stair_deep_min(model.lower, model.lower_ceiling, h, wx, wy)
            cache[key] = deep_min_grid
        sx, sy = cells_of_aabb(space, amin, amax)
        if sx.stop <= sx.start or sy.stop <= sy.start:
            continue
        i, j = sx.start, sy.start
        if i >= deep_min_grid.shape[0] or j >= deep_min_grid.shape[1]:
            continue
        deep_min = float(deep_min_grid[i, j])
        violation += max(0.0, float(amax[2]) - deep_min - HM_STAIR_SLACK)
    return violation


# --- §5.1: 属性クラスタリング項（W_CLUSTER、QA#62、既定OFF） ------------------------


def _item_category(is_soft: bool, is_priority: bool) -> str:
    """QA#62の4分類（優先/ソフト/優先かつソフト/どちらでもない）。"""
    if is_soft and is_priority:
        return "soft_prio"
    if is_soft:
        return "soft"
    if is_priority:
        return "prio"
    return "none"


def cluster_score(placed_items: list) -> float:
    """decode確定後の実配置から、直接支持関係（XY重なり・天面=底面接触）にある
    ペアのうち同一属性（QA#62の4分類）どうしの組数を数える（design文書§5.1.3）。

    v36（`HM_W_SOFT_CAP`）の失敗機構（高さでcogを直接押し上げる）を避けるため、
    高さを一切参照しない——`bench_run.py::_quality_proxies::_covered_by_other_kind`と
    同型の幾何近似（XY重なり + 天面/底面が2cm以内で接触）で「直接支持」を判定する。
    """
    n = len(placed_items)
    score = 0.0
    for a in range(n):
        ta = placed_items[a]
        cat_a = _item_category(ta.is_soft, ta.is_priority)
        for b in range(n):
            if a == b:
                continue
            tb = placed_items[b]
            # tb が ta の直上に載っているか（ta=支持側、tb=被支持側）。
            if tb.aabb_min_rel[0] >= ta.aabb_max_rel[0] or tb.aabb_max_rel[0] <= ta.aabb_min_rel[0]:
                continue
            if tb.aabb_min_rel[1] >= ta.aabb_max_rel[1] or tb.aabb_max_rel[1] <= ta.aabb_min_rel[1]:
                continue
            if abs(float(tb.aabb_min_rel[2]) - float(ta.aabb_max_rel[2])) > 0.02:
                continue
            cat_b = _item_category(tb.is_soft, tb.is_priority)
            if cat_a == cat_b and cat_a != "none":
                score += 1.0
    return score


# --- §5: fill/cog（`order.py::_fill_and_cog` と同型の目的関数用近似） ----------------


def fill_and_cog(spaces, all_placed: dict[int, list]) -> tuple[float, float]:
    """充填率（AABB体積比の近似）と質量加重cog（相対Z）を返す（探索の目的関数専用、
    最終スコアの厳密計算ではない。`order.py::_fill_and_cog` と同じ近似方式）。"""
    total_item_vol = 0.0
    num = 0.0
    den = 0.0
    for items in all_placed.values():
        for it in items:
            total_item_vol += float(np.prod(np.asarray(it.size, dtype=np.float64)))
            cz = 0.5 * (float(it.aabb_min_rel[2]) + float(it.aabb_max_rel[2]))
            num += float(it.weight) * cz
            den += float(it.weight)
    total_cont_vol = 0.0
    for space in spaces:
        span = np.asarray(space.inner_max_rel, dtype=np.float64) - np.asarray(space.inner_min_rel, dtype=np.float64)
        total_cont_vol += float(np.prod(span))
    fill = total_item_vol / total_cont_vol if total_cont_vol > 0.0 else 0.0
    cog = num / den if den > 0.0 else 0.0
    return fill, cog


# --- 複数コンテナへのオーケストレーション ---------------------------------------------


def decode_all(
    spaces: list, base_placed: dict[int, list], item_specs: dict[int, dict],
    order1: list[int], order2: list[int], order3: list[int],
    orn_map: dict[int, int], container_map: dict[int, int],
    slot_footprint: dict[int, tuple[float, float]] | None = None,
):
    """全コンテナぶんのdecodeを実行する（design文書§3.4「decodeはコンテナごとに独立」）。

    `order1/2/3` は `item_specs` の全キーの3順列（グローバル）。コンテナごとに
    その順列を対象idで絞り込んだ部分列（相対順序は保持）を作り、`decode_container`を
    独立に呼ぶ（コンテナ間に物理干渉はないため独立に解ける）。

    Args:
        spaces: `ContainerSpace`一覧（indexがcontainer_idxと一致）。
        base_placed: `container_idx -> list[PlacedItem]`（初期積載品、design文書§2.6）。
            呼び出し側のリストは変更しない（内部でコピーする）。
        item_specs: 未配置荷物のみ（`id -> {"size","mass","is_soft","is_priority"}`）。
        order1/2/3: `item_specs`のキー集合の3順列。
        orn_map: `id -> 0..5`。
        container_map: `id -> container_idx`。
        slot_footprint: `decode_container`へそのまま渡す位置決めオーバーライド
            （`shelf_seed_genes`の列整合用、既定None）。

    Returns:
        `(committed_all, plan_all, failed_all, state)`。
        `committed_all`: `container_idx -> list[id]`（確定順）。
        `plan_all`: `id -> (container_idx, pos_rel, orientation)`。
        `failed_all`: 置けなかったid集合。
        `state`: 最終的な`PackingState`（`state.placed`が全コンテナの確定配置）。
    """
    from .state import PackingState

    n_cont = len(spaces)
    placed_copy = {c: list(base_placed.get(c, [])) for c in range(n_cont)}
    state = PackingState(
        containers=list(spaces), placed=placed_copy, pool=[],
        ems={c: [] for c in range(n_cont)}, ems_truncation={c: 0.0 for c in range(n_cont)}, meta={},
    )

    committed_all: dict[int, list[int]] = {c: [] for c in range(n_cont)}
    plan_all: dict[int, tuple] = {}
    failed_all: set[int] = set()

    for cidx in range(n_cont):
        ids_here = [i for i in order1 if container_map.get(i) == cidx]
        if not ids_here:
            continue
        ids_set = set(ids_here)
        o2 = [i for i in order2 if i in ids_set]
        o3 = [i for i in order3 if i in ids_set]
        specs_here = {i: item_specs[i] for i in ids_here}
        orn_here = {i: orn_map[i] for i in ids_here}
        committed, plan_entries, failed = decode_container(
            spaces[cidx], state, cidx, specs_here, ids_here, o2, o3, orn_here,
            slot_footprint=slot_footprint,
        )
        committed_all[cidx] = committed
        plan_all.update(plan_entries)
        failed_all |= failed

    return committed_all, plan_all, failed_all, state


def to_order_and_plan(
    committed_all: dict[int, list[int]], plan_all: dict[int, tuple], all_ids: list[int],
) -> tuple[list[int], dict[int, tuple], dict[int, int]]:
    """`decode_all`の出力を`rollout_plan.py::_order_and_plan`と同一契約の
    `(order, plan, rank)`へ変換する。

    **実装の簡略化（design文書§4.3からの差分）**: design文書は「奥→手前・支持DAG順の
    トポロジカルソート」による到着順序の再構成を想定していたが、`decode_container`が
    既に`order1`（=支持関係を保証する物理drop順）の順で確定させているため、
    その確定順（コンテナはindex昇順で連結、コンテナ間に依存関係は無いため任意の連結順で
    よい）をそのまま採用する。§19準拠の最終担保はどのみち`seq_triple_optimize`の
    実物理検証（floorとの比較、design文書§4.4）であり、ここでの順序付けは
    「検証を通りやすくする」ためのヒューリスティックに過ぎないため、追加の再ソートを
    省いても安全性は損なわれない。
    """
    placed_order: list[int] = []
    for cidx in sorted(committed_all):
        placed_order.extend(committed_all[cidx])
    plan_d = dict(plan_all)
    rank = {fid: i for i, fid in enumerate(placed_order)}
    rest = [i for i in all_ids if i not in plan_d]
    return placed_order + rest, plan_d, rank


# --- design文書§3.5: 初期解（棚/行パッキングによる構成的シード） ---------------------


def _shelf_rows(
    ids: list[int], footprint: dict[int, tuple[float, float]], width: float, depth: float,
    *, headroom_probe=None, item_height: dict[int, float] | None = None,
) -> tuple[list[list[int]], list[int]]:
    """`ids`順に幅`width`・奥行き`depth`の1層へnext-fitで詰める（単純な棚パッキング）。

    行の累積奥行きが`depth`（コンテナの実y-extent）を超える手前で打ち切り、
    収まらなかった残りは`overflow`として返す（呼び出し側の`_shelf_layers`が
    次の層の入力として再利用する）。

    容量判定には`footprint`の値へ`_GAP`を加えたものを使う（`decode_container`が
    実際に`decode_xy`へ渡すfootprintは`_GAP`ぶん大きい、§3.4手順4のコメント参照）。
    ここで生の`footprint`のまま判定すると、行数・列数を`_GAP`の余地を考慮せず
    決めてしまい、実際に`decode_xy`で座標を確定した時点で数cm単位で壁・行間から
    はみ出す（実装時に発見: real_000相当のタスクでfounderの荷物が奥壁を
    2.5cm超過してinclusion判定に落ちる不具合として顕在化した）。

    `headroom_probe`（既定None）: `(cx, cy, wx, wy) -> 頭上余裕[m]`を返す関数
    （呼び出し側が空コンテナの`ContainerHeightModel`から構築する）。指定時、
    各荷物の暫定中心位置での頭上余裕が`item_height[id]`未満ならその位置を
    拒否しoverflowへ回す——cut/棚構造で床が持ち上がったり天井が下がったりする
    領域（real_000相当のタスクで実在）にfounderを置くと、着地後に頭上超過で
    inclusion/support判定が落ち、その列に積んだ全層が連鎖的に失敗する問題への
    対処（design文書§10.4実装ノート、実装時に発見）。座標は`_GAP`パディング後の
    暫定位置（呼び出し側のoffset+累積カーソル）で概算する——`decode_xy`の
    最長路が確定させる実座標とは若干ズレうるが、cut/棚の粗い回避には十分。
    """
    rows: list[list[int]] = []
    cur: list[int] = []
    cursor = 0.0
    row_depth = 0.0
    used_depth = 0.0
    overflow: list[int] = []
    for id_ in ids:
        wx, wy = footprint[id_]
        wx_g, wy_g = wx + _GAP, wy + _GAP
        would_wrap = cur and cursor + wx_g > width + 1e-9
        if would_wrap and used_depth + row_depth + wy_g > depth + 1e-9:
            overflow.append(id_)
            continue
        if would_wrap:
            rows.append(cur)
            used_depth += row_depth
            cur = []
            cursor = 0.0
            row_depth = 0.0
        if used_depth + row_depth + wy_g > depth + 1e-9:
            overflow.append(id_)
            continue
        if headroom_probe is not None:
            cx = cursor + wx / 2.0
            cy = used_depth + row_depth + wy / 2.0
            headroom = headroom_probe(cx, cy, wx, wy)
            if headroom is None or headroom < item_height[id_] - 1e-6:
                overflow.append(id_)
                continue
        cur.append(id_)
        cursor += wx_g
        row_depth = max(row_depth, wy_g)
    if cur:
        rows.append(cur)
    return rows, overflow


def _shelf_columns(
    ids: list[int], footprint: dict[int, tuple[float, float]],
    width: float, depth: float, height_limit: float, layer_height: dict[int, float],
    *, space=None,
) -> tuple[list[list[list[int]]], dict[int, tuple[float, float]]]:
    """列（スロット）ベースの棚パッキングで多層シードを構成する（design文書§10.4(1)、
    §9で課題として残していた「多層シード未実装」への対処）。

    **旧実装（層ごとに独立`_shelf_rows`）の問題**: 各層が独自にfootprintで
    行分割するため、層が異なると列境界が実寸差ぶんズレる。ズレたまま段積みすると、
    上の層の荷物が下の層の荷物からわずかにはみ出し、支持率判定で落ちる候補が
    多発する（real_000相当のタスクで確定率が11/41に留まった主因と推定）。

    **本実装の方針**: 床面の「列（スロット）」を最下層（founder）の配置だけで
    1回確定し、以降の層は各列のfounder footprintに収まる荷物を高さ予算内で
    繰り返し積む**スタック**として構成する。同じ列の全層は`slot_footprint`
    （`decode_container`のオーバーライド、founderのfootprintをそのまま使う）を
    共有するため、`decode_xy`が算出する(x,y)中心が列内で完全に一致する
    ——層間の支持整合が構成的に保証される。

    `space`（既定None＝矩形コンテナ想定でheadroom未考慮）: 実コンテナの
    `ContainerSpace`。指定時、空コンテナの`ContainerHeightModel`から頭上余裕の
    粗いプローブを構築し`_shelf_rows`へ渡す——cut/棚構造で床が持ち上がる・
    天井が下がる領域（real_000相当のタスクで実在）にfounderを置いてしまうと、
    その列に積んだ全層が連鎖的にinclusion/support判定で落ちる問題への対処
    （実装時に発見、`_shelf_rows`のdocstring参照）。

    Returns:
        `(layers, slot_footprint)`。
        `layers[L]`: 深さL（0=最下層）に荷物を持つ列だけを集めた行のリスト
        （`shelf_seed_genes`が要求する`list[list[int]]`と同型）。真のoverflow
        （床の列にも収まらなかった残り）は幅だけで再パッキングした
        「もう1層」として`layers`の末尾に追加済み（下記コメント参照、
        Gamma1/2/3への単純追加によるx方向暴走を避けるため）。
        `slot_footprint`: `id -> founderのfootprint`（列の全層で共通）。
    """
    def _build_headroom_probe(space):
        if space is None:
            return None
        from .container_space import cells_of_aabb
        from .heightmap import ContainerHeightModel

        model = ContainerHeightModel(space, [])
        ox = float(space.inner_min_rel[0]) + _GAP / 2.0
        oy = float(space.inner_min_rel[1]) + _GAP / 2.0

        def _probe(cx, cy, wx, wy):
            amin = np.array([ox + cx - wx / 2.0, oy + cy - wy / 2.0, 0.0])
            amax = np.array([ox + cx + wx / 2.0, oy + cy + wy / 2.0, 0.0])
            sx, sy = cells_of_aabb(space, amin, amax)
            if sx.stop <= sx.start or sy.stop <= sy.start:
                return None
            land = float(model.lower[sx, sy].max())
            ceil_ = float(model.lower_ceiling[sx, sy].min())
            return ceil_ - land
        return _probe

    headroom_probe = _build_headroom_probe(space)
    order = sorted(ids, key=lambda i: -(footprint[i][0] * footprint[i][1]))
    rows, _floor_overflow = _shelf_rows(
        order, footprint, width, depth, headroom_probe=headroom_probe, item_height=layer_height,
    )

    # `_floor_overflow`（`_shelf_rows`が返す、床の新規行に入らなかった残り）と
    # `remaining_pool`（`ids`からfoundersを除いたもの）は集合として同一
    # （`_shelf_rows`は`order`の全idをfoundersかoverflowのどちらかに割り振るため）。
    # 以前は両方を`overflow`の算出に使い、重複id（同じidがGamma1/2/3に2回現れる
    # ＝順列として不正）を生む実装バグがあった（実装時に確定率が11/41→2/41へ
    # 悪化する形で発覚）。`remaining_pool`だけを正とする。
    founders = {f for row in rows for f in row}
    remaining_pool = [i for i in ids if i not in founders]
    used: set[int] = set(founders)
    column_rows: list[list[list[int]]] = []
    for row in rows:
        row_stacks: list[list[int]] = []
        for founder in row:
            fwx, fwy = footprint[founder]
            stack = [founder]
            h_used = layer_height[founder]
            cand_pool = sorted(
                (i for i in remaining_pool
                 if i not in used and footprint[i][0] <= fwx + 1e-9 and footprint[i][1] <= fwy + 1e-9),
                key=lambda i: layer_height[i],  # 薄い荷物から積む（層数を稼ぐ）
            )
            for cand in cand_pool:
                if h_used + layer_height[cand] > height_limit + 1e-9:
                    continue
                stack.append(cand)
                used.add(cand)
                h_used += layer_height[cand]
            row_stacks.append(stack)
        column_rows.append(row_stacks)

    max_depth = max((len(s) for row in column_rows for s in row), default=0)
    layers: list[list[list[int]]] = []
    for depth_l in range(max_depth):
        layer_rows = [
            [s[depth_l] for s in row if len(s) > depth_l]
            for row in column_rows
        ]
        layer_rows = [r for r in layer_rows if r]
        layers.append(layer_rows)

    slot_footprint: dict[int, tuple[float, float]] = {}
    for row in column_rows:
        for stack in row:
            fp = footprint[stack[0]]
            for id_ in stack:
                slot_footprint[id_] = fp

    # 真のoverflow（床にも既存の列にも収まらなかった残り）は、以前は単純に
    # Gamma1/2/3の末尾へ同じ順で追加していたが、これはGamma1・Gamma2の相対順序が
    # 一致するため`classify_pairs`で全ペアが'x'関係になり、実座標へ変換すると
    # x方向へ延々と連なって壁を大きく超過する（実装時に確定率2/41への
    # 悪化として発覚：real_000相当タスクでcxが12mまで伸びた）。
    # 幅だけで（奥行き無制限で）再度`_shelf_rows`にかけ、既存の層構成と同じ
    # 一般式が扱える「もう1層」として`layers`へ追加することで、少なくとも
    # x方向へは幅で折り返す（それでも深さ超過で大半は失敗するが、暴走はしない）。
    true_overflow = [i for i in remaining_pool if i not in used]
    if true_overflow:
        # 深さ無制限のこの最終救済passではheadroom_probeを適用しない——適用すると、
        # 頭上余裕を満たす位置を一度も試せなかった荷物が`_shelf_rows`のoverflowへ
        # 再度落ち、行き場を失ってGamma1/2/3の順列から**完全に消える**（実装時に
        # 発見: real_000相当タスクで41件中16件が忽然と消える不具合として顕在化、
        # `shelf_seed_genes`は全idを含む順列を返す契約なので致命的）。
        # ここでの目的は「全idに何らかのGamma位置を必ず与える」ことであり、
        # 実際の実行可能性判定はどのみち`decode_container`（実座標確定後の
        # `_try_place_lower`）が最終的に行うため、深さ無制限pass自体は
        # headroom未考慮のままで安全（不可行なら単にfailedへ回るだけ）。
        overflow_rows, _unused = _shelf_rows(true_overflow, footprint, width, float("inf"))
        if overflow_rows:
            layers.append(overflow_rows)
            for row in overflow_rows:
                for id_ in row:
                    slot_footprint[id_] = footprint[id_]
    return layers, slot_footprint


def shelf_seed_genes(
    order_by_container: dict[int, list[int]], footprint: dict[int, tuple[float, float]],
    container_width: dict[int, float], container_depth: dict[int, float],
    container_height: dict[int, float], layer_height: dict[int, float],
    spaces: dict[int, object] | None = None,
) -> tuple[list[int], list[int], list[int], dict[int, tuple[float, float]]]:
    """棚（shelf）パッキングの列/層割当から直接 Gamma1/Gamma2/Gamma3 を構成する
    （design文書§3.5「初期解はv52実際の配置とほぼ一致させる」の具体化。段積み・
    列整合対応版、実装時に単層版→独立層版の順に拡張した。§10.4(1)参照）。

    導出（`classify_pairs`の定義から機械的に従う。層をL=1..K、各層内の行をR=1..M_Lとする。
    層・行は`_shelf_columns`が返す列構造由来——同じ列の異なる層は`slot_footprint`を
    共有するfounder（最下層）のfootprintで揃っている）:

      Gamma1 = 層1(行1左→右, 行2左→右, …) → 層2(同様) → … → 層K
      Gamma2 = 層K(各層内は行**逆順**、行内は左→右) → … → 層1
      Gamma3 = 層K(各層内は行**順順**=Gamma1と同じ、行内は左→右) → … → 層1

    （Gamma2・Gamma3は共に層の並びを**反転**するが、層内の行の並びだけが異なる
    ——Gamma2は反転、Gamma3は非反転。この非対称性が下表の分類を生む）:

      同層・同行のペア: Gamma1・Gamma2で相対順序が一致（行順序反転は同一行内の
        左右関係に影響しない）→ 'x'
      同層・異なる行のペア: Gamma2は層内行順序が反転しているため不一致 → Gamma3は
        層内行順序がGamma1と同じなので一致 → 'y'
      異なる層のペア: Gamma2・Gamma3はともに層の並びを反転しているため、
        どちらもGamma1と不一致 → 'z'

    （単層(K=1)の場合、層の反転は自明（長さ1の列の反転は同一）に帰着し、
    旧実装（Gamma3=Gamma1、z関係なし）と完全に一致する。）

    `container_height`（実z-extent）に達するまで`_shelf_columns`で列を積む。
    真のoverflow（床の列にも収まらなかった残り）は`_shelf_columns`が幅だけで
    再パッキングして「もう1層」として`layers`へ組み込み済みのため、本関数は
    特別扱いしない（以前はGamma1/2/3の末尾へ単純追加しておりx方向に暴走する
    不具合があった、`_shelf_columns`のコメント参照）。

    Returns:
        `(g1, g2, g3, slot_footprint)`。`slot_footprint`は
        `decode_container(..., slot_footprint=...)`へそのまま渡す。
    """
    g1: list[int] = []
    g2: list[int] = []
    g3: list[int] = []
    slot_footprint: dict[int, tuple[float, float]] = {}
    for cidx in sorted(order_by_container):
        layers, cont_slot_footprint = _shelf_columns(
            order_by_container[cidx], footprint, container_width[cidx], container_depth[cidx],
            container_height[cidx], layer_height,
            space=(spaces.get(cidx) if spaces else None),
        )
        slot_footprint.update(cont_slot_footprint)

        for layer in layers:
            for row in layer:
                g1.extend(row)

        for layer in reversed(layers):
            for row in reversed(layer):
                g2.extend(row)

        for layer in reversed(layers):
            for row in layer:
                g3.extend(row)
    return g1, g2, g3, slot_footprint


# --- design文書§6: `agent.py::optimize()` 統合エントリポイント ----------------------


def _envf_local(name: str, default: float) -> float:
    """`agent.py`の同名ヘルパと同型。未設定・数値化不能は既定値へフォールバック。"""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _build_item_specs(item_list: list[dict]) -> dict[int, dict]:
    specs: dict[int, dict] = {}
    for it in item_list:
        idx = int(it["index"])
        specs[idx] = {
            "size": np.array([float(it["length"]), float(it["width"]), float(it["height"])], dtype=np.float64),
            "mass": float(it.get("mass", 0.0)),
            "is_soft": bool(it.get("is_soft")),
            "is_priority": bool(it.get("is_prioritized")),
        }
    return specs


def _flat_orientation(size: Vec3) -> int:
    """扁平姿勢（最小辺を鉛直に）の向きを1つ返す（実測489/489が扁平で置かれる、findings §19.9）。"""
    from .rollout_plan import _flat_orientations
    opts = _flat_orientations(size)
    return int(opts[0][0]) if opts else 0


def _seed_container_assignment(item_specs: dict[int, dict], container_list: list[dict]) -> dict[int, int]:
    """優先品を優先コンテナへ、残りを体積バランスで割り当てる初期解（design文書§3.5・§9未決事項3、
    v1では固定ヒューリスティックとして実装し、探索対象には含めない簡略化）。"""
    n_cont = len(container_list)
    prio_conts = [i for i, c in enumerate(container_list) if bool(c.get("is_prioritized"))]
    load = [0.0] * n_cont
    assign: dict[int, int] = {}
    by_vol = sorted(item_specs.items(), key=lambda kv: -float(np.prod(kv[1]["size"])))
    for id_, spec in by_vol:
        candidates = prio_conts if (spec["is_priority"] and prio_conts) else list(range(n_cont))
        cidx = min(candidates, key=lambda c: load[c])
        assign[id_] = cidx
        load[cidx] += float(np.prod(spec["size"]))
    return assign


def _verify_plan_via_rollout(task: dict, order: list[int], plan: dict, rank: dict, total: int, budget_s: float):
    """design文書§4.4: `plan`を実物理（`_RolloutEnv`）で1手ずつ実行し `(n_placed, volume)` を返す。

    `agent.py::_run_pipeline`の消費ロジック（可視プールのうちplan該当品でrank最小のものを
    planどおり適用、無ければ既存貪欲へフォールバック）を、採用判定のための忠実な
    シミュレーションとして再現する。既存 `_RolloutEnv`/HF-040 の物理検証パターン
    （`agent.py`の`GH_ORDER_SEARCH_VERIFY`分岐）と同一の考え方。
    """
    from .rollout_plan import _RolloutEnv
    from .types import Candidate

    e = _RolloutEnv(task, order=order)
    try:
        t0 = time.monotonic()
        done = False
        guard = 0
        while not done and guard < total + 5:
            guard += 1
            if time.monotonic() - t0 > budget_s:
                break
            pool = e.env.stream_manager.visible_pool
            visible = [(j, int(po.index)) for j, po in enumerate(pool) if int(po.index) in plan]
            if visible:
                j, fid = min(visible, key=lambda t: rank.get(t[1], 1 << 30))
                ci, pos, orn = plan[fid]
                cand = Candidate(
                    item_idx=int(j), container_idx=int(ci), ems_id=0, orientation=int(orn),
                    pos_rel=np.asarray(pos, dtype=np.float64), osize=np.ones(3, dtype=np.float64),
                )
                _placed_ok, done = e.apply(cand)
            else:
                cs = e.cands(1)
                if not cs:
                    break
                _placed_ok, done = e.apply(cs[0])
        return e.n_placed, e.volume
    finally:
        e.close()


def seq_triple_optimize(
    container_list: list[dict], item_list: list[dict], lookahead_k: int, budget_s: float = 150.0,
    post_fn=None,
):
    """課題A（`la=1`）向けSequence Triple計画（design文書§6の統合スケッチに対応）。

    3順列・向き・コンテナ割当の遺伝子空間をヒルクライミング（`order.py::_apply_move`を
    再利用、決定論的マルチスタート＋停滞時アンカー切替、design文書§3.5/§10.4(2)）で
    探索し、`decode_all`（§3.4-4.2、既定OFFの`W_CLUSTER`込み§5.1）を評価器にする。
    採用は§4.4のとおり実物理（`_RolloutEnv`）で既定貪欲floorと比較し、
    `(n_placed, volume)`がfloorを上回った時のみ行う（`plan_optimize`/`deeplow_optimize`
    と同じ非退行原則）。

    `post_fn`（既定None＝恒等関数）: **floorの構成に必須**（実装時に発見した重大な
    バグへの対処）。`agent.py::optimize()`の既定経路は
    `_post(plan_order_forward_sim(...))`であり（`_post`はsoft/priority/heavy/tall
    シフト・インターリーブをまとめた順序後処理、agent.py:623）、`GH_ORDER_SEARCH_VERIFY`
    （HF-040）のfloorも`_post(plan_order_forward_sim(...))`を使っている
    （agent.py:685）。本関数が`post_fn`無しで生の`plan_order_forward_sim`をfloorに
    使うと、**実際のv52本番挙動より弱いfloor**と比較することになり、v52を下回る
    候補を誤って「floorを上回った」と判定して採用してしまう——実装時、f2_0000で
    fill 28.55→20.12・np 0.65→0.50という明確な退行を誤って採用する形で発覚した
    （agent.py側の呼び出しで`post_fn=_post`を渡す契約、§6参照）。
    候補自身の`order`には適用しない（`plan`/`rank`と対応した明示的な位置決めを
    崩さないため。§4.3参照）。

    Returns:
        `(order, plan, rank)`（`rollout_plan.py`の契約と同一）または`None`
        （非採用・材料不正・例外時。呼び出し側=agent.pyが既存順序計画へフォールバックする）。
    """
    try:
        from .constants import BEAM_W, ORDER_STAGNANT_RESTART, GridParams
        from .container_space import build_container_space
        from .order import _apply_move
        from .state import _packed_item_to_placed

        identity = (lambda order: order) if post_fn is None else post_fn
        t0 = time.monotonic()
        deadline = t0 + budget_s
        cell = GridParams().cell

        spaces = [build_container_space(cd, i, cell) for i, cd in enumerate(container_list)]
        if not spaces:
            return None
        base_placed: dict[int, list] = {}
        for i, cd in enumerate(container_list):
            base_placed[i] = [_packed_item_to_placed(pi, spaces[i]) for pi in (cd.get("packed_items") or [])]

        item_specs = _build_item_specs(item_list)
        all_ids = list(item_specs.keys())
        n = len(all_ids)
        if n == 0:
            return None

        # §4.4: floorを最初に、固定予算で検証する（HF-040`GH_ORDER_SEARCH_VERIFY`と
        # 同一パターン、`agent.py:685`）。**実装時に発見した重大なバグへの対処**:
        # 当初はSA探索が終わった後の残り予算（縮小した端数）でfloorを検証していたが、
        # floorの検証（`_construct(rule="greedy")`、毎手候補探索を伴うため高コスト）と
        # 自分の候補の検証（`_verify_plan_via_rollout`、計画を直接適用するだけで安価）は
        # 単位時間あたりに処理できる荷物数が大きく異なる。SA消費後の少ない残り予算を
        # 両者に同じ割合で配分すると、**高コストなfloor側だけが時間切れで早期に打ち切られ、
        # 実際より大幅に少ない`floor_np`を報告してしまう**——f2_0000（80品/2容器）で
        # `floor_np=34`という過小評価が生じ、実際のv52本番実行（np=52相当）を大きく
        # 下回っていたにもかかわらず、自分の候補（40品）が「floorを上回った」と誤判定
        # されて採用される事故が発生した。floorの検証だけを他と独立に、固定の
        # `verify_reserve_s`（HF-040と同じ25秒）で先に完了させることで、SAが後から
        # どれだけ時間を使っても比較の公平性が損なわれないようにする。
        margin_s = 5.0
        verify_reserve_s = 25.0
        from .order import plan_order_forward_sim
        from .rollout_plan import _RolloutEnv, _construct, reconstruct_task

        task = reconstruct_task(container_list, item_list, lookahead_k)
        floor_budget = max(10.0, min(90.0, deadline - time.monotonic() - margin_s))
        floor_order = identity(plan_order_forward_sim(item_list, container_list, floor_budget))
        ef = _RolloutEnv(task, order=floor_order)
        try:
            floor_np, _plan_unused = _construct(
                ef, ef.total, budget_s=verify_reserve_s, rule="greedy", record=False,
            )
            floor_vol = float(ef.volume)
        finally:
            ef.close()

        container_map = _seed_container_assignment(item_specs, container_list)
        orn_map = {i: _flat_orientation(item_specs[i]["size"]) for i in all_ids}

        footprint0 = {id_: item_footprint(item_specs[id_]["size"], orn_map[id_])[:2] for id_ in all_ids}
        layer_height = {id_: item_footprint(item_specs[id_]["size"], orn_map[id_])[2] for id_ in all_ids}
        width_map = {
            c: float(spaces[c].inner_max_rel[0] - spaces[c].inner_min_rel[0]) for c in range(len(spaces))
        }
        depth_map = {
            c: float(spaces[c].inner_max_rel[1] - spaces[c].inner_min_rel[1]) for c in range(len(spaces))
        }
        height_map = {
            c: float(spaces[c].inner_max_rel[2] - spaces[c].inner_min_rel[2]) for c in range(len(spaces))
        }
        space_map = {c: spaces[c] for c in range(len(spaces))}

        w_fill = _envf_local("GH_SEQ_W_FILL", 0.268)
        w_cog = _envf_local("GH_SEQ_W_COG", 0.261)
        w_stair = _envf_local("GH_SEQ_W_STAIR", 50.0)
        w_cluster = _envf_local("GH_SEQ_W_CLUSTER", 0.0)  # design文書§5.1、既定OFF
        hz_ref = max((float(c["height"]) for c in container_list), default=1.0)
        n_cont = len(spaces)

        # slot_footprintは遺伝子と一緒にタプルで運ぶ（アンカー＝seedごとに列構造が
        # 異なるため）。SAの近傍操作を経ても元アンカーのslot_footprintを固定のまま
        # 使い続ける——遺伝子（Gamma1/2/3・向き・コンテナ割当）が変異してもx/yの
        # 非重なり保証は常にdecode_xyの最長路そのものから来るため、footprintの値に
        # 依存しない（固定のままでも安全性は損なわれない、§10.4実装ノート）。
        def _evaluate(genes):
            order1, order2, order3, orn, cmap, slot_fp = genes
            committed_all, plan_all, failed_all, state = decode_all(
                spaces, base_placed, item_specs, order1, order2, order3, orn, cmap,
                slot_footprint=slot_fp,
            )
            fill, cog = fill_and_cog(spaces, state.placed)
            stair = sum(stair_violation(spaces[c], state.placed[c]) for c in range(n_cont))
            cluster = 0.0
            if w_cluster != 0.0:
                cluster = sum(cluster_score(state.placed[c]) for c in range(n_cont))
            n_committed = n - len(failed_all)
            j_val = (
                w_fill * fill * 100.0 + w_cog * (1.0 - cog / hz_ref) * 100.0
                - w_stair * stair + w_cluster * cluster
            )
            return j_val, n_committed, committed_all, plan_all

        # design文書§10.4(2): 決定論的マルチスタート。単一の「体積降順」シードだけでは
        # real_000相当のタスク（cut/棚構造・大きい荷物比率）で局所的に不利な列構造に
        # 嵌りやすいと判明したため、`order.py::search_order_composite`と同型の
        # 複数アンカー＋停滞時ローテーション方式を導入する（実装時に単一シードの
        # ヒルクライミングだけではfloorに届かなかったことを踏まえた拡張）。
        seed_keys = [
            ("volume_desc", lambda i: -float(np.prod(item_specs[i]["size"]))),
            ("footprint_desc", lambda i: -(footprint0[i][0] * footprint0[i][1])),
            ("thin_first", lambda i: layer_height[i]),
            ("thick_first", lambda i: -layer_height[i]),
            ("mass_desc", lambda i: -float(item_specs[i].get("mass", 0.0))),
        ]
        anchors: list[tuple] = []
        for _name, key in seed_keys:
            order_by_container: dict[int, list[int]] = {c: [] for c in range(len(spaces))}
            for id_ in sorted(all_ids, key=key):
                order_by_container[container_map[id_]].append(id_)
            g1, g2, g3, slot_fp = shelf_seed_genes(
                order_by_container, footprint0, width_map, depth_map, height_map, layer_height,
                spaces=space_map,
            )
            if sorted(g1) != sorted(all_ids):
                continue  # 防御: このシードが不正なら候補から除外する（他のシードで継続）
            anchors.append((g1, g2, g3, dict(orn_map), dict(container_map), slot_fp))
        if not anchors:
            return None  # 防御: 全シードが不正なら安全側で不採用

        rng = np.random.default_rng(20260811)
        evaluated = [(genes, _evaluate(genes)) for genes in anchors]
        evaluated.sort(key=lambda t: (t[1][1], t[1][0]), reverse=True)
        anchor_pool = evaluated[: max(1, BEAM_W)]

        active = 0
        cur_genes, (cur_j, cur_n, cur_committed, cur_plan) = anchor_pool[active]
        best_j, best_n = cur_j, cur_n
        best_committed, best_plan = cur_committed, cur_plan

        # 自分の候補の実物理検証（後述）に固定`verify_reserve_s`を残す（floorは既に検証済み）。
        search_deadline = deadline - verify_reserve_s - margin_s
        guard_iters = 0
        stagnant = 0
        while time.monotonic() < search_deadline and guard_iters < 200000:
            guard_iters += 1
            o1, o2, o3, orn, cmap, slot_fp = (
                list(cur_genes[0]), list(cur_genes[1]), list(cur_genes[2]),
                dict(cur_genes[3]), dict(cur_genes[4]), cur_genes[5],
            )
            move = int(rng.integers(4))
            if move == 0:
                o1 = _apply_move(o1, int(rng.integers(3)), rng)
            elif move == 1:
                o2 = _apply_move(o2, int(rng.integers(3)), rng)
            elif move == 2:
                o3 = _apply_move(o3, int(rng.integers(3)), rng)
            else:
                id_ = all_ids[int(rng.integers(n))]
                if n_cont > 1 and rng.random() < 0.3:
                    cmap[id_] = int(rng.integers(n_cont))
                else:
                    orn[id_] = int(rng.integers(6))
            neighbor = (o1, o2, o3, orn, cmap, slot_fp)
            j_val, n_committed, committed_all, plan_all = _evaluate(neighbor)
            if (n_committed, j_val) > (cur_n, cur_j):
                cur_genes = neighbor
                cur_j, cur_n, cur_committed, cur_plan = j_val, n_committed, committed_all, plan_all
                stagnant = 0
                if (n_committed, j_val) > (best_n, best_j):
                    best_j, best_n = j_val, n_committed
                    best_committed, best_plan = committed_all, plan_all
            else:
                stagnant += 1
                if stagnant >= ORDER_STAGNANT_RESTART and len(anchor_pool) > 1:
                    active = (active + 1) % len(anchor_pool)
                    cur_genes, (cur_j, cur_n, cur_committed, cur_plan) = anchor_pool[active]
                    stagnant = 0

        order, plan, rank = to_order_and_plan(best_committed, best_plan, all_ids)
        if sorted(order) != sorted(all_ids):
            return None  # 防御: 完全順列でなければ不採用

        # §4.4: 採用前に既定貪欲floor（既に固定予算で検証済み、上記）と実物理で比較する
        # （非退行フロア、HF-040と同一パターン）。自分の候補も同じ固定`verify_reserve_s`で
        # 検証する（floorとの比較を公平にするため、実装時に発見したバグの教訓）。
        remain2 = deadline - time.monotonic()
        if remain2 < 5.0:
            return None
        seq_np, seq_vol = _verify_plan_via_rollout(
            task, order, plan, rank, n, min(verify_reserve_s, remain2),
        )

        adopted = (seq_np, seq_vol) > (floor_np, floor_vol)
        if os.environ.get("GH_SEQ_TRIPLE_FORCE", "0") != "0":
            # 診断専用（可視化・調査目的、design文書§11.9）: §4.4のfloor比較を無視して
            # 常に自分の候補を採用する。提出物では絶対に使わない（既定OFF、GH_SEQ_TRIPLE単独
            # では効果が無く、この変数を明示的に立てた場合のみ働く二重ゲート）。
            adopted = True
        if os.environ.get("GH_SEQ_TRIPLE_DEBUG", "0") != "0":
            # HF-040の`GH_ORDER_SEARCH_DEBUG`（agent.py）と同型の診断出力。
            best_seed_n = max((a[1][1] for a in evaluated), default=-1)
            print(
                f"HF-045-DEBUG n={n} best_seed_n={best_seed_n} sa_best_n={best_n} "
                f"floor=({floor_np},{floor_vol:.4f}) seq=({seq_np},{seq_vol:.4f}) "
                f"adopted={adopted} elapsed={time.monotonic()-t0:.1f}s",
                flush=True,
            )
        if adopted:
            return order, plan, rank
        return None
    except Exception:
        if os.environ.get("GH_SEQ_TRIPLE_DEBUG", "0") != "0":
            import traceback
            traceback.print_exc()
        return None
