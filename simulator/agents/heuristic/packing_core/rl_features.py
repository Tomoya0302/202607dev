"""候補・グローバル文脈 → 特徴ベクトル変換（DRL方策計画 Phase 1、`docs/HANDOFF.md` 参照）。

`heightmap.py::decide_candidates` が返す安全性フィルタ済み `Candidate` 群を、
`packing_core/rl_model.py` のスコアリングネットワークへ渡せる固定長ベクトルへ変換する。
副作用のない純粋関数のみで構成し、学習・推論の両方から同一実装を共有する
（学習/推論間の特徴量ドリフトを防ぐ、計画Phase 1の方針）。

既存資産の再利用:
- 候補の安定性特徴量（支持率・粗さ・tippiness等12項目）は
  `heightmap.py::support_features`（HF-022、`STAB_FEATURE_NAMES`）をそのまま呼ぶ。
  独自に再実装しない。
- 候補の幾何・veto系フラグは `Candidate.features` 辞書（`stair_bad`/`soft_bad`/`prio_bad`等、
  カスケードの3段階で「消さず・desperate段のみ許可」対象を示す）をそのまま使う。
"""
from __future__ import annotations

import numpy as np

from .heightmap import STAB_FEATURE_NAMES, support_features
from .types import Candidate

# 候補1件あたりの特徴次元数（このモジュールの出力形状の唯一の定義源）。
#   pos_rel(3) + osize(3) + orientation one-hot(6) + score(1)
#   + features フラグ(7: hm_upper/soft_nonflat/floor/stair_bad/rough_bad/soft_bad/prio_bad)
#   + support_features(12, STAB_FEATURE_NAMES) + item属性(mass/is_soft/is_prioritized = 3)
CAND_FEATURE_DIM = 3 + 3 + 6 + 1 + 7 + len(STAB_FEATURE_NAMES) + 3

# グローバル文脈の特徴次元数。
#   pool: n_pool, pool_mass_sum, pool_n_soft, pool_n_prio (4)
#   containers: n_containers (1)
#   + コンテナごとの要約統計を最大 MAX_CONTAINERS 件、各4次元
#     (n_placed, mass_placed, mean_top_z, max_top_z) にパディング
MAX_CONTAINERS = 6
CTX_FEATURE_DIM = 4 + 1 + MAX_CONTAINERS * 4

_CAND_FLAG_KEYS = ("hm_upper", "soft_nonflat", "floor", "stair_bad", "rough_bad",
                    "soft_bad", "prio_bad")


def _item_meta_for(state, item_idx: int) -> dict:
    """`state.pool`（`ItemSpec`一覧）から `item_idx` に対応する荷物属性を引く。"""
    for spec in state.pool:
        if int(spec.idx) == int(item_idx):
            return {"mass": float(spec.weight), "is_soft": bool(spec.is_soft),
                    "is_prioritized": bool(spec.is_priority)}
    return {"mass": 0.0, "is_soft": False, "is_prioritized": False}


def candidate_feature_vector(cand: Candidate, model, item_meta: dict) -> np.ndarray:
    """1候補分の特徴ベクトルを返す。shape (CAND_FEATURE_DIM,), float32。"""
    orn_onehot = np.zeros(6, dtype=np.float32)
    orn_onehot[int(cand.orientation) % 6] = 1.0

    flags = np.array([1.0 if cand.features.get(k) else 0.0 for k in _CAND_FLAG_KEYS],
                      dtype=np.float32)

    stab = support_features(model, cand, item_meta)
    stab_vec = np.array([float(stab.get(k, 0.0)) for k in STAB_FEATURE_NAMES], dtype=np.float32)

    item_vec = np.array([
        float(item_meta.get("mass", 0.0)),
        1.0 if item_meta.get("is_soft") else 0.0,
        1.0 if item_meta.get("is_prioritized") else 0.0,
    ], dtype=np.float32)

    vec = np.concatenate([
        np.asarray(cand.pos_rel, dtype=np.float32).reshape(3),
        np.asarray(cand.osize, dtype=np.float32).reshape(3),
        orn_onehot,
        np.array([float(cand.score)], dtype=np.float32),
        flags,
        stab_vec,
        item_vec,
    ])
    assert vec.shape == (CAND_FEATURE_DIM,), (vec.shape, CAND_FEATURE_DIM)
    return vec


def candidate_batch_features(candidates: list[Candidate], models, state) -> np.ndarray:
    """複数候補をまとめて特徴行列へ変換する。shape (N, CAND_FEATURE_DIM), float32。

    候補0件のときは shape (0, CAND_FEATURE_DIM) を返す。
    """
    if not candidates:
        return np.zeros((0, CAND_FEATURE_DIM), dtype=np.float32)
    rows = []
    item_meta_cache: dict[int, dict] = {}
    for cand in candidates:
        if cand.item_idx not in item_meta_cache:
            item_meta_cache[cand.item_idx] = _item_meta_for(state, cand.item_idx)
        model = models[cand.container_idx]
        rows.append(candidate_feature_vector(cand, model, item_meta_cache[cand.item_idx]))
    return np.stack(rows, axis=0)


def global_context_vector(state) -> np.ndarray:
    """現在の状態（pool・各コンテナの積載状況）をグローバル文脈ベクトルへ変換する。

    shape (CTX_FEATURE_DIM,), float32。コンテナ数が MAX_CONTAINERS を超える分は切り捨て、
    不足分は0パディングする（既定タスクは1〜2コンテナのため通常は発生しない）。
    """
    pool = state.pool
    n_pool = float(len(pool))
    pool_mass = float(sum(float(s.weight) for s in pool))
    pool_n_soft = float(sum(1 for s in pool if s.is_soft))
    pool_n_prio = float(sum(1 for s in pool if s.is_priority))

    n_containers = float(len(state.containers))
    cont_rows = []
    for cidx in range(len(state.containers)):
        if cidx >= MAX_CONTAINERS:
            break
        placed = state.placed.get(cidx, [])
        n_placed = float(len(placed))
        mass_placed = float(sum(float(p.weight) for p in placed))
        if placed:
            tops = [float(p.aabb_max_rel[2]) for p in placed]
            mean_top = float(np.mean(tops))
            max_top = float(np.max(tops))
        else:
            mean_top = 0.0
            max_top = 0.0
        cont_rows.append([n_placed, mass_placed, mean_top, max_top])
    while len(cont_rows) < MAX_CONTAINERS:
        cont_rows.append([0.0, 0.0, 0.0, 0.0])

    vec = np.concatenate([
        np.array([n_pool, pool_mass, pool_n_soft, pool_n_prio], dtype=np.float32),
        np.array([n_containers], dtype=np.float32),
        np.asarray(cont_rows, dtype=np.float32).reshape(-1),
    ])
    assert vec.shape == (CTX_FEATURE_DIM,), (vec.shape, CTX_FEATURE_DIM)
    return vec
