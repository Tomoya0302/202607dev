"""候補再ランキング用スコアリングネットワーク（DRL方策計画 Phase 1、`docs/HANDOFF.md` 参照）。

学習（`simulator/training/rl_packing/`, GPU）・推論（`agent.py` の `GH_RL_POLICY` フック, CPU）の
両方から本モジュールを共有インポートする。アーキテクチャの定義源をここ1箇所に絞ることで、
学習済みチェックポイントと推論側モデル定義のドリフトを防ぐ（計画Phase 1の方針）。

設計方針: `decide_candidates` が返す安全性フィルタ済み候補集合に対する順列不変な
再ランキングのみを学習する（候補生成・実行可能性判定は一切学習しない、既存の
GOPT/PCT型文献のPlacement Generator部分をそのまま既存heightmap.pyに委ねるハイブリッド構成）。
CPU・8秒/手の推論予算に収めるため、GOPT型の重厚なTransformerは狙わず、共有MLP
（deep sets）+ 軽量self-attention1層という最小構成にする。
"""
from __future__ import annotations

import torch
from torch import nn

from .rl_features import CAND_FEATURE_DIM, CTX_FEATURE_DIM


class RLPolicyNet(nn.Module):
    """候補特徴行列 (N, CAND_FEATURE_DIM) + 文脈ベクトル (CTX_FEATURE_DIM,)
    → 候補ごとのスコア (N,) を返す、順列不変な軽量スコアリングネット。

    Args:
        hidden_dim: 候補embeddingの次元。既定64（CPU推論の速度を優先し小さめに保つ）。
        n_attn_heads: 候補間self-attentionのヘッド数。0でattention層を無効化する
            （候補ごと独立スコアのdeep sets構成のみになる）。
    """

    def __init__(self, hidden_dim: int = 64, n_attn_heads: int = 4):
        super().__init__()
        in_dim = CAND_FEATURE_DIM + CTX_FEATURE_DIM
        self.encoder = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.n_attn_heads = n_attn_heads
        if n_attn_heads > 0:
            self.attn = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=n_attn_heads, batch_first=True)
            self.attn_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, cand_feats: torch.Tensor, ctx_feat: torch.Tensor) -> torch.Tensor:
        """Args:
            cand_feats: shape (N, CAND_FEATURE_DIM), float32。N=0のとき shape (0,) を返す。
            ctx_feat: shape (CTX_FEATURE_DIM,), float32。

        Returns:
            shape (N,) の候補ごとスコア（logit、softmaxやargmaxは呼び出し側の責務）。
        """
        n = cand_feats.shape[0]
        if n == 0:
            return cand_feats.new_zeros(0)
        ctx_rep = ctx_feat.unsqueeze(0).expand(n, -1)
        x = torch.cat([cand_feats, ctx_rep], dim=-1)
        h = self.encoder(x)  # (N, hidden_dim)
        if self.n_attn_heads > 0:
            h_seq = h.unsqueeze(0)  # (1, N, hidden_dim), batch_first
            attn_out, _ = self.attn(h_seq, h_seq, h_seq, need_weights=False)
            h = self.attn_norm(h + attn_out.squeeze(0))
        scores = self.head(h).squeeze(-1)  # (N,)
        return scores


def load_policy(checkpoint_path: str, *, device: str = "cpu") -> RLPolicyNet:
    """推論用: チェックポイントからモデルを読み込む（`GPU_TRAINING.md` の運用原則どおり
    `map_location="cpu"` 相当で読み込み、推論はCPUのみで完結させる）。
    """
    model = RLPolicyNet()
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model
