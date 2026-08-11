"""模倣学習ウォームスタート（DRL方策計画 Phase 2a、`docs/HANDOFF.md` 参照）。

`collect_imitation_data.py` が集めた (候補特徴行列, 文脈ベクトル, ラベル) を教師データとして
`packing_core/rl_model.py::RLPolicyNet` を学習する。目的はゼロからのRLではなく、
「v52既定ヒューリスティックと同等の選択ができる状態」から次のRLファインチューン
（Phase 2b/3）を始めること（GH_DEEPLOW/Sequence Tripleの教訓——単純な初期解は
蓄積されたスコアリングに正面から負ける——を踏まえた計画上の理由）。

損失は候補集合内の多クラス分類（cross entropy、target=教師ラベルindex）。候補数 N は
ステップごとに可変なので、バッチはサンプル単位のループ＋勾配蓄積で構成する
（N別のpaddingを避けた最小実装、データ規模が大きくなれば要見直し）。

使い方:
    cd simulator && source .venv/bin/activate
    python -m training.rl_packing.imitation \
        --data-dir artifacts/rl_training/imitation/f1 \
        --data-dir artifacts/rl_training/imitation/f2 \
        --out-dir artifacts/rl_training/imitation_ckpt --epochs 20
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
from datetime import datetime

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None  # ローカルCPU venvにtensorboard未導入でも動くようにする（GPU学習コンテナでは利用可）

from agents.heuristic.packing_core.rl_model import RLPolicyNet


class _NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


def _load_episodes(data_dirs: list[str]) -> list[list[dict]]:
    """episode単位でサンプルをまとめて返す（train/val分割をepisode単位で行うため）。"""
    episodes = []
    for d in data_dirs:
        for path in sorted(glob.glob(os.path.join(d, "ep_*.pkl"))):
            with open(path, "rb") as f:
                samples = pickle.load(f)
            if samples:
                episodes.append(samples)
    return episodes


def _split_train_val(episodes: list[list[dict]], val_frac: float, seed: int
                      ) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(episodes))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(episodes) * val_frac))) if len(episodes) > 1 else 0
    val_eps = [episodes[i] for i in idx[:n_val]]
    train_eps = [episodes[i] for i in idx[n_val:]]
    train = [s for ep in train_eps for s in ep]
    val = [s for ep in val_eps for s in ep]
    return train, val


@torch.no_grad()
def _evaluate(model: RLPolicyNet, samples: list[dict], device: str) -> tuple[float, float]:
    if not samples:
        return float("nan"), float("nan")
    model.eval()
    total_loss = 0.0
    n_correct = 0
    for s in samples:
        cand_feats = torch.from_numpy(s["cand_feats"]).to(device)
        ctx_feat = torch.from_numpy(s["ctx_feat"]).to(device)
        label = int(s["label"])
        scores = model(cand_feats, ctx_feat)
        loss = F.cross_entropy(scores.unsqueeze(0), torch.tensor([label], device=device))
        total_loss += float(loss.item())
        if int(torch.argmax(scores).item()) == label:
            n_correct += 1
    model.train()
    return total_loss / len(samples), n_correct / len(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", action="append", required=True,
                         help="collect_imitation_data.py の --out-dir。複数指定可。")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-accum", type=int, default=16,
                         help="候補数Nがサンプルごとに違うためバッチ化せず、この件数ごとに"
                              "勾配を蓄積してoptimizer.step()する（疑似バッチサイズ）。")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    episodes = _load_episodes(args.data_dir)
    if not episodes:
        raise SystemExit(f"no episodes found under {args.data_dir}")
    train_samples, val_samples = _split_train_val(episodes, args.val_frac, args.seed)
    print(f"episodes: {len(episodes)}  train_samples: {len(train_samples)}  "
          f"val_samples: {len(val_samples)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    writer = (SummaryWriter(log_dir=os.path.join(out_dir, "tensorboard"))
              if SummaryWriter is not None else _NullWriter())

    model = RLPolicyNet().to(args.device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    rng = np.random.default_rng(args.seed)
    global_step = 0
    best_val_acc = -1.0
    for epoch in range(args.epochs):
        order = np.arange(len(train_samples))
        rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        n_correct = 0
        for i, idx in enumerate(order):
            s = train_samples[idx]
            cand_feats = torch.from_numpy(s["cand_feats"]).to(args.device)
            ctx_feat = torch.from_numpy(s["ctx_feat"]).to(args.device)
            label = int(s["label"])
            scores = model(cand_feats, ctx_feat)
            loss = F.cross_entropy(scores.unsqueeze(0), torch.tensor([label], device=args.device))
            (loss / args.grad_accum).backward()
            running_loss += float(loss.item())
            if int(torch.argmax(scores).item()) == label:
                n_correct += 1
            if (i + 1) % args.grad_accum == 0 or i == len(order) - 1:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            global_step += 1

        train_loss = running_loss / max(len(order), 1)
        train_acc = n_correct / max(len(order), 1)
        val_loss, val_acc = _evaluate(model, val_samples, args.device)
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/acc", val_acc, epoch)
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_acc > best_val_acc or not val_samples:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epochs": args.epochs}, os.path.join(out_dir, "training_state.pt"))
    writer.close()
    print(f"\nbest val_acc: {best_val_acc:.3f}")
    print(f"checkpoint (state_dict only, for agents/heuristic/agent.py 推論用): "
          f"{os.path.join(out_dir, 'model.pt')}")
    print(f"TensorBoard: {os.path.join(out_dir, 'tensorboard')}")


if __name__ == "__main__":
    main()
