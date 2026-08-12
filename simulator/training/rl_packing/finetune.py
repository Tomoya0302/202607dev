"""RLファインチューン本体（DRL方策計画 Phase 2b/3、`docs/HANDOFF.md` 参照）。

Phase 2a の模倣学習チェックポイントを初期値に、方策勾配（REINFORCE + 移動平均baseline）で
ファインチューンする。1手の決定は「`decide_candidates`が返す安全性フィルタ済み候補集合から
1つを選ぶ」カテゴリカル分布なので、無効な行動が生成されることは構造的にない
（探索中も本番同様、既存の安全性クリティカルなロジックには一切触れない）。

報酬はローカルで正確に計算できる2値（`fill_score`/`num_placed_items`）+ Phase 0で検証した
支持多角形ベース安定性proxy（`scripts/bench_run.py::_static_equilibrium_proxy`）の
線形結合。cog_score/stability_score はサーバー側にしか実装が無いため、直接の学習信号には
できない（計画の既知の限界、Phase 0の合意どおり）。

自己対戦データ収集は `ProcessPoolExecutor` でCPU並列化する。各反復の直前に現在の重みを
一時チェックポイントへ保存し、ワーカーはそこから読み込む（プロセス間でtensorを直接渡さず、
既存の `_RolloutEnv`/`eval_chunked.py` と同じ「ファイル経由」の単純さを踏襲）。
方策勾配の対数確率は、収集後にメインプロセスで同じ重み・同じ入力に対し勾配ありで
再計算する（決定論的なforward passなので収集時no_gradの結果と完全に一致し、stale/
off-policyにならない）。

使い方（GPU学習コンテナ内、`docker-compose.gpu.yml` 経由を想定。CPUでも動作する）:
    cd simulator && source .venv/bin/activate  # または GPU学習コンテナ内
    python -m training.rl_packing.finetune \
        --init-ckpt artifacts/rl_training/imitation_ckpt/20260811_141349/model.pt \
        --out-dir artifacts/rl_training/finetune_ckpt \
        --iterations 2000 --episodes-per-iter 24 --workers 12 \
        --eval-every 20 --checkpoint-every 20
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None


class _NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


class _StepBudget:
    def __init__(self, t0: float, soft: float, hard: float):
        self.t0 = t0
        self.soft = soft
        self.hard = hard

    def over_soft(self) -> bool:
        return (time.monotonic() - self.t0) > self.soft

    def over_hard(self) -> bool:
        return (time.monotonic() - self.t0) > self.hard


def compute_reward(fill_score: float, num_placed: float, eq_frac_violation: float,
                    np_weight: float, eq_weight: float) -> float:
    """report済みPhase 0 findings に基づく報酬合成。

    fill_score は0-100スケールでそのまま使う。num_placed(0-1) は `np_weight` 倍して
    fill_scoreと同程度の大きさへ揃える（局所npが本番npへ最も信頼できる転移を示す指標
    という`docs/findings_2026-07.md`の知見を反映し、他の2項より軽視しない）。
    equilibrium proxy はfrac_violation（違反した荷物の割合、0-1）をペナルティとして使う
    （Phase 0で実測: 既存cog proxyがr=-0.001だったのに対し、この指標はfrac_violationが
    実cog/stability双方とマイナス方向に強く相関、r=-0.81/-0.73）。
    """
    return (float(fill_score)
            + np_weight * float(num_placed) * 100.0
            - eq_weight * float(eq_frac_violation) * 100.0)


def run_selfplay_episode(task_config: dict, ckpt_path: str, k: int,
                          np_weight: float, eq_weight: float, temperature: float,
                          policy_soft: float = 4.5, policy_hard: float = 5.5) -> dict | None:
    """1エピソードを現在方策（サンプリング）で実行し、軌跡+報酬を返す。例外時は None。

    ワーカープロセス側で呼ばれる想定（picklableな引数のみ、モデルはckpt_pathから都度読み込む）。
    """
    try:
        from agents.heuristic.agent import Agent
        from agents.heuristic.packing_core.heightmap import decide_candidates
        from agents.heuristic.packing_core.rl_features import (
            candidate_batch_features, global_context_vector,
        )
        from agents.heuristic.packing_core.rl_model import load_policy
        from agents.heuristic.packing_core.state import build_state, make_action
        from scripts.bench_run import _static_equilibrium_proxy
        from src.ground_handling.env import GroundHandlingEnv

        model = load_policy(ckpt_path, device="cpu")
        agent = Agent(module_path="agents/heuristic/")
        env = GroundHandlingEnv(config=task_config, verbose=False, render_mode=None)
        env.reset_settings()
        init_states = env.get_init_states()
        agent.get_init_states(init_states)

        if task_config["agent"].get("optimize"):
            item_list = env.get_info_for_optimization()
            order = agent.optimize(item_list)
            if not env.set_item_order(order):
                env.close()
                return None

        env.reset_item_stream()
        obs, info = env.reset(seed=42)
        terminated = truncated = False

        trajectory: list[dict] = []
        while not terminated and not truncated:
            init = {"optimize": obs.get("optimize", agent.optimize_enabled),
                    "lookahead_k": obs.get("lookahead_k", agent.lookahead_k),
                    "container_list": obs.get("container_list", agent.container_list)}
            state = build_state(obs, init, compute_ems=False)
            budget = _StepBudget(t0=time.monotonic(), soft=policy_soft, hard=policy_hard)
            candidates, models = decide_candidates(state, budget, k=k)
            if not candidates:
                break

            cand_feats = candidate_batch_features(candidates, models, state)
            ctx_feat = global_context_vector(state)
            with torch.no_grad():
                scores = model(torch.from_numpy(cand_feats), torch.from_numpy(ctx_feat))
                probs = torch.softmax(scores / temperature, dim=0)
                action_idx = int(torch.multinomial(probs, 1).item())

            trajectory.append({"cand_feats": cand_feats, "ctx_feat": ctx_feat,
                                "action": action_idx})
            best = candidates[action_idx]
            action = make_action(item_idx=best.item_idx, container_idx=best.container_idx,
                                  pos_rel=best.pos_rel, orientation=best.orientation)
            obs, reward, terminated, truncated, info = env.step(action)

        fill_score, out_items = env.evaluator.calculate_fill_rate(env.container_manager.containers)
        num_total = env.num_total_items
        num_placed = (sum(len(c.packed_items) for c in env.container_manager.containers)
                      / max(1, num_total))
        eq = _static_equilibrium_proxy(env)
        env.close()

        total_reward = compute_reward(fill_score, num_placed, eq.get("frac_violation", 0.0),
                                       np_weight, eq_weight)
        return {"trajectory": trajectory, "reward": total_reward, "fill_score": float(fill_score),
                "num_placed": float(num_placed),
                "eq_frac_violation": float(eq.get("frac_violation", 0.0)),
                "n_steps": len(trajectory)}
    except Exception as exc:  # ワーカー内例外はこのエピソードだけスキップする
        import traceback
        traceback.print_exc()
        return None


def _sample_tasks(rng: np.random.Generator, n: int, seed_base: int) -> list[tuple[int, int]]:
    """family1/family2を交互に、重複しないseedで抽出する。"""
    from scripts.gen_tasks import _FAMILY_TEMPLATES  # noqa: F401  (存在確認のみ)
    out = []
    for i in range(n):
        family = 1 if i % 2 == 0 else 2
        seed = seed_base + int(rng.integers(0, 10_000_000))
        out.append((family, seed))
    return out


def _policy_gradient_update(model, optimizer, results: list[dict], baselines: list[float],
                             entropy_coef: float) -> dict:
    """収集済み軌跡群から方策勾配を計算し1回optimizer.step()する。REINFORCE、報酬は
    エピソード終端のみ（`compute_reward`）で全ステップに一律ブロードキャストする
    （fill/np/equilibriumはいずれもエピソード完了後にしか確定しないため妥当）。

    `baselines`は`results`と1対1対応（2026-08-12改訂: 家族別の移動平均を使う。
    family1/family2で到達可能なfill/np水準が体系的に異なるため、共通の1本の
    baselineだと「家族間の差」がadvantageに漏れ込み勾配のノイズ源になっていた）。
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_steps = sum(len(r["trajectory"]) for r in results) or 1
    total_pg_loss = 0.0
    total_entropy = 0.0
    for r, baseline in zip(results, baselines):
        advantage = r["reward"] - baseline
        for step in r["trajectory"]:
            cand_feats = torch.from_numpy(step["cand_feats"])
            ctx_feat = torch.from_numpy(step["ctx_feat"])
            scores = model(cand_feats, ctx_feat)
            log_probs = F.log_softmax(scores, dim=0)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum()
            pg_loss = -log_probs[step["action"]] * advantage
            loss = (pg_loss - entropy_coef * entropy) / total_steps
            loss.backward()
            total_pg_loss += float(pg_loss.item())
            total_entropy += float(entropy.item())
    optimizer.step()
    return {"pg_loss": total_pg_loss / total_steps, "entropy": total_entropy / total_steps,
            "total_steps": total_steps}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-ckpt", type=str, required=True,
                         help="Phase 2a模倣学習のstate_dictチェックポイント")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--episodes-per-iter", type=int, default=24)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--k", type=int, default=64)
    # 2026-08-12: 初回本番実行（3e-4, entropy_coef=0.01, 単一baseline）はiter 15〜600の
    # eval reward前半/後半平均が59.84/59.84と完全に横ばいだった（ユーザー指摘で発覚）。
    # 家族別baseline（本文参照）に加え、学習率を引き上げて動きを出やすくする。まず小規模
    # diagnostic（--iterations 150程度）でeval reward trendが動くか確認してから本番再開すること。
    parser.add_argument("--lr", type=float, default=1e-3)
    # fill_score は0-100スケール。num_placed(0-1)は*100で同スケールに揃えた上でnp_weight倍、
    # eq_frac_violation(0-1、通常ほぼ0)も同様。np_weight/eq_weightは「fill_score基準の相対重み」
    # であり、30/50のような大きい値を入れると事実上その項だけの最適化になってしまう
    # （2026-08-11のsmoke testで実際に踏んだ失敗、reward全体の97%がnp項という結果だった）。
    parser.add_argument("--np-weight", type=float, default=0.5)
    parser.add_argument("--eq-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0,
                         help="自己対戦サンプリング時のsoftmax温度。>1で探索を強める。")
    parser.add_argument("--entropy-coef", type=float, default=0.003)
    parser.add_argument("--baseline-momentum", type=float, default=0.9)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-n", type=int, default=6)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()

    from agents.heuristic.packing_core.rl_model import RLPolicyNet
    from scripts.gen_tasks import gen_task

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    writer = (SummaryWriter(log_dir=os.path.join(out_dir, "tensorboard"))
              if SummaryWriter is not None else _NullWriter())

    model = RLPolicyNet()
    model.load_state_dict(torch.load(args.init_ckpt, map_location="cpu", weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    tmp_ckpt = os.path.join(out_dir, "current.pt")
    torch.save(model.state_dict(), tmp_ckpt)

    baselines_by_family: dict[int, float] = {}  # 2026-08-12改訂: 家族別移動平均（本文参照）
    best_eval_reward = -1e18
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for it in range(args.iterations):
            tasks = _sample_tasks(rng, args.episodes_per_iter, args.seed + it * 1000)
            futures = [
                pool.submit(run_selfplay_episode, gen_task(fam, seed)["000"], tmp_ckpt,
                            args.k, args.np_weight, args.eq_weight, args.temperature)
                for fam, seed in tasks
            ]
            raw_results = [f.result() for f in futures]
            results, families = [], []
            for (fam, _seed), r in zip(tasks, raw_results):
                if r is not None and r["trajectory"]:
                    results.append(r)
                    families.append(fam)
            if not results:
                print(f"iter={it:05d} no valid episodes, skipping")
                continue

            rewards = [r["reward"] for r in results]
            mean_reward = float(np.mean(rewards))
            for fam in set(families):
                fam_rewards = [r["reward"] for r, f in zip(results, families) if f == fam]
                fam_mean = float(np.mean(fam_rewards))
                if fam not in baselines_by_family:
                    baselines_by_family[fam] = fam_mean
                else:
                    baselines_by_family[fam] = (args.baseline_momentum * baselines_by_family[fam]
                                                 + (1 - args.baseline_momentum) * fam_mean)
            baselines = [baselines_by_family[f] for f in families]
            baseline = float(np.mean(list(baselines_by_family.values())))  # ログ表示用

            stats = _policy_gradient_update(model, optimizer, results, baselines, args.entropy_coef)
            torch.save(model.state_dict(), tmp_ckpt)

            elapsed = time.time() - t_start
            fill_mean = float(np.mean([r["fill_score"] for r in results]))
            np_mean = float(np.mean([r["num_placed"] for r in results]))
            eqv_mean = float(np.mean([r["eq_frac_violation"] for r in results]))
            print(f"iter={it:05d} n_ep={len(results)} elapsed={elapsed/60:.1f}min "
                  f"reward={mean_reward:.2f} baseline={baseline:.2f} "
                  f"fill={fill_mean:.2f} np={np_mean:.3f} eq_viol={eqv_mean:.3f} "
                  f"pg_loss={stats['pg_loss']:.4f} entropy={stats['entropy']:.3f}")
            writer.add_scalar("selfplay/reward_mean", mean_reward, it)
            writer.add_scalar("selfplay/baseline", baseline, it)
            writer.add_scalar("selfplay/fill_mean", fill_mean, it)
            writer.add_scalar("selfplay/np_mean", np_mean, it)
            writer.add_scalar("selfplay/eq_violation_mean", eqv_mean, it)
            writer.add_scalar("train/pg_loss", stats["pg_loss"], it)
            writer.add_scalar("train/entropy", stats["entropy"], it)

            if (it + 1) % args.checkpoint_every == 0:
                ckpt_path = os.path.join(out_dir, f"model_iter{it+1:05d}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"  checkpoint saved: {ckpt_path}")

            if (it + 1) % args.eval_every == 0:
                eval_tasks = _sample_tasks(np.random.default_rng(args.seed - 1),
                                            args.eval_n, args.seed - 1)
                eval_futures = [
                    pool.submit(run_selfplay_episode, gen_task(fam, seed)["000"], tmp_ckpt,
                                args.k, args.np_weight, args.eq_weight, 1e-6)  # ほぼargmax相当
                    for fam, seed in eval_tasks
                ]
                eval_results = [f.result() for f in eval_futures]
                eval_results = [r for r in eval_results if r is not None]
                if eval_results:
                    eval_reward = float(np.mean([r["reward"] for r in eval_results]))
                    eval_fill = float(np.mean([r["fill_score"] for r in eval_results]))
                    eval_np = float(np.mean([r["num_placed"] for r in eval_results]))
                    print(f"  [eval] n={len(eval_results)} reward={eval_reward:.2f} "
                          f"fill={eval_fill:.2f} np={eval_np:.3f}")
                    writer.add_scalar("eval/reward", eval_reward, it)
                    writer.add_scalar("eval/fill", eval_fill, it)
                    writer.add_scalar("eval/np", eval_np, it)
                    if eval_reward > best_eval_reward:
                        best_eval_reward = eval_reward
                        torch.save(model.state_dict(), os.path.join(out_dir, "model_best.pt"))

    torch.save(model.state_dict(), os.path.join(out_dir, "model_final.pt"))
    writer.close()
    print(f"\ndone. final checkpoint: {os.path.join(out_dir, 'model_final.pt')}")
    print(f"best eval checkpoint: {os.path.join(out_dir, 'model_best.pt')} "
          f"(reward={best_eval_reward:.2f})")


if __name__ == "__main__":
    main()
