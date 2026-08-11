"""DRL方策（候補再ランキング型）の学習コード（`docs/HANDOFF.md` DRL方策計画）。

提出zipには含めない。推論側（`agents/heuristic/packing_core/rl_model.py`/`rl_features.py`）と
モデル定義を共有し、ここには学習ループ・データ収集・自己対戦のみを置く
（`GPU_TRAINING.md` §10「学習コードと学習済みモデルは artifacts/ 配下」の運用原則に対応）。
"""
