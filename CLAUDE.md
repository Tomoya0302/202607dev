# CLAUDE.md

## Project

NEDO Challenge 2026向けのGround Handlingシミュレーター開発リポジトリ。

```text
nedo-challenge-2026/
├── docs/
├── simulator/
├── .vscode/
└── .devcontainer/
```

主な実装対象は `simulator/`。

## Environments

* **GUI確認**: WSL 2 / Ubuntu 24.04 / Python 3.11 venv
* **評価・実行**: 提供されたCPU用Dockerコンテナ
* **GPU学習**: `docker-compose.gpu.yml` のCUDAコンテナ
* **エディタ**: Windows版VS Code + WSL / Dev Containers

## Common Commands

### GUI確認

```bash
cd simulator
source .venv/bin/activate
python -m scripts.run_test --render-mode human --verbose True
```

### CPU評価

```bash
cd simulator
docker compose up -d --build
docker compose exec dev1 python -m scripts.run_test --verbose True
```

### GPU確認・学習

```bash
cd simulator
docker compose -f docker-compose.gpu.yml run --rm train \
  python -m scripts.check_gpu

docker compose -f docker-compose.gpu.yml run --rm train \
  python -m scripts.train_smoke
```

## Development Rules

* 変更前に関連コードと設定を確認する。
* 変更は小さく保ち、既存インターフェースを不用意に変えない。
* Python 3.11と3.12の両方で解釈可能な構文を使う。
* 学習時はGPUを利用してよいが、提出・評価時の推論はCPUで動作可能にする。
* 最終確認は必ず提供されたCPU評価コンテナで行う。
* モデル、ログ、生成物は `simulator/artifacts/` または `simulator/results/` に保存する。
* `.venv/`、`__pycache__/`、生成ログ、`*:Zone.Identifier` はコミットしない。
* 不明な仕様は推測で大きく変更せず、既存コード・README・docsを優先する。

## Code Quality

* 読みやすい命名、型ヒント、短い関数を優先する。
* 重要なロジックには簡潔なコメントを付ける。
* 修正後は可能な範囲で構文確認・テスト・実行確認を行う。
* 作業完了時に、変更ファイル、変更内容、実行した確認コマンドを簡潔に報告する。
