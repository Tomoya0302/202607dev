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

## Git Workflow

### Branches

* `main`: 安定したチェックポイントを保持する。
* `develop`: 実装済み機能を統合し、評価するためのブランチ。
* `feature/<algorithm-name>`: アルゴリズム単位の実装ブランチ。

新しいアルゴリズムの実装は、`develop`からブランチを作成する。

```bash
git switch develop
git pull
git switch -c feature/<algorithm-name>
```

アルゴリズム名は、英小文字とハイフンを使用する。

```text
feature/greedy-baseline
feature/ppo-agent
feature/mcts-planner
```

実装と確認が完了した機能は、`develop`へ統合する。複数の独立したアルゴリズムを同じfeatureブランチへ混在させない。

### Commits

最初の環境構築コミットは以下。

```text
[INIT] set up development and training environments
```

以後のコミットメッセージも、次の形式に統一する。

```text
[TYPE] concise description
```

使用する主な種別:

* `[INIT]`: 初期構築
* `[FEAT]`: 新機能・アルゴリズムの追加
* `[FIX]`: 不具合修正
* `[REFACTOR]`: 挙動を変えない構造改善
* `[TEST]`: テストの追加・修正
* `[DOCS]`: 文書・コメントの更新
* `[CONFIG]`: Docker、VS Code、依存関係などの設定変更
* `[CHORE]`: その他の保守作業

例:

```text
[FEAT] add greedy placement baseline
[FIX] handle invalid action positions
[REFACTOR] separate observation preprocessing
[TEST] add placement validator tests
[DOCS] document GPU training workflow
[CONFIG] update CUDA training dependencies
```

コミットメッセージは英語で、命令形または簡潔な現在形に統一する。1コミットには原則として1つの目的だけを含める。

### Before Committing

コミット前に以下を確認する。

```bash
git status
git diff
```

可能な範囲で、関連する構文確認、テスト、実行確認を行う。

以下はコミットしない。

* `.venv/`
* `__pycache__/`
* `.env`
* 学習ログ
* 一時的な評価結果
* モデルチェックポイント
* `*:Zone.Identifier`
* 秘密情報や認証情報

Claude Codeは、ユーザーから明示的な指示がない限り、コミット、マージ、リベース、タグ作成、リモートへのpushを行わない。

## Development Rules

* 変更前に関連コードと設定を確認する。
* 変更は小さく保ち、既存インターフェースを不用意に変えない。
* Python 3.11と3.12の両方で解釈可能な構文を使う。
* 学習時はGPUを利用してよいが、提出・評価時の推論はCPUで動作可能にする。
* 最終確認は必ず提供されたCPU評価コンテナで行う。
* モデル、ログ、生成物は `simulator/artifacts/` または `simulator/results/` に保存する。
* `.venv/`、`__pycache__/`、生成ログ、`*:Zone.Identifier` はコミットしない。
* 不明な仕様は推測で大きく変更せず、既存コード、`README.md`、`docs/`を優先する。

## Code Quality

* 読みやすい命名、型ヒント、短い関数を優先する。
* 重要なロジックには簡潔なコメントを付ける。
* 修正後は可能な範囲で構文確認、テスト、実行確認を行う。
* 作業完了時に、変更ファイル、変更内容、実行した確認コマンドを簡潔に報告する。
