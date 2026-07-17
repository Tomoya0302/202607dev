# GPU学習環境（WSL 2 + Docker Engine + RTX 4060 Ti）

この追加構成は、提供されたCPU評価環境を変更せず、GPU学習だけを別コンテナで行うためのものです。

- 評価: `dockerfiles/Dockerfile` / CPU版PyTorch
- 学習: `dockerfiles/Dockerfile.train` / CUDA版PyTorch
- GUI: WSLのPython 3.11 venv

## 0. ファイル配置

このZIPの中身を `simulator/` の直下へコピーします。既存の評価用ファイルは上書きしません。

```text
simulator/
├── dockerfiles/
│   ├── Dockerfile
│   └── Dockerfile.train
├── requirements/
│   └── train.txt
├── scripts/
│   ├── check_gpu.py
│   └── train_smoke.py
├── docker-compose.yml
├── docker-compose.gpu.yml
└── GPU_TRAINING.md
```

## 1. Windows側ドライバ確認

PowerShellで実行します。

```powershell
nvidia-smi
```

RTX 4060 Tiが表示されない場合は、Windows用NVIDIAドライバを更新します。
WSL内にLinux版NVIDIAディスプレイドライバを入れてはいけません。

## 2. WSL側からGPUを確認

```bash
/usr/lib/wsl/lib/nvidia-smi
```

通常の `nvidia-smi` で見つからない場合は、次を追加できます。

```bash
echo 'export PATH=/usr/lib/wsl/lib:$PATH' >> ~/.bashrc
source ~/.bashrc
nvidia-smi
```

## 3. NVIDIA Container ToolkitをWSL Ubuntuへ導入

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## 4. DockerからGPUが見えるか確認

```bash
docker run --rm --gpus all \
  nvidia/cuda:12.8.1-base-ubuntu24.04 \
  nvidia-smi
```

ここでRTX 4060 Tiが表示されてから、学習イメージをビルドします。

## 5. 学習コンテナをビルド

```bash
cd ~/nedo-challenge-2026/simulator
export LOCAL_UID="$(id -u)"
export LOCAL_GID="$(id -g)"
mkdir -p artifacts

docker compose -f docker-compose.gpu.yml build train
```

CUDA開発イメージとPyBulletのビルドを含むため、初回イメージは大きくなります。

## 6. CUDA/PyTorch診断

```bash
docker compose -f docker-compose.gpu.yml run --rm train \
  python -m scripts.check_gpu
```

最後に以下が出れば成功です。

```text
CUDA smoke test: PASS
```

## 7. GPU学習のスモークテスト

```bash
docker compose -f docker-compose.gpu.yml run --rm train \
  python -m scripts.train_smoke
```

出力はホスト側の以下へ残ります。

```text
artifacts/smoke/<日時>/model.pt
artifacts/smoke/<日時>/tensorboard/
```

TensorBoardを起動する場合:

```bash
docker compose -f docker-compose.gpu.yml run --rm \
  -p 6006:6006 train \
  tensorboard --logdir artifacts --host 0.0.0.0 --port 6006
```

Windowsのブラウザで `http://localhost:6006` を開きます。

## 8. 対話シェル

```bash
docker compose -f docker-compose.gpu.yml run --rm train bash
```

コンテナ内ではプロジェクトが `/workspace` にマウントされています。

## 9. 評価用CPU環境の確認

GPU学習後も、提出前は必ず従来の評価コンテナで確認します。

```bash
docker compose up -d --build
docker compose exec dev1 bash
python -m scripts.run_test --verbose True
```

GPUで学習したモデルを提出に含める場合も、モデル読み込みと推論がCPUのみで完了することを確認してください。

## 10. 運用上の原則

1. 学習コードと学習済み重みは `artifacts/` へ保存する。
2. 評価用DockerfileへCUDA依存を追加しない。
3. エージェントでは `map_location="cpu"` を指定してモデルを読み込めるようにする。
4. 提出前にCPU版PyTorchで速度・メモリ・タイムアウトを測る。
5. RTX 4060 TiのVRAMを超える場合は、batch size削減、混合精度、勾配蓄積を使う。
