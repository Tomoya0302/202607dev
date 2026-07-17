"""Tiny GPU training loop with TensorBoard/checkpoint output.

This is infrastructure validation, not a solution algorithm for the challenge.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run scripts/check_gpu.py first.")

    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("artifacts") / "smoke" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    model = nn.Sequential(
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    writer = SummaryWriter(log_dir=output_dir / "tensorboard")

    model.train()
    for step in range(200):
        features = torch.randn(4096, 64, device=device)
        targets = (
            0.4 * features[:, 0]
            - 0.2 * features[:, 1]
            + 0.1 * features[:, 2].square()
        ).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(features)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        writer.add_scalar("train/loss", loss.item(), step)
        if step % 25 == 0 or step == 199:
            print(f"step={step:03d} loss={loss.item():.6f}")

    checkpoint = output_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "steps": 200,
        },
        checkpoint,
    )
    writer.close()
    print(f"Checkpoint : {checkpoint}")
    print(f"TensorBoard: {output_dir / 'tensorboard'}")


if __name__ == "__main__":
    main()
