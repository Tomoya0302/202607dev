"""Fail-fast CUDA/PyTorch diagnostic for the training container."""

from __future__ import annotations

import platform
import sys
import time

import torch


def gib(value: int) -> float:
    return value / (1024**3)


def main() -> int:
    print(f"Python        : {sys.version.split()[0]}")
    print(f"Platform      : {platform.platform()}")
    print(f"PyTorch       : {torch.__version__}")
    print(f"Torch CUDA    : {torch.version.cuda}")
    print(f"cuDNN         : {torch.backends.cudnn.version()}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not visible from PyTorch.", file=sys.stderr)
        return 1

    count = torch.cuda.device_count()
    print(f"GPU count     : {count}")
    for index in range(count):
        props = torch.cuda.get_device_properties(index)
        print(
            f"GPU {index}        : {props.name} | "
            f"compute {props.major}.{props.minor} | "
            f"VRAM {gib(props.total_memory):.2f} GiB"
        )

    device = torch.device("cuda:0")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Small forward/backward workload to verify allocation, kernels, and gradients.
    x = torch.randn((2048, 2048), device=device, requires_grad=True)
    y = torch.randn((2048, 2048), device=device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    loss = (x @ y).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    print(f"Smoke loss    : {loss.item():.6f}")
    print(f"Elapsed       : {elapsed:.3f} s")
    print(f"GPU allocated : {gib(allocated):.3f} GiB")
    print(f"GPU reserved  : {gib(reserved):.3f} GiB")
    print("CUDA smoke test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
