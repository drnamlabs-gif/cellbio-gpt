# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stage 0: verify the training environment (RTX 5090, thread caps, mixed precision)."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "10")
os.environ.setdefault("MKL_NUM_THREADS", "10")

import time

import torch

torch.set_num_threads(10)


def main() -> None:
    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("No CUDA device — activate .venv-gpu before running.")
        return

    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    print(f"GPU: {props.name}, {props.total_memory / 1e9:.1f} GB, CC {props.major}.{props.minor}")
    print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")

    # matmul throughput at the sizes stage-3 pretraining will use
    n = 4096
    for dtype in (torch.float32, torch.bfloat16):
        a = torch.randn(n, n, device=dev, dtype=dtype)
        b = torch.randn(n, n, device=dev, dtype=dtype)
        for _ in range(3):  # warmup
            a @ b
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        iters = 20
        for _ in range(iters):
            a @ b
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        tflops = 2 * n**3 / dt / 1e12
        print(f"{str(dtype):>15}: {n}x{n} matmul {dt * 1e3:.2f} ms -> {tflops:.1f} TFLOP/s")


if __name__ == "__main__":
    main()
