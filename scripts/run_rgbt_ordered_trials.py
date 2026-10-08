#!/usr/bin/env python3
"""Wait for complete audited caches, then run one shared-refiner training trial."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--channels", type=int, choices=[96, 192], required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    work = root / "outputs/rgbt/joint_training_20260920"
    expected = {"cache_ordered_train": 224, "cache_ordered_val": 24}
    print("WAITING for complete training/validation caches", expected, flush=True)
    deadline = time.monotonic() + 1800
    while any(
        len(list((work / name).glob("*.pt"))) != count
        for name, count in expected.items()
    ):
        if time.monotonic() > deadline:
            raise TimeoutError("Caches did not complete; training was not started")
        time.sleep(10)
    command = [
        sys.executable,
        str(root / "scripts/train_rgbt_joint.py"),
        "--output",
        str(work / args.run),
        "--device",
        args.device,
        "--train-cache",
        str(work / "cache_ordered_train"),
        "--val-cache",
        str(work / "cache_ordered_val"),
        "--steps",
        "10000",
        "--eval-every",
        "1000",
        "--lr",
        "0.0001",
        "--channels",
        str(args.channels),
        "--attention-pairs",
        "2",
        "--ordered-updates",
        "--init-refiner",
        str(work / f"ordered_init{args.channels}.pt"),
        "--distill-weight",
        "0.2",
        "--teacher-image-weight",
        "8",
        "--update-distill-weight",
        "5",
        "--robust-distill",
        "--covariance-distill",
        "--perceptual-weight",
        "0.5",
        "--distill-only",
        "--seed",
        str(20260929 if args.channels == 96 else 20260930),
    ]
    print("START", command, flush=True)
    log = work / f"{args.run}.log"
    env = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
    }
    with log.open("x") as stream:
        result = subprocess.run(
            command,
            cwd=root,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print("FINISHED", result.returncode, log, flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
