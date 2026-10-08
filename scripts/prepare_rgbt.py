#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from depth_anything_3.rgbt.data import prepare_dataset

p = argparse.ArgumentParser()
p.add_argument("--root", default="/home/lf/feedforward/data")
p.add_argument("--output", default="outputs/rgbt/data_336")
p.add_argument("--height", type=int, default=252)
p.add_argument("--width", type=int, default=336)
a = p.parse_args()
prepare_dataset(a.root, a.output, (a.height, a.width))
