#!/usr/bin/env python3
"""Merge completed paired evaluation shards with duplicate/missing-case checks."""

import argparse
import hashlib
import html
import json
from pathlib import Path

import torch
from evaluate_rgbt_joint import aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--expected-cases", type=int, default=766)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output)
    rows, configurations = [], []
    for shard in range(args.shards):
        if not (output / f"index_{shard}.html").exists():
            raise RuntimeError(f"Shard {shard} did not finish")
        rows += [
            json.loads(line)
            for line in (output / f"rows_{shard}.jsonl").read_text().splitlines()
        ]
        configurations.append(json.loads((output / f"config_{shard}.json").read_text()))
    # torch.save embeds the archive filename; equal tensors in refiner_0/1.pt
    # therefore need not have equal file hashes. Check integrity then contents.
    reference = None
    for shard, config in enumerate(configurations):
        snapshot = output / f"refiner_{shard}.pt"
        if (
            hashlib.sha256(snapshot.read_bytes()).hexdigest()
            != config["evaluated_refiner_sha256"]
        ):
            raise ValueError(f"Evaluated checkpoint changed in shard {shard}")
        state = torch.load(snapshot, map_location="cpu", weights_only=False)
        if reference is not None:
            for key in ["format", "refiner_config", "step"]:
                if state[key] != reference[key]:
                    raise ValueError(f"Checkpoint {key} differs between shards")
            if state["state_dict"].keys() != reference["state_dict"].keys() or any(
                not torch.equal(value, reference["state_dict"][name])
                for name, value in state["state_dict"].items()
            ):
                raise ValueError(
                    "Different learned weights were evaluated across shards"
                )
            for key in [
                "teacher",
                "data",
                "split",
                "views",
                "targets",
                "shards",
                "include_base",
            ]:
                if config.get(key) != configurations[0].get(key):
                    raise ValueError(f"Evaluation option {key} differs between shards")
        reference = state
    keys = [(row["scene"], row["views"], row["target"]) for row in rows]
    assert len(keys) == len(set(keys)) == args.expected_cases
    policy = json.loads(
        (root / "outputs/rgbt/joint_training_20260920/quality_policy.json").read_text()
    )
    rows.sort(key=lambda row: (row["split"], row["scene"], row["views"], row["target"]))
    result = aggregate(rows, policy)
    (output / "results_all.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "rows_all.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    cards = []
    for row in rows:
        title = f"{row['scene']} / {row['views']} views / {row['target']}"
        filename = f"{row['views']}_{row['target']}.jpg"
        cards.append(
            f'<article><h3>{html.escape(title)}</h3><img loading="lazy" src="{html.escape(filename)}"></article>'
        )
        if "base" in row:
            cards.append(
                f"<p>Target / base reconstruction / shared refiner</p>"
                f'<img loading="lazy" src="base_{html.escape(filename)}">'
            )
    (output / "index_all.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Shared refinement evaluation</title><style>body{max-width:1100px;margin:30px auto;font:16px system-ui}img{max-width:100%}</style><h1>Target / original model / shared refiner</h1>'
        + "".join(cards)
    )
    print(
        json.dumps(
            {key: result[key] for key in ["cases", "scenes", "passed", "summary_delta"]}
        )
    )


if __name__ == "__main__":
    main()
