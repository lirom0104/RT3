"""Resume must preserve the training recipe and the protected shared weights."""

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_rgbt_joint import configure_training, configured_args

from depth_anything_3.rgbt.learned_joint import LearnedJointRefiner


class JointTrainingTests(unittest.TestCase):
    def test_resume_restores_recipe_and_honors_explicit_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            recipe = Path(tmp) / "recipe.json"
            recipe.write_text(json.dumps({"lr": 0.2, "preserve_first_update": False}))
            state = Path(tmp) / "last.pt"
            torch.save(
                {
                    "format": "da3-rgbt-learned-refiner-training",
                    "args": {
                        "lr": 0.1,
                        "preserve_first_update": True,
                        "output": "old",
                        "device": "cuda:9",
                        "smoke": True,
                    },
                },
                state,
            )
            parser = argparse.ArgumentParser()
            for key in ["output", "device"]:
                parser.add_argument("--" + key, required=True)
            for key in ["config", "resume"]:
                parser.add_argument("--" + key)
            parser.add_argument("--lr", type=float, default=1.0)
            parser.add_argument("--preserve-first-update", action="store_true")
            parser.add_argument("--smoke", action="store_true")
            args = configured_args(
                parser,
                [
                    "--config",
                    str(recipe),
                    "--resume",
                    str(state),
                    "--output",
                    "new",
                    "--device",
                    "cpu",
                    "--lr",
                    "0.01",
                ],
            )
            self.assertEqual((args.output, args.device, args.lr), ("new", "cpu", 0.01))
            self.assertTrue(args.preserve_first_update)
            self.assertFalse(args.smoke)

    def test_optimizer_updates_shared_weights_without_changing_first_step(self):
        torch.set_num_threads(2)
        model = LearnedJointRefiner(
            attention_pairs=2,
            ordered_updates=True,
            phase_modulation=True,
            preserve_first_update=True,
            shared_residual=True,
        )
        configure_training(model)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0)
        # Exercise hooks with nonzero gradients on every potentially protected row.
        sum(p.sum() for p in model.parameters() if p.requires_grad).backward()
        optimizer.step()
        after = model.state_dict()
        for name, value in before.items():
            if name.startswith(
                ("core.neck.", "core.blocks.0.", "core.blocks.2.", "core.head.0.")
            ):
                self.assertTrue(torch.equal(value, after[name]), name)
            elif name.startswith(("core.phase_embedding.", "core.phase_modulation.")):
                self.assertTrue(torch.equal(value[0], after[name][0]), name)
                self.assertFalse(torch.equal(value[1:], after[name][1:]), name)
            elif name.startswith("core.head.2."):
                protected = [3, 4, 5, 10, 14]
                self.assertTrue(
                    torch.equal(value[protected], after[name][protected]), name
                )
                self.assertFalse(torch.equal(value[0], after[name][0]), name)
        self.assertFalse(
            torch.equal(
                before["core.shared_residual.2.bias"],
                after["core.shared_residual.2.bias"],
            )
        )


if __name__ == "__main__":
    unittest.main()
