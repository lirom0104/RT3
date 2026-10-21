#!/usr/bin/env python3
"""Train DA3 Giant thermal camera tokens + LoRA; supports torchrun and resume."""

import argparse
from contextlib import nullcontext
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from depth_anything_3.thermal.adapter import (
    AdapterConfig, adapter_state_dict, install_adapter, load_adapter, load_giant,
)
from depth_anything_3.thermal.hybrid_data import SEARDataset, build_manifest
from depth_anything_3.thermal.hybrid_loss import geometry_loss, frozen_adapter_forward, confidence_preservation_loss, rgb_preservation_loss
from depth_anything_3.thermal.preservation import frozen_base_forward
from depth_anything_3.thermal.resume import map_resume_batch
from depth_anything_3.utils.logger import logger


class RepairedSEARDataset(SEARDataset):
    """Keep the supplied field of view, including supervised/unsupervised sky."""
    def __getitem__(self, index):
        scene_idx, _ = self.items[index]
        frame = self.frames[scene_idx][0]["rgb"]
        height, width = frame["h"], frame["w"]
        if height % 14 or width % 14:
            raise ValueError(f"Processed resolution must be divisible by 14: {height}x{width}")
        self.height, self.width = height, width
        return super().__getitem__(index)


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def atomic_save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/thermal_giant_hybrid_20261011.json"))
    parser.add_argument("--output")
    parser.add_argument("--checkpoint-dir", help="Weight directory; defaults to <repository>/checkpoints/<output directory name>")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--allow-world-size-change", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--views", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--accumulation", type=int)
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    initial, _ = parser.parse_known_args()
    config = json.loads(Path(initial.config).read_text())
    parser.set_defaults(**config)
    args = parser.parse_args()
    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(Path(__file__).resolve().parents[1] / "checkpoints" / Path(args.output).name)
    return args


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def forward(model, batch):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(batch["images"], thermal_mask=batch["thermal_mask"],
                     ref_view_strategy="first", use_ray_pose=False, infer_gs=False)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    sums, counts, rows = {}, {}, []
    for batch in loader:
        batch = move_batch(batch, device)
        _, metrics = geometry_loss(forward(model, batch), batch)
        mode = batch["mode"][0]
        values = {k: float(v) for k, v in metrics.items()}
        rows.append({"scene": batch["scene"][0], "mode": mode, **values})
        for group in ("all", mode):
            counts[group] = counts.get(group, 0) + 1
            sums.setdefault(group, {k: 0.0 for k in values})
            for k, v in values.items():
                sums[group][k] += v
    model.train()
    return {group: {k: v / counts[group] for k, v in totals.items()} for group, totals in sums.items()}, rows


def main():
    args = arguments()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Giant training requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl", device_id=device)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    logger.level = 1
    output = Path(args.output).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    if rank == 0:
        if (checkpoint_dir / "last.pt").exists() and not args.resume:
            raise FileExistsError(f"Checkpoint already exists in {checkpoint_dir}; use --resume or a new checkpoint directory")
        output.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    manifest = build_manifest(args.data_root, args.official_split, args.difficult_scenes)
    for group in ("train", "eval", "held_out", "excluded"):
        for scene in manifest[group]:
            scene["transforms_sha256"] = hashlib.sha256(Path(scene["transforms"]).read_bytes()).hexdigest()
    manifest_digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    settings = vars(args) | {"world_size": world_size, "torch": str(torch.__version__), "manifest_sha256": manifest_digest}
    if rank == 0:
        if not args.resume:
            atomic_json(output / "run_config.json", settings)
            atomic_json(output / "data_manifest.json", manifest)
        print(json.dumps({"event": "data", "scenes": {k: len(manifest[k]) for k in ("train", "eval", "held_out", "excluded")},
                          "train_pairs": sum(s["pairs"] for s in manifest["train"]), "world_size": world_size}), flush=True)
    train_data = RepairedSEARDataset(manifest["train"], args.views, args.height, args.width, True, args.seed)
    val_data = RepairedSEARDataset(manifest["eval"], args.views, args.height, args.width, False, args.seed)
    sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    train_loader = DataLoader(train_data, batch_size=1, sampler=sampler, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=1, num_workers=args.workers, pin_memory=True)
    model = load_giant(args.base_checkpoint)
    config = AdapterConfig(args.rank, args.alpha, args.dropout, args.start_block, True)
    install_adapter(model, config)
    load_adapter(model, args.initialize_adapter)
    model.to(device)
    teacher_state = torch.load(args.initialize_adapter, map_location='cpu', weights_only=True)
    teacher_weights = {n:v.to(device) for n,v in teacher_state['adapter'].items()}
    del teacher_state
    settings['initial_adapter_sha256'] = hashlib.sha256(Path(args.initialize_adapter).read_bytes()).hexdigest()
    if rank == 0: atomic_json(output / 'run_config.json', settings)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / args.accumulation)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = max(1, round(total_steps * args.warmup_ratio))
    # Constant with warmup, matching SEAR's optimizer schedule.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warmup_steps))
    step, start_epoch, start_batch, best, stale = 0, 0, 0, float("inf"), 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        old_settings = state["settings"]
        for key in ("views", "height", "width", "manifest_sha256", "base_checkpoint", "epochs",
                    "learning_rate", "weight_decay", "warmup_ratio", "rank", "alpha", "dropout", "start_block", "seed",
                    "teacher_weight", "teacher_valid_weight", "initialize_adapter", "confidence_weight"):
            if old_settings[key] != settings[key]:
                raise ValueError(f"Resume setting changed: {key}")
        load_adapter(model, state)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, start_epoch, start_batch = state["step"], state["epoch"], state["next_batch"]
        start_batch = map_resume_batch(
            old_settings, settings, start_batch, len(train_data), args.allow_world_size_change
        )
        if rank == 0:
            print(json.dumps({"event": "resume", "step": step, "epoch": start_epoch + 1,
                              "old_world_size": old_settings["world_size"], "world_size": world_size,
                              "old_next_batch": state["next_batch"], "next_batch": start_batch,
                              "effective_views_per_step": world_size * args.accumulation * args.views}), flush=True)
        best, stale = state["best"], state["stale"]
        rng = state["rng"][rank]
        torch.set_rng_state(rng["cpu"])
        torch.cuda.set_rng_state(rng["cuda"], device)
        del state
        if rank == 0:
            atomic_json(output / "resume_config.json", settings)
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False) if world_size > 1 else model
    model.train()
    if rank == 0:
        print(json.dumps({"event": "model_ready", "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                          "steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "warmup_steps": warmup_steps}), flush=True)

    def record(value):
        if rank == 0:
            value = {"time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), **value}
            if isinstance(value.get("best"), float) and not math.isfinite(value["best"]):
                value["best"] = None
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(value) + "\n")
            print(json.dumps(value), flush=True)

    def save(epoch, next_batch, is_best=False):
        rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device)}
        rng_states = [None] * world_size
        if world_size > 1:
            dist.all_gather_object(rng_states, rng)
        else:
            rng_states[0] = rng
        if rank == 0:
            adapter = {"format_version": 1, "adapter_config": model.thermal_adapter_config,
                       "adapter": adapter_state_dict(model), "base_checkpoint": model.base_checkpoint,
                       "step": step, "epoch": epoch, "settings": settings}
            state = adapter | {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                               "next_batch": next_batch, "best": best, "stale": stale, "rng": rng_states}
            atomic_save(checkpoint_dir / "last.pt", state)
            atomic_save(checkpoint_dir / "adapter_last.pt", adapter)
            if is_best:
                atomic_save(checkpoint_dir / "adapter_best.pt", adapter)
        if world_size > 1:
            dist.barrier()

    def validation(epoch, baseline=False):
        nonlocal best, stale
        if rank == 0:
            metrics, rows = evaluate(model, val_loader, device)
            # Select by depth accuracy, not camera loss, for the hybrid run.
            score = (metrics["rgb"]["abs_rel"] + metrics["mixed"]["abs_rel"] + metrics["thermal"]["abs_rel"]) / 3
            improved = score < best
            if improved:
                best, stale = score, 0
            elif not baseline:
                stale += 1
            atomic_json(output / ("baseline_eval.json" if baseline else "latest_eval.json"), {"step": step, "epoch": epoch, "metrics": metrics, "scenes": rows})
            if improved:
                atomic_json(output / "best_eval.json", {"step": step, "epoch": epoch, "metrics": metrics, "scenes": rows})
            record({"event": "baseline" if baseline else "validation", "step": step, "epoch": epoch,
                    "metrics": metrics, "selection_score": score, "best": best, "stale": stale})
            info = [best, stale, improved]
        else:
            info = [None, None, None]
        if world_size > 1:
            dist.broadcast_object_list(info, src=0, device=device)
        best, stale, improved = info
        return improved

    if args.eval_only:
        validation(start_epoch, baseline=True)
        if world_size > 1:
            dist.destroy_process_group()
        return
    if not args.resume and not args.skip_initial_eval:
        validation(0, baseline=True)
        save(0, 0, is_best=True)
    started = time.monotonic()
    stop = False
    for epoch in range(start_epoch, args.epochs):
        train_data.epoch = epoch
        sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        pending = {}
        for batch_idx, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_idx < start_batch:
                continue
            group_start = (batch_idx // args.accumulation) * args.accumulation
            group_size = min(args.accumulation, len(train_loader) - group_start)
            sync = (batch_idx + 1) % args.accumulation == 0 or batch_idx + 1 == len(train_loader)
            context = wrapped.no_sync() if world_size > 1 and not sync else nullcontext()
            with context:
                batch = move_batch(batch, device)
                teacher_depth = None
                with frozen_adapter_forward(wrapped, teacher_weights) as teacher, torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    teacher_output = teacher(batch['images'], thermal_mask=batch['thermal_mask'], ref_view_strategy='first', use_ray_pose=False, infer_gs=False)
                    teacher_conf = teacher_output['depth_conf'].detach().float()
                    del teacher_output
                if args.teacher_weight > 0 and (~batch["thermal_mask"]).any():
                    with frozen_base_forward(wrapped) as teacher, torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        teacher_output = teacher(batch["images"], thermal_mask=None,
                            ref_view_strategy="first", use_ray_pose=False, infer_gs=False)
                        teacher_depth = teacher_output["depth"].detach().float()
                        del teacher_output
                prediction = forward(wrapped, batch)
                loss, metrics = geometry_loss(prediction, batch)
                if teacher_depth is not None:
                    preservation = rgb_preservation_loss(prediction, teacher_depth,
                        batch["valid"], batch["thermal_mask"], args.teacher_valid_weight)
                    loss = loss + args.teacher_weight * preservation
                    metrics["rgb_preservation"] = preservation.detach()
                    metrics["loss"] = loss.detach()
                else:
                    metrics["rgb_preservation"] = loss.detach() * 0
                confidence_loss = confidence_preservation_loss(prediction, teacher_conf)
                loss = loss + args.confidence_weight * confidence_loss
                metrics['confidence_preservation'] = confidence_loss.detach()
                metrics['loss'] = loss.detach()
                del prediction, teacher_depth, teacher_conf
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at {step}, scene {batch['scene']}")
                (loss / group_size).backward()
            for key, value in metrics.items():
                pending[key] = pending.get(key, torch.zeros_like(value)) + value / group_size
            if not sync:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
            if step == 0 and rank == 0:
                gradients = {n: float(p.grad.norm()) if p.grad is not None else None for n, p in model.named_parameters() if p.requires_grad}
                atomic_json(output / "first_step_gradients.json", gradients)
                record({"event": "gradients", "parameters_with_grad": sum(v is not None for v in gradients.values()),
                        "adapter_tensors": len(gradients), "nonzero_gradients": sum(v is not None and v > 0 for v in gradients.values())})
            learning_rate = scheduler.get_last_lr()[0]
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if world_size > 1:
                for value in pending.values():
                    dist.all_reduce(value)
                    value /= world_size
            if step % args.log_every == 0 or step <= 2:
                record({"event": "train", "epoch": epoch + 1, "step": step, "batch": batch_idx + 1,
                        **{k: float(v) for k, v in pending.items()}, "grad_norm": float(grad_norm),
                        "learning_rate": learning_rate, "elapsed_seconds": round(time.monotonic() - started, 1),
                        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30})
            pending = {}
            if step % args.save_every == 0:
                save(epoch, batch_idx + 1)
            if args.max_steps and step >= args.max_steps:
                save(epoch, batch_idx + 1)
                stop = True
                break
        if stop:
            break
        improved = validation(epoch + 1)
        save(epoch + 1, 0, is_best=improved)
        if stale >= args.patience:
            record({"event": "early_stop", "epoch": epoch + 1, "step": step, "best": best})
            break
    record({"event": "finished", "step": step, "best": best, "max_steps_stop": stop})
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
