"""Unified parameter-controlled CTC trainer using Strategy pattern & src/recognition core."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from src.recognition.bn_calibration import (
    freeze_all_bn_eval,
    snapshot_bn_buffers,
    verify_bn_buffers_identical,
)
from src.recognition.canonical_ctc import canonical_ctc_inputs
from src.recognition.checkpointing import resolve_checkpoint_state, save_atomic_with_sha256
from src.recognition.data import IMAGENET_MEAN, IMAGENET_STD, collate, decode
from src.recognition.model import ARCH_SPECS, MenuRecognizer
from src.recognition.optimization import (
    build_parameter_controlled_optimizer,
    cosine_warmup_lr_scale,
    snapshot_frozen_params,
    verify_frozen_params_identical,
)
from src.recognition.strategies import (
    BatchSamplingStrategy,
    BucketEpochBatchStrategy,
    FixedPlanBatchStrategy,
    TrainConfig,
    preset_module_policy,
)


class RenderedLineDataset(Dataset):
    """Dataset reading rendered line manifests (V60-V75 format)."""

    def __init__(self, records: list[dict], images_dir: Path, char_to_id: dict[str, int]):
        self.records = records
        self.images_dir = images_dir
        self.char_to_id = char_to_id
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        arr = np.asarray(Image.open(self.images_dir / r["image_rel"]).convert("RGB"), dtype=np.float32) / 255.0
        img = (torch.from_numpy(arr).permute(2, 0, 1) - self.mean) / self.std
        text = r["canonical_text"]
        target = torch.tensor([self.char_to_id[c] for c in text], dtype=torch.long)
        return img, target, float(r.get("orientation_flag", 0)), int(r["valid_width"]), text, False


class _SkipFirstBatchesSampler(Sampler[list[int]]):
    """Skips the first `skip_batches` batches of an underlying batch sampler on resume."""

    def __init__(self, inner: Sampler[list[int]], skip_batches: int):
        self.inner = inner
        self.skip_batches = max(0, int(skip_batches))

    def __iter__(self):
        it = iter(self.inner)
        for _ in range(self.skip_batches):
            next(it, None)
        yield from it

    def __len__(self) -> int:
        return max(0, len(self.inner) - self.skip_batches)


def _update_ema(
    ema_state: dict[str, torch.Tensor],
    model: nn.Module,
    decay: float,
    trainable_param_names: set[str],
) -> None:
    """Update EMA on trainable parameters only; copy frozen params & buffers bit-identically."""
    with torch.no_grad():
        for k, v in model.state_dict().items():
            if k in trainable_param_names and v.is_floating_point():
                ema_state[k].mul_(decay).add_(v.detach(), alpha=1.0 - decay)
            else:
                ema_state[k].copy_(v.detach())


def _verify_ema_baselines(
    ema_state: dict[str, torch.Tensor],
    bn_baseline: dict[str, torch.Tensor],
    frozen_baseline: dict[str, torch.Tensor],
) -> None:
    for k, v0 in bn_baseline.items():
        if not torch.equal(ema_state[k].detach().cpu(), v0):
            raise RuntimeError(f"EMA BN buffer drifted: {k}")
    for k, v0 in frozen_baseline.items():
        if not torch.equal(ema_state[k].detach().cpu(), v0):
            raise RuntimeError(f"EMA frozen parameter drifted: {k}")


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _config_signature(config: TrainConfig, source_sha256: str, num_records: int) -> str:
    payload = {
        "stage_name": config.stage_name,
        "source_sha256": source_sha256,
        "num_records": num_records,
        "height_pool": config.height_pool,
        "stage_strides": [list(s) for s in config.stage_strides],
        "expected_h": config.expected_h,
        "bilstm_layers": config.bilstm_layers,
        "temporal": config.temporal,
        "orientation": config.orientation,
        "temporal_norm": config.temporal_norm,
        "use_bilstm": config.use_bilstm,
        "frozen_modules": list(config.frozen_modules),
        "freeze_bn_affine": config.freeze_bn_affine,
        "freeze_bn_buffers": config.freeze_bn_buffers,
        "module_lrs": dict(sorted(config.module_lrs.items())),
        "default_lr": config.default_lr,
        "weight_decay": config.weight_decay,
        "weight_decay_policy": config.weight_decay_policy,
        "grad_clip_norm": config.grad_clip_norm,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "warmup_steps": config.warmup_steps,
        "min_lr_ratio": config.min_lr_ratio,
        "ema_decay": config.ema_decay,
        "seed": config.seed,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def run_training(config: TrainConfig, batch_strategy: BatchSamplingStrategy | None = None) -> dict:
    """Execute parameter-controlled training with BN/frozen-weight bit-identity audits & full resume."""
    t0 = time.time()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = bool(config.use_bf16 and device.type == "cuda")

    vocab_path = config.vocab_path or (config.base_dir / "payload" / "vocab.json")
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    char_to_id = vocab["char_to_id"]
    num_classes = len(char_to_id) + 1

    model = MenuRecognizer(
        classes=num_classes,
        pretrained=False,
        temporal=config.temporal,
        orientation=config.orientation,
        temporal_norm=config.temporal_norm,
        use_bilstm=config.use_bilstm,
        height_pool=config.height_pool,
        stage_strides=config.stage_strides,
        expected_h=config.expected_h,
        bilstm_layers=config.bilstm_layers,
    )
    state, resolved_key, actual_source_sha, _ = resolve_checkpoint_state(
        config.source_ckpt, arm_key=config.source_key, expected_sha256=config.expected_source_sha
    )
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.train()

    bns = freeze_all_bn_eval(model) if config.freeze_bn_buffers else []
    bn_baseline = snapshot_bn_buffers(model) if config.freeze_bn_buffers else {}

    opt, opt_summary = build_parameter_controlled_optimizer(
        model,
        module_lrs=config.module_lrs,
        frozen_modules=config.frozen_modules,
        freeze_bn_affine=config.freeze_bn_affine,
        weight_decay=config.weight_decay,
        default_lr=config.default_lr,
        weight_decay_policy=config.weight_decay_policy,
    )
    trainable_param_names = {n for n, p in model.named_parameters() if p.requires_grad}
    frozen_baseline = snapshot_frozen_params(model) if (config.frozen_modules or config.freeze_bn_affine) else {}
    base_lrs = [g["lr"] for g in opt.param_groups]
    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    records = [json.loads(l) for l in config.rendered_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    if config.smoke_only:
        records = records[: min(len(records), config.batch_size)]

    cfg_sig = _config_signature(config, actual_source_sha, len(records))

    if batch_strategy is None:
        if config.requires_fixed_plan and not config.smoke_only and config.plan_path is None:
            raise ValueError(
                f"Stage {config.stage_name!r} requires a validated fixed batch plan (--plan-path)"
            )
        if config.plan_path is not None and not config.smoke_only:
            batch_strategy = FixedPlanBatchStrategy(
                config.plan_path, expected_batches=config.expected_fixed_batches
            )
        else:
            batch_strategy = BucketEpochBatchStrategy()

    sampler = batch_strategy.make_sampler(records, config)
    total_steps = 2 if config.smoke_only else batch_strategy.total_steps(sampler, config)
    save_steps = set(config.save_every_steps) if not config.smoke_only else set()

    checksums: dict[str, str] = {}
    step_logs: list[dict] = []
    global_step = 0
    start_epoch = 1
    resume_batch_cursor = 0
    cond_grad_norm_step1 = None
    first_decode_sample = None

    if config.resume_ckpt is not None and not config.smoke_only:
        ckpt = torch.load(config.resume_ckpt, map_location="cpu", weights_only=False)
        if ckpt.get("config_signature") != cfg_sig:
            raise ValueError(
                f"Resume checkpoint {config.resume_ckpt} config_signature mismatch: "
                f"{ckpt.get('config_signature')} != {cfg_sig}"
            )
        if ckpt.get("source_sha256") != actual_source_sha:
            raise ValueError(
                f"Resume checkpoint source_sha256 mismatch: {ckpt.get('source_sha256')} != {actual_source_sha}"
            )
        model.load_state_dict(ckpt["raw"], strict=True)
        ema_loaded = ckpt["ema"]
        for k in ema_state:
            ema_state[k].copy_(ema_loaded[k].to(device))
        opt.load_state_dict(ckpt["optimizer"])
        global_step = int(ckpt["step"])
        ckpt_epoch = int(ckpt["epoch"])
        ckpt_cursor = int(ckpt.get("batch_cursor", 0))
        batches_per_epoch = len(sampler)
        if ckpt_cursor >= batches_per_epoch:
            start_epoch = ckpt_epoch + 1
            resume_batch_cursor = 0
        else:
            start_epoch = ckpt_epoch
            resume_batch_cursor = ckpt_cursor
        checksums = dict(ckpt.get("checksums", {}))
        step_logs = list(ckpt.get("step_logs", []))
        cond_grad_norm_step1 = ckpt.get("cond_grad_norm_step1")
        first_decode_sample = ckpt.get("first_decode_sample")
        if "rng_state" in ckpt:
            _restore_rng_state(ckpt["rng_state"])

    ds = RenderedLineDataset(records, config.base_dir / "images", char_to_id)
    epoch_list = [1] if config.smoke_only else list(range(start_epoch, config.epochs + 1))

    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    def _save_ckpt(ckpt_name: str, epoch_idx: int, batch_cursor: int, extra: dict | None = None) -> None:
        bn_cnt = verify_bn_buffers_identical(model, bn_baseline) if config.freeze_bn_buffers else 0
        fr_cnt = verify_frozen_params_identical(model, frozen_baseline) if frozen_baseline else 0
        _verify_ema_baselines(ema_state, bn_baseline, frozen_baseline)
        payload = {
            "stage": config.stage_name,
            "step": global_step,
            "epoch": epoch_idx,
            "batch_cursor": batch_cursor,
            "source_sha256": actual_source_sha,
            "config_signature": cfg_sig,
            "raw": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "ema": {k: v.detach().cpu() for k, v in ema_state.items()},
            "optimizer": opt.state_dict(),
            "rng_state": _capture_rng_state(),
            "bn_buffer_count": bn_cnt,
            "frozen_tensor_count": fr_cnt,
            "checksums": dict(checksums),
            "step_logs": list(step_logs),
            "cond_grad_norm_step1": cond_grad_norm_step1,
            "first_decode_sample": first_decode_sample,
        }
        if extra:
            payload.update(extra)
        checksums[ckpt_name] = save_atomic_with_sha256(config.out_dir / ckpt_name, payload)
        _write_json_atomic(config.out_dir / "checksums.json", checksums)

    for epoch in epoch_list:
        model.train()
        if config.freeze_bn_buffers:
            freeze_all_bn_eval(model)
        batch_strategy.on_epoch_start(sampler, epoch)
        skip = resume_batch_cursor if epoch == start_epoch else 0
        epoch_sampler = _SkipFirstBatchesSampler(sampler, skip) if skip > 0 else sampler
        loader = DataLoader(
            ds,
            batch_sampler=epoch_sampler,
            collate_fn=collate,
            num_workers=config.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        epoch_loss_sum = 0.0
        epoch_batches = 0
        batch_cursor = skip

        for batch in loader:
            global_step += 1
            batch_cursor += 1
            lr_scale = cosine_warmup_lr_scale(
                global_step, total_steps, warmup_steps=config.warmup_steps, min_ratio=config.min_lr_ratio
            )
            for g, blr in zip(opt.param_groups, base_lrs):
                g["lr"] = blr * lr_scale

            images = batch["images"].to(device, non_blocking=True)
            ori = batch["orientation"].to(device, non_blocking=True)
            in_lens = batch["input_lengths"]
            targets = batch["targets"].to(device, non_blocking=True)
            tgt_lens = batch["lengths"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(images, ori, in_lens)

            log_probs, local_targets = canonical_ctc_inputs(logits, targets, char_to_id)
            loss = ctc(log_probs, local_targets, in_lens.to(device), tgt_lens)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {global_step}: {loss.item()}")
            loss.backward()

            if global_step == 1:
                cond_w = model.condition[0].weight
                cond_grad_norm_step1 = float(cond_w.grad[:, 256, 0].norm().item()) if cond_w.grad is not None else 0.0
                if config.verify_nonzero_cond_grad and cond_grad_norm_step1 <= 0.0:
                    raise RuntimeError(
                        f"Expected non-zero condition.0.weight[:, 256, 0] grad, got {cond_grad_norm_step1}"
                    )
                decoded = decode(logits.detach().float(), char_to_id, in_lens)
                first_decode_sample = {"truth": batch["texts"][0], "pred": decoded[0]}

            torch.nn.utils.clip_grad_norm_(trainable_params, config.grad_clip_norm)
            opt.step()
            _update_ema(ema_state, model, decay=config.ema_decay, trainable_param_names=trainable_param_names)

            loss_val = float(loss.item())
            epoch_loss_sum += loss_val
            epoch_batches += 1

            if global_step == 1 or global_step % 250 == 0 or global_step in save_steps:
                msg = {"step": global_step, "epoch": epoch, "loss": round(loss_val, 5), "lr_scale": round(lr_scale, 5)}
                step_logs.append(msg)
                print(f"[{config.stage_name}] {json.dumps(msg)}", flush=True)

            if global_step in save_steps:
                _save_ckpt(f"step_{global_step:04d}.pt", epoch, batch_cursor)

            if config.smoke_only and global_step >= 2:
                break

        if config.save_each_epoch and not config.smoke_only:
            _save_ckpt(
                f"epoch_{epoch:02d}.pt",
                epoch,
                batch_cursor,
                extra={"epoch_avg_loss": round(epoch_loss_sum / max(1, epoch_batches), 5)},
            )

        if config.smoke_only:
            break

    bn_cnt_end = verify_bn_buffers_identical(model, bn_baseline) if config.freeze_bn_buffers else 0
    fr_cnt_end = verify_frozen_params_identical(model, frozen_baseline) if frozen_baseline else 0
    _verify_ema_baselines(ema_state, bn_baseline, frozen_baseline)

    audit = {
        "stage": config.stage_name,
        "job_id": config.job_id,
        "smoke_only": config.smoke_only,
        "resumed_from": str(config.resume_ckpt) if config.resume_ckpt else None,
        "config_signature": cfg_sig,
        "device": str(device),
        "bf16": use_bf16,
        "source_ckpt": str(config.source_ckpt),
        "source_key": resolved_key,
        "source_sha256": actual_source_sha,
        "total_steps": global_step,
        "planned_total_steps": total_steps,
        "bn_module_count": len(bns),
        "bn_buffer_count": bn_cnt_end,
        "bn_bit_identical": bool(config.freeze_bn_buffers),
        "frozen_tensor_count": fr_cnt_end,
        "frozen_bit_identical": bool(frozen_baseline),
        "ema_frozen_bit_identical": True,
        "cond_grad_norm_step1": cond_grad_norm_step1,
        "first_decode_sample": first_decode_sample,
        "optimizer_summary": opt_summary,
        "elapsed_seconds": round(time.time() - t0, 2),
        "checksums": checksums,
        "step_logs": step_logs,
    }
    if not config.smoke_only:
        _write_json_atomic(config.out_dir / "checksums.json", checksums)
        _write_json_atomic(config.out_dir / "train_audit.json", audit)
        if config.job_id:
            _write_json_atomic(
                config.base_dir / "DONE.json",
                {
                    "job_id": config.job_id,
                    "stage": config.stage_name,
                    "checksums": checksums,
                    "total_steps": global_step,
                    "elapsed_seconds": audit["elapsed_seconds"],
                },
            )
    return audit


def _parse_cli() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["v72_arm_a", "v72_arm_b", "v73"], default="v73")
    p.add_argument("--arch", choices=sorted(ARCH_SPECS.keys()), default="v60_ref")
    p.add_argument("--stage-name", default=None)
    p.add_argument("--base-dir", required=True)
    p.add_argument("--source-ckpt", required=True)
    p.add_argument("--source-key", default="ema")
    p.add_argument("--expected-source-sha", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--rendered-manifest", required=True)
    p.add_argument("--vocab-path", default=None)
    p.add_argument("--plan-path", default=None)
    p.add_argument("--resume-ckpt", default=None)
    p.add_argument("--job-id", default=None)
    p.add_argument("--freeze", nargs="*", default=None, help="Override frozen module prefixes")
    p.add_argument("--freeze-bn-affine", action="store_true", default=None)
    p.add_argument("--no-freeze-bn-buffers", action="store_true")
    p.add_argument("--lr", nargs="*", default=[], help="Override module LRs as module=lr (e.g. encoder=2e-7)")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--weight-decay-policy", choices=["all", "exclude_norm_bias"], default="all")
    p.add_argument("--grad-clip-norm", type=float, default=5.0)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--ema-decay", type=float, default=0.998)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261173)
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--render-replay-count", type=int, default=None, help="If set, render replay manifest before training")
    p.add_argument("--render-workers", type=int, default=8)
    p.add_argument("--no-combine-yolo", action="store_true", help="Skip YOLO combine & 521-term check during render")
    args = p.parse_args()

    policy = preset_module_policy(args.preset)
    arch_spec = ARCH_SPECS[args.arch]
    module_lrs = dict(policy["module_lrs"])
    for item in args.lr:
        k, v = item.split("=", 1)
        module_lrs[k] = float(v)

    cfg = TrainConfig(
        stage_name=args.stage_name or args.preset,
        base_dir=Path(args.base_dir),
        source_ckpt=Path(args.source_ckpt),
        source_key=args.source_key,
        expected_source_sha=args.expected_source_sha,
        out_dir=Path(args.out_dir),
        rendered_manifest=Path(args.rendered_manifest),
        vocab_path=Path(args.vocab_path) if args.vocab_path else None,
        plan_path=Path(args.plan_path) if args.plan_path else None,
        resume_ckpt=Path(args.resume_ckpt) if args.resume_ckpt else None,
        job_id=args.job_id,
        height_pool=arch_spec["height_pool"],
        stage_strides=arch_spec["stage_strides"],
        expected_h=arch_spec["expected_h"],
        bilstm_layers=arch_spec["bilstm_layers"],
        frozen_modules=tuple(args.freeze) if args.freeze is not None else tuple(policy["frozen_modules"]),
        freeze_bn_affine=args.freeze_bn_affine if args.freeze_bn_affine is not None else bool(policy["freeze_bn_affine"]),
        freeze_bn_buffers=not args.no_freeze_bn_buffers,
        module_lrs=module_lrs,
        weight_decay=args.weight_decay,
        weight_decay_policy=args.weight_decay_policy,
        grad_clip_norm=args.grad_clip_norm,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        ema_decay=args.ema_decay,
        epochs=args.epochs if args.epochs is not None else int(policy["epochs"]),
        save_every_steps=tuple(policy["save_every_steps"]),
        save_each_epoch=bool(policy["save_each_epoch"]),
        requires_fixed_plan=bool(policy.get("requires_fixed_plan", False)),
        expected_fixed_batches=policy.get("expected_fixed_batches"),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        use_bf16=not args.no_bf16,
        smoke_only=args.smoke_only,
    )
    return cfg, args


if __name__ == "__main__":
    cfg, cli_args = _parse_cli()
    if cli_args.render_replay_count is not None and not cfg.smoke_only:
        from src.recognition.render.renderer_v74 import render_v74_replay_and_combine

        render_audit = render_v74_replay_and_combine(
            cfg.base_dir,
            workers=cli_args.render_workers,
            expected_replay_count=cli_args.render_replay_count,
            combine_with_yolo=not cli_args.no_combine_yolo,
        )
        print(f"[{cfg.stage_name}] RENDER_OK {json.dumps(render_audit)}", flush=True)
    out = run_training(cfg)
    if cfg.smoke_only:
        print(f"E2E_SMOKE_OK {json.dumps({'stage': out['stage'], 'total_steps': out['total_steps']})}", flush=True)
    else:
        print(json.dumps(out, indent=2), flush=True)
