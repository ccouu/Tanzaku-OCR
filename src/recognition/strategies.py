"""Strategy-pattern abstractions and parameter-controlled configs for recognition training & evaluation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
import random
from typing import Any

from torch.utils.data import Sampler


@dataclass
class TrainConfig:
    """Purely parameter-driven training specification (replaces per-experiment trainer copies)."""

    stage_name: str
    base_dir: Path
    source_ckpt: Path
    out_dir: Path
    rendered_manifest: Path
    source_key: str | None = "ema"
    expected_source_sha: str | None = None
    vocab_path: Path | None = None
    plan_path: Path | None = None
    resume_ckpt: Path | None = None
    job_id: str | None = None

    # Model architecture parameters
    height_pool: str = "conv2d"
    stage_strides: tuple[tuple[int, int], ...] = ((2, 2), (2, 2), (2, 1), (2, 1), (2, 1))
    expected_h: int = 2
    bilstm_layers: int = 1
    temporal: bool = True
    orientation: bool = True
    temporal_norm: str = "batch"
    use_bilstm: bool = True

    # Freeze / unfreeze & per-module LR parameters
    frozen_modules: tuple[str, ...] = ()
    freeze_bn_affine: bool = False
    freeze_bn_buffers: bool = True
    module_lrs: dict[str, float] = field(default_factory=lambda: {
        "encoder": 2e-7,
        "projection": 1e-6,
        "condition": 1e-6,
        "temporal": 1e-6,
        "recurrent": 1e-6,
        "classifier": 1e-6,
    })
    default_lr: float = 1e-6
    weight_decay: float = 1e-4
    weight_decay_policy: str = "all"
    grad_clip_norm: float = 5.0

    # Schedule, EMA & saving parameters
    epochs: int = 5
    batch_size: int = 32
    warmup_steps: int = 200
    min_lr_ratio: float = 0.1
    ema_decay: float = 0.998
    save_every_steps: tuple[int, ...] = ()
    save_each_epoch: bool = True
    requires_fixed_plan: bool = False
    expected_fixed_batches: int | None = None
    num_workers: int = 4
    seed: int = 20261173
    use_bf16: bool = True
    verify_nonzero_cond_grad: bool = True
    smoke_only: bool = False

    def __post_init__(self):
        self.base_dir = Path(self.base_dir)
        self.source_ckpt = Path(self.source_ckpt)
        self.out_dir = Path(self.out_dir)
        self.rendered_manifest = Path(self.rendered_manifest)
        if self.vocab_path is not None:
            self.vocab_path = Path(self.vocab_path)
        if self.plan_path is not None:
            self.plan_path = Path(self.plan_path)
        if self.resume_ckpt is not None:
            self.resume_ckpt = Path(self.resume_ckpt)
        if "condition" in self.frozen_modules:
            self.verify_nonzero_cond_grad = False


def preset_module_policy(preset: str) -> dict[str, Any]:
    """Return parameter-control presets matching V68-V73 recipes."""
    presets = {
        "v72_arm_a": {
            "frozen_modules": ("encoder", "projection", "condition", "temporal"),
            "freeze_bn_affine": True,
            "module_lrs": {"recurrent": 1e-5, "classifier": 1e-6},
            "epochs": 1,
            "save_every_steps": (500, 1000, 1500, 2000),
            "save_each_epoch": False,
            "requires_fixed_plan": True,
            "expected_fixed_batches": 2000,
        },
        "v72_arm_b": {
            "frozen_modules": (),
            "freeze_bn_affine": False,
            "module_lrs": {
                "encoder": 2e-7,
                "projection": 1e-6,
                "condition": 1e-6,
                "temporal": 1e-6,
                "recurrent": 1e-5,
                "classifier": 1e-6,
            },
            "epochs": 1,
            "save_every_steps": (500, 1000, 1500, 2000),
            "save_each_epoch": False,
            "requires_fixed_plan": True,
            "expected_fixed_batches": 2000,
        },
        "v73": {
            "frozen_modules": (),
            "freeze_bn_affine": False,
            "module_lrs": {
                "encoder": 2e-7,
                "projection": 1e-6,
                "condition": 1e-6,
                "temporal": 1e-6,
                "recurrent": 1e-6,
                "classifier": 1e-6,
            },
            "epochs": 5,
            "save_every_steps": (),
            "save_each_epoch": True,
            "requires_fixed_plan": False,
            "expected_fixed_batches": None,
        },
    }
    if preset not in presets:
        raise ValueError(f"Unknown preset {preset!r}; available={sorted(presets)}")
    return presets[preset]


# ---------------------------------------------------------------------------
# Batch Sampling Strategies
# ---------------------------------------------------------------------------


class BatchSamplingStrategy(ABC):
    """Strategy that constructs a PyTorch Sampler and reports planned total steps."""

    @abstractmethod
    def make_sampler(self, records: list[dict], config: TrainConfig) -> Sampler[list[int]]:
        ...

    @abstractmethod
    def total_steps(self, sampler: Sampler[list[int]], config: TrainConfig) -> int:
        ...

    def on_epoch_start(self, sampler: Sampler[list[int]], epoch: int) -> None:
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)


class _FixedPlanSampler(Sampler[list[int]]):
    def __init__(self, batches: list[list[int]]):
        self.batches = batches

    def __iter__(self):
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)


class FixedPlanBatchStrategy(BatchSamplingStrategy):
    """Replays an exact precomputed batch plan (e.g. V64/V72 2,000x32 matched plan)."""

    def __init__(self, plan_path: Path, expected_batches: int | None = None):
        self.plan_path = Path(plan_path)
        self.expected_batches = expected_batches

    def make_sampler(self, records: list[dict], config: TrainConfig) -> Sampler[list[int]]:
        plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
        batches = plan.get("batches")
        if not isinstance(batches, list) or not batches:
            raise ValueError(f"Fixed plan {self.plan_path} has no batches")
        expected = self.expected_batches if self.expected_batches is not None else config.expected_fixed_batches
        if expected is not None and len(batches) != expected:
            raise ValueError(
                f"Fixed plan {self.plan_path} has {len(batches)} batches, expected {expected}"
            )
        n_records = len(records)
        for bi, b in enumerate(batches):
            if not b or any((not isinstance(idx, int) or idx < 0 or idx >= n_records) for idx in b):
                raise ValueError(f"Fixed plan {self.plan_path} batch {bi} contains out-of-range indices for {n_records} records")
        return _FixedPlanSampler(batches)

    def total_steps(self, sampler: Sampler[list[int]], config: TrainConfig) -> int:
        return len(sampler) * config.epochs


class _BucketEpochSampler(Sampler[list[int]]):
    def __init__(self, records: list[dict], batch_size: int = 32, seed: int = 20261173):
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.by_bucket: dict[int, list[int]] = defaultdict(list)
        for i, r in enumerate(records):
            self.by_bucket[int(r.get("bucket", r.get("canvas", 512)))].append(i)
        self._num_batches = sum((len(idxs) + batch_size - 1) // batch_size for idxs in self.by_bucket.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch * 9973)
        batches: list[list[int]] = []
        for _, idxs in sorted(self.by_bucket.items()):
            pool = list(idxs)
            rng.shuffle(pool)
            for s in range(0, len(pool), self.batch_size):
                batches.append(pool[s : s + self.batch_size])
        rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return self._num_batches


class BucketEpochBatchStrategy(BatchSamplingStrategy):
    """Full-pass bucketed batching with drop_last=False (used in V65, V68-V71, V73)."""

    def make_sampler(self, records: list[dict], config: TrainConfig) -> Sampler[list[int]]:
        return _BucketEpochSampler(records, batch_size=config.batch_size, seed=config.seed)

    def total_steps(self, sampler: Sampler[list[int]], config: TrainConfig) -> int:
        return len(sampler) * config.epochs


# ---------------------------------------------------------------------------
# Evaluation Guard Strategies
# ---------------------------------------------------------------------------


class GuardStrategy(ABC):
    """Strategy for evaluating candidate eligibility against a source baseline vector."""

    @abstractmethod
    def check(self, cand_vec: dict[str, Any], src_vec: dict[str, Any]) -> dict[str, Any]:
        ...


class V65ShortGuardStrategy(GuardStrategy):
    """Standard V65/V72/V73 regression guards on heldout cohorts + V65 probe improvement."""

    def check(self, cand_vec: dict[str, Any], src_vec: dict[str, Any]) -> dict[str, Any]:
        guards = {
            "g1_digits_heldout_le_src_plus_001": cand_vec["digits_heldout_cer"] <= src_vec["digits_heldout_cer"] + 0.01 + 1e-12,
            "g2_price_exact_ge_src_minus_002": cand_vec["price_exact_rate"] >= src_vec["price_exact_rate"] - 0.02 - 1e-12,
            "g3_brackets_heldout_le_src_plus_001": cand_vec["brackets_heldout_cer"] <= src_vec["brackets_heldout_cer"] + 0.01 + 1e-12,
            "g4_long512_clean_le_src_plus_001": cand_vec["long512_clean_cer"] <= src_vec["long512_clean_cer"] + 0.01 + 1e-12,
            "g5_long512_noisy_le_src_plus_001": cand_vec["long512_noisy_cer"] <= src_vec["long512_noisy_cer"] + 0.01 + 1e-12,
            "g6_original_bucket_le_src_plus_001": cand_vec["original_bucket_cer"] <= src_vec["original_bucket_cer"] + 0.01 + 1e-12,
            "g7_probe_v65_macro_cer_improves": cand_vec["probe_v65_macro_cer_1_2_sml"] < src_vec["probe_v65_macro_cer_1_2_sml"] - 1e-12,
        }
        return {"passed_all": all(guards.values()), "guards": guards}


class NativeHVGuardStrategy(GuardStrategy):
    """Original V68-V71 native-vertical & horizontal 2400-sample source-relative guards."""

    def check(self, cand_vec: dict[str, Any], src_vec: dict[str, Any]) -> dict[str, Any]:
        guards = {
            "g_vert_probe_vertical_le_src_plus_001": cand_vec["vert_probe_vertical_cer"] <= src_vec["vert_probe_vertical_cer"] + 0.01 + 1e-12,
            "g_vert_probe_horiz_le_src_plus_001": cand_vec["vert_probe_horizontal_cer"] <= src_vec["vert_probe_horizontal_cer"] + 0.01 + 1e-12,
        }
        return {"passed_all": all(guards.values()), "guards": guards}


class CompositeGuardStrategy(GuardStrategy):
    """Combines multiple GuardStrategy instances with logical AND."""

    def __init__(self, *strategies: GuardStrategy):
        self.strategies = strategies

    def check(self, cand_vec: dict[str, Any], src_vec: dict[str, Any]) -> dict[str, Any]:
        merged: dict[str, bool] = {}
        for strat in self.strategies:
            merged.update(strat.check(cand_vec, src_vec)["guards"])
        return {"passed_all": all(merged.values()), "guards": merged}
