"""Tests for refactored Strategy-pattern training, parameter-controlled LR/freezing, and checkpoint resolution."""
from pathlib import Path
import torch

from src.recognition.bn_calibration import freeze_all_bn_eval, snapshot_bn_buffers, verify_bn_buffers_identical
from src.recognition.checkpointing import resolve_checkpoint_state, save_atomic_with_sha256
from src.recognition.model import MenuRecognizer
from src.recognition.optimization import (
    build_parameter_controlled_optimizer,
    cosine_warmup_lr_scale,
    snapshot_frozen_params,
    verify_frozen_params_identical,
)
from src.recognition.strategies import (
    CompositeGuardStrategy,
    NativeHVGuardStrategy,
    TrainConfig,
    V65ShortGuardStrategy,
    preset_module_policy,
)


def _tiny_model(height_pool="conv2d"):
    return MenuRecognizer(
        classes=32,
        pretrained=False,
        temporal=True,
        orientation=True,
        temporal_norm="batch",
        use_bilstm=True,
        height_pool=height_pool,
    )


def test_height_pool_conv2d_and_mean_forward():
    x = torch.randn(2, 3, 64, 128)
    ori = torch.tensor([0.0, 1.0])
    lengths = torch.tensor([32, 24], dtype=torch.long)
    for hp in ("mean", "conv2d"):
        m = _tiny_model(height_pool=hp).eval()
        out = m(x, ori, lengths)
        assert out.shape == (32, 2, 32)


def test_parameter_controlled_freeze_and_lr_presets():
    m = _tiny_model("conv2d")
    freeze_all_bn_eval(m)
    bn_snap = snapshot_bn_buffers(m)
    assert len(bn_snap) == 147

    # Arm A: freeze encoder, projection, condition, temporal, and BN affine
    p_a = preset_module_policy("v72_arm_a")
    opt_a, sum_a = build_parameter_controlled_optimizer(
        m,
        module_lrs=p_a["module_lrs"],
        frozen_modules=p_a["frozen_modules"],
        freeze_bn_affine=p_a["freeze_bn_affine"],
    )
    frozen_snap = snapshot_frozen_params(m)
    lrs_a = {g["name"]: g["lr"] for g in opt_a.param_groups}
    assert set(lrs_a.keys()) == {"recurrent", "classifier"}
    assert lrs_a["recurrent"] == 1e-5
    assert lrs_a["classifier"] == 1e-6
    assert sum_a["frozen_tensors"] > 0

    # Dummy forward/backward/step must keep frozen params & BN buffers bit-identical
    x = torch.randn(2, 3, 64, 128)
    ori = torch.tensor([0.0, 1.0])
    lengths = torch.tensor([32, 24], dtype=torch.long)
    opt_a.zero_grad()
    loss = m(x, ori, lengths).square().mean()
    loss.backward()
    opt_a.step()
    assert verify_frozen_params_identical(m, frozen_snap) == len(frozen_snap)
    assert verify_bn_buffers_identical(m, bn_snap) == 147

    # V73: all modules train, encoder=2e-7, all others=1e-6
    p_73 = preset_module_policy("v73")
    opt_73, _ = build_parameter_controlled_optimizer(
        m,
        module_lrs=p_73["module_lrs"],
        frozen_modules=p_73["frozen_modules"],
        freeze_bn_affine=p_73["freeze_bn_affine"],
    )
    lrs_73 = {g["name"]: g["lr"] for g in opt_73.param_groups}
    assert lrs_73["encoder"] == 2e-7
    assert lrs_73["recurrent"] == 1e-6


def test_checkpoint_key_aliases_and_sha256(tmp_path: Path):
    m = _tiny_model("conv2d")
    p = tmp_path / "ckpt.pt"
    sha = save_atomic_with_sha256(p, {"model": m.state_dict(), "ema_model": m.state_dict()})
    state, resolved, got_sha, _ = resolve_checkpoint_state(p, arm_key="raw", expected_sha256=sha)
    assert resolved == "model"
    assert got_sha == sha
    assert len(state) > 10
    _, resolved_ema, _, _ = resolve_checkpoint_state(p, arm_key="ema", expected_sha256=sha)
    assert resolved_ema == "ema_model"


def test_guard_strategies_and_cosine_schedule():
    src = {
        "digits_heldout_cer": 0.05,
        "price_exact_rate": 0.70,
        "brackets_heldout_cer": 0.20,
        "long512_clean_cer": 0.14,
        "long512_noisy_cer": 0.24,
        "original_bucket_cer": 0.20,
        "probe_v65_macro_cer_1_2_sml": 0.12,
        "vert_probe_vertical_cer": 0.075,
        "vert_probe_horizontal_cer": 0.054,
    }
    good = dict(src, probe_v65_macro_cer_1_2_sml=0.08, vert_probe_vertical_cer=0.072, vert_probe_horizontal_cer=0.055)
    comp = CompositeGuardStrategy(V65ShortGuardStrategy(), NativeHVGuardStrategy())
    assert comp.check(good, src)["passed_all"] is True
    bad = dict(good, vert_probe_vertical_cer=0.095)
    assert comp.check(bad, src)["passed_all"] is False
    assert abs(cosine_warmup_lr_scale(200, 2000, 200, 0.1) - 1.0) < 1e-6
    assert abs(cosine_warmup_lr_scale(2000, 2000, 200, 0.1) - 0.1) < 1e-6


def test_train_config_auto_disables_cond_grad_when_condition_frozen(tmp_path: Path):
    cfg = TrainConfig(
        stage_name="arm_a",
        base_dir=tmp_path,
        source_ckpt=tmp_path / "src.pt",
        out_dir=tmp_path / "out",
        rendered_manifest=tmp_path / "m.jsonl",
        frozen_modules=("encoder", "condition"),
    )
    assert cfg.verify_nonzero_cond_grad is False


def test_h4_k41_arch_forward_and_weight_decay_policies():
    import json
    from src.recognition.model import ARCH_SPECS

    spec = ARCH_SPECS["h4_k41"]
    m = MenuRecognizer(
        classes=32,
        pretrained=False,
        temporal=True,
        orientation=True,
        temporal_norm="batch",
        use_bilstm=True,
        height_pool=spec["height_pool"],
        stage_strides=spec["stage_strides"],
        expected_h=spec["expected_h"],
        bilstm_layers=spec["bilstm_layers"],
    ).eval()
    x = torch.randn(2, 3, 64, 128)
    ori = torch.tensor([0.0, 1.0])
    lengths = torch.tensor([32, 24], dtype=torch.long)
    out = m(x, ori, lengths)
    assert out.shape == (32, 2, 32)

    opt_all, sum_all = build_parameter_controlled_optimizer(m, weight_decay=0.01, weight_decay_policy="all")
    assert sum_all["weight_decay_policy"] == "all"
    assert all(g["weight_decay"] == 0.01 for g in opt_all.param_groups)

    opt_ex, sum_ex = build_parameter_controlled_optimizer(
        m, weight_decay=0.01, weight_decay_policy="exclude_norm_bias"
    )
    assert sum_ex["weight_decay_policy"] == "exclude_norm_bias"
    wd_by_name = {g["name"]: g["weight_decay"] for g in opt_ex.param_groups}
    assert wd_by_name["encoder_decay"] == 0.01
    assert wd_by_name["encoder_no_decay"] == 0.0


def _make_tiny_training_fixture(tmp_path: Path, n_samples: int = 4):
    import json
    from PIL import Image

    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    vocab = {"char_to_id": {"あ": 1, "い": 2, "う": 3, "（": 4, "）": 5, "(": 6, ")": 7}}
    (tmp_path / "payload").mkdir(parents=True, exist_ok=True)
    vocab_path = tmp_path / "payload" / "vocab.json"
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")

    records = []
    for i in range(n_samples):
        rel = f"line_{i:02d}.png"
        Image.new("RGB", (64, 64), color=(240, 240, 240)).save(images_dir / rel)
        records.append({
            "image_rel": rel,
            "canonical_text": "あい",
            "orientation_flag": i % 2,
            "valid_width": 64,
            "bucket": 64,
        })
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")

    m = MenuRecognizer(
        classes=8,
        pretrained=False,
        temporal=True,
        orientation=True,
        temporal_norm="batch",
        use_bilstm=True,
        height_pool="conv2d",
    )
    src_ckpt = tmp_path / "source.pt"
    src_sha = save_atomic_with_sha256(src_ckpt, {"raw": m.state_dict(), "ema": m.state_dict()})
    return vocab_path, manifest, src_ckpt, src_sha


def test_trainer_v2_ema_frozen_identity_incremental_checksums_and_resume(tmp_path: Path):
    import json
    from src.recognition.trainer_v2 import _update_ema, run_training

    vocab_path, manifest, src_ckpt, src_sha = _make_tiny_training_fixture(tmp_path, n_samples=4)

    # 1. Verify _update_ema keeps frozen params & BN buffers bit-identical across 20 updates
    m = _tiny_model("conv2d")
    freeze_all_bn_eval(m)
    bn_snap = snapshot_bn_buffers(m)
    build_parameter_controlled_optimizer(m, frozen_modules=("encoder",), freeze_bn_affine=True)
    frozen_snap = snapshot_frozen_params(m)
    trainable_names = {n for n, p in m.named_parameters() if p.requires_grad}
    ema = {k: v.detach().clone() for k, v in m.state_dict().items()}
    for _ in range(20):
        with torch.no_grad():
            for n, p in m.named_parameters():
                if n in trainable_names:
                    p.add_(0.01)
        _update_ema(ema, m, decay=0.998, trainable_param_names=trainable_names)
    for k, v0 in bn_snap.items():
        assert torch.equal(ema[k].cpu(), v0)
    for k, v0 in frozen_snap.items():
        assert torch.equal(ema[k].cpu(), v0)

    # 2. Uninterrupted 2-epoch run vs interrupted after epoch 1 + resumed for epoch 2
    out_full = tmp_path / "out_full"
    cfg_full = TrainConfig(
        stage_name="resume_test",
        base_dir=tmp_path,
        source_ckpt=src_ckpt,
        source_key="raw",
        expected_source_sha=src_sha,
        out_dir=out_full,
        rendered_manifest=manifest,
        vocab_path=vocab_path,
        frozen_modules=("encoder",),
        freeze_bn_affine=True,
        epochs=2,
        batch_size=2,
        num_workers=0,
        use_bf16=False,
        job_id="job_full_1",
    )
    audit_full = run_training(cfg_full)
    assert (out_full / "checksums.json").exists()
    assert set(audit_full["checksums"].keys()) == {"epoch_01.pt", "epoch_02.pt"}
    done_obj = json.loads((tmp_path / "DONE.json").read_text(encoding="utf-8"))
    assert done_obj["job_id"] == "job_full_1"

    out_res = tmp_path / "out_res"
    cfg_ep1 = TrainConfig(
        stage_name="resume_test",
        base_dir=tmp_path,
        source_ckpt=src_ckpt,
        source_key="raw",
        expected_source_sha=src_sha,
        out_dir=out_res,
        rendered_manifest=manifest,
        vocab_path=vocab_path,
        frozen_modules=("encoder",),
        freeze_bn_affine=True,
        epochs=1,
        batch_size=2,
        num_workers=0,
        use_bf16=False,
    )
    # Note: config_signature includes `epochs`, so simulate interruption at epoch 1 of a 2-epoch run
    # by copying out_full/epoch_01.pt and resuming to epoch 2!
    out_res.mkdir(parents=True, exist_ok=True)
    ep1_copy = out_res / "epoch_01.pt"
    ep1_copy.write_bytes((out_full / "epoch_01.pt").read_bytes())

    cfg_resume = TrainConfig(
        stage_name="resume_test",
        base_dir=tmp_path,
        source_ckpt=src_ckpt,
        source_key="raw",
        expected_source_sha=src_sha,
        out_dir=out_res,
        rendered_manifest=manifest,
        vocab_path=vocab_path,
        resume_ckpt=ep1_copy,
        frozen_modules=("encoder",),
        freeze_bn_affine=True,
        epochs=2,
        batch_size=2,
        num_workers=0,
        use_bf16=False,
    )
    run_training(cfg_resume)
    ck_full = torch.load(out_full / "epoch_02.pt", map_location="cpu", weights_only=False)
    ck_res = torch.load(out_res / "epoch_02.pt", map_location="cpu", weights_only=False)
    assert ck_full["step"] == ck_res["step"]
    assert ck_full["epoch_avg_loss"] == ck_res["epoch_avg_loss"]
    for k in ck_full["raw"]:
        assert torch.equal(ck_full["raw"][k], ck_res["raw"][k]), f"raw mismatch on {k}"
        assert torch.equal(ck_full["ema"][k], ck_res["ema"][k]), f"ema mismatch on {k}"


def test_v72_preset_requires_fixed_plan_and_colab_probe_job_ownership(tmp_path: Path):
    import json
    import tarfile
    import pytest
    from src.recognition.colab_runner import build_probe_script, make_bundle_tar
    from src.recognition.trainer_v2 import run_training

    vocab_path, manifest, src_ckpt, src_sha = _make_tiny_training_fixture(tmp_path, n_samples=4)
    p_a = preset_module_policy("v72_arm_a")
    assert p_a["requires_fixed_plan"] is True
    assert p_a["expected_fixed_batches"] == 2000

    cfg_no_plan = TrainConfig(
        stage_name="v72_arm_a",
        base_dir=tmp_path,
        source_ckpt=src_ckpt,
        source_key="raw",
        expected_source_sha=src_sha,
        out_dir=tmp_path / "out_v72",
        rendered_manifest=manifest,
        vocab_path=vocab_path,
        frozen_modules=tuple(p_a["frozen_modules"]),
        freeze_bn_affine=p_a["freeze_bn_affine"],
        module_lrs=p_a["module_lrs"],
        epochs=1,
        requires_fixed_plan=True,
        expected_fixed_batches=2000,
        num_workers=0,
        use_bf16=False,
    )
    with pytest.raises(ValueError, match="requires a validated fixed batch plan"):
        run_training(cfg_no_plan)

    # Probe script must reject stale DONE.json from another job_id
    probe_src = build_probe_script(str(tmp_path), "new_job_123", ["sub"])
    (tmp_path / "DONE.json").write_text(json.dumps({"job_id": "stale_job_999"}), encoding="utf-8")
    ns = {}
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(probe_src, ns)
    probe_out = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert probe_out["done"] is False

    (tmp_path / "DONE.json").write_text(json.dumps({"job_id": "new_job_123"}), encoding="utf-8")
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        exec(probe_src, {})
    probe_out2 = json.loads(buf2.getvalue().strip().splitlines()[-1])
    assert probe_out2["done"] is True

    # Bundle tar packages src/recognition without requiring a code/ folder
    bundle = tmp_path / "bundle.tar.gz"
    make_bundle_tar(tmp_path, src_ckpt, bundle, bundle_dirs=("payload",))
    with tarfile.open(bundle, "r:gz") as tf:
        names = tf.getnames()
    assert any(n.endswith("src/recognition/trainer_v2.py") or n.endswith("src\\recognition\\trainer_v2.py") for n in names)
    assert any(n.endswith("payload/source_ckpt.pt") or n.endswith("payload\\source_ckpt.pt") for n in names)
