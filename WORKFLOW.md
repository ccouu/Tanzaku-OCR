# Project workflow

1. **Collect resources and sources:** retain originals, licenses, checksums, photo identities and the existing generation ledger under `data/raw/` and `data/resources/`.
2. **Prepare Stage 1 data:** build the shared lexicon and synthetic base; keep each augmentation run separate under `data/processed/`. Audit sample units and related-source splits before combining data.
3. **Review Stage 2 pages:** consolidate into `data/annotations/menu_obb/`, generate proposals and explicitly review/save labels. Predictions remain separate from reviewed labels. Reference crops never become detector pages.
4. **Freeze a detector dataset:** prepare a fresh directory under `data/datasets/detection/`. Preserve source-connected groups and reviewed-only admission. Keep the snapshot and its `dataset.zip` immutable.
5. **Train and evaluate:** choose a fresh output run under `outputs/detection/training/`; retain logs, recovery checkpoints and evaluation together. Select weights into `checkpoints/` with dataset/run provenance.
6. **Inspect fresh inference:** keep overlays and predictions under `outputs/detection/inference/`. The current 100-photo collection is unlabelled; it has no measured accuracy score.
7. **Stage 3 handoff:** V27 is the active recognition implementation. Its20-epoch run reached4.48% seen-font/5.27% held-out-font CER, but failed provisional GenAI-menu retention. Reviewed real transcriptions and real-menu validation remain pending. Stage4 follows validated recognition. See [Stage3 guide](docs/stage3_recognition/README.md); older experiments are archived.

The current selected detector uses the reviewed v2 dataset with 862 training and 351 validation images; there is no independent test split. Validation accuracy gates remain unmet. Reorganization does not authorize promotion of proposals or a new training run.

See [Stage 2 training](docs/stage2_detection/stage2_training_v2.md) for explicit Colab commands and recovery. Agent routing is documented separately in [orchestration](docs/orchestration.md) and `AGENTS.md`.
