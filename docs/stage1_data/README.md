# Stage 1 data preparation

The synthetic base is `data/processed/synthetic_base_v1/`; shared resources are `data/resources/fonts/`, `data/raw/` and `data/processed/lexicon/`. Explicit custom `--data-root` directories keep the original self-contained layout.

Build commands below change derived data; they are not needed to inspect the existing datasets:

```powershell
python -m scripts.download_fonts --jmdict
python -m src.data.lexicon_builder
python -m src.data.synthetic_renderer --workers 1
```

Offline checks can write fresh reports without replacing the historical evidence:

```powershell
python -m scripts.data.verify_stage1_dataset --report outputs/reorganization/stage1_audit.json
python -m scripts.data.verify_menu_augmentation --data-root data/processed/augmentation/run_500 --report outputs/reorganization/augmentation_500_audit.json
```

The 1,100-pair and 500-pair augmentation runs remain separate. Do not combine correlated crops as independent examples. Retain the existing generation ledger and source licenses.

[Summary and limitations](stage1_summary.md) · [Augmentation](menu_augmentation.md) · [Photo collection](tabelog_scraper_review.md)
