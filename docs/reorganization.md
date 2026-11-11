# Workspace reorganization

Code is grouped by capability, data by lifecycle, and guides by stage. `workspace_paths.json` is the explicit old-to-new artifact map. Python path resolution accepts these legacy paths at supported entrypoints. No symlinks are required.

Documentation, including historical guides, now uses current filesystem locations. Implementation links point into `src/detection/`, with the review server and HTML under `src/detection/review/`. Current command examples use `python -m src.detection...`; the previous `python -m scripts.stage2_yolo...` commands remain supported compatibility entrypoints. Paths in frozen data artifacts retain their original provenance values.

Historical manifests, provenance snapshots, dataset archives, labels and image bytes are retained unchanged. Old paths inside them are historical evidence, resolved through the map. New consolidation records include `source_kind`; legacy records are classified through the same compatibility reader. Frozen snapshots keep their existing records and class IDs.

`outputs/reorganization/inventory.json` records each moved file's original path, destination, byte count and SHA-256. Run `python tools/reorganize_workspace.py --verify` to recheck migrated files. Audit CLIs support `--report` to preserve historical reports. New dataset archives are stored inside their snapshot as `dataset.zip`.

The inaccessible legacy `data/.pytest-stage1-tmp/` scratch directory remains in place because Windows denied access. All new test scratch is under `.cache/test-runs/`. `.stage2_packages/` is retained for the existing dependency bootstrap.

No datasets were rebuilt, no model was retrained, and no proposals were promoted to reviewed labels. No Git repository was initialized. Existing license notices and the generation ledger remain with the assets.

The one-time migration scripts refuse overwrites. `refactor_layout.py` and `update_layout_docs.py` describe the completed source migration and must not be rerun on the reorganized tree.

Validation completed: 77 regression tests passed (67 before the refactor); all 92,619 migrated files matched their recorded hashes. Stage 1, Stage 2 and both augmentation audits passed with new reports under `outputs/reorganization/`. Checks also covered 19 legacy command entrypoints, standalone worker help, documentation links, loading existing review annotations, frozen v1/v2 archive hashes, disjoint source groups and the selected checkpoint hash. No Colab training session was started.
