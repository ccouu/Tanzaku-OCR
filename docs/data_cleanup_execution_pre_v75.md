# Recognition output cleanup — 2026-11-17

User authorized pruning useful old output artifacts while **not deleting anything newer than V74**. This is a new bounded execution, not execution of the obsolete broad cleanup proposal.

## Completed scope

Only classic numbered experiment/pilot/training roots **V74 or earlier** under `outputs/recognition/` were eligible. Newer and ambiguous families were protected wholesale, including **all H4 and L2 V01 folders**, the newer V60 architecture ablations/follow-up, and original V75/V75B/V76/V77. No directory was recursively deleted.

**68,848 files removed, 10.735 GiB of logical file contents.** After approximately 55.39 MiB of audit/recovered-source records, estimated net logical reclamation is **10.681 GiB**. Actual filesystem free-space change can differ due to allocation/compression and concurrent activity. No deletion errors.

| Removed class | Files | Approximate GiB |
|---|---:|---:|
| Superseded intermediate epoch checkpoints | 61 | 2.570 |
| Superseded intermediate update checkpoints | 100 | 3.656 |
| Smoke/preflight checkpoints | 31 | 0.879 |
| Completed upload bundles | 8 | 0.717 |
| Completed upload chunks | 88 | 1.479 |
| Old generated training images | 68,560 | 1.435 |

Generated-image deletion was confined to `training/v34_broad_fullpasses6/corpus/images/` and `training/v40_new_characters_2000/targeted/images/train/`. Their labels/manifests and code remain. These training renders must be regenerated if those exact old runs are repeated. All evaluation/probe/confirmation images and all review pages were retained.

## Preserved and verified

- Selected/best/final/initial and important lineage states, plus **124 code/documentation-referenced checkpoint paths**.
- Key V53 epoch9, V60 epoch10, V68 epoch13, H4 V74 epoch2, original V75 epoch4 and L2 V01 phase1 epoch2 / phase2 epochs6 and8 exist after cleanup.
- Promoted V43, rollback V38, detector weights and registry hashes are unchanged.
- Every protected newer/ambiguous experiment and all `outputs/recognition/evaluation/` and `review/` inventories match their pre-cleanup file size/mtime snapshots.
- All existing extracted code, JSON/JSONL/CSV reports, metrics, audits, labels, split manifests, vocabularies and fonts remain. Raw/processed datasets, `src/`, scripts, environments and Stage2 outputs were outside deletion scope.
- Before removing bundles, **251 source/configuration entries** were recovered into the cleanup audit's `recovered_bundle_source/`, deduplicated by content hash. Preserve that directory with the audit; it contains source recovery copies, not disposable training data.

## Audit and reproducibility

[Cleanup report](../outputs/recognition/cleanup/pre_v75_20261117/report.json) and sibling artifacts:

- `deletion_plan.json`: exact paths, original sizes/mtime and rationale.
- `deletion_manifest.jsonl`: deletion intent/completion journal.
- `retained_dependency_checkpoints.json`: retained references.
- `protected_snapshot.json` and `protected_hashes.json`: verification evidence.
- `recovered_source_manifest.json` and `recovered_bundle_source/`: recovered bundle source/configuration text.
- `cleanup.py`: bounded file-only implementation; **do not blindly rerun execution against its already-consumed plan**.

Historical checksum inventories and reports were not rewritten to hide pruned intermediate files. A historical manifest may therefore list a checkpoint that was deliberately removed; consult the deletion journal before diagnosing it as corruption. No new training, model promotion, Git commit or push occurred.
