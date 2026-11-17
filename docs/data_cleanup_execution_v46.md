# Post-V46 data cleanup

Deleted 26,015 audited disposable files (12.51 GiB logical file bytes). Training and final evaluation completed before deletion.

Unique source menus, annotations, fonts, active corpora, source code, review pages, environments, V38 rollback, V43 promoted weights and all non-cache checkpoints were preserved. Protected checksums, current recognizer load/forward, and training/detection imports passed.

Exact path/SHA256 deletion manifest and journal: `outputs/recognition/cleanup/v46_post_training/`. See `report.json` for actual file-byte totals, free-disk delta and skipped files. The broader historical cleanup plan was not executed.

This is a historical execution record, not a current disk inventory or authorization for another deletion. Later V53–V77 datasets/checkpoints/review exports must be assessed separately; see [current status](stage3_recognition/current_status.md).
