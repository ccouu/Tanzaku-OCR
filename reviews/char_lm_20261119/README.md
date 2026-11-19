# Char-LM fixes — reviewed candidates, NOT applied

The source under `src/recognition/char_lm_beam.py` and the current inspector retain their existing implementation. Files here are isolated, tested candidates awaiting an implementation decision. They are not imported by production entrypoints.

- `candidates/src_char_lm_beam_patch.diff` and `inspector_stateless_patch.diff`: valid patches against the current source snapshot. Both pass `git apply --check`; do not apply blindly after source changes.
- Corresponding full candidate modules: immutable LM tables/provenance, explicit schema and decode-mode compatibility, stateless per-request options, true alpha-zero beam, optional R1/R2 routing and HTTP error handling.
- `test_candidate_char_lm.py`: concurrency, schema, brute-force CTC and routing checks. Run from repository root with `python -m pytest reviews/char_lm_20261119/candidates/test_candidate_char_lm.py -q`.
- `migrate_lm_to_v2.py`: explicit model-table packaging helper. It does not run automatically. Preserve original LM assets and compare table/prediction parity before deployment.

Only published helper path discovery was made repository-relative. Original audited candidates, outputs, datasets and model files were not modified by packaging. Candidate fixes require a complete module/inspector migration, not copying only the API names into the existing server.

See [validation and limitations](../../docs/stage3_recognition/char_lm_validation.md). The L2_v06 **recognizer** registry promotion is separate from decoder-candidate approval.
