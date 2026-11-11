# Parent validation of Gemini deliverables

Validated all5,069 candidate IDs exactly once against the original export,
including normalized surface forms,allowed classes and confidence values.
Reconciliation approved496 unique current-vocabulary terms:114 one-character
and382 two-character. These comprise485 high-confidence independent agreements
and11 separately sourced essential supplements. The main SQLite dictionary hash
still matches the reconciliation manifest.

The reviewer classified749 candidates practical,478 valid but uncommon,368
irrelevant and3,474 uncertain. This is full record coverage,not exhaustive manual
verification of every term. The approved list is conservative and incomplete.
The57 suggested supplements remain suggestions; only the11 separately checked
essentials were added in this pilot version. Other suggestions and disagreements
need further review.

Numerical corrections to the raw Gemini summary:1,407 out-of-vocabulary
candidates are27.76% of5,069,not37.23%;the supplement JSON contains57 unique
terms,not62. Its illustrative `天ぷら` example is three
characters,not two; the actual approved list was checked to contain only1/2
characters. The source deliverable is preserved unchanged for provenance.

Direct read-only SQL checks confirmed inappropriate menu-default senses for
several existing entries:ピザ has the slang gloss `fat, ugly person`;中生 is read
なかて with an agricultural sense;大福 has a tea-related reading/gloss;生 is read
なまり;杯 is readつき;麺 has the gloss `flour`. These are sense-selection problems
for menu usage,not proof that the Japanese surface forms are invalid. No main
dictionary readings,translations or schema were modified. The reviewer's precise
claims about upstream implementation causes and regulatory supplement evidence
were not independently established by this validation.

The reconciled-list10-update GPU smoke completed for both arms. Full generation
then exposed rare256px two-character crop layouts incompatible with arbitrary
replacement terms. Before any full-run updates,the attempt was preserved as
`v31_curated_short_failed_layout`. The corrected policy replays those58 pairs
identically in both arms and selects the same total70% curated quota from eligible
128px pairs. No glyphs are squeezed/clipped and physical batch buckets remain
matched. The corrected pilot artifacts record the exact exception/counts.
