# V34 full-pass broad-mixture continuation

User authorized the long run after V33 failed, with the additional requirement
that all521 terms eligible in V33 appear at least once in training.

## Completed result

V34 early stopped after2full epochs,8002updates and256000presentations. The
completion audit verifies all521required exact labels each epoch,with each
original/generated view visited once and both partial batches retained. No
training remains running. All4trained evaluations failed retention;best.pt
is tensor-identical to V30,last.pt equals trained epoch_02.pt at update8002.

Resized clean/noisy CER improves13.33/16.61→9.76/12.06%,but seen-font CER
worsens4.97→7.67%,original clean4.95→7.24%,historical10→12.37%. Deletions
dominate the clean-data regression. Larger short confirmation improves,but
older clean1-char and original pooled1/2-char retention still fail. YOLO stays
0/38exact. Retain V30;do not promote V34 or extend the same recipe.

Read the [final analysis](../../outputs/recognition/training/v34_broad_fullpasses6/analysis.md)
and [completion audit](../../outputs/recognition/training/v34_broad_fullpasses6/completion_audit.json).

## Recipe and exact coverage

- Start from V30 control/best.pt,update2000. Keep mean-height pooling,
  MobileNetV4,temporal BN and packed BiLSTM. No V32 height convolution.
- Original64000 clean training records plus32000 newly rendered clean/noisy
  pairs:128000 image records per complete epoch. Keep15 fonts and broad
  historical label distribution. No curated50,tier weighting,new morphology,
  stronger blur or fixed384 stream.
- Freeze the corpus before gradient updates. All original records and both views
  of every generated pair are visited exactly once per epoch. Same-width batches
  contain16original+8clean+8noisy;bucket-tail batches retain the same50/25/25
  proportions and are not dropped. There are4001updates per complete epoch.
- Native render size24–120,H64 proportional resize,next128/256/384 bucket,
  border replication,mild appearance recipe inherited from V29/V30.
- The exact521-term set is taken from V33 sampling.json meaningful_draws keys.
  The original64000 records contain117of these terms as exact whole-line labels.
  Count originals plus the proposed broad generated corpus,protect already-covered
  required labels,and override only redundant compatible short slots for terms
  still entirely missing. Each override retains text length,font support and
  bucket. This minimal coverage correction is explicitly user-authorized and
  audited separately;it does not restore the failed50%curated mixture.
- Training asserts521/521 exact-label coverage before the first gradient update.
  Corpus/manifest.json records each term's exact count and every overridden pair.
  Clean/noisy views are two images of one base render,not independent fonts.

## Training and selection

- Maximum6 full epochs (24006updates),AdamW,encoder initial5e-7/head3e-6,
  fully unfrozen,cosine decay to20%of initial rates,clip5. No decay on norm/bias.
- Existing V30 selection score and retention rules,plus the previous short-stratum
  checks. Frozen larger confirmation set:1000one-char and1000two-char base renders,
  each clean/noisy (4000images),half known/half held-out font cases. This is an
  image holdout,not unseen labels. Confirmation strata also use source+1pp caps.
- Validate every2000updates within an epoch and at epoch end,skipping redundant
  checks within100updates of the end. Stop after3checks without an eligible
  improvement,only after finishing the current epoch. Preserve V30 fallback.
- best.pt is retention-gated;last.pt and epoch checkpoints preserve actual trained
  states. metrics.jsonl contains every evaluation. Checkpoints include optimizer,
  RNG,epoch and batch position; --resume requires matching config/code/corpus.
- Preparation caches completed render pairs to permit restart. No artifacts are
  deleted or pushed to GitHub.

## Execution

Code:[fullpass_application_train.py](../../outputs/recognition/fullpass_application_train.py).
Launcher:[run_v34.ps1](../../outputs/recognition/run_v34.ps1).
Output:`outputs/recognition/training/v34_broad_fullpasses6/`.
Logs:`outputs/recognition/v34.stdout.log`, `v34.stderr.log`.

The two-epoch GPU smoke completed12updates/384presentations and verified full
small-corpus coverage. The full-pass planner test verifies no dropped/duplicated
records,paired views,partial bucket batches,stream ratios and deterministic plans.
The run is launched in a hidden background process. After preparation it starts
training automatically. Completion/failure is sent once through Herdr to Codex;
there is no agent status-polling loop.
