# AGY operating guide: delegated OCR experiments on Colab

This guide lets the user instruct AGY directly when Sol6/Codex is unavailable.
It also documents the Herdr handoff developed from V53 through V77. Current model/data status is in [Stage3 current status](stage3_recognition/current_status.md). Command examples were
checked against installed Herdr0.9.5 and `colab` CLI help on this project machine;
discover current syntax again before running them. This is an operating guide,
not authorization to start a new paid GPU run or to change an active experiment.
The user's specific experiment instructions and approved WORK_PLAN take priority
over generic defaults here or older architecture skills.

Project root: `D:\profolio\japanese_translation`.


## Current implementation and reliability corrections

- Read [current status](stage3_recognition/current_status.md) and the current recognition/Colab skills before configuring another job. No generic default overrides an explicit approved experiment.
- The reusable Strategy refactor is under [request-changes review](../outputs/recognition/review/refactor_v68_v73_review/REVIEW.md): model parity passed, trainer/runner integration did not. Fix and verify the gate before using them. Existing validated experiments use narrow adapters; do not copy per-version trainer/model bodies or silently modify protected src to get a job running.
- `colab sessions` inventories runtimes; `colab ls` lists files within a runtime. Distinguish legitimate empty inventory from auth/network errors. Use `colab new -s <owned-name> --gpu A100`, never `--name`. Check existing owner/job before creating or restarting.
- Require AGY to **code** controller launch/completion/failure notifications to a discovered `agent == "agy"` recipient: `herdr session list`, then `herdr --session <name> agent list`, then session-scoped `agent prompt` without `--wait`. Match the experiment workspace and re-resolve the pane before sending; no hardcoded IDs or controller-to-Codex fallback. Persist/retry undelivered events. AGY can forward user-requested reports to the coordinator separately. See the Colab skill for routing tests and failure handling. The installed CLI does not support the shorthand `herdr sessions`.
- Match record IDs, numeric render seeds, producer/consumer manifest paths and helper signatures in an actual unpacked-bundle test. Imports/py_compile alone miss runtime contracts. ProcessPool smoke must use real old/new/synthetic records and both orientation flags. Assert the actual smoke batch count, not two updates when only one batch exists.
- Validate exact required-term labels and actual consumed IDs; V72/V73's faulty counter obscured a real475/521 gap. Priority海胆/軟骨/雲丹/馬刺 require both orientations. Restore coverage without calling previously trained labels “unseen”.
- V76 Attempt2 policy: preserve native aspect ratio atH64, no width-driven height padding. Over512 uses100px-overlap windows; partial training labels need validated alignment. Test splits cannot use ground-truth alignment. Shared crop exporters still letterbox, so audit actual pixels and rebuild from native display crops. Geometry and vocabulary changes invalidate claims of direct old-baseline parity unless evaluated separately.
- Freeze BN statistics in both raw andEMA; average only allowed trainable weights, copy frozen tensors/buffers exactly. All-weight random training may update BN only when explicitly approved. V75B supports the current frozen policy, not a universal rule.
- Save full model/EMA/optimizer/scheduler/RNG/sampler cursor/config/manifest hashes. Publish atomic checksums after every checkpoint, verify job-owned DONE/PID and expected full artifact inventory before terminating the runtime. Stale DONE files cannot complete a new job. Evaluate locally after retrieval; an evaluator exception does not authorize rerunning training.
- True heldout font status requires file hashes against the training pool, not a folder name. Verified local Windows fonts are recorded in the existing synthetic-holdout audit. A train-font fallback is not heldout. Keep new priority probes' post-training construction disclosed.
- Distinguish corrected native splitting from historical letterboxed evaluation, 1920 short-paired versus2400 all-length H/V probes, collection-qualified menu IDs, eligible selected versus experimental continuation, and source/final raw/EMA. Menu-disjoint development sets repeatedly inspected are not fresh blind production tests.

## 1. Who does what

| Role | Responsibility |
|---|---|
| User | Defines objective, source model, budget, data restrictions and approval conditions. |
| Sol6/Codex, when present | Writes the concrete plan, sends it to AGY, reviews final evidence. |
| AGY | Configures validated components through narrow adapters, constructs data, validates, launches Colab, preserves checkpoints, retrieves results and writes the final handover. |
| Background controller | Checks machine status, syncs artifacts, verifies hashes and sends terminal notifications without repeated LLM turns. |

When the user instructs AGY directly, AGY takes responsibility for the entire
approved plan; it must not wait for an absent Codex session. If the run parameters
are missing, construct a concrete reviewable plan before requesting the missing
decision. Do not repeatedly ask about routine choices already covered by approval.
For an approved conditional plan, e.g. 'pilot then3epochs if gates pass', implement
that condition in code and proceed automatically only when the declared gates pass.

## 2. Communication through Herdr

### Discover, identify, send once

Use the intended user's live Herdr session. Read AGENTS.md and `herdr --skill`;
follow the installed skill's caller-context requirements. Do not spoof HERDR_ENV,
restart the user's server, or create an unrelated session to bypass a missing
connection. If the expected server/session is unavailable, preserve the prepared
plan and explain the actionable connection blocker.

```powershell
herdr status server
herdr agent list
herdr pane list
herdr agent read <verified-agy-pane> --source recent-unwrapped --lines 40
herdr agent prompt --help
```

Choose an AGY identity from live JSON, using its name or explicit pane ID. Do not
guess from focused UI, sidebar order, or a previous run. Historically AGY was
`w2:p3` and Sol6 was `w2:p1`; those are examples, not permanent addresses.
Read an unfamiliar idle/done pane before submitting input; a working agent may
have unrelated work. Do not interrupt or replace it. Reuse the existing AGY
session. If a new agent is explicitly needed and authorized, create an available
pane first and pass native flags after `--`, for example:

```text
herdr agent start <unique-name> --kind agy --pane <verified-shell-pane> -- --mode auto --model <verified-supported-model-id> --effort medium
```

Use a currently supported, user-approved model ID; do not blindly copy old Argon
names from historical instructions or restart a working agent to change models.

Save the full plan to the new experiment's WORK_PLAN.md, then submit a short
message pointing to it. PowerShell here-strings/structured subprocess arguments
avoid shell expansion and broken multiline quoting:

```powershell
$experimentPrompt = @'
Read <experiment-root>/WORK_PLAN.md and execute the approved plan.
Writes only inside <experiment-root>; existing source/data/models/docs are read-only.
Implement, validate, launch, retrieve and audit. Maximum2Markdown documents total.
Do not merely return another plan. Callback to <verified-sender-pane>, if available.
No training/promotion beyond the authorized budget and conditions.
'@
herdr agent prompt <verified-agy-pane> $experimentPrompt --wait --until working --timeout 15000
```

Herdr's `agent_prompted`/working response confirms task acceptance, NOT data
generation or training launch. A timeout does not prove non-delivery: read agent
state/output before retrying. Never blindly dispatch duplicate GPU jobs.

### Notifications, not conversation polling

Require these notifications:

1. **Launch acknowledgement:** real hardware verified, actual generated-data e2e smoke and first optimizer update passed, exact coverage/geometry audits saved, detached worker alive and durable sync/controller active. Wrapper PID alone is insufficient.
2. **One terminal completion or actionable failure:** actual phase/epochs/updates,
   handover path, retrieval/checksum status and blockers. No premature 'completed'.

Do not spend LLM tokens repeatedly requesting status. A plain background program
may check at a modest interval (e.g.90–180seconds); this is machine supervision,
not an agent conversation loop or a Colab lifetime-bypass keepalive.
Do not wait for the recipient to reply before completing an authorized next phase.
If Codex is absent, write status/HANDOVER locally and notify the user in AGY's
own session. A failed Herdr delivery does not mean training failed: preserve the
message and delivery error locally and distinguish those two states.

### The Windows permission issue we actually observed

Herdr sometimes returned PermissionDenied from a restricted child process even
though the Herdr server was an ordinary unelevated process. Launching the
authorized helper from an unrestricted **normal-user** execution context fixed
IPC. This was not proof that administrator rights were necessary.

Inspect the actual error and execution context. Do not change ACLs/security
settings, enable broad access, restart Herdr, or send approval keystrokes blindly.
If the execution environment requires approval for that context, use its normal
approval mechanism with a specific justification. User secrets never belong in
logs, prompts, payloads, Git, or diagnostic environment dumps.

## 3. Mandatory isolated experiment layout

Use a new, non-overwriting path such as:

```text
outputs/recognition/experiments/<new-run>/
  WORK_PLAN.md             approved plan; AGY reads it
  HANDOVER.md              AGY's one final implementation/result/resume summary
  code/                    copied/adapted implementation and controllers
  payload/                 permitted source weights, vocab, text/font assets
  manifests/               configs, hashes, data frequencies, gates, audits
  logs/                    launcher, sync and notification logs
  downloads/               verified remote artifacts
```

Maximum **two Markdown documents per experiment**: WORK_PLAN.md and HANDOVER.md.
Use JSON/JSONL/CSV for detailed metrics and state. Do not create extra progress,
analysis, final-summary or README Markdown files for the same run.

Unless the user specifically authorizes otherwise, AGY may not modify `docs/`,
`src/`, existing `scripts/`, source datasets, existing experiment artifacts,
checkpoints, the production registry, Git, or protected/system files. Read-only
reference access is allowed. Copy needed implementation into code/ before changing
it. Set PYTHONDONTWRITEBYTECODE=1 for reference imports to avoid incidental writes.
Do not clean up files or auto-promote a model as part of training.

## 4. Write a concrete WORK_PLAN before implementation

Every plan must specify:

- Exact source checkpoint/state (`model` raw or `ema_model`) and source hash,
  or explicitly random initialization with `pretrained=False` and no weight load.
- Architecture, vocabulary/blank IDs, preprocessing, orientation and CTC lengths.
- Corpus record count, disjoint construction slots, font/character coverage,
  augmentation rates, replay source and evaluation split rules.
- Which weights train, BN affine policy, and separately the BN-buffer policy.
- Effective batch/microbatch, optimizer groups, LR, warmup, schedule horizon,
  training precision, update/epoch budget and partial-batch handling.
- Exact metrics, selection gates, raw/EMA identities and conditional phases.
- Colab hardware/session, detachment, persistence, retrieval and resume strategy.
- Permitted writes, notification routing and handover requirements.

Do not infer the source from `best.pt` alone: historical V43 best.pt was an
unchanged source fallback, while V53 best.pt really contains epoch9 raw weights.
Inspect metadata and compare model tensors with the claimed epoch/arm.

Keep architecture changes separate from a data experiment unless requested.
The old vision skill describes a different generic ViT layout; it must not
override an explicit MobileNetV4/CTC run plan.

## 5. Colab CLI and a real A100

The installed package is colab-cli, but its executable here is **`colab`**.
Do not assume the command is named `colab-cli`. Start with local help:

```powershell
colab --help
colab skill
colab sessions
colab new --help
colab exec --help
colab upload --help
colab download --help
```

Examples after choosing an unused session name:

```powershell
colab new -s <run-session> --gpu A100
colab upload -s <run-session> <local-bundle.zip> /content/<run-bundle.zip>
colab exec -s <run-session> -f <local-probe-or-launch-script.py>
colab download -s <run-session> /content/<run>/manifests/metrics.jsonl <local-download-path>
```

Verify in remote code: CUDA available, actual GPU name contains A100, memory
bytes, BF16 support and library versions. Report GB/GiB units accurately.
An accepted session request or a successful upload does not prove A100 training.
Do not silently substitute local GPU/CPU or another accelerator. Do not stop or
reuse another experiment's session. Use existing CLI authentication; if interactive
authentication is needed, request that specific user action without reading or
printing credential files.

### Payload and remote generation

Upload only approved code, text/config/vocabulary manifests, licensed fonts and,
for fine-tuning, the exact authorized checkpoint. Random-init runs upload no model
weights. Do not upload local synthetic-image corpora when remote generation was
required. Include actual dependency versions and license/font hashes.

Build and render train and evaluation images on Colab **before training**. Long
generation should itself be resumable or detached; do not depend on an output-
silence timeout to infer failure. Extract only the trusted run bundle and validate
archive entries stay inside the dedicated remote directory.

## 6. Dataset construction contract

### Meaning and coverage

Use explicit manifest records rather than sampling arbitrary characters at
training time. Include source label, canonical label, stream, font hash, seed,
appearance, geometry, physical bucket, valid width and image hash/provenance.
Meaningful phrases must use compatible menu grammar. Pair qualifiers with the
appropriate dish category: portions, counts, preparation, tax, availability, etc.
Allow randomness among compatible choices, not unrelated random words.

Digit anchors should balance0–9 and include isolated digits, repeated digits,
short numbers, realistic prices and units. Balanced anchor presentations are
different from equal token counts—prices naturally contain many zeros.
Preserve original rendered text while normalizing fullwidth decimal digits to
ASCII. Do not apply broad NFKC or silently change the saved vocabulary IDs.

Count actual label properties separately from construction slots. A20%digit
slot can produce40%digit-containing lines because other slots also contain
prices. Similarly, a15%long-line slot does not mean15%of images use512pixels.
Audit actual physical widths, glyph heights, text lengths, single-character
coverage, fonts,0–9tokens and target glyph frequencies before training.

If the plan requires each character >=N occurrences or all521eligible terms,
enforce and test it. Reserve mandatory coverage before spending the corpus
budget. V53 failed its >=50character target for1,113canonical entries and left
three genuine characters unseen; record a failed target rather than claiming
full compliance. Normalized-away width aliases require separate visual counts.
Do not satisfy 'meaningful' coverage with made-up gibberish around rare glyphs.

### Full passes, replay and split separation

Full pass = every record ID in the declared run corpus exactly once per epoch.
Bucket/shuffle records, retain all final partial batches, and audit consumed IDs.
Do not use weighted replacement or steps-per-epoch truncation and call that a
full pass. Derive update count from actual bucket tails/gradient accumulation.

For50%replay, say whether that is all original records or a specified stratified
subset. Preserve source record IDs. If Colab images are gone, regenerate using
saved manifests/fonts/seeds/pipeline; verify hashes when available. A regenerated
view with different pixels is not an identical original, even if its label matches.

Keep evaluation seeds, image hashes, fonts and text independently controlled.
Report both unique-string and row-level train/eval overlap. Separate known-font,
held-out-font and unseen-text results; a different seed is not a no-overlap proof.
Check font hashes and family identities, not just filenames. An unseen-font score
on repeated training text is valid font-transfer evidence but not unseen-text
recognition. Do not use evaluation failures to select new training labels.
Create stratified clean-train probes: slicing the first500manifest records can
accidentally evaluate only the first stream, as happened in V53.

### H64 preprocessing: exact implementation matters

For the established recognizer, unless the plan explicitly changes it:

1. RGB, aspect-preserving height normalization toward64px; no independent width
   squeeze. Use actual valid pixel width, not character count, for bucket choice.
2. Physical buckets128/256/384/512; all1char lines use minimum128.
3. Real protective background guards and independently variable left/right
   margins. Reduce excess background or route wider; never clip glyph strokes.
4. Right padding uses edge replication/BORDER_REPLICATE. Do not force every line
   to384 or append artificial black/zero regions.
5. Verify orientation contract: actual vertical Japanese layout, CCW90rotation
   into a strip and orientation flag1; native horizontal strips use flag0.
6. ImageNet RGB mean/std for the current V43/V53 family; verify the actual source
   config rather than assuming a new normalization matches an old checkpoint.
7. Effective horizontal stride4: stem(2,2), first block(2,2), three later(2,1).
   EncoderH64/W384 yields H2/W96. V43/V53 use height mean + Conv1d; V60+ research uses Conv2d(960,256,(2,1),stride=(2,1)) height collapse. Both preserve T96; set checkpoint-compatible height_pool explicitly.
8. Compute `valid_t=ceil(valid_width/4)` after geometry changes. Use it in packed
   BiLSTM, fp32 CTCLoss and decoding. Physical bucket padding is not valid text.
9. Check CTC feasibility, including adjacent repeated target characters.

Pool ASCII/fullwidth parentheses before CTC collapse and normalize references
consistently. Keep saved physical IDs intact. For loss, use the tested alias-
column compaction; explicit-inf alias columns previously caused CTC-backward NaNs.
Keep literal and canonical metrics separate to expose normalization effects.

V51 showed that changing trained H64/stride4 weights directly to H96 or stride2
caused major regressions. Such changes need an explicitly authorized matched
architecture experiment, not an incidental preprocessing 'fix'.

### Augmentation

Keep the approved clean/mild/morphology rates and log actual delivered rates.
Use small morphology kernels, correct ink-mask polarity, and balanced dilation/
erosion; reject erased thin strokes and clipping. Mild3x3blur/noise is different
from5x5/7x7heavy appearance. Vary glyph size, qualifier size, spacing and margins
without changing the text label. Evaluate fixed deterministic views separately.

## 7. Training policy and honest pilot gates

BN affine parameters and running-stat buffers are separate decisions:

| Mode | Parameters | BN running statistics |
|---|---|---|
| V53-style random training | All trainable; no pretrained weights/teacher | Update explicitly, as approved. |
| V54-style focused continuation | All learnable weights incl. BN affine at low LR | Freeze all147buffers to the exact source checkpoint. |

For frozen BN: after model.train(), put BN modules into eval and assert means,
variances and counters remain bit-identical, including EMA snapshots. Restoring
eval/train mode is not enough unless the BN policy is reapplied. Do not silently
freeze to V30/V43 statistics when the source is V53.

Use optimizer groups with no weight decay on biases/norm/ndim<=1 parameters.
Verify intended parameters belong to the optimizer and receive updates. Use a
stable fp32 CTC loss, exact valid lengths, finite-loss checks and gradient clipping.
Match warmup/cosine horizon to actual planned updates. Record any intentional
optimizer/scheduler reset between pilot and long phase. No hidden LR escalation.

Run a short implementation smoke before the authorized budget: finite forward/
backward, classifier dimensions, trainable-module changes and correct BN policy.
It is not evidence that recognition gates passed. Evaluation uses model.eval();
restore the approved training mode afterward. Do not select on augmented training
loss or a nonrepresentative clean subset alone.

For conditional training, put numeric gates in the approved plan BEFORE the
pilot. Store source baselines, candidate metrics and gate decisions in JSON.
Check exact denominators and unrounded numbers. If no candidate passes, retrieve
the pilot and stop; do not weaken thresholds or substitute 'loss decreased'. If
one passes and continuation was already authorized, continue without waiting for
an absent coordinator. Preserve selected pilot, all epoch snapshots and final
states even if later epochs regress. Never confuse last with best or raw with EMA.

## 8. Detached execution and persistence

Colab background execution does **not** guarantee unlimited runtime. Detached
processes survive a launch-client exit, not destruction of the VM. A local sync
controller also stops if the user's computer sleeps or shuts down. Prefer actual
durable remote storage where available; otherwise state the residual risk clearly.

The successful V53 remote launch used this pattern inside a short remote script:

```python
import os, subprocess, sys
from pathlib import Path

base = Path('/content/<unique-run>')
(base / 'logs').mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env['PYTHONPATH'] = str(base / 'code')
env['PYTHONUNBUFFERED'] = '1'
with (base / 'logs' / 'train.log').open('a', encoding='utf-8') as log:
    worker = subprocess.Popen(
        [sys.executable, '-u', str(base / 'code' / '<trainer>.py'),
         '--base-dir', str(base)],
        cwd=base, env=env, stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True, close_fds=True,
    )
(base / 'logs' / 'train.pid').write_text(str(worker.pid))
```

Adapt arguments to the actual trainer. Verify worker liveness and advancing
status after the launch exec returns; PID existence alone can be stale. Inspect
startup errors before acknowledging launch. Do not run the whole job under one
fragile foreground CLI call. For long image generation, detach the whole remote
pipeline or checkpoint the generation phase too.

The local transfer controller must run independently of the initiating agent.
On Windows, use Start-Process with WindowStyle Hidden and proper argument lists,
logs and PID/status files. Do not use an administrator shell by default. Avoid
blocking LLM waits; the background controller handles modest-rate machine checks.

In V53, `colab drivemount` required interactive TTY authentication. Its workaround
was atomic runtime writes plus epoch-wise local retrieval, not a magically durable
`/content` directory. If authentication is necessary, request that specific action
or use an explicitly approved alternative. Do not assert Drive mounting succeeded
without evidence, and never mount/copy another user's private data.

## 9. Checkpoints, resume, retrieval and completion order

Each epoch checkpoint needs model/raw and EMA identity, optimizer/scheduler,
RNG, epoch/update and sample progress, vocabulary/config, and corpus/consumption
hashes. Save mid-epoch resume state when practical. A resume must restore actual
state and the remaining record order; reloading weights alone is not a faithful
resume. Include a tested, exact resume command in HANDOVER.md.

Write checkpoints atomically to a temporary file, then rename. Produce SHA-256
only after finalization. Keep unique epoch filenames and preserve existing sources.
Download to a .part path, compare with the REMOTE expected hash, and atomically
finalize locally only on a match. Retry transfers without labeling partial files
complete. A local inventory hash proves inventory consistency, not remote-to-local
retrieval equality unless compared with the remote digest.

Do not stop a runtime when an arbitrary subset of files verifies. Required
artifacts include every epoch and selected/last checkpoint, raw/EMA identities,
resume state, code/config/vocab, font assets/licenses or reproducible sources,
corpus/frequency/split/evaluation manifests, predictions where needed, metrics,
full-pass/BN/optimizer audits and complete logs. Download small preview sheets;
do not duplicate all generated images locally unless requested. Record remaining
unrecoverable or runtime-only assets honestly.

Correct terminal sequence:

1. Trainer writes completed status, final audits and immutable artifact hashes.
2. Controller retrieves all required outputs and verifies remote/local hashes.
3. AGY finalizes HANDOVER.md with actual results, deviations, resume commands and
   retrieval status; no launch-time 'still running' header remains.
4. Stop ONLY this run's owned Colab session; check status confirms released.
5. Write terminal local status and send the one completion callback. If delivery
   fails, save the callback text and error for the user; do not retrain/restart.

Transient CLI/network/authentication errors are not proof of worker death. Keep
'unknown/unreachable' distinct from 'failed'; inspect persistent status/logs before
claiming either. In an actual failure, preserve the newest valid resume state,
logs and partial verified retrieval, and report the exact blocking condition.

## 10. Evaluation definitions and results that must not be conflated

- Report CER as edits/reference characters, plus exact-line count and D/I/S.
  CER can exceed100% due to insertions. State literal versus canonical decoding.
- Separate clean/noisy,1char/2char, known/held-out font, ordinary/long, and
  synthetic/YOLO results. A5%new-generator CER is not a5%real-menu CER.
- Opening/closing bracket recall is distinct from correctly reading the enclosed
  qualifier and the rest of the line. Give integer counts and matching rules.
- Digit recall must specify alignment-based versus multiset counting. Multiset
  recall can credit a digit in the wrong order; it is not price correctness.
- Report ordered digit-sequence exactness and actual standalone-price accuracy.
  Exactness of a long descriptive line containing円 is a different metric.
- When comparing models, reuse identical ordered images, normalization, physical
  buckets, valid lengths and decoder parameters. Verify old-model prediction
  parity before interpreting a new-model gain or regression.
- For YOLO compare paired menus, not assumed-independent crops; report source
  labels as VLM-produced where applicable. Development sets repeatedly inspected
  or used for selection are not fresh blind production tests.
- Beam search and lexicon reranking are separate optional experiments. Keep
  unknown terms possible and audit price/digit changes. Do not silently change
  the primary decoder or compare beam scores with an earlier greedy baseline.

Useful findings from the actual sequence: V53 learned clean brackets much better,
but matched local testing gave58.60%YOLOCER versus41.50%V43 and39.73%V48.
It was not promoted. This is why independent matched local evaluation follows
synthetic training before any production decision.

## 11. What HANDOVER.md must contain

Keep one concise document with:

1. Completed/blocked/failed state and actual Colab release/retrieval status.
2. Exact source, initialization, raw/EMA/selected/last checkpoint identities.
3. What actually ran: corpus sizes, passes, updates, fonts, preprocessing,
   learning rates, parameter/BN policy, duration and measured throughput.
4. Epoch/pilot results with metric definitions and denominators; all gate outcomes.
5. Coverage/split/hash audits, reproduction changes, exclusions and plan deviations.
6. Artifact paths, remote/local checksum verification and precise resume commands.
7. Limitations, what remains untested, and confirmation that no auto-promotion or
   protected-file changes occurred. Never claim all requirements passed when they did not.

Detailed per-image results, hashes and frequencies go in machine-readable files.
Checkpoint inventories should identify aliases/deduplicated identical states,
not merely say 'all outputs downloaded'.

## 12. Copy-paste instruction when Sol6 is unavailable

The user can give AGY this message, replacing the bracketed fields:

> Read `docs/agy_colab_training_workflow.md`, AGENTS.md and the approved
> `[experiment-root]/WORK_PLAN.md` read-only. You are responsible for implementing,
> validating, generating data remotely, launching the authorized Colab A100 job,
> preserving/retrieving checkpoints and writing the final HANDOVER.md. Sol6 is
> unavailable: do not wait for it or turn the job into repeated planning messages.
> Writes are restricted to `[experiment-root]`; do not change docs/, existing
> source/data/models/registry/Git. Keep at most WORK_PLAN.md and HANDOVER.md as
> Markdown documents. Follow the approved initialization, BN policy, dataset,
> preprocessing, budgets and numeric pilot gates exactly. If the pilot passes
> and continuation is authorized, execute the continuation automatically; if not,
> stop and report. Use the installed `colab` CLI, a verified A100, detached
> execution and checksum-verified persistence/retrieval. Send one verified launch
> acknowledgement and one terminal result in this session; also use
> `[verified-Herdr-callback]` if available, without waiting for a reply. Do not
> repeatedly poll with LLM turns. Report genuine authentication/hardware/storage
> blockers; never expose secrets, overwrite sources, or auto-promote.

If no approved run plan exists, add the source/model mode, data mix, corpus size,
epochs/update budget and desired pass criteria to the user request first. This
guide supplies execution discipline, not permission to invent those decisions.

## 13. Read-only examples from this project

- Random-initialized10full-pass recipe and final handover:
  `outputs/recognition/experiments/v53_random_colab_full10/{WORK_PLAN.md,HANDOVER.md}`.
- V53 isolated implementations: `code/corpus_builder.py`, `renderer.py`,
  `train_v53.py`, `remote_launch.py`, `sync_and_complete.py`, `final_audit.py`.
  Treat these as historical read-only references. Do not duplicate trainer/model/orchestrator bodies or reuse hardcoded sessions/paths/callback IDs. Prefer parameterized validated components and thin adapters; the generic src trainer/runner remain gated until the recorded review fixes and integration tests pass.
- Conditional focused continuation:
  `outputs/recognition/experiments/v54_short_digits_colab/WORK_PLAN.md`.
- Matched local evaluation:
  `outputs/recognition/experiments/v53_random_colab_full10/local_evaluation_plan.json`
  and `evaluation_local/report.json`.

These are examples, not blanket permission to repeat runs. V53's stale initial
handover, nonrepresentative train-clean subset and sparse character coverage are
known pitfalls to improve, not patterns to copy.
