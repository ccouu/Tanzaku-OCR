# V35 matched frozen-BN pilot

User authorized the2000-update continuation after the four-way zero-update BN
diagnostic. This run tests freezing statistics during training rather than
post-hoc swapping co-adapted states.

## Completed outcome

V35 completed2000updates,63980presentations,313/521exact-term encounters.
All147BNbuffers stayed unchanged;all98affine tensors received nonzero gradients
and changed final values. Baseline predictions reproduce exactly. Frozen BN
improves most matched normal-BN metrics:seen5.52→4.94%,resized clean11.21→10.24%,
noisy14.30→13.15%,historical10.29→9.92%,strong appearance63.75→62.32%.
It still fails pooled original2-char retention (57/400=14.25%,cap13.5%). No trained
checkpoint passes all gates;best.pt remains V30,last.pt is trained update2000.
No longer training is running and no checkpoint is promoted. See the
[full analysis](../../outputs/recognition/pilots/v35_frozen_bn_2000/analysis.md).

## Authorized continuation: finish the first full pass

Completed:4001total updates,128000records,all521terms once or more. Independent
audit confirms no repeated/dropped records,AdamW step4001,147fixedBNbuffers,
and exact resume prediction parity. Frozen-BN final original clean4.86%,seen5.03%,
held5.50%,historical10.51%,resized clean/noisy10.18/12.67%. Two short gates still
fail:pooled1-char61/200(cap60),2-char56/400(cap54). No trained checkpoint passed
all gates;best.pt stays V30,last.pt is trained4001. No second epoch is running.
Read the [full-pass analysis](../../outputs/recognition/pilots/v35_frozen_bn_fullpass4001/analysis.md).

User approved continuing to4001total updates. The separate continuation output
is `outputs/recognition/pilots/v35_frozen_bn_fullpass4001/`;original V35 artifacts
remain unchanged. Code is
[finish_frozen_bn_pass.py](../../outputs/recognition/finish_frozen_bn_pass.py),
launcher[run_v35_fullpass.ps1](../../outputs/recognition/run_v35_fullpass.ps1).

Restore V35 update2000 model,AdamW moments/step counts and Torch/CUDA RNG.
Resume validation must exactly reproduce V35 step2000predictions before training.
Use only V34 epoch1 batches2001–4001,with original global cosine schedule and
fixed147BNbuffers;gamma/beta remain trainable. No optimizer restart or data
regeneration. A two-update GPU resume smoke passed.

Combined prefix+continuation is128000presentations with each original and each
generated view exactly once,all521required exact labels included. No records are
repeated to force coverage. Evaluate2500/3000/3500/4001;skip the redundant4000
check one update before the end. Selection and source-relative gates stay unchanged.
Normal-BN equal-budget control is V34 update_004001.json. Stop at4001;no second
epoch or automatic promotion. Completion/failure sends one Herdr message.

- Source:V30 control/best.pt,unchanged mean-height model.
- Frozen pixels,labels,valid lengths and exact first2000ordered batches from
  V34 epoch_01_plan.json. No data regeneration or changed term allocation.
- Same fresh AdamW,seed20261212,gradient clipping5,BF16CNN/head and FP32recurrence/CTC.
- Initial encoder5e-7/head3e-6,exact V34 cosine trajectory over24006planned
  updates. Do not compress the schedule into2000updates.
- After each model.train(),put all49BN modules into eval mode. Freeze only
  running_mean,running_var,num_batches_tracked (147buffers);all98gamma/beta
  parameter tensors remain trainable. All remaining model weights train normally.
- Assert buffers remain bit-identical and affine gradients exist. The GPU
  smoke passed2updates and finite CTC loss,with all98affine gradients present.
- Reproduce V30 baseline predictions exactly against V34 initial.json before
  training. Evaluate at500/1000/1500/2000with unchanged selection and retention
  gates,including the larger clean/noisy1/2-char confirmation strata.
- Equal-budget normal-BN control is V34 update_002000.json. It is already
  available,so another normal-BN training arm is unnecessary. Final versus
  best eligible checkpoint must be reported separately.
- This is a partial pass. The full frozen corpus covers all521required exact
  labels,but this prefix may not encounter all521. sampling.json records exact
  encountered counts and missing terms;do not alter matching to force coverage.
- No longer training or promotion is automatic.

Code:[frozen_bn_pilot.py](../../outputs/recognition/frozen_bn_pilot.py).
Launcher:[run_v35.ps1](../../outputs/recognition/run_v35.ps1).
Output:`outputs/recognition/pilots/v35_frozen_bn_2000/`.
One-shot completion/failure message through Herdr;no polling loop.
