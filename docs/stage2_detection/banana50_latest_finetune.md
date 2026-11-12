# Latest-only Banana50 detector fine-tuning

## Completed result

Completed56epochs,earlystoppedafter20withoutimprovement;selectedbestepoch36.
Independent audit verified45train/5validationmenus,all50latestreviewedpages/
668boxes,allimage/label/annotationsourcehashes,nocrosssplitduplicatehashes,and
zerooldertrainingimages. Candidatecopymatchesrunsbest.pt;rootcheckpointSHA
unchanged. No trainingrunning.

| Metric on the five validation menus | Original | Fine-tuned best |
|---|---:|---:|
| Precision |96.55%|97.46%|
| Recall |96.05%|99.41%|
| mAP50 |98.84%|99.41%|
| mAP50-95 |86.29%|89.75%|

Candidate:`outputs/detection/training/banana50_finetune_v3/yolo11n_menu_obb_banana50_best.pt`.
Details:`review.md`,`completion_verification.json`,`evaluation_report.json`
in that outputfolder. Lastcheckpointisepoch56,notselectedbest. Fivegenerated
validationmenuswereusedforselection;noindependenttestorolderrealretention
benchmarkrun. RecognitionV37usedoldcachedcrops,notthenewdetectoroutputs.
Noautomaticreplacement/deploymentorend-to-endCERclaim.

User requested fine-tuning using **only the latest annotation** after reviewing
all50pages in `data/annotations/gemini_banana_mobile_menus_50/labels/`.
Frozen snapshot:`data/datasets/detection/banana50_latest_v3/`.

| Split | Menus | Horizontal boxes | Vertical boxes |
|---|---:|---:|---:|
| Train |45|456|159|
| Validation |5|38|15|

All668boxes come from explicitly human-reviewed JSON/text pairs;proposals and
earlier reviewed_v1/v2datasets contribute zero images. Image hashes,dimensions,
convex OBBgeometry,JSON/text agreement and review flags were checked. Original
humanannotationfiles are preserved. Source pages are copied byte-for-byte into
an immutable snapshot with annotation/provenance hashes in training_manifest.json.

Validationmenus010/020/030/040/050match the existing Stage3menuholdout and remain
excluded from detectortraining. Validation selects checkpoints,so is not an
independenttest. Duplicate image hashes do not cross splits.

Start:`checkpoints/yolo11n_menu_obb_best.pt`,SHA256
`853659a53572315238798842f81ffcbfada5dd2a23629a0844cedf29258169bd`.
Fine-tune allweights,not fromscratch:AdamWLR1e-4,cosinefinal0.1,100epochmaximum,
early-stop patience20,batch8,nbs16,imgsz640,3warmupepochs,warmupbiasLR1e-4.
Mildgeometryrotation5degrees/translation.03/scale.15/perspective.0002,mosaic.25
disabledlast10epochs. No mirrors,90degreeorientationflips,mixup,erasing or
additionaldatasets. FP32CUDAtraining;AMPchecks/downloadsdisabled. Packages
already installed locally;allruntimecaches/logging/settings under workspace,
offline mode,no automatic installs or remote experimenttracking.

Baselineval of the original checkpoint and finalval of selectedbest use the
same five menus. Newcandidate:
`outputs/detection/training/banana50_finetune_v3/yolo11n_menu_obb_banana50_best.pt`.
Originalcheckpointsfile is never replaced automatically. `runs/menu_obb/weights/`
keepsbest/last;results.csv records epochs. evaluation_report.json comparesbaseline
andfinetunedmetrics,butdoesnotestablishretentiononolderrealmenus.

Code:`scripts/stage2_yolo/finetune_banana_latest.py`.
Hiddenlauncher:`scripts/stage2_yolo/run_banana_finetune.ps1`.
Output:`outputs/detection/training/banana50_finetune_v3/`.

LauncherstagedafteractiveV37recognitionpilot:uses native Wait-Process on the
V37launcherPID,no polling loop andno concurrentGPUtraining. Requires completed
V37statusbeforetraining;dependencyfailure sendsoneHerdrnotification for recovery.
Initialqueue failed beforetraining because PowerShellJSON had already parsed an
ISOtimestamp intoDateTime,andre-parsing its culture-dependent string swapped
day/month. Fixed by preservingDateTimetype/useinvariantISOparsing;initialfailure
savedin initial_queue_failure.json. Correctedqueue is active.

Readstatus.json/launcher_status.json for queued/training/completedstate.
stdout.log/stderr.log contain output;process.json records launcherPID.
Onecompletion/failureHerdrmessage to w2:p1,noconstantpolling. When V37finishes,
review its saved results on CPU while this detector uses theGPU.
