# Menus051-100fine-tuning with previous validation retained

## Completed result: retain v3

Completed21epochs;besttrainedepoch1,then20withoutimprovement. Independent audit
passed:45latesttrainingmenus,5new+5previousvalidation,687train/126valboxes,
55snapshotpages,reviewflags/geometry/JSONtext/hashchecks,nooverlap. Previous
validationbytesidenticalandbaselinepreviousmetricsreproducev3exactly. Source
checkpointunchangedandcandidatecopymatchesrunsbest.pt. No trainingrunning.

| Validation mAP50-95 | Starting v3 | v4best | Change(pp) |
|---|---:|---:|---:|
| Combined10menus |89.44%|87.86%|-1.58|
| Latest5menus |89.33%|88.55%|-0.78|
| Previous5menus |89.75%|87.30%|-2.45|

Do notpromotev4. Besttrainedepochisnotbetterthanstartingcheckpoint. Previous
precisionalsodeclines97.46→95.18%whilerecallrises99.41→99.96%;mAP50nearly
unchanged. Tighterlocalizationqualityregressedratherthanbroadnewdetectiongain.
Neithernewnorpreviousvalidationbeatsbaseline. Traininglossimproveswithout
validationgain;causecouldincludefeature/BNdrift,augmentationorannotationpolicy,
butnoneisisolatedbythisrun. Validationonly5generatedmenus/grouplimitsclaims.

Preferred detector remainsv3:
`outputs/detection/training/banana50_finetune_v3/yolo11n_menu_obb_banana50_best.pt`.
v4candidateisretainedforinspection,notautomaticallyinstalled. Fullmetricsand
independentaudit:`review.md`,`completion_verification.json`inthev4outputfolder.
Beforeanotherlatest-onlyrun,inspectannotationconsistencyandtestalowerLR/
frozenbackbone-or-BNcontrolifauthorized. Do notextendthisrunautomatically.

User authorized latest-label fine-tuning and explicitly requested the previous
validation set also be used. Newrunstartsfromthelatestv3bestcheckpoint,not
theoriginalpre-v3weights. Humanannotationfilesandallpriorcheckpointsarepreserved.

## Dataset

Snapshot:`data/datasets/detection/banana51_100_v4/`.

| Group | Pages | Horizontal boxes | Vertical boxes |
|---|---:|---:|---:|
| Latest training |45|545|142|
| Latest validation |5|58|15|
| Previous validation |5|38|15|

All50newpages051-100areexplicitlyreviewed,760boxes. Trainusesonlythelatest45
pages/687boxes. Validationcombinesnew5menus/73boxeswithprevious5menus/53boxes:
55totalpagesand813boxesinsnapshot. Previousbatchtrainingimagesarenotincluded.

Newvalidationmenus:060,070,080,086,100. Aninitialevenlyspaced060/070/080/090/100
choicecontainednoverticalboxes,sowereplaced090with086beforetraining. Theunused
preflightsnapshotispreservedat
`outputs/detection/preflight_snapshots/banana51_100_horizontal_only_holdout/`.
Previousvalidation010/020/030/040/050isbyte-for-bytefromthefrozenv3snapshot.
Source/image/label/reviewflags/geometryandcrosssplitduplicatehasheswerechecked.

Main`dataset.yaml`selectscombined10menuvalidation. `dataset_latest.yaml`and
`dataset_previous.yaml`provideindependentsubsetreporting,notindependenttestsets.
Bothvalidationgroupscontainhorizontalandverticalboxes. Allvalidationmenusstay
outoftraining. FutureStage3cropsmusthonortheseexactmenuholdoutidentities.

## Training and evaluation

Source:`outputs/detection/training/banana50_finetune_v3/yolo11n_menu_obb_banana50_best.pt`
(v3bestepoch36,SHAf851ba684051e0ecd2bb0f9e1cb8914a7b2ed4ce347f287608efac24a3936592).
Samecontrolledfine-tuningsettingsasv3:100epochmaximum,patience20,AdamW1e-4,
cosineend0.1,batch8/nbs16,imgsz640,mildgeometry/coloraugmentation,noflips,
FP32CUDA,noAMPselfcheckdownloads. Installedlocalruntimeonly,offlinecaches.

Evaluateinitialandfinetunedbestonthreegroups:combined/latest/previous. Combined
10menuvalidationmAP50-95selectsbest;separatereportsrevealprevious-setforgetting
thatcouldotherwisebehiddenbynew-menuimprovement. Nogrouppasscapsinvented.
Thesevalidationpagesareusedforselection;noindependentreal-menutestclaim.

Code:`scripts/stage2_yolo/finetune_banana_batch2.py`.
Hiddenlauncher:`scripts/stage2_yolo/run_banana_batch2_finetune.ps1`.
Output:`outputs/detection/training/banana51_100_finetune_v4/`.
Read`status.json`,stdout/stderrlogs,results.csvandfinalevaluation_report.json.
Candidatewillbe`yolo11n_menu_obb_banana100_best.pt`;existingroot/v3weightsare
notreplaced. OneHerdrcompletion/failuremessage,no pollingloop.
