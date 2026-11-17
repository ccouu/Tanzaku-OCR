"""Fail-closed cleanup of audited disposable data after V46 completion only.

No active corpus, source menu, annotation, font, checkpoint, review, source code,
or environment is a deletion candidate. Cache test fixtures are disposable.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import stat
import time

from src.recognition.utils import sha,write_json
from src.recognition.current import REGISTRY,load_current_recognizer,current_checkpoint

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'outputs/recognition/training/v46_focused_recurrent_4000'
AUDIT=ROOT/'outputs/recognition/cleanup/v46_post_training'
CACHES=('.cache/test-runs','.cache/pip-recognition')
SMOKES=('v29_application_mixture_smoke','v30_short_retention_smoke','v30_short_retention_smoke_final',
        'v30_short_retention_smoke_verified','v31_routing_smoke','v31_parent_preflight_smoke',
        'v31_reconciled_smoke','v31_curated_short_failed_layout','v32_height_smoke','v32_height_smoke_v2',
        'v33_tiered_smoke','v34_fullpass_smoke','v35_frozen_bn_smoke','v35_resume_smoke','v45_latin_head_only_smoke')
IMAGE_SUFFIXES={'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'}


def checked(path):
    """Validate both lexical ancestry and resolved target; reject all reparse links."""
    p=Path(os.path.abspath(path));resolved=p.resolve()
    assert p!=ROOT and p.is_relative_to(ROOT) and resolved==p and resolved.is_relative_to(ROOT),str(p)
    for part in [p,*p.parents]:
        if part==ROOT:break
        if part.exists():
            info=part.lstat()
            assert not part.is_symlink() and not getattr(info,'st_file_attributes',0)&stat.FILE_ATTRIBUTE_REPARSE_POINT,str(part)
    return p


def files_under(folder):
    folder=checked(folder)
    if not folder.exists():return []
    result=[]
    for current,dirs,files in os.walk(folder,followlinks=False):
        checked(current)
        for name in dirs:checked(Path(current)/name)
        for name in files:
            p=checked(Path(current)/name)
            if p.is_file():result.append(p)
    return result


def source_references(folder):
    relative=folder.relative_to(ROOT).as_posix();hits=[]
    # Source references are conservative blockers, even for old launchers.
    for base in ('src','scripts','tests','outputs/recognition/review','outputs/detection/crop_review'):
        if not (ROOT/base).exists():continue
        for p in (ROOT/base).rglob('*'):
            if p.suffix not in ('.py','.ps1','.json','.yaml','.yml'):continue
            if p.resolve()==Path(__file__).resolve():continue
            text=p.read_text(encoding='utf-8-sig',errors='replace').replace('\\\\','/').replace('\\','/')
            if relative in text:hits.append(p.relative_to(ROOT).as_posix())
    return hits


def plan(cutoff):
    candidates=[];skipped=[];allowed_roots=[]
    def append(p,reason):
        info=p.stat()
        if info.st_mtime>cutoff:
            skipped.append(dict(path=p.relative_to(ROOT).as_posix(),reason='Modified after audit cutoff; preserve'));return
        candidates.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=info.st_size,mtime_ns=info.st_mtime_ns,sha256=sha(p),reason=reason))
    for rel in CACHES:
        folder=checked(ROOT/rel);allowed_roots.append(rel)
        for p in files_under(folder):append(p,'Disposable pytest fixture or installed-package download cache; no environment files')
    for name in SMOKES:
        rel='outputs/recognition/pilots/'+name;folder=checked(ROOT/rel)
        if not folder.exists():continue
        refs=source_references(folder)
        if refs:skipped.append(dict(path=rel,reason='Conservative source-reference blocker',references=refs));continue
        allowed_roots.append(rel)
        for p in files_under(folder):
            if p.suffix.lower() in IMAGE_SUFFIXES:append(p,'Obsolete explicitly allowlisted smoke/failed-layout rendered image; keep reports and checkpoints')
    # Delete a duplicate dataset only when the complete relative-file inventory
    # AND every file hash match the preserved canonical crop directory.
    duplicate=checked(ROOT/'data/processed/gemini_recognition')
    canonical=checked(ROOT/'data/processed/gemini_banana_mobile_menus_50_crops')
    if duplicate.exists():
        a={p.relative_to(duplicate).as_posix():p for p in files_under(duplicate)}
        b={p.relative_to(canonical).as_posix():p for p in files_under(canonical)}
        refs=source_references(duplicate)
        equal=bool(a) and a.keys()==b.keys() and not refs
        if equal:equal=all(p.stat().st_size==b[k].stat().st_size and sha(p)==sha(b[k]) for k,p in a.items())
        if equal:
            allowed_roots.append(duplicate.relative_to(ROOT).as_posix())
            for p in a.values():append(p,'Complete dataset duplicate verified by relative paths, sizes and SHA256 against canonical preserved crops')
        else:skipped.append(dict(path=duplicate.relative_to(ROOT).as_posix(),reason='Not proven exact duplicate, or source-referenced; preserve',references=refs))
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),cutoff=cutoff,allowed_roots=allowed_roots,
        candidates=candidates,skipped=skipped,planned_bytes=sum(r['bytes'] for r in candidates),
        policy='Only explicit disposable caches, unreferenced smoke images, and completely hash-identical duplicate dataset. All checkpoints outside disposable test cache protected.')


def main():
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--audit-only',action='store_true',help='Inventory only; return before all unlink/rmdir operations')
    mode.add_argument('--execute',action='store_true',help='Delete only after completed V46 training/evaluation artifacts exist')
    args=parser.parse_args()
    os.chdir(ROOT);AUDIT.mkdir(parents=True,exist_ok=True)
    # A stable cutoff prevents cleanup from collecting files created while training.
    cutoff_path=AUDIT/'cutoff.json'
    if not cutoff_path.exists():write_json(cutoff_path,dict(timestamp=time.time()-3600))
    cutoff=json.loads(cutoff_path.read_text())['timestamp']
    if args.execute:
        report=json.loads((WORK/'report.json').read_text())
        status=json.loads((WORK/'status.json').read_text())
        assert report['completed'] and status['state']=='completed'
        assert (WORK/'analysis.md').is_file() and (WORK/'final_raw.pt').is_file() and (WORK/'final_ema.pt').is_file()
    protected={p.as_posix():sha(p) for p in [REGISTRY.resolve(),current_checkpoint(),ROOT/'checkpoints/menu_recognizer_v38.pt',
        ROOT/'outputs/recognition/pilots/v43_digits_prices_2000/final_raw.pt']}
    manifest=plan(cutoff);write_json(AUDIT/'deletion_manifest.json',manifest)
    print(json.dumps(dict(execute=args.execute,files=len(manifest['candidates']),planned_GiB=manifest['planned_bytes']/2**30,
                         skipped_groups=len(manifest['skipped']))),flush=True)
    if not args.execute:return
    before=shutil.disk_usage(ROOT).free;deleted=0;count=0;skipped=[]
    journal=AUDIT/'deleted.jsonl'
    with journal.open('a',encoding='utf-8') as stream:
        for row in manifest['candidates']:
            p=checked(ROOT/row['path'])
            assert any(p.is_relative_to(checked(ROOT/r)) for r in manifest['allowed_roots'])
            if not p.exists():continue
            st=p.stat()
            if st.st_mtime_ns!=row['mtime_ns'] or st.st_size!=row['bytes'] or sha(p)!=row['sha256']:
                skipped.append(dict(path=row['path'],reason='Changed since audit'));continue
            try:p.unlink()
            except OSError as exc:
                skipped.append(dict(path=row['path'],reason=type(exc).__name__));continue
            deleted+=row['bytes'];count+=1;stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
    for root in manifest['allowed_roots']:
        folder=checked(ROOT/root)
        if not folder.exists():continue
        for current,dirs,files in os.walk(folder,topdown=False,followlinks=False):
            p=checked(current)
            try:p.rmdir()
            except OSError:pass
    assert all(sha(Path(p))==h for p,h in protected.items())
    model,ck=load_current_recognizer('cpu');assert len(ck['vocab']['char_to_id'])==1779
    import torch
    torch.set_num_threads(4)
    with torch.inference_mode():
        out=model(torch.zeros(1,3,64,128),torch.tensor([0]),torch.tensor([32]))
        assert out.shape==(32,1,1780) and torch.isfinite(out).all()
    # Source imports retain their historical helper modules; never archive code.
    import scripts.recognition.train_v46_focused
    import src.detection.line_crops
    result=dict(completed=True,deleted_files=count,deleted_bytes=deleted,deleted_GiB=deleted/2**30,
        free_disk_delta_bytes=shutil.disk_usage(ROOT).free-before,skipped=skipped,
        protected_checksums_unchanged=True,promoted_model_load_and_forward_pass=True,
        active_training_and_detection_imports_passed=True,manifest=str(AUDIT/'deletion_manifest.json'),
        note='File-byte total is exact; free-disk delta may include unrelated background activity. Unique source data and all non-cache checkpoints preserved.')
    write_json(AUDIT/'report.json',result)
    (AUDIT/'report.md').write_text('# Post-V46 cleanup\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (ROOT/'docs/data_cleanup_execution_v46.md').write_text(
        '# Post-V46 data cleanup\n\n'
        f"Deleted {count:,} audited disposable files ({deleted/2**30:.2f} GiB logical file bytes). "
        'Training and final evaluation completed before deletion.\n\n'
        'Unique source menus, annotations, fonts, active corpora, source code, review pages, environments, '
        'V38 rollback, V43 promoted weights and all non-cache checkpoints were preserved. '
        'Protected checksums, current recognizer load/forward, and training/detection imports passed.\n\n'
        'Exact path/SHA256 deletion manifest and journal: `outputs/recognition/cleanup/v46_post_training/`. '
        'See `report.json` for actual file-byte totals, free-disk delta and skipped files. '
        'The broader historical cleanup plan was not executed.\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
