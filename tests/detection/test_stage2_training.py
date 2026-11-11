import math
from pathlib import Path
import pytest

from scripts.stage2_yolo.prepare_training import orientation, grouped_split, group_keys


def record(ident, info=None, sha=None):
    return dict(id=ident,sha256=sha or ident,source_info=info or {})


def test_price_orientation_uses_rotated_rectangle_axes():
    p=[[0,0],[80,0],[80,20],[0,20]]
    angle=math.radians(40)
    turned=[[x*math.cos(angle)-y*math.sin(angle),x*math.sin(angle)+y*math.cos(angle)] for x,y in p]
    assert orientation(turned)[0]=='dish_horizontal'
    assert orientation([[0,0],[20,0],[20,80],[0,80]])[0]=='dish_vertical'
    assert orientation([[0,0],[20,0],[20,20],[0,20]])[0]=='dish_horizontal'


def test_connected_content_background_and_original_photo_stay_together():
    records=[record('menu',{'source_menu_id':'A'}),
             record('composite',{'source_menu_id':'A','background_group':'123','background_sha256':'rawhash'}),
             record('background',{'restaurant_id':'123'},'rawhash')]
    records += [record(str(i)) for i in range(20)]
    split,groups=grouped_split(records)
    assert split['menu']==split['composite']==split['background']
    assert any(set(g)=={'menu','composite','background'} for g in groups)
    assert set(split.values())=={'train','val','test'}
    assert grouped_split(records)==(split,groups)
    for a in records:
        for b in records:
            if split[a['id']]!=split[b['id']]:
                assert not group_keys(a)&group_keys(b)


def test_unsplittable_components_fail():
    with pytest.raises(ValueError,match='three independent'):
        grouped_split([record('a',{'source_group':'same'}),record('b',{'source_group':'same'})])


def test_source_ratios_with_related_generated_pair():
    from collections import Counter
    from scripts.stage2_yolo.prepare_training import source_ratio_split, source_kind
    records=[]
    for i in range(10):
        r=record('real'+str(i),{'restaurant_id':str(i)})
        r['source_root']='data/raw/tabelog_menus';records.append(r)
    for i in range(20):
        r=record('gen'+str(i),{'source_menu_id':str(i//2)})
        r['source_root']='data/processed/augmentation/run_500';records.append(r)
    split,groups=source_ratio_split(records)
    assert Counter(split[r['id']] for r in records if source_kind(r)!='generated')=={'train':5,'val':5}
    assert Counter(split[r['id']] for r in records if source_kind(r)=='generated')=={'train':18,'val':2}
    for group in groups:assert len({split[ident] for ident in group})==1
    assert set(split.values())=={'train','val'}


def test_atomic_recovery_checkpoint_publication(tmp_path):
    from types import SimpleNamespace
    import json
    from scripts.stage2_yolo.train_yolo_obb import publish_recovery, sha
    last=tmp_path/'last.pt';last.write_bytes(b'resumable last')
    best=tmp_path/'best.pt';best.write_bytes(b'best model')
    trainer=SimpleNamespace(epoch=0,last=last,best=best,metrics={'map':.7})
    state=tmp_path/'status.json'
    recovery=publish_recovery(trainer,tmp_path,state,100,{})
    assert recovery['epoch']==1
    assert sha(recovery['files']['last']['path'])==recovery['files']['last']['sha256']
    trainer.epoch=1;last.write_bytes(b'epoch2')
    unchanged=publish_recovery(trainer,tmp_path,state,100,recovery)
    assert unchanged['epoch']==1
    assert Path(recovery['files']['last']['path']).read_bytes()==b'resumable last'
    assert json.loads(state.read_text())['epoch']==2
    assert not list(tmp_path.rglob('*.tmp'))


def test_snapshot_excludes_unreviewed_and_backs_up_price_labels(tmp_path):
    import json
    import zipfile
    from PIL import Image
    from src.common import dump_json, digest
    from scripts.stage2_yolo.common import annotation, save_pair
    from scripts.stage2_yolo.prepare_training import prepare
    root=tmp_path/'review';(root/'images').mkdir(parents=True)
    records=[]
    for i in range(12):
        ident=f'image{i}';image=root/'images'/(ident+'.png')
        Image.new('RGB',(100,80),(i*20,10,20)).save(image)
        r=dict(id=ident,image=image.name,sha256=digest(image),width=100,height=80,
               role='page',source_root='data/raw/tabelog_menus',source_path=str(image),source_info={})
        doc=annotation(r,[dict(label='price_tag',points=[[10,10],[30,10],[30,60],[10,60]])],{})
        doc['flags']['reviewed']=i!=11
        save_pair(root/'labels',r,doc);records.append(r)
    dump_json(root/'manifest.json',dict(records=records))
    output=tmp_path/'snapshot'
    report=prepare(root,output)
    assert report['images']==11 and report['excluded_unreviewed']==1
    assert report['price_conversion_count']==11
    assert all(r['id']!='image11' for r in report['records'])
    assert json.loads((Path(report['backup_path'])/'image0.json').read_text())['shapes'][0]['label']=='price_tag'
    assert json.loads((root/'labels/image0.json').read_text())['shapes'][0]['label']=='dish_vertical'
    assert json.loads((root/'labels/image11.json').read_text())['shapes'][0]['label']=='price_tag'
    with zipfile.ZipFile(output/'dataset.zip') as z:
        assert 'dataset.zip' not in z.namelist()
        assert not any('image11' in name for name in z.namelist())
        for name in z.namelist():
            if name.startswith('labels/'):
                assert z.read(name).split()[0]==b'1'
