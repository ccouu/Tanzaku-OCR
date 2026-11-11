from collections import Counter
from src.recognition.font_training import epoch_plan,gate,lr_factor


def rows_fixture():
    rows=[]
    for i in range(640):
        n=(i%10)+1;span=min(370,n*36)
        rows.append(dict(text='日'*n,glyph_box=[0,0,span,64],cohort='old' if i%2 else 'new',
                         morphology_actual='dilate' if i%4==0 else 'erode' if i%4==1 else 'none'))
    return rows


def test_full_pass_nearest_buckets_and_exact_small_morph_mix():
    rows=rows_fixture();batches,audit=epoch_plan(rows,1,123)
    keys=[k for b in batches for k in b]
    assert Counter(k[0] for k in keys)==Counter(range(640))
    assert sum(k[1] for k in keys)==32 and sum(k[2] for k in keys)==320
    assert all(len({k[4] for k in b})==1 and len(b)<=32 for b in batches)
    assert all(k[4]==128 for k in keys if len(rows[k[0]]['text'])==1)
    assert epoch_plan(rows,1,123)==(batches,audit)
    assert epoch_plan(rows,2,123)[1]['plan_sha256']!=audit['plan_sha256']


def test_lr_nonzero_endpoint():
    assert lr_factor(1,20)==lr_factor(2,20)==1
    assert abs(lr_factor(20,20)-.1)<1e-12
    assert lr_factor(10,20)>lr_factor(19,20)>lr_factor(20,20)


def test_retention_gate_rejects_drift():
    report={'seen':{'0.0':{'cohort_macro':{'old':.08}}}}
    for k in ('original_bucket','historical_bucket','yolo_provisional'):report[k]={'0.0':{'all':{'cer':.08}}}
    baseline={k:.07 for k in ('original_fonts','original_bucket','historical_bucket','yolo_provisional')}
    assert gate(report,baseline)['eligible']
    report['historical_bucket']['0.0']['all']['cer']=.081
    assert gate(report,baseline)['failed']==['historical_bucket']
