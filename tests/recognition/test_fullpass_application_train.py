from collections import Counter
import random

from outputs.recognition.fullpass_application_train import full_plan
from src.recognition.font_training import row_canvas


def test_full_pass_preserves_partial_buckets_and_paired_views():
    original=[];rendered=[]
    for width,count,span in ((128,18,60),(256,34,180),(384,6,300)):
        original.extend(dict(text='文字',glyph_box=[0,0,span,64]) for _ in range(count))
        for _ in range(count//2):
            pair=len(rendered)//2
            rendered.extend(dict(index=pair,view=view,canvas=width) for view in ('clean_control','texture_color_noise'))
    state=random.getstate();a=full_plan(original,rendered,1,34);b=full_plan(original,rendered,2,34)
    assert random.getstate()==state
    assert a==full_plan(original,rendered,1,34) and a!=b
    assert sum(map(len,a))==116
    assert any(len(batch)<32 for batch in a)
    assert Counter(i for batch in a for k,i in batch if k=='original')==Counter(range(58))
    assert Counter(i for batch in a for k,i in batch if k!='original')==Counter(range(58))
    for batch in a:
        assert len({row_canvas(original[i]) if k=='original' else rendered[i]['canvas'] for k,i in batch})==1
        streams=Counter(k for k,i in batch)
        assert streams['original']==2*streams['clean_control']==2*streams['texture_color_noise']
        pairs=Counter(rendered[i]['index'] for k,i in batch if k!='original')
        assert all(v==2 for v in pairs.values())
