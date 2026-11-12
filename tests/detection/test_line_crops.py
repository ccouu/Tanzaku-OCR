import math
import numpy as np
from PIL import Image
from src.detection.line_crops import process,overlap,recognition_input,rectify,inside


def rect(x,y,w,h,angle=0,label='dish_horizontal',conf=.9,id='a'):
    q=np.array([[-w/2,-h/2],[w/2,-h/2],[w/2,h/2],[-w/2,h/2]])
    r=math.radians(angle);q=q@np.array([[math.cos(r),math.sin(r)],[-math.sin(r),math.cos(r)]])+[x,y]
    return dict(id=id,points=q.tolist(),label=label,confidence=conf)


def test_rotated_iou_not_aabb():
    a=rect(100,100,140,10,45);b=rect(90,110,140,10,45,id='b')
    assert overlap(a['points'],b['points'])['iou']<.01
    assert len(process([a,b],300,300)['crops'])==2


def test_duplicate_and_contained_word_preserve_complete_line():
    a=rect(100,50,180,20,conf=.6,id='full');b=rect(50,50,40,20,conf=.95,id='word');c=rect(100,50,180,20,conf=.5,id='dup')
    r=process([a,b,c],300,100);assert len(r['crops'])==1
    assert np.ptp(np.array(r['crops'][0]['core'])[:,0])==180
    assert len([d for d in r['decisions'] if d['action']=='suppress_duplicate'])==2


def test_merge_horizontal_and_vertical_fragments():
    h=process([rect(40,50,40,20,id='a'),rect(65,50,40,20,id='b')],200,200)
    assert len(h['crops'])==1 and h['crops'][0]['source_ids']==['a','b']
    v=process([rect(50,40,20,40,label='dish_vertical',id='a'),rect(50,65,20,40,label='dish_vertical',id='b')],200,200)
    assert len(v['crops'])==1 and v['crops'][0]['orientation_flag']==1


def test_duplicate_union_does_not_discard_extreme_strokes():
    a=rect(60,50,100,20,id='a');b=rect(64,50,100,20,id='b',conf=.8)
    r=process([a,b],160,100);assert len(r['crops'])==1
    assert all(inside(p,r['crops'][0]['core']) for d in (a,b) for p in d['points'])


def test_neighboring_rows_and_distant_price_not_merged():
    r=process([rect(60,50,70,20,id='a'),rect(90,64,70,20,id='b'),rect(190,50,30,20,id='price')],240,150)
    assert len(r['crops'])==3
    assert r['conflicts'] # Genuine core overlap is flagged,not destroyed.


def test_chain_does_not_bridge_rows():
    r=process([rect(30,50,30,20,id='a'),rect(50,54,30,20,id='b'),rect(70,58,30,20,id='c')],150,100)
    assert len(r['crops'])==2


def test_adjacent_regions_only_merge_in_explicit_preview_mode():
    detections=[rect(40,50,40,20,id='a'),rect(85,50,40,20,id='b')]
    assert len(process(detections,200,100)['crops'])==2
    assert len(process(detections,200,100,min_merge_iom=0)['crops'])==1


def test_padding_only_overlap_is_removed_without_reducing_cores():
    a=rect(60,50,100,20,id='a');b=rect(60,72,100,20,id='b');r=process([a,b],150,120)
    assert len(r['crops'])==2 and not r['conflicts']
    assert overlap(r['crops'][0]['crop_polygon'],r['crops'][1]['crop_polygon'])['intersection']<.001
    for c in r['crops']:
        assert abs(np.ptp(np.array(c['core'])[:,1])-20)<1e-6
        assert all(inside(p,c['crop_polygon']) for p in c['core'])


def test_crossing_orientations_not_merged():
    r=process([rect(80,80,100,20,id='h'),rect(80,80,20,100,label='dish_vertical',id='v')],160,160)
    assert len(r['crops'])==2 and r['conflicts']


def test_recognition_widths_and_rectification():
    for w,expected in ((90,128),(200,256),(350,384),(480,512),(800,512)):
        im,meta=recognition_input(Image.new('RGB',(w,64),'white'))
        assert im.size==(expected,64) and meta['input_length']==(meta['valid_width']+3)//4
        assert meta['vertical_letterbox']==(w>512)
    image=Image.new('RGB',(20,100),'white');image.paste('red',(0,0,20,50));image.paste('blue',(0,50,20,100))
    r=process([rect(10,50,20,100,label='dish_vertical')],20,100)
    strip=rectify(image,r['crops'][0]['crop_polygon'])
    assert strip.width>strip.height and strip.getpixel((5,5))[0]>200 and strip.getpixel((strip.width-5,5))[2]>200
