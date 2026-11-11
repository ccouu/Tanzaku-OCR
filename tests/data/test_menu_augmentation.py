import json
import random

import numpy as np
from PIL import Image, ImageDraw
import pytest

from src.common import digest, dump_json
from src.data.menu_augmentation import (AugmentationConfig, build, corners, extract_ctc_line,
    grouped_split, homography, menu_transform, project_menu_plane, scene_transform,
    transform_points, validate_polygon, warp_image)


def test_projective_mapping_and_inverse():
    source = corners(200,100)
    target = np.array([[40,60],[300,30],[280,220],[60,200]],dtype=float)
    matrix = homography(source,target)
    assert np.allclose(transform_points(source,matrix),target)
    interior = np.array([[70,30],[100,80]],dtype=float)
    assert np.allclose(transform_points(transform_points(interior,matrix),np.linalg.inv(matrix)),interior)


def test_transformed_pixels_follow_polygon():
    image = Image.new('RGBA',(100,100),'white')
    ImageDraw.Draw(image).rectangle((30,40,50,60),fill='black')
    matrix = homography(corners(100,100),np.array([[30,25],[140,30],[130,145],[20,120]]))
    output = warp_image(image,matrix,(180,180))
    x,y = transform_points([[40,50]],matrix)[0]
    assert output.getpixel((round(x),round(y)))[:3] == (0,0,0)
    assert output.getpixel((0,0))[3]==0


def test_geometry_never_clips_full_menu():
    for source_size in [(1052,1494),(640,640),(1600,500)]:
        for seed in range(30):
            rng = random.Random(seed)
            matrix,pose = menu_transform(source_size,640,rng,30,10,10,1.2)
            polygon = transform_points(corners(*source_size),matrix)
            validate_polygon(polygon,(640,640))
            assert polygon.min()>0 and polygon.max()<640
            assert 0<=pose['pitch_magnitude_degrees']<=30
            assert abs(pose['pitch_degrees'])==pose['pitch_magnitude_degrees']
            assert -10<=pose['roll_degrees']<=10 and -10<=pose['yaw_degrees']<=10
            placement = scene_transform(640,rng,1.2,1.2,polygon)
            scene_polygon = transform_points(polygon,placement)
            validate_polygon(scene_polygon,(640,640))
            assert placement[0,0]==pytest.approx(1.2)
            assert np.allclose(np.ptp(scene_polygon,axis=0),1.2*np.ptp(polygon,axis=0))


def test_pitch_yaw_and_roll_have_their_physical_axes():
    base = project_menu_plane((200,100),0,0,0)
    assert np.allclose(base,(corners(200,100)-[100,50])/200)
    forward = project_menu_plane((200,100),30,0,0)
    backward = project_menu_plane((200,100),-30,0,0)
    assert np.linalg.norm(forward[1]-forward[0])>np.linalg.norm(forward[2]-forward[3])
    assert np.linalg.norm(backward[1]-backward[0])<np.linalg.norm(backward[2]-backward[3])
    yaw = project_menu_plane((200,100),0,0,10)
    assert np.linalg.norm(yaw[2]-yaw[1])>np.linalg.norm(yaw[3]-yaw[0])
    roll = project_menu_plane((200,100),0,10,0)
    edge = roll[1]-roll[0]
    assert np.degrees(np.arctan2(edge[1],edge[0]))==pytest.approx(10)


def test_vertical_ctc_preserves_top_to_bottom_reading_order():
    image = Image.new('L',(64,384),255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((20,20,40,60),fill=0)
    draw.rectangle((10,300,50,350),fill=80)
    crop = extract_ctc_line(image,corners(64,384),1)
    assert crop.size==(384,64)
    assert np.array_equal(np.asarray(crop),np.asarray(image.transpose(Image.Transpose.ROTATE_90)))


def test_grouped_split_and_config_validation():
    a = grouped_split([f'restaurant-{i}' for i in range(10)],42)
    assert a==grouped_split(reversed(list(a)),42)
    assert list(a.values()).count('train')==8
    assert list(a.values()).count('validation')==list(a.values()).count('test')==1
    with pytest.raises(ValueError):
        grouped_split(['one','two'],42)
    with pytest.raises(ValueError):
        AugmentationConfig(max_pitch=90).validate()


def tiny_inputs(root):
    data = root/'input'
    menus = data/'genai/menu_images'
    menus.mkdir(parents=True)
    for i in range(3):
        path = menus/f'menu-{i+1:03d}.png'
        Image.new('RGB',(120,160),(150+i*20,180,210)).save(path)
        dump_json(path.with_suffix('.json'),dict(id=i+1,status='completed',image='menu_images/'+path.name,
            sha256=digest(path),intended_text=['焼き鳥定食','ラーメン定食','季節の刺身'][i],transcription_verified=False))
    yolo = data/'yolo_obb'
    pages = []
    for i,split in enumerate(('train','validation','test')):
        image = yolo/f'images/{split}/0000.png'
        image.parent.mkdir(parents=True)
        canvas = Image.new('RGB',(128,128),'white')
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((12,22,85,36),fill=(10+i*20,10,10))
        canvas.save(image)
        polygon = [[10,20],[90,20],[90,40],[10,40]]
        label = yolo/f'labels/{split}/0000.txt'
        label.parent.mkdir(parents=True)
        label.write_text('0 '+' '.join(str(v/128) for xy in polygon for v in xy)+'\n')
        pages.append(dict(image=image.relative_to(data).as_posix(),label=label.relative_to(data).as_posix(),
            split=split,items=[dict(class_id=0,text='焼き鳥定食',corners=polygon)]))
    dump_json(yolo/'pages.json',pages)
    backgrounds = data/'backgrounds'
    backgrounds.mkdir()
    for i in range(3):
        Image.new('RGB',(400,300),(50+i*30,120,80)).save(backgrounds/f'restaurant-{i}.jpg')
    return menus,yolo,backgrounds


def test_end_to_end_preserves_text_and_does_not_invent_generated_labels(tmp_path):
    menus,yolo,backgrounds = tiny_inputs(tmp_path)
    output = tmp_path/'augmentation'
    manifest = build(output,menus,yolo,backgrounds,AugmentationConfig(size=320,workers=2))
    assert manifest['menu_images']==manifest['composite_images']==6
    assert manifest['ctc_lines']==3
    assert not list(output.rglob('*.txt'))
    report = json.loads((output/'verification_report.json').read_text())
    assert report['status']=='passed'
    for row in manifest['samples']:
        metadata = json.loads((output/row['menu']['label']).read_text(encoding='utf8'))
        if row['id'].startswith('genai-'):
            assert metadata['annotations']==[] and metadata['text_verified'] is False
            assert metadata['intended_text']
        else:
            assert metadata['annotations'][0]['text']=='焼き鳥定食'
    # The same sample seed must produce identical image bytes with another
    # worker count; thread scheduling must not change the training corpus.
    repeat = tmp_path/'augmentation-repeat'
    second = build(repeat,menus,yolo,backgrounds,AugmentationConfig(size=320,workers=1))
    for first,other in zip(manifest['samples'],second['samples']):
        for kind in ('menu','composite'):
            assert digest(output/first[kind]['image'])==digest(repeat/other[kind]['image'])
    with pytest.raises(FileExistsError):
        build(output,menus,yolo,backgrounds,AugmentationConfig(size=320,workers=1))


def test_audit_rejects_text_corruption(tmp_path):
    from scripts.data.verify_menu_augmentation import verify
    menus,yolo,backgrounds = tiny_inputs(tmp_path)
    output = tmp_path/'augmentation'
    manifest = build(output,menus,yolo,backgrounds,AugmentationConfig(size=320,workers=1,export_ctc=False))
    row = next(r for r in manifest['samples'] if r['id'].startswith('synthetic-'))
    path = output/row['menu']['label']
    metadata = json.loads(path.read_text(encoding='utf8'))
    metadata['annotations'][0]['text']='違う文字列'
    dump_json(path,metadata)
    with pytest.raises(ValueError,match='Transcription'):
        verify(output)


def test_exact_count_works_below_and_above_source_count(tmp_path):
    menus,yolo,backgrounds = tiny_inputs(tmp_path)
    for count in (5,8):
        output = tmp_path/f'augmentation-{count}'
        manifest = build(output,menus,yolo,backgrounds,
            AugmentationConfig(size=320,workers=2,export_ctc=False),count=count)
        assert manifest['menu_images']==manifest['composite_images']==count
        assert len({r['id'] for r in manifest['samples']})==count
        assert manifest['requested_samples']==count
        assert manifest['available_source_menus']==6
        for row in manifest['samples']:
            label = json.loads((output/row['composite']['label']).read_text(encoding='utf8'))
            assert .5<=label['menu_scale']<=1.2
            assert label['background_selection']=='seeded_uniform_within_split'
