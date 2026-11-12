"""Deterministic positive edge padding and mild horizontal aspect stretch."""
import random
import numpy as np
from PIL import Image
from src.detection.line_crops import recognition_input


def jitter_parameters(size,seed):
    w,h=size;rng=random.Random(seed)
    ratios={side:rng.uniform(.02,.05) for side in ('left','right','top','bottom')}
    pixels={side:max(1,round(ratios[side]*(w if side in ('left','right') else h))) for side in ratios}
    stretch=rng.uniform(.95,1.05)
    stretched_width=max(1,round((w+pixels['left']+pixels['right'])*stretch))
    padded_height=h+pixels['top']+pixels['bottom']
    scale=min(64/padded_height,512/stretched_width)
    valid=min(512,max(1,round(stretched_width*scale)));height=min(64,max(1,round(padded_height*scale)))
    canvas=next(c for c in (128,256,384,512) if valid<=c)
    return dict(seed=seed,source_size=[w,h],margin_ratios=ratios,margin_pixels=pixels,horizontal_stretch=stretch,
                stretched_width=stretched_width,padded_height=padded_height,valid_width=valid,canvas=canvas,
                input_length=(valid+3)//4,scaled_height=height,vertical_letterbox=height<64)


def apply_crop_jitter(strip,params):
    strip=strip.convert('RGB')
    if list(strip.size)!=params['source_size']:raise ValueError('Native crop size changed')
    p=params['margin_pixels']
    padded=np.pad(np.asarray(strip),((p['top'],p['bottom']),(p['left'],p['right']),(0,0)),mode='edge')
    im=Image.fromarray(padded).resize((params['stretched_width'],params['padded_height']),Image.Resampling.BICUBIC)
    normalized,meta=recognition_input(im)
    for key in ('valid_width','canvas','input_length','scaled_height','vertical_letterbox'):assert meta[key]==params[key]
    return normalized,meta
