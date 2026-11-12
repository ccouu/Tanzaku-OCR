import numpy as np
from PIL import Image
from src.recognition.crop_jitter import jitter_parameters,apply_crop_jitter
from src.recognition.vocabulary import split_existing_classifier,fused_state
from src.recognition.retention_distillation import teacher_kl
import torch
from torch import nn


def test_crop_jitter_is_deterministic_positive_padding_and_bucketed():
    im=Image.new('RGB',(170,45),(40,80,120));p=jitter_parameters(im.size,42)
    assert p==jitter_parameters(im.size,42)
    assert all(.02<=x<=.05 for x in p['margin_ratios'].values())
    assert all(x>0 for x in p['margin_pixels'].values()) and .95<=p['horizontal_stretch']<=1.05
    a,meta=apply_crop_jitter(im,p);b,_=apply_crop_jitter(im,p)
    assert np.array_equal(a,b) and a.size==(meta['canvas'],64)
    assert meta['input_length']==(meta['valid_width']+3)//4 and not meta['glyph_clipped']


def test_extreme_long_crop_fits_without_clipping():
    im=Image.new('RGB',(2000,40),'white');p=jitter_parameters(im.size,55)
    out,meta=apply_crop_jitter(im,p)
    assert out.size==(512,64) and meta['valid_width']==512 and meta['scaled_height']<64


def test_split_keeps_every_pretrained_classifier_row():
    full=nn.Conv1d(4,7,1);m=nn.Module();m.classifier=split_existing_classifier(full,5)
    exported=fused_state(m)
    assert torch.equal(exported['classifier.weight'],full.weight)
    assert torch.equal(exported['classifier.bias'],full.bias)


def test_teacher_kl_excludes_padding_and_teacher_gradient():
    teacher=torch.randn(3,1,5,requires_grad=True);student=teacher.detach().clone();student[2]+=torch.arange(5)*10;student.requires_grad_()
    loss=teacher_kl(student,teacher,torch.tensor([2]));loss.backward()
    assert abs(float(loss.detach()))<1e-6 and teacher.grad is None and not student.grad[2].any()
