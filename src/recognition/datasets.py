"""V27 font masks, clean/historical retention views and guarded bucket layouts."""
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from .data import preprocess
from .utils import sha,fit_margins

BUCKETS=(128,256,384)

def canvas_for(width):
    if not isinstance(width,int) or width<1:raise ValueError('Width must be a positive integer')
    for canvas in BUCKETS:
        if width<=canvas:return canvas
    raise ValueError('Crop exceeds384; routing must not clip or resize it')

def canonical_tensor(parent,box,mean,std,canvas=384,guard=16):
    """Content pixels are fixed; real background guards precede edge extension."""
    if guard<1:raise ValueError('A real background guard is required')
    left,top,right,bottom=box
    if not (0<=left<right<=parent.width and 0<=top<bottom<=parent.height):raise ValueError('Invalid content box')
    head,tail,capped=fit_margins(right-left,guard,guard,384)
    if min(head,tail)<1:raise ValueError('Cannot add safe guards without changing content')
    arr=np.asarray(parent.convert('RGB'))
    border=np.concatenate((arr[0],arr[-1],arr[:,0],arr[:,-1]))
    fill=tuple(np.median(border,axis=0).astype(np.uint8).tolist())
    valid=Image.new('RGB',(head+right-left+tail,parent.height),fill)
    valid.paste(parent.crop((left,0,right,parent.height)).convert('RGB'),(head,0))
    if canvas<valid.width:raise ValueError('Physical canvas is smaller than valid content')
    padded=Image.fromarray(np.pad(np.asarray(valid),((0,0),(0,canvas-valid.width),(0,0)),mode='edge'))
    tensor,_,compressed=preprocess(padded,mean,std,'batch',max_width=canvas)
    if compressed:raise ValueError('Unexpected compression')
    return tensor,valid.width,dict(head=head,tail=tail,capped=capped,glyph_width=right-left)

class FontLines(Dataset):
    def __init__(self,root,split,mean,std,arm='clean'):
        self.root=root;self.mean=mean;self.std=std;self.arm=arm;self.view='fixed';self.verified=set()
        self.rows=[r for r in map(json.loads,(root/'labels.jsonl').read_text(encoding='utf-8').splitlines()) if r['split']==split]
        self.mapping=json.loads((root/'vocab.json').read_text(encoding='utf-8'))['char_to_id']

    def __len__(self):return len(self.rows)

    def __getitem__(self,key):
        if isinstance(key,int):i=key;canvas=None
        else:i,_,_,canvas=key
        row=self.rows[i];aug=self.arm=='morph';relative=row['aug_image'] if aug else row['image'];box=row['aug_box'] if aug else row['glyph_box']
        path=self.root/relative
        if i not in self.verified:
            if sha(path)!=(row['aug_sha256'] if aug else row['sha256']):raise ValueError('Mask hash changed')
            self.verified.add(i)
        with Image.open(path) as mask:arr=np.array(mask)[:,box[0]:box[2]]
        span=arr.shape[1];guard=4 if len(row['text'])==1 else 16
        if canvas is None:
            canvas=canvas_for(min(384,span+2*guard))
        head,tail,_=fit_margins(span,guard,guard,canvas)
        if min(head,tail)<1:raise ValueError('No safe background guards')
        bg=row['background'];ink=20 if bg==242 else 235
        pixels=np.rint(bg+(ink-bg)*(arr.astype(np.float32)/255)).clip(0,255).astype(np.uint8)
        valid=head+span+tail;image=Image.new('L',(canvas,64),bg);image.paste(Image.fromarray(pixels),(head,0))
        tensor,_,compressed=preprocess(image.convert('RGB'),self.mean,self.std,'batch')
        if compressed:raise ValueError('Unexpected glyph compression')
        return tensor,torch.tensor([self.mapping[c] for c in row['text']]),row['orientation_flag'],valid,row['text'],False

class CleanViews(Dataset):
    def __init__(self,root,split,mean,std,view='bucket'):
        self.root=root; self.mean=mean; self.std=std; self.view=view;self.verified=set()
        self.rows=[r for r in map(json.loads,(root/'labels.jsonl').read_text(encoding='utf-8').splitlines()) if r['split']==split]
        self.mapping=json.loads((root/'vocab.json').read_text(encoding='utf-8'))['char_to_id']

    def __len__(self): return len(self.rows)

    def __getitem__(self,key):
        if isinstance(key,int):
            i=key;row=self.rows[i];guard=4 if len(row['text'])==1 else 16
            canvas=canvas_for(min(384,row['glyph_box'][2]-row['glyph_box'][0]+2*guard))
            jitter=self.view=='jitter';seed=12345+i*104729
            if self.view!='bucket':canvas=384
        else:i,jitter,seed,canvas=key
        row=self.rows[i];path=self.root/row['image']
        if i not in self.verified:
            if sha(path)!=row['sha256']:raise ValueError('Clean image changed')
            self.verified.add(i)
        with Image.open(path) as im: parent=im.convert('RGB')
        if self.view=='native' and isinstance(key,int): tensor,width,_=preprocess(parent,self.mean,self.std,'batch')
        elif not jitter:
            tensor,width,_=canonical_tensor(parent,row['glyph_box'],self.mean,self.std,canvas=canvas,
                                           guard=4 if len(row['text'])==1 else 16)
        else:
            # A real right background guard prevents replicating a glyph's final column.
            left,_,right,_=row['glyph_box'];span=right-left
            tail=min(4,384-span);head=min(random.Random(seed).randint(0,32),384-span-tail)
            bg=parent.getpixel((0,0));valid=Image.new('RGB',(head+span+tail,64),bg)
            valid.paste(parent.crop((left,0,right,64)),(head,0));width=valid.width
            padded=Image.fromarray(np.pad(np.asarray(valid),((0,0),(0,384-width),(0,0)),mode='edge'))
            tensor,_,_=preprocess(padded,self.mean,self.std,'batch')
        return tensor,torch.tensor([self.mapping[c] for c in row['text']]),row['orientation_flag'],width,row['text'],False

class Minimum128(CleanViews):
    """Preserve V23 guards/pixels/lengths; change only 1/2-character canvas."""
    def __getitem__(self,key):
        i=key if isinstance(key,int) else key[0]
        if len(self.rows[i]['text'])<=2:
            sample=super().__getitem__((i,False,0,128))
            if sample[3]>128:raise ValueError('Cannot fit valid span without glyph resizing')
            return sample
        return super().__getitem__(key)

class GuardedLines(Dataset):
    def __init__(self,root,split,mean,std,crop_root=None,native=False):
        self.root=root;self.mean,self.std=mean,std;self.native=native
        self.rows=[r for r in map(json.loads,(root/'labels.jsonl').read_text(encoding='utf-8').splitlines()) if r['split']==split]
        self.crop_rows=([r for r in map(json.loads,(crop_root/'labels.jsonl').read_text(encoding='utf-8').splitlines()) if r['split']==split] if crop_root else self.rows)
        self.mapping=json.loads((root/'vocab.json').read_text(encoding='utf-8'))['char_to_id'];self.verified=set()
        if [(r['text'],r['orientation_flag']) for r in self.rows]!=[(r['text'],r['orientation_flag']) for r in self.crop_rows]:raise ValueError('Crop metadata mismatch')
        self.widths=[384 if native else self._guarded_width(r) for r in self.crop_rows]

    @staticmethod
    def _guarded_width(row):
        span=row['glyph_box'][2]-row['glyph_box'][0]
        return min(384,span+32)

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index];path=(self.root/row['image']).resolve()
        if not path.is_relative_to(self.root.resolve()):raise ValueError('Unsafe input path')
        if index not in self.verified:
            if sha(path)!=row['sha256']:raise ValueError('Image hash differs')
            self.verified.add(index)
        with Image.open(path) as im:image=im.convert('RGB')
        if self.native:
            tensor,width,compressed=preprocess(image,self.mean,self.std,'batch')
            if compressed:raise ValueError('Unexpected resizing')
        else:tensor,width,_=canonical_tensor(image,self.crop_rows[index]['glyph_box'],self.mean,self.std)
        if width!=self.widths[index]:raise ValueError('Routing width differs from rendered width')
        return tensor,torch.tensor([self.mapping[c] for c in row['text']]),row['orientation_flag'],width,row['text'],False
