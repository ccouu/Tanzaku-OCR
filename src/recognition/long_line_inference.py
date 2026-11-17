"""H64 long-line routing and source-coordinate CTC overlap ownership."""
import math
import numpy as np
import torch
from PIL import Image
from .data import preprocess


def native_strip(image):
    image=image.convert('RGB')
    return image.resize((max(1,round(image.width*64/image.height)),64),Image.Resampling.BICUBIC)


def bucket_image(image,max_width=768):
    """Aspect-preserving fit, with explicit letterboxing when max_width is exceeded."""
    scale=min(64/image.height,max_width/image.width)
    w=max(1,min(max_width,round(image.width*scale)));h=max(1,min(64,round(image.height*scale)))
    resized=image.convert('RGB').resize((w,h),Image.Resampling.BICUBIC)
    canvas=next(c for c in (128,256,384,512,768) if c>=w and c<=max_width)
    top=(64-h)//2
    arr=np.pad(np.asarray(resized),((top,64-h-top),(0,canvas-w),(0,0)),mode='edge')
    return Image.fromarray(arr),dict(valid_width=w,canvas=canvas,input_length=(w+3)//4,scaled_height=h)


def window_ranges(width,max_width=512,overlap=100):
    """Balance valid window widths; overlap is in H64-normalized pixels."""
    if width<1 or not 0<=overlap<max_width:raise ValueError('Invalid width or overlap')
    n=max(1,math.ceil((width-overlap)/(max_width-overlap)))
    size=math.ceil((width+(n-1)*overlap)/n)
    return [(j*(size-overlap),min(width,j*(size-overlap)+size)) for j in range(n)]


def ctc_tokens(path):
    """Keep a token's complete argmax run, including legitimate repeated letters."""
    result=[];start=0
    while start<len(path):
        end=start+1
        while end<len(path) and path[end]==path[start]:end+=1
        if path[start]:result.append((int(path[start]),start,end))
        start=end
    return result


def owned_tokens(paths,ranges):
    """Assign overlap runs to the window owning their source-coordinate center."""
    if len(paths)!=len(ranges):raise ValueError('Path/window count mismatch')
    seams=[(a[1]+b[0])/2 for a,b in zip(ranges,ranges[1:])]
    result=[]
    for i,(path,(left,right)) in enumerate(zip(paths,ranges)):
        lo=seams[i-1] if i else -math.inf;hi=seams[i] if i<len(seams) else math.inf
        for token,start,end in ctc_tokens(path):
            x=left+2*(start+end)
            if lo<=x<hi:result.append(token)
    return result


def predict_long(model,image,cfg,mapping,device,precision='fp16',mode='wide768',orientation=0):
    strip=native_strip(image);ranges=window_ranges(strip.width) if mode=='overlap512' else [(0,strip.width)]
    paths=[]
    with torch.inference_mode():
        for left,right in ranges:
            fitted,meta=bucket_image(strip.crop((left,0,right,64)),512 if mode=='overlap512' else 768 if mode=='wide768' else 512)
            tensor,_,_=preprocess(fitted,cfg['mean'],cfg['std'],'batch',max_width=meta['canvas'])
            lengths=torch.tensor([meta['input_length']])
            with torch.autocast('cuda',dtype=torch.float16 if precision=='fp16' else torch.bfloat16):
                logits=model(tensor[None].to(device),torch.tensor([orientation],device=device),lengths)
            paths.append(logits[:meta['input_length'],0].argmax(-1).cpu().tolist())
    tokens=owned_tokens(paths,ranges) if mode=='overlap512' else [c for c,_,_ in ctc_tokens(paths[0])]
    reverse={v:k for k,v in mapping.items()}
    return ''.join(reverse[c] for c in tokens)
