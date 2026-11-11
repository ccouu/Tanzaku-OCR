"""V27 image normalization, valid CTC lengths, collation and decoding."""
import numpy as np
import torch
from PIL import Image

def output_length(width):
    return (int(width) + 3) // 4


def minimum_ctc_steps(text):
    return len(text) + sum(a == b for a, b in zip(text, text[1:]))


def preprocess(image, mean, std, padding='bucket', max_width=384):
    if max_width < 1 or (padding == 'bucket' and max_width != 384):
        raise ValueError('Extended widths require batch padding')
    image = image.convert('RGB')
    width = max(1, round(image.width * 64 / image.height))
    compressed = width > max_width
    width = min(width, max_width)
    image = image.resize((width, 64), Image.Resampling.BICUBIC)
    if padding not in ('bucket', 'batch'):
        raise ValueError('Unknown recognition padding policy')
    bucket = (192 if width <= 192 else 384) if padding == 'bucket' else width
    arr = np.asarray(image)
    border = np.concatenate((arr[0], arr[-1], arr[:, 0], arr[:, -1]))
    fill = np.median(border, axis=0).astype(np.uint8)
    padded = np.empty((64, bucket, 3), dtype=np.uint8)
    padded[:] = fill
    padded[:, :width] = arr
    tensor = torch.from_numpy(padded.copy()).permute(2, 0, 1).float().div_(255)
    tensor = (tensor - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]
    return tensor, width, compressed


def collate(samples, padded_width=None):
    max_width = max(s[0].shape[-1] for s in samples)
    if padded_width is not None:
        if not isinstance(padded_width,int) or padded_width < max_width:
            raise ValueError('Physical padded width must contain every sample')
        max_width = padded_width
    images = []
    for image, *_ in samples:
        if image.shape[-1] < max_width:
            image = torch.cat((image, image[:, :, -1:].expand(-1, -1, max_width-image.shape[-1])), dim=2)
        images.append(image)
    return dict(images=torch.stack(images), targets=torch.cat([s[1] for s in samples]),
                lengths=torch.tensor([len(s[1]) for s in samples], dtype=torch.long),
                orientation=torch.tensor([s[2] for s in samples]),
                input_lengths=torch.tensor([output_length(s[3]) for s in samples]),
                texts=[s[4] for s in samples], compressed=[s[5] for s in samples])


def decode(logits, mapping, lengths=None):
    reverse = {v: k for k, v in mapping.items()}
    paths = logits.argmax(-1).transpose(0, 1).cpu().tolist()
    texts = []
    for i, path in enumerate(paths):
        result, previous = [], None
        for token in path[:int(lengths[i]) if lengths is not None else len(path)]:
            if token != previous and token != 0:
                result.append(reverse[token])
            previous = token
        texts.append(''.join(result))
    return texts


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        following = [i]
        for j, cb in enumerate(b, 1):
            following.append(min(following[-1]+1, row[j]+1, row[j-1]+(ca != cb)))
        row = following
    return row[-1]
