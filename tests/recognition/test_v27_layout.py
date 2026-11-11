"""Runtime dependency and data-integrity checks for the V27-only tree."""
import ast
import json
from pathlib import Path

import torch

from src.recognition.data import collate,minimum_ctc_steps
from src.recognition.datasets import FontLines
from src.recognition.utils import sha


def test_active_modules_have_no_experiment_import_dependencies():
    root=Path('src/recognition');modules={p.stem for p in root.glob('*.py')}
    for path in root.glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node,ast.ImportFrom) and node.level:
                assert node.module.split('.')[0] in modules,(path,node.module)


def test_v27_font_manifest_and_sample_alignment():
    root=Path('data/datasets/recognition/v27_fonts')
    manifest=json.loads((root/'manifest.json').read_text())
    assert sha(root/'labels.jsonl')==manifest['labels_sha256']
    checkpoint=torch.load('outputs/recognition/training/v27_font_full20/epoch_20.pt',map_location='cpu',weights_only=False)
    cfg=checkpoint['config'];ds=FontLines(root,'seen',cfg['mean'],cfg['std'])
    sample=ds[0];batch=collate([sample])
    assert sample[0].shape[-1] in (128,256,384)
    assert batch['input_lengths'][0]>0 and batch['input_lengths'][0]>=minimum_ctc_steps(sample[4])
    assert ds.mapping==checkpoint['vocab']['char_to_id']


def test_frozen_transfer_inputs_are_local_and_unchanged():
    root=Path('data/datasets/recognition/v27_validation')
    transfer=json.loads((root/'transfer.json').read_text(encoding='utf-8'))
    assert len(transfer['yolo_provisional'])==38 and len(transfer['strong_appearance'])==100
    for rows in transfer.values():
        for row in rows:
            path=Path(row['image'])
            assert path.resolve().is_relative_to(root.resolve())
            assert sha(path)==row['sha256']
