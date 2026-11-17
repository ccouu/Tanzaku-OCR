# L2 V01 — latest completed research checkpoint

`epoch_08.pt` is the original, unmodified Phase2 epoch8 checkpoint, stored in **Git LFS**. It contains separate `raw` and `ema` states plus training-resume metadata. It is a research artifact, **not a production promotion**. Production registry V43 is unchanged; the V43 weights are not included here.

After cloning, run `git lfs pull`. Verify the downloaded file against `metadata.json`. Training images, crop datasets, fonts and experiment bundles are excluded from Git. `vocab.json` is the checkpoint's required character-to-ID mapping, not a training dataset.

```python
import json
import torch
from src.recognition.model import ARCH_SPECS, MenuRecognizer

vocab = json.load(open("checkpoints/l2_v01/vocab.json", encoding="utf-8"))
# Load only a trusted checkpoint from this repository's verified LFS artifact.
checkpoint = torch.load("checkpoints/l2_v01/epoch_08.pt",
                        map_location="cpu", weights_only=False)
model = MenuRecognizer(classes=len(vocab["characters"]) + 1,
                       pretrained=False, use_bilstm=True,
                       **ARCH_SPECS["h4_k41"])
model.load_state_dict(checkpoint["raw"], strict=True)  # or checkpoint["ema"]
model.eval()
```

Inputs are H64 ImageNet-normalized RGB, appropriate orientation flags and valid sequence lengths `ceil(valid_width/4)`. Native vertical glyph columns are rotated counterclockwise once before recognition with flag1. This example loads weights; it does not silently resize arbitrary camera images or implement long-line splitting.

See [dataset/training history](../../docs/stage3_recognition/l2_v01_dataset_training_summary.md) and [model family lineage](../../docs/stage3_recognition/model_lineage.md). Epoch6 RAW remains a separate useful camera-adaptation candidate; epoch8 is supplied as the latest completed state, not a universal best.
