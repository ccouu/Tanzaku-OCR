"""MobileNetV4 spatial encoder with an orientation-conditioned CTC head."""
import torch
from torch import nn
from torch.nn import functional as F

BACKBONE = 'mobilenetv4_conv_small.e2400_r224_in1k'


class ChannelLayerNorm(nn.Module):
    """Normalize channels independently at each time, never across padded time."""
    def __init__(self,channels,eps=1e-5):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(channels))
        self.bias=nn.Parameter(torch.zeros(channels))
        self.eps=eps

    def forward(self,x):
        return F.layer_norm(x.transpose(1,2),(x.shape[1],),self.weight,self.bias,self.eps).transpose(1,2)


def temporal_normalization(channels,kind):
    if kind=='batch':
        return nn.BatchNorm1d(channels)
    if kind=='layer':
        return ChannelLayerNorm(channels)
    raise ValueError('Unknown temporal normalization')


class TemporalBlock(nn.Module):
    def __init__(self, channels=256, norm='batch'):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2, groups=channels, bias=False),
            temporal_normalization(channels,norm), nn.ReLU(),
            nn.Conv1d(channels, channels, 1, bias=False), temporal_normalization(channels,norm))

    def forward(self, x):
        return torch.relu(x + self.body(x))


class ResidualBiLSTM(nn.Module):
    """Packed recurrent residual; zero projection preserves the pretrained path."""
    def __init__(self, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            256, 128, num_layers=num_layers, bidirectional=True, batch_first=True, dropout=0.0
        )
        self.output = nn.Linear(256, 256)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, lengths):
        if lengths is None:
            raise ValueError('BiLSTM requires valid feature sequence lengths')
        lengths = torch.as_tensor(lengths).detach().cpu()
        if (lengths.ndim != 1 or len(lengths) != features.shape[0]
                or lengths.dtype not in (torch.int32, torch.int64)
                or torch.any(lengths < 1) or torch.any(lengths > features.shape[-1])):
            raise ValueError('Invalid recurrent sequence lengths')
        # FP32 recurrent computation is portable across CUDA packed-LSTM kernels.
        with torch.autocast(features.device.type, enabled=False):
            packed = nn.utils.rnn.pack_padded_sequence(
                features.transpose(1, 2).float(), lengths, batch_first=True, enforce_sorted=False)
            encoded, _ = self.lstm(packed)
            encoded, _ = nn.utils.rnn.pad_packed_sequence(
                encoded, batch_first=True, total_length=features.shape[-1])
            residual = self.output(encoded).transpose(1, 2)
            valid = torch.arange(features.shape[-1], device=features.device)[None, :] < lengths.to(features.device)[:, None]
            residual = residual * valid[:, None, :]
        return features + residual.to(features.dtype)


ARCH_SPECS = {
    'v60_ref': {
        'stage_strides': ((2, 2), (2, 2), (2, 1), (2, 1), (2, 1)),
        'expected_h': 2,
        'height_pool': 'conv2d',
        'bilstm_layers': 1,
    },
    'v60_delay_s4_random10': {
        'stage_strides': ((2, 2), (2, 1), (2, 1), (2, 1), (2, 2)),
        'expected_h': 2,
        'height_pool': 'conv2d',
        'bilstm_layers': 1,
    },
    'h4_k41': {
        'stage_strides': ((2, 2), (2, 1), (2, 1), (2, 1), (1, 2)),
        'expected_h': 4,
        'height_pool': 'conv2d_h4',
        'bilstm_layers': 1,
    },
    'v60_h4_k41_random10': {
        'stage_strides': ((2, 2), (2, 1), (2, 1), (2, 1), (1, 2)),
        'expected_h': 4,
        'height_pool': 'conv2d_h4',
        'bilstm_layers': 1,
    },
}


class MenuRecognizer(nn.Module):
    def __init__(
        self,
        classes,
        pretrained=False,
        temporal=True,
        orientation=True,
        temporal_norm='batch',
        use_bilstm=False,
        height_pool='mean',
        stage_strides=((2, 2), (2, 2), (2, 1), (2, 1), (2, 1)),
        expected_h=2,
        bilstm_layers=1,
    ):
        super().__init__()
        import timm
        if height_pool not in ('mean', 'conv2d', 'conv2d_h4'):
            raise ValueError(f'Unsupported height_pool={height_pool!r}')
        if len(stage_strides) != 5:
            raise ValueError(f'Expected 5 stage_strides, got {stage_strides}')
        self.height_pool = height_pool
        self.expected_h = int(expected_h)
        self.stage_strides = tuple(tuple(s) for s in stage_strides)
        self.encoder = timm.create_model(BACKBONE, pretrained=pretrained, num_classes=0)
        # Enumerate the actual spatial downsampling convolutions, not stage aliases.
        downsamples = [(name, layer) for name, layer in self.encoder.named_modules()
                       if isinstance(layer, nn.Conv2d) and tuple(layer.stride) == (2, 2)]
        if len(downsamples) != 5:
            raise ValueError(f'Unexpected backbone downsamplers: {[n for n, _ in downsamples]}')
        self.stride_changes = []
        for (name, layer), target_stride in zip(downsamples, self.stage_strides):
            before = tuple(layer.stride)
            layer.stride = tuple(target_stride)
            if tuple(target_stride) != before:
                self.stride_changes.append(name)
        if self.encoder.num_features != 960:
            raise ValueError('Expected 960 spatial feature channels')
        # forward_features excludes these ImageNet classification head modules.
        self.encoder.conv_head = nn.Identity()
        self.encoder.norm_head = nn.Identity()
        self.encoder.act2 = nn.Identity()
        if height_pool == 'conv2d':
            self.projection = nn.Conv2d(960, 256, kernel_size=(2, 1), stride=(2, 1), padding=0, bias=True)
        elif height_pool == 'conv2d_h4':
            self.projection = nn.Conv2d(960, 256, kernel_size=(4, 1), stride=(4, 1), padding=0, bias=True)
        else:
            self.projection = nn.Conv1d(960, 256, 1)
        self.temporal_norm=temporal_norm
        self.use_orientation = orientation
        self.condition = nn.Sequential(nn.Conv1d(257, 256, 1), nn.ReLU())
        self.temporal = nn.Sequential(TemporalBlock(norm=temporal_norm), TemporalBlock(norm=temporal_norm)) if temporal else nn.Identity()
        self.classifier = nn.Conv1d(256, classes, 1)
        self.recurrent = ResidualBiLSTM(num_layers=bilstm_layers) if use_bilstm else None

    def forward(self, images, orientation, lengths=None):
        feat_4d = self.encoder.forward_features(images)
        if self.height_pool in ('conv2d', 'conv2d_h4'):
            if feat_4d.shape[2] != self.expected_h:
                raise ValueError(
                    f'Expected encoder height H={self.expected_h} before Conv2d projection, got {tuple(feat_4d.shape)}'
                )
            features = self.projection(feat_4d).squeeze(2)
        else:
            features = self.projection(feat_4d.mean(dim=2))
        flag = orientation.to(features.dtype).reshape(-1, 1, 1).expand(-1, 1, features.shape[-1])
        if not self.use_orientation:
            flag = torch.zeros_like(flag)
        features = self.condition(torch.cat((features, flag), dim=1))
        features = self.temporal(features)
        if self.recurrent is not None:
            features = self.recurrent(features, lengths)
        return self.classifier(features).permute(2, 0, 1)

    def freeze_encoder(self, frozen):
        self.encoder.requires_grad_(not frozen)
        if frozen:
            self.encoder.eval()


def load_recognizer_state(model,state,initialize_recurrent=False):
    """Strict load, explicitly dropping only migrated temporal BN buffers for LN."""
    state=dict(state)
    removed=[]
    for name,layer in model.named_modules():
        if isinstance(layer,ChannelLayerNorm):
            for suffix in ('running_mean','running_var','num_batches_tracked'):
                key=name+'.'+suffix
                if key in state:
                    state.pop(key);removed.append(key)
    if initialize_recurrent:
        if getattr(model,'recurrent',None) is None or any(k.startswith('recurrent.') for k in state):
            raise ValueError('Recurrent initialization requires a base checkpoint and a recurrent model')
        # Supply only the new branch keys; every existing key remains strict.
        state.update({k:v for k,v in model.state_dict().items() if k.startswith('recurrent.')})
    model.load_state_dict(state,strict=True)
    return removed
