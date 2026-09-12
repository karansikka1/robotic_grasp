"""Official VC-1 ViT-L encoder weights in existing torchvision modules.

Architecture/preprocessing reference: facebookresearch/eai-vc.
Pinned weights: facebook/vc1-large revision 8a47f311ef3a8e0b3f58e4249700c9c5b36012c9.
No timm, hydra, or omegaconf runtime dependency. Pool size 1 uses the official
CLS embedding; sizes 2/4 optionally pool the normalized spatial patch outputs.
"""
from functools import partial
import os
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import VisionTransformer
from torchvision.transforms import functional as TF, InterpolationMode


REVISION = '8a47f311ef3a8e0b3f58e4249700c9c5b36012c9'


def convert_encoder_weights(source):
    """Convert timm VC-1 names; discard only the published MAE decoder/mask."""
    result = {}
    direct = {'cls_token': 'class_token', 'pos_embed': 'encoder.pos_embedding',
              'patch_embed.proj.weight': 'conv_proj.weight', 'patch_embed.proj.bias': 'conv_proj.bias',
              'norm.weight': 'encoder.ln.weight', 'norm.bias': 'encoder.ln.bias'}
    parts = {'norm1': 'ln_1', 'norm2': 'ln_2', 'attn.qkv.weight': 'self_attention.in_proj_weight',
             'attn.qkv.bias': 'self_attention.in_proj_bias', 'attn.proj': 'self_attention.out_proj',
             'mlp.fc1': 'mlp.0', 'mlp.fc2': 'mlp.3'}
    for key, value in source.items():
        if key.startswith('decoder') or key == 'mask_token':
            continue
        if key in direct:
            target = direct[key]
        elif key.startswith('blocks.'):
            _, index, suffix = key.split('.', 2)
            if suffix in parts:
                suffix = parts[suffix]
            else:
                prefix, parameter = suffix.rsplit('.', 1)
                suffix = parts[prefix] + '.' + parameter
            target = f'encoder.layers.encoder_layer_{index}.{suffix}'
        else:
            raise ValueError(f'Unexpected VC-1 encoder tensor: {key}')
        result[target] = value
    return result


def preprocess_vc1(images, device):
    """MuJoCo flip, official bicubic short-edge resize and center crop, ImageNet norm."""
    images = torch.as_tensor(images, device=device)
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError('Expected RGB BHWC or HWC input')
    images = images.permute(0, 3, 1, 2).flip(-2)
    images = TF.resize(images, 256, interpolation=InterpolationMode.BICUBIC, antialias=True)
    images = TF.center_crop(images, [224, 224]).float().div(255)
    return TF.normalize(images, [.485, .456, .406], [.229, .224, .225])


class FrozenVC1Features(nn.Module):
    def __init__(self, pretrained=True, pool_size=1):
        super().__init__()
        if pool_size not in (1, 2, 4):
            raise ValueError('VC-1 pooling must be 1, 2, or 4')
        self.pool_size = pool_size
        self.output_dim = 1024 * pool_size ** 2
        self.encoder = VisionTransformer(image_size=224, patch_size=16, num_layers=24,
                                         num_heads=16, hidden_dim=1024, mlp_dim=4096,
                                         norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.encoder.heads = nn.Identity()
        if pretrained:
            path = Path(os.environ.get('VC1_CHECKPOINT', '/workspace/.cache/vc1-large/pytorch_model.bin'))
            source = torch.load(path, map_location='cpu', weights_only=True)['model']
            self.encoder.load_state_dict(convert_encoder_weights(source), strict=True)
        self.requires_grad_(False)

    def tokens(self, images):
        tokens = self.encoder._process_input(images)
        cls = self.encoder.class_token.expand(len(images), -1, -1)
        return self.encoder.encoder(torch.cat((cls, tokens), dim=1))

    def forward(self, images):
        tokens = self.tokens(images)
        if self.pool_size == 1:
            return tokens[:, 0]
        spatial = tokens[:, 1:].transpose(1, 2).reshape(-1, 1024, 14, 14)
        return F.adaptive_avg_pool2d(spatial, self.pool_size).flatten(1)
