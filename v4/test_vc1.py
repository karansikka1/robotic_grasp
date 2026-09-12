"""Strict VC-1 conversion, reference attention, and image preprocessing checks."""
from functools import partial
import unittest

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import VisionTransformer
from torchvision.transforms import functional as TF, InterpolationMode

from v4.vc1 import convert_encoder_weights, preprocess_vc1


def reference_tokens(images, source, heads):
    """Independent functional implementation of the official timm-style VC-1 path."""
    patches = F.conv2d(images, source['patch_embed.proj.weight'], source['patch_embed.proj.bias'], stride=16)
    tokens = patches.flatten(2).transpose(1, 2) + source['pos_embed'][:, 1:]
    cls = (source['cls_token'] + source['pos_embed'][:, :1]).expand(len(images), -1, -1)
    tokens = torch.cat((cls, tokens), dim=1)
    layers = sum(key.endswith('attn.qkv.weight') for key in source)
    dimension = tokens.shape[-1]
    for index in range(layers):
        prefix = f'blocks.{index}.'
        def norm(x, name):
            return F.layer_norm(x, (dimension,), source[prefix + name + '.weight'], source[prefix + name + '.bias'], 1e-6)
        def linear(x, name):
            return F.linear(x, source[prefix + name + '.weight'], source[prefix + name + '.bias'])
        qkv = linear(norm(tokens, 'norm1'), 'attn.qkv').reshape(len(images), tokens.shape[1], 3, heads, dimension // heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attention = ((q @ k.transpose(-2, -1)) * (dimension // heads) ** -.5).softmax(-1)
        values = (attention @ v).transpose(1, 2).reshape_as(tokens)
        tokens = tokens + linear(values, 'attn.proj')
        tokens = tokens + linear(F.gelu(linear(norm(tokens, 'norm2'), 'mlp.fc1')), 'mlp.fc2')
    return F.layer_norm(tokens, (dimension,), source['norm.weight'], source['norm.bias'], 1e-6)


class VC1Test(unittest.TestCase):
    def test_conversion_matches_reference_attention(self):
        torch.manual_seed(0)
        source = {'cls_token': torch.randn(1, 1, 32), 'pos_embed': torch.randn(1, 197, 32),
                  'patch_embed.proj.weight': torch.randn(32, 3, 16, 16) * .02,
                  'patch_embed.proj.bias': torch.randn(32) * .02,
                  'norm.weight': torch.ones(32), 'norm.bias': torch.zeros(32)}
        for index in range(2):
            for name, out_dim, in_dim in [('attn.qkv', 96, 32), ('attn.proj', 32, 32),
                                         ('mlp.fc1', 128, 32), ('mlp.fc2', 32, 128)]:
                source[f'blocks.{index}.{name}.weight'] = torch.randn(out_dim, in_dim) * .02
                source[f'blocks.{index}.{name}.bias'] = torch.randn(out_dim) * .02
            for name in ['norm1', 'norm2']:
                source[f'blocks.{index}.{name}.weight'] = torch.ones(32)
                source[f'blocks.{index}.{name}.bias'] = torch.zeros(32)
        model = VisionTransformer(image_size=224, patch_size=16, num_layers=2, num_heads=4,
                                  hidden_dim=32, mlp_dim=128, norm_layer=partial(nn.LayerNorm, eps=1e-6)).eval()
        model.heads = nn.Identity()
        model.load_state_dict(convert_encoder_weights(source), strict=True)
        images = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            actual = model.encoder(torch.cat((model.class_token.expand(2, -1, -1), model._process_input(images)), dim=1))
            expected = reference_tokens(images, source, 4)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        self.assertEqual(len(convert_encoder_weights({**source, 'mask_token': torch.zeros(1), 'decoder_pred.weight': torch.zeros(1)})), len(source))
        with self.assertRaises(ValueError):
            convert_encoder_weights({'unexpected': torch.zeros(1)})

    def test_native_transform_batch_matches_online_and_official_ops(self):
        images = torch.randint(256, (2, 256, 256, 3), dtype=torch.uint8)
        actual = preprocess_vc1(images, 'cpu')
        self.assertEqual(tuple(actual.shape), (2, 3, 224, 224))
        for index in range(2):
            torch.testing.assert_close(actual[index], preprocess_vc1(images[index], 'cpu')[0], rtol=0, atol=0)
            expected = images[index].permute(2, 0, 1).flip(-2)
            expected = TF.resize(expected, 256, interpolation=InterpolationMode.BICUBIC, antialias=True)
            expected = TF.center_crop(expected, [224, 224]).float() / 255
            expected = TF.normalize(expected, [.485, .456, .406], [.229, .224, .225])
            torch.testing.assert_close(actual[index], expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
