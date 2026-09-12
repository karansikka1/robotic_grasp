"""BC temporal encoders and architecture-aware checkpoint loading."""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from v1.model import ACTION_DIM, PrivilegedPPOPolicy
from v2.model import TemporalPPOPolicy


class FrozenResNetFeatures(nn.Module):
    """ImageNet ResNet-18 retaining a configurable spatial grid."""

    def __init__(self, pretrained=True, pool_size=1):
        super().__init__()
        if pool_size not in (1, 2, 4):
            raise ValueError('RGB pool size must be 1, 2, or 4')
        network = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.pool_size = pool_size
        self.output_dim = network.fc.in_features * pool_size ** 2
        self.features = nn.Sequential(*list(network.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.requires_grad_(False)

    def forward(self, images):
        features = self.features(images)
        if min(features.shape[-2:]) < self.pool_size:
            raise ValueError('Image resolution is too small for the requested spatial grid')
        return self.pool(features).flatten(1)


class TemporalTransformer(nn.Module):
    """Causal attention over frame embeddings with learned temporal positions."""

    def __init__(self, dimension, history_length, heads, layers, feedforward_dim, dropout):
        super().__init__()
        self.positions = nn.Embedding(history_length, dimension)
        nn.init.normal_(self.positions.weight, std=0.02)
        # Construct layers independently rather than cloning identical weights.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dimension, nhead=heads, dim_feedforward=feedforward_dim,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
            ) for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(dimension)
        self.register_buffer('causal_mask', torch.ones(history_length, history_length,
                                                     dtype=torch.bool).triu(1), persistent=False)

    def forward(self, frames):
        if frames.shape[-2] != self.positions.num_embeddings:
            raise ValueError('Feature history length does not match the transformer configuration')
        values = frames + self.positions.weight
        for layer in self.layers:
            values = layer(values, src_mask=self.causal_mask)
        return self.norm(values)


class TransformerTemporalPolicy(TemporalPPOPolicy):
    """Existing visual/proprio features and actor, with a temporal transformer."""

    def __init__(self, *, transformer_heads=4, transformer_layers=2,
                 transformer_feedforward_dim=512, transformer_dropout=0.0,
                 rgb_backbone='mobilenet_v3_small', camera_fusion='sum', rgb_pool_size=1,
                 use_previous_action=True,
                 **kwargs):
        if rgb_backbone not in ('mobilenet_v3_small', 'resnet18', 'vc1_vitl'):
            raise ValueError('Unknown RGB backbone')
        if camera_fusion not in ('sum', 'concat'):
            raise ValueError('Camera fusion must be sum or concat')
        if rgb_pool_size not in (1, 2, 4) or (rgb_backbone == 'mobilenet_v3_small' and rgb_pool_size != 1):
            raise ValueError('Spatial RGB grids require ResNet-18 or VC-1 and pool size 1, 2, or 4')
        super().__init__(**kwargs)
        self.use_previous_action = bool(use_previous_action)
        self.rgb_backbone_name = rgb_backbone
        self.camera_fusion = camera_fusion
        self.rgb_pool_size = rgb_pool_size
        if rgb_backbone == 'resnet18':
            self.rgb_backbone = FrozenResNetFeatures(pretrained=kwargs.get('pretrained', True),
                                                     pool_size=rgb_pool_size)
        if rgb_backbone == 'vc1_vitl':
            from v4.vc1 import FrozenVC1Features
            self.rgb_backbone = FrozenVC1Features(pretrained=kwargs.get('pretrained', True), pool_size=rgb_pool_size)
        if rgb_backbone in ('resnet18', 'vc1_vitl') or camera_fusion == 'concat':
            self.rgb_projection = nn.Sequential(
                nn.Linear(self.rgb_backbone.output_dim * (2 if camera_fusion == 'concat' else 1),
                          self.embedding_dim), nn.ReLU(),
            )
            nn.init.orthogonal_(self.rgb_projection[0].weight, gain=2**0.5)
            nn.init.zeros_(self.rgb_projection[0].bias)
        if transformer_heads < 1 or self.embedding_dim % transformer_heads:
            raise ValueError('embedding_dim must be divisible by positive transformer_heads')
        if transformer_layers < 1 or transformer_feedforward_dim < 1:
            raise ValueError('Transformer layer count and feedforward dimension must be positive')
        if not 0 <= transformer_dropout < 1:
            raise ValueError('transformer_dropout must be in [0, 1)')
        self.transformer_config = dict(
            transformer_heads=transformer_heads, transformer_layers=transformer_layers,
            transformer_feedforward_dim=transformer_feedforward_dim,
            transformer_dropout=transformer_dropout,
        )
        self.temporal_encoder = TemporalTransformer(
            self.embedding_dim, self.history_length, transformer_heads, transformer_layers,
            transformer_feedforward_dim, transformer_dropout,
        )
        self.temporal_fusion = nn.Sequential(
            nn.Linear(self.embedding_dim + ACTION_DIM, self.embedding_dim), nn.Tanh(),
        )
        nn.init.orthogonal_(self.temporal_fusion[0].weight, gain=2**0.5)
        nn.init.zeros_(self.temporal_fusion[0].bias)

    def _rgb_tensor(self, image):
        if self.rgb_backbone_name == 'vc1_vitl':
            from v4.vc1 import preprocess_vc1
            return preprocess_vc1(image, self.device)
        return super()._rgb_tensor(image)

    def get_model_config(self):
        return {**super().get_model_config(), **self.transformer_config,
                'rgb_backbone': self.rgb_backbone_name, 'camera_fusion': self.camera_fusion,
                'rgb_pool_size': self.rgb_pool_size, 'use_previous_action': self.use_previous_action}

    def fused_embedding(self, features):
        if self.camera_fusion == 'concat':
            rgb = self.rgb_projection(torch.cat((features['front'], features['wrist']), dim=-1))
            frames = self.fusion_norm(rgb + self.depth_projection(features['depth'])
                                      + self.proprio_encoder(features['proprio']))
        else:
            frames = PrivilegedPPOPolicy.fused_embedding(self, features)
        # The newest token can attend to all past frames in the supplied window.
        latest = self.temporal_encoder(frames)[..., -1, :]
        # Mask at the shared model boundary for both cached training and online inference.
        # Keep cache labels intact so gripper-switch diagnostics remain meaningful.
        previous = features['previous_action']
        if not self.use_previous_action:
            previous = torch.zeros_like(previous)
        return self.temporal_fusion(torch.cat((latest, previous), dim=-1))


POLICY_CLASSES = {'single': PrivilegedPPOPolicy, 'history': TemporalPPOPolicy,
                  'transformer': TransformerTemporalPolicy}


def policy_kind(policy):
    if isinstance(policy, TransformerTemporalPolicy):
        return 'transformer'
    return 'history' if isinstance(policy, TemporalPPOPolicy) else 'single'


def load_policy(checkpoint_path, *, device='cpu'):
    """Load v4 BC models, including earlier single-frame and history checkpoints."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['model_config']
    kind = checkpoint.get('policy_type', 'history' if 'history_length' in config else 'single')
    if kind == 'spatial_lstm':
        from v4.recurrent import SpatialLSTMPolicy
        policy_class = SpatialLSTMPolicy
    else:
        policy_class = POLICY_CLASSES[kind]
    policy = policy_class(pretrained=False, **config)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    return policy.to(device).eval()
