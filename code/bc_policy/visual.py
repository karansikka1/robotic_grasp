"""RGB BC models extracted from the experiments, without run-management dependencies."""
from collections import OrderedDict

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights

from bc_data.state_policy import FixedRotationHead, StageAuxStateLSTMPolicy, PROPRIO_KEYS
from .data import RGB_KEYS

INPUT_KEYS = RGB_POLICY_KEYS = (*RGB_KEYS, *PROPRIO_KEYS)



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


def rgb_observation(observation):
    return {key: observation[key] for key in RGB_POLICY_KEYS}


class RGBSpatialLSTMPolicy(nn.Module):
    def __init__(self, *, hidden_dim=128, image_size=128, pool_size=4, stage_classes=17, pretrained=False):
        super().__init__()
        self.hidden_dim, self.image_size = hidden_dim, image_size
        self.pool_size, self.stage_classes = pool_size, stage_classes
        self.rgb_backbone = FrozenResNetFeatures(pretrained=pretrained, pool_size=pool_size)
        self.rgb_projection = nn.Sequential(nn.Linear(2*self.rgb_backbone.output_dim, hidden_dim), nn.ReLU())
        self.proprio_encoder = nn.Sequential(nn.Linear(16, hidden_dim), nn.ReLU())
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.actor = FixedRotationHead(hidden_dim)
        self.stage_head = nn.Linear(hidden_dim, stage_classes)
        self.register_buffer('image_mean', torch.tensor([.485,.456,.406]).reshape(1,3,1,1))
        self.register_buffer('image_std', torch.tensor([.229,.224,.225]).reshape(1,3,1,1))
        self.register_buffer('proprio_mean', torch.zeros(16))
        self.register_buffer('proprio_scale', torch.ones(16))
        self.reset_history()
        self.train(False)

    @property
    def device(self):
        return self.proprio_mean.device

    def train(self, mode=True):
        super().train(mode)
        self.rgb_backbone.eval()
        return self

    def preprocess(self, images):
        values = torch.as_tensor(images, device=self.device)
        if values.ndim == 3: values = values.unsqueeze(0)
        if values.ndim != 4 or values.shape[-1] != 3: raise ValueError('Expected BHWC RGB')
        values = values.permute(0,3,1,2).float().div(255).flip(-2)
        values = F.interpolate(values, size=(self.image_size,self.image_size), mode='bilinear',
                               align_corners=False, antialias=True)
        return (values-self.image_mean)/self.image_std

    @torch.no_grad()
    def encode_rgb(self, front, wrist):
        self.rgb_backbone.eval()
        return {'front':self.rgb_backbone(self.preprocess(front)),
                'wrist':self.rgb_backbone(self.preprocess(wrist))}

    @torch.no_grad()
    def extract_frozen_features(self, observation):
        # Explicit whitelist: no depth, object coordinates, teacher state or prior actions.
        inputs = rgb_observation(observation)
        encoded = self.encode_rgb(inputs[RGB_KEYS[0]], inputs[RGB_KEYS[1]])
        encoded['proprio'] = torch.cat([torch.as_tensor(inputs[k],device=self.device).flatten()
                                       for k in PROPRIO_KEYS]).float().unsqueeze(0)
        return encoded

    def sequence_with_stage(self, features, hidden=None):
        rgb = self.rgb_projection(torch.cat((features['front'],features['wrist']),dim=-1))
        proprio = (features['proprio']-self.proprio_mean)/self.proprio_scale
        fused = self.fusion_norm(rgb+self.proprio_encoder(proprio))
        values, hidden = self.lstm(fused,hidden)
        return self.actor(values), hidden, self.stage_head(values)

    def sequence(self, features, hidden=None):
        logits, hidden, _ = self.sequence_with_stage(features,hidden)
        return logits,hidden

    def reset_history(self):
        self._hidden = None

    @torch.no_grad()
    def predict(self, observation, *, deterministic=True):
        if not deterministic: raise ValueError('Use deterministic BC inference')
        self.eval()
        features = {k:v.unsqueeze(0) for k,v in self.extract_frozen_features(observation).items()}
        logits,self._hidden = self.sequence(features,self._hidden)
        return logits[0,0].tanh().cpu().numpy()

    def get_model_config(self):
        return dict(hidden_dim=self.hidden_dim,image_size=self.image_size,
                    pool_size=self.pool_size,stage_classes=self.stage_classes)

    def sequence_images(self, images, proprio, hidden=None):
        batch, time = proprio.shape[:2]
        encoded = self.encode_rgb(*(images[key] for key in RGB_KEYS))
        features = {key: value.reshape(batch, time, -1) for key, value in encoded.items()}
        features['proprio'] = proprio
        return self.sequence_with_stage(features, hidden)


class ColorLocalizer(nn.Module):
    def __init__(self,arm='spatial_softmax',feature_channels=128,grid_size=16):
        super().__init__();self.arm=arm
        self.feature_channels,self.grid_size=feature_channels,grid_size
        self.register_buffer('proprio_mean',torch.zeros(16));self.register_buffer('proprio_scale',torch.ones(16))
        self.register_buffer('position_mean',torch.zeros(3,3));self.register_buffer('position_scale',torch.ones(3,3))
        if arm=='direct_4x4':
            width=16384+16
        elif arm=='spatial_softmax':
            self.heatmap_head=nn.Sequential(nn.Conv2d(feature_channels,32,1),nn.ReLU(),nn.Conv2d(32,3,1))
            self.visibility_head=nn.Linear(feature_channels,3)
            axis=(torch.arange(grid_size).float()+.5)/grid_size*2-1
            yy,xx=torch.meshgrid(axis,axis,indexing='ij')
            self.register_buffer('grid',torch.stack((xx,yy),dim=-1).reshape(grid_size*grid_size,2))
            width=2*3*(feature_channels+2+1)+16
        else:raise ValueError(arm)
        self.position_head=nn.Sequential(nn.Linear(width,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,9))

    def forward(self,features):
        proprio=(features['proprio']-self.proprio_mean)/self.proprio_scale
        extra={}
        if self.arm=='direct_4x4':
            values=torch.cat((features['front'].float(),features['wrist'].float(),proprio),dim=-1)
        else:
            maps=torch.stack((features['front'],features['wrist']),dim=1).float().reshape(-1,self.feature_channels,self.grid_size,self.grid_size)
            logits=self.heatmap_head(maps).flatten(2)
            probabilities=logits.softmax(-1)
            uv=probabilities@self.grid
            pooled=probabilities@maps.flatten(2).transpose(1,2)
            visibility=self.visibility_head(maps.mean((-2,-1)))
            values=torch.cat((pooled,uv,visibility.sigmoid().unsqueeze(-1)),dim=-1).reshape(len(proprio),-1)
            values=torch.cat((values,proprio),dim=-1)
            extra={'uv':uv.reshape(-1,2,3,2),'heatmap_logits':logits.reshape(-1,2,3,self.grid_size*self.grid_size),
                   'visibility_logits':visibility.reshape(-1,2,3)}
        normalized=self.position_head(values).reshape(-1,3,3)
        return {'positions':normalized*self.position_scale+self.position_mean,'normalized_positions':normalized,**extra}


class VisionTrunk(nn.Module):
    def __init__(self, backbone, prefix_half_quantization=False):
        super().__init__()
        self.prefix_half_quantization = bool(prefix_half_quantization)
        network={'resnet18':resnet18,'resnet34':resnet34,'resnet50':resnet50}[backbone](weights=None)
        self.features=nn.Sequential(*list(network.children())[:6])
        self.register_buffer('image_mean',torch.tensor([.485,.456,.406]).reshape(1,3,1,1))
        self.register_buffer('image_std',torch.tensor([.229,.224,.225]).reshape(1,3,1,1))
        self.finetune_stages=();self.requires_grad_(False);self.eval()

    def train(self,mode=True):
        super().train(False)
        return self

    def configure_finetuning(self,stages=()):
        if any(stage not in ('layer1','layer2') for stage in stages):raise ValueError('Unsupported CNN stage')
        self.finetune_stages=tuple(stages);self.requires_grad_(False)
        for stage in stages:self.features[{'layer1':4,'layer2':5}[stage]].requires_grad_(True)
        self.eval()  # Keep BatchNorm running statistics fixed, including trainable stages.

    def encode(self,observations):
        with torch.set_grad_enabled(torch.is_grad_enabled() and bool(self.finetune_stages)):
            return self._encode(observations)

    def _encode(self,observations):
        output={}
        for key,view in zip(RGB_KEYS,('front','wrist')):
            pixels=torch.as_tensor(observations[key],device=self.image_mean.device)
            if pixels.ndim==3:pixels=pixels.unsqueeze(0)
            assert pixels.ndim==4 and pixels.shape[-1]==3
            values=pixels.permute(0,3,1,2).float().div(255).flip(-2)
            values=F.interpolate(values,size=(128,128),mode='bilinear',align_corners=False,antialias=True)
            values=(values-self.image_mean)/self.image_std
            if self.prefix_half_quantization:
                # Matched CNN continuation trained layer2 after a float16 layer1 boundary.
                values=self.features[:5](values).half().float()
                values=self.features[5](values)
            else:
                values=self.features(values)
            output[view]=values.flatten(1).half()
        return output


class VisionStateLSTM(nn.Module):
    """Visual-feature policy with an optional frozen CNN geometry-input branch."""
    def __init__(self,config):
        super().__init__();self.config=config
        self.vision=VisionTrunk(config['backbone'], config.get('prefix_half_quantization', False))
        self.vision.configure_finetuning(config.get('bc_finetune_stages',()))
        channels=config['localizer']['feature_channels']
        self.spatial_head=nn.Sequential(nn.Conv2d(channels,32,1),nn.ReLU(),nn.Conv2d(32,3,1))
        c=config['controller'];hidden=c['hidden_dim'];grid=config['localizer']['grid_size']
        self.adapter=nn.Sequential(nn.Linear(2*(32+3)*grid*grid+16,hidden),nn.ReLU())
        source=StageAuxStateLSTMPolicy(torch.zeros(c['input_dim']),torch.ones(c['input_dim']),
            hidden_dim=hidden,lstm_layers=c['lstm_layers'],stage_classes=c['stage_classes'],
            action_dim=c['action_dim'],recurrent_activation=c.get('recurrent_activation','tanh'))
        self.lstm,self.actor,self.stage_head=source.lstm,source.actor,source.stage_head
        self.location_head=nn.Linear(hidden,9)
        self.register_buffer('proprio_mean',torch.zeros(16));self.register_buffer('proprio_scale',torch.ones(16))
        self.register_buffer('position_mean',torch.zeros(3,3));self.register_buffer('position_scale',torch.ones(3,3))
        if config.get('predicted_geometry_inputs',False):
            self.input_localizer=ColorLocalizer(**config['localizer']).requires_grad_(False)
            self.geometry_adapter=nn.Linear(18,hidden,bias=False)
            nn.init.zeros_(self.geometry_adapter.weight)
            self.register_buffer('offset_mean',torch.zeros(3,3));self.register_buffer('offset_scale',torch.ones(3,3))
        self.reset_history()

    @property
    def device(self):return self.proprio_mean.device

    def encode(self,observations):return self.vision.encode(observations)

    @torch.no_grad()
    def geometry_features(self,features,proprio):
        batch,time=proprio.shape[:2]
        self.input_localizer.eval()
        predicted=self.input_localizer({'front':features['front'].reshape(batch*time,-1),'wrist':features['wrist'].reshape(batch*time,-1),'proprio':proprio.reshape(batch*time,16)})['positions'].reshape(batch,time,3,3)
        offsets=predicted-proprio[...,7:10].unsqueeze(-2)
        absolute=(predicted-self.input_localizer.position_mean)/self.input_localizer.position_scale
        relative=(offsets-self.offset_mean)/self.offset_scale
        return torch.cat((absolute.flatten(-2),relative.flatten(-2)),dim=-1)

    def sequence_from_features(self,features,proprio,hidden=None):
        batch,time=proprio.shape[:2];channels=self.config['localizer']['feature_channels'];grid=self.config['localizer']['grid_size']
        maps=torch.stack([features[k].reshape(batch*time,channels,grid,grid) for k in ('front','wrist')],dim=1).reshape(-1,channels,grid,grid).float()
        appearance=self.spatial_head[1](self.spatial_head[0](maps))
        logits=self.spatial_head[2](appearance)
        probabilities=logits.flatten(2).softmax(-1).reshape(-1,3,grid,grid)
        # Preserve every spatial cell; optional geometry is a separate frozen-head readout.
        spatial=torch.cat((appearance,probabilities),dim=1).reshape(batch,time,-1)
        normalized=(proprio-self.proprio_mean)/self.proprio_scale
        inputs=torch.cat((spatial,normalized),dim=-1)
        if self.config.get('predicted_geometry_inputs',False):
            encoded=self.adapter[1](self.adapter[0](inputs)+self.geometry_adapter(self.geometry_features(features,proprio)))
        else:encoded=self.adapter(inputs)
        memory,hidden=self.lstm(encoded,hidden)
        # Auxiliary location outputs are downstream of the recurrent/action representation.
        positions=self.location_head(memory).reshape(batch,time,3,3)*self.position_scale+self.position_mean
        return self.actor(memory),hidden,self.stage_head(memory),positions

    def reset_history(self):self._hidden=None

    def get_model_config(self):
        return self.config

    def sequence_images(self, images, proprio, hidden=None):
        logits, hidden, stage, _ = self.sequence_from_features(self.encode(images), proprio, hidden)
        return logits, hidden, stage

    @torch.no_grad()
    def predict(self,observation,deterministic=True):
        if not deterministic:raise ValueError('Deterministic BC policy')
        self.eval()
        legal={key:observation[key] for key in INPUT_KEYS}
        features=self.encode(legal)
        proprio=torch.cat([torch.as_tensor(legal[k],device=self.device,dtype=torch.float32).flatten() for k in PROPRIO_KEYS])
        logits,self._hidden,_,_=self.sequence_from_features(features,proprio.reshape(1,1,16),self._hidden)
        return logits[0,0].tanh().cpu().numpy()


class PixelReader:
    """Read only the current RGB chunk; never load whole trajectories of images."""
    def __init__(self, entries):
        self.entries = entries
        self.stops = np.cumsum([entry['steps'] for entry in entries])
        self.files = OrderedDict()

    def read(self, indices):
        images = {key: [] for key in RGB_KEYS}
        for index in indices.flatten().tolist():
            episode = int(np.searchsorted(self.stops, index, side='right'))
            frame = index - (int(self.stops[episode - 1]) if episode else 0)
            if episode not in self.files:
                self.files[episode] = h5py.File(self.entries[episode]['path'], 'r')
                if len(self.files) > 8:
                    self.files.popitem(last=False)[1].close()
            self.files.move_to_end(episode)
            for key in RGB_KEYS:
                images[key].append(self.files[episode][f'observations/{key}'][frame])
        return {key: torch.from_numpy(np.stack(values)) for key, values in images.items()}

    def close(self):
        for file in self.files.values():
            file.close()
        self.files.clear()


def make_visual_policy(kind, proprio, checkpoint=None, finetune_cnn=False):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False) if checkpoint else None
    if kind == 'rgb_lstm':
        policy = RGBSpatialLSTMPolicy(**saved['model_config']) if saved else RGBSpatialLSTMPolicy(pretrained=True)
        if not saved:
            policy.proprio_mean.copy_(proprio.mean(0))
            policy.proprio_scale.copy_(proprio.std(0, correction=int(len(proprio) > 1)).clamp_min(.01))
    elif kind == 'rgb_spatial':
        if not saved:
            raise ValueError('rgb_spatial requires --init-checkpoint from prepare_visual or an earlier spatial BC run')
        config = dict(saved['model_config'])
        if config.get('predicted_geometry_inputs') and finetune_cnn:
            raise ValueError('The predicted-geometry comparison requires a frozen backbone')
        config['bc_finetune_stages'] = ['layer1', 'layer2'] if finetune_cnn else []
        policy = VisionStateLSTM(config)
    else:
        raise ValueError(kind)
    if saved:
        if saved.get('uses_teacher_stage') or saved.get('uses_depth') or saved.get('uses_privileged_state'):
            raise ValueError('Visual checkpoints must use only RGB and permitted robot observations')
        policy.load_state_dict(saved['policy_state_dict'], strict=True)
    return policy
