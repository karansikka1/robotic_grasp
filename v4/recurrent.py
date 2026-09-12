"""Episode-persistent LSTM policies for visual and privileged-state BC."""
import torch
from torch import nn

from v1.model import PrivilegedPPOPolicy
from v4.model import FrozenResNetFeatures


class SpatialLSTMPolicy(PrivilegedPPOPolicy):
    """One RGB/depth/proprio frame per step; no previous-action input."""
    def __init__(self, *, lstm_layers=1, **kwargs):
        super().__init__(**kwargs)
        if lstm_layers < 1:
            raise ValueError('lstm_layers must be positive')
        self.lstm_layers = lstm_layers
        self.rgb_backbone = FrozenResNetFeatures(pretrained=kwargs.get('pretrained', True), pool_size=4)
        self.rgb_projection = nn.Sequential(nn.Linear(16384, self.embedding_dim), nn.ReLU())
        nn.init.orthogonal_(self.rgb_projection[0].weight, gain=2**.5)
        nn.init.zeros_(self.rgb_projection[0].bias)
        self.lstm = nn.LSTM(self.embedding_dim, self.embedding_dim, num_layers=lstm_layers, batch_first=True)
        self.reset_history()

    def sequence(self, features, hidden=None):
        rgb = self.rgb_projection(torch.cat((features['front'], features['wrist']), dim=-1))
        frames = self.fusion_norm(rgb + self.depth_projection(features['depth'])
                                  + self.proprio_encoder(features['proprio']))
        encoded, hidden = self.lstm(frames, hidden)
        return self.actor(encoded), hidden

    def reset_history(self):
        self._hidden = None

    @torch.no_grad()
    def predict(self, observation, *, deterministic=True):
        if not deterministic:
            raise ValueError('BC evaluation uses deterministic actions')
        self.eval()
        features = {key: value.reshape(1, 1, -1) for key, value in self.extract_frozen_features(observation).items()}
        logits, self._hidden = self.sequence(features, self._hidden)
        return logits[0, 0].tanh().cpu().numpy()

    def get_model_config(self):
        return {**super().get_model_config(), 'lstm_layers': self.lstm_layers}


class StateLSTMPolicy(nn.Module):
    """Exact state + stage, with an episode-local recurrent state."""
    def __init__(self, mean, scale, hidden_dim=128, lstm_layers=1):
        super().__init__()
        self.register_buffer('mean', mean.clone())
        self.register_buffer('scale', scale.clone())
        self.hidden_dim, self.lstm_layers = hidden_dim, lstm_layers
        self.encoder = nn.Sequential(nn.Linear(len(mean), hidden_dim), nn.ReLU())
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.actor = nn.Linear(hidden_dim, 7)
        self.reset_history()

    def sequence(self, features, hidden=None):
        encoded = self.encoder((features['state'] - self.mean) / self.scale)
        values, hidden = self.lstm(encoded, hidden)
        return self.actor(values), hidden

    def reset_history(self):
        self._hidden = None

    @torch.no_grad()
    def predict_state(self, values):
        self.eval()
        logits, self._hidden = self.sequence({'state': values.reshape(1, 1, -1)}, self._hidden)
        return logits[:, 0].tanh()

    def get_model_config(self):
        return {'input_dim': len(self.mean), 'hidden_dim': self.hidden_dim, 'lstm_layers': self.lstm_layers}


def load_state_lstm(path, device='cpu'):
    saved = torch.load(path, map_location=device, weights_only=False)
    weights = saved['policy_state_dict']
    config = saved['model_config']
    policy = StateLSTMPolicy(weights['mean'], weights['scale'], hidden_dim=config['hidden_dim'],
                             lstm_layers=config['lstm_layers']).to(device)
    policy.load_state_dict(weights)
    return policy.eval()
