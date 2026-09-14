"""State BC models; share the original collection policy's input/checkpoint schema."""
import torch
from torch import nn

from bc_data.state_policy import FixedRotationHead, StateLSTMPolicy, StageAuxStateLSTMPolicy

STATE_TYPES = ('state_mlp', 'state_lstm', 'state_lstm_aux')


class StateMLP(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer('mean', mean.clone())
        self.register_buffer('scale', scale.clone())
        self.network = nn.Sequential(nn.Linear(len(mean), 128), nn.ReLU(),
                                     nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 7))
        self.network[-1] = FixedRotationHead.from_seven_head(self.network[-1])

    def forward(self, state):
        return self.network((state - self.mean) / self.scale)

    @torch.no_grad()
    def predict_state(self, state):
        self.eval()
        return self(state).tanh()

    def get_model_config(self):
        return {'input_dim': len(self.mean), 'action_dim': 4}


def make_state_policy(kind, mean, scale):
    if kind == 'state_mlp':
        return StateMLP(mean, scale)
    cls = StageAuxStateLSTMPolicy if kind == 'state_lstm_aux' else StateLSTMPolicy
    if kind not in STATE_TYPES:
        raise ValueError(kind)
    return cls(mean, scale, hidden_dim=128, lstm_layers=1, action_dim=4)


def load_state_checkpoint(path, kind, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('use_stage') or saved.get('uses_teacher_stage'):
        raise ValueError('Only policies without teacher-stage inputs are supported')
    weights = saved.get('policy_state_dict', saved.get('state_dict'))
    policy = make_state_policy(kind, weights['mean'], weights['scale'])
    policy.load_state_dict(weights, strict=True)
    return policy.to(device).eval()


def load_policy(checkpoint, kind, device='cpu'):
    """Load a packaged state or RGB policy for evaluation."""
    if kind in STATE_TYPES:
        return load_state_checkpoint(checkpoint, kind, device)
    if kind not in ('rgb_lstm', 'rgb_spatial'):
        raise ValueError(kind)
    from .visual import RGBSpatialLSTMPolicy, VisionStateLSTM
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if any(saved.get(key) for key in ('uses_teacher_stage', 'uses_depth', 'uses_privileged_state')):
        raise ValueError('Visual policies must use only RGB and permitted robot measurements')
    policy = (RGBSpatialLSTMPolicy(**saved['model_config']) if kind == 'rgb_lstm'
              else VisionStateLSTM(saved['model_config']))
    policy.load_state_dict(saved['policy_state_dict'], strict=True)
    return policy.to(device).eval()
