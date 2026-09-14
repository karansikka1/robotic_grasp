"""Original privileged state-input BC policy, used only to visit correction states."""
import numpy as np
import torch
from torch import nn

PROPRIO_KEYS = ('robot0_joint_pos', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos')
STAGES = tuple(f'{moving}_{stage}_over_{support}' for moving, support in (('green', 'red'), ('blue', 'green')) for stage in ('move_above', 'descend', 'close', 'lift', 'transfer', 'lower', 'release', 'retreat')) + ('settle',)

class FixedRotationHead(nn.Linear):
    """Stores only four output rows; no trainable rotation outputs exist."""
    def __init__(self, in_features):
        super().__init__(in_features, 4)

    def forward(self, values):
        four = super().forward(values)
        return torch.cat((four[..., :3], torch.zeros_like(four[..., :3]), four[..., 3:]), dim=-1)

    @classmethod
    def from_seven_head(cls, head):
        # Match fresh seven-head initialization for retained rows without consuming
        # additional RNG draws (including downstream stage head and minibatch order).
        with torch.random.fork_rng(devices=[]):
            result = cls(head.in_features).to(device=head.weight.device, dtype=head.weight.dtype)
        with torch.no_grad():
            result.weight.copy_(head.weight[[0, 1, 2, 6]])
            result.bias.copy_(head.bias[[0, 1, 2, 6]])
        return result

def state_features(positions, eef, proprio, stages, *, use_stage=True):
    """Shared offline/online packing. Relative positions are derived, not targets."""
    positions = np.asarray(positions).reshape(-1, 3, 3)
    eef = np.asarray(eef).reshape(-1, 3)
    proprio = np.asarray(proprio).reshape(-1, 16)
    one_hot = (np.eye(len(STAGES))[np.asarray([STAGES.index(stage) for stage in stages])]
               if use_stage else np.zeros((len(positions), len(STAGES))))
    return np.concatenate((positions.reshape(-1, 9), eef, proprio,
                           (positions - eef[:, None, :]).reshape(-1, 9), one_hot), axis=1).astype(np.float32)

class StateLSTMPolicy(nn.Module):
    """Exact state + stage, with an episode-local recurrent state."""
    def __init__(self, mean, scale, hidden_dim=128, lstm_layers=1, action_dim=7,
                 recurrent_activation='tanh'):
        super().__init__()
        self.register_buffer('mean', mean.clone())
        self.register_buffer('scale', scale.clone())
        if action_dim not in (4, 7):
            raise ValueError("action_dim must be 4 or 7")
        self.action_dim = action_dim
        if recurrent_activation != 'tanh':
            raise ValueError('This collection package supports the original tanh-LSTM checkpoint only')
        self.recurrent_activation = recurrent_activation
        self.hidden_dim, self.lstm_layers = hidden_dim, lstm_layers
        self.encoder = nn.Sequential(nn.Linear(len(mean), hidden_dim), nn.ReLU())
        recurrent_class = nn.LSTM
        self.lstm = recurrent_class(hidden_dim, hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.actor = nn.Linear(hidden_dim, 7)
        if action_dim == 4:
            self.actor = FixedRotationHead.from_seven_head(self.actor)
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
        return {'input_dim': len(self.mean), 'hidden_dim': self.hidden_dim, 'lstm_layers': self.lstm_layers,
                'action_dim': self.action_dim, 'recurrent_activation': self.recurrent_activation}

class StageAuxStateLSTMPolicy(StateLSTMPolicy):
    """Parallel stage-prediction head; labels and predictions are never action inputs."""
    def __init__(self, mean, scale, hidden_dim=128, lstm_layers=1, stage_classes=17, action_dim=7,
                 recurrent_activation='tanh'):
        super().__init__(mean, scale, hidden_dim, lstm_layers, action_dim=action_dim,
                         recurrent_activation=recurrent_activation)
        self.stage_classes = stage_classes
        self.stage_head = nn.Linear(hidden_dim, stage_classes)

    def sequence_with_stage(self, features, hidden=None):
        encoded = self.encoder((features['state'] - self.mean) / self.scale)
        values, hidden = self.lstm(encoded, hidden)
        return self.actor(values), hidden, self.stage_head(values)

    def sequence(self, features, hidden=None):
        logits, hidden, _ = self.sequence_with_stage(features, hidden)
        return logits, hidden

    def get_model_config(self):
        return {**super().get_model_config(), 'stage_classes': self.stage_classes}

def load_state_lstm(path, device='cpu'):
    saved = torch.load(path, map_location=device, weights_only=False)
    weights = saved['policy_state_dict']
    config = saved['model_config']
    cls = StageAuxStateLSTMPolicy if config.get('stage_classes') else StateLSTMPolicy
    extra = {'stage_classes': config['stage_classes']} if config.get('stage_classes') else {}
    policy = cls(weights['mean'], weights['scale'], hidden_dim=config['hidden_dim'],
                 lstm_layers=config['lstm_layers'], action_dim=config.get('action_dim', 7),
                 recurrent_activation=config.get('recurrent_activation', 'tanh'), **extra).to(device)
    policy.load_state_dict(weights)
    return policy.eval()
