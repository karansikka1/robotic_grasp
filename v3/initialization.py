"""Initialize a fresh experiment from a saved state policy, with provenance."""

import hashlib
from pathlib import Path

import torch

from v3.model import StatePPOPolicy


def initialize_policy(checkpoint_path=None, *, hidden_dim=None, device='cpu'):
    if checkpoint_path is None:
        return StatePPOPolicy(hidden_dim=128 if hidden_dim is None else hidden_dim).to(device), None
    path = Path(checkpoint_path).resolve(strict=True)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if not checkpoint.get('uses_privileged_state') or checkpoint['config'].get('task_name') != 'v2-green-lift':
        raise ValueError('Initialization requires a v3 state-based green-lift checkpoint')
    model_config = checkpoint['model_config']
    if hidden_dim is not None and hidden_dim != model_config['hidden_dim']:
        raise ValueError('--hidden-dim must match the initialization checkpoint')
    policy = StatePPOPolicy(**model_config)
    # Includes actor, critic, and exploration log_std. The trainer creates Adam
    # anew; optimizer moments, old rollouts, and step counters are not restored.
    policy.load_state_dict(checkpoint['policy_state_dict'])
    provenance = {
        'checkpoint': str(path),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'global_step': checkpoint['global_step'],
        'update': checkpoint['update'],
        'source_task': checkpoint['config']['task'],
        'optimizer_restored': False,
    }
    return policy.to(device), provenance
