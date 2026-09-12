"""Causal observation windows with the preceding teacher action for BC."""

import torch


def add_history(features, actions, entries, history_length):
    """Build windows after splitting/caching, with identical padding to v2.

    Frame t predicts action t; the previous-action input is action t-1 only.
    Each trajectory starts with repeated frame zero and a zero previous action.
    """
    if not isinstance(history_length, int) or history_length < 1:
        raise ValueError('history_length must be a positive integer')
    if sum(entry['steps'] for entry in entries) != len(actions):
        raise ValueError('Trajectory lengths do not match the feature cache')
    if any(len(value) != len(actions) for value in features.values()):
        raise ValueError('Feature and action counts differ')
    indices, previous = [], []
    start = 0
    for entry in entries:
        length = entry['steps']
        if length < 1:
            raise ValueError('Empty trajectory')
        time = torch.arange(length).unsqueeze(1)
        offsets = torch.arange(1 - history_length, 1).unsqueeze(0)
        indices.append((time + offsets).clamp_min(0) + start)
        previous.append(torch.cat([torch.zeros_like(actions[start:start + 1]),
                                   actions[start:start + length - 1]], dim=0))
        start += length
    indices = torch.cat(indices)
    return {**{name: value[indices] for name, value in features.items()},
            'previous_action': torch.cat(previous)}
