"""Replay and validate the three review-only visible disturbance examples."""

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import av
import h5py
import numpy as np

from motion_planning.simulator import Simulator
from v4.data import write_manifest
from v4.record_trajectory import OBSERVATION_KEYS
from v4.teacher import read_oracle_state
from v4.visible_perturbation_pilots import ROOT, PreviewController


def audit(result):
    np.random.seed(result['layout_seed'])
    simulator = Simulator(has_renderer=False)
    try:
        np.random.seed(result['layout_seed'])
        simulator.reset()
        observation = simulator.step(np.zeros(7, dtype=np.float32))
        controller = PreviewController(result['category'], result['noise_seed'], read_oracle_state(simulator))
        with h5py.File(result['trajectory'], 'r') as f:
            assert f.attrs['review_only']
            assert f.attrs['success'] == result['success']
            intended, executed = f['actions'][:], f['executed_actions'][:]
            assert np.isfinite(intended).all() and np.isfinite(executed).all()
            assert np.abs(intended).max() <= 1 and np.abs(executed).max() <= 1
            np.testing.assert_array_equal(intended[:, 3:6], executed[:, 3:6])
            if result['category'] != 'drop_block':
                np.testing.assert_array_equal(intended[:, 6], executed[:, 6])
            positions, grip = f['oracle/object_positions'][:], f['oracle/eef_position'][:]
            stages, phases = f['stage'].asstr()[:], f['phase'][:]
            metadata = {k: v[:] for k, v in f['perturbation'].items()}
            for key in OBSERVATION_KEYS:
                np.testing.assert_array_equal(observation[key], f[f'observations/{key}'][0])
            for step in range(len(intended)):
                state = read_oracle_state(simulator)
                np.testing.assert_array_equal(state.positions, positions[step])
                np.testing.assert_array_equal(state.eef_position, grip[step])
                decision, action, info = controller.command(simulator, state, step)
                assert not decision.done
                assert decision.stage == stages[step] and decision.phase == phases[step]
                np.testing.assert_array_equal(decision.action, intended[step])
                np.testing.assert_array_equal(action, executed[step])
                for key, value in info.items():
                    np.testing.assert_array_equal(value, metadata[key][step])
                observation = simulator.step(action)
            decision, _, _ = controller.command(simulator, read_oracle_state(simulator), len(intended))
            assert decision.done and observation['task_complete']
            assert f['next_task_complete'][-10:].all()
            for key in OBSERVATION_KEYS:
                np.testing.assert_array_equal(observation[key], f[f'terminal_observation/{key}'][:])
            assert controller.event_start == result['event_start']
            assert controller.event_end == result['event_end']
            assert np.any(metadata['changed'])
            if result['category'] == 'drop_block':
                g = f['oracle/grasped'][:, 1]
                assert g[result['event_start']]
                assert not g[result['event_start']:result['event_end']].all()
                assert g[result['regrasp_step']]
                assert result['green_downward_travel_cm'] > 8
        video_frames = {}
        for filename in ('raw.mp4', 'review.mp4'):
            with av.open(str(Path(result['video']).parent / filename)) as video:
                video_frames[filename] = sum(1 for _ in video.decode(video=0))
            assert video_frames[filename] == result['steps'] + 1
        return {'category': result['category'], 'passed': True, 'steps': result['steps'],
                'all_positions_commands_stages_and_metadata_exact': True,
                'initial_terminal_observations_exact': True, 'stable_stack_success': True,
                'decoded_video_frames': video_frames}
    finally:
        simulator.close()


def main():
    data = json.loads((ROOT / 'examples.json').read_text())
    assert len(data['examples']) == 3
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn')) as pool:
        results = list(pool.map(audit, data['examples']))
    report = {'passed': True, 'examples': results, 'review_only': True}
    write_manifest(report, ROOT / 'validation.json')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
