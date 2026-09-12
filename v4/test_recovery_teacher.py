"""Object/phase coverage, gripper faults, and phase-preserving recovery checks."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from v4.collect_recovery import make_plan
from v4.control_diagnostics import STAGES
from v4.recovery_teacher import DisturbanceConfig, RecoveryTeacher
from v4.teacher import Stage, TeacherDecision, TeacherFailure


def state():
    return SimpleNamespace(positions=np.array([[0, 0, .82], [0, 0, .87], [.15, .10, .825]]),
                           eef_position=np.array([.15, .10, .90]),
                           body_velocities=np.zeros((3, 6)), grasped=np.array([False, False, False]),
                           object_contacts=np.array([True, False, False]))


class RecoveryTests(unittest.TestCase):
    def test_frozen_plan_has_balanced_objects_and_held_out_phases(self):
        plan = make_plan()
        self.assertEqual(plan, make_plan())
        self.assertEqual(len(plan['slots']), 300)
        self.assertEqual(sum(s['partition'] == 'validation' for s in plan['slots']), 30)
        self.assertEqual(sum(s['save_video'] for s in plan['slots']), 10)
        seeds = [c['seed'] for s in plan['slots'] for c in s['candidates']]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertFalse(set(seeds) & set(plan['excluded_seeds']))
        for category in ('gaussian_burst', 'drop_block', 'sideways_error', 'gripper_interrupt'):
            rows = [s for s in plan['slots'] if s['category'] == category]
            self.assertEqual(sum(s['config']['target_phase'] == 0 for s in rows), 25)
            self.assertEqual({s['config']['target_phase'] for s in rows if s['partition'] == 'validation'}, {0, 1})
            self.assertEqual({s['config']['target_phase'] for s in rows if s['save_video']}, {0, 1})
        gaussian = [s for s in plan['slots'] if s['category'] == 'gaussian_burst']
        self.assertEqual({(s['config']['target_phase'], s['config']['trigger']) for s in gaussian},
                         {(p, t) for p in (0, 1) for t in ('approach', 'descent', 'lift', 'transfer')})

    def test_gripper_fault_labels_remain_close_and_only_execution_opens(self):
        s = state()
        config = DisturbanceConfig(category='gripper_interrupt', target_phase=1, trigger='delay_close', duration=4)
        controller = RecoveryTeacher(config, 12, s)
        controller.teacher.phase = 1
        action = np.array([.1, .2, .3, 0, 0, 0, 1], dtype=np.float32)
        controller.teacher.decide = Mock(return_value=TeacherDecision(action, 'blue_close_over_green', 1))
        for i in range(7):
            decision, executed, meta = controller.command(None, s, i)
            np.testing.assert_array_equal(decision.action, action)
            np.testing.assert_array_equal(executed[:6], action[:6])
            self.assertEqual(executed[6], -1 if i < 4 else 1)
            self.assertEqual(meta['active'], i < 4)
        self.assertEqual(controller.event_start, 0)
        self.assertEqual(controller.changed_steps, 4)

    def test_blue_recovery_preserves_phase_and_existing_base(self):
        s = state()
        controller = RecoveryTeacher(DisturbanceConfig(category='drop_block', target_phase=1, trigger='lift'), 1, s)
        controller.teacher.phase = 1
        controller._start_recovery(s, 10, 'lost grasp')
        decision = controller._recovery_decision(s, 11)
        self.assertEqual(decision.phase, 1)
        self.assertEqual(decision.stage, 'blue_retreat_over_green')
        self.assertIn(decision.stage, STAGES)
        self.assertEqual(decision.action[6], -1)
        self.assertIsNone(controller._recovery_decision(s, 31))
        self.assertEqual(controller.teacher.phase, 1)
        self.assertEqual(controller.teacher.stage, Stage.MOVE_ABOVE)
        self.assertIsNone(controller.recovery_start)
        self.assertEqual(controller.recoveries[0]['end'], 31)
        s.object_contacts[0] = False
        with self.assertRaisesRegex(TeacherFailure, 'base disturbed'):
            controller._start_recovery(s, 32, 'lost grasp')

    def test_gaussian_changes_only_xy_and_is_reproducible_over_held_samples(self):
        s = state()
        s.eef_position = s.positions[2] + np.array([0, 0, .12])
        config = DisturbanceConfig(category='gaussian_burst', target_phase=1, trigger='approach',
                                  duration=12, sample_hold_steps=3)
        controllers = [RecoveryTeacher(config, 8, s) for _ in range(2)]
        action = np.array([0, 0, -.2, 0, 0, 0, -1], dtype=np.float32)
        for c in controllers:
            c.teacher.phase = 1
            c.teacher.decide = Mock(return_value=TeacherDecision(action, 'blue_move_above_over_green', 1))
        outputs = []
        for i in range(15):
            left = controllers[0].command(None, s, i)
            right = controllers[1].command(None, s, i)
            np.testing.assert_array_equal(left[1], right[1])
            np.testing.assert_array_equal(left[1][2:], action[2:])
            self.assertLessEqual(np.abs(left[1]).max(), 1)
            outputs.append(left[1])
        np.testing.assert_array_equal(outputs[0], outputs[1])
        self.assertFalse(np.array_equal(outputs[2], outputs[3]))
        np.testing.assert_array_equal(outputs[12], action)


if __name__ == '__main__':
    unittest.main()
