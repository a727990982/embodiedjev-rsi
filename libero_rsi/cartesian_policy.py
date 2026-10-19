"""Jev-as-Policy motor Choices adapted to LIBERO's synchronous control loop.

Direction questions and endpoint-clamped control adapted from YuanKJing/
Jev-as-Policy (jev_policy.py and app.py); see THIRD_PARTY_NOTICES.md.
Reference points here come from the existing visual-estimate library, not
simulator object positions. Directions use world axes, not camera directions.
"""
import numpy as np

BLOCK_STEPS = 8
ALIGN_TOL_M = .002  # Less than position tolerance even across three axes.
FREE_SPEED_M_S = .14
CONTACT_SPEED_M_S = .06
SIGNS = {'positive': 1, 'negative': -1, 'stay': 0}


def motor_request(intent, definition, tcp, target, fingers):
    origin = np.asarray(tcp, dtype=float).reshape(3)
    reference = np.asarray(target, dtype=float).reshape(3)
    if not np.isfinite(origin).all() or not np.isfinite(reference).all():
        raise ValueError('Nonfinite Cartesian reference')
    offset = reference - origin
    relation = {axis: ('aligned' if abs(offset[i]) < ALIGN_TOL_M else
                      'positive' if offset[i] > 0 else 'negative')
                for i, axis in enumerate('xyz')}
    desired = 'stay' if fingers == 'keep' else fingers
    if desired not in ('open', 'close', 'stay'):
        raise ValueError('Unknown gripper command')
    context = {'selected_intent': intent, 'intent_definition': definition,
               'coordinate_frame': 'world XYZ in metres; not image left/right',
               'source': 'library waypoint from visual estimates and robot proprioception; not object ground truth',
               'tcp_xyz_m': origin.tolist(), 'reference_target_xyz_m': reference.tolist(),
               'target_offset_m': dict(zip('xyz', offset.tolist())),
               'target_relative_to_gripper': relation, 'requested_fingers': desired,
               'execution': 'One bounded control block, then fresh observation and a new intent. '
                            'in_progress does not mean waypoint reached. Wrong-way axes are held.'}
    questions = {}
    for axis in 'xyz':
        questions[axis] = {'type': 'choice',
            'instructions': f'Read motor_context.target_relative_to_gripper.{axis}. '
                            'Select that sign, or stay when aligned. Do not select a new target or intent.',
            'criteria': {'positive': f'Move along positive world {axis}.',
                         'negative': f'Move along negative world {axis}.',
                         'stay': f'Hold world {axis}.'}}
    questions['fingers'] = {'type': 'choice',
        'instructions': 'Follow motor_context.requested_fingers for the selected intent. '
                        'stay preserves the current command. Existing identity, grasp and release checks still apply.',
        'criteria': {'open': 'Open or keep open.', 'close': 'Close or keep closed.',
                     'stay': 'Keep the current gripper command.'}}
    return context, questions


def motor_choices(answers):
    if set(answers) != {'x', 'y', 'z', 'fingers'}:
        raise ValueError('Motor reply must contain x/y/z/fingers')
    motor = {key: value['choice'] for key, value in answers.items()}
    if any(motor[a] not in SIGNS for a in 'xyz') or motor['fingers'] not in ('open', 'close', 'stay'):
        raise ValueError('Invalid motor choice')
    return motor


def signed_delta(error, motor, speed, dt):
    """Commanded displacement cap per axis; no reversal or endpoint overshoot."""
    error = np.asarray(error, dtype=float).reshape(3)
    sign = np.array([SIGNS[motor[a]] for a in 'xyz'])
    return np.where(error * sign > 0, sign * np.minimum(np.abs(error), speed * dt), 0.)
