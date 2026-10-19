"""Measured motion and visual-waypoint checks; no scene-object ground truth."""
from math import dist, isfinite


def project(point, camera_position, camera_rotation, focal, size):
    delta = [a-b for a, b in zip(point, camera_position)]
    local = [sum(camera_rotation[j][i] * delta[j] for j in range(3)) for i in range(3)]
    if local[2] >= -1e-6:
        return None
    return [size/2 + focal * local[0]/-local[2], size/2 - focal * local[1]/-local[2]]


def grasp_position(tcp, grasp, object_width, object_height):
    """Require proximity in XY AND height; being above the target is insufficient."""
    # Allow controller tolerance + visual estimate error, but never an above-target gap.
    xy_tol = max(.010, min(.015, .5 * object_width))
    z_tol = max(.010, min(.015, .5 * object_height))
    xy = dist(tcp[:2], grasp[:2])
    z = abs(tcp[2] - grasp[2])
    return {'ready': isfinite(xy + z) and xy <= xy_tol and z <= z_tol,
            'xy_error_m': xy, 'height_error_m': z,
            'xy_tolerance_m': xy_tol, 'height_tolerance_m': z_tol,
            'source': 'current proprioception versus visual grasp estimate; not object truth'}


def release_position(tcp, place, clearance):
    return (dist(tcp[:2], place[:2]) <= .04
            and abs(tcp[2] - (place[2] + clearance)) <= .015)


def same_motion(old, target, orientation):
    """An unchanged failed waypoint/orientation must not be retried blindly."""
    return (old is not None and dist(old['target_xyz_m'], target) <= .006
            and all(abs(a - b) <= 3 for a, b in zip(old['orientation_deg'], orientation)))


def recovery_progress(intents, results, positions, grasp, contact_parts):
    """Measure completed preparation/approach cycles, not individual motor success.

    Compare all endpoints to the SAME visual waypoint so changing the estimate
    alone is not progress. A real descent attempt starts a new validation cycle.
    """
    preparations = {'clear_reapproach', 'change_approach_yaw', 'back_off'}
    history = list(zip(intents, results, positions, contact_parts))
    tail = []
    for row in reversed(history):
        intent, result, _, _ = row
        if intent not in preparations | {'move_above_object'}:
            break
        if result not in ('reached', 'at_target') and not result.startswith('recovery_prepared:'):
            break
        tail.append(row)
    tail.reverse()
    approaches = [i for i, row in enumerate(tail) if row[0] == 'move_above_object']
    count = sum(row[0] in preparations for row in tail)
    out = {'preparations_without_descent': count, 'no_progress': False,
           'note': 'Preparation and reaching an approach waypoint do not verify clearance or graspability. '
                   'Use current images to choose a descent test, a different approach, or diagnosis.'}
    # Two completed preparation cycles between three approach endpoints.
    if len(approaches) < 3 or tail[-1][0] != 'move_above_object':
        return out
    window = tail[approaches[-3]:]
    if sum(row[0] in preparations for row in window) < 2:
        return out
    endpoints = [row for row in window if row[0] == 'move_above_object']
    errors = [dist(row[2], grasp) for row in endpoints]
    improvement = errors[0] - min(errors[1:])
    cleared = bool(set(endpoints[0][3]) - set(endpoints[-1][3]))
    out.update(no_progress=improvement < .006 and not cleared,
               approach_error_m=errors, error_reduction_m=improvement,
               contact_parts_cleared=cleared)
    return out


def contact_summary(pairs, robot_parts):
    """Report only the robot part in contact, never the other body's identity."""
    environment, self_contact = set(), set()
    for a, b in pairs:
        pa, pb = robot_parts.get(a), robot_parts.get(b)
        if pa and pb:
            self_contact.update((pa, pb))
        elif pa or pb:
            environment.add(pa or pb)
    return {'environment_contact_parts': sorted(environment),
            'self_contact_parts': sorted(self_contact),
            'note': 'Robot-part contact booleans, including fixtures. No contacted-object identity. '
                    'Contact can be intended and does not by itself prove an obstruction.'}


def motion_report(start, end, target, start_step, end_step, result, before, during, after):
    return {'result': result, 'start_step': start_step, 'end_step': end_step,
            'start_xyz_m': list(start), 'end_xyz_m': list(end), 'target_xyz_m': list(target),
            'requested_delta_m': [b-a for a, b in zip(start, target)],
            'actual_delta_m': [b-a for a, b in zip(start, end)],
            'remaining_distance_m': dist(end, target),
            'contacts_before': before,
            'environment_contact_parts_during_motion': sorted(during['environment']),
            'self_contact_parts_during_motion': sorted(during['self']),
            'contacts_after': after,
            'interpretation': 'Stall means insufficient motion, not a diagnosed collision. '
                              'Use before/after images and contact locations to diagnose the cause.'}
