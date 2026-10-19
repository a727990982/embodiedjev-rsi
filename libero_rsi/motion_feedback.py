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
