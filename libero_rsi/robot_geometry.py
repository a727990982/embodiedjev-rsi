"""Robot-only geometry and camera projections; no object poses or physics stepping.

Compiled mesh vertices are already scaled/centered by MuJoCo. Transform them with
compiled geom poses, not authored XML offsets. Bounds are conservative boxes,
not silhouettes, collision tests, visibility masks or future configurations.
"""
from itertools import product
import numpy as np
from PIL import Image, ImageDraw

PREFIXES = ('robot0_', 'gripper0_', 'mount0_')
CORNERS = np.array(list(product((-1., 1.), repeat=3)))
EDGES = [(i, j) for i in range(8) for j in range(i+1, 8) if np.sum(CORNERS[i] != CORNERS[j]) == 1]
INSTRUCTIONS = '''
Robot geometry assistance:
robot_geometry contains ONLY this robot's model dimensions, current joint/body poses
and calibrated camera projections at observation_step. Lengths are meters, rotations
are row-major local-to-world matrices, and joint angles are radians. Bounds are
conservative boxes, not exact surfaces. Projected boxes can show THROUGH scene objects;
2D overlap does not prove occlusion or collision. Original images remain authoritative.
R-number labels in overlays map to bodies[].label and bodies[].name in the geometry table.
Compare finger, palm/wrist and arm clearance separately. Use image/depth evidence for
surrounding objects; their hidden geometry is unknown. Missing geometry is not clearance.
A proposed yaw is not an IK solution, future arm pose or collision-free trajectory.
Return an additional robot_geometry_advice object with:
  collision: "no_visible_obstruction" | "possible_obstruction" | "uncertain"
  visibility: "target_visible" | "robot_occlusion" | "scene_occlusion" | "uncertain"
  evidence: short visual evidence naming the view and robot part, with uncertainty
  suggestion: a conditional adjustment or observation suggestion for Jev
  verify: what to inspect after the chosen action
This object is advisory ONLY. Jev chooses the next action. Do not copy geometry-only
suggestions into executable plan steps; existing independently justified recovery
plans remain subject to the existing protocol.
'''


def _name(model, kind, index):
    method = getattr(model, kind + '_id2name', None)
    return (method(index) if method else getattr(model, kind)(index).name) or ''


def _body_array(data, legacy, native):
    return getattr(data, legacy) if hasattr(data, legacy) else getattr(data, native)


def _box(points):
    lo, hi = np.min(points, axis=0), np.max(points, axis=0)
    return (lo+hi)/2, (hi-lo)/2


def _geom_bounds(model, g):
    kind, size = int(model.geom_type[g]), np.asarray(model.geom_size[g], float)
    center = np.zeros(3)
    if kind == 2: half = np.repeat(size[0], 3)  # sphere
    elif kind == 3: half = np.array([size[0], size[0], size[1]+size[0]])  # capsule
    elif kind == 4 or kind == 6: half = size.copy()  # ellipsoid / box
    elif kind == 5: half = np.array([size[0], size[0], size[1]])  # cylinder
    elif kind == 7:
        mesh = int(model.geom_dataid[g])
        if mesh < 0: raise ValueError('missing mesh')
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        center, half = _box(np.asarray(model.mesh_vert[start:start+count], float))
    else:
        raise ValueError(f'unsupported robot geom type {kind}')
    if not np.all(np.isfinite([center, half])): raise ValueError('nonfinite geom bounds')
    return center + CORNERS * half


def snapshot(model, data, tcp, tcp_quat, cameras, step):
    """Read only named robot bodies/joints/geoms and the explicitly requested cameras.

    cameras maps a logical view to (model camera name, image width, image height).
    No full-state qpos/body array is copied into the result.
    """
    out = {'observation_step': step, 'status': 'available', 'units': 'meters and radians',
           'source': 'robot model and own forward-kinematic state; camera calibration',
           'bounds_note': 'Body-local conservative boxes; parent offsets are frame offsets, not physical link lengths. '
                          'No scene-object geometry, IK, future pose or collision/visibility test.',
           'tcp_xyz_m': np.asarray(tcp).tolist(), 'tcp_orientation_xyzw': np.asarray(tcp_quat).tolist(),
           'bodies': [], 'joints': [], 'cameras': {}, 'omissions': []}
    if not hasattr(model, 'nbody'):
        out.update(status='unavailable', omissions=['robot model has no body geometry'])
        return out
    own = {b for b in range(1, model.nbody) if _name(model, 'body', b).startswith(PREFIXES)}
    if not own:
        out.update(status='unavailable', omissions=['no robot0/gripper0/mount0 bodies found'])
        return out
    xpos = _body_array(data, 'body_xpos', 'xpos')
    xmat = _body_array(data, 'body_xmat', 'xmat')
    for b in sorted(own):
        name = _name(model, 'body', b)
        pos, rot = np.asarray(xpos[b]), np.asarray(xmat[b]).reshape(3, 3)
        parent = int(model.body_parentid[b])
        body = {'name': name, 'label': f'R{len(out["bodies"])+1}',
                'parent': _name(model, 'body', parent) if parent in own else 'robot_root',
                'parent_offset_m': np.asarray(model.body_pos[b]).tolist(),
                'position_world_m': pos.tolist(), 'rotation_world': rot.tolist(),
                'tcp_offset_body_m': (rot.T @ (np.asarray(tcp)-pos)).tolist(), 'bounds': []}
        groups = {'collision': [], 'visual': []}
        for g in range(int(model.body_geomadr[b]), int(model.body_geomadr[b]+model.body_geomnum[b])):
            try:
                points = _geom_bounds(model, g)
                world = points @ np.asarray(data.geom_xmat[g]).reshape(3, 3).T + data.geom_xpos[g]
                collision = bool(model.geom_contype[g] or model.geom_conaffinity[g])
                groups['collision' if collision else 'visual'].extend((world-pos) @ rot)
            except (AttributeError, IndexError, ValueError) as ex:
                out['omissions'].append(f'{name} geom {g}: {ex}')
        for kind, points in groups.items():
            if points:
                center, half = _box(points)
                body['bounds'].append({'kind': kind, 'center_body_m': center.tolist(),
                                       'dimensions_m': (2*half).tolist()})
        out['bodies'].append(body)
    for j in range(model.njnt):
        if int(model.jnt_bodyid[j]) not in own: continue
        kind = int(model.jnt_type[j])
        if kind not in (2, 3):
            out['omissions'].append(f'{_name(model,"joint",j)}: non hinge/slide joint omitted')
            continue
        out['joints'].append({'name': _name(model, 'joint', j),
            'body': _name(model, 'body', int(model.jnt_bodyid[j])),
            'type': 'hinge' if kind == 3 else 'slide',
            'position': float(data.qpos[int(model.jnt_qposadr[j])]),
            'axis_body': np.asarray(model.jnt_axis[j]).tolist(),
            'origin_body_m': np.asarray(model.jnt_pos[j]).tolist(),
            'limits': np.asarray(model.jnt_range[j]).tolist() if model.jnt_limited[j] else None})
    for view, (name, width, height) in cameras.items():
        try:
            method = getattr(model, 'camera_name2id', None)
            c = int(method(name) if method else model.camera(name).id)
            if c < 0: raise ValueError('camera absent')
            # This harness uses perspective fovy cameras. Do not silently misproject other models.
            if hasattr(model, 'cam_orthographic') and model.cam_orthographic[c]:
                raise ValueError('orthographic camera unsupported')
            if hasattr(model, 'cam_sensorsize') and np.any(model.cam_sensorsize[c]):
                raise ValueError('sensor-size intrinsics unsupported by fovy projection')
            f = .5 * height / np.tan(np.deg2rad(float(model.cam_fovy[c]))/2)
            camera = {'name': name, 'image_size': [width, height], 'focal_px': float(f),
                'principal_px': [width/2, height/2], 'position_world_m': data.cam_xpos[c].tolist(),
                'rotation_world': data.cam_xmat[c].reshape(3,3).tolist(),
                'convention': 'Matches vertically flipped harness RGB: +x right, +y up, camera looks -z. '
                              'Projected bounds are not visibility masks.'}
            camera['projected_bounds'] = project_bodies(out['bodies'], camera)
            out['cameras'][view] = camera
        except (AttributeError, IndexError, KeyError, ValueError) as ex:
            out['omissions'].append(f'camera {view}: {ex}')
    if out['omissions']: out['status'] = 'partial'
    return out


def projected_edges(body, bound, camera):
    points = np.asarray(bound['center_body_m']) + CORNERS * np.asarray(bound['dimensions_m']) / 2
    world = points @ np.asarray(body['rotation_world']).T + body['position_world_m']
    local = (world-camera['position_world_m']) @ np.asarray(camera['rotation_world'])
    lines = []
    for i, j in EDGES:
        a, b = local[i].copy(), local[j].copy()
        if a[2] >= -.001 and b[2] >= -.001: continue
        if a[2] >= -.001: a = b + (a-b) * ((-.001-b[2])/(a[2]-b[2]))
        if b[2] >= -.001: b = a + (b-a) * ((-.001-a[2])/(b[2]-a[2]))
        def pixel(v):
            return [camera['principal_px'][0] + camera['focal_px']*v[0]/-v[2],
                    camera['principal_px'][1] - camera['focal_px']*v[1]/-v[2]]
        lines.append((pixel(a), pixel(b)))
    return lines


def project_bodies(bodies, camera):
    result = []
    for body in bodies:
        for bound in body['bounds']:
            lines = projected_edges(body, bound, camera)
            if not lines: continue
            pixels = np.asarray(lines).reshape(-1, 2)
            lo, hi = pixels.min(0), pixels.max(0)
            w, h = camera['image_size']
            if hi[0] < 0 or hi[1] < 0 or lo[0] >= w or lo[1] >= h: continue
            result.append({'body': body['name'], 'label': body.get('label', body['name']), 'kind': bound['kind'],
                'pixel_bounds_clipped': [float(max(0,lo[0])), float(max(0,lo[1])),
                                         float(min(w-1,hi[0])), float(min(h-1,hi[1]))]})
    return result


def overlay(rgb, geometry, view, route=None):
    im = Image.fromarray(np.asarray(rgb).copy()); draw = ImageDraw.Draw(im)
    camera = geometry['cameras'][view]
    for body in geometry['bodies']:
        for bound in body['bounds']:
            color = 'cyan' if bound['kind'] == 'collision' else 'magenta'
            for a, b in projected_edges(body, bound, camera): draw.line([tuple(a), tuple(b)], fill=color, width=1)
    labels, occupied = set(), []
    for box in camera['projected_bounds']:
        if box['label'] in labels: continue
        labels.add(box['label'])
        x, y = box['pixel_bounds_clipped'][:2]
        x, y = min(x, im.width-30), max(26, min(y, im.height-14))
        while any(abs(x-px)<30 and abs(y-py)<12 for px,py in occupied) and y<im.height-26:
            y += 12
        occupied.append((x,y))
        draw.rectangle(draw.textbbox((x,y),box['label']), fill='black')
        draw.text((x,y), box['label'], fill='yellow')
    if route:
        previous = None
        for key, label in [('tcp_xyz_m','TCP'), ('approach_xyz_m','approach'), ('descent_xyz_m','grasp')]:
            v = (np.asarray(route[key])-camera['position_world_m']) @ np.asarray(camera['rotation_world'])
            if v[2] >= -.001:
                previous = None
                continue
            pixel = [camera['principal_px'][0]+camera['focal_px']*v[0]/-v[2],
                     camera['principal_px'][1]-camera['focal_px']*v[1]/-v[2]]
            if previous is not None: draw.line([tuple(previous),tuple(pixel)], fill='orange', width=2)
            draw.text(tuple(pixel), label, fill='orange')
            previous = pixel
    draw.rectangle((0,0,im.width,25), fill='black')
    draw.text((3,2), f'{view} robot boxes: cyan collision, magenta visual', fill='white')
    draw.text((3,13), 'Conservative projection; NOT visibility or clearance proof', fill='yellow')
    return im


def advice(raw):
    """Normalize only descriptive fields; never expose executable plan data."""
    if not isinstance(raw, dict): return None
    result = {}
    for key, values in {'collision': {'no_visible_obstruction','possible_obstruction','uncertain'},
                         'visibility': {'target_visible','robot_occlusion','scene_occlusion','uncertain'}}.items():
        value = raw.get(key)
        result[key] = value if isinstance(value, str) and value in values else 'uncertain'
    for key in ('evidence', 'suggestion', 'verify'):
        result[key] = raw[key].strip()[:500] if isinstance(raw.get(key), str) else ''
    if not result['evidence']:
        result.update(collision='uncertain', visibility='uncertain', evidence='No visual evidence supplied.')
    return result
