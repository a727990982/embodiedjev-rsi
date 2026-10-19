"""Metric camera-depth panels and regional evidence; no scene object state."""
import numpy as np
from PIL import Image, ImageDraw

INSTRUCTIONS = '''
Depth-aware path selection (required on every visual request):
Read robot_geometry.depth_views and the attached RGB-geometry/depth panels.
Left: robot geometry over RGB. Right: registered metric depth. Answer pixel
coordinates ALWAYS refer to the original FRONT RGB, not the combined panel.
Depth is camera-axis distance in metres, NOT world height or Euclidean range.
Blue is near, red is far; black is invalid/unknown. The fixed colour scale
saturates: use numeric tile statistics for exact regional distances. Each grid
cell describes an image region, not merely the grasp point. Missing depth is
unknown, never free space. Use calibrated camera axes for world motion reasoning.
For motion planning compare the full corridor for a direct route, a raised route
and a lateral detour. Account for fingers, palm/wrist, arm and held-object size;
a clear TCP line is insufficient. Write comparisons in robot_geometry_advice:
path_candidates is a list of route, assessment (preferred|blocked|unknown), and
evidence naming the camera, grid cells, depth values and robot parts. selected_path
names the preferred route or "reobserve" if evidence is insufficient. suggestion
explains which existing motion/recovery actions to consider and verify explains
what to check after the next short segment. Do not invent new executable actions.
Visible free space supports a candidate, not a whole-arm swept-volume collision
certificate. Behind surfaces, out-of-view regions, invalid depth and future arm
configurations remain unknown. For identity-only checks inspect depth but do not
invent a route selection. Never infer hidden scene geometry from simulator names.
'''


def panel(annotated_rgb, depth, *, view, step, grid=8):
    """Keep registered pixels at native size; add a fixed-scale legend and grid."""
    rgb = np.asarray(annotated_rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    valid = np.zeros((h, w), dtype=bool)
    status = 'unavailable'
    values = np.full((h, w), np.nan)
    if depth is not None:
        values = np.asarray(depth, dtype=float)
        if values.shape != (h, w):
            raise ValueError(f'{view} RGB/depth shape mismatch: {(h,w)} vs {values.shape}')
        valid = np.isfinite(values) & (values > 0)
        status = 'available' if valid.any() else 'invalid'
    low, high = .1, 2.0
    scaled = np.clip((np.where(valid, values, low)-low)/(high-low), 0, 1)
    colour = np.stack([scaled, 1-np.abs(2*scaled-1), 1-scaled], axis=-1)
    pixels = np.uint8(np.clip(colour*255, 0, 255)); pixels[~valid] = 0
    result = Image.new('RGB', (2*w, h+48), 'black')
    result.paste(Image.fromarray(rgb), (0, 0)); result.paste(Image.fromarray(pixels), (w, 0))
    draw = ImageDraw.Draw(result)
    tiles = []
    for row in range(grid):
        for col in range(grid):
            x0, x1 = col*w//grid, (col+1)*w//grid
            y0, y1 = row*h//grid, (row+1)*h//grid
            region = values[y0:y1, x0:x1]; mask = valid[y0:y1, x0:x1]
            samples = region[mask]
            tile = {'cell': f'r{row}c{col}', 'pixel_bounds_xyxy': [x0,y0,x1,y1],
                    'valid_fraction': round(float(mask.mean()), 3) if mask.size else 0.,
                    'depth_m_min_p10_median': [round(float(v), 4) for v in
                        (samples.min(), np.percentile(samples, 10), np.median(samples))] if samples.size else None}
            tiles.append(tile)
            draw.rectangle((w+x0,y0,w+max(x0,x1-1),max(y0,y1-1)), outline=(90,90,90))
            draw.text((w+x0+2,y0+2), tile['cell'], fill='white', stroke_width=1, stroke_fill='black')
    draw.text((4,h+2), f'{view} | step {step} | left: RGB geometry | right: DEPTH {status}', fill='white')
    draw.text((4,h+16), 'Camera-axis metres. Black: unknown. Grid refers to original pixels.', fill='white')
    for x in range(w):
        t = x/max(1,w-1)
        draw.line((w+x,h+30,w+x,h+39), fill=(int(255*t),int(255*(1-abs(2*t-1))),int(255*(1-t))))
    draw.text((4,h+31), 'BLUE <=0.10 m -> RED >=2.00 m (clipped)', fill='white')
    info = {'status': status, 'observation_step': step, 'image_size': [w,h],
            'units': 'metres along camera optical axis', 'colour_range_m': [low,high],
            'valid_fraction': round(float(valid.mean()), 4),
            'saturated_near_fraction': round(float((valid & (values < low)).mean()), 4),
            'saturated_far_fraction': round(float((valid & (values > high)).mean()), 4),
            'tiles': tiles, 'grid_shape': [grid,grid],
            'layout': 'Native RGB/geometry left; native aligned depth right; footer is not image data.'}
    return result, info
