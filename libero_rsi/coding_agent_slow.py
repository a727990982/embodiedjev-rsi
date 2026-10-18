"""Coding-agent mid-episode monitor + episode-start "Great Wall" planner.

Replaces or dual-runs Astra slow-think with a coding-agent style monitor that
reads text context (policy decision lines, experience_brief, PATCH/GOAL_ID) plus optional
keyframe still paths, and returns the SAME episode-plan JSON the harness already
applies via `_apply_episode_plan`.

Also provides episode-start global planning (env EPISODE_START_PLAN) that emits
an ordered predictive plan (push_aside / clear / relook / …) before the first
approach — RSI start-of-episode reflection; mid-episode slow-think only fine-tunes.

Backends (env CODING_AGENT_SLOW):
  unset      — codex
  0 / off    — coding monitor disabled (harness uses Astra)
  mock       — deterministic local plan; no network (smoke / CI)
  codex      — `codex exec` on PATH (or CODING_AGENT_CODEX_BIN). Optional proxy via
               CODING_AGENT_HTTP_PROXY / CODING_AGENT_SOCKS; unset means direct.
  dual       — call coding agent AND leave Astra path for A/B logging (harness dual)

Episode-start (env EPISODE_START_PLAN):
  unset      — codex (independent of the slow-think override)
  0 / off    — disabled
  mock|codex — start planner backend
  astra      — harness uses Astra with the start-plan prompt
  auto/empty — inherit CODING_AGENT_SLOW (or PERCEIVE) when those are on

This module is intentionally freestanding so smokes need no MuJoCo / Astra.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ALLOWED_STEPS = frozenset({
    'relook', 'raise_redescend', 'lateral_recenter', 'change_approach_yaw',
    'confirm_contact', 'clear_reapproach', 'adjust', 'push_aside',
})
ALLOWED_ISSUES = frozenset({
    'ok', 'wrong_object', 'tip_push', 'empty_grasp', 'place_miss',
    'blocked_by_neighbour', 'object_pushed_by_descent', 'other',
})
STALL_STEPS = frozenset({
    'raise_redescend', 'lateral_recenter', 'change_approach_yaw',
    'confirm_contact', 'clear_reapproach', 'push_aside', 'adjust',
})

CODEX_BIN = os.environ.get('CODING_AGENT_CODEX_BIN', 'codex')
CODEX_MODEL = os.environ.get('CODING_AGENT_CODEX_MODEL', 'gpt-6-astra')
CODEX_EFFORT = os.environ.get('CODING_AGENT_CODEX_EFFORT', 'high')
# Optional proxy. Empty (the default) means a direct connection.
DEFAULT_HTTP_PROXY = os.environ.get('CODING_AGENT_HTTP_PROXY', '').strip()
DEFAULT_SOCKS = os.environ.get('CODING_AGENT_SOCKS', '').strip()

MONITOR_PROMPT = """You are a mid-episode coding-agent MONITOR for a Jev-controlled robot arm
(LIBERO pick/place). You do NOT write control code. You only decide whether to
intervene THIS EPISODE and emit a short structured plan the harness already knows
how to apply (same schema as Astra slow-think).

Language goal: {lang}

You receive:
- recent_diag: last policy/decision lines (text)
- experience_brief: short when→prefer recoveries already in the library
- patch_goal: current PATCH offsets + GOAL_ID confirmation
- keyframe_paths: 0–4 still image paths (front+wrist composites when present)
- trigger: why you were woken (stall / spectator_defer / hard_identity / ask_astra / …)

EmbodiedEvo analogy: there the coding agent writes solution.py from observations/.
Here Jev already selects intents; you only patch the episode plan / recovery.

Reply ONLY one JSON object (no markdown fences):
{{
  "goal_object_where": short phrase or "not visible",
  "held_or_between_fingers": short phrase or "nothing",
  "is_goal_object": true | false | null,
  "matched_language_goal": true | false | null,
  "issue": one of {issues},
  "intervene": true only if issue != "ok" AND an episode-only fix is needed now,
  "diagnosis": one short sentence,
  "lesson": "" or one GENERAL qualitative sentence (NO numbers/units/object names),
  "plan": [] or ordered steps, each one of:
    {{"step":"relook"}},
    {{"step":"raise_redescend"}},
    {{"step":"lateral_recenter"}},
    {{"step":"change_approach_yaw"}},
    {{"step":"confirm_contact"}},
    {{"step":"clear_reapproach"}},
    {{"step":"adjust","grasp_dz_objh":-0.5..0.5,"tilt_deg":0..30,"yaw_deg":-90..90}},
    {{"step":"push_aside","blocker_px":[u,v],"direction":"away_from_target"|"toward_image_left"|"toward_image_right"|"toward_camera"|"away_from_camera","distance_objw":0.5..2.0}}
}}

Rules:
- Identity wrong/unsure → intervene + lead with relook.
- Identity confirmed + neighbour block → prefer push_aside or clear_reapproach (not relook-only).
- Empty close / stalled descent → raise_redescend or lateral_recenter.
- Prefer intervene=false when progressing normally with confirmed identity.
- Fine-tune only: honour episode_start_plan phase_order (incl. post-place retreat / next-object cycles); do not drop them.
- Episode-only: do not invent library-wide recoveries here.

Context JSON:
{context_json}
"""


def backend_mode(raw: Optional[str] = None) -> str:
    v = (raw if raw is not None else os.environ.get('CODING_AGENT_SLOW', 'codex')).strip().lower()
    if v in ('', '0', 'off', 'false', 'no', 'astra'):
        return 'off'
    if v in ('1', 'on', 'true', 'yes'):
        return 'mock'  # safe default when flipped on without naming a backend
    if v in ('mock', 'codex', 'dual'):
        return v
    return 'off'


def pack_context(
    *,
    lang: str,
    trigger: str,
    recent_diag: Sequence[Any],
    experience_brief: Any,
    patch: Optional[Dict[str, Any]] = None,
    goal_id: Any = None,
    keyframe_paths: Optional[Sequence[str]] = None,
    slow_ctx: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the monitor context blob (text-first; images are paths)."""
    paths = [str(p) for p in (keyframe_paths or []) if p][:4]
    ctx = {
        'language_goal': lang,
        'trigger': trigger,
        'recent_diag': list(recent_diag)[-40:],
        'experience_brief': experience_brief,
        'patch_goal': {
            'patch': dict(patch or {}),
            'goal_id': goal_id,
        },
        'keyframe_paths': paths,
        'slow_ctx': slow_ctx or {},
    }
    if extra:
        ctx['extra'] = extra
    return ctx


def build_prompt(ctx: Dict[str, Any]) -> str:
    return MONITOR_PROMPT.format(
        lang=ctx.get('language_goal') or '',
        issues=' | '.join(sorted(ALLOWED_ISSUES)),
        context_json=json.dumps(ctx, ensure_ascii=False, indent=2, default=str)[:12000],
    )


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or '').strip()
    if not text:
        raise ValueError('empty_monitor_reply')
    # Strip common fences
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Last JSON object in the dump
    m = re.search(r'\{[\s\S]*\}\s*$', text)
    if not m:
        raise ValueError('no_json_in_monitor_reply')
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError('monitor_reply_not_object')
    return obj


def validate_intervention(parsed: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Normalize + validate plan JSON for applicator acceptance.

    Returns (ok, errors, normalized).
    """
    errs: List[str] = []
    out = dict(parsed or {})
    issue = str(out.get('issue') or 'other')
    if issue not in ALLOWED_ISSUES:
        errs.append(f'bad_issue:{issue}')
        issue = 'other'
        out['issue'] = issue
    intervene = bool(out.get('intervene')) and issue != 'ok'
    out['intervene'] = intervene
    plan = out.get('plan')
    if plan is None:
        plan = []
    if not isinstance(plan, list):
        errs.append('plan_not_list')
        plan = []
    norm_plan = []
    for st in plan[:3]:
        if not isinstance(st, dict):
            errs.append('plan_step_not_object')
            continue
        step = st.get('step')
        if step not in ALLOWED_STEPS:
            errs.append(f'bad_step:{step}')
            continue
        clean = {'step': step}
        if step == 'adjust':
            for k in ('grasp_dz_objh', 'tilt_deg', 'tilt_dir_deg', 'yaw_deg'):
                if k in st:
                    try:
                        clean[k] = float(st[k])
                    except (TypeError, ValueError):
                        errs.append(f'adjust_{k}_bad')
        if step == 'push_aside':
            if 'blocker_px' in st and isinstance(st['blocker_px'], (list, tuple)) and len(st['blocker_px']) >= 2:
                try:
                    clean['blocker_px'] = [float(st['blocker_px'][0]), float(st['blocker_px'][1])]
                except (TypeError, ValueError):
                    errs.append('blocker_px_bad')
            if 'direction' in st:
                clean['direction'] = str(st['direction'])
            if 'distance_objw' in st:
                try:
                    clean['distance_objw'] = float(st['distance_objw'])
                except (TypeError, ValueError):
                    errs.append('distance_objw_bad')
        norm_plan.append(clean)
    out['plan'] = norm_plan
    if intervene and not norm_plan:
        errs.append('intervene_without_plan')
    if intervene and issue in ('empty_grasp', 'tip_push', 'blocked_by_neighbour', 'object_pushed_by_descent'):
        if not any(st.get('step') in STALL_STEPS for st in norm_plan):
            # Soft: applicator would inject a default; mark warning-style error
            errs.append('stall_missing_feedback_step')
    ok = not any(e.startswith('bad_') or e in ('plan_not_list', 'intervene_without_plan') for e in errs)
    # stall_missing is soft — still ok for applicator (it forces a default)
    hard = [e for e in errs if e != 'stall_missing_feedback_step']
    return (len(hard) == 0), errs, out


def mock_intervene(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic mock: neighbour → push_aside; empty/stall → raise; else ok."""
    trigger = str(ctx.get('trigger') or '')
    slow = ctx.get('slow_ctx') or {}
    preds = (slow.get('predicates') or {}) if isinstance(slow, dict) else {}
    last = str(slow.get('intent_result') or '')
    # Never treat a field name, a false predicate, or an instruction as evidence.
    blob = (trigger + ' ' + last + ' ' + ' '.join(k for k, v in preds.items() if v is True)).lower()
    goal_id = (ctx.get('patch_goal') or {}).get('goal_id')
    confirmed = goal_id.get('confirmed') if isinstance(goal_id, dict) else goal_id
    if confirmed is not True:
        return {
            'matched_language_goal': None, 'is_goal_object': None,
            'issue': 'wrong_object', 'intervene': True,
            'diagnosis': 'mock: identity unverified — relook before contact',
            'lesson': 'Verify target identity before contact.', 'plan': [{'step': 'relook'}],
        }

    # Confirmed grasp (harness proprio+lift): do not undo with push_aside / retreat
    if isinstance(slow, dict) and (slow.get('grasp_secured') or slow.get('grasp_confirm')):
        return {
            'goal_object_where': 'language goal held between fingers',
            'held_or_between_fingers': 'language-goal object',
            'is_goal_object': True,
            'matched_language_goal': True,
            'issue': 'ok',
            'intervene': False,
            'diagnosis': 'mock: grasp confirmed — lift/carry, do not push_aside',
            'lesson': 'After a confirmed close, lift; do not retreat or push neighbours while holding.',
            'plan': [],
        }

    # After local-deadend global relook: do NOT keep proposing push_aside
    if slow.get('local_deadend_global_relook') is True or 'local_deadend' in trigger:
        return {
            'goal_object_where': 'language goal after global front relook',
            'held_or_between_fingers': 'nothing',
            'is_goal_object': None,
            'matched_language_goal': True,
            'issue': 'other',
            'intervene': True,
            'diagnosis': 'mock: local dead-end cleared — replan from global view without another push_aside',
            'lesson': 'When local near/wrist views have no solution, retreat, look at the global front view, then replan; do not loop push_aside.',
            'plan': [{'step': 'raise_redescend'}],
        }
    # Push already exhausted in context → raise rather than another shove
    try:
        n_push = int(slow.get('push_aside_n') or 0) if isinstance(slow, dict) else 0
        max_push = int(slow.get('max_push_aside') or 2) if isinstance(slow, dict) else 2
    except Exception:
        n_push, max_push = 0, 2
    if n_push >= max_push:
        return {
            'goal_object_where': 'language goal; neighbour push budget exhausted',
            'held_or_between_fingers': 'nothing',
            'is_goal_object': None,
            'matched_language_goal': True,
            'issue': 'other',
            'intervene': True,
            'diagnosis': 'mock: push_aside cap reached — raise and re-approach from global view',
            'lesson': 'Do not burn the horizon repeating push_aside; raise clear and replan.',
            'plan': [{'step': 'raise_redescend'}],
        }

    if slow.get('neighbour_observed') is True:
        return {
            'goal_object_where': 'language goal near neighbourhood clutter',
            'held_or_between_fingers': 'nothing',
            'is_goal_object': None,
            'matched_language_goal': True,
            'issue': 'blocked_by_neighbour',
            'intervene': True,
            'diagnosis': 'mock: neighbour blocking grasp angle',
            'lesson': 'When a neighbour blocks the grasp, clear or push aside then re-approach before closing.',
            'plan': [{'step': 'push_aside', 'direction': 'away_from_target', 'distance_objw': 1.0}],
        }
    if any(k in blob for k in ('empty', 'empty_grasp', 'stalled', 'timeout', 'descent_servo_stalled', 'hard_stall')):
        return {
            'goal_object_where': 'language goal under approach',
            'held_or_between_fingers': 'nothing',
            'is_goal_object': None,
            'matched_language_goal': True,
            'issue': 'empty_grasp' if 'empty' in blob else 'other',
            'intervene': True,
            'diagnosis': 'mock: stalled / empty close — raise and redescend',
            'lesson': 'After an empty close or stalled descent, raise clear then re-approach shallower.',
            'plan': [{'step': 'raise_redescend'}],
        }
    if any(k in blob for k in ('identity', 'wrong_object', 'unmatched', 'hard_identity')):
        return {
            'goal_object_where': 'uncertain',
            'held_or_between_fingers': 'nothing',
            'is_goal_object': False,
            'matched_language_goal': False,
            'issue': 'wrong_object',
            'intervene': True,
            'diagnosis': 'mock: identity unstable — relook language goal',
            'lesson': 'When target identity is unsure, relook before closing.',
            'plan': [{'step': 'relook'}],
        }
    return {
        'goal_object_where': 'ok',
        'held_or_between_fingers': 'nothing',
        'is_goal_object': None,
        'matched_language_goal': True,
        'issue': 'ok',
        'intervene': False,
        'diagnosis': 'mock: no intervention',
        'lesson': '',
        'plan': [],
    }


def _proxy_env() -> Dict[str, str]:
    """Copy the process env and apply an optional proxy.

    CODING_AGENT_HTTP_PROXY is used when that endpoint accepts a connection.
    Otherwise CODING_AGENT_SOCKS is applied when set. Neither default is baked in.
    """
    env = dict(os.environ)
    http = DEFAULT_HTTP_PROXY
    socks = DEFAULT_SOCKS
    if http:
        try:
            import socket
            rest = http.split('://', 1)[1] if '://' in http else http
            host, port_s = rest.rsplit(':', 1)
            port = int(port_s.split('/')[0])
            with socket.create_connection((host, port), timeout=0.3):
                env['http_proxy'] = http
                env['https_proxy'] = http
                env['HTTP_PROXY'] = http
                env['HTTPS_PROXY'] = http
                env.pop('ALL_PROXY', None)
                env.pop('all_proxy', None)
                return env
        except Exception:
            pass
    if socks:
        env['ALL_PROXY'] = socks
        env['all_proxy'] = socks
    return env


def _resolve_codex_bin() -> str:
    raw = (CODEX_BIN or 'codex').strip()
    p = Path(raw)
    if p.is_file():
        return str(p)
    found = shutil.which(raw)
    if found:
        return found
    raise FileNotFoundError(f'codex_bin_missing:{raw}')


def call_codex(prompt: str, image_paths: Sequence[str], *, timeout_s: float = 180.0) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Invoke Codex CLI non-interactively; parse JSON from stdout.

    Prompt is always passed on stdin with argv `-` because `-i/--image` is
    variadic and would swallow a trailing prompt string as another image path
    (symptoms: "Reading prompt from stdin... No prompt provided").
    """
    bin_path = _resolve_codex_bin()
    cmd = [
        bin_path, 'exec',
        '--skip-git-repo-check',
        '-m', CODEX_MODEL,
        '-c', f'model_reasoning_effort="{CODEX_EFFORT}"',
    ]
    for p in list(image_paths)[:4]:
        if p and Path(p).is_file():
            cmd.extend(['-i', str(p)])
    cmd.append('-')  # read prompt from stdin (safe with variadic -i)
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=_proxy_env(),
        cwd=str(Path(__file__).resolve().parent),
    )
    wall = time.time() - t0
    out = (proc.stdout or '') + '\n' + (proc.stderr or '')
    if proc.returncode != 0 and not proc.stdout:
        raise RuntimeError(f'codex_exit_{proc.returncode}:{(proc.stderr or "")[:500]}')
    parsed = _extract_json(proc.stdout or out)
    rec = {
        'backend': 'codex',
        'model': CODEX_MODEL,
        'effort': CODEX_EFFORT,
        'wall_s': round(wall, 3),
        'cost_usd': 0.0,  # Codex CLI does not always expose; ledger separately if needed
        'returncode': proc.returncode,
        'stdout_chars': len(proc.stdout or ''),
    }
    return parsed, rec


def call_monitor(
    ctx: Dict[str, Any],
    *,
    mode: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run monitor backend; return (normalized_parsed, rec).

    Raises on hard failure; caller may fall back to Astra.
    """
    mode = backend_mode(mode)
    if mode == 'off':
        raise RuntimeError('coding_agent_slow_disabled')
    prompt = build_prompt(ctx)
    images = list(ctx.get('keyframe_paths') or [])
    # dual runs Codex for comparison; the harness separately applies Astra.
    if mode == 'dual':
        mode = 'codex'
    if mode == 'mock':
        raw = mock_intervene(ctx)
        ok, errs, norm = validate_intervention(raw)
        rec = {
            'backend': 'mock',
            'wall_s': 0.0,
            'cost_usd': 0.0,
            'validate_ok': ok,
            'validate_errs': errs,
            'dual': mode == 'dual',
        }
        if not ok:
            raise ValueError(f'mock_invalid:{errs}')
        return norm, rec
    if mode == 'codex':
        raw, rec = call_codex(prompt, images)
        ok, errs, norm = validate_intervention(raw)
        rec = dict(rec)
        rec['validate_ok'] = ok
        rec['validate_errs'] = errs
        if not ok:
            raise ValueError(f'codex_invalid:{errs}')
        return norm, rec
    raise RuntimeError(f'unknown_mode:{mode}')


def recent_diag_from_file(path: str, limit: int = 30) -> List[Any]:
    p = Path(path)
    if not p.is_file():
        return []
    lines = p.read_text(errors='replace').splitlines()
    out = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            out.append({'raw': line[:300]})
    return out


def collect_keyframe_paths(ep_dir: str, limit: int = 4) -> List[str]:
    """Prefer recent slow/relook/fb stills, then start.jpg."""
    d = Path(ep_dir)
    if not d.is_dir():
        return []
    ranked: List[Tuple[int, float, Path]] = []
    for p in d.glob('*.jpg'):
        name = p.name
        rank = 0
        if name.startswith('slow'):
            rank = 3
        elif name.startswith('relook') or name.startswith('fb') or name.startswith('escape'):
            rank = 2
        elif name.startswith('start'):
            rank = 1
        ranked.append((rank, p.stat().st_mtime, p))
    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [str(p) for _, _, p in ranked[:limit]]


def applicator_accepts(parsed: Dict[str, Any]) -> bool:
    """Mirror of harness acceptance gate used by smoke (no MuJoCo)."""
    ok, errs, norm = validate_intervention(parsed)
    if not ok:
        return False
    if not norm.get('intervene'):
        return True  # ok / no-op is valid
    plan = norm.get('plan') or []
    return all(isinstance(st, dict) and st.get('step') in ALLOWED_STEPS for st in plan)


# --- harness bridge helpers (imported only when CODING_AGENT_SLOW is on) ---

def prepare_slow_context(
    *,
    lang: str,
    trigger: str,
    slow_ctx_json: str,
    experience_brief: Any,
    patch: Dict[str, Any],
    goal_id: Any,
    ep_dir: str,
    keyframe_paths: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Capture policy-visible context on the caller thread. Never read simulator diagnostics."""
    try:
        slow_ctx = json.loads(slow_ctx_json) if isinstance(slow_ctx_json, str) else (slow_ctx_json or {})
    except json.JSONDecodeError:
        slow_ctx = {'raw': str(slow_ctx_json)[:2000]}
    diag = []
    # Also pull decision log if present next to ep_dir
    dec_log = str(Path(ep_dir) / 'decisions.jsonl')
    if Path(dec_log).is_file():
        diag = diag + recent_diag_from_file(dec_log, 20)
    ctx = pack_context(
        lang=lang,
        trigger=trigger,
        recent_diag=diag,
        experience_brief=experience_brief,
        patch=patch,
        goal_id=goal_id,
        keyframe_paths=list(keyframe_paths) if keyframe_paths is not None else collect_keyframe_paths(ep_dir, 4),
        slow_ctx=slow_ctx,
    )
    # Detach nested dicts/lists from live harness state before starting a worker.
    from copy import deepcopy
    return deepcopy(ctx)


def harness_call_from_slow(*, mode: Optional[str] = None, **kwargs):
    return call_monitor(prepare_slow_context(**kwargs), mode=mode)


# ---------------------------------------------------------------------------
# Episode-start / relook keypoint perception via coding agent
# Env CODING_AGENT_PERCEIVE=0|mock|codex (default off). Same flag covers
# perceive_start AND mid-episode relook/escape_look (anything calling
# harness perceive()). verify_grasp_identity stays on Astra.
# ---------------------------------------------------------------------------

PERCEIVE_PROMPT = """You are the perception module of a single Franka arm with a parallel gripper (top-down grasps).
Task instruction (language goal): "{lang}"
Image size: {img_size}x{img_size} px from a fixed front (agentview) camera; robot base at the TOP of the image. Pixel v grows downward.
Optional extra stills may show the wrist camera — pixel coordinates ALWAYS refer to the FRONT agentview image only.

IDENTITY FIRST (do this before locking any keypoints):
- Read the language goal and NAME the object that must be moved and the place region, using only what the instruction says plus what you see.
- If several similar cans/bottles/boxes are visible, pick the one that best matches the language phrase; if you cannot tell, set matched_language_goal to null.
- grasp_px / object_center_px MUST lie on THAT language-goal object only. place_px MUST lie on the language-goal place region only. Never lock points on a neighbour.

Phase / context: {phase}
{extra_context}

Reply ONLY one JSON object (no markdown fences):
{{
  "goal_object_named": short name of the language-goal object,
  "goal_object_where": one short phrase locating that object in THIS image (or "not visible"),
  "place_region_named": short name of the place region,
  "place_region_where": one short phrase locating that place region (or "not visible"),
  "matched_language_goal": true if grasp/object_center are on the language-goal object AND place_px on the place region; false if wrong; null if unsure,
  "grasp_px": [u, v],
  "grasp_yaw_deg": gripper rotation about vertical (-90..90; 0 = fingers close along image left-right),
  "grasp_depth_m": how far below that visible point the fingertips should go (0.00-0.06),
  "object_center_px": [u, v],
  "object_height_m": height estimate,
  "object_width_m": width across the gripper fingers,
  "place_px": [u, v],
  "place_clearance_m": height above place point to release (0.00-0.10),
  "object_held_by_gripper": true/false,
  "confidence": 0..1
}}

Uncertain ≠ proceed: never claim matched_language_goal=true when unsure.
"""


def perceive_backend_mode(raw: Optional[str] = None) -> str:
    v = (raw if raw is not None else os.environ.get('CODING_AGENT_PERCEIVE', '0')).strip().lower()
    if v in ('', '0', 'off', 'false', 'no', 'astra'):
        return 'off'
    if v in ('1', 'on', 'true', 'yes'):
        return 'mock'
    if v in ('mock', 'codex'):
        return v
    return 'off'


def _as_px(val: Any, img_size: int, errs: List[str], name: str) -> Optional[List[float]]:
    if not isinstance(val, (list, tuple)) or len(val) < 2:
        errs.append(f'{name}_missing')
        return None
    try:
        u, v = float(val[0]), float(val[1])
    except (TypeError, ValueError):
        errs.append(f'{name}_bad')
        return None
    if not (0 <= u < img_size and 0 <= v < img_size):
        errs.append(f'{name}_oob')
    return [u, v]


def validate_keypoints(parsed: Dict[str, Any], *, img_size: int = 512) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Normalize Astra-shaped keypoint JSON so _commit_keypoints path works."""
    errs: List[str] = []
    out = dict(parsed or {})
    for name in ('grasp_px', 'object_center_px', 'place_px'):
        px = _as_px(out.get(name), img_size, errs, name)
        if px is not None:
            out[name] = px
    if out.get('object_center_px') is None and out.get('grasp_px') is not None:
        out['object_center_px'] = list(out['grasp_px'])
    for k, lo, hi, default in (
        ('grasp_yaw_deg', -90.0, 90.0, 0.0),
        ('grasp_depth_m', 0.0, 0.06, 0.02),
        ('object_height_m', 0.005, 0.2, 0.03),
        ('object_width_m', 0.01, 0.2, 0.05),
        ('place_clearance_m', 0.0, 0.10, 0.04),
        ('confidence', 0.0, 1.0, 0.5),
    ):
        if k not in out or out[k] is None:
            out[k] = default
            continue
        try:
            out[k] = float(out[k])
            if out[k] < lo or out[k] > hi:
                out[k] = float(max(lo, min(hi, out[k])))
        except (TypeError, ValueError):
            errs.append(f'{k}_bad')
            out[k] = default
    m = out.get('matched_language_goal')
    if m is True or str(m).lower() == 'true':
        out['matched_language_goal'] = True
    elif m is False or str(m).lower() == 'false':
        out['matched_language_goal'] = False
    else:
        out['matched_language_goal'] = None
    held = out.get('object_held_by_gripper')
    out['object_held_by_gripper'] = held is True or str(held).lower() == 'true'
    for k in ('goal_object_named', 'goal_object_where', 'place_region_named', 'place_region_where'):
        if k not in out or out[k] is None:
            out[k] = 'unknown'
        else:
            out[k] = str(out[k])[:120]
    hard = [e for e in errs if e.endswith('_missing') or e.endswith('_bad') or e.endswith('_oob')]
    # grasp_px missing/bad is fatal; oob is soft-warn but still reject for safety
    ok = out.get('grasp_px') is not None and out.get('place_px') is not None
    ok = ok and not any(e.startswith('grasp_px_') or e.startswith('place_px_') for e in hard)
    return ok, errs, out


def mock_perceive(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic keypoints for smoke (cream-ish layout from prior LIBERO agentview)."""
    lang = str(ctx.get('language_goal') or '').lower()
    phase = str(ctx.get('phase') or 'perceive_start')
    # Default cream cheese → basket layout (approx. from prior Astra perceive_start)
    grasp = [196.0, 310.0]
    place = [416.0, 243.0]
    named = 'cream cheese'
    place_named = 'basket'
    if 'ketchup' in lang:
        grasp, named = [280.0, 300.0], 'ketchup'
    elif 'butter' in lang:
        grasp, named = [340.0, 280.0], 'butter'
    elif 'soup' in lang or 'alphabet' in lang:
        grasp, named = [150.0, 340.0], 'alphabet soup'
    matched = True
    if 'unmatched' in phase or 'force_null' in str(ctx.get('extra_context') or ''):
        matched = None
    return {
        'goal_object_named': named,
        'goal_object_where': f'{named} on the table near image centre',
        'place_region_named': place_named,
        'place_region_where': f'{place_named} toward image right',
        'matched_language_goal': matched,
        'grasp_px': grasp,
        'grasp_yaw_deg': 5.0,
        'grasp_depth_m': 0.015,
        'object_center_px': list(grasp),
        'object_height_m': 0.025,
        'object_width_m': 0.06,
        'place_px': place,
        'place_clearance_m': 0.04,
        'object_held_by_gripper': False,
        'confidence': 0.85,
    }


def build_perceive_prompt(ctx: Dict[str, Any]) -> str:
    extra = ctx.get('extra_context') or ''
    if extra:
        extra = f'Context (current robot state / why you are asked again): {extra}'
    return PERCEIVE_PROMPT.format(
        lang=ctx.get('language_goal') or '',
        img_size=int(ctx.get('img_size') or 512),
        phase=ctx.get('phase') or 'perceive_start',
        extra_context=extra,
    )


def call_perceive(
    ctx: Dict[str, Any],
    *,
    mode: Optional[str] = None,
    image_paths: Optional[Sequence[str]] = None,
    timeout_s: float = 180.0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run perceive backend; return (normalized_keypoints, rec)."""
    mode = perceive_backend_mode(mode)
    if mode == 'off':
        raise RuntimeError('coding_agent_perceive_disabled')
    img_size = int(ctx.get('img_size') or 512)
    images = list(image_paths or ctx.get('image_paths') or [])[:4]
    if mode == 'mock':
        raw = mock_perceive(ctx)
        ok, errs, norm = validate_keypoints(raw, img_size=img_size)
        rec = {
            'backend': 'mock',
            'wall_s': 0.0,
            'cost_usd': 0.0,
            'validate_ok': ok,
            'validate_errs': errs,
            'phase': ctx.get('phase'),
        }
        if not ok:
            raise ValueError(f'mock_perceive_invalid:{errs}')
        return norm, rec
    if mode == 'codex':
        prompt = build_perceive_prompt(ctx)
        raw, rec = call_codex(prompt, images, timeout_s=timeout_s)
        ok, errs, norm = validate_keypoints(raw, img_size=img_size)
        rec = dict(rec)
        rec['validate_ok'] = ok
        rec['validate_errs'] = errs
        rec['phase'] = ctx.get('phase')
        rec['backend'] = 'codex_perceive'
        if not ok:
            raise ValueError(f'codex_perceive_invalid:{errs}')
        return norm, rec
    raise RuntimeError(f'unknown_perceive_mode:{mode}')


def harness_call_perceive(
    *,
    lang: str,
    phase: str,
    image_paths: Sequence[str],
    context: Optional[str] = None,
    img_size: int = 512,
    mode: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Entry used by run_ep_fbpatch.perceive when CODING_AGENT_PERCEIVE != off."""
    ctx = {
        'language_goal': lang,
        'phase': phase,
        'extra_context': context or '',
        'img_size': img_size,
        'image_paths': list(image_paths)[:4],
    }
    return call_perceive(ctx, mode=mode, image_paths=image_paths)


# ---------------------------------------------------------------------------
# Episode-start "Great Wall" global plan (predictive; before first approach)
# Env EPISODE_START_PLAN=0|mock|codex|astra|auto
#   unset: codex. Explicit empty/auto inherits CODING_AGENT_SLOW/PERCEIVE.
#   Explicit 0/off disables planning even if CODING_AGENT_SLOW is on.
# Prefer Codex; harness may fall back to Astra when mode=astra or codex fails.
# Plan step vocabulary matches mid-episode slow-think → `_apply_episode_plan`.
# ---------------------------------------------------------------------------

ALLOWED_RISKS = frozenset({
    'neighbour_block', 'grasp_height', 'identity_ambiguous', 'place_ambiguous',
    'path_occlusion', 'tip_risk', 'other',
})
ALLOWED_PHASES = frozenset({
    'clear_neighbour', 'approach', 'grasp', 'lift', 'place', 'retreat', 'relook',
})
ALLOWED_STEP1 = frozenset({
    'push_aside', 'clear_reapproach', 'raise_redescend', 'relook', 'approach', 'none',
})

START_PLAN_PROMPT = """You are the episode-START global planner ("Great Wall") for a Jev-controlled
Franka arm (LIBERO pick/place). BEFORE any approach/grasp/place motion, emit ONE ordered
predictive long-horizon skeleton. Mid-episode slow-think only fine-tunes — YOU own the
full episode phase_order; do NOT leave place→next-object or post-place clearance to hard gates.

Language goal: {lang}

You receive:
- start stills (front agentview; optional wrist) — pixel coords refer to FRONT image
- experience_brief: library recoveries (when→prefer) + lessons (when→instead). READ them.
  Bake applicable general triggers into phase_order / diagnosis / plan (no brand hardcodes).
- start_hints: identity gate result, named goal/place from perceive, neighbour prior tip
- NO proprio contact yet (episode start) — predict from geometry / clutter / language + brief

EmbodiedEvo / RSI analogy: this is start-of-episode reflection that becomes an episode prior.
Invent predictive detail from experience_brief + the image — do not wait for stall contact.

Reply ONLY one JSON object (no markdown fences):
{{
  "matched_language_goal": true | false | null,
  "issue": one of {issues},
  "intervene": true if a pre-grasp clearance / adjust / relook is needed BEFORE approach,
  "predicted_risks": [] or subset of {risks},
  "step1_must": one of {step1},
  "phase_order": ordered phases from {phases} — complete skeleton for the WHOLE language goal,
  "diagnosis": one short sentence (what you foresee, citing brief triggers when used),
  "lesson": "" or one GENERAL qualitative sentence (NO numbers/units/object brand names),
  "plan": [] or ordered steps (max 3), each one of:
    {{"step":"relook"}},
    {{"step":"raise_redescend"}},
    {{"step":"lateral_recenter"}},
    {{"step":"change_approach_yaw"}},
    {{"step":"confirm_contact"}},
    {{"step":"clear_reapproach"}},
    {{"step":"adjust","grasp_dz_objh":-0.5..0.5,"tilt_deg":0..30,"yaw_deg":-90..90}},
    {{"step":"push_aside","blocker_px":[u,v],"direction":"away_from_target"|"toward_image_left"|"toward_image_right"|"toward_camera"|"away_from_camera","distance_objw":0.5..2.0}}
}}

Rules:
- READ experience_brief.recoveries and experience_brief.lessons first. For each applicable when→prefer / when→instead, bake it into phase_order (and diagnosis). Use GENERAL triggers only — no object-brand hardcodes in lesson text.
- After EVERY place, include retreat (lift clear of container / leave rim) BEFORE the next approach. This is predictive clearance so the arm does not tip or scrape the place region — not something for mid-episode hard gates to invent later.
- If language_goal names multiple objects / "both" / "and": phase_order MUST contain one complete cycle per object: (optional clear_neighbour) → approach → grasp → lift → place → retreat, then the next object… ending with a final retreat. Do not stop after the first place; do not omit the second place.
- Mid-episode slow-think only fine-tunes; this start plan owns the long-horizon skeleton.
- If a neighbour sits close enough that the OPEN gripper path or fingers may hit it before a clean bilateral grasp → set issue=blocked_by_neighbour, intervene=true, step1_must=push_aside (or clear_reapproach), phase_order MUST begin with clear_neighbour, and plan MUST LEAD with that step. Include blocker_px on the blocking body in the FRONT image when visible. Clearance requires start_hints.goal_id_confirmed=true; otherwise lead with relook. Your match assertion does not override the identity gate.
- Identity mismatched or unconfirmed → step1_must=relook, lead plan with relook. Conditional experience lessons are not observations of current clutter.
- Clear path + identity OK + single object → intervene=false, plan=[], step1_must=approach or none, phase_order=["approach","grasp","lift","place","retreat"].
- Episode-only: do not invent library edits or scene-specific absolute yaw angles.
- No object-brand hardcodes in lesson text.

Context JSON:
{context_json}
"""


def start_plan_backend_mode(raw: Optional[str] = None) -> str:
    """Default to Codex; explicit empty/auto inherits CODING_AGENT_SLOW/PERCEIVE."""
    if raw is None:
        raw = os.environ.get('EPISODE_START_PLAN', 'codex')
    v = (raw or '').strip().lower()
    if v in ('0', 'off', 'false', 'no'):
        return 'off'
    if v in ('mock', 'codex', 'astra'):
        return v
    if v in ('1', 'on', 'true', 'yes', 'auto', ''):
        # Inherit mid-episode coding-agent backend when pilot already on.
        slow = backend_mode(os.environ.get('CODING_AGENT_SLOW', 'codex'))
        if slow == 'dual':
            return 'codex'
        if slow in ('mock', 'codex'):
            return slow
        # Also inherit perceive=codex as a signal the pilot wants Codex at start.
        perc = perceive_backend_mode(os.environ.get('CODING_AGENT_PERCEIVE', '0'))
        if perc == 'codex':
            return 'codex'
        if perc == 'mock':
            return 'mock'
        if v in ('1', 'on', 'true', 'yes', 'auto'):
            return 'mock'  # explicit on without backend → safe mock
        return 'off'
    return 'off'


def pack_start_context(
    *,
    lang: str,
    experience_brief: Any,
    start_hints: Optional[Dict[str, Any]] = None,
    keyframe_paths: Optional[Sequence[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    paths = [str(p) for p in (keyframe_paths or []) if p][:4]
    ctx = {
        'language_goal': lang,
        'trigger': 'episode_start_plan',
        'experience_brief': experience_brief,
        'start_hints': start_hints or {},
        'keyframe_paths': paths,
    }
    if extra:
        ctx['extra'] = extra
    return ctx


def build_start_prompt(ctx: Dict[str, Any]) -> str:
    return START_PLAN_PROMPT.format(
        lang=ctx.get('language_goal') or '',
        issues=' | '.join(sorted(ALLOWED_ISSUES)),
        risks=' | '.join(sorted(ALLOWED_RISKS)),
        step1=' | '.join(sorted(ALLOWED_STEP1)),
        phases=' | '.join(sorted(ALLOWED_PHASES)),
        context_json=json.dumps(ctx, ensure_ascii=False, indent=2, default=str)[:12000],
    )


def _lang_multi_object_cues(lang: str) -> bool:
    """Heuristic: language names multiple pick targets (both / the A and the B).

    Does NOT treat single-object 'pick X and place/put it' as multi-object.
    """
    t = (lang or '').lower()
    if re.search(r'\bboth\b', t):
        return True
    # Classic dual noun phrases: "the cream cheese and the butter"
    if re.search(r'\bthe\s+[^,]{1,48}?\s+and\s+the\s+', t):
        return True
    # "A and B" where B is not a place/put verb or pronoun
    m = re.search(
        r'\band\s+(?!place\b|put\b|set\b|drop\b|move\b|it\b|them\b|that\b|this\b|then\b)([a-z0-9_\-]+)',
        t,
    )
    if m and any(k in t for k in ('place', 'put', 'pick', 'into', 'basket', 'drawer', 'cabinet')):
        return True
    return False


def _ensure_retreat_after_place(phases: List[str]) -> List[str]:
    """After every place, ensure a retreat follows before the next phase."""
    out: List[str] = []
    for i, ph in enumerate(phases):
        out.append(ph)
        if ph == 'place':
            nxt = phases[i + 1] if i + 1 < len(phases) else None
            if nxt != 'retreat':
                out.append('retreat')
    return out


def _default_object_cycle() -> List[str]:
    return ['approach', 'grasp', 'lift', 'place', 'retreat']


def validate_start_plan(
    parsed: Dict[str, Any],
    *,
    lang: Optional[str] = None,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Normalize start-plan JSON; plan steps share mid-episode vocabulary.

    Soft normalizations (do not hard-fail):
      - insert retreat after every place that lacks one
      - multi_object_incomplete_phases when lang has multi cues but <2 places
        (optional stub second cycle appended)
    """
    ok, errs, out = validate_intervention(parsed)
    lang_txt = lang if lang is not None else str(
        out.get('language_goal') or parsed.get('language_goal') or ''
    )
    # predicted_risks
    risks_raw = out.get('predicted_risks')
    if risks_raw is None:
        risks_raw = []
    if not isinstance(risks_raw, list):
        errs.append('predicted_risks_not_list')
        risks_raw = []
    risks = []
    for r in risks_raw[:6]:
        rs = str(r)
        if rs not in ALLOWED_RISKS:
            errs.append(f'bad_risk:{rs}')
            continue
        risks.append(rs)
    out['predicted_risks'] = risks
    # step1_must
    s1 = str(out.get('step1_must') or 'none')
    if s1 not in ALLOWED_STEP1:
        errs.append(f'bad_step1:{s1}')
        s1 = 'none'
    out['step1_must'] = s1
    # phase_order (allow two full cycles + clear/relook headers)
    po_raw = out.get('phase_order')
    if po_raw is None:
        po_raw = []
    if not isinstance(po_raw, list):
        errs.append('phase_order_not_list')
        po_raw = []
    phases = []
    for ph in po_raw[:16]:
        ps = str(ph)
        if ps not in ALLOWED_PHASES:
            errs.append(f'bad_phase:{ps}')
            continue
        phases.append(ps)
    if not phases:
        if s1 == 'push_aside' or (out.get('plan') and out['plan'][0].get('step') == 'push_aside'):
            phases = ['clear_neighbour'] + _default_object_cycle()
        elif s1 == 'relook' or (out.get('plan') and out['plan'] and out['plan'][0].get('step') == 'relook'):
            phases = ['relook'] + _default_object_cycle()
        else:
            phases = list(_default_object_cycle())
    # Always normalize place → retreat pairing
    before = list(phases)
    phases = _ensure_retreat_after_place(phases)[:16]
    if phases != before:
        errs.append('place_retreat_normalized')  # soft
    # Multi-object soft check / optional stub second cycle
    if _lang_multi_object_cues(lang_txt):
        n_place = sum(1 for p in phases if p == 'place')
        if n_place < 2:
            errs.append('multi_object_incomplete_phases')  # soft
            # Append a second full cycle stub so the skeleton is usable
            for ph in _default_object_cycle():
                if len(phases) >= 16:
                    break
                phases.append(ph)
            phases = _ensure_retreat_after_place(phases)[:16]
            errs.append('multi_object_cycle_stubbed')  # soft
    out['phase_order'] = phases
    # Consistency: neighbour risk / step1 push should lead plan with push/clear
    if out.get('intervene') and s1 in ('push_aside', 'clear_reapproach'):
        plan = out.get('plan') or []
        if not plan or plan[0].get('step') not in ('push_aside', 'clear_reapproach'):
            # Soft inject — applicator needs an executable first step
            lead = {'step': s1}
            if s1 == 'push_aside':
                lead['direction'] = 'away_from_target'
                lead['distance_objw'] = 1.0
            out['plan'] = [lead] + [st for st in plan if st.get('step') != s1][:2]
            if 'blocked_by_neighbour' not in (out.get('issue') or ''):
                out['issue'] = 'blocked_by_neighbour'
            errs.append('step1_plan_injected')  # soft
    hard = [e for e in errs if e.startswith('bad_') or e in (
        'plan_not_list', 'intervene_without_plan', 'predicted_risks_not_list', 'phase_order_not_list',
    )]
    # Soft markers ignored for ok
    soft = {
        'step1_plan_injected', 'stall_missing_feedback_step',
        'place_retreat_normalized', 'multi_object_incomplete_phases',
        'multi_object_cycle_stubbed',
    }
    hard = [e for e in hard if e not in soft]
    return (len(hard) == 0), errs, out


def mock_start_plan(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic start plan: observed blocker + confirmed identity → push_aside.

    Always ends each place with retreat. Multi-object language ("both" / " and ")
    gets a second approach→grasp→lift→place→retreat cycle so mock tests match Codex intent.
    """
    hints = ctx.get('start_hints') or {}
    lang = str(ctx.get('language_goal') or '')
    multi = _lang_multi_object_cues(lang)
    cycle = _default_object_cycle()
    second = list(cycle) if multi else []

    def _phases(*prefix: str) -> List[str]:
        return list(prefix) + list(cycle) + second

    # Conditional lessons and instructional hints are priors, not scene observations.
    neighbour = hints.get('neighbour_observed') is True
    id_ok = hints.get('goal_id_confirmed')
    matched = hints.get('identity_matched')
    verify = hints.get('verify_confirmed')
    clearly_bad = (
        matched is False or str(matched).lower() == 'false'
        or verify is False or str(verify).lower() == 'false'
    )
    if clearly_bad or id_ok is not True:
        return {
            'matched_language_goal': False if clearly_bad else None,
            'issue': 'wrong_object' if clearly_bad else 'other',
            'intervene': True,
            'predicted_risks': ['identity_ambiguous'],
            'step1_must': 'relook',
            'phase_order': _phases('relook'),
            'diagnosis': 'mock: start identity rejected — relook before approach',
            'lesson': 'When start identity is rejected, relook before the first approach.',
            'plan': [{'step': 'relook'}],
        }
    if neighbour:
        blocker = None
        # Optional seed px from hints (general; not task-name)
        bp = hints.get('suggested_blocker_px')
        if isinstance(bp, (list, tuple)) and len(bp) >= 2:
            try:
                blocker = [float(bp[0]), float(bp[1])]
            except (TypeError, ValueError):
                blocker = None
        step = {'step': 'push_aside', 'direction': 'away_from_target', 'distance_objw': 1.0}
        if blocker:
            step['blocker_px'] = blocker
        return {
            'matched_language_goal': True,
            'issue': 'blocked_by_neighbour',
            'intervene': True,
            'predicted_risks': ['neighbour_block'],
            'step1_must': 'push_aside',
            'phase_order': _phases('clear_neighbour'),
            'diagnosis': (
                'mock: neighbour near target — clear before first approach'
                + ('; then full cycles for each language object' if multi else '')
            ),
            'lesson': 'When a neighbour may block the grasp path, push aside or clear before approaching.',
            'plan': [step],
        }
    return {
        'matched_language_goal': True if id_ok is not False else None,
        'issue': 'ok',
        'intervene': False,
        'predicted_risks': [],
        'step1_must': 'approach',
        'phase_order': _phases(),
        'diagnosis': (
            'mock: clear path — full place cycles'
            + (' for each language object' if multi else ' then retreat')
        ),
        'lesson': '',
        'plan': [],
    }


def gate_start_plan(plan: Dict[str, Any], confirmed: bool) -> Dict[str, Any]:
    """Planning assertions cannot authorize contact with an unverified target."""
    out = dict(plan)
    match = out.get('matched_language_goal')
    if confirmed is not True or match is False or str(match).lower() == 'false':
        out.update(intervene=True, step1_must='relook', plan=[{'step': 'relook'}],
                   matched_language_goal=False if match is False else None,
                   issue='wrong_object' if match is False else 'other')
        out['phase_order'] = ['relook'] + [p for p in out.get('phase_order', []) if p not in ('relook', 'clear_neighbour')]
        out['predicted_risks'] = list(dict.fromkeys(list(out.get('predicted_risks') or []) + ['identity_ambiguous']))
    return out


def call_start_plan(
    ctx: Dict[str, Any],
    *,
    mode: Optional[str] = None,
    timeout_s: float = 180.0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run start-plan backend; return (normalized_plan, rec)."""
    mode = start_plan_backend_mode(mode)
    if mode == 'off':
        raise RuntimeError('episode_start_plan_disabled')
    images = list(ctx.get('keyframe_paths') or [])[:4]
    if mode == 'mock':
        raw = mock_start_plan(ctx)
        ok, errs, norm = validate_start_plan(raw, lang=str(ctx.get('language_goal') or ''))
        rec = {
            'backend': 'mock_start',
            'wall_s': 0.0,
            'cost_usd': 0.0,
            'validate_ok': ok,
            'validate_errs': errs,
        }
        if not ok:
            raise ValueError(f'mock_start_plan_invalid:{errs}')
        return norm, rec
    if mode == 'codex':
        prompt = build_start_prompt(ctx)
        raw, rec = call_codex(prompt, images, timeout_s=timeout_s)
        ok, errs, norm = validate_start_plan(raw, lang=str(ctx.get('language_goal') or ''))
        rec = dict(rec)
        rec['validate_ok'] = ok
        rec['validate_errs'] = errs
        rec['backend'] = 'codex_start'
        if not ok:
            raise ValueError(f'codex_start_plan_invalid:{errs}')
        return norm, rec
    if mode == 'astra':
        # Harness supplies Astra via its own client; this module only documents the mode.
        raise RuntimeError('astra_start_plan_must_be_called_from_harness')
    raise RuntimeError(f'unknown_start_plan_mode:{mode}')


def harness_call_start_plan(
    *,
    lang: str,
    experience_brief: Any,
    start_hints: Optional[Dict[str, Any]] = None,
    ep_dir: str,
    image_paths: Optional[Sequence[str]] = None,
    mode: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Entry used by run_ep_fbpatch after start identity, before first approach."""
    paths = list(image_paths or [])[:4]
    if not paths:
        # Prefer start stills already written by harness
        for name in ('start.jpg', 'start_id_verify.jpg', 'start_id_relook.jpg', 'start_wrist.jpg'):
            p = str(Path(ep_dir) / name)
            if Path(p).is_file():
                paths.append(p)
        paths = paths[:4]
    ctx = pack_start_context(
        lang=lang,
        experience_brief=experience_brief,
        start_hints=start_hints,
        keyframe_paths=paths,
    )
    return call_start_plan(ctx, mode=mode)
