"""LIBERO copy of rsi_mujoco_v2 dsl (same validator, LIBERO observed vars).
v2 restricted expression DSL + DIMENSIONAL validator (no hard-coded scene lengths).
Every length must come from an observed quantity; numeric literals may only be dimensionless factors
(multiplying/dividing an observed length, or compared with ratios/counts). 0 is allowed anywhere."""
import ast, re
import numpy as np

ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Call, ast.Name,
           ast.Load, ast.Constant, ast.Subscript, ast.Index, ast.Slice, ast.Tuple, ast.List,
           ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
           ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.Pow, ast.Mod)

def vec(x, y, z): return np.array([float(x), float(y), float(z)])
def dist(a, b): return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))
def dist_xy(a, b): return float(np.linalg.norm(np.asarray(a, float)[:2] - np.asarray(b, float)[:2]))
def max_abs_xy(a, b): return float(np.max(np.abs(np.asarray(a, float)[:2] - np.asarray(b, float)[:2])))
FUNCS = {'vec': vec, 'dist': dist, 'dist_xy': dist_xy, 'max_abs_xy': max_abs_xy,
         'max': max, 'min': min, 'abs': abs, 'float': float, 'bool': bool}

# observed quantities (all from proprioception, tactile proxy, Astra episode-start perception, or task statement)
# LIBERO: proprioception (tcp, opening, opening_at_close, tcp_at_close), robot spec (open_max),
# Astra episode-start / re-look perception (grasp, objc, place, obj_h, clear). No simulator object poses.
LENGTH_VARS = ('tcp', 'grasp', 'objc', 'place', 'opening', 'opening_at_close', 'tcp_at_close', 'obj_h', 'obj_w', 'clear', 'open_max')
COUNT_VARS = (
    'n_same', 'step', 'n_close', 'n_relook', 'clear_reapproach_n', 'push_aside_n',
    'place_lower_stall_n', 'place_lift_lower_cycles',
)
BOOL_VARS = ('contact', 'left_contact', 'right_contact', 'ever_contact', 'closed_cmd', 'released', 'seen_held')
STR_VARS = ('prev', 'last_result')
STATE_VARS = LENGTH_VARS + COUNT_VARS + BOOL_VARS + STR_VARS

class DimError(ValueError):
    pass

# dims: 'L' length, 'N' dimensionless/count, 'C' bare numeric literal (nonzero), 'Z' zero literal, 'B' bool, 'S' string, 'X' other (L^2, 1/L)
def _unify_add(a, b, expr):
    s = {a, b}
    if 'L' in s and 'C' in s:
        raise DimError(f'bare length constant added/compared to a length in {expr!r}: write it as k*<observed scale> (e.g. 0.5*obj_h)')
    if 'X' in s: return 'X'
    if 'L' in s: return 'L'
    if s <= {'C', 'Z'}: return 'C' if 'C' in s else 'Z'
    return 'N' if s & {'N', 'C', 'Z'} else (a if a == b else 'X')

def dim(n, expr, preds):
    if isinstance(n, ast.Expression): return dim(n.body, expr, preds)
    if isinstance(n, ast.Constant):
        if isinstance(n.value, str): return 'S'
        if isinstance(n.value, bool): return 'B'
        return 'Z' if n.value == 0 else 'C'
    if isinstance(n, ast.Name):
        if n.id in LENGTH_VARS: return 'L'
        if n.id in COUNT_VARS: return 'N'
        if n.id in BOOL_VARS or n.id in preds: return 'B'
        if n.id in STR_VARS: return 'S'
        raise DimError(f'unknown name {n.id!r}')
    if isinstance(n, ast.Subscript):
        return dim(n.value, expr, preds)
    if isinstance(n, ast.UnaryOp):
        d = dim(n.operand, expr, preds); return 'B' if isinstance(n.op, ast.Not) else d
    if isinstance(n, ast.BoolOp):
        for v in n.values: dim(v, expr, preds)
        return 'B'
    if isinstance(n, ast.Compare):
        ds = [dim(n.left, expr, preds)] + [dim(c, expr, preds) for c in n.comparators]
        for a, b in zip(ds, ds[1:]): _unify_add(a, b, expr)
        return 'B'
    if isinstance(n, ast.IfExp):
        dim(n.test, expr, preds); return _unify_add(dim(n.body, expr, preds), dim(n.orelse, expr, preds), expr)
    if isinstance(n, ast.BinOp):
        a, b = dim(n.left, expr, preds), dim(n.right, expr, preds)
        if isinstance(n.op, (ast.Add, ast.Sub, ast.Mod)): return _unify_add(a, b, expr)
        if isinstance(n.op, ast.Mult):
            if 'L' in (a, b) and not ({a, b} <= {'L', 'C', 'Z', 'N'}): return 'X'
            if a == 'L' and b == 'L': return 'X'
            if 'L' in (a, b): return 'L'
            return _unify_add(a, b, expr) if 'X' not in (a, b) else 'X'
        if isinstance(n.op, ast.Div):
            if a == 'L' and b == 'L': return 'N'
            if a == 'L': return 'L'
            if b == 'L': return 'X'
            return 'N' if 'N' in (a, b) else a
        if isinstance(n.op, ast.Pow):
            return 'X' if a == 'L' else ('N' if a == 'N' else a)
    if isinstance(n, ast.Call):
        f = n.func.id; ds = [dim(x, expr, preds) for x in n.args]
        if f == 'vec':
            for d in ds:
                if d == 'C': raise DimError(f'bare length constant inside vec(...) in {expr!r}: use 0 or k*<observed scale>')
            return 'L'
        if f in ('dist', 'dist_xy', 'max_abs_xy'): return 'L'
        if f in ('max', 'min'):
            d = ds[0]
            for e in ds[1:]: d = _unify_add(d, e, expr)
            return d
        if f in ('abs', 'float'): return ds[0]
        if f == 'bool': return 'B'
    if isinstance(n, (ast.Tuple, ast.List)):
        for e in n.elts: dim(e, expr, preds)
        return 'X'
    return 'X'

def check(expr, extra_names=()):
    tree = ast.parse(expr, mode='eval')
    for n in ast.walk(tree):
        if not isinstance(n, ALLOWED):
            raise ValueError(f'disallowed syntax {type(n).__name__} in {expr!r}')
        if isinstance(n, ast.Name) and n.id not in FUNCS and n.id not in STATE_VARS and n.id not in extra_names:
            raise ValueError(f'unknown name {n.id!r} in {expr!r}')
        if isinstance(n, ast.Call) and not (isinstance(n.func, ast.Name) and n.func.id in FUNCS):
            raise ValueError(f'disallowed call in {expr!r}')
    return compile(tree, '<lib>', 'eval'), tree

def check_dims(expr, preds=()):
    _, tree = check(expr, extra_names=tuple(preds))
    return dim(tree, expr, set(preds))

_NUM_TEXT = re.compile(r'(\d*\.\d+)|(\d+\s*(m|cm|mm|metre|meter)s?\b)', re.I)
def check_text(t, where):
    m = _NUM_TEXT.search(t or '')
    if m:
        raise DimError(f'{where}: text contains a numeric length/threshold {m.group(0)!r}; express thresholds as predicates over observed quantities and refer to them by name')

def evaluate(expr, ns):
    code, _ = check(expr, extra_names=tuple(k for k in ns if k not in FUNCS))
    return eval(code, {'__builtins__': {}}, {**FUNCS, **ns})
