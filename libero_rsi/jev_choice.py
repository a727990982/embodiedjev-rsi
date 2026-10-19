"""Bundled TypeSafe Jev 1.14 image-capable choice transport."""
import json
import math
import os
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

MODEL = 'jev-1.14.0'
ENDPOINT = 'https://api.typesafe.ai/v1/systemone'


def choose(state, instructions, criteria, *, assets, meta=None):
    key = os.environ.get('TYPESAFE_API_KEY', '').strip()
    if not key:
        raise RuntimeError('Set TYPESAFE_API_KEY for TypeSafe Jev 1.14')
    if not criteria or not assets:
        raise ValueError('Jev decisions require current camera assets and candidates')
    body = {'model': MODEL, 'state': state, 'assets': assets,
            'questions': {'next_intent': {'type': 'choice', 'instructions': instructions,
                                          'criteria': criteria}}}
    request = Request(ENDPOINT, data=json.dumps(body, allow_nan=False).encode(), headers={
        'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json',
        'TypeSafe-Danger-Zone-Images': 'true',
    })
    start = time.monotonic()
    try:
        with urlopen(request, timeout=60) as response:
            data = json.loads(response.read())
    except HTTPError as exc:
        # The service response is not needed to diagnose authentication/status;
        # avoid printing credentials or large model inputs in tracebacks.
        raise RuntimeError(f'TypeSafe HTTP {exc.code}') from None
    except Exception as exc:
        raise RuntimeError(f'TypeSafe transport failed: {type(exc).__name__}') from None
    if data.get('model') != MODEL:
        raise ValueError('Unexpected TypeSafe response model')
    answer = data['answers']['next_intent']
    choice, probabilities = answer['choice'], answer['probabilities']
    if choice not in criteria or set(probabilities) != set(criteria):
        raise ValueError('TypeSafe response does not match candidate set')
    if any(isinstance(p, bool) or not isinstance(p, (int, float))
           or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
        raise ValueError('Invalid TypeSafe choice probabilities')
    return choice, probabilities, time.monotonic() - start
