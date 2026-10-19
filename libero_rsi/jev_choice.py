"""Bundled TypeSafe Jev 1.14 image-capable choice transport."""
import json
import math
import os
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

MODEL = 'jev-1.14.0'
ENDPOINT = 'https://api.typesafe.ai/v1/systemone'


def choose_questions(state, questions, *, assets):
    """Ask independent Choice questions against one current observation."""
    key = os.environ.get('TYPESAFE_API_KEY', '').strip()
    if not key:
        raise RuntimeError('Set TYPESAFE_API_KEY for TypeSafe Jev 1.14')
    if not questions or not assets or any(not q.get('criteria') for q in questions.values()):
        raise ValueError('Jev decisions require current camera assets and candidates')
    body = {'model': MODEL, 'state': state, 'assets': assets,
            'questions': questions}
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
    answers = data.get('answers')
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise ValueError('TypeSafe response does not match question set')
    for name, question in questions.items():
        answer = answers[name]
        if not isinstance(answer, dict):
            raise ValueError('Invalid TypeSafe choice answer')
        choice, probabilities = answer.get('choice'), answer.get('probabilities')
        criteria = question['criteria']
        if (not isinstance(choice, str) or choice not in criteria
                or not isinstance(probabilities, dict) or set(probabilities) != set(criteria)):
            raise ValueError('TypeSafe response does not match candidate set')
        if any(isinstance(p, bool) or not isinstance(p, (int, float))
               or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
            raise ValueError('Invalid TypeSafe choice probabilities')
    return answers, time.monotonic() - start


def choose(state, instructions, criteria, *, assets, meta=None):
    answers, latency = choose_questions(state, {
        'next_intent': {'type': 'choice', 'instructions': instructions, 'criteria': criteria},
    }, assets=assets)
    answer = answers['next_intent']
    return answer['choice'], answer['probabilities'], latency
