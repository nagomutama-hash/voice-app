"""未公開の音量安定感試作。芯・支え・呼吸法は評価しない。"""
import numpy as np

VERSION = 'volume-stability-draft-4'
# ユーザー指定3例を固定。下側の200%=0点は未校正の試験用延長。
CV_ANCHORS = [0, 53.103, 74.932, 76.119, 200]
SCORE_ANCHORS = [20, 19, 15.7, 15.7, 1]

def score_from_cv(cv):
    if cv is None or not np.isfinite(cv) or cv < 0:
        return None
    return round(float(np.interp(cv, CV_ANCHORS, SCORE_ANCHORS)), 1)

def measure(audio, sr):
    y = np.asarray(audio, dtype=float)
    result = {'version': VERSION, 'approved_for_public': False,
              'status': 'unavailable', 'reference_score': None,
              'rms_cv_percent': None, 'reason': 'invalid_or_insufficient_audio'}
    if y.ndim != 1 or sr < 8000 or not np.isfinite(y).all() or len(y) < sr:
        return result
    if np.mean(np.abs(y) >= .999) > .01:
        return dict(result, reason='clipping')
    hop = round(sr * .1)
    n = len(y) // hop
    rms = np.sqrt(np.mean(y[:n * hop].reshape(n, hop) ** 2, axis=1))
    if np.percentile(rms, 95) < 1e-7:
        return result
    db = 20 * np.log10(np.maximum(rms, 1e-15))
    mask = db >= np.percentile(db, 95) - 30
    selected = rms[mask]
    seconds = float(mask.sum() * hop / sr)
    if seconds < 1:
        return result
    cv = float(np.std(selected) / np.mean(selected) * 100)
    return dict(result, status='provisional', reason=None,
                rms_cv_percent=round(cv, 3), reference_score=score_from_cv(cv),
                included_seconds=round(seconds, 3), score_scale=20,
                note='relative_level_gate_is_not_speech_recognition')
