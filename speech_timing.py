"""Shared timing calculation without loading the other voice metrics."""
import numpy as np


def speech_timing(audio, sr):
    """20msの音量枠で前後無音と180ms以上の内部休止を除く近似。音声認識ではない。"""
    hop = max(1, round(sr * 0.02))
    count = int(np.ceil(len(audio) / hop))
    padded = np.pad(audio, (0, count * hop - len(audio)))
    frames = padded.reshape(count, hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    threshold = max(10 ** (-55 / 20), float(np.percentile(rms, 95)) * 10 ** (-30 / 20))
    active = rms >= threshold
    # 短い子音の閉鎖・小音量部分は「間」に数えず、発音時間に含める。
    indices = np.flatnonzero(active)
    if not len(indices):
        return {'articulation_seconds': 0.0, 'span_seconds': 0.0, 'pause_seconds': 0.0}, active, rms
    first, last = int(indices[0]), int(indices[-1])
    for left, right in zip(indices[:-1], indices[1:]):
        if (right - left - 1) * hop / sr < 0.18:
            active[left:right + 1] = True
    durations = np.minimum(hop, np.maximum(0, len(audio) - np.arange(count) * hop)) / sr
    articulation = float(np.sum(durations[active]))
    span = float(np.sum(durations[first:last + 1]))
    return {
        'articulation_seconds': round(articulation, 4),
        'span_seconds': round(span, 4),
        'pause_seconds': round(max(0, span - articulation), 4),
    }, active, rms
