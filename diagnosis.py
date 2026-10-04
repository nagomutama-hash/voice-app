"""5項目の検証用測定。仮の換算値を公開用の診断基準として扱わない。"""
import json
from pathlib import Path

import librosa
import numpy as np
from auxiliary_measurements import extract_auxiliary, unavailable
from volume_stability_draft import measure as measure_stability, score_from_cv

CONFIG = json.loads((Path(__file__).parent / 'config' / 'diagnosis_draft.json').read_text(encoding='utf-8'))


from speech_timing import speech_timing


def draft_score(metric, raw):
    if raw is None or not np.isfinite(raw):
        return None, 'unknown'
    definition = CONFIG['metrics'][metric]
    if metric == 'speed' and definition.get('method') == 'teacher_piecewise_linear':
        low, ideal, high = (definition['anchors'][k] for k in ('slow','ideal','fast'))
        a,b = (low,ideal) if raw <= ideal[0] else (ideal,high)
        score = a[1] + (raw-a[0])*(b[1]-a[1])/(b[0]-a[0])
        # 丸め前で判定。6.99が表示丸めで7.0になっても保留する。
        if score < definition['hold_below_score'] - 1e-10:
            return None, 'unknown'
        lower,upper = definition['target_range']
        return round(float(np.clip(score,0,20)),1), 'slow' if raw < lower else 'fast' if raw > upper else 'optimal'
    if metric == 'power':
        return score_from_cv(raw), 'within_reference' if raw <= definition['target_range'][1] else 'low'
    score = round(float(np.interp(raw, definition['raw_breakpoints'], definition['score_breakpoints'])), 1)
    low, high = definition['target_range']
    direction = 'low' if raw < low else 'high' if raw > high else 'within_reference'
    if metric == 'speed':
        direction = {'low': 'slow', 'high': 'fast', 'within_reference': 'optimal'}[direction]
    return score, direction


def build_diagnosis(audio, sr, prompt_id, reading_complete, speech_rate=None, speed_measurement=None):
    timing, active, rms = speech_timing(audio, sr)
    hop = round(sr * 0.02)
    active_samples = np.repeat(active, hop)[:len(audio)]
    signal = audio[active_samples]
    if not len(signal):
        raise ValueError('No active signal')
    # 周波数分析では区間の連結による偽の立ち上がりを避け、元の時間列から選ぶ。
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop, n_fft=1024)[0][:len(active)]
    flatness = librosa.feature.spectral_flatness(y=audio, hop_length=hop, n_fft=1024)[0][:len(active)]
    crossing = librosa.feature.zero_crossing_rate(y=audio, hop_length=hop, frame_length=1024)[0][:len(active)]
    onset = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset, sr=sr, hop_length=hop, units='frames')
    onset_frames = onset_frames[onset_frames < len(active)]
    onset_rate = float(np.sum(active[onset_frames])) / max(timing['articulation_seconds'], 0.001)
    selected_rms = rms[active]
    modulation = float(np.std(selected_rms) / max(np.mean(selected_rms), 1e-8) * 100)
    crossing_value = float(np.mean(crossing[active[:len(crossing)]]))
    flatness_value = float(np.mean(flatness[active[:len(flatness)]]))
    unit = lambda value, low, high: float(np.clip((value - low) / (high - low), 0, 1))
    articulation_index = (
        unit(onset_rate, 0.4, 4.5) * 0.35 + unit(crossing_value, 0.015, 0.16) * 0.25
        + unit(flatness_value, 0.002, 0.10) * 0.20 + unit(modulation, 10, 80) * 0.20
    )
    harmonic = librosa.effects.harmonic(audio)
    harmonic_ratio = float(np.mean(harmonic[active_samples] ** 2) / max(np.mean(signal ** 2), 1e-12) * 100)
    stability = measure_stability(audio, sr)
    # 旧区切り回数へフォールバックしない。未認識は速度だけ保留。
    speed_measurement = speed_measurement or {'status':'unavailable','reason':'asr_unavailable'}
    speed = speed_measurement.get('mora_per_second') if speed_measurement.get('status')=='experimental' else None
    if speed is not None and (not isinstance(speed,(int,float)) or not np.isfinite(speed) or not 1<=speed<=20):
        speed=None
    raw = {
        'brightness': float(np.mean(centroid[active[:len(centroid)]])),
        'articulation': articulation_index,
        'power': stability['rms_cv_percent'],
        'speed': speed,
        'resonance': harmonic_ratio,
    }
    clipped = float(np.mean(np.abs(audio) >= 0.999))
    metrics = {}
    for key, value in raw.items():
        score, direction = draft_score(key, value)
        reason = speed_measurement.get('reason','recognition_failed') if key == 'speed' and score is None else None
        if key == 'speed' and value is not None and score is None:
            reason = 'speed_out_of_reference'
        if key == 'speed' and prompt_id == CONFIG['speed_retest']['id'] and not reading_complete:
            score, direction, reason = None, 'unknown', 'reading_not_confirmed'
        if key == 'power' and score is None:
            reason = stability['reason']
        if clipped > 0.01:
            score, direction, reason = None, 'unknown', 'clipping'
        metrics[key] = {
            'raw_value': round(value, 6) if value is not None else None,
            'reference_score': score, 'direction': direction,
            'status': 'unavailable' if score is None else 'provisional',
            'reason': reason,
        }
    try:
        type_auxiliary = extract_auxiliary(audio, sr, active, timing)
    except Exception:
        # 補助測定だけの失敗で既存の診断成功を失わせない。
        type_auxiliary = unavailable('measurement_failed')
    return {
        'schema_version': CONFIG['schema_version'],
        'measurement_version': CONFIG['measurement_version'],
        'calibration_version': CONFIG['calibration_version'],
        'score_scale': CONFIG['score_scale'],
        'total_score': round(sum(m['reference_score'] for m in metrics.values()), 1) if all(m['reference_score'] is not None for m in metrics.values()) else None,
        'approved_for_public': False,
        'prompt_id': CONFIG['speed_retest']['id'] if prompt_id == CONFIG['speed_retest']['id'] else CONFIG['reading']['id'],
        'reading_complete': bool(reading_complete) if prompt_id == CONFIG['speed_retest']['id'] else False,
        'metrics': metrics, 'timing': timing,
        'type_auxiliary': type_auxiliary,
        'auxiliary': {'onset_rate': round(onset_rate, 4), 'zero_crossing': round(crossing_value, 6),
                      'spectral_flatness': round(flatness_value, 6), 'rms_cv': round(modulation, 4),
                      'clipped_fraction': round(clipped, 6)},
    }
