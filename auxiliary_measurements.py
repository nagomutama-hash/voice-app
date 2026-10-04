"""分類検証用の非表示測定。値を力み・感情・自然さの点数にしない。"""
import librosa
import numpy as np

VERSION = 'type-auxiliary-experimental-1'


def unavailable(reason):
    return {'version': VERSION, 'status': 'unavailable', 'reason': reason,
            'classification_enabled': False, 'pitch': None, 'intensity': None,
            'spectrum': None, 'h1_h2': {'status': 'deferred', 'reason': 'vowel_and_formant_correction_not_validated'}}


def extract_auxiliary(audio, sr, active, timing):
    """音声・有声区間不足は保留。入力も既存測定も変更しない。"""
    audio = np.asarray(audio, dtype=float)
    if (audio.ndim != 1 or len(audio) < 2048 or not np.all(np.isfinite(audio))
            or sr < 8000 or not len(active)):
        return unavailable('invalid_or_short_input')
    if np.mean(np.abs(audio) >= .999) > .01:
        return unavailable('clipping')
    if timing.get('articulation_seconds', 0) < .3:
        return unavailable('insufficient_active_audio')
    hop, size = max(1, round(sr * .02)), 2048
    frames = librosa.util.frame(np.pad(audio, (size // 2, size // 2)),
                               frame_length=size, hop_length=hop)
    count = min(frames.shape[1], len(active))
    frames = frames[:, :count]
    rms = np.sqrt(np.mean(frames ** 2, axis=0))
    audible = np.asarray(active[:count], dtype=bool) & (rms >= 10 ** (-55 / 20))
    if np.sum(audible) * hop / sr < .3:
        return unavailable('insufficient_active_audio')
    f0 = librosa.yin(audio, sr=sr, fmin=60, fmax=min(1000, sr / 4),
                     frame_length=size, hop_length=hop)[:count]
    centered = frames - np.mean(frames, axis=0)
    acf = librosa.autocorrelate(centered, max_size=size, axis=0)
    lags = np.clip(np.rint(sr / np.maximum(f0, 1)).astype(int), 1, size - 1)
    periodicity = acf[lags, np.arange(count)] / np.maximum(acf[0], 1e-15)
    voiced = audible & np.isfinite(f0) & (f0 > 60) & (f0 < min(1000, sr / 4)) & (periodicity >= .65)
    voiced_seconds = float(np.sum(voiced) * hop / sr)
    result = unavailable('insufficient_periodic_audio')
    result.update(status='partial', intensity={
        'active_level_range_db': round(float(np.diff(np.percentile(20 * np.log10(rms[audible]), [10, 90]))[0]), 4),
        'active_seconds': round(float(np.sum(audible) * hop / sr), 4),
        'pause_seconds': timing.get('pause_seconds'),
    })
    # 0.65と0.3秒は測定品質の暫定ガードであり、タイプ分類の閾値ではない。
    if voiced_seconds < .3:
        return result
    pitches = f0[voiced]
    low, high = np.percentile(pitches, [10, 90])
    result.update(status='experimental', reason=None, pitch={
        'p10_hz': round(float(low), 4), 'p90_hz': round(float(high), 4),
        'range_semitones': round(float(12 * np.log2(high / low)), 4),
        'median_hz': round(float(np.median(pitches)), 4),
        'voiced_seconds': round(voiced_seconds, 4),
        'periodicity_median': round(float(np.median(periodicity[voiced])), 4),
        'trace': [{'seconds': round(float(i * hop / sr), 4), 'hz': round(float(f0[i]), 3)}
                  for i in np.flatnonzero(voiced)[::max(1, int(np.ceil(np.sum(voiced) / 80)))]],
    })
    # 声道の共鳴を補正していないスペクトル特徴。声帯音源や力みの測定とは呼ばない。
    power = np.abs(np.fft.rfft(centered[:, voiced] * np.hanning(size)[:, None], axis=0)) ** 2
    freq = np.fft.rfftfreq(size, 1 / sr)
    low_band, high_band = (freq >= 300) & (freq < 1500), (freq >= 1500) & (freq <= 4000)
    ratio = 10 * np.log10(np.maximum(np.sum(power[high_band], axis=0), 1e-20) /
                         np.maximum(np.sum(power[low_band], axis=0), 1e-20))
    result['spectrum'] = {'high_to_low_energy_db_median': round(float(np.median(ratio)), 4),
                          'low_band_hz': [300, 1500], 'high_band_hz': [1500, 4000],
                          'formant_corrected': False}
    return result
