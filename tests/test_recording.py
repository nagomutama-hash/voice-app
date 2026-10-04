import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


def wav(seconds=5, amplitude=0.1):
    t = np.arange(int(seconds * 16000)) / 16000
    signal = amplitude * np.sin(2 * np.pi * 150 * t)
    output = io.BytesIO()
    sf.write(output, signal, 16000, format="WAV")
    return output.getvalue()


def analyze(data):
    return client.post('/analyze', files={'file': ('recording.wav', data, 'audio/wav')})


@pytest.mark.parametrize('data,code,status', [
    (b'', 'empty_file', 422),
    (b'not audio', 'invalid_audio', 422),
    (wav(0.5), 'too_short', 422),
    (wav(22), 'too_long', 422),
    (wav(5, 0), 'silence', 422),
    (wav(5, 0.0001), 'silence', 422),
    (b'x' * (main.MAX_AUDIO_BYTES + 1), 'file_too_large', 413),
], ids=['empty', 'invalid', 'short', 'long', 'silent', 'quiet', 'oversize'])
def test_invalid_recording_has_no_scores(data, code, status):
    response = analyze(data)
    assert response.status_code == status
    assert response.json()['code'] == code
    assert response.json()['success'] is False
    assert 'stats' not in response.json()
    assert 'detail' not in response.json()


def test_pitch_failure_is_not_a_success(monkeypatch):
    monkeypatch.setattr(main.librosa, 'yin', lambda *args, **kwargs: np.zeros(126))
    assert analyze(wav()).json()['code'] == 'no_voice'


def test_just_under_five_seconds_is_rejected():
    assert analyze(wav(4.999)).json()['code'] == 'too_short'


def test_brief_voice_in_long_silence_is_rejected(monkeypatch):
    def brief_pitch(audio, **kwargs):
        values = np.zeros(len(audio) // 256 + 1)
        values[:20] = 150
        return values
    monkeypatch.setattr(main.librosa, 'yin', brief_pitch)
    assert analyze(wav(10)).json()['code'] == 'no_voice'


def test_valid_audio_still_returns_original_six_scores():
    response = analyze(wav())
    assert response.status_code == 200, response.text
    data = response.json()
    stats = data['stats']
    assert stats['has_pitch']
    assert stats['scores'] == main._compute_voice_scores(
        stats['pitch_std'], stats['mean_hz'], stats['rms_cv'],
        stats['brightness_hz'], stats['harmonic_ratio'], stats['speech_rate'], stats['rms_trend'],
    )
    assert set(stats['scores']) == {'intonation', 'dynamics', 'brightness', 'resonance', 'tempo', 'sustain'}


def test_internal_error_does_not_expose_traceback(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError('private-server-details')
    monkeypatch.setattr(main.librosa, 'load', fail)
    response = analyze(wav())
    assert response.status_code == 500
    assert response.json()['code'] == 'analysis_failed'
    assert 'private-server-details' not in response.text


@pytest.mark.parametrize('path', ['/', '/speed-retest', '/help/microphone', '/mic-test'])
def test_pages_and_microphone_policy(path):
    response = client.get(path)
    assert response.status_code == 200
    assert response.headers['permissions-policy'] == 'microphone=(self)'
    if path not in ['/', '/speed-retest']:
        assert 'googletagmanager' not in response.text
        assert 'fbq(' not in response.text


def test_fixed_reading_never_calls_asr(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('ASR must not be used')
    monkeypatch.setattr(main, 'measure_speech_speed', forbidden)
    fields = {'measurement_mode': 'five_preview', 'prompt_id': main.DIAGNOSIS_CONFIG['reading']['id']}
    for completed in [False, True]:
        response = client.post('/analyze', data={**fields, 'reading_complete': str(completed).lower()},
                              files={'file': ('recording.wav', wav(6), 'audio/wav')})
        assert response.status_code == 200
        metric = response.json()['diagnosis']['metrics']['speed']
        if completed:
            assert metric['raw_value'] == pytest.approx(41 / 6, abs=1e-5)
            assert metric['reference_score'] is not None
        else:
            assert metric['reference_score'] is None
            assert metric['reason'] == 'reading_not_confirmed'
