import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from diagnosis import CONFIG, build_diagnosis, draft_score, speech_timing
from main import app


def tone(seconds=2.0, amplitude=0.1):
    t = np.arange(round(seconds * 16000)) / 16000
    return amplitude * np.sin(2 * np.pi * 150 * t)


def test_prompt_mora_count():
    assert CONFIG['reading']['mora_count'] == 41
    assert CONFIG['approved_for_public'] is False
    assert CONFIG['comparison']['allow_improvement_claim'] is False


def test_pauses_do_not_change_articulation_duration():
    a = np.concatenate([tone(), np.zeros(8000), tone()])
    b = np.concatenate([np.zeros(4000), tone(), np.zeros(24000), tone(), np.zeros(4000)])
    timing_a, _, _ = speech_timing(a, 16000)
    timing_b, _, _ = speech_timing(b, 16000)
    assert timing_a['articulation_seconds'] == pytest.approx(4, abs=0.041)
    assert timing_b['articulation_seconds'] == pytest.approx(4, abs=0.041)
    assert timing_b['pause_seconds'] == pytest.approx(timing_a['pause_seconds'] + 1, abs=0.041)


def test_short_closures_are_part_of_pronunciation():
    audio = np.concatenate([tone(), np.zeros(1600), tone()])
    assert speech_timing(audio, 16000)[0]['articulation_seconds'] == 4.1


def test_lengthening_pronunciation_changes_rate():
    fast = speech_timing(tone(4), 16000)[0]['articulation_seconds']
    slow = speech_timing(tone(8), 16000)[0]['articulation_seconds']
    assert 32 / fast == 8
    assert 32 / slow == 4


@pytest.mark.parametrize('key,low,mid,high', [('brightness',400,1800,3200), ('speed',2,7,12)])
def test_target_is_not_always_louder_brighter_faster(key, low, mid, high):
    if key=='speed':
        assert draft_score(key,low)[0] is None
        assert draft_score(key,high)[0] is None
        assert draft_score(key,mid)[0] is not None
        return
    assert draft_score(key,mid)[0] > draft_score(key,low)[0]
    assert draft_score(key,mid)[0] > draft_score(key,high)[0]


def test_unconfirmed_fixed_reading_is_withheld():
    for prompt, completed in [(CONFIG['reading']['id'],False)]:
        d = build_diagnosis(tone(),16000,prompt,completed,speed_measurement={'status':'experimental','mora_per_second':6.4})
        assert d['metrics']['speed']['reference_score'] is None
        assert d['metrics']['speed']['status'] == 'unavailable'


def test_clipping_withholds_all_scores():
    d = build_diagnosis(np.clip(tone(amplitude=2),-1,1),16000,CONFIG['reading']['id'],True)
    assert all(m['reference_score'] is None for m in d['metrics'].values())


def test_preview_api_responds_with_five_metrics_and_versions(monkeypatch):
    monkeypatch.setattr('main.measure_speech_speed',lambda *args:{'status':'experimental','mora_per_second':6.4})
    output=io.BytesIO()
    sf.write(output,tone(5),16000,format='WAV')
    client=TestClient(app)
    response=client.post('/analyze',files={'file':('test.wav',output.getvalue(),'audio/wav')},data={
        'measurement_mode':'five_preview','prompt_id':CONFIG['reading']['id'],'reading_complete':'true'})
    assert response.status_code == 200, response.text
    d=response.json()['diagnosis']
    assert set(d['metrics']) == {'brightness','articulation','power','speed','resonance'}
    assert d['metrics']['speed']['raw_value'] >= 0
    assert d['measurement_version'] == CONFIG['measurement_version']
    assert d['approved_for_public'] is False
    assert client.get('/api/diagnosis-config').json()['reading']['id'] == d['prompt_id']


def test_twenty_point_total_and_missing_total():
    d = build_diagnosis(tone(5),16000,CONFIG['reading']['id'],True,speed_measurement={'status':'experimental','mora_per_second':6.4})
    assert d['score_scale'] == 20
    assert d['total_score'] == round(sum(m['reference_score'] for m in d['metrics'].values()),1)
    missing = build_diagnosis(tone(5),16000,'wrong',False,speech_rate=1.25)
    assert missing['total_score'] is None
    assert missing['metrics']['speed']['status']=='unavailable'
    assert missing['metrics']['speed']['raw_value'] is None
    assert draft_score('power', 53.103)[0] == 19
    assert draft_score('articulation', .65)[0] == 14.4
