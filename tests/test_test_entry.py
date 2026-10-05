from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient
import pytest
import main
from test_entry import TEST_PREFIX, TEST_END, entry_access_allowed, render_entry_html


@pytest.fixture(autouse=True)
def enable_local_route_tests(monkeypatch):
    # Override the ordinary local-test exemption: these tests exercise the public gate.
    monkeypatch.setattr(main, 'ACCESS_CLOSED', True)


def test_closed_old_urls_and_apis_are_not_reopened():
    client = TestClient(main.app)
    for path in ['/', '/?preview=five', '/speed-retest?preview=five',
                 '/api/diagnosis-config', '/static/index.html', '/mic-test',
                 '/help/microphone', '/docs']:
        response = client.get(path)
        assert 'このURLからのご利用は終了しました' in response.text
        assert response.headers['permissions-policy'] == 'microphone=()'
    for path in ['/analyze', '/advice', '/test-202610-extra/analyze']:
        assert client.post(path).status_code == 403


def test_new_entry_assets_config_and_microphone_links():
    client = TestClient(main.app)
    for path in ['', '/', '/speed-retest', '/help/microphone', '/mic-test']:
        response = client.get(TEST_PREFIX + path)
        assert response.status_code == 200
        assert response.headers['permissions-policy'] == 'microphone=(self)'
        assert 'src="/static/' not in response.text
        assert 'href="/"' not in response.text
        assert TEST_PREFIX + '/static/test-entry.js' in response.text
    assert client.get(TEST_PREFIX + '/static/test-entry.js').status_code == 200
    assert client.get(TEST_PREFIX + '/api/diagnosis-config').json()['score_scale'] == 20


def test_new_analysis_only_accepts_fixed_reading_mode(monkeypatch):
    client = TestClient(main.app)
    called = []
    monkeypatch.setattr(main, 'analyze_audio', lambda *args: called.append(args) or {'success': True})
    file = {'file': ('test.wav', b'placeholder', 'audio/wav')}
    assert client.post(TEST_PREFIX + '/analyze', files=file,
                       data={'measurement_mode': 'legacy'}).status_code == 422
    assert called == []
    response = client.post(TEST_PREFIX + '/analyze', files=file,
        data={'measurement_mode': 'five_preview', 'prompt_id': 'prompt', 'reading_complete': 'true'})
    assert response.json()['success']
    assert called[0][1:] == ('five_preview', 'prompt', True)


def test_october_deadline_and_exact_namespace():
    before = datetime(2026, 10, 31, 14, 59, 59, tzinfo=timezone.utc)
    assert entry_access_allowed(TEST_PREFIX, 'GET', before)
    assert entry_access_allowed(TEST_PREFIX + '/analyze', 'POST', before)
    for path in [TEST_PREFIX, TEST_PREFIX + '/analyze', TEST_PREFIX + '/static/index.html']:
        for method in ['GET', 'POST']:
            assert not entry_access_allowed(path, method, TEST_END)
    for path in ['/', TEST_PREFIX + '-other', TEST_PREFIX + '/advice', TEST_PREFIX + '/docs']:
        assert not entry_access_allowed(path, 'GET', before)


def test_expired_http_entry_is_closed(monkeypatch):
    monkeypatch.setattr(main, 'entry_access_allowed', lambda *args: False)
    client = TestClient(main.app)
    assert client.post(TEST_PREFIX + '/analyze').status_code == 403
    response = client.get(TEST_PREFIX)
    assert response.headers['permissions-policy'] == 'microphone=()'
