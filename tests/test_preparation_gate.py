from fastapi.testclient import TestClient
import main


def test_old_entries_and_direct_api_are_closed(monkeypatch):
    monkeypatch.setattr(main, 'ACCESS_CLOSED', True)
    client = TestClient(main.app)
    for path in ['/', '/?preview=five', '/speed-retest?preview=five', '/api/diagnosis-config', '/docs']:
        response = client.get(path)
        assert '準備中' in response.text
        assert response.headers['cache-control'] == 'no-store'
        assert response.headers['permissions-policy'] == 'microphone=()'
    for path in ['/analyze', '/diagnose']:
        response = client.post(path)
        assert response.status_code == 403
        assert response.json()['code'] == 'test_closed'
