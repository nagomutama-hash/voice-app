"""October test entry, kept separate from the closed legacy URLs."""
from datetime import datetime, timezone
from pathlib import Path

TEST_PREFIX = '/test-202610'
TEST_END = datetime(2026, 11, 30, 15, 0, tzinfo=timezone.utc)  # 12/1 00:00 JST
READ_PATHS = {
    TEST_PREFIX, TEST_PREFIX + '/', TEST_PREFIX + '/speed-retest',
    TEST_PREFIX + '/help/microphone', TEST_PREFIX + '/mic-test',
    TEST_PREFIX + '/api/diagnosis-config',
}


def entry_access_allowed(path, method, now=None):
    if (now or datetime.now(timezone.utc)) >= TEST_END:
        return False
    if method in ('GET', 'HEAD'):
        return path in READ_PATHS or path.startswith(TEST_PREFIX + '/static/')
    return method == 'POST' and path == TEST_PREFIX + '/analyze'


def render_entry_html(filename):
    source = (Path(__file__).resolve().parent / 'static' / filename).read_text(encoding='utf-8')
    source = source.replace('7秒無料声診断', '声診断アプリ Ver.2.0')
    for path in ('/static/', '/help/microphone', '/mic-test'):
        source = source.replace(path, TEST_PREFIX + path)
    # Only literal root links; never alter external links or JavaScript division.
    source = source.replace('href="/"', 'href="' + TEST_PREFIX + '/"')
    if filename != 'index.html':
        source = source.replace('<head>', '<head><script src="' + TEST_PREFIX + '/static/test-entry.js"></script>', 1)
    return source
