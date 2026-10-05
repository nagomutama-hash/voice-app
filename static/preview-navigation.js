(function () {
    'use strict';
    // Preserve only the known preview mode, never arbitrary return URLs or query data.
    const prefix = '/test-202610';
    const pathname = location.pathname || '';
    const testEntry = pathname === prefix || pathname.startsWith(prefix + '/');
    if (!testEntry && new URLSearchParams(location.search).get('preview') !== 'five') return;
    const paths = new Set(['/', '/help/microphone', '/mic-test']);
    for (const link of document.querySelectorAll('a[href]')) {
        const href = link.getAttribute('href');
        const original = testEntry && href.startsWith(prefix + '/') ? href.slice(prefix.length) : href;
        if (paths.has(original)) link.setAttribute('href', (testEntry ? prefix : '') + original + '?preview=five');
    }
})();
