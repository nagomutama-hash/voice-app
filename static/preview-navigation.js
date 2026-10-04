(function () {
    'use strict';
    // Preserve only the known preview mode, never arbitrary return URLs or query data.
    if (new URLSearchParams(location.search).get('preview') !== 'five') return;
    const paths = new Set(['/', '/help/microphone', '/mic-test']);
    for (const link of document.querySelectorAll('a[href]')) {
        const href = link.getAttribute('href');
        if (paths.has(href)) link.setAttribute('href', href + '?preview=five');
    }
})();
