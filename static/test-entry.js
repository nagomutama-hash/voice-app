(function (root) {
    'use strict';
    const prefix = '/test-202610';
    const path = typeof location === 'undefined' ? '' : location.pathname || '';
    const active = path === prefix || path.startsWith(prefix + '/');
    root.voiceAppPath = value => active && value.startsWith('/') && !value.startsWith('//')
        && value !== prefix && !value.startsWith(prefix + '/') ? prefix + value : value;
    root.VOICE_TEST_ENTRY = active;
})(globalThis);
