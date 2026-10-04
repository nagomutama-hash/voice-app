(function (root) {
    'use strict';
    function stopStream(stream) {
        stream?.getTracks().forEach(track => track.stop());
    }
    function createRecorder(stream, Recorder = root.MediaRecorder) {
        if (typeof Recorder !== 'function') throw new DOMException('録音非対応', 'NotSupportedError');
        const types = ['audio/webm;codecs=opus', 'audio/mp4', 'audio/webm', 'audio/ogg;codecs=opus'];
        for (const mimeType of types) {
            if (typeof Recorder.isTypeSupported === 'function' && Recorder.isTypeSupported(mimeType)) {
                try { return new Recorder(stream, { mimeType }); } catch (_) { /* 端末の既定形式へ */ }
            }
        }
        return new Recorder(stream);
    }
    async function permissionState() {
        try { return (await navigator.permissions?.query({ name: 'microphone' }))?.state || 'unknown'; }
        catch (_) { return 'unknown'; }
    }
    function acquireMicrophone(timeoutMs = 30000) {
        if (!root.isSecureContext) return Promise.reject(new DOMException('HTTPSが必要です', 'SecurityError'));
        if (!navigator.mediaDevices?.getUserMedia) return Promise.reject(new DOMException('マイク非対応', 'NotSupportedError'));
        return new Promise((resolve, reject) => {
            let expired = false;
            const timer = setTimeout(() => {
                expired = true;
                reject(new DOMException('許可待ち時間を超えました', 'TimeoutError'));
            }, timeoutMs);
            navigator.mediaDevices.getUserMedia({ audio: true, video: false }).then(stream => {
                clearTimeout(timer);
                if (expired) stopStream(stream); else resolve(stream);
            }, error => { clearTimeout(timer); reject(error); });
        });
    }
    function errorCopy(name) {
        if (['NotAllowedError', 'PermissionDeniedError', 'SecurityError'].includes(name))
            return ['マイクの許可が必要です', '下の「設定方法を見る」からマイクを許可してください。一度拒否すると、再試行だけでは許可画面が出ない場合があります。'];
        if (['NotReadableError', 'AbortError'].includes(name))
            return ['マイクを開けませんでした', '通話・録音など、マイクを使うほかのアプリを閉じてから、もう一度お試しください。'];
        if (['NotFoundError', 'DevicesNotFoundError'].includes(name))
            return ['マイクが見つかりません', 'マイクやBluetooth機器の接続を確認してから、もう一度お試しください。'];
        if (name === 'NotSupportedError')
            return ['この画面では録音できません', 'SafariまたはChromeで診断ページを開いてください。アプリ内の画面では動かない場合があります。'];
        if (name === 'TimeoutError')
            return ['マイクの確認に時間がかかっています', '許可画面が出ていれば選択を終えてから、もう一度お試しください。'];
        return ['録音を開始できませんでした', '少し待ってから再試行してください。続く場合は「設定方法を見る」を確認してください。'];
    }
    function validStats(stats) {
        return stats?.has_pitch === true && ['intonation', 'dynamics', 'brightness', 'resonance', 'tempo', 'sustain']
            .every(key => typeof stats.scores?.[key] === 'number' && Number.isFinite(stats.scores[key]) && stats.scores[key] >= 1 && stats.scores[key] <= 10);
    }
    root.RecordingSupport = { stopStream, createRecorder, permissionState, acquireMicrophone, errorCopy, validStats };
    if (typeof module !== 'undefined') module.exports = root.RecordingSupport;
})(globalThis);
