const button = document.getElementById('testButton');
const statusBox = document.getElementById('testStatus');
let generation = 0;
let activeStream = null;
window.addEventListener('pagehide', () => { generation++; RecordingSupport.stopStream(activeStream); activeStream=null; });
window.addEventListener('pageshow', () => { button.disabled=false; });
button.addEventListener('click', async () => {
    if (button.disabled) return;
    const attempt=++generation;
    let stream=null;
    button.disabled = true;
    statusBox.textContent = 'マイクの許可を確認しています...';
    document.getElementById('testCode').textContent = '';
    try {
        stream = await RecordingSupport.acquireMicrophone();
        if (attempt!==generation) return;
        activeStream=stream;
        RecordingSupport.stopStream(activeStream);
        statusBox.textContent = 'マイクを開くことができました。診断画面へ戻って録音をお試しください。音量や聞き取りやすさは、この確認では判定していません。';
    } catch (error) {
        if (attempt!==generation) return;
        const [title, copy] = RecordingSupport.errorCopy(error.name);
        statusBox.textContent = `${title}。${copy}`;
        const permission = await RecordingSupport.permissionState();
        if (attempt!==generation) return;
        document.getElementById('testCode').textContent = `確認コード：${error.name || 'UnknownError'} / ${permission} / mic-test`;
    } finally {
        RecordingSupport.stopStream(stream);
        if(attempt===generation){activeStream = null;button.disabled = false;}
    }
});
