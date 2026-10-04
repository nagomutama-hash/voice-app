const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const support = require('../static/recording-support.js');
const html = fs.readFileSync('static/index.html', 'utf8');
const code = html.slice(html.indexOf('// ===== 状態管理 ====='), html.lastIndexOf('</script>'));
const scores = Object.fromEntries(['intonation','dynamics','brightness','resonance','tempo','sustain'].map(k => [k,9]));

function harness(overrides = {}, extras = {}) {
    const elements = {}, events = [], timers = new Map(),listeners={};
    const element = id => elements[id] ||= {textContent:'', innerHTML:'', disabled:false,
        classList:{add(){},remove(){},toggle(){}}, addEventListener(){},scrollIntoView(){},focus(){},pause(){},load(){},removeAttribute(key){delete this[key]}};
    const window = {isSecureContext:true,addEventListener(name,fn){listeners[name]=fn},gtag(...args){events.push(args)}};
    const context = vm.createContext({window, navigator:{userAgent:'test'}, document:{getElementById:element,querySelector:element},
        RecordingSupport:{...support, ...overrides}, MediaRecorder: function(){}, console, Date,
        setTimeout(fn){const id=timers.size+1;timers.set(id,fn);return id},clearTimeout(id){timers.delete(id)},setInterval(){return 1},clearInterval(){},
        Blob, URL, AbortController, FormData, ...extras});
    vm.runInContext(code, context);
    return {context,elements,events,listeners,run:source=>vm.runInContext(source,context)};
}

test('missing, nonfinite and incomplete scores are rejected', () => {
    assert.equal(support.validStats({has_pitch:false}),false);
    assert.equal(support.validStats({has_pitch:true,scores:{}}),false);
    assert.equal(support.validStats({has_pitch:true,scores:{...scores,tempo:NaN}}),false);
    assert.equal(support.validStats({has_pitch:true,scores}),true);
});
test('invalid response cannot display success, change history or emit completion', () => {
    const h=harness();
    h.run('previousScores = {tempo:7}');
    assert.throws(()=>h.run('renderResults({stats:{has_pitch:false}})'),/話し声/);
    assert.equal(h.run('previousScores.tempo'),7);
    assert.equal(h.events.length,0);
});
test('all nine scores are not pressure classification', () => {
    const h=harness();h.context.input=scores;
    assert.equal(h.run('detectVoiceType(input)'),'expressive');
});
test('MP4 is selected when WebM is unsupported', () => {
    class Recorder {static isTypeSupported(t){return t==='audio/mp4'} constructor(stream,options){this.options=options}}
    assert.equal(support.createRecorder({},Recorder).options.mimeType,'audio/mp4');
});
test('browser default is used when explicit formats fail', () => {
    class Recorder {static isTypeSupported(){return true} constructor(stream,options){if(options)throw Error('unsupported');this.default=true}}
    assert.equal(support.createRecorder({},Recorder).default,true);
});
test('double click acquires only one stream; stop blocks new recording while processing', async () => {
    let resolve, requests=0, stopped=0;
    const recorder={state:'recording',stream:{getTracks:()=>[{stop(){stopped++}}]},start(){},stop(){this.state='inactive'}};
    const h=harness({acquireMicrophone(){requests++;return new Promise(r=>resolve=r)},createRecorder(){return recorder}});
    const first=h.run('startRecording()');
    await h.run('startRecording()');
    assert.equal(requests,1);
    resolve(recorder.stream);await first;
    assert.equal(h.run('recordingState'),'recording');
    h.run('stopRecording()');await h.run('startRecording()');
    assert.equal(requests,1);assert.equal(stopped,1);
    assert.equal(h.run('recordingState'),'processing');
});
test('constructor failure releases microphone and restores retry', async () => {
    let stopped=0;
    const h=harness({acquireMicrophone:async()=>({getTracks:()=>[{stop(){stopped++}}]}),
        createRecorder(){throw new DOMException('unsupported','NotSupportedError')},permissionState:async()=> 'granted'});
    await h.run('startRecording()');
    assert.equal(stopped,1);assert.equal(h.run('recordingState'),'idle');
    assert.match(h.elements.micErrorCode.textContent,/NotSupportedError \/ granted \/ recorder/);
});

test('microphone disconnect discards partial audio and allows recording again without diagnosis',async()=>{
    const ended=[];let stopped=0,requests=0;
    const stream={getTracks:()=>[{stop(){stopped++},addEventListener(name,fn){if(name==='ended')ended.push(fn)}}]};
    const recorder={state:'inactive',stream,start(){this.state='recording'},stop(){this.state='inactive';this.onstop?.()}};
    const h=harness({acquireMicrophone:async()=>{requests++;return stream},createRecorder:()=>recorder});
    await h.run('startRecording()');
    recorder.ondataavailable({data:new Blob(['partial'])});
    h.run('onRecordingStop=()=>{throw Error("must not diagnose interrupted audio")}');
    ended[0]();
    assert.equal(stopped,1);assert.equal(h.run('recordedChunks.length'),0);
    assert.equal(h.run('recordingState'),'idle');assert.equal(h.elements.recordBtn.disabled,false);
    assert.match(h.elements.statusText.textContent,/接続が切れました/);
    assert.equal(h.events.some(e=>['recording_complete','five_diagnosis_complete'].includes(e[1])),false);
    await h.run('startRecording()');assert.equal(requests,2);assert.equal(h.run('recordingState'),'recording');
    ended[0]();assert.equal(h.run('recordingState'),'recording');
});

test('track end after a normal stop leaves the completed recording available for analysis',async()=>{
    let ended;
    const stream={getTracks:()=>[{stop(){},addEventListener(name,fn){ended=fn}}]};
    const recorder={state:'inactive',stream,start(){this.state='recording'},stop(){this.state='inactive'}};
    const h=harness({acquireMicrophone:async()=>stream,createRecorder:()=>recorder});
    await h.run('startRecording()');recorder.ondataavailable({data:new Blob(['complete'])});
    h.run('stopRecording()');ended();
    assert.equal(h.run('recordingState'),'processing');assert.equal(h.run('recordingAborted'),false);
    assert.equal(h.run('recordedChunks.length'),1);assert.equal(typeof recorder.onstop,'function');
});
test('late permission result from a departed page releases stream', async () => {
    let resolve,stopped=0;
    const h=harness({acquireMicrophone:()=>new Promise(r=>resolve=r)});
    const first=h.run('startRecording()');h.run('recordingGeneration += 1');
    resolve({getTracks:()=>[{stop(){stopped++}}]});await first;
    assert.equal(stopped,1);
});
test('microphone check never records, uploads, or sends analytics', () => {
    const source=fs.readFileSync('static/microphone-test.js','utf8');
    assert.doesNotMatch(source,/fetch\(|XMLHttpRequest|\.start\(|gtag\(|trackEvent\(/);
});
test('permission timeout releases a stream that arrives later', async () => {
    let resolve, stopped=0;
    const context=vm.createContext({isSecureContext:true, navigator:{mediaDevices:{getUserMedia:()=>new Promise(r=>resolve=r)}},
        DOMException, setTimeout, clearTimeout});
    vm.runInContext(fs.readFileSync('static/recording-support.js','utf8'),context);
    await assert.rejects(vm.runInContext('RecordingSupport.acquireMicrophone(5)',context),{name:'TimeoutError'});
    resolve({getTracks:()=>[{stop(){stopped++}}]});
    await new Promise(r=>setImmediate(r));
    assert.equal(stopped,1);
});
test('decode failure restores controls without updating history or sending completion', async () => {
    const h=harness();
    h.run('previousScores={tempo:7}; mediaRecorder={mimeType:"audio/mp4",stream:{getTracks:()=>[]}}; recordedChunks=[new Blob(["test"])]; convertToWav=async()=>{throw new Error("decode failed")};');
    await h.run('onRecordingStop()');
    assert.equal(h.run('recordingState'),'idle');
    assert.equal(h.run('previousScores.tempo'),7);
    assert.equal(h.events.some(e=>e[1]==='recording_complete'),false);
    assert.equal(h.events.some(e=>e[1]==='analysis_failed'),true);
});

function previewHarness(confirmed=true){
    const received=[];
    const ui={init:async()=>{},beginRecording(){},prepareReading:async()=>confirmed,render(data,blob){if(!data.diagnosis)throw Error('invalid');received.push({data,blob});return false;}};
    const h=harness({}, {location:{search:'?preview=five'},URLSearchParams,FiveMetricUI:ui});
    h.run('mediaRecorder={mimeType:"audio/mp4",stream:{getTracks:()=>[]}}; recordedChunks=[new Blob(["voice"])];convertToWav=async()=>new Blob(["wav"]);');
    return {...h,received};
}
test('only successful preview analysis passes the original audio to the result',async()=>{
    const h=previewHarness();h.run('uploadAndAnalyze=async()=>({diagnosis:{measurement_version:"m",calibration_version:"c",prompt_id:"p"}})');
    await h.run('onRecordingStop()');assert.equal(h.received.length,1);assert.equal(await h.received[0].blob.text(),'voice');
    assert.equal(h.run('recordingUrl'),null);assert.equal(h.run('recordedChunks.length'),0);
});
test('reading retry and analysis failure cannot replace successful comparison audio',async()=>{
    for(const confirmed of [false,true]){
        const h=previewHarness(confirmed);h.run('uploadAndAnalyze=async()=>{throw Error("failed")};');
        await h.run('onRecordingStop()');assert.equal(h.received.length,0);assert.equal(h.run('recordingState'),'idle');
    }
});
test('invalid preview response preserves old result and emits no success',async()=>{
    const h=previewHarness();h.run('uploadAndAnalyze=async()=>({})');await h.run('onRecordingStop()');
    assert.equal(h.received.length,0);assert.equal(h.events.some(e=>e[1]==='five_diagnosis_complete'),false);
});

test('a delayed old permission error cannot unlock a new recording after returning',async()=>{
    let finishPermission;
    const recorder={state:'recording',stream:{getTracks:()=>[]},start(){},stop(){this.state='inactive'}};
    const h=harness({acquireMicrophone:async()=>{throw new DOMException('denied','NotAllowedError')},permissionState:()=>new Promise(r=>finishPermission=r),createRecorder:()=>recorder});
    const first=h.run('startRecording()');await new Promise(r=>setImmediate(r));
    h.listeners.pagehide();h.listeners.pageshow();
    h.context.RecordingSupport.acquireMicrophone=async()=>recorder.stream;
    await h.run('startRecording()');assert.equal(h.run('recordingState'),'recording');
    const events=h.events.length;finishPermission('denied');await first;
    assert.equal(h.run('recordingState'),'recording');assert.equal(h.elements.recordBtn.disabled,true);assert.equal(h.events.length,events);
});
test('a delayed recorder error cannot unlock a newer recording',async()=>{
    let finishPermission;
    const recorder={state:'recording',stream:{getTracks:()=>[]},start(){this.state='recording'},stop(){this.state='inactive'}};
    const h=harness({acquireMicrophone:async()=>recorder.stream,permissionState:()=>new Promise(r=>finishPermission=r),createRecorder:()=>recorder});
    await h.run('startRecording()');const failed=recorder.onerror({error:{name:'AbortError'}});
    h.listeners.pagehide();h.listeners.pageshow();await h.run('startRecording()');
    finishPermission('granted');await failed;assert.equal(h.run('recordingState'),'recording');
});
test('explicit preview telemetry version is preserved and localhost stays silent',()=>{
    const h=harness();h.run('trackEvent("five_exit_click",{app_version:"v2-five-preview-1"})');
    assert.equal(h.events[0][2].app_version,'v2-five-preview-1');
    const local=harness({}, {localPreview:true});local.run('trackEvent("five_exit_click",{})');assert.equal(local.events.length,0);
});
test('microphone check ignores late response after leaving and starting again',async()=>{
    const elements={},handlers={},requests=[];let stopped=0;
    const get=id=>elements[id]||=( {textContent:'',disabled:false,addEventListener(name,fn){this[name]=fn}} );
    const context=vm.createContext({document:{getElementById:get},window:{addEventListener(name,fn){handlers[name]=fn}},
        RecordingSupport:{...support,acquireMicrophone:()=>new Promise(r=>requests.push(r))}});
    vm.runInContext(fs.readFileSync('static/microphone-test.js','utf8'),context);
    const first=get('testButton').click();handlers.pagehide();handlers.pageshow();const second=get('testButton').click();
    requests[0]({getTracks:()=>[{stop(){stopped++}}]});await first;
    assert.equal(stopped,1);assert.equal(get('testButton').disabled,true);assert.doesNotMatch(get('testStatus').textContent,/開くことができました/);
    requests[1]({getTracks:()=>[]});await second;assert.equal(get('testButton').disabled,false);assert.match(get('testStatus').textContent,/開くことができました/);
});
