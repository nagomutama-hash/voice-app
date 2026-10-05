(function(root) {
    'use strict';
    const appPath = path => typeof root.voiceAppPath === 'function' ? root.voiceAppPath(path) : path;
    const KEYS = ['brightness','articulation','power','speed','resonance'];
    const STORAGE_KEY = 'voice-app-v2:five-metric-history:1';
    const DIRECTIONS = ['low','high','within_reference','slow','fast','optimal','unknown'];
    function valid(d) {
        return d?.schema_version === 'five-metric-diagnosis-1' &&
            typeof d.measurement_version === 'string' && typeof d.calibration_version === 'string' &&
            [10,20].includes(d.score_scale ?? 10) &&
            (d.prompt_id === null || typeof d.prompt_id === 'string') &&
            KEYS.every(key => {
                const m = d.metrics?.[key];
                return m && ['provisional','unavailable'].includes(m.status) && DIRECTIONS.includes(m.direction) &&
                    (m.raw_value === null || (typeof m.raw_value === 'number' && Number.isFinite(m.raw_value))) &&
                    (m.status === 'unavailable' ? m.reference_score === null :
                        typeof m.raw_value === 'number' && Number.isFinite(m.raw_value) && typeof m.reference_score === 'number' && Number.isFinite(m.reference_score) && m.reference_score >= 1 && m.reference_score <= (d.score_scale ?? 10));
            });
    }
    function compact(d) {
        if (!valid(d)) throw new Error('診断結果を確認できませんでした。もう一度録音してください。');
        const auxiliary=d.type_evidence||d.type_auxiliary;
        const pitch=auxiliary?.pitch?.range_semitones,spectrum=auxiliary?.spectrum?.high_to_low_energy_db_median;
        const evidence=auxiliary?.status==='experimental'&&Number.isFinite(pitch)&&pitch>=0&&pitch<=120&&Number.isFinite(spectrum)&&Math.abs(spectrum)<=200?
            {type_evidence:{status:'experimental',pitch:{range_semitones:pitch},spectrum:{high_to_low_energy_db_median:spectrum}}}:{};
        return {schema_version:d.schema_version,measurement_version:d.measurement_version,
            ...evidence,
            calibration_version:d.calibration_version,prompt_id:d.prompt_id,score_scale:d.score_scale ?? 10,
            metrics:Object.fromEntries(KEYS.map(key => {
                const m=d.metrics[key];return [key,{raw_value:m.raw_value,reference_score:m.reference_score,direction:m.direction,status:m.status,
                    reason:['clipping','reading_not_confirmed','asr_unavailable','asr_busy','recognition_failed','speech_not_recognized','insufficient_speech','reading_uncertain','speed_out_of_reference','fixed_reading_mismatch'].includes(m.reason)?m.reason:null}];
            }))};
    }
    function compare(current, previous) {
        if (!valid(current) || !valid(previous) || ['schema_version','measurement_version','calibration_version','prompt_id'].some(k=>current[k]!==previous[k])) return null;
        if ((current.score_scale ?? 10)!==(previous.score_scale ?? 10)) return null;
        return Object.fromEntries(KEYS.map(key => {
            const a=previous.metrics[key].reference_score,b=current.metrics[key].reference_score;
            return [key,a===null||b===null?null:Math.round((b-a)*10)/10];
        }));
    }
    function createHistory(storage, now = () => Date.now()) {
        let fallback = [], persisted = Boolean(storage);
        function trial(t,d){
            if(!t || !/^[a-z_]+-\d{2}$/.test(t.hint_id) || !Number.isInteger(t.hint_version) || !KEYS.includes(t.target_metric) || !compare(d,t.before) || !/^[a-z]+$/.test(t.mechanism_type) || ![null,'fast','slow'].includes(t.speed_direction))return null;
            return {hint_id:t.hint_id,hint_version:t.hint_version,target_metric:t.target_metric,speed_direction:t.speed_direction,mechanism_type:t.mechanism_type,before:compact(t.before)};
        }
        const clean = rows => Array.isArray(rows) ? rows.filter(row => row && Number.isFinite(row.at) && row.at<=now()+60000 && now()-row.at<30*86400000 && valid(row.diagnosis)).slice(-20).map(row=>({at:row.at,diagnosis:compact(row.diagnosis),...(trial(row.trial,row.diagnosis)?{trial:trial(row.trial,row.diagnosis)}:{})})) : [];
        function read() {
            if (!persisted) return clean(fallback);
            let stored;
            try { stored=storage.getItem(STORAGE_KEY); } catch (_) { persisted=false; return clean(fallback); }
            try { fallback=clean(JSON.parse(stored||'[]')); } catch (_) { fallback=[]; }
            if (stored && stored !== JSON.stringify(fallback)) {
                try { storage.setItem(STORAGE_KEY,JSON.stringify(fallback)); } catch (_) { persisted=false; }
            }
            return fallback;
        }
        function write(rows) {
            fallback=clean(rows);
            if (persisted) {try {storage.setItem(STORAGE_KEY,JSON.stringify(fallback));} catch (_) {persisted=false;}}
        }
        return {read, append(d,t){const rows=read();rows.push({at:now(),diagnosis:compact(d),trial:t});write(rows);},
            clear(){
                fallback=[];
                if(!storage)return true;
                try{storage.removeItem(STORAGE_KEY);return true;}
                catch(_){
                    try{storage.setItem(STORAGE_KEY,'[]');return true;}
                    catch(_){persisted=false;return false;}
                }
            },
            get persisted(){return persisted}};
    }
    function total(d){return valid(d)&&KEYS.every(k=>d.metrics[k].reference_score!==null)?Math.round(KEYS.reduce((n,k)=>n+d.metrics[k].reference_score,0)*10)/10:null;}
    function baseline(rows,d,at){const date=new Date(at).toDateString();return rows.find(r=>new Date(r.at).toDateString()===date&&compare(d,r.diagnosis))?.diagnosis ?? null;}
    const api = {valid,compact,compare,createHistory,total,baseline};
    root.FiveMetricCore=api;
    if (typeof module !== 'undefined') module.exports=api;

    let config, history, panel, catalog=[], pending=null, armed=null, activeTrial=null, experience={}, audioStore;
    let fixedMode=false, completedReading=false;
    function pauseAudio(){
        for(const audio of document.querySelectorAll('#fiveResults audio, #audioPlayer'))audio.pause();
    }
    function audioHTML(){
        const clips=audioStore?.snapshot()||{};
        return `<details class="card five-optional" id="fiveAudioCompare"><summary>録音した声を聴き比べる（任意）</summary><div class="five-optional-content">${clips.current?`${clips.previous?'<label for="fivePreviousAudio">前の声（比較元の診断）</label><audio id="fivePreviousAudio" controls preload="metadata"></audio>':'<p>比較元の音声はこのページにありません。もう一度録音すると、2つの声を聴き比べられます。</p>'}<label for="fiveCurrentAudio">今回の声（この診断の録音）</label><audio id="fiveCurrentAudio" controls preload="metadata"></audio><p>声の自然さや聞き取りやすさを、同じ再生音量で確かめてください。</p>`:'<p>音声は履歴に保存していません。このページで新たに録音すると、声を聴いて確認できます。</p>'}<p class="five-draft-note">聴き比べ用の音声は、このページを開いている間だけ保持します。再読み込みやページ移動で消えます。</p></div></details>`;
    }
    function connectAudio(){
        const clips=audioStore?.snapshot()||{};
        for(const [id,url] of [['fivePreviousAudio',clips.previous],['fiveCurrentAudio',clips.current]]){
            const audio=document.getElementById(id);if(!audio||!url)continue;
            audio.src=url;
            audio.addEventListener('play',()=>{for(const other of document.querySelectorAll('#fiveResults audio, #audioPlayer'))if(other!==audio)other.pause();});
        }
    }
    let shown=[];
    let supplementSuccesses=0,lastNaturalShown=-Infinity,inlineVariant=-1;
    const groupLabels={brightness:'明るさ',articulation:'滑舌',power:'声の安定感',speed_fast:'速度を少し下げる',speed_slow:'速度を少し上げる',resonance:'響き'};
    function hintEvent(name,t){if(typeof root.trackEvent==='function')root.trackEvent(name,{hint_id:t.hint_id,hint_version:t.hint_version,target_metric:t.target_metric,speed_direction:t.speed_direction||'none',app_version:'v2-five-preview-1'});}
    function hintPanel(d){
        const rows=history.read(), past=rows.filter(r=>r.trial);
        const section=document.createElement('section');section.className='card';section.id='fiveHints';
        const switchMetric=root.HintEngine.switchMetric(rows,catalog,d);
        const groups=Object.keys(groupLabels).filter(group=>catalog.some(h=>h.group===group&&root.HintEngine.eligible(d,h)));
        section.innerHTML='<h2>次に試す改善ヒント</h2><p class="five-draft-note">無理や違和感があれば中止してください。</p>'+
            (switchMetric?'<p>次は別の項目で、声の変化を確かめてみましょう。</p>':'')+
            '<div id="fiveHintChoices"></div><div id="fiveHintText" aria-live="polite"></div>';
        panel.append(section);
        const choices=section.querySelector('#fiveHintChoices');
        const available=switchMetric?groups.filter(g=>!catalog.some(h=>h.group===g&&h.target_metric===switchMetric)):groups;
        if(!groups.length)section.querySelector('#fiveHintText').textContent='今回は方向に合うヒントを選べません。録音条件をそろえて、もう一度お試しください。';
        const offered=available.length?available:groups;
        const clips=audioStore?.snapshot()||{};
        const supplement=root.SupplementComments?.build(d,catalog,rows,{excludeMetric:switchMetric,audioAvailable:Boolean(clips.current&&clips.previous)});
        const recommended=supplement?.targets.length? (supplement.targets.length===1?offered.find(g=>catalog.some(h=>h.group===g&&h.target_metric===supplement.targets[0]&&root.HintEngine.eligible(d,h)))||null:null):root.HintEngine.recommendGroup(catalog,d,offered);
        [recommended,...offered.filter(group=>group!==recommended)].filter(Boolean).forEach(group=>{
            const button=document.createElement('button');button.type='button';
            button.className='btn-rerecord'+(group===recommended?' five-recommended':'');
            button.textContent=group===recommended?'おすすめ：'+groupLabels[group]:groupLabels[group];
            button.addEventListener('click',()=>show(group));choices.append(button);
        });
        if(recommended)show(recommended);
        function show(group){
            const hint=root.HintEngine.next(catalog,d,rows,group,shown),box=section.querySelector('#fiveHintText');
            if(!hint){box.textContent='この項目の候補を表示できません。別の項目を選んでください。';return;}
            shown.push(hint.hint_id);
            hintEvent('five_hint_view',hint);
            box.innerHTML=`<h3>${groupLabels[group]}</h3><p>選んだ項目で、この方法による変化を確かめます。</p><p><strong>${escape(hint.hint_text)}</strong></p><button type="button" class="btn-rerecord" id="fiveTryHint">このヒントを試す</button> <button type="button" id="fiveOtherHint">別の方法を見る</button>`;
            pending={hint_id:hint.hint_id,hint_version:hint.hint_version,target_metric:hint.target_metric,speed_direction:hint.speed_direction,mechanism_type:hint.mechanism_type,before:compact(d)};
            box.querySelector('#fiveOtherHint').onclick=()=>show(group);
            box.querySelector('#fiveTryHint').onclick=()=>{
                const record=document.getElementById('recordBtn');if(record.disabled)return;
                armed=pending;hintEvent('five_hint_rerecord_click',armed);
                const reminder=document.getElementById('fiveReadyHint');
                const previous=rows.at(-2)?.diagnosis;
                const comments=KEYS.map(k=>root.RecordingChange?.select(d,previous,baseline(rows,d,rows.at(-1)?.at??Date.now()),k,false)?.text||'').join(' ');
                const change=root.RecordingChange?.select(d,previous,baseline(rows,d,rows.at(-1)?.at??Date.now()),hint.target_metric,false);
                const natural=supplement?.text||supplement?.trial?'':root.SupplementComments?.naturalText({state:change?.state,existingText:comments,hintText:hint.hint_text,successCount:supplementSuccesses,lastShown:lastNaturalShown,variant:supplementSuccesses,complete:KEYS.every(k=>d.metrics[k].status==='provisional')});
                if(natural)lastNaturalShown=supplementSuccesses;
                reminder.textContent='今回のヒント：'+hint.hint_text+(natural?' '+natural:'');reminder.classList.remove('hidden');
                document.getElementById('statusText').textContent='準備ができたら「録音を始める」を押してください。';
                record.scrollIntoView({behavior:'smooth',block:'start'});record.focus({preventScroll:true});
            };
        }
        if(past.length){
            const list=document.createElement('section');list.className='card';list.innerHTML='<details class="five-optional"><summary>この端末で試した方法を見る</summary><div class="five-optional-content"></div></details>';
            past.slice(-5).reverse().forEach(row=>{
                const h=catalog.find(h=>h.hint_id===row.trial.hint_id&&h.hint_version===row.trial.hint_version);
                const change=root.HintEngine.delta(row),p=document.createElement('p');
                p.textContent=`${new Date(row.at).toLocaleString('ja-JP')}：${labels[row.trial.target_metric]}／${h?.hint_text||'以前の版のヒント'} 対象の参考値 ${change===null?'比較保留':`${change>0?'+':''}${change.toFixed(1)}点`}${change>=0.5?'（参考値が上がった方法として記録）':''}`;list.querySelector('.five-optional-content').append(p);
            });list.hidden=true;panel.append(list);
        }
    }
    const labels = {brightness:'明るさ',articulation:'滑舌',power:'声の安定感',speed:'話す速度',resonance:'響き'};
    const details = {
        brightness:'声に含まれる高音・低音の成分のバランスを見ています。高い点数でも、さらに明るくする必要はありません。',
        articulation:'音の切り替わりなどから推定した参考値です。言葉の聞き取りやすさは、実際の声を聴いて確認する必要があります。',
        power:'話している間の音量のムラの少なさを見ています。音量の変化が少ないほど高い参考点になります。無音や間の除外は推定です。端末と口の距離をそろえて比べてください。',
        speed:'録音した言葉を文字に起こし、読みの拍数を話している時間で割った参考値です。前後の無音は除き、途中の間は含めます。聞き間違いや読みの違いで数値が変わることがあります。',
        resonance:'声に含まれる調和した音の成分を見ています。声の豊かさすべてを表す数値ではありません。',
    };
    const escape = value => String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    function stateLabel(key,m) {
        if(m.status==='unavailable') return '今回は判定を保留';
        if(key==='speed')return {slow:'遅めの目安',optimal:'基準範囲の目安',fast:'速めの目安'}[m.direction];
        return {low:'参考範囲より低め',high:'参考範囲より高め',within_reference:'参考範囲内'}[m.direction];
    }
    function display(d, previous, restored) {
        if(inlineVariant<0)inlineVariant=Math.max(0,history.read().length-1);
        else if(!restored)inlineVariant++;
        const latest=history.read().at(-1),compared=compare(d,previous);
        const delta=compared && Object.values(compared).some(value=>typeof value==='number')?compared:null;
        const first=baseline(history.read(),d,latest?.at ?? Date.now());
        const dayDelta=compare(d,first);
        const clips=audioStore?.snapshot()||{};
        const context={baseline:first,audioAvailable:Boolean(clips.current&&clips.previous),excludeMetric:root.HintEngine.switchMetric(history.read(),catalog,d),commentVariant:inlineVariant};
        const eligibleGroups=Object.keys(groupLabels).filter(group=>catalog.some(h=>h.group===group&&root.HintEngine.eligible(d,h)));
        const availableGroups=context.excludeMetric?eligibleGroups.filter(g=>!catalog.some(h=>h.group===g&&h.target_metric===context.excludeMetric)):eligibleGroups;
        const recommendedGroup=root.HintEngine.recommendGroup(catalog,d,availableGroups.length?availableGroups:eligibleGroups);
        context.recommendedMetric=catalog.find(h=>h.group===recommendedGroup&&root.HintEngine.eligible(d,h))?.target_metric??null;
        const supplement=root.SupplementComments?.build(d,catalog,history.read(),context);
        context.inlineSupplements=true;context.supplement=supplement;
        if(supplement?.targets.length)context.recommendedMetric=supplement.targets.length===1?supplement.targets[0]:null;
        const view=root.ResultExperience.describe(d,previous,latest?.trial,context);
        panel.classList.remove('hidden');
        pauseAudio();
        panel.innerHTML = root.ResultExperience.summaryHTML(d,previous,latest?.trial,restored,context);
        if(d.metrics.speed.reference_score===null){
            const retry=document.createElement('section');retry.className='card';
            retry.innerHTML='<h2>速度を適切に判定できませんでした</h2><p>ほかの4項目は下で確認できます。総合点は保留しています。</p><p>指定の文章で測り直してください。'+(fixedMode?'再測定でも判定できない場合は、速度を保留します。':'ページを移動すると、聴き比べ用の音声は消えます。')+'</p><a class="btn-rerecord" href="'+appPath('/speed-retest?preview=five')+'">指定の文章で測り直す</a>';
            panel.prepend(retry);
        }
        if(delta){
            const beforeTotal=root.FiveMetricCore.total(previous),afterTotal=root.FiveMetricCore.total(d);
            const totalDelta=beforeTotal!==null&&afterTotal!==null?Math.round((afterTotal-beforeTotal)*10)/10:null;
            const totalHTML=totalDelta===null?'':`<div class="five-comparison-row"><h3>総合</h3><p class="five-score-transition"><span>${beforeTotal.toFixed(1)} → ${afterTotal.toFixed(1)}</span><strong class="five-delta-badge ${root.ResultExperience.deltaClass(totalDelta)}">${totalDelta>0?'+':''}${totalDelta.toFixed(1)}点</strong></p></div>`;
            const changes=KEYS.map(key=>{
                const numbers=root.ResultExperience.metricChangeHTML(d,previous,key);
                return `<div class="five-comparison-row"><h3>${labels[key]}</h3>${numbers||'<p>比較保留</p>'}</div>`;
            }).join('');
            panel.querySelector('.five-overview').insertAdjacentHTML('beforebegin',`<section class="card five-comparison-grid" aria-label="前回と今回の点数比較">${totalHTML}${changes}</section>`);
        }
        const metricHTML = KEYS.map(key=>{
                const m=d.metrics[key],change=delta?.[key];
                const selected=root.RecordingChange?.select(d,previous,first,key,context.audioAvailable);
                const diff=root.ResultExperience.metricChangeHTML(d,previous,key);
                const description=m.reason==='clipping'?'録音した音が大きすぎて、波形がつぶれています。端末を少し離し、普段の声で録り直してください。':m.reason==='reading_not_confirmed'?'固定文を最後まで読めたことを確認してから、もう一度お試しください。':details[key];
                return `<section class="card"><h3>${labels[key]} <span class="five-score">${m.reference_score===null?'—':m.reference_score.toFixed(1)}<small> / ${d.score_scale ?? 10}</small></span></h3><p><strong>${stateLabel(key,m)}</strong></p><p>${description}</p>${diff}${typeof dayDelta?.[key]==='number'?`<p>今日最初との差：${dayDelta[key]>0?'+':''}${dayDelta[key].toFixed(1)}点</p>`:''}${selected?`<p class="five-change-comment">${escape(selected.text)}</p>`:''}</section>`;
            }).join('');
        if(latest?.trial){
            const h=catalog.find(h=>h.hint_id===latest.trial.hint_id&&h.hint_version===latest.trial.hint_version);
            const change=root.HintEngine.delta(latest),summary=document.createElement('section');summary.className='card';
            const selected=root.RecordingChange?.select(d,previous,first,latest.trial.target_metric,context.audioAvailable);
            const message=selected?selected.text:change===null?'対象項目を比較できなかったため、この方法の変化は判定を保留します。':change>=0.5?'対象の参考値が上がりました。録音した声を聴いて、自然さや聞き取りやすさも確かめてください。':change<=-0.5?'対象の参考値が下がりました。無理に続けず、別の方法や項目を試してみましょう。':'今回は対象の数値に大きな変化はありませんでした。別の方法との相性も確かめてみましょう。';
            summary.innerHTML=`<h2>試した方法</h2><p>${escape(labels[latest.trial.target_metric])}</p>${root.ResultExperience.metricChangeHTML(d,previous,latest.trial.target_metric)}`;
            panel.append(summary);
        }
        hintPanel(d);
        panel.insertAdjacentHTML('beforeend',root.ResultExperience.zoomHTML(view,experience.exit));
        panel.insertAdjacentHTML('beforeend', `<details class="card five-optional"><summary>項目ごとの数値・説明を見る</summary><div class="five-optional-content">${metricHTML}</div></details>` + audioHTML());
        if(['127.0.0.1','localhost'].includes(location.hostname)&&root.TypeCandidate?.explain){
            const report=root.TypeCandidate.explain(d);
            const details=document.createElement('details');details.className='card five-optional';
            details.innerHTML='<summary>先生用：タイプ判定の記録</summary><p>判定基準・各項目の点数・低さの数を保存できます。音声は含みません。</p><pre style="white-space:pre-wrap;overflow-wrap:anywhere"></pre><button type="button" class="btn-rerecord">判定記録を保存</button>';
            details.querySelector('pre').textContent=JSON.stringify(report,null,2);
            details.querySelector('button').addEventListener('click',()=>{
                const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));
                const a=document.createElement('a');a.href=url;a.download='voice-type-decision-'+Date.now()+'.json';
                document.body.append(a);try{a.click();}finally{a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
            });panel.append(details);
        }
        const audioDetails=document.getElementById('fiveAudioCompare');
        audioDetails.addEventListener('toggle',()=>{if(!audioDetails.open)pauseAudio();});
        connectAudio();
        const exit=panel.querySelector('.five-exit');
        if(exit)exit.addEventListener('click',()=>{
            if(typeof root.trackEvent==='function')root.trackEvent('five_exit_click',{destination_type:experience.exit.type,result_state:view.state,experience_version:experience.version||'unknown',app_version:'v2-five-preview-1'});
        });
    }
    root.FiveMetricUI = {
        async init() {
            document.body.classList.add('studio-preview');
            const response = await fetch(appPath('/api/diagnosis-config'));
            if(!response.ok)throw new Error('診断の準備ができませんでした。ページを再読み込みしてください。');
            config=await response.json();
            fixedMode=true;
            if(fixedMode)config.reading={...config.reading,id:config.speed_retest.id,text:config.speed_retest.text};
            root.RecordingChange.configure(config);
            const hints=await fetch(appPath('/static/hints-draft.json'));
            if(!hints.ok)throw new Error('ヒントの準備ができませんでした。ページを再読み込みしてください。');
            catalog=await hints.json();
            // 案内先の設定取得に失敗しても、録音・診断は続けられる。
            try{const r=await fetch(appPath('/static/experience-config.json'));experience=r.ok?await r.json():{};}catch(_){experience={};}
            if(!experience||typeof experience!=='object')experience={};
            let storage;try{storage=root.localStorage}catch(_){}
            history=createHistory(storage);
            audioStore=root.AudioComparison.createStore(root.URL,compare);
            document.getElementById('audioPlayer').addEventListener('play',()=>{for(const audio of document.querySelectorAll('#fiveResults audio'))audio.pause();});
            const rows=history.read(); // 不正・期限切れ・旧形式の保存値を表示しない。
            document.querySelector('.record-guide').innerHTML=`<strong>好きな内容を5〜10秒ほど話してください（例）</strong><span class="record-example">${escape(config.reading.text)}</span><p>口と端末の距離を毎回そろえて、普段の声で話してください。話し終えたら録音を止めてください。</p>`;
            document.querySelector('.record-guide').append(document.getElementById('timer'));
            if(fixedMode){
                document.querySelector('.record-guide strong').textContent='次の文章を最後まで、普段の声で読んでください。';
                const note=document.createElement('p');note.textContent='文章を変えたり省略したりせず、最後まで読んでください。';
                document.querySelector('.record-guide').append(note);
            }
            document.getElementById('statusText').textContent='準備ができたら「録音を始める」を押してください。';
            document.querySelector('.hero-lead').hidden=true;
            document.querySelector('.hero-tagline').hidden=true;
            document.querySelector('.profile-line').hidden=true;
            document.querySelector('.authority-strip').innerHTML='<span class="five-authority-badge">MC・ナレーター・DJ <strong>35年以上</strong></span><span class="five-authority-badge">ボイストレーナー歴 <strong>20年以上</strong></span><span class="five-authority-badge">受講者累計 <strong>2万人以上</strong></span>';
            document.getElementById('recordBtn').textContent='🎤 録音を始める';
            document.getElementById('statusText').textContent='準備ができたら「録音を始める」を押してください。';
            panel=document.getElementById('fiveResults');
            const privacy=document.getElementById('fiveHistoryNotice');
            privacy.classList.remove('hidden');
            privacy.innerHTML='<details><summary>保存・テストについて</summary><p>結果と試した方法をこの端末・ブラウザ内に最大20回・30日間保存します。音声は履歴に保存しません。点数・基準・ヒントは検証中で、声の良し悪しや性格、練習の効果を断定するものではありません。</p></details><p id="fiveStorageStatus" hidden></p><button type="button" id="fiveDeleteHistory">この端末の診断履歴を消す</button>';
            if(!history.persisted)document.getElementById('fiveStorageStatus').hidden=false;
            if(!history.persisted)document.getElementById('fiveStorageStatus').textContent='このブラウザでは履歴を保存できません。今回はページを開いている間だけ比較します。';
            document.getElementById('fiveDeleteHistory').addEventListener('click',()=>{
                this.clearAudio();root.TypeReview?.clear();const cleared=history.clear();pending=armed=activeTrial=null;shown=[];panel.innerHTML='';panel.classList.add('hidden');
                document.getElementById('fiveStorageStatus').hidden=false;
                document.getElementById('fiveStorageStatus').textContent=cleared?'この端末の診断履歴を消しました。':
                    '画面内の履歴と音声は消しましたが、ブラウザに保存した履歴を削除できませんでした。ブラウザの設定から、このサイトのデータを削除してください。';
            });
            const last=rows.at(-1)?.diagnosis;
            if(last && last.measurement_version===config.measurement_version && last.calibration_version===config.calibration_version && last.prompt_id===config.reading.id) display(last,rows.at(-2)?.diagnosis,true);
            root.TypeReview?.notice();
            root.dispatchEvent(new Event('five-preview-ready'));
        },
        beginRecording(){completedReading=false;pauseAudio();if(!armed)document.getElementById('fiveReadyHint').classList.add('hidden');activeTrial=armed;armed=null;},
        setPlaybackBusy(busy){for(const audio of document.querySelectorAll('#fiveResults audio')){audio.controls=!busy;if(busy)audio.pause();}},
        clearAudio(){pauseAudio();audioStore?.clear();const section=document.getElementById('fiveAudioCompare');if(section)section.innerHTML='<summary>録音した声を聴き比べる（任意）</summary><p>音声の一時保持を終了しました。新たに録音すると、声を聴いて確認できます。</p>';},
        // 指定文の案内を前提に停止後すぐ診断する。完読を自動検証する機能ではない。
        async prepareReading(){completedReading=fixedMode;return true;},
        addRequestFields(form){form.append('measurement_mode','five_preview');form.append('prompt_id',config.reading.id);form.append('reading_complete',completedReading?'true':'false')},
        render(data,blob=null){
            const d=compact(data.diagnosis),priorRow=history.read().at(-1),last=priorRow?.diagnosis;
            // 補助測定は今回の表示だけに使い、音声・高さ推移を履歴へ保存しない。
            d.type_auxiliary=data.diagnosis.type_auxiliary;
            history.append(d,activeTrial);
            supplementSuccesses++;
            pauseAudio();audioStore?.commit(blob,history.read().at(-1),priorRow);
            if(activeTrial)hintEvent('five_hint_rerecord_complete',activeTrial);
            activeTrial=null;pending=null;
            document.getElementById('fiveReadyHint').classList.add('hidden');
            display(d,last,false);
            root.TypeReview?.capture(data.diagnosis);
            if(!history.persisted)document.getElementById('fiveStorageStatus').hidden=false;
            if(!history.persisted)document.getElementById('fiveStorageStatus').textContent='このブラウザでは履歴を保存できません。今回はページを開いている間だけ比較します。';
            panel.scrollIntoView({behavior:'smooth',block:'start'});
            return Boolean(compare(d,last));
        },
    };
})(globalThis);
