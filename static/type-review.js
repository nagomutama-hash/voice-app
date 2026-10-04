(function(root){
    'use strict';
    const types={expressive:'表現者',soft:'ふわっと優しすぎ',pushy:'圧強めセールス',cerebral:'頭でっかち説明',unclear:'こもり声・不鮮明',uncertain:'判断保留'};
    const degrees={good:'良い（良さとして評価）',none:'問題はない（普通）',mild:'少し気になる',strong:'強く気になる',uncertain:'判断できない'};
    const numeric=v=>typeof v==='number'&&Number.isFinite(v)?v:null;
    const pick=(o,keys)=>o?Object.fromEntries(keys.map(k=>[k,numeric(o[k])])):null;
    function measurement(d){
        if(!root.FiveMetricCore.valid(d)||!d.type_auxiliary)return null;
        const a=d.type_auxiliary;
        return {diagnosis:root.FiveMetricCore.compact(d),total_score:root.FiveMetricCore.total(d),
            auxiliary:{version:typeof a.version==='string'?a.version:'unknown',status:['experimental','partial','unavailable'].includes(a.status)?a.status:'unavailable',
                classification_enabled:false,pitch:pick(a.pitch,['p10_hz','p90_hz','range_semitones','median_hz','voiced_seconds','periodicity_median']),
                intensity:pick(a.intensity,['active_level_range_db','active_seconds','pause_seconds']),
                spectrum:pick(a.spectrum,['high_to_low_energy_db_median']),h1_h2_status:'deferred'}};
    }
    function label(values){
        if(!Object.hasOwn(types,values.type)||!['strain','flatness','weak_core','listening_burden'].every(k=>Object.hasOwn(degrees,values[k])))throw Error('全項目の聞き分けを選んでください。迷う場合は「判断できない」を選べます。');
        if(!['natural','artificial','uncertain'].includes(values.expression))throw Error('声の表情の自然さを選んでください。');
        if(!/^(?:[1-9]|1\d|20)$/.test(values.participant))throw Error('話者番号を選んでください。');
        if(!['pc','iphone','android','other'].includes(values.device))throw Error('録音端末を選んでください。');
        return Object.fromEntries(['type','strain','flatness','weak_core','listening_burden','expression','participant','device'].map(k=>[k,values[k]]));
    }
    const enabled=()=>typeof location!=='undefined'&&['127.0.0.1','localhost'].includes(location.hostname)&&new URLSearchParams(location.search).get('preview')==='five'&&new URLSearchParams(location.search).get('review')==='types';
    let entries=[],current=null,sequence=0;
    const session=Date.now().toString(36);
    function notice(){
        if(!enabled())return;
        const panel=document.createElement('section');panel.className='card';panel.id='typeReviewNotice';
        panel.innerHTML='<h2>玉井先生の聞き分け確認</h2><p>普段の声で録音し、結果の下にある確認欄へ進んでください。声の表情を無理につける必要はありません。</p><p>聞き分けと測定値は、このページ内で一時保持します。「聞き分けを記録して保存」を押したときだけ、音声を含まないファイルをPCへ保存します。再読み込みすると、未保存の確認データは消えます。</p>';
        document.getElementById('fiveHistoryNotice').after(panel);
    }
    function select(name,title,options){
        return `<label for="review-${name}">${title}</label><select id="review-${name}" name="${name}" required><option value="">選んでください</option>${Object.entries(options).map(([value,text])=>`<option value="${value}">${text}</option>`).join('')}</select>`;
    }
    function capture(d){
        if(!enabled())return;
        const data=measurement(d);if(!data)return;
        current={sample_id:session+'-'+(++sequence),synthetic:location.pathname.startsWith('/test-'),...data};
        const old=document.getElementById('typeReviewPanel');if(old)old.remove();
        const panel=document.createElement('section');panel.id='typeReviewPanel';panel.className='card';
        panel.innerHTML='<h2>今回の声をどう聴きましたか？</h2><p>測定値を見る前に、実際に聴いた印象で選んでください。「良い」と「問題はない（普通）」は区別して選べます。点数から選ぶ必要はありません。</p><p id="typeReviewStatus" aria-live="polite" tabindex="-1"></p><form id="typeReviewForm" novalidate>'+
            select('participant','話者番号（同じ人は同じ番号）',Object.fromEntries(Array.from({length:20},(_,i)=>[i+1,'話者 '+(i+1)])))+
            select('device','録音した端末',{pc:'PC',iphone:'iPhone',android:'Android',other:'その他'})+
            select('type','今回のタイプ',types)+select('strain','力み・リラックス',{...degrees,good:'リラックスしていて良い'})+select('flatness','声の表情・抑揚',{...degrees,good:'声に自然な表情があり良い'})+
            select('weak_core','声の芯',{...degrees,good:'芯があり良い'})+select('listening_burden','聞き取りやすさ',{...degrees,good:'言葉を聞き取りやすく良い'})+
            select('expression','声の表情の自然さ',{natural:'自然',artificial:'無理につけている・不自然',uncertain:'判断できない'})+
            '<button type="submit" class="btn-rerecord">聞き分けを記録して保存（音声なし）</button></form><button type="button" id="typeReviewExport" class="btn-rerecord">記録済みデータをもう一度保存</button><details><summary>記録した測定値を見る</summary><pre id="typeReviewValues"></pre></details>';
        document.getElementById('fiveResults').append(panel);
        const form=panel.querySelector('form'),status=panel.querySelector('#typeReviewStatus');
        function showStatus(message,error=false){
            status.textContent=message;status.dataset.state=error?'error':'success';
            status.scrollIntoView({behavior:'smooth',block:'center'});status.focus({preventScroll:true});
        }
        function exportEntries(){
            if(!entries.length){showStatus('先に聞き分けを入力して「聞き分けを記録して保存」を押してください。',true);return;}
            const blob=new Blob([JSON.stringify({schema_version:'tamai-type-review-2',classification_enabled:false,samples:entries},null,2)],{type:'application/json'});
            const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='voice-type-review-'+session+'.json';
            document.body.append(a);
            try{a.click();}finally{a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
            showStatus(`聞き分け${entries.length}件を記録し、保存用ファイルを作成しました。ブラウザのダウンロードを確認してください。音声は含みません。`);
        }
        form.addEventListener('submit',event=>{
            event.preventDefault();
            try{
                const controls=[...form.querySelectorAll('select')];
                for(const control of controls)control.removeAttribute('aria-invalid');
                const missing=controls.filter(control=>!control.value);
                if(missing.length){
                    for(const control of missing)control.setAttribute('aria-invalid','true');
                    const names=missing.map(control=>form.querySelector(`label[for="${control.id}"]`).textContent);
                    showStatus('未入力：'+names.join('、')+'。迷う評価は「判断できない」を選べます。',true);
                    return;
                }
                const expert=label(Object.fromEntries(new FormData(form)));
                const sample={...current,expert};const index=entries.findIndex(e=>e.sample_id===sample.sample_id);
                if(index>=0)entries[index]=sample;else entries.push(sample);

                panel.querySelector('#typeReviewValues').textContent=JSON.stringify(sample.auxiliary,null,2);
                exportEntries();
            }catch(error){showStatus('保存できませんでした：'+error.message+'。入力内容は残っています。',true);}
        });
        panel.querySelector('#typeReviewExport').addEventListener('click',()=>{try{exportEntries();}catch(error){showStatus('保存できませんでした：'+error.message,true);}});
    }
    function clear(){entries=[];current=null;document.getElementById('typeReviewPanel')?.remove();}
    const api={measurement,label,notice,capture,clear};root.TypeReview=api;
    if(typeof module!=='undefined')module.exports=api;
})(globalThis);
