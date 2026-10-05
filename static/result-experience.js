(function(root){
    'use strict';
    const keys=['brightness','articulation','power','speed','resonance'];
    const labels={brightness:'明るさ',articulation:'滑舌',power:'声の安定感',speed:'話す速度',resonance:'響き'};
    const esc=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const inRange=m=>['within_reference','optimal'].includes(m.direction);
    function coaching(d,exclude=null){
        const available=keys.filter(k=>d.metrics[k]?.status==='provisional');
        const strong=available.filter(k=>inRange(d.metrics[k])).sort((a,b)=>d.metrics[b].reference_score-d.metrics[a].reference_score)[0];
        const eligible=available.filter(k=>k==='speed'?['slow','fast'].includes(d.metrics[k].direction):!['high','unknown'].includes(d.metrics[k].direction));
        const remaining=eligible.filter(k=>k!==exclude);
        const target=(remaining.length?remaining:eligible).sort((a,b)=>d.metrics[a].reference_score-d.metrics[b].reference_score)[0];
        return {strong:strong||null,target:target||null};
    }
    function coachingHTML(d,exclude){
        const c=coaching(d,exclude);
        if(!c.strong&&!c.target)return '';
        const good=c.strong?`${labels[c.strong]}は参考範囲内です。今回の声の良いところとして、聴いて確かめてみましょう。`:'測定できた項目があります。録音した声とあわせて、今回の良いところを確かめましょう。';
        const challenge=c.target?`次は${labels[c.target]}を試してみましょう。ヒントを試せる項目の中で、今回の点数が最も低い項目です。`:'今回は方向に合うヒントを選べません。録音条件と、実際の声を確認しましょう。';
        return `<div class="five-coaching"><h3>今回の良いところ</h3><p>${esc(good)}</p><h3>次に試すポイント</h3><p>${esc(challenge)}</p></div>`;
    }
    function describe(d,previous,trial,context={}){
        const core=root.FiveMetricCore;
        if(!core.valid(d))throw new Error('Invalid diagnosis');
        const available=keys.filter(k=>d.metrics[k].status==='provisional');
        const within=available.filter(k=>inRange(d.metrics[k]));
        const delta=core.compare(d,previous);
        const changes=delta?keys.filter(k=>typeof delta[k]==='number'&&Math.abs(delta[k])>=0.5):[];
        const trialDelta=trial && keys.includes(trial.target_metric)?core.compare(d,trial.before)?.[trial.target_metric]:null;
        let state='first';
        if(available.length===0)state='unavailable';
        else if(available.length<5)state='partial';
        else if(typeof trialDelta==='number')state=trialDelta>=0.5?'increase':trialDelta<=-0.5?'decrease':'trial_small';
        else if(within.length===5)state='balanced';
        else if(delta && keys.some(k=>typeof delta[k]==='number'))state=changes.length?'changed':'small';
        const chosen=trial?.target_metric?root.RecordingChange?.select(d,previous,context.baseline,trial.target_metric,context.audioAvailable):null;
        if(chosen)state=({clear_rise:'increase',rise:'increase',decline:'decrease',maintained:'trial_small',high_maintained:'high_maintained'})[chosen.state];
        const headings={high_maintained:'今回も高い点数を保てています！声の良さが引き続き表れています。',first:'まずは、今回の声の傾向を確かめましょう',balanced:'5項目とも参考範囲に入っています',increase:'試した項目の参考値が上がりました',decrease:'試した項目の参考値が下がりました',trial_small:'試した項目には、大きな数値の変化はありませんでした',small:'今回は数値の大きな変化はありませんでした',changed:'前回から参考値に変化がありました',partial:'一部の項目は判定を保留しています',unavailable:'今回は点数の判定を保留しています'};
        let overview=available.length===0?'録音条件を確認して、普段の声で録り直してください。':within.length===5?'今の声を無理に変えず、録音した声の自然さや聞き取りやすさも確かめてみましょう。':within.length?`今回、${within.map(k=>labels[k]).join('・')}は参考範囲内でした。ほかの項目は、下の説明と録音した声をあわせて確認してください。`:'録音条件をそろえながら、各項目の説明と録音した声をあわせて確認してください。';
        if(state==='partial')overview='測定できた項目だけを表示しています。判定できなかった項目を平均点として補ってはいません。';
        const largest=changes.sort((a,b)=>Math.abs(delta[b])-Math.abs(delta[a]))[0];
        const changeText=largest?`前回との差が最も大きい項目：${labels[largest]} ${delta[largest]>0?'+':''}${delta[largest].toFixed(1)}点` : '';
        return {state,heading:headings[state],overview,changeText,within,available,delta};
    }
    function metricChangeHTML(d,previous,key){
        const delta=root.FiveMetricCore.compare(d,previous)?.[key];
        if(typeof delta!=='number')return '';
        const before=previous.metrics[key].reference_score,after=d.metrics[key].reference_score;
        return `<p class="five-score-transition"><span>${before.toFixed(1)} <span aria-label="から">→</span> ${after.toFixed(1)}</span><strong class="five-delta-badge ${deltaClass(delta)}">${delta>0?'+':''}${delta.toFixed(1)}点</strong></p>`;
    }
    function deltaClass(delta){return delta>=2?'five-delta-rise':delta<=-2?'five-delta-fall':'';}
    function radar(d,previous,exclude=null,recommendedMetric=undefined){
        if(!root.FiveMetricCore.valid(d)||keys.some(k=>d.metrics[k].reference_score===null))return '<p class="five-chart-note">判定を保留した項目があるため、グラフは表示していません。</p>';
        const scale=d.score_scale ?? 10;
        const point=(i,value)=>{const a=-Math.PI/2+i*2*Math.PI/5;return `${(190+Math.cos(a)*value*100/scale).toFixed(1)},${(145+Math.sin(a)*value*100/scale).toFixed(1)}`;};
        const polygon=values=>values.map((v,i)=>point(i,v)).join(' ');
        const old=root.FiveMetricCore.compare(d,previous)&&keys.every(k=>previous.metrics[k].reference_score!==null);
        const grid=[.2,.4,.6,.8,1].map(v=>v*scale).map(n=>`<polygon points="${polygon(keys.map(()=>n))}" fill="none" stroke="#b7c9d8"/>`).join('');
        const target=recommendedMetric===undefined?coaching(d,exclude).target:recommendedMetric;
        const text=keys.map((k,i)=>{const [x,y]=point(i,scale*1.3).split(',');return `<text x="${x}" y="${y}" text-anchor="middle" dominant-baseline="middle" fill="#24445f" font-size="15" font-weight="700">${labels[k]}<tspan x="${x}" dy="17" class="five-radar-value">${d.metrics[k].reference_score.toFixed(1)}</tspan></text>`;}).join('');
        const dots=keys.map((k,i)=>{const [x,y]=point(i,d.metrics[k].reference_score).split(',');return `<circle cx="${x}" cy="${y}" r="${k===target?6:4.5}" fill="${k===target?'#f4ac66':'#62dded'}" stroke="#112235" stroke-width="2"><title>${labels[k]} ${d.metrics[k].reference_score.toFixed(1)}点${k===target?'：次に試す項目':''}</title></circle>`;}).join('');
        return `<figure class="five-radar"><svg viewBox="0 0 380 310" role="img" aria-label="5項目の参考スコア。各項目に今回の点数を表示。">${grid}${old?`<polygon points="${polygon(keys.map(k=>previous.metrics[k].reference_score))}" fill="none" stroke="#62758a" stroke-width="1.5" stroke-dasharray="5 4"/>`:''}<polygon points="${polygon(keys.map(k=>d.metrics[k].reference_score))}" fill="#1f8fbd33" stroke="#14779e" stroke-width="3"/>${dots}${text}<text x="198" y="49" fill="#62758a" font-size="11">${scale}</text><text x="198" y="139" fill="#62758a" font-size="11">0</text></svg><figcaption>実線：今回${old?' ／ 点線：前回':''}（各${scale}点満点）<br><span class="five-recommend-dot" aria-hidden="true">●</span> 次に試すおすすめ項目</figcaption></figure>`;
    }
    function summaryHTML(d,previous,trial,restored,context={}){
        const view=describe(d,previous,trial,context);
        const candidate=root.TypeCandidate?.classify(d);
        const selected=candidate?.types??(candidate?.type?[candidate.type]:[]);
        const expressiveProgress=context.inlineSupplements?root.TypeCandidate?.experiment(d).expressiveProgress:null;
        const guide='<p class="five-type-guide">今回の録音から見た声のタイプの目安です。話し方や録音環境によって、結果が変わることがあります。</p>';
        const typeHTML=selected.length?guide+(selected.length===1?
            `<h2 class="five-type-title">${esc(root.TypeCandidate.labels[selected[0]])}タイプです</h2>${typeCommentHTML(selected[0],d,previous,trial,context)}`:
            `<h2 class="five-type-title">${selected.map(type=>esc(root.TypeCandidate.labels[type])).join(' ／ ')}</h2><p>今回は、両方の傾向が見られます。</p>${selected.map(type=>`<details><summary>${esc(root.TypeCandidate.labels[type])}の説明</summary>${typeCommentHTML(type,d,previous,trial,context)}</details>`).join('')}`):
            guide+`<h2>今回の声の特徴</h2>${expressiveProgress?.near?'':`<p>${candidate?.reason==='measurement_unavailable'?'声の特徴を十分に測定できませんでした。静かな場所でもう一度録音してください。':'今回はタイプを絞れませんでした。各項目の結果を参考にしてください。'}</p>`}${unclassifiedHTML(d,context.excludeMetric,context)}`;
        const total=root.FiveMetricCore.total(d),high=root.RecordingChange?.totalComment(d);
        const totalHTML=d.score_scale===20?`<p class="five-total">総合 <strong>${total===null?'判定保留':total.toFixed(1)+' / 100点'}</strong></p>${high?`<p>${esc(high)}</p>`:''}`:'';
        return `<section class="card five-radar-card"><h2>5項目のバランス</h2>${totalHTML}${radar(d,previous,context.excludeMetric,context.recommendedMetric)}</section><section class="card five-overview">${typeHTML}${view.available.length?'':`<p>${esc(view.overview)}</p>`}</section>`;
    }
    function unclassifiedHTML(d,exclude,context={}){
        const c=coaching(d,exclude);
        let html=`<section class="five-comment-step"><h3>あなたの声の個性</h3><p>${c.strong?`${labels[c.strong]}の参考値は、今回の参考範囲に入っています。録音した声とあわせて、今ある良さを確かめてみましょう。`:'測定できた項目と録音した声をあわせて、今回の声の特徴を確かめてみましょう。'}</p></section><section class="five-comment-step five-comment-challenge"><h3>このままだと</h3><p>数値だけでは、声の芯や力み、実際の聞き取りやすさまでは決められません。タイプ名を当てはめるよりも、今回の声と取り組みやすい項目を確認しましょう。</p></section><section class="five-comment-step"><h3>この声の可能性</h3><p>${c.target?`${labels[c.target]}のヒントを試し、前後の声を聴き比べてみましょう。`:'普段の声で、録音条件をそろえてもう一度試してみましょう。'}点数だけにこだわらず、自然に話せる感覚と、言葉の届き方を確かめていきます。</p></section>`;
        if(context.inlineSupplements&&d?.score_scale===20&&Object.values(d.metrics).every(m=>m.status==='provisional')){
            const supplement=context.supplement;
            const target=supplement?.targets?.length===1?supplement.targets[0]:context.recommendedMetric||c.target;
            const risk={
                brightness:'声の表情が乏しいと、内容は分かっても、気持ちが相手に届きにくくなることがあります。',
                articulation:'言葉が不鮮明になると、相手は聞き取ることに力を使います。伝えたい内容まで、受け取りにくくなることがあります。',
                power:'大切な言葉まで声が届かないと、伝えたい内容が十分に受け取られないことがあります。',
                speed:'話すペースが合わないと、相手が内容を追いにくくなることがあります。伝えたいことが十分に届かないのは、もったいないところです。',
                resonance:'声の良さが十分に伝わらないと、相手が話の内容に耳を傾けにくくなることがあります。'
            };
            const future={
                brightness:'声に表情が加わると、分かりやすさに、あなたの気持ちも重なる話し方へ近づきます。',
                articulation:'言葉が明確になると、相手の聞き取る負担が減り、伝えたい内容に耳を傾けてもらいやすくなります。',
                power:'大切な言葉がしっかり届くと、相手が安心して耳を傾けられる声へ近づきます。',
                speed:'話すペースが整うと、相手が内容を受け取りやすく、言葉が届く話し方へ近づきます。',
                resonance:'声の良さを引き出すことで、相手が耳を傾けやすく、伝えたい言葉が届く声を目指せます。'
            };
            const action=target?`まずは「${labels[target]}」を一つ。`:'';
            const replacement={
                'このままだと':risk[target]||'声の良さが十分に伝わらないままでは、伝えたい内容まで受け取りにくくなることがあります。',
                'この声の可能性':action+(future[target]||'今ある良さを活かし、あなたの言葉が相手に届く声を目指せます。')
            };
            for(const [heading,text] of Object.entries(replacement))html=html.replace(new RegExp('(<h3>'+heading+'</h3><p>)([^<]*)(</p>)'),(all,start,old,end)=>text.length<=old.length?start+esc(text)+end:all);
            const progress=root.TypeCandidate?.experiment(d).expressiveProgress;
            if(progress?.near){
                const near={
                    'あなたの声の個性':progress.tone==='developing'?'声の良さが整ってきています。今ある良さを活かして、表現者タイプを目指していけます。':'全体として、表現者タイプに近い点数です。今ある声の良さを大切にしていきましょう。',
                    'このままだと':'今の良さがあっても、大切な言葉が十分に届かないと、あなたの魅力を活かしきれないことがあります。',
                    'この声の可能性':progress.tone==='developing'?'今ある良さを残しながら、一つずつ整えていきましょう。あなたの言葉がもっと相手に届く声を目指せます。':`表現者タイプまで、あと一歩。目安まで、あと${progress.gap.toFixed(1)}点です。今ある良さを活かし、言葉がもっと相手に届く声を目指せます。`
                };
                for(const [heading,text] of Object.entries(near))html=html.replace(new RegExp('(<h3>'+heading+'</h3><p>)([^<]*)(</p>)'),(all,start,old,end)=>start+esc(text)+end);
            }
        }
        return html;
    }
    function typeCommentHTML(type,d,previous,trial,context={}){
        const steps=root.ThreeStepComments?.build(type,d,previous,trial,context);
        if(steps)return `<section class="five-comment-step"><h3>あなたの声の個性</h3><p>${esc(steps.individuality)}</p></section><section class="five-comment-step five-comment-challenge"><h3>このままだと</h3><p>${esc(steps.challenge)}</p></section><section class="five-comment-step"><h3>この声の可能性</h3><p>${esc(steps.potential)}</p>${steps.change?`<p>${esc(steps.change)}</p>`:''}</section>`;
        const material=(root.TypeCommentMaterial||[]).filter(e=>e.type===type);
        const pick=section=>material.find(e=>e.section===section)?.text;
        const feature=material.find(e=>!e.section)?.text;
        return [['',pick('strength')], [type==='expressive'?'特徴':'課題',feature],['伸びしろ',pick('potential')]].filter(([,text])=>text).map(([title,text])=>`${title?`<h3>${title}</h3>`:''}<p>${esc(text)}</p>`).join('');
    }
    function typeGalleryHTML(){
        return `<details class="card five-optional"><summary>5タイプの診断文を見る（テスト確認用）</summary><div class="five-optional-content"><p>文章全体を確認するための一覧です。ここにあるタイプすべてが、今回のあなたの判定ではありません。</p>${Object.entries(root.TypeCandidate?.labels||{}).map(([type,label])=>`<section><h2>${esc(label)}</h2>${typeCommentHTML(type)}<p>ヒントを試し、もう一度録音して変化を確かめましょう。</p></section>`).join('')}</div></details>`;
    }
    function destination(config){
        if(!config || config.enabled!==true || !['line','form','video'].includes(config.type))return null;
        try{const u=new URL(config.url);if(u.protocol!=='https:'||u.username||u.password)return null;return {url:u.href,type:config.type,label:{line:'LINE登録でZoom無料声診断を受ける',form:'Zoom無料声診断の応募案内へ',video:'声診断の解説動画を見る'}[config.type]};}catch(_){return null;}
    }
    function zoomCopy(state){
        const copy={
            increase:'今回、試した項目の参考値が上がりました。プロのボイストレーナーが、実際の声を聴き、どんな変化があったのか、無理なく再現できる方法は何かを一緒に確かめます。',
            decrease:'今回は参考値が下がりました。録音条件や力みなど、数値だけでは理由を特定できません。プロのボイストレーナーが、声を聴き、その方法が合うか、別の方法がよいかを一緒に確かめます。',
            small:'今回は数値の大きな変化がありませんでした。プロのボイストレーナーが、声や話す様子を確認し、どこから取り組むとよいか、あなたに合う方法を一緒に探します。',
            trial_small:'試した項目には、数値の大きな変化がありませんでした。プロのボイストレーナーが、声や話す様子を確認し、その方法との相性や、別の取り組み方を一緒に探します。',
            balanced:'今回は5項目とも参考範囲内でした。プロのボイストレーナーが、実際の声を聴き、仕事や人前で話す場面に合わせて、今ある声の強みをどう活かすかをお伝えします。',
            changed:'前回から参考値に変化がありました。プロのボイストレーナーが、実際の声を聴き、聞き手にどう伝わる変化か、あなたの目的に合っているかを一緒に確かめます。',
            first:'短い録音だけでは分からない声の癖や、仕事で声を使うときの悩みもあります。プロのボイストレーナーが、声を聴き、あなたの目的に合う練習と取り組む順番をお伝えします。',
            partial:'今回は一部の測定を保留しました。数値が出なかったことを声の弱点とは判断できません。Zoomでは実際の声を聴きながら、悩みや目的に合わせた方法を確認します。',
            unavailable:'今回は録音条件を確認して、もう一度お試しください。声の悩みを直接相談したい場合は、Zoomで実際の声を聴きながら練習方法を一緒に確認できます。'
        };
        return copy[state]||copy.first;
    }
    function zoomHTML(view,config){
        const link=destination(config);
        const button=link?`<a class="five-exit" href="${esc(link.url)}" target="_blank" rel="noopener noreferrer">${link.label}</a>`:'';
        return `<section class="card five-next-step"><h3>次のSTEP</h3><p>30分のZoom無料声診断では、オンラインで実際の声を聴きながら、あなたの声の癖や、伝わりにくくなっているポイントを一緒に確認します。今ある声の良さを活かし、もっと伝わりやすくなるための改善方法を、その場でお伝えします。</p><p>Zoom無料声診断は、LINE登録から無料で受けられます。登録特典もご用意しています。すでにLINE登録済みで、まだ診断を受けていない方もご利用いただけます。</p></section><section class="card five-zoom" id="fiveZoom"><h2>Zoom無料声診断を受ける</h2><div class="five-line-gift"><p>🎁 LINE登録特典</p><strong>やってはいけない<br>「逆効果になる声の練習」リスト</strong><p>友だち追加後すぐにお届けします。</p></div><p>35年以上・2万人以上の声を見てきたプロが、あなたの声の強みや、声で損しているポイントを確認し、あなたに合った改善方法をわかりやすくお伝えします。</p>${button}<p class="five-line-reassurance">無理な勧誘・営業は一切ありません。<br>安心してお気軽にご利用ください。</p></section>`;
    }
    const api={describe,radar,summaryHTML,destination,zoomCopy,zoomHTML,metricChangeHTML,deltaClass,coaching,typeGalleryHTML,typeCommentHTML};root.ResultExperience=api;
    if(typeof module!=='undefined')module.exports=api;
})(globalThis);
