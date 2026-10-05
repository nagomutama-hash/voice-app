(function(root){
    'use strict';
    // Draft candidates are enabled only in the opt-in preview, never approved for public use.
    function eligible(d,h){
        const m=d?.metrics?.[h.target_metric];
        if(!m || m.status!=='provisional' || h.lifecycle_state!=='preview')return false;
        if(h.target_metric==='speed')return m.direction===h.speed_direction;
        return !['high','unknown'].includes(m.direction);
    }
    function trials(rows,catalog,d){
        return rows.filter(r=>r.trial && root.FiveMetricCore.compare(d,r.diagnosis) &&
            catalog.some(h=>h.hint_id===r.trial.hint_id && h.hint_version===r.trial.hint_version));
    }
    function delta(row){return root.FiveMetricCore.compare(row.diagnosis,row.trial.before)?.[row.trial.target_metric]??null;}
    function highMaintained(row,rows){
        if(!root.RecordingChange?.select||!root.FiveMetricCore.baseline)return false;
        const first=root.FiveMetricCore.baseline(rows,row.diagnosis,row.at);
        return root.RecordingChange.select(row.diagnosis,row.trial.before,first,row.trial.target_metric,false)?.state==='high_maintained';
    }
    function next(catalog,d,rows,group,shown=[]){
        const past=trials(rows,catalog,d);
        const pool=catalog.filter(h=>h.group===group && eligible(d,h));
        const used=[...past.map(r=>r.trial.hint_id),...shown];
        const fresh=pool.filter(h=>!used.includes(h.hint_id));
        const candidates=fresh.length?fresh:pool.filter(h=>h.hint_id!==used.at(-1));
        const previous=catalog.find(h=>h.hint_id===used.at(-1));
        return candidates.sort((a,b)=>{
            if(!fresh.length){const age=used.lastIndexOf(a.hint_id)-used.lastIndexOf(b.hint_id);if(age)return age;}
            const kind=Number(a.mechanism_type===previous?.mechanism_type)-Number(b.mechanism_type===previous?.mechanism_type);
            if(kind)return kind;
            const failed=h=>past.filter(r=>r.trial.mechanism_type===h.mechanism_type && r.trial.target_metric===h.target_metric && delta(r)!==null && delta(r)<0.5).length;
            return failed(a)-failed(b)||a.initial_priority-b.initial_priority;
        })[0]||null;
    }
    function switchMetric(rows,catalog,d){
        const past=trials(rows,catalog,d),last=past.at(-1);
        if(!last)return null;
        const change=delta(last);
        if(change===null||highMaintained(last,rows))return null;
        if(change>=0.5)return last.trial.target_metric;
        const previous=past.at(-2);
        return previous && previous.trial.target_metric===last.trial.target_metric &&
            previous.trial.hint_id!==last.trial.hint_id && !highMaintained(previous,rows) && previous.trial.speed_direction===last.trial.speed_direction && delta(previous)!==null && delta(previous)<0.5?last.trial.target_metric:null;
    }
    function shouldSwitch(rows,catalog,d){return Boolean(switchMetric(rows,catalog,d));}
    function recommendGroup(catalog,d,groups){
        const available=groups.filter(group=>catalog.some(h=>h.group===group&&eligible(d,h)));
        const type=root.TypeCandidate?.classify(d)?.type;
        // 顔の表情、響き、速すぎる場合の間を、既存の方向に合うヒントへ接続。
        // 音量安定感を声の芯や力みの代用にしない。
        const preferred={cerebral:'brightness',soft:'resonance',pushy:'speed_fast'}[type];
        if(available.includes(preferred))return preferred;
        return available.slice().sort((a,b)=>{
            const metric=group=>catalog.find(h=>h.group===group&&eligible(d,h)).target_metric;
            return d.metrics[metric(a)].reference_score-d.metrics[metric(b)].reference_score;
        })[0]||null;
    }
    const api={eligible,next,delta,shouldSwitch,switchMetric,recommendGroup};
    root.HintEngine=api;
    if(typeof module!=='undefined')module.exports=api;
})(globalThis);
