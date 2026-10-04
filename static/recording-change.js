(function(root){
    'use strict';
    let policy=null,comments={},totals=[];
    function configure(config){
        policy=config.change_policy;
        comments=Object.fromEntries((config.change_comments||[]).map(e=>[e.section.replace('recording_change_',''),e]));
        totals=config.total_comments||[];
    }
    function select(current,previous,baseline,key,audioAvailable=false){
        const core=root.FiveMetricCore;
        if(!policy || current?.score_scale!==20 || previous?.score_scale!==20)return null;
        const delta=core.compare(current,previous)?.[key];
        if(typeof delta!=='number')return null;
        const high=core.compare(current,baseline)&&baseline.score_scale===20&&
            baseline.metrics[key].reference_score>=policy.high_score_baseline_min&&
            current.metrics[key].reference_score>=policy.high_score_current_min&&
            (key!=='speed'||current.metrics[key].direction==='optimal'&&baseline.metrics[key].direction==='optimal');
        const state=delta<=policy.large_decline_max?'decline':delta>=policy.bands[0].min_inclusive?'clear_rise':
            delta>=policy.bands[1].min_inclusive?'rise':high?'high_maintained':
            delta<=policy.bands[3].max_inclusive?'decline':'maintained';
        const entry=comments[state];
        if(!entry)return null;
        return {state,delta,comment_id:entry.id,text:state==='clear_rise'&&!audioAvailable?entry.text_without_comparison_audio:entry.text};
    }
    function totalComment(d){
        if(d?.score_scale!==20)return null;
        const score=root.FiveMetricCore.total(d);
        return score===null?null:totals.find(e=>score>=e.score_min_inclusive&&(e.score_max_exclusive===undefined||score<e.score_max_exclusive))?.text??null;
    }
    const api={configure,select,totalComment};root.RecordingChange=api;
    if(typeof module!=='undefined')module.exports=api;
})(globalThis);
