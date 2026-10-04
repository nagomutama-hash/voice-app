(function(root){
 'use strict';
 // 承認素材の選択と組合せ。実行時の自由生成や音声の外部送信は行わない。
 function build(type,d,previous,trial,context={}){
  const base=root.ThreeStepMaterial?.types[type];
  if(!base)return null;
  const variant=Math.abs(Math.floor(context.commentVariant||0))%3;
  const sentences=base.potential.match(/[^。]+。/g)||[base.potential];
  // 中間の文だけを切り替え、個性・課題の判断と未来の核は保つ。
  const potential=variant===1&&sentences.length>2?sentences.filter((_,i)=>i!==1).join(''):
   variant===2&&sentences.length>2?[sentences[0],...sentences.slice(2),sentences[1]].join(''):base.potential;
  let selected=null,key=trial?.target_metric;
  const before=trial?trial.before:previous;
  const delta=root.FiveMetricCore.compare(d,before);
  if(!trial&&delta)key=Object.keys(delta).filter(k=>typeof delta[k]==='number').sort((a,b)=>Math.abs(delta[b])-Math.abs(delta[a]))[0];
  if(key&&delta)selected=root.RecordingChange?.select(d,before,context.baseline,key,context.audioAvailable);
  const labels={brightness:'明るさ',articulation:'滑舌',power:'声の安定感',speed:'話す速度',resonance:'響き'};
  const change=selected?`${labels[key]}について、${selected.text}`:'';
  return {individuality:base.individuality,challenge:base.challenge,potential,change,state:selected?.state||'base',variant};
 }
 root.ThreeStepComments={build};if(typeof module!=='undefined')module.exports=root.ThreeStepComments;
})(globalThis);
