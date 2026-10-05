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
  const result={individuality:base.individuality,challenge:base.challenge,potential,change,state:selected?.state||'base',variant};
  if(context.inlineSupplements){
   const split=text=>text.match(/[^。！？]+[。！？]/g)||[text];
   const individual=split(base.individuality),challenge=split(base.challenge),future=split(base.potential);
   result.individuality=variant===1&&individual.length>2?individual.slice(0,-1).join(''):variant===2&&individual.length>2?[individual[0],individual.at(-1)].join(''):base.individuality;
   // 課題は聴き手への影響を保つ。練習案内へ置き換えない。
   result.challenge=variant===1?challenge.slice(0,2).join(''):variant===2&&challenge.length>1?challenge[1]:base.challenge;
   const supplement=context.supplement;
   const targets=supplement?.targets||[];
   const action=targets.length===1?`まずは「${labels[targets[0]]}」を一つ。`:targets.length>1?'まずは一つ、取り組む項目を選んでみましょう。':'';
   const core=type==='expressive'?future.slice(0,2).join(''):future[0];
   // 必ず声がどう届くようになるかで締める。数値の上下で未来を上書きしない。
   const endings=[core,future.at(-1),core][variant];
   const addition=supplement?.kind==='near'&&targets.length===1?`「${labels[targets[0]]}」は、今の良さを残しながらあと一歩。`:action;
   result.potential=addition.length+endings.length<=base.potential.length?addition+endings:core;
   if(type==='unclear'){
    const strong=Object.keys(labels).filter(k=>d?.metrics?.[k]?.status==='provisional'&&['optimal','within_reference'].includes(d.metrics[k].direction)).sort((a,b)=>d.metrics[b].reference_score-d.metrics[a].reference_score)[0];
    result.individuality=strong?`「${labels[strong]}」は、今回の参考範囲に入っています。今ある声の良さを大切にしていきましょう。`:'あなたの声には、これから引き出していける良さがあります。言葉の届き方を、一つずつ整えていきましょう。';
   }
   if(selected&&['rise','clear_rise','high_maintained'].includes(selected.state)){
    const praise=selected.state==='high_maintained'?`「${labels[key]}」も、高い点数を保てています。`:`「${labels[key]}」の点数も、前回より上がりました。`;
    const head=split(result.individuality)[0];
    if(head.length+praise.length<=base.individuality.length)result.individuality=head+praise;
   }
   result.change='';
   const progress=root.TypeCandidate?.experiment(d).expressiveProgress;
   if(progress?.near){
    result.individuality=progress.tone==='developing'?'声の良さが整ってきています。今ある良さを活かして、表現者タイプを目指していけます。':'全体として、表現者タイプに近い点数です。今ある声の良さを大切にしていきましょう。';
    result.challenge='今の良さがあっても、大切な言葉が十分に届かないと、あなたの魅力を活かしきれないことがあります。';
    result.potential=progress.tone==='developing'?'今ある良さを残しながら、一つずつ整えていきましょう。あなたの言葉がもっと相手に届く声を目指せます。':`表現者タイプまで、あと一歩。目安まで、あと${progress.gap.toFixed(1)}点です。今ある良さを活かし、言葉がもっと相手に届く声を目指せます。`;
   }
  }
  return result;
 }
 root.ThreeStepComments={build};if(typeof module!=='undefined')module.exports=root.ThreeStepComments;
})(globalThis);
