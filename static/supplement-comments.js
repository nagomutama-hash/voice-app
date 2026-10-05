(function(root){
 'use strict';
 const keys=['brightness','articulation','power','speed','resonance'];
 const labels={brightness:'明るさ',articulation:'滑舌',power:'声の安定感',speed:'話す速度',resonance:'響き'};
 const natural=['声を作ろうとしすぎず、いつもの声でヒントを試してみてください。','力を入れすぎず、リラックスして話してみましょう。','焦らず、いつもの声で試してみてください。'];
 function choose(d,catalog,trial,exclude=null){
  if(d?.score_scale!==20||!root.FiveMetricCore.valid(d))return {kind:null,targets:[]};
  const measured=keys.filter(k=>d.metrics[k].status==='provisional');
  const eligible=k=>k!==exclude&&catalog.some(h=>h.target_metric===k&&root.HintEngine.eligible(d,h));
  if(trial&&trial.target_metric!==exclude)return {kind:null,targets:eligible(trial.target_metric)?[trial.target_metric]:[],trial:true};
  const low=measured.filter(k=>d.metrics[k].reference_score<14);
  const kind=low.length?'focus':measured.length===5?'near':null;
  const candidates=(low.length?low:measured.filter(k=>d.metrics[k].reference_score>=14&&d.metrics[k].reference_score<14.5)).filter(eligible);
  if(!kind||!candidates.length)return {kind:null,targets:[]};
  const minimum=Math.min(...candidates.map(k=>d.metrics[k].reference_score));
  return {kind,targets:candidates.filter(k=>d.metrics[k].reference_score===minimum),partial:measured.length<5};
 }
 function text(selection,index=0,audioAvailable=false){
  if(!selection.kind||!selection.targets.length)return '';
  if(selection.targets.length>1)return `今回は「${selection.targets.map(k=>labels[k]).join('」「')}」が同じ点数です。下から一つ選んで、ヒントを試してみましょう。`;
  const name=labels[selection.targets[0]],variant=index%3;
  if(selection.kind==='focus')return (selection.partial?'測れた項目では、':'')+[
   `今回は、まず「${name}」から試してみましょう。一度に全部を変えなくて大丈夫です。`,
   `まずは「${name}」を一つ。ヒントを試して、声がどう変わるか確かめてみましょう。`,
   `今回、先に取り組みたいのは「${name}」です。ほかの項目は、ひとまずそのままで大丈夫です。`][variant];
  return [
   `「${name}」は、今回の目安まであと${selection.gap.toFixed(1)}点です。今の声を大きく変えずに、ヒントを一つ試してみましょう。`,
   `あと少し見てみたいのは「${name}」です。今の良さを残しながら、ヒントを試してみましょう。`,
   `今回は「${name}」をもう少し。ヒントを試して、${audioAvailable?'前後の声':'今回の声'}を確かめてみましょう。`][variant];
 }
 function build(d,catalog,rows,context={}){
  const latest=rows.at(-1),selection=choose(d,catalog,latest?.trial,context.excludeMetric);
  if(selection.kind==='near'&&selection.targets.length===1)selection.gap=14.5-d.metrics[selection.targets[0]].reference_score;
  const comparable=rows.filter(r=>root.FiveMetricCore.compare(d,r.diagnosis));
  selection.text=text(selection,Math.max(0,comparable.length-1),context.audioAvailable);
  return selection;
 }
 // 録音準備時だけ使用。診断成功2回を挟むまで次の自然案内を出さない。
 function naturalText({state,existingText='',hintText='',successCount=0,lastShown=-Infinity,variant=0,complete=true}){
  if(!complete||!['decline','maintained'].includes(state)||successCount-lastShown<=2||/リラックス|力を|声を作|自然|いつもの声|無理に声を張らず|焦らず/.test(existingText+' '+hintText))return '';
  return natural[variant%3];
 }
 const api={choose,build,text,naturalText};root.SupplementComments=api;
 if(typeof module!=='undefined')module.exports=api;
})(globalThis);
