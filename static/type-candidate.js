(function(root){
 'use strict';
 const keys=['brightness','articulation','power','speed','resonance'];
 const labels={expressive:'表現者',soft:'ふわっと優しすぎ',pushy:'圧強めセールス',cerebral:'頭でっかち説明',unclear:'こもり声・不鮮明'};
 function experiment(d){
  const version='teacher-numeric-20261005-4';
  if(!root.FiveMetricCore.valid(d))return {version,status:'incomplete',types:[]};
  const scores=Object.fromEntries(keys.map(k=>[k,d.metrics[k].status==='provisional'?d.metrics[k].reference_score*20/(d.score_scale??10):null]));
  const values=Object.values(scores),complete=values.every(Number.isFinite);
  const total=complete?Math.round(values.reduce((sum,v)=>sum+v,0)*10)/10:null;
  const floorPassed=complete&&values.every(v=>v>12);
  const gap=floorPassed&&total<80?Math.round((80-total)*10)/10:null;
  const near=floorPassed&&total>=76&&total<80;
  const expressiveProgress={total,qualified:floorPassed&&total>=80,near,gap,
   tone:near?(gap<=2.5&&values.filter(v=>v<14).length<=2?'one_step':'developing'):null};
  const lowCounts=Object.fromEntries(keys.map(k=>[k,scores[k]===null?null:scores[k]<10?2:scores[k]<14?1:0]));
  const lowTotal=Object.values(lowCounts).reduce((sum,v)=>sum+(v??0),0);
  const first=[];
  if(lowTotal>=4)first.push('unclear');
  if(expressiveProgress.qualified)first.push('expressive');
  const remaining=[];
  if(!first.length){
   if(scores.brightness!==null&&scores.resonance!==null&&scores.brightness<14&&scores.resonance<14)remaining.push('cerebral');
   if(scores.brightness!==null&&scores.resonance!==null&&scores.power!==null&&scores.brightness>=14.5&&scores.resonance<14&&scores.power<14)remaining.push('soft');
   if(scores.power!==null&&scores.speed!==null&&scores.power>=16&&scores.speed<=14)remaining.push('pushy');
  }
  const types=first.length?first:remaining;
  return {version,status:complete?(types.length?'matched':'outside'):'incomplete',types,scores,lowCounts,lowTotal,expressiveProgress,
   boundaryKeys:keys.filter(k=>scores[k]!==null&&scores[k]>=14&&scores[k]<14.5),
   speedDirection:d.metrics.speed.direction,
   provisionalDefinition:'表現者は総合80.0以上・全項目12.0より高い。76.0以上80.0未満・同じ下限では差と項目のまとまりで近さの言葉を選ぶ仮表示。欠測は低得点に数えません。'};
 }
 function classify(d){
  const report=experiment(d);
  if(!root.FiveMetricCore.valid(d)||keys.filter(k=>k!=='speed').some(k=>d.metrics[k].status!=='provisional'))return {type:null,types:[],reason:'measurement_unavailable',version:report.version};
  return {type:report.types.length===1?report.types[0]:null,types:report.types,
   reason:report.types.length?'numeric_rules':report.status==='incomplete'?'measurement_unavailable':'outside_rules',version:report.version};
 }
 function explain(d){return {version:'type-decision-3',selected:classify(d),hypothesis:experiment(d),limitation:'録音から見たタイプの目安です。欠測を低得点として数えません。'};}
 const api={classify,explain,experiment,labels};root.TypeCandidate=api;if(typeof module!=='undefined')module.exports=api;
})(globalThis);
