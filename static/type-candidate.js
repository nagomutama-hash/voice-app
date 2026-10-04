(function(root){
 'use strict';
 const keys=['brightness','articulation','power','speed','resonance'];
 const labels={expressive:'表現者',soft:'ふわっと優しすぎ',pushy:'圧強めセールス',cerebral:'頭でっかち説明',unclear:'こもり声・不鮮明'};
 function candidate(type,note){return {type,reason:'auxiliary_trial',label:labels[type]+'タイプの候補',note};}
 // テスト用の数値傾向。聞き取り・力み・顔・芯を確認した判定ではない。
 function classify(d){
  if(!root.FiveMetricCore.valid(d)||keys.some(k=>d.metrics[k].status!=='provisional'))return {type:null,reason:'partial'};
  const scale=d.score_scale??10;
  const scores=keys.map(k=>d.metrics[k].reference_score/scale);
  // 速さの適正から外れたことを、言葉が聞き取れない根拠にしない。
  const clarityScores=scores.filter((_,i)=>keys[i]!=='speed');
  if(clarityScores.filter(v=>v<.6).length>=3||clarityScores.some(v=>v<.4))return {type:'unclear',reason:'low_score_pattern',label:'こもり声・不鮮明タイプの候補',note:'速度以外の複数の低い参考点、または特に低い参考点があるためのテスト候補です。実際の聞き取りやすさも確認してください。'};
  const aux=d.type_evidence||d.type_auxiliary;
  const spectrum=aux?.spectrum?.high_to_low_energy_db_median;
  const pitch=aux?.pitch?.range_semitones;
  // 未校正の代理条件。音量安定感は芯・力み・抑揚の代用にしない。
  if(aux?.status==='experimental'){
   if(Number.isFinite(spectrum)&&spectrum>-5&&scores[1]>=.8)return candidate('pushy','高い帯域の成分と音の切り替わりの参考点を組み合わせた仮候補です。力みを直接測定した判定ではありません。');
   if(Number.isFinite(pitch)&&pitch<7.5&&Number.isFinite(spectrum)&&spectrum<-15&&scores[1]<.7)return candidate('cerebral','高さの変化幅、帯域の成分、音の切り替わりを組み合わせた仮候補です。顔の表情・考え方・自然な抑揚を判定したものではありません。');
   if(Number.isFinite(spectrum)&&spectrum<-8&&scores[0]<.9&&scores[1]>=.6&&scores[1]<.8&&scores[4]>=.8)return candidate('soft','帯域の成分と明るさ・音の切り替わり・響きの参考点を組み合わせた、話し方のテスト用候補です。芯を直接測っておらず、芯のある声を含む可能性があります。');
  }
  if(scores.every(v=>v>=.73)&&keys.every(k=>['within_reference','optimal'].includes(d.metrics[k].direction)))return {type:'expressive',reason:'balanced_scores',label:'表現者タイプの候補',note:'5項目の参考点がそろっている数値傾向です。力みや声の表情まで確認した判定ではありません。'};
  return {type:null,reason:'insufficient_type_evidence'};
 }
 const api={classify,labels};root.TypeCandidate=api;if(typeof module!=='undefined')module.exports=api;
})(globalThis);
