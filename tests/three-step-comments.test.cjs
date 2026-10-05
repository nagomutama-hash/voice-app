const {test}=require('node:test');const assert=require('node:assert/strict');const fs=require('node:fs');
require('../static/diagnosis-preview.js');require('../static/three-step-material.js');
const change=require('../static/recording-change.js');
change.configure({change_policy:JSON.parse(fs.readFileSync('knowledge_comments/recording_change_policy_v1.json')),change_comments:JSON.parse(fs.readFileSync('knowledge_comments/approved/recording_change_comments_v1.json')).entries});
const comments=require('../static/three-step-comments.js');
const make=score=>({schema_version:'five-metric-diagnosis-1',score_scale:20,measurement_version:'m',calibration_version:'c',prompt_id:'p',metrics:Object.fromEntries(['brightness','articulation','power','speed','resonance'].map(k=>[k,{status:'provisional',raw_value:1,direction:k==='speed'?'optimal':'within_reference',reference_score:score}]))});
test('every type varies its wording while preserving the approved individuality and challenge',()=>{
 for(const type of ['soft','pushy','cerebral','unclear','expressive']){
  const variants=[0,1,2].map(commentVariant=>comments.build(type,make(15),null,null,{commentVariant}));
  assert.equal(new Set(variants.map(v=>v.potential)).size,3);
  assert.equal(new Set(variants.map(v=>v.individuality)).size,1);
  assert.equal(new Set(variants.map(v=>v.challenge)).size,1);
  assert.ok(variants.every(v=>v.change===''));
 }
});
test('trial comments use the trial baseline and withhold incompatible measurements',()=>{
 const before=make(15),d=make(16);const trial={target_metric:'power',before};
 const result=comments.build('soft',d,make(19),trial,{baseline:before});
 assert.equal(result.state,'rise');assert.match(result.change,/声の安定感/);
 assert.doesNotMatch(result.change,/芯が.*改善|腹式/);
 trial.before=make(15);trial.before.measurement_version='old';
 assert.equal(comments.build('soft',d,before,trial).change,'');
});
test('ordinary rerecording chooses the largest measured change and recognises high maintenance',()=>{
 const before=make(15),d=make(15);d.metrics.resonance.reference_score=12;
 assert.equal(comments.build('soft',d,before,null).state,'decline');
 const high=make(19);assert.equal(comments.build('expressive',high,high,{target_metric:'power',before:high},{baseline:high}).state,'high_maintained');
});

test('three steps keep motivation and a future ending within original length limits',()=>{
 for(const type of Object.keys(global.ThreeStepMaterial.types))for(const commentVariant of [0,1,2])for(const kind of ['focus','near',null]){
  const base=global.ThreeStepMaterial.types[type],before=make(15),d=make(14);
  const result=comments.build(type,d,before,null,{inlineSupplements:true,commentVariant,supplement:{kind,targets:kind?['resonance']:[],text:'まずは「響き」を一つ。'}});
  for(const section of ['individuality','challenge','potential'])assert.ok(result[section].length<=base[section].length,`${type}/${section}`);
  assert.doesNotMatch(result.challenge,/ヒント|まずは|点数/);
  assert.match(result.challenge,/届|伝わ|受け取|身構|活か|魅力|引き込|急か|押し切/);
  assert.match(result.potential,/近づ|目指|届|伝わ|耳を傾け|なります/);
  assert.doesNotMatch(result.potential,/点数|前回より下が/);
  assert.equal(result.change,'');
  if(kind==='focus')assert.match(result.potential,/「響き」/);
 }
});
test('positive changes reinforce praise while declines never replace the future',()=>{
 const before=make(15);
 for(const score of [14,16]){
  const result=comments.build('expressive',make(score),before,null,{inlineSupplements:true,commentVariant:1});
  if(score===16)assert.match(result.individuality,/前回より上がりました/);
  assert.doesNotMatch(result.potential,/点数|下がりました/);
  assert.match(result.potential,/伝わる声/);
 }
});
test('unclear starts with a measured strength without inventing one for missing data',()=>{
 const d=make(12);d.metrics.articulation.reference_score=16;
 assert.match(comments.build('unclear',d,null,null,{inlineSupplements:true}).individuality,/「滑舌」/);
 for(const m of Object.values(d.metrics))m.direction='low';
 assert.doesNotMatch(comments.build('unclear',d,null,null,{inlineSupplements:true}).individuality,/参考範囲に入っています/);
});
test('near expressive shows the real total gap while retaining three steps and length limits',()=>{
 require('../static/type-candidate.js');
 const d=make(15);[17.2,16.4,13.9,19.2,13.2].forEach((v,i)=>d.metrics[Object.keys(d.metrics)[i]].reference_score=v);
 for(const type of Object.keys(global.ThreeStepMaterial.types)){
  const result=comments.build(type,d,null,null,{inlineSupplements:true});
  assert.match(result.individuality,/表現者タイプに近い/);
  assert.match(result.challenge,/魅力を活かしきれない/);
  assert.match(result.potential,/あと0.1点/);
  for(const section of ['individuality','challenge','potential'])assert.ok(result[section].length<=global.ThreeStepMaterial.types[type][section].length);
 }
 d.metrics.resonance.reference_score=12;
 assert.doesNotMatch(comments.build('soft',d,null,null,{inlineSupplements:true}).potential,/表現者タイプの目安まで/);
});
