const {test}=require('node:test');
const assert=require('node:assert/strict');
require('../static/diagnosis-preview.js');
const experience=require('../static/result-experience.js');
const make=()=>({schema_version:'five-metric-diagnosis-1',measurement_version:'m1',calibration_version:'c1',prompt_id:'p1',metrics:Object.fromEntries(['brightness','articulation','power','speed','resonance'].map(k=>[k,{status:'provisional',direction:k==='speed'?'fast':'low',raw_value:1,reference_score:6}]))});
const missing=(d,k)=>{d.metrics[k]={status:'unavailable',direction:'unknown',raw_value:null,reference_score:null};return d;};

test('coaching praises a measured reference feature and excludes wrong-direction challenges',()=>{
 const d=make();d.metrics.speed.direction='optimal';d.metrics.speed.reference_score=10;
 d.metrics.brightness.direction='high';d.metrics.brightness.reference_score=2;
 d.metrics.power.reference_score=3;
 assert.deepEqual(experience.coaching(d),{strong:'speed',target:'power'});
 assert.notEqual(experience.coaching(d,'power').target,'power');
 missing(d,'power');assert.notEqual(experience.coaching(d).target,'power');
 const html=experience.summaryHTML(d);
 assert.doesNotMatch(html,/次に試すポイント|今回の良いところ/);
});
test('single high scores do not claim balanced voice or infer personality',()=>{
 const d=make();for(const m of Object.values(d.metrics))m.reference_score=9;
 const v=experience.describe(d);assert.equal(v.state,'first');assert.doesNotMatch(experience.summaryHTML(d),/圧強め|セールス|安定しています/);
});
test('all five reference directions are required for balanced message',()=>{
 const d=make();for(const [k,m] of Object.entries(d.metrics))m.direction=k==='speed'?'optimal':'within_reference';
 assert.equal(experience.describe(d).state,'balanced');missing(d,'power');assert.equal(experience.describe(d).state,'partial');
});
test('missing scores never form a zero-filled radar chart',()=>{
 const d=missing(make(),'power');assert.doesNotMatch(experience.radar(d),/<svg|<polygon/);
 for(const k of Object.keys(d.metrics))missing(d,k);assert.equal(experience.describe(d).state,'unavailable');
});
test('radar previous polygon requires complete comparable version',()=>{
 const d=make(),p=make();assert.match(experience.radar(d,p),/点線：前回/);
 p.calibration_version='other';assert.doesNotMatch(experience.radar(d,p),/点線：前回/);
 p.calibration_version='c1';missing(p,'speed');assert.doesNotMatch(experience.radar(d,p),/点線：前回/);
});
test('trial uses its own baseline and preserves signed change',()=>{
 const before=make(),d=make(),unrelated=make();unrelated.metrics.power.reference_score=2;
 for(const [score,state] of [[6.5,'increase'],[5.5,'decrease'],[6.1,'trial_small']]){
  d.metrics.power.reference_score=score;assert.equal(experience.describe(d,unrelated,{target_metric:'power',before}).state,state);
 }
});
test('small target change does not hide large change in another metric',()=>{
 const d=make(),p=make();d.metrics.speed.reference_score=7;
 const v=experience.describe(d,p,{target_metric:'power',before:p});assert.equal(v.state,'trial_small');assert.match(v.changeText,/話す速度 \+1.0/);
});
test('incompatible baselines do not produce change claims',()=>{
 const d=make(),p=make();p.measurement_version='old';p.metrics.power.reference_score=1;
 const v=experience.describe(d,p,{target_metric:'power',before:p});assert.equal(v.state,'first');assert.equal(v.changeText,'');
});
test('largest absolute change includes decreases',()=>{
 const d=make(),p=make();d.metrics.power.reference_score=4;d.metrics.speed.reference_score=7;
 assert.match(experience.describe(d,p).changeText,/声の安定感 -2.0/);
});
test('an unavailable previous recording cannot be called unchanged',()=>{
 const d=make(),p=make();for(const key of Object.keys(p.metrics))missing(p,key);
 assert.equal(experience.describe(d,p).state,'first');assert.equal(experience.describe(d,p).changeText,'');
});
test('pending, malformed, executable and credential URLs cannot create CTA links',()=>{
 for(const config of [null,{}, {enabled:false,type:'line',url:'https://lin.ee/0uZqUu4'},...['javascript:alert(1)','http://example.com','https://user:pass@example.com','bad'].map(url=>({enabled:true,type:'line',url}))]){
  assert.equal(experience.destination(config),null);assert.doesNotMatch(experience.zoomHTML({state:'first'},config),/href=/);
 }
});
test('approved LINE configuration opens exact destination without result data',()=>{
 const config=require('../static/experience-config.json');
 assert.deepEqual(experience.destination(config.exit),{url:'https://lin.ee/0uZqUu4',type:'line',label:'LINE登録でZoom無料声診断を受ける'});
 const html=experience.zoomHTML({state:'increase'},config.exit);assert.match(html,/noopener noreferrer/);assert.match(html,/逆効果になる声の練習/);
});
test('result variants all have contextual copy and avoid guaranteed improvement',()=>{
 for(const state of ['first','increase','decrease','small','trial_small','balanced','changed','partial','unavailable']){
  assert.ok(experience.zoomCopy(state).length>30);assert.doesNotMatch(experience.zoomCopy(state),/必ず|改善しました|原因は.*です/);
 }
});
