const {test}=require('node:test');
const assert=require('node:assert/strict');
require('../static/diagnosis-preview.js');
require('../static/hint-engine.js');
const comments=require('../static/supplement-comments.js');
const keys=['brightness','articulation','power','speed','resonance'];
const make=values=>({schema_version:'five-metric-diagnosis-1',measurement_version:'m',calibration_version:'c',prompt_id:'p',score_scale:20,metrics:Object.fromEntries(keys.map((k,i)=>[k,{status:'provisional',direction:k==='speed'?'fast':'low',raw_value:1,reference_score:values[i]}]))});
const catalog=keys.map(k=>({target_metric:k,lifecycle_state:'preview',speed_direction:k==='speed'?'fast':null}));
const build=(d,rows=[{diagnosis:d}],context={})=>comments.build(d,catalog,rows,context);
test('14 and 14.5 boundaries distinguish focus, near, and no supplement',()=>{
 for(const [score,kind] of [[13.9,'focus'],[14,'near'],[14.4,'near'],[14.5,null]]){
  const d=make([15,score,16,17,15]);assert.equal(build(d).kind,kind);
 }
 const d=make([15,14.2,16,17,15]);assert.match(build(d).text,/あと0.3点/);
});
test('tied targets invite a choice without inventing a priority',()=>{
 const d=make([13,13,16,17,15]),result=build(d);
 assert.deepEqual(result.targets,['brightness','articulation']);assert.match(result.text,/一つ選んで/);
});
test('missing scores cannot become low scores or near claims',()=>{
 const d=make([15,13,16,17,15]);d.metrics.speed={status:'unavailable',direction:'unknown',raw_value:null,reference_score:null};
 assert.match(build(d).text,/測れた項目では/);
 d.metrics.articulation.reference_score=14.2;assert.equal(build(d).kind,null);
});
test('unknown speed direction and high power do not offer mismatched hints',()=>{
 const d=make([15,15,13,13,15]);d.metrics.speed.direction='unknown';d.metrics.power.direction='high';
 assert.equal(build(d).text,'');
});
test('trial target stays until the existing switch decision releases it',()=>{
 const d=make([12,13,16,17,15]),rows=[{diagnosis:d,trial:{target_metric:'articulation'}}];
 assert.deepEqual(build(d,rows).targets,['articulation']);assert.equal(build(d,rows).text,'');
 assert.deepEqual(build(d,rows,{excludeMetric:'articulation'}).targets,['brightness']);
});
test('restoring the same history keeps the text; new comparable results rotate it',()=>{
 const d=make([15,13,16,17,15]),rows=[{diagnosis:d}];
 assert.equal(build(d,rows).text,build(d,rows).text);
 const first=build(d,rows).text;rows.push({diagnosis:d});assert.notEqual(build(d,rows).text,first);
 const incompatible=make([15,13,16,17,15]);incompatible.prompt_id='other';
 assert.equal(build(d,[{diagnosis:incompatible},...rows]).text,build(d,rows).text);
});
test('natural guidance suppresses repeated meaning and waits two successes',()=>{
 const input={state:'decline',successCount:4,lastShown:1};
 assert.ok(comments.naturalText(input));
 for(const existingText of ['リラックスして','声を作ろうとせず','焦らず'])assert.equal(comments.naturalText({...input,existingText}),'');
 assert.equal(comments.naturalText({...input,hintText:'無理に声を張らずに届ける'}),'');
 for(const successCount of [1,2,3])assert.equal(comments.naturalText({...input,successCount}),'');
 assert.equal(comments.naturalText({...input,complete:false}),'');
 assert.equal(comments.naturalText({...input,state:'rise'}),'');
});
