const {test}=require('node:test');const assert=require('node:assert/strict');
require('../static/diagnosis-preview.js');const types=require('../static/type-candidate.js');
require('../static/type-comment-material.js');const experience=require('../static/result-experience.js');const hints=require('../static/hint-engine.js');
const keys=['brightness','articulation','power','speed','resonance'];
const make=values=>({schema_version:'five-metric-diagnosis-1',score_scale:20,measurement_version:'m',calibration_version:'c',prompt_id:'p',metrics:Object.fromEntries(keys.map((k,i)=>[k,{status:'provisional',raw_value:1,direction:k==='speed'?'optimal':'within_reference',reference_score:values[i]}]))});
test('weighted lows use strict fourteen and ten boundaries and unclear has priority',()=>{
 assert.equal(types.classify(make([12,15,15,9.8,12])).type,'unclear');
 assert.equal(types.classify(make([13.8,14.7,13.3,17,8.8])).type,'unclear');
 assert.equal(types.classify(make([13,15,15,10,13])).type,'cerebral');
 const report=types.experiment(make([14,14,14,14,14]));assert.equal(report.lowTotal,0);assert.equal(report.status,'outside');
 assert.equal(types.experiment(make([10,14,14,14,14])).lowTotal,1);
});
test('expressive uses total eighty and strictly excludes twelve or below',()=>{
 assert.equal(types.classify(make([17,17,14,19,13])).type,'expressive');
 assert.equal(types.classify(make([17,17,14,18.9,13])).type,null);
 assert.equal(types.classify(make([18,18,16,16,12])).type,null);
 assert.equal(types.classify(make([18,18,16,16,12.1])).type,'expressive');
 assert.equal(types.classify(make([18,18,18,13,13])).type,'expressive');
 const progress=types.experiment(make([17.2,16.4,13.9,19.2,13.2])).expressiveProgress;
 assert.deepEqual(progress,{total:79.9,qualified:false,near:true,gap:0.1,tone:'one_step'});
 assert.equal(types.experiment(make([16,16,14,19,13])).expressiveProgress.near,true);
 assert.equal(types.experiment(make([16,16,14,18.9,13])).expressiveProgress.near,true);
 assert.equal(types.experiment(make([18,18,16,15.9,12])).expressiveProgress.near,false);
});
test('flexible language uses both total gap and distribution within the fixed band',()=>{
 assert.equal(types.experiment(make([16.1,15.5,14.7,14.7,16.8])).expressiveProgress.tone,'one_step');
 assert.equal(types.experiment(make([16.7,16,15.8,12.9,15])).expressiveProgress.tone,'developing');
 assert.equal(types.experiment(make([13,13,13,20,20])).expressiveProgress.tone,'developing');
 assert.equal(types.experiment(make([16,16,15,14,15])).expressiveProgress.near,true);
 assert.equal(types.experiment(make([16,16,15,13.9,15])).expressiveProgress.near,false);
 const missing=make([16,16,16,16,16]);missing.metrics.speed={status:'unavailable',direction:'unknown',reference_score:null,raw_value:null};
 assert.equal(types.experiment(missing).expressiveProgress.near,false);
});
test('soft and pressure use their agreed boundaries without spectral or articulation gates',()=>{
 assert.equal(types.classify(make([14.5,15,13.9,15,13.9])).type,'soft');
 assert.equal(types.classify(make([14.4,15,13.9,15,13.9])).type,null);
 assert.equal(types.classify(make([15,13,16,14,15])).type,'pushy');
 assert.equal(types.classify(make([15,13,15.9,14,15])).type,null);
 assert.equal(types.classify(make([15,13,16,14.1,15])).type,null);
});
test('overlaps are explicit and do not silently drive a single type hint',()=>{
 const d=make([13,15,16,14,13]);const c=types.classify(d);
 assert.deepEqual(c.types,['cerebral','pushy']);assert.equal(c.type,null);
 const html=experience.summaryHTML(d);assert.match(html,/頭でっかち説明 ／ 圧強めセールス/);assert.match(html,/両方の傾向/);
});
test('missing speed can preserve supported types but does not count as low or allow expressive',()=>{
 const d=make([11.7,14.2,12.8,15,5.9]);d.metrics.speed={status:'unavailable',raw_value:null,reference_score:null,direction:'unknown'};
 assert.equal(types.classify(d).type,'unclear');assert.equal(types.experiment(d).lowCounts.speed,null);
 d.metrics.resonance.reference_score=12;assert.equal(types.classify(d).type,'cerebral');
 d.metrics.brightness.reference_score=16;d.metrics.power.reference_score=16;d.metrics.resonance.reference_score=16;
 assert.equal(types.classify(d).reason,'measurement_unavailable');
 d.metrics.brightness.status='unavailable';d.metrics.brightness.raw_value=null;d.metrics.brightness.reference_score=null;assert.equal(types.classify(d).type,null);
});
test('decision is unchanged by auxiliary data or history compaction and leaves points unchanged',()=>{
 const d=make([16,16,16,16,16]);d.type_auxiliary={status:'experimental',spectrum:{high_to_low_energy_db_median:-2},pitch:{range_semitones:5,trace:[{hz:150}]},audio:'private'};
 const before=JSON.stringify(d),report=types.explain(d);assert.equal(report.selected.type,'expressive');
 assert.equal(JSON.stringify(d),before);assert.doesNotMatch(JSON.stringify(report),/private|trace|hz/);
 assert.deepEqual(types.classify(globalThis.FiveMetricCore.compact(d)),types.classify(d));
});
test('outside conditions and failed measurement have different messages and no forced soft label',()=>{
 const d=make([14,14,14,14,14]);let html=experience.summaryHTML(d);
 assert.match(html,/タイプを絞れませんでした/);assert.match(html,/目安です/);assert.doesNotMatch(html,/ふわっと優しすぎタイプです/);
 d.metrics.brightness={status:'unavailable',raw_value:null,reference_score:null,direction:'unknown'};
 html=experience.summaryHTML(d);assert.match(html,/十分に測定できませんでした/);
});
test('observed teacher examples follow the numeric criteria including adjacent type differences',()=>{
 for(const v of [[16.1,15.3,14.9,18.2,16.6],[17.2,16,14.5,18.4,19]])assert.equal(types.classify(make(v)).type,'expressive');
 for(const v of [[15.7,15.1,15.4,15.5,15.1],[15.1,14.5,16.5,17.4,15.2],[16.7,15.6,15.3,17.3,14.2]])assert.equal(types.classify(make(v)).type,null);
 assert.equal(types.classify(make([11.5,14.3,12.1,17.3,12.7])).type,'cerebral');
 assert.equal(types.classify(make([16.5,15.2,16.6,9,14])).type,'pushy');
 assert.equal(types.classify(make([15.7,14.6,15.8,8.6,14.3])).type,null);
});
