const {test}=require('node:test');
const assert=require('node:assert/strict');
const core=require('../static/diagnosis-preview.js');
const engine=require('../static/hint-engine.js');
const catalog=require('../static/hints-draft.json');
const make=()=>({schema_version:'five-metric-diagnosis-1',measurement_version:'m1',calibration_version:'c1',prompt_id:'p1',metrics:Object.fromEntries(['brightness','articulation','power','speed','resonance'].map(k=>[k,{status:'provisional',direction:k==='speed'?'fast':'low',raw_value:1,reference_score:6}]))});
function row(id,change){const h=catalog.find(h=>h.hint_id===id),before=make(),diagnosis=make();diagnosis.metrics[h.target_metric].reference_score+=change;return {at:Date.now(),diagnosis,trial:{hint_id:id,hint_version:1,target_metric:h.target_metric,speed_direction:h.speed_direction,mechanism_type:h.mechanism_type,before}};}
test('catalog keeps unique IDs and required metadata including approved short wording',()=>{
 assert.equal(new Set(catalog.map(h=>h.hint_id)).size,catalog.length);
 for(const h of catalog){assert.ok(h.hint_text.length>0&&h.hint_text.length<=65);if(h.content_approval!=='approved')assert.ok(h.hint_text.length>=25);assert.ok(['preview','disabled'].includes(h.lifecycle_state));assert.equal(h.evidence_tier,'candidate_unvalidated');assert.ok(h.safety_note);}
});
test('user-rejected hints stay excluded through repeated rotation and restored history',()=>{
 const rejected=['articulation-01','power-02','speed_fast-01','speed_fast-02','speed_slow-05','speed_slow-02','speed_slow-03','resonance-02','resonance-04'];
 assert.equal(catalog.filter(h=>h.lifecycle_state==='preview').length,48);
 for(const id of rejected){const h=catalog.find(h=>h.hint_id===id);assert.equal(h.lifecycle_state,'disabled');assert.ok(h.disabled_reason);}
 for(const group of new Set(catalog.map(h=>h.group))){
  const d=make(),shown=[];if(group==='speed_slow')d.metrics.speed.direction='slow';
  for(let i=0;i<20;i++){const h=engine.next(catalog,d,rejected.map(id=>row(id,0)),group,shown);assert.ok(h);assert.ok(!rejected.includes(h.hint_id));shown.push(h.hint_id);}
 }
});
test('speed never proposes opposite direction or adjusts optimal speed',()=>{
 const d=make();assert.equal(engine.next(catalog,d,[],'speed_slow'),null);
 d.metrics.speed.direction='optimal';assert.equal(engine.next(catalog,d,[],'speed_fast'),null);
 d.metrics.speed.direction='slow';assert.ok(engine.next(catalog,d,[],'speed_slow'));
});

test('approved fast-speed hints rotate only for fast results and retain old trial labels',()=>{
 const d=make(),selected=catalog.filter(h=>h.group==='speed_fast'&&engine.eligible(d,h));
 assert.equal(selected.length,8);assert.ok(selected.every(h=>h.content_approval==='approved'&&h.speed_direction==='fast'));
 const old=catalog.find(h=>h.hint_id==='speed_fast-03'),history=core.createHistory(null);
 history.append(d,row(old.hint_id,0).trial);
 const restored=history.read()[0].trial;
 assert.equal(catalog.find(h=>h.hint_id===restored.hint_id&&h.hint_version===restored.hint_version).hint_text,old.hint_text);
 const shown=[];
 for(let i=0;i<24;i++){const h=engine.next(catalog,d,history.read(),'speed_fast',shown);assert.ok(selected.includes(h));shown.push(h.hint_id);}
 assert.equal(new Set(shown.slice(0,8)).size,8);
 for(const direction of ['slow','optimal','unknown']){d.metrics.speed.direction=direction;assert.equal(engine.next(catalog,d,[],'speed_fast'),null);}
 d.metrics.speed.direction='fast';d.metrics.speed.status='unavailable';assert.equal(engine.next(catalog,d,[],'speed_fast'),null);
});
test('unavailable and excessive input cannot receive increasing hints',()=>{
 const d=make();d.metrics.power.direction='high';assert.equal(engine.next(catalog,d,[],'power'),null);
 d.metrics.brightness.status='unavailable';assert.equal(engine.next(catalog,d,[],'brightness'),null);
});

test('approved slow-speed hints rotate only for slow results and retain old trial labels',()=>{
 const d=make();d.metrics.speed.direction='slow';
 const selected=catalog.filter(h=>h.group==='speed_slow'&&engine.eligible(d,h));
 assert.equal(selected.length,8);assert.ok(selected.every(h=>h.content_approval==='approved'&&h.speed_direction==='slow'));
 const old=catalog.find(h=>h.hint_id==='speed_slow-01'),history=core.createHistory(null),trial=row(old.hint_id,0).trial;
 trial.before.metrics.speed.direction='slow';history.append(d,trial);
 const restored=history.read()[0].trial;
 assert.equal(catalog.find(h=>h.hint_id===restored.hint_id&&h.hint_version===restored.hint_version).hint_text,old.hint_text);
 const shown=[];
 for(let i=0;i<24;i++){const h=engine.next(catalog,d,history.read(),'speed_slow',shown);assert.ok(selected.includes(h));shown.push(h.hint_id);}
 assert.equal(new Set(shown.slice(0,8)).size,8);
 for(const direction of ['fast','optimal','unknown']){d.metrics.speed.direction=direction;assert.equal(engine.next(catalog,d,[],'speed_slow'),null);}
 d.metrics.speed.direction='slow';d.metrics.speed.status='unavailable';assert.equal(engine.next(catalog,d,[],'speed_slow'),null);
});
test('skipped and completed hints rotate before reuse',()=>{
 const d=make(),shown=['brightness-07'],rows=[row('brightness-06',0)];
 for(let i=0;i<6;i++){const h=engine.next(catalog,d,rows,'brightness',shown);assert.ok(!shown.includes(h.hint_id));assert.notEqual(h.hint_id,'brightness-06');shown.push(h.hint_id);}
 assert.notEqual(engine.next(catalog,d,rows,'brightness',shown).hint_id,shown.at(-1));
});

test('eight approved brightness hints replace drafts without rewriting stored trial text',()=>{
 const selected=catalog.filter(h=>h.group==='brightness'&&engine.eligible(make(),h));
 assert.equal(selected.length,8);assert.ok(selected.every(h=>h.content_approval==='approved'));
 const old=catalog.find(h=>h.hint_id==='brightness-01');
 assert.equal(old.hint_text,'親しい人にほほえみかけるくらい、口角をほんの少し上げた表情で話してみましょう。');
 const history=core.createHistory(null);history.append(make(),row(old.hint_id,0).trial);
 const restored=history.read()[0].trial;
 assert.equal(catalog.find(h=>h.hint_id===restored.hint_id&&h.hint_version===restored.hint_version).hint_text,old.hint_text);
 const shown=[];
 for(let i=0;i<24;i++){const h=engine.next(catalog,make(),history.read(),'brightness',shown);assert.ok(selected.includes(h));shown.push(h.hint_id);}
 assert.equal(new Set(shown.slice(0,8)).size,8);
});
test('a different mechanism is preferred while fresh alternatives remain',()=>{
 assert.notEqual(engine.next(catalog,make(),[],'resonance',['resonance-06']).mechanism_type,'preparation');
});

test('all six approved pools retain historical labels and exclude every legacy draft',()=>{
 const groups=['brightness','articulation','power','speed_fast','speed_slow','resonance'];
 for(const group of groups){
  const d=make();if(group==='speed_slow')d.metrics.speed.direction='slow';
  const selected=catalog.filter(h=>h.group===group&&engine.eligible(d,h));
  assert.equal(selected.length,8);assert.ok(selected.every(h=>h.content_approval==='approved'));
  const history=core.createHistory(null);
  for(let i=1;i<=5;i++){
   const id=`${group}-${String(i).padStart(2,'0')}`,old=catalog.find(h=>h.hint_id===id);
   assert.equal(old.lifecycle_state,'disabled');history.append(d,row(id,0).trial);
   const restored=history.read().at(-1).trial;
   assert.equal(catalog.find(h=>h.hint_id===restored.hint_id&&h.hint_version===restored.hint_version).hint_text,old.hint_text);
  }
  const shown=[];
  for(let i=0;i<24;i++){const h=engine.next(catalog,d,history.read(),group,shown);assert.ok(selected.includes(h));shown.push(h.hint_id);}
  assert.equal(new Set(shown.slice(0,8)).size,8);
 }
});

test('approved power pool preserves the full stance instruction and prior history',()=>{
 const selected=catalog.filter(h=>h.group==='power'&&engine.eligible(make(),h));
 assert.equal(selected.length,8);assert.ok(selected.every(h=>h.content_approval==='approved'));
 assert.equal(catalog.find(h=>h.hint_id==='power-12').hint_text,'両足を肩幅に開いて立ち、足の裏で床を踏みしめた姿勢で話しましょう。');
 const old=catalog.find(h=>h.hint_id==='power-01'),history=core.createHistory(null);
 history.append(make(),row(old.hint_id,0).trial);
 const restored=history.read()[0].trial;
 assert.equal(catalog.find(h=>h.hint_id===restored.hint_id&&h.hint_version===restored.hint_version).hint_text,old.hint_text);
 const shown=[];
 for(let i=0;i<24;i++){const h=engine.next(catalog,make(),history.read(),'power',shown);assert.ok(selected.includes(h));shown.push(h.hint_id);}
 assert.equal(new Set(shown.slice(0,8)).size,8);
});

test('approved articulation pool rotates all eight and preserves prior trials',()=>{
 const selected=catalog.filter(h=>h.group==='articulation'&&engine.eligible(make(),h));
 assert.equal(selected.length,8);assert.ok(selected.every(h=>h.content_approval==='approved'));
 assert.equal(catalog.find(h=>h.hint_id==='articulation-12').hint_text,'録音前に「ウイウイウイスキー」と3回言ってから話しましょう。');
 const old=catalog.find(h=>h.hint_id==='articulation-02'),history=core.createHistory(null);
 history.append(make(),row(old.hint_id,0).trial);
 const restored=history.read()[0].trial;
 assert.equal(catalog.find(h=>h.hint_id===restored.hint_id&&h.hint_version===restored.hint_version).hint_text,old.hint_text);
 const shown=[];
 for(let i=0;i<24;i++){const h=engine.next(catalog,make(),history.read(),'articulation',shown);assert.ok(selected.includes(h));shown.push(h.hint_id);}
 assert.equal(new Set(shown.slice(0,8)).size,8);
});
test('two small or negative changes switch metric; one does not',()=>{
 const a=row('brightness-01',0),b=row('brightness-02',-0.5);
 assert.equal(engine.shouldSwitch([a],catalog,make()),false);
 assert.equal(engine.shouldSwitch([a,b],catalog,make()),true);
});
test('positive reference change suggests another metric without claiming improvement',()=>{
 assert.equal(engine.shouldSwitch([row('brightness-01',0.5)],catalog,make()),true);
});
test('incompatible versions and missing target scores do not count as failures',()=>{
 const a=row('brightness-01',0);a.diagnosis.calibration_version='other';assert.equal(engine.shouldSwitch([a,a],catalog,make()),false);
 const b=row('brightness-02',0);b.diagnosis.metrics.brightness.reference_score=null;b.diagnosis.metrics.brightness.status='unavailable';assert.equal(engine.shouldSwitch([b,b],catalog,make()),false);
});
test('trial linkage restores and strips arbitrary personal or audio fields',()=>{
 const storage={value:null,getItem(){return this.value},setItem(k,v){this.value=v},removeItem(){this.value=null}};
 const a=row('power-01',0.5);a.trial.audio='private';a.trial.before.email='private';core.createHistory(storage).append(a.diagnosis,a.trial);
 const restored=core.createHistory(storage).read()[0];assert.equal(engine.delta(restored),0.5);assert.doesNotMatch(storage.value,/private|audio|email/);
});
test('ordinary recordings and malformed trials cannot become hint attempts',()=>{
 const store=core.createHistory(null);store.append(make());store.append(make(),{hint_id:'invalid'});
 assert.ok(store.read().every(r=>!r.trial));
});
test('switch target comes from compatible trials, not a newer incompatible record',()=>{
 const a=row('brightness-01',0),b=row('brightness-02',0),c=row('power-01',0.5);c.diagnosis.calibration_version='other';
 assert.equal(engine.switchMetric([a,b,c],catalog,make()),'brightness');
});
