const {test}=require('node:test');
const assert=require('node:assert/strict');
const {valid,compact,compare,createHistory}=require('../static/diagnosis-preview.js');
const make=()=>({schema_version:'five-metric-diagnosis-1',measurement_version:'measure-1',calibration_version:'draft-1',prompt_id:'reading-1',
    metrics:Object.fromEntries(['brightness','articulation','power','speed','resonance'].map(key=>[key,{raw_value:1,reference_score:6,direction:'low',status:'provisional'}]))});
const memory=()=>({value:null,getItem(){return this.value},setItem(k,v){this.value=v},removeItem(){this.value=null}});
test('version and prompt changes prevent comparing incompatible results',()=>{
    for(const key of ['measurement_version','calibration_version','prompt_id']){const a=make(),b=make();b[key]='other';assert.equal(compare(a,b),null)}
});
test('missing scores remain unavailable, never become five',()=>{
    const d=make();d.metrics.speed={raw_value:null,reference_score:null,direction:'unknown',status:'unavailable'};
    assert.equal(valid(d),true);assert.equal(compare(d,make()).speed,null);
    d.metrics.speed.reference_score=NaN;assert.equal(valid(d),false);
});
test('comparison preserves actual sign and decimal change',()=>{
    const a=make(),b=make();a.metrics.power.reference_score=6.5;a.metrics.resonance.reference_score=5.9;
    assert.equal(compare(a,b).power,0.5);assert.equal(compare(a,b).resonance,-0.1);
});
test('persistent history restores from a fresh store',()=>{
    const storage=memory();createHistory(storage).append(make());
    assert.equal(createHistory(storage).read().length,1);
});
test('only approved result fields are saved, never audio or identifying fields',()=>{
    const d=make();d.audio='blob';d.name='private';d.email='private';d.metrics.brightness.audio='blob';
    const stored=JSON.stringify(compact(d));assert.doesNotMatch(stored,/audio|private|email/);
});
test('corrupt or obsolete storage does not break recording',()=>{
    const storage=memory();storage.value='{bad';assert.deepEqual(createHistory(storage).read(),[]);
    storage.value=JSON.stringify([{at:Date.now(),diagnosis:{}}]);assert.deepEqual(createHistory(storage).read(),[]);
});
test('quota failure falls back to this page, retaining prior entries',()=>{
    const storage=memory(),store=createHistory(storage);store.append(make());
    storage.setItem=()=>{throw Error('quota')};store.append(make());
    assert.equal(store.persisted,false);assert.equal(store.read().length,2);
});
test('old results expire, history is capped, and delete clears it',()=>{
    let now=Date.now();const storage=memory(),store=createHistory(storage,()=>now);
    for(let i=0;i<25;i++)store.append(make());assert.equal(store.read().length,20);
    now+=31*86400000;assert.equal(store.read().length,0);
    store.append(make());store.clear();assert.equal(storage.value,null);assert.equal(store.read().length,0);
});

test('deletion failure is reported and cannot silently restore stale data in this page',()=>{
    const storage=memory(),store=createHistory(storage);store.append(make());
    storage.removeItem=()=>{throw Error('denied')};storage.setItem=()=>{throw Error('denied')};
    assert.equal(store.clear(),false);assert.deepEqual(store.read(),[]);assert.equal(store.persisted,false);assert.ok(storage.value);
});
test('deletion falls back to emptying its own key if removal alone fails',()=>{
    const storage=memory(),store=createHistory(storage);store.append(make());storage.removeItem=()=>{throw Error('denied')};
    assert.equal(store.clear(),true);assert.deepEqual(store.read(),[]);assert.equal(storage.value,'[]');
});
