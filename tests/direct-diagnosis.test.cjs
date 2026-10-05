const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');

test('fixed-reading recording proceeds immediately without a confirmation screen',async()=>{
 const source=fs.readFileSync('static/diagnosis-preview.js','utf8');
 const context=vm.createContext({document:new Proxy({}, {get(){throw Error('confirmation UI must not be accessed');}})});
 vm.runInContext(source.replace('let fixedMode=false','let fixedMode=true'),context);
 assert.equal(await context.FiveMetricUI.prepareReading(),true);
 const html=fs.readFileSync('static/index.html','utf8');
 assert.doesNotMatch(html,/id="readingReview"|id="readingConfirmed"|最後まで読めた・診断する/);
});
