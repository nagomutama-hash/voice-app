const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const source=fs.readFileSync('static/test-entry.js','utf8');
test('new entry keeps API and static requests inside its own namespace',()=>{
 for(const pathname of ['/test-202610','/test-202610/','/test-202610/mic-test']){
  const c={location:{pathname}};vm.runInNewContext(source,c);
  assert.equal(c.VOICE_TEST_ENTRY,true);
  for(const p of ['/analyze','/api/diagnosis-config','/static/hints-draft.json','/'])
   assert.equal(c.voiceAppPath(p),'/test-202610'+p);
  for(const p of ['/test-202610','/test-202610/analyze','//elsewhere.test','https://lin.ee/0uZqUu4'])
   assert.equal(c.voiceAppPath(p),p);
 }
});
test('local preview and unrelated paths keep their original requests',()=>{
 for(const pathname of ['/','/help/microphone','/test-202610-other']){
  const c={location:{pathname}};vm.runInNewContext(source,c);
  assert.equal(c.VOICE_TEST_ENTRY,false);assert.equal(c.voiceAppPath('/analyze'),'/analyze');
 }
});
test('help navigation returns to new entry even without preview query',()=>{
 const paths=['/','/help/microphone','/mic-test','/test-202610/','https://lin.ee/0uZqUu4'];
 const links=paths.map(href=>({href,getAttribute(){return this.href},setAttribute(_,v){this.href=v}}));
 vm.runInNewContext(fs.readFileSync('static/preview-navigation.js','utf8'),{
  location:{pathname:'/test-202610/mic-test',search:''},URLSearchParams,
  document:{querySelectorAll:()=>links}
 });
 assert.deepEqual(links.map(l=>l.href),['/test-202610/?preview=five','/test-202610/help/microphone?preview=five',
  '/test-202610/mic-test?preview=five','/test-202610/?preview=five','https://lin.ee/0uZqUu4']);
});
