const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const source=fs.readFileSync('static/preview-navigation.js','utf8');
function navigate(search,hrefs){
 const links=hrefs.map(href=>({href,getAttribute(){return this.href},setAttribute(k,v){this.href=v}}));
 vm.runInNewContext(source,{location:{search},URLSearchParams,document:{querySelectorAll:()=>links}});
 return links.map(l=>l.href);
}
test('preview mode survives diagnosis to help to microphone check and back',()=>{
 const [help]=navigate('?preview=five',['/help/microphone']);
 const [mic]=navigate(new URL(help,'https://example.test').search,['/mic-test']);
 assert.deepEqual(navigate(new URL(mic,'https://example.test').search,['/','/help/microphone']),['/?preview=five','/help/microphone?preview=five']);
});
test('legacy mode is unchanged and unknown parameters are not forwarded',()=>{
 const paths=['/','/help/microphone','/mic-test'];
 for(const search of ['', '?preview=other','?return=https://outside.test'])assert.deepEqual(navigate(search,paths),paths);
 assert.deepEqual(navigate('?preview=five&private=secret',paths),paths.map(p=>p+'?preview=five'));
 const external=['https://lin.ee/0uZqUu4','#safari','//outside.test/','/?preview=five'];
 assert.deepEqual(navigate('?preview=five',external),external);
});
