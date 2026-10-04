'use strict';
// camreal/calibration/web/app.js in a node vm with just enough DOM for its logic (run by test_calibration_page.py).
// node calibration_page.js <scenario>: exits non-zero on a failed check.
const assert=require('assert'),fs=require('fs'),path=require('path'),vm=require('vm');
const APP=fs.readFileSync(path.join(__dirname,'..','calibration','web','app.js'),'utf8');
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const MARKERS=[...'ABCD'].flatMap((row,i)=>[1,2,3].map(c=>({id:row+c,x:[.6,1,1.5,2][i],y:[.4,0,-.4][c-1]})));
const PIXEL=Object.fromEntries(MARKERS.map((m,i)=>[m.id,[200+100*(i%3),300+150*Math.floor(i/3)]]));   // >GRAB apart
const KEY='camreal-calibrate:sha';
const TOKEN_ERROR='토큰이 맞지 않습니다. 페이지를 새로 고치세요 (calibrate를 다시 실행하면 토큰이 바뀝니다).';

class El{
  constructor(tag){Object.assign(this,{tag,className:'',hidden:false,disabled:false,src:'',width:0,height:0,kids:[],text:'',on:{}});
    const c=new Set();this.classList={add:k=>c.add(k),remove:k=>c.delete(k),contains:k=>c.has(k)}}
  get textContent(){return this.text+this.kids.map(k=>typeof k==='string'?k:k.textContent).join('')}
  set textContent(v){this.text=String(v);this.kids=[]}
  append(...kids){this.kids.push(...kids)}
  addEventListener(type,f){(this.on[type]=this.on[type]||[]).push(f)}
  fire(type,props){const e=Object.assign({type,defaultPrevented:false,preventDefault(){this.defaultPrevented=true}},props);
    for(const f of this.on[type]||[])f(e);return e}
  getBoundingClientRect(){return {left:0,top:0,width:this.width,height:this.height}}   // 1 screen px = 1 image px
  setPointerCapture(){}
  getContext(){return new Proxy({},{get:(o,k)=>k in o?o[k]:()=>{},set:(o,k,v)=>(o[k]=v,true)})}
}

// What /api/fit answers for these clicks (the real server's numbers do not matter to the page logic).
function fitted(points,extra){const ids=Object.keys(points),n=ids.length;
  return Object.assign({n,H_i2g:n>=4?[[1,0,0],[0,1,0],[0,0,1]]:null,errors_cm:n>=5?Object.fromEntries(ids.map(k=>[k,.5])):{},
    rms_cm:n>=5?.5:null,pose:n>=4?{x_m:.1,y_m:0,height_m:.2,pitch_deg:10,yaw_deg:0,roll_deg:0}:null,suspect:null,savable:n>=6,
    warned:[],uncovered:[],reason:n>=6?`저장 가능: 마커 ${n}개, RMS 0.5 cm.`:'마커를 더 찍으세요.',bev:n>=4?'data:image/png;base64,AA==':null},extra)}

async function page(store){
  const els={},storage=new Map(Object.entries(store||{})),document=new El('document'),window=new El('window');
  const session={width:1480,height:1080,markers:MARKERS,ost:'/home/car/camera_calibration/ost.yaml',ost_kind:'1주차 학생 파일',frame:'data/bags/calib',
    out:'data/calibration/car.yaml',image_sha256:'sha',hfov_deg:86.7,bev:{x_range_m:[.2,4],y_range_m:[-1.5,1.5],resolution_m:.01},
    warn_cm:5,reject_cm:20,pass_cm:3,min_fit:4,min_loo:5,min_save:6,ahead_m:1,token:'t1'};
  const p={els,storage,document,window,fit:points=>({status:200,body:fitted(points)}),saves:0};
  document.getElementById=id=>els[id]||(els[id]=new El(id));
  document.createElement=tag=>new El(tag);
  document.body=new El('body');
  const context=vm.createContext({document,window,setTimeout,clearTimeout,requestAnimationFrame:f=>setTimeout(f,0),
    localStorage:{getItem:k=>storage.has(k)?storage.get(k):null,setItem:(k,v)=>void storage.set(k,String(v))},
    Image:class{set src(v){this.complete=true;this.naturalWidth=session.width;setTimeout(()=>this.onload&&this.onload())}},
    fetch:async(url,options)=>{const body=options.body&&JSON.parse(options.body);
      const r=url==='/api/session'?{status:200,body:session}:url==='/api/fit'?p.fit(body.points):(p.saves++,{status:200,body:{saved:session.out,backup:null,rms_cm:.5}});
      return {ok:r.status<400,status:r.status,json:async()=>r.body}}});
  p.ev=expr=>vm.runInContext(expr,context);
  p.idle=async()=>{for(let i=0;i<300;i++){if(p.ev('!!session&&fitted===version&&!document.body.classList.contains("pending")'))return;await sleep(10)}
    throw Error('the fit never settled')};
  p.click=async(id,[du,dv]=[0,0])=>{const [u,v]=PIXEL[id],c=els.canvas;
    c.fire('pointerdown',{button:0,clientX:u+du+.5,clientY:v+dv+.5,pointerId:1});c.fire('pointerup',{});await p.idle()};
  p.key=key=>document.fire('keydown',{key,code:key==='s'?'KeyS':key});
  p.row=id=>els.list.kids[MARKERS.findIndex(m=>m.id===id)].onclick();
  p.points=()=>JSON.parse(p.ev('JSON.stringify(points)'));
  p.status=()=>({text:els.status.textContent,kind:els.status.className});
  vm.runInContext(APP,context,{filename:'app.js'});
  await p.idle();
  return p;
}
const clickAll=async p=>{for(const m of MARKERS)await p.click(m.id)};

const scenarios={
  async arrows_scroll_once_every_marker_is_placed(){
    const p=await page();
    await clickAll(p);
    assert.strictEqual(p.ev('selected'),null,'the last placed marker stays selected');
    assert.match(p.els.hint.textContent,/모두/);
    p.els.canvas.fire('pointermove',{clientX:5,clientY:5});
    assert(!p.key('ArrowDown').defaultPrevented,'ArrowDown did not scroll');
    assert.deepStrictEqual(p.points().D3,PIXEL.D3);
    // A point picked from the list moves only while the mouse is over the image.
    p.row('D3');p.els.canvas.fire('pointerleave',{});
    assert(!p.key('ArrowDown').defaultPrevented,'ArrowDown over the list moved D3');
    assert.deepStrictEqual(p.points().D3,PIXEL.D3);
    p.els.canvas.fire('pointermove',{clientX:5,clientY:5});
    assert(p.key('ArrowDown').defaultPrevented);
    await p.idle();
    assert.deepStrictEqual(p.points().D3,[PIXEL.D3[0],PIXEL.D3[1]+1]);
  },
  async restored_clicks_leave_nothing_selected(){
    const points=Object.fromEntries(MARKERS.slice(0,11).map(m=>[m.id,PIXEL[m.id]]));
    const p=await page({[KEY]:JSON.stringify({token:'t0',points,skipped:['D3']})});
    assert.strictEqual(Object.keys(p.points()).length,11);
    assert.strictEqual(p.ev('selected'),null);
    assert.strictEqual(JSON.parse(p.storage.get(KEY)).token,'t1','the restored record was not claimed by this run');
  },
  async skip_hint_for_markers_outside_the_image(){
    const p=await page();
    assert.match(p.els.hint.textContent,/^A1 .*클릭.*안 보이면 S/);
  },
  async a_tab_of_a_stopped_run_stops_writing(){
    const p=await page();
    await p.click('A1');
    p.fit=()=>({status:403,body:{error:TOKEN_ERROR}});   // calibrate was restarted: this tab's token is gone
    await p.click('A2');
    const kept=p.storage.get(KEY);
    assert.deepStrictEqual(p.status(),{text:TOKEN_ERROR,kind:'error'});
    await p.click('A3');p.row('A1');p.key('ArrowDown');p.key('s');p.key('Delete');await p.idle();
    assert.strictEqual(p.storage.get(KEY),kept,'a dead tab still writes localStorage');
    assert(!('A3' in p.points()) && p.els.save.disabled && p.status().text===TOKEN_ERROR);
    assert(!p.window.fire('beforeunload',{}).defaultPrevented,'closing the dead tab asks to stay');
  },
  async a_newer_run_claims_the_record(){
    const p=await page();
    await p.click('A1');
    // calibrate restarted on the same port and its page opened in another tab, which took over the record.
    const newer=JSON.stringify({token:'t2',points:{A1:PIXEL.A1},skipped:[]});
    p.storage.set(KEY,newer);
    await p.click('A2');
    assert.strictEqual(p.storage.get(KEY),newer,'the old tab overwrote the newer run\'s clicks');
    assert(!('A2' in p.points()) && p.els.save.disabled && p.status().kind==='error' && /다른 탭/.test(p.status().text));
  },
  async another_tab_of_the_same_run_takes_over(){
    const p=await page();
    await clickAll(p);
    assert(!p.els.save.disabled);
    const other=JSON.stringify({token:'t1',points:{A1:[210,300]},skipped:[]});
    p.storage.set(KEY,other);p.window.fire('storage',{key:KEY,newValue:other});
    assert(p.els.save.disabled && /다른 탭/.test(p.status().text));
    p.row('A1');p.els.canvas.fire('pointermove',{clientX:5,clientY:5});p.key('ArrowLeft');await p.idle();
    assert.strictEqual(p.storage.get(KEY),other);
    assert.deepStrictEqual(p.points().A1,PIXEL.A1);
    p.window.fire('storage',{key:'other-key',newValue:'{}'});   // another page's key changes nothing
  },
  async a_tab_does_not_save_over_another_tabs_clicks(){
    const p=await page();
    await clickAll(p);
    p.storage.set(KEY,JSON.stringify({token:'t1',points:{A1:[210,300]},skipped:[]}));   // its storage event not delivered yet
    await p.els.save.onclick();
    assert(p.saves===0 && p.els.save.disabled && /다른 탭/.test(p.status().text));
  },
  async a_failed_fit_clears_the_result_pane(){
    const p=await page();
    for(const m of MARKERS.slice(0,6))await p.click(m.id);
    assert(!p.els.bev.hidden && /RMS 0.50 cm/.test(p.els.rms.textContent) && /0.200 m/.test(p.els.pose.textContent));
    p.fit=()=>({status:400,body:{error:'markers.yaml이(가) calibrate를 시작한 뒤에 바뀌어 이 화면에는 반영되지 않았습니다.'}});
    await p.click('C1');
    assert(p.els.bev.hidden && p.els.rms.textContent==='' && p.els.pose.textContent==='','the old BEV, RMS or pose stayed');
    assert(p.els.bevNote.textContent!=='' && p.status().kind==='error' && p.els.save.disabled);
  },
  async a_savable_result_with_warnings_is_not_green(){
    const p=await page();
    for(const m of MARKERS.slice(0,6))await p.click(m.id);
    assert.strictEqual(p.status().kind,'ok');
    p.fit=points=>({status:200,body:fitted(points,{warned:['A3','B3'],reason:'저장은 되지만 A3 12.0 cm·B3 5.6 cm가 5 cm를 넘습니다.'})});
    await p.click('C1');
    assert(p.status().kind==='problem' && !p.els.save.disabled);
    p.fit=points=>({status:200,body:fitted(points,{uncovered:['near-left'],reason:'저장은 되지만 왼쪽에 찍은 마커가 없습니다.'})});
    await p.click('C2');
    assert(p.status().kind==='problem' && !p.els.save.disabled);
    assert.match(p.els.legend.textContent,/점검 기준.*3 cm/);
  },
};

const name=process.argv[2];
if(!scenarios[name]){console.error(`scenarios: ${Object.keys(scenarios).join(' ')}`);process.exit(2)}
scenarios[name]().then(()=>process.exit(0),e=>{console.error(e&&e.stack||e);process.exit(1)});
