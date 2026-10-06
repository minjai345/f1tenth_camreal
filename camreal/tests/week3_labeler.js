'use strict';
// camreal/week3_labeler.js in a node vm with just enough DOM for its logic (run by test_week3_lab.py).
// node week3_labeler.js <scenario>: exits non-zero on a failed check.
const assert=require('assert'),fs=require('fs'),path=require('path'),vm=require('vm');
const JS=fs.readFileSync(path.join(__dirname,'..','week3_labeler.js'),'utf8');
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const BEV={x_range_m:[.2,4],y_range_m:[-1.5,1.5],resolution_m:.01}, VMAX=2.5, SCALE=2;

class El{
  constructor(tag){Object.assign(this,{tag,width:0,height:0,text:'',on:{}})}
  get textContent(){return this.text}
  set textContent(v){this.text=String(v)}
  addEventListener(type,f){(this.on[type]=this.on[type]||[]).push(f)}
  fire(type,props){const e=Object.assign({type,preventDefault(){}},props);for(const f of this.on[type]||[])f(e)}
  getBoundingClientRect(){return {left:0,top:0,width:this.width,height:this.height}}   // 1 screen px = 1 canvas px
  setPointerCapture(){}
  getContext(){return new Proxy({},{get:(o,k)=>k in o?o[k]:()=>{},set:(o,k,v)=>(o[k]=v,true)})}
}

// What camreal.week3_lab.Labeler answers; the real one also snaps again and checks the range.
async function page(n,{pre={},fail=false}={}){
  const els={},document=new El('document');
  document.getElementById=id=>els[id]||(els[id]=new El(id));
  const labels=Array.from({length:n},(_,i)=>pre[i]||{status:'unlabeled',waypoint_m:null}),calls=[];
  const counts=()=>Object.fromEntries(['accepted','rejected','unlabeled'].map(s=>[s,labels.filter(l=>l.status===s).length]));
  const reply=i=>({label:labels[i],counts:counts(),statuses:labels.map(l=>l.status)});
  const server={frame:i=>Object.assign({index:i,n,image:'data:image/png;base64,'+i},reply(i)),
    save:(i,status,x,y)=>{if(fail)throw Error('저장 실패: 테스트');labels[i]={status,waypoint_m:status==='accepted'?[x,y]:null};return reply(i)}};
  const call=async(name,args)=>{calls.push([name,...args]);return server[name](...args)};
  const CONFIG={n,ahead_m:1,bev:BEV,scale:SCALE,prefix:'t.'};
  const context=vm.createContext({document,CONFIG,call,setTimeout,
    Image:class{set src(v){this._src=v;this.complete=true;this.naturalWidth=300;setTimeout(()=>this.onload&&this.onload())}get src(){return this._src}}});
  vm.runInContext(JS,context,{filename:'week3_labeler.js'});
  const p={els,calls,labels,text:id=>els[id].textContent,settle:()=>sleep(20)};
  p.click=async(x,y)=>{const c=els.cv,u=(BEV.y_range_m[1]-y)/BEV.resolution_m*SCALE,v=(VMAX-x)/BEV.resolution_m*SCALE;
    c.fire('pointerdown',{clientX:u,clientY:v,pointerId:1});c.fire('pointerup',{});await p.settle()};
  p.key=async key=>{document.fire('keydown',{key});await p.settle()};
  p.saves=()=>calls.filter(c=>c[0]==='save');
  await p.settle();
  return p;
}

const scenarios={
  async loadsFirstFrame(){
    const p=await page(3);
    assert.deepStrictEqual(p.calls,[['frame',0]]);
    assert.strictEqual(p.text('pos'),'1 / 3');
    assert.strictEqual(p.text('counts'),'승인 0 · 제외 0 · 미작업 3');
    assert.strictEqual(p.text('state'),'미작업');
    assert.match(p.text('msg'),/1 m 원이 좌우 테이프의 가운데와 만나는 곳을 클릭하세요/);
    assert.strictEqual(p.els.cv.width,600);assert.strictEqual(p.els.cv.height,560);   // 3 m wide, -0.3..2.5 m deep, 2x
  },
  async clickSnapsToTheCircle(){
    const p=await page(3);
    await p.click(1.5,0);
    assert.match(p.text('msg'),/앞 1\.00 m, 왼쪽 0\.00 m/);
    await p.click(.6,.8);
    assert.match(p.text('msg'),/앞 0\.60 m, 왼쪽 0\.80 m/);
    assert.strictEqual(p.saves().length,0);
  },
  async enterSavesAndMovesOn(){
    const p=await page(3);
    await p.click(1.2,.9);
    await p.key('Enter');
    const [, i,status,x,y]=p.saves()[0];
    assert.strictEqual(i,0);assert.strictEqual(status,'accepted');
    assert.ok(Math.abs(Math.hypot(x,y)-1)<1e-9&&Math.abs(y/x-.75)<1e-9);
    assert.strictEqual(p.text('pos'),'2 / 3');
    assert.strictEqual(p.text('counts'),'승인 1 · 제외 0 · 미작업 2');
    await p.click(1,0);await p.key('Enter');
    await p.click(1,0);await p.key('Enter');
    assert.strictEqual(p.text('pos'),'3 / 3');
    assert.match(p.text('msg'),/다 했습니다/);
  },
  async enterWithoutPointWarns(){
    const p=await page(3);
    await p.key('Enter');
    assert.strictEqual(p.saves().length,0);
    assert.match(p.text('msg'),/먼저 1 m 원 위를 클릭하세요/);
  },
  async xRejectsAndMovesOn(){
    const p=await page(3);
    await p.key('x');
    assert.deepStrictEqual(p.saves()[0],['save',0,'rejected',null,null]);
    assert.strictEqual(p.text('pos'),'2 / 3');
    assert.strictEqual(p.text('counts'),'승인 0 · 제외 1 · 미작업 2');
  },
  async clickOutsideTheBevWarns(){
    const p=await page(3);
    await p.click(-.2,0);                       // behind the rear axle: the circle point there is outside the BEV
    assert.match(p.text('msg'),/BEV 범위 밖입니다/);
    await p.key('Enter');
    assert.strictEqual(p.saves().length,0);
  },
  async arrowsMoveBetweenFrames(){
    const p=await page(3);
    await p.key('ArrowLeft');
    assert.strictEqual(p.text('pos'),'1 / 3');assert.strictEqual(p.calls.length,1);
    await p.key('ArrowRight');assert.strictEqual(p.text('pos'),'2 / 3');
    await p.key('ArrowRight');await p.key('ArrowRight');assert.strictEqual(p.text('pos'),'3 / 3');
    await p.key('ArrowLeft');assert.strictEqual(p.text('pos'),'2 / 3');
  },
  async todoJumpsToTheNextUnlabeled(){
    const p=await page(3,{pre:{1:{status:'accepted',waypoint_m:[1,0]}}});
    p.els.todo.onclick();await p.settle();
    assert.strictEqual(p.text('pos'),'3 / 3');
    await p.key('x');
    p.els.todo.onclick();await p.settle();
    assert.strictEqual(p.text('pos'),'1 / 3');
    await p.key('x');
    p.els.todo.onclick();await p.settle();
    assert.match(p.text('msg'),/미작업 프레임이 없습니다/);
  },
  async serverErrorIsShown(){
    const p=await page(3,{fail:true});
    await p.click(1,0);await p.key('Enter');
    assert.match(p.text('msg'),/저장 실패: 테스트/);
    assert.strictEqual(p.text('pos'),'1 / 3');
  },
};

const name=process.argv[2];
if(!scenarios[name]){console.error('unknown scenario '+name);process.exit(2)}
scenarios[name]().then(()=>process.exit(0),e=>{console.error(e);process.exit(1)});
