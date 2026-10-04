'use strict';
// Floor marker crosses clicked on the undistorted image. The server fits, judges and saves; it trusts only the clicks.
// Points are OpenCV pixel coordinates (integer = pixel centre): image point (u, v) sits at canvas (u+0.5, v+0.5).
const $=id=>document.getElementById(id), canvas=$('canvas'), ctx=canvas.getContext('2d'), image=new Image();
const ZOOM=4, LENS=180, GRAB=10, STATUS={todo:'미작업',placed:'찍음',skipped:'건너뜀',suspect:'다시 찍기'}, STORE='camreal-calibrate:';
const COLOR={none:'#53e5cf',na:'#8894a8',good:'#5fd68f',warn:'#ffc76b',bad:'#ff7a85',selected:'#ffb454'};
const POSE=[['카메라 높이','height_m',3,' m'],['pitch (+아래)','pitch_deg',1,'°'],['yaw (+왼쪽)','yaw_deg',1,'°'],
  ['roll (+오른쪽 아래)','roll_deg',1,'°'],['x (후륜축 앞)','x_m',3,' m'],['y (왼쪽 +)','y_m',3,' m']];
const NUDGE={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]};
const points=Object.create(null), skipped=new Set();   // no prototype: ids such as "constructor" are plain keys
let session, selected=null, result=null, drag=null, lens=null, version=0, fitted=-1, timer=0, frame=0, busy=false, savedKey='';
const scale=()=>canvas.width/canvas.getBoundingClientRect().width;   // canvas (= image) px per screen px
const own=(o,k)=>Object.prototype.hasOwnProperty.call(o,k);
const key=()=>JSON.stringify(session.markers.map(m=>points[m.id]||null));
const state=id=>points[id]?'placed':skipped.has(id)?'skipped':'todo';
const errorOf=id=>result&&points[id]&&own(result.errors_cm,id)?result.errors_cm[id]:undefined;
const grade=e=>e===undefined?'none':e===null?'na':e<=session.warn_cm?'good':e<=session.reject_cm?'warn':'bad';
const colorOf=id=>id===selected?COLOR.selected:COLOR[grade(errorOf(id))];
const clamp=([u,v])=>[Math.min(Math.max(u,0),session.width-1),Math.min(Math.max(v,0),session.height-1)];
const status=(text,kind)=>{$('status').textContent=text;$('status').className=kind||''};
const fixed=(x,digits)=>(+x.toFixed(digits)||0).toFixed(digits);   // never "-0.0"
async function api(path,body){const options=body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-Calibration-Token':session.token},body:JSON.stringify(body)};
  const r=await fetch(path,options).catch(()=>{throw Error('서버에 연결할 수 없습니다. python3 -m camreal calibrate가 실행 중인지 확인하세요.')}),value=await r.json();if(!r.ok)throw Error(value.error||r.status);return value}
function eventPixel(e){const r=canvas.getBoundingClientRect();return clamp([(e.clientX-r.left)*canvas.width/r.width-.5,(e.clientY-r.top)*canvas.height/r.height-.5])}

function redraw(){if(!frame)frame=requestAnimationFrame(()=>{frame=0;draw()})}
function cross(x,y,arm,gap,width,color){ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();
  for(const [dx,dy] of [[1,0],[-1,0],[0,1],[0,-1]]){ctx.moveTo(x+dx*gap,y+dy*gap);ctx.lineTo(x+dx*arm,y+dy*arm)}ctx.stroke()}
function draw(){
  if(!session)return;const s=scale();
  ctx.imageSmoothingEnabled=true;ctx.fillStyle='#242a35';ctx.fillRect(0,0,canvas.width,canvas.height);
  if(image.complete&&image.naturalWidth)ctx.drawImage(image,0,0);
  ctx.font=`bold ${15*s}px system-ui,sans-serif`;ctx.lineJoin='round';
  for(const m of session.markers){const p=points[m.id];if(!p)continue;
    const x=p[0]+.5,y=p[1]+.5,c=colorOf(m.id);cross(x,y,14*s,3*s,(m.id===selected?3:2)*s,c);
    ctx.lineWidth=4*s;ctx.strokeStyle='#121722';ctx.strokeText(m.id,x+9*s,y-9*s);ctx.fillStyle=c;ctx.fillText(m.id,x+9*s,y-9*s)}
  if(lens)magnifier(s);
}
function magnifier(s){
  // ZOOM screen px per image px, beside the cursor (flipped at the edges) so it never hides the spot being clicked.
  const size=LENS*s,half=LENS/ZOOM/2,f=size/(2*half),[u,v]=lens,sx=u+.5-half,sy=v+.5-half;
  let x=u+.5+24*s,y=v+.5-24*s-size;
  if(x+size>canvas.width)x=u+.5-24*s-size;
  if(y<0)y=v+.5+24*s;
  ctx.save();ctx.beginPath();ctx.rect(x,y,size,size);ctx.clip();
  ctx.fillStyle='#000';ctx.fillRect(x,y,size,size);ctx.imageSmoothingEnabled=false;
  if(image.complete&&image.naturalWidth)ctx.drawImage(image,sx,sy,2*half,2*half,x,y,size,size);
  cross(x+size/2,y+size/2,size/2,8*s,s,'#ffffffaa');
  for(const m of session.markers){const p=points[m.id];if(p)cross(x+(p[0]+.5-sx)*f,y+(p[1]+.5-sy)*f,22*s,5*s,2*s,colorOf(m.id))}
  ctx.restore();ctx.strokeStyle='#edf1f8';ctx.lineWidth=2*s;ctx.strokeRect(x,y,size,size);
}

function list(){
  const body=$('list');body.textContent='';
  for(const m of session.markers){const tr=document.createElement('tr'),e=errorOf(m.id),st=result&&result.suspect===m.id&&points[m.id]?'suspect':state(m.id);
    tr.className=m.id===selected?'selected':'';tr.onclick=()=>{selected=m.id;list();redraw()};
    for(const [text,cls] of [[m.id,''],[m.x.toFixed(3),'num'],[m.y.toFixed(3),'num'],[STATUS[st],st],   // mm: tape readings
      [e===undefined?'':e===null?'검증 불가':e.toFixed(1)+' cm','num '+grade(e)]]){const td=document.createElement('td');td.textContent=text;td.className=cls;tr.append(td)}
    body.append(tr)}
  const s=session.markers.find(m=>m.id===selected);
  $('hint').textContent=s?`${s.id} (x ${s.x.toFixed(3)} m, y ${s.y.toFixed(3)} m): `+(points[s.id]?'드래그나 방향키로 조정':'영상에서 십자 중심을 클릭'):'목록에서 마커를 고르세요';
  buttons();
}
function buttons(){$('save').disabled=busy||!result||!result.savable||fitted!==version||key()===savedKey;$('skip').disabled=$('clear').disabled=!selected}
function next(){const ids=session.markers.map(m=>m.id),i=ids.indexOf(selected);
  for(let j=1;j<=ids.length;j++){const id=ids[(i+j)%ids.length];if(state(id)==='todo'){selected=id;return}}}

// Clicks are kept per image (the PNG's sha256), so restarting calibrate on the same frame does not cost a re-click.
function remember(){try{localStorage.setItem(STORE+session.image_sha256,JSON.stringify({points,skipped:[...skipped]}))}catch(e){}}
function recall(){let saved=null;try{saved=JSON.parse(localStorage.getItem(STORE+session.image_sha256))}catch(e){}
  const p=saved&&typeof saved.points==='object'&&saved.points||{},skip=saved&&Array.isArray(saved.skipped)?saved.skipped:[];
  for(const m of session.markers){const q=own(p,m.id)?p[m.id]:null;
    if(Array.isArray(q)&&q.length===2&&q.every(Number.isFinite)&&q[0]>=0&&q[0]<session.width&&q[1]>=0&&q[1]<session.height)points[m.id]=q;
    else if(skip.includes(m.id))skipped.add(m.id)}
  return Object.keys(points).length}
function changed(){version++;remember();document.body.classList.add('pending');clearTimeout(timer);timer=setTimeout(fit,150);buttons()}
function move(id,p){points[id]=clamp(p).map(c=>Math.round(c*100)/100);skipped.delete(id);changed()}
function skip(){if(!selected)return;skipped.add(selected);if(points[selected]){delete points[selected];changed()}else remember();next();list();redraw()}
function clear(){if(!selected)return;skipped.delete(selected);if(points[selected]){delete points[selected];changed()}else remember();list();redraw()}
async function fit(){
  const v=version;
  try{const r=await api('/api/fit',{points});if(v!==version)return;result=r;fitted=v;show()}   // older than the clicks: a newer fit follows
  catch(e){if(v!==version)return;result=null;fitted=v;document.body.classList.remove('pending');status(e.message,'error');list();redraw()}
}
function show(){
  const r=result;document.body.classList.remove('pending');
  $('bev').hidden=!r.bev;if(r.bev)$('bev').src=r.bev;
  $('bevNote').textContent=r.bev?'':r.n<session.min_fit?`마커를 ${session.min_fit}개 이상 찍으면 보입니다.`:'BEV를 만들 수 없습니다. 아래 메시지를 보세요.';
  $('rms').textContent=`마커 ${r.n}개 · RMS ${r.rms_cm==null?'-':r.rms_cm.toFixed(2)+' cm'}`;
  $('pose').textContent='';
  for(const [label,text] of [...POSE.map(([l,name,digits,unit])=>[l,r.pose?fixed(r.pose[name],digits)+unit:'-']),
    ['수평 화각 (new_K)',session.hfov_deg.toFixed(1)+'°']]){const dt=document.createElement('span'),dd=document.createElement('b');dt.textContent=label;dd.textContent=text;$('pose').append(dt,dd)}
  status(r.reason,r.savable?'ok':'problem');list();redraw();
}
async function save(){
  if($('save').disabled)return;busy=true;buttons();const k=key();
  try{const r=await api('/api/save',{points});savedKey=k;
    status(`저장했습니다: ${r.saved}${r.backup?` (이전 파일은 ${r.backup})`:''} · RMS ${r.rms_cm.toFixed(2)} cm`,'ok')}
  catch(e){status(e.message,'error')}finally{busy=false;buttons()}
}

function nearest(p){let best=null,d=GRAB*scale();
  for(const m of session.markers){const q=points[m.id],e=q?Math.hypot(q[0]-p[0],q[1]-p[1]):Infinity;if(e<d){d=e;best=m.id}}return best}
canvas.addEventListener('pointerdown',e=>{
  if(!session||e.button!==0)return;const p=eventPixel(e),hit=nearest(p);
  if(hit){selected=hit;drag={id:hit,dx:points[hit][0]-p[0],dy:points[hit][1]-p[1],placed:false}}
  else if(selected){drag={id:selected,dx:0,dy:0,placed:!points[selected]};move(selected,p)}
  else return;
  canvas.setPointerCapture(e.pointerId);lens=points[drag.id];list();redraw()});
canvas.addEventListener('pointermove',e=>{if(!session)return;const p=eventPixel(e);
  if(drag){move(drag.id,[p[0]+drag.dx,p[1]+drag.dy]);lens=points[drag.id]}else lens=p;redraw()});
function release(){if(!drag)return;if(drag.placed)next();drag=null;list();redraw()}
canvas.addEventListener('pointerup',release);canvas.addEventListener('pointercancel',release);
canvas.addEventListener('pointerleave',()=>{if(!drag){lens=null;redraw()}});
document.addEventListener('keydown',e=>{
  if(!session||e.altKey||e.ctrlKey||e.metaKey)return;const d=NUDGE[e.key];
  if(d&&selected&&points[selected]){e.preventDefault();const p=points[selected];move(selected,[p[0]+d[0],p[1]+d[1]]);lens=points[selected];redraw()}
  else if(e.code==='KeyS'){skip()}else if(e.key==='Delete'||e.key==='Backspace'){e.preventDefault();clear()}});
window.addEventListener('beforeunload',e=>{if(session&&Object.keys(points).length&&key()!==savedKey){e.preventDefault();e.returnValue=''}});
window.addEventListener('resize',redraw);
$('skip').onclick=skip;$('clear').onclick=clear;$('save').onclick=save;
(async()=>{try{session=await api('/api/session');canvas.width=session.width;canvas.height=session.height;savedKey=key();
  const restored=recall(),b=session.bev;
  $('ost').textContent=`ost.yaml: ${session.ost} (${session.ost_kind})`;$('size').textContent=`${session.width}×${session.height}`;
  $('frame').textContent=`프레임: ${session.frame}`+(restored?` · 이전에 찍은 점 ${restored}개를 불러옴`:'');$('out').textContent=`저장: ${session.out}`;
  $('bevSpec').textContent=`x ${b.x_range_m[0]}~${b.x_range_m[1]} m · y ${b.y_range_m[0]}~${b.y_range_m[1]} m · ${b.resolution_m} m/px`;
  [['good',`${session.warn_cm} cm 이하`],['warn',`${session.warn_cm}~${session.reject_cm} cm`],['bad',`${session.reject_cm} cm 초과`]].forEach(([cls,text],i)=>{
    const span=document.createElement('span');span.className=cls;span.textContent=text;$('legend').append(i?' · ':' ',span)});
  selected=session.markers[0].id;if(state(selected)!=='todo')next();list();draw();
  image.onload=redraw;image.onerror=()=>status('왜곡 보정 영상을 불러오지 못했습니다.','error');image.src='/image/undistorted.png';
  await fit()}catch(e){status(e.message,'error')}})();
