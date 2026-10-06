'use strict';
// Click labeling inside a Colab output cell (inlined by camreal/week3_lab.py, which defines CONFIG and call()).
// call('frame',[i]) -> {index,n,image,label:{status,waypoint_m},counts,statuses}
// call('save',[i,status,x,y]) -> {label,counts,statuses}; the server snaps the point onto the ahead_m circle again.
(()=>{
const $=id=>document.getElementById(id), canvas=$('cv'), ctx=canvas.getContext('2d');
const STATUS={unlabeled:'미작업',accepted:'승인',rejected:'제외'};
const B=CONFIG.bev, R=B.resolution_m, S=CONFIG.scale, AHEAD=CONFIG.ahead_m;
const VMAX=Math.min(B.x_range_m[1],Math.max(2,2.5*AHEAD)), VMIN=-.3;   // near the car, down to just behind the rear axle
canvas.width=Math.round((B.y_range_m[1]-B.y_range_m[0])/R*S);canvas.height=Math.round((VMAX-VMIN)/R*S);
const toPixel=([x,y])=>[(B.y_range_m[1]-y)/R*S,(VMAX-x)/R*S];
const toMetre=([u,v])=>[VMAX-v*R/S,B.y_range_m[1]-u*R/S];
const image=new Image();
let index=0, label=null, point=null, statuses=[], busy=false;
const msg=text=>{$('msg').textContent=text};

function onRing(p){   // same snapping as camreal ring_point: keep the direction from the rear axle, fix the distance
  const r=Math.hypot(p[0],p[1]);if(r<1e-6)return null;
  const q=[p[0]*AHEAD/r,p[1]*AHEAD/r];
  return q[0]>=B.x_range_m[0]&&q[0]<=B.x_range_m[1]&&q[1]>=B.y_range_m[0]&&q[1]<=B.y_range_m[1]?q:null;
}

function draw(){
  ctx.fillStyle='#303746';ctx.fillRect(0,0,canvas.width,canvas.height);
  const top=(B.x_range_m[1]-VMAX)/R, rows=(VMAX-B.x_range_m[0])/R;   // BEV rows from VMAX down to the BEV's near edge
  if(image.complete&&image.naturalWidth)ctx.drawImage(image,0,top,image.naturalWidth,rows,0,0,canvas.width,rows*S);
  ctx.strokeStyle='#ffffff40';ctx.fillStyle='#fff';ctx.font='14px sans-serif';ctx.lineWidth=1;
  for(let x=0;x<=VMAX;x+=.5){const v=toPixel([x,0])[1];ctx.beginPath();ctx.moveTo(0,v);ctx.lineTo(canvas.width,v);ctx.stroke();ctx.fillText(x.toFixed(1)+' m',4,v-4)}
  const [ou,ov]=toPixel([0,0]),radius=AHEAD/R*S;
  ctx.strokeStyle='#53e5cf';ctx.lineWidth=2;ctx.setLineDash([10,6]);ctx.beginPath();ctx.arc(ou,ov,radius,0,Math.PI*2);ctx.stroke();ctx.setLineDash([]);
  ctx.strokeStyle='#fff';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(ou-12,ov);ctx.lineTo(ou+12,ov);ctx.moveTo(ou,ov-12);ctx.lineTo(ou,ov+12);ctx.stroke();
  if(point){const [u,v]=toPixel(point);ctx.strokeStyle='#ffb454';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(ou,ov);ctx.lineTo(u,v);ctx.stroke();
    ctx.fillStyle='#ffb454';ctx.beginPath();ctx.arc(u,v,8,0,Math.PI*2);ctx.fill()}
}

function show(r){
  statuses=r.statuses;const c=r.counts;
  $('counts').textContent=`승인 ${c.accepted} · 제외 ${c.rejected} · 미작업 ${c.unlabeled}`;
  $('state').textContent=STATUS[label.status];
}

async function load(i){
  if(busy||i<0||i>=CONFIG.n)return;
  busy=true;
  try{const r=await call('frame',[i]);index=i;label=r.label;point=label.waypoint_m;image.src=r.image;
    $('pos').textContent=`${i+1} / ${CONFIG.n}`;show(r);
    msg(point?'저장된 점입니다. 고치려면 다시 클릭하고 Enter':`${AHEAD} m 원이 좌우 테이프의 가운데와 만나는 곳을 클릭하세요`);draw()}
  catch(e){msg(e.message)}finally{busy=false}
}

async function save(status){
  if(busy)return;
  if(status==='accepted'&&!point){msg(`먼저 ${AHEAD} m 원 위를 클릭하세요`);return}
  busy=true;let saved=false;
  try{const r=await call('save',[index,status,...(status==='accepted'?point:[null,null])]);label=r.label;point=label.waypoint_m;show(r);saved=true}
  catch(e){msg(e.message)}finally{busy=false}
  if(!saved)return;
  if(index+1<CONFIG.n)await load(index+1);
  else{draw();msg(statuses.includes('unlabeled')?'마지막 프레임입니다. "다음 미작업"으로 남은 프레임을 하세요':'다 했습니다. 아래 셀로 넘어가세요')}
}

function nextTodo(){
  for(let k=1;k<=CONFIG.n;k++){const i=(index+k)%CONFIG.n;if(statuses[i]==='unlabeled')return load(i)}
  msg('미작업 프레임이 없습니다. 아래 셀로 넘어가세요');
}

function place(e){
  const r=canvas.getBoundingClientRect(),q=onRing(toMetre([(e.clientX-r.left)*canvas.width/r.width,(e.clientY-r.top)*canvas.height/r.height]));
  if(!q){msg('BEV 범위 밖입니다. 원이 차선과 만나는 앞쪽을 클릭하세요');return}
  point=q;msg(`점 (앞 ${q[0].toFixed(2)} m, 왼쪽 ${q[1].toFixed(2)} m) · 맞으면 Enter`);draw();
}
let dragging=false;
canvas.addEventListener('pointerdown',e=>{if(busy)return;dragging=true;canvas.setPointerCapture(e.pointerId);place(e)});
canvas.addEventListener('pointermove',e=>{if(dragging&&!busy)place(e)});
canvas.addEventListener('pointerup',()=>{dragging=false});
document.addEventListener('keydown',e=>{
  if(e.key==='Enter'){e.preventDefault();save('accepted')}else if(e.key==='x'||e.key==='X')save('rejected');
  else if(e.key==='ArrowLeft')load(index-1);else if(e.key==='ArrowRight')load(index+1)});
$('prev').onclick=()=>load(index-1);$('next').onclick=()=>load(index+1);$('todo').onclick=nextTodo;
$('accept').onclick=()=>save('accepted');$('reject').onclick=()=>save('rejected');
image.onload=draw;
load(0);
})();
