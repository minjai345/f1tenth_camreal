'use strict';
// One waypoint per frame: the lane centre where it crosses the ahead_m circle around the rear axle.
const $=id=>document.getElementById(id), canvas=$('canvas'), ctx=canvas.getContext('2d');
const STATUS={unlabeled:'미작업',accepted:'승인',rejected:'제외'};
let project, index=0, annotation, point=null, dirty=false, dragging=false, busy=false, generation=0;
let viewMinX, viewMaxX;
const bevImage=new Image();
const res=()=>project.bev.resolution_m;
const pointToPixel=([x,y])=>[(project.bev.y_range_m[1]-y)/res(),(viewMaxX-x)/res()];
const pixelToPoint=([u,v])=>[viewMaxX-v*res(),project.bev.y_range_m[1]-u*res()];
const eventPixel=e=>{const r=canvas.getBoundingClientRect();return[(e.clientX-r.left)*canvas.width/r.width,(e.clientY-r.top)*canvas.height/r.height]};
const feedback=text=>$('feedback').textContent=text;
async function api(path,body){const options=body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-Label-Token':project.token},body:JSON.stringify(body)};const r=await fetch(path,options),value=await r.json();if(!r.ok)throw Error(value.error||r.status);return value}

function onRing(p){
  // Same snapping as the server: keep the direction from the rear axle, fix the distance.
  const r=Math.hypot(p[0],p[1]);if(r<1e-6)return null;
  const q=[p[0]*project.ahead_m/r,p[1]*project.ahead_m/r],b=project.bev;
  return q[0]>=b.x_range_m[0]&&q[0]<=b.x_range_m[1]&&q[1]>=b.y_range_m[0]&&q[1]<=b.y_range_m[1]?q:null;
}

function draw(){
  if(!project)return;const b=project.bev,r=res();
  ctx.fillStyle='#303746';ctx.fillRect(0,0,canvas.width,canvas.height);
  const top=(b.x_range_m[1]-viewMaxX)/r,bh=(viewMaxX-b.x_range_m[0])/r;
  if(bevImage.complete&&bevImage.naturalWidth)ctx.drawImage(bevImage,0,top,bevImage.naturalWidth,bh,0,0,canvas.width,bh);
  // Hatched: behind the BEV range, no camera image.
  ctx.save();ctx.beginPath();ctx.rect(0,bh,canvas.width,canvas.height-bh);ctx.clip();
  ctx.strokeStyle='#536078';ctx.lineWidth=1;for(let v=bh;v<canvas.height+canvas.width;v+=20){ctx.beginPath();ctx.moveTo(0,v);ctx.lineTo(canvas.width,v-canvas.width/3);ctx.stroke()}
  ctx.restore();
  ctx.strokeStyle='#ffffff33';ctx.fillStyle='#eee';ctx.font='15px sans-serif';
  for(let x=0;x<=viewMaxX;x+=.5){const v=pointToPixel([x,0])[1];ctx.beginPath();ctx.moveTo(0,v);ctx.lineTo(canvas.width,v);ctx.stroke();ctx.fillText(x.toFixed(1)+'m',5,v-4)}
  const [ou,ov]=pointToPixel([0,0]),radius=project.ahead_m/r;
  ctx.strokeStyle='#53e5cf';ctx.lineWidth=2;ctx.setLineDash([10,6]);ctx.beginPath();ctx.arc(ou,ov,radius,0,Math.PI*2);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle='#53e5cf';ctx.fillText(`${project.ahead_m} m`,ou+radius*.72+6,ov-radius*.72);
  ctx.strokeStyle='white';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(ou-12,ov);ctx.lineTo(ou+12,ov);ctx.moveTo(ou,ov-12);ctx.lineTo(ou,ov+12);ctx.stroke();
  if(point){const [u,v]=pointToPixel(point);ctx.strokeStyle='#ffb454';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(ou,ov);ctx.lineTo(u,v);ctx.stroke();
    ctx.fillStyle='#ffb454';ctx.beginPath();ctx.arc(u,v,8,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#121722';ctx.lineWidth=2;ctx.stroke()}
}

function place(e){const q=onRing(pixelToPoint(eventPixel(e)));if(!q){feedback('BEV 범위 밖입니다. 원이 차선과 만나는 앞쪽을 클릭하세요.');return}point=q;dirty=true;feedback(`waypoint (x ${q[0].toFixed(2)} m, y ${q[1].toFixed(2)} m) · 맞으면 Enter`);draw()}
canvas.addEventListener('pointerdown',e=>{if(busy)return;dragging=true;canvas.setPointerCapture(e.pointerId);place(e)});
canvas.addEventListener('pointermove',e=>{if(dragging&&!busy)place(e)});
canvas.addEventListener('pointerup',()=>dragging=false);canvas.addEventListener('pointercancel',()=>dragging=false);

function counts(){const c={unlabeled:0,accepted:0,rejected:0};project.frames.forEach(f=>c[f.status]++);$('counts').textContent=`승인 ${c.accepted} · 제외 ${c.rejected} · 미작업 ${c.unlabeled}`}
function refreshChoices(){project.frames.forEach((f,i)=>$('frames').options[i].textContent=`${i+1} / ${project.frames.length} · ${f.id} · ${STATUS[f.status]}`);$('frames').value=String(index);counts()}

async function load(i){
  if(busy||i<0||i>=project.frames.length)return;
  if(dirty&&!confirm('저장하지 않은 waypoint를 버리고 이동할까요?')){$('frames').value=String(index);return}
  busy=true;const g=++generation;
  try{index=i;const f=project.frames[i];annotation=await api('/api/frame/'+f.id);if(g!==generation)return;
    point=annotation.waypoint_m?annotation.waypoint_m.slice():null;dirty=false;$('note').value=annotation.note||'';
    $('status').textContent=STATUS[annotation.status]||annotation.status;refreshChoices();
    $('raw').src='/images/raw/'+f.id;bevImage.src='/images/bev/'+f.id;
    $('details').textContent=`image timestamp(ns): ${f.header_stamp_ns} · frame: ${f.camera_frame} · revision: ${annotation.revision}`;
    feedback(point?'저장된 waypoint입니다.':`${project.ahead_m} m 원이 차선 중앙과 만나는 곳을 클릭하세요.`);draw()}
  catch(e){feedback(e.message)}finally{busy=false}
}

async function save(status){
  if(busy)return;if(status==='accepted'&&!point){feedback(`먼저 ${project.ahead_m} m 원 위에 waypoint를 찍으세요.`);return}
  busy=true;let saved=false;
  try{annotation=await api('/api/annotation/'+project.frames[index].id,{revision:annotation.revision,status,note:$('note').value,waypoint_m:status==='accepted'?point:null});
    dirty=false;saved=true;project.frames[index].status=status;$('status').textContent=STATUS[status];refreshChoices();
    feedback(`${STATUS[status]} 저장`)}
  catch(e){feedback(e.message)}finally{busy=false}
  if(saved&&index+1<project.frames.length)await load(index+1);
}

function nextTodo(){for(let k=1;k<=project.frames.length;k++){const i=(index+k)%project.frames.length;if(project.frames[i].status==='unlabeled')return load(i)}feedback('미작업 프레임이 없습니다.')}
bevImage.onload=draw;
$('prev').onclick=()=>load(index-1);$('next').onclick=()=>load(index+1);$('todo').onclick=nextTodo;$('frames').onchange=()=>load(Number($('frames').value));
$('accept').onclick=()=>save('accepted');$('reject').onclick=()=>save('rejected');$('note').oninput=()=>dirty=true;
document.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName))return;
  if(e.key==='Enter'){e.preventDefault();save('accepted')}else if(e.key==='x'||e.key==='X'){save('rejected')}
  else if(e.key==='ArrowLeft'){load(index-1)}else if(e.key==='ArrowRight'){load(index+1)}});
window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue=''}});
(async()=>{try{project=await api('/api/project');$('session').textContent=project.session_id;
  project.frames.forEach((f,i)=>{const option=document.createElement('option');option.value=String(i);$('frames').append(option)});
  const b=project.bev;viewMinX=Math.min(-.3,b.x_range_m[0]);viewMaxX=Math.min(b.x_range_m[1],Math.max(2,2.5*project.ahead_m));
  canvas.width=Math.round((b.y_range_m[1]-b.y_range_m[0])/b.resolution_m);canvas.height=Math.round((viewMaxX-viewMinX)/b.resolution_m);
  $('instructions').textContent=`십자 = 후륜축. 점선 원 = 후륜축에서 ${project.ahead_m} m. 원이 좌우 테이프의 중앙(${project.path_line})과 만나는 곳을 클릭하세요. 드래그하면 원을 따라 움직입니다.`;
  if(project.calibration_status==='assumed'){$('warning').hidden=false;$('warning').textContent='가정 캘리브레이션(ASSUMED)으로 만든 BEV입니다. 화면 확인용이며, 실측 보정으로 다시 준비하기 전에는 학습 라벨로 쓰지 마세요.'}
  await load(0)}catch(e){feedback(e.message)}})();
