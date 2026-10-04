"""Student entrypoint: python -m camreal {calibrate,prepare,label,export}. Recording itself is ros2 bag record."""
import argparse
import math
import os
from pathlib import Path
import re
import shutil
import yaml


def session_name(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',value):
        raise ValueError('이름은 영문/숫자/_/- 1~80자로 지정하세요.')
    return value


def rel(path):
    """Show paths the way students typed them (relative to the repository root)."""
    path=Path(path)
    return os.path.relpath(path) if path.is_relative_to(Path.cwd()) else str(path)


def load_course(path):
    path=Path(path)
    if not path.is_file():
        raise ValueError(f'설정 파일이 없습니다: {path}. camreal/config/course.yaml을 복사해 만들고 model, calibration 경로를 채우세요.')
    c=yaml.safe_load(path.read_text())
    required={'model','calibration','image_topic','paths','sessions','sampling'}
    if not isinstance(c,dict) or set(c)!=required:
        old=' (이전 버전 설정입니다: bundle/training 대신 model을 씁니다)' if isinstance(c,dict) and 'bundle' in c else ''
        raise ValueError(f'설정 최상위 항목은 {sorted(required)}이어야 합니다.{old}')
    sections={'paths':{'bags','projects','datasets'},'sessions':{'train','val'},'sampling':{'interval_s','max_frames'}}
    for key,fields in sections.items():
        if not isinstance(c[key],dict) or set(c[key])!=fields:
            raise ValueError(f'{key} 항목은 {sorted(fields)}이어야 합니다.')
    for key in ('model','calibration'):
        if not isinstance(c[key],str) or not c[key].strip():raise ValueError(f'{key} 경로가 비었습니다.')
        c[key]=str(Path(c[key]).expanduser().resolve())
    for key,value in c['paths'].items():
        if not isinstance(value,str) or not value.strip():raise ValueError(f'paths.{key} 경로가 비었습니다.')
        c['paths'][key]=str(Path(value).expanduser().resolve())
    seen=set()
    for split in ('train','val'):
        values=c['sessions'][split]
        if not isinstance(values,list) or not values:raise ValueError(f'sessions.{split}에 세션 이름이 필요합니다.')
        for value in values:
            session_name(value)
            if value in seen:raise ValueError('train/val에 같은 세션을 중복 지정할 수 없습니다.')
            seen.add(value)
    sampling=c['sampling']
    if type(sampling['max_frames']) is not int or sampling['max_frames']<=0:raise ValueError('max_frames는 양의 정수여야 합니다.')
    interval=sampling['interval_s']
    if not isinstance(interval,(int,float)) or not math.isfinite(interval) or interval<0:raise ValueError('interval_s가 잘못되었습니다.')
    if not isinstance(c['image_topic'],str) or not c['image_topic'].startswith('/') or any(x.isspace() for x in c['image_topic']):
        raise ValueError('image_topic은 실제 절대 ROS 토픽 이름이어야 합니다.')
    return c


def calibrate(args):
    """Week-1 ost.yaml + one floor frame -> marker clicks in the browser -> course calibration (data/calibration/car.yaml)."""
    if (args.session is None)==(args.image is None):
        raise ValueError('bag 세션 이름과 --image 중 하나만 지정하세요. 예: python3 -m camreal calibrate calib 또는 --image frame.png')
    c=load_course(args.config)
    from camreal.calibration import core
    from camreal.calibration.server import CalibrationSession, serve
    ost=Path(args.ost).expanduser()
    if not ost.is_file():
        option=core.reference_option(ost)
        raise FileNotFoundError(f'ost.yaml이 없습니다: {args.ost}. 1주차 결과를 그 위치에 두세요'+(f' (또는 {option}로 기준 파일 지정).' if option else '.'))
    intr=core.read_ost(ost)
    print(f'ost.yaml: {rel(intr.path)} ({core.ost_kind(intr.path)}) · {intr.width}x{intr.height}',flush=True)
    if args.image is not None:
        source=Path(args.image).expanduser().resolve()
        if not source.is_file():raise FileNotFoundError(f'이미지 파일이 없습니다: {rel(source)}')
        frame,stamp=core.read_image(source),None
        print(f'프레임: {rel(source)}',flush=True)
    else:
        source=Path(c['paths']['bags'])/session_name(args.session)
        try:frame,stamp=core.read_bag_frame(source,c['image_topic'])
        except ImportError as exc:
            raise ValueError(f'bag을 읽으려면 ROS 2가 필요합니다 ({exc.name or exc} 없음). source /opt/ros/humble/setup.bash 후 다시 실행하거나 --image를 쓰세요.') from exc
        print(f'프레임: {rel(source)}의 {c["image_topic"]} 가운데 메시지 (stamp {stamp} ns)',flush=True)
    core.check_resolution(frame,intr)
    markers_path=Path(args.markers).expanduser().resolve()
    if not markers_path.is_file():
        raise FileNotFoundError(f'마커 파일이 없습니다: {rel(markers_path)}. camreal/config/markers.yaml을 복사해 줄자로 잰 값으로 고치세요.')
    markers,markers_sha256=core.load_markers(markers_path)   # the sha256 of the bytes parsed: save refuses a later edit
    print(f'마커: {rel(markers_path)} ({len(markers)}개)',flush=True)
    model=Path(c['model'])
    if (model/'checkpoint.json').is_file():
        from camreal.checkpoint import load_config
        cfg=load_config(model)
        print(f'BEV 규격: {rel(model/"checkpoint.json")} (주행 모델과 같음)',flush=True)
    else:
        from camsim import config
        cfg=config.load()
        print(f'BEV 규격: camsim 기본 설정 ({rel(model)}에 checkpoint.json이 없음)',flush=True)
    serve(CalibrationSession(intr,frame,core.undistort(frame,intr),markers,cfg,Path(c['calibration']),
        dict(frame=str(source),frame_stamp_ns=stamp,markers=str(markers_path),markers_sha256=markers_sha256)),args.port)


def run(args):
    if args.command=='calibrate':return calibrate(args)
    c=load_course(args.config)
    projects=Path(c['paths']['projects'])
    if args.command=='prepare':
        from camreal.labeling.bag import extract_bag, existing_project
        session=session_name(args.session)
        bag,output=Path(c['paths']['bags'])/session,projects/session
        print(f'입력: {rel(bag)} ({c["image_topic"]}) · 모델: {rel(c["model"])} · 보정: {rel(c["calibration"])}',flush=True)
        existing=existing_project(output,bag,c['image_topic'],c['model'],c['calibration'],session,
            c['sampling']['interval_s'],c['sampling']['max_frames'])
        if existing is not None:
            print(f"이미 준비됨: {rel(output)} ({len(existing['frames'])}장). 기존 이미지와 라벨을 유지합니다.\n"
                  f"다음: python3 -m camreal label {session}")
            return
        result=extract_bag(bag,c['image_topic'],c['model'],c['calibration'],output,session,
            c['sampling']['interval_s'],c['sampling']['max_frames'])
        n=len(result['frames'])
        print(f"출력: {rel(output)} · 원본 raw/ {n}장 + BEV bev/ {n}장\n다음: python3 -m camreal label {session}")
    elif args.command=='label':
        from camreal.labeling.server import serve
        serve(projects/session_name(args.session),args.port)
    else:
        from camreal.labeling.core import export_dataset
        from camreal.evaluate import evaluate, report
        output=Path(c['paths']['datasets'])/session_name(args.name)
        archive=output.with_suffix('.zip')
        if output.exists() or archive.exists():
            raise ValueError(f'{rel(output)}이(가) 이미 있습니다. 다른 이름으로 실행하세요: python3 -m camreal export {args.name}_v2')
        print(f"입력: {rel(projects)}의 승인 라벨 · train {c['sessions']['train']} / val {c['sessions']['val']}",flush=True)
        manifest=export_dataset([projects/s for s in c['sessions']['train']],[projects/s for s in c['sessions']['val']],output)
        print(f"1/3 데이터셋: {rel(output)} (train {manifest['counts']['train']}장 · val {manifest['counts']['val']}장, "
              'camsim DiskDataset 형식)',flush=True)
        shutil.make_archive(str(output),'zip',root_dir=output.parent,base_dir=output.name)
        print(f'2/3 Colab 업로드용 압축: {rel(archive)}',flush=True)
        print(f'3/3 현재 모델({rel(c["model"])})의 실데이터 오차 계산',flush=True)
        result=evaluate(c['model'],output,output/'baseline')
        print(report(result))
        print(f"오차가 큰 프레임부터 보기: {rel(output/'baseline'/'index.html')}")


def main(argv=None):
    from camreal.calibration import WEEK1_OST
    p=argparse.ArgumentParser(description='Camreal · 3주차 실차 데이터 실습 (기록은 ros2 bag record 사용)')
    commands=p.add_subparsers(dest='command',required=True)
    for command,help_text in [('calibrate','1주차 ost.yaml + 바닥 마커 클릭 → 지면 캘리브레이션 (data/calibration/car.yaml)'),
                              ('prepare','rosbag → 원본/BEV 프레임 추출 (data/bags/세션 → data/labeling/세션)'),
                              ('label','브라우저에서 1 m waypoint 라벨링'),
                              ('export','승인 라벨 → 학습용 데이터셋 (data/datasets/이름) + 현재 모델 오차')]:
        sub=commands.add_parser(command,help=help_text)
        sub.add_argument('--config',default='data/camreal.yaml',help='설정 파일')
        if command in ('prepare','label'):sub.add_argument('session',help='기록 세션 이름 (예: run_train)')
        if command in ('calibrate','label'):sub.add_argument('--port',type=int,default=8765)
        if command=='export':sub.add_argument('name',help='데이터셋 이름 (예: week3_real)')
        if command=='calibrate':
            sub.add_argument('session',nargs='?',help='주차 칸에 세우고 기록한 bag 세션 이름 (예: calib). --image와 둘 중 하나')
            sub.add_argument('--image',help='bag 대신 쓸 원본 해상도 이미지 (PNG/JPG)')
            sub.add_argument('--ost',default=WEEK1_OST,help='1주차 ost.yaml (기본: %(default)s)')
            sub.add_argument('--markers',default='data/calibration/markers.yaml',help='마커 실측 좌표 (기본: %(default)s)')
    args=p.parse_args(argv)
    from camsim.config import ConfigError
    try:run(args)
    except (ValueError,OSError,KeyError,ConfigError) as exc:p.exit(2,f'오류: {exc}\n')


if __name__=='__main__':main()
