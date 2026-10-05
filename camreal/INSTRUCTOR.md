# Camreal — 담당자 준비

학생은 [실습 문서](README.md)의 설치와 0~8단계를 따라간다. 담당자는 아래를 수업 전에 준비한다.

## 레포 구조

```text
camsim/     2주차 시뮬 실습 (정태 레포 jeongtaek1m/f1tenth_gym 그대로. 여기서 고치지 않는다)
camreal/
  __main__.py         학생 명령: calibrate / prepare / label / export
  calibration/        지면 캘리브레이션: 1주차 ost.yaml + 바닥 마커 클릭 → data/calibration/car.yaml
  checkpoint.py       camsim 5장 산출물(model.onnx + checkpoint.json) 읽기, onnxruntime 추론
  preprocessing.py    실차 왜곡 보정 · IPM · 학습 가시 영역
  labeling/           rosbag 추출 · 브라우저 라벨링 · camsim 형식 export
  evaluate.py         실데이터 오차 (export가 시뮬 모델 기준으로 자동 실행)
  tools/              담당자 도구
  config/             course.yaml · markers.yaml(마커 템플릿) · recording_qos.yaml · ost_reference_1920x1200.yaml(기준 ost.yaml, 4번에서 추가)
  ros2/camsim_driver/ 실차 ROS 노드 2개: waypoint_node → /waypoint → pure_pursuit_node (상세: ros2/camsim_driver/README.md)
data/  out/           rosbag·라벨·모델·결과 (Git 제외)
```

camsim이 바뀌면 받아 온다. 이 브랜치는 정태 레포(`jt` remote)의 main 위에 camreal을 얹은 것이다.

```bash
cd ~/f1tenth_gym
git pull jt main     # camsim 업데이트. 그다음 colcon build 다시 (아래)
```

## 수업 전 체크리스트

### 1. ROS 패키지와 차량 스택

학생이 직접 설치할 때는 [차량 스택 설치 문서](CAR_STACK.md)를 따라 하게 한다(명령 복사로 끝나게 작성, 새 폴더 빌드로 검증함).
차가 여러 대면 차마다 `ROS_DOMAIN_ID`를 다르게 둔다(CAR_STACK.md 7단계, 학생 문서 설치 4번). 같은 공유기에서 같은 ID면 다른 차의 `/drive`와
`/joy`가 이 차를 움직이고, 주행 노드는 다른 차의 `/waypoint` 때문에 속도 0에 머문다.
학생 문서는 차에 붙은 라벨을 읽게 하니 차마다 라벨 세 개를 붙인다.

- 차 번호(1~101, 차마다 다르게): 설치 4번에서 `ROS_DOMAIN_ID`로 넣는 수. 조교가 설치할 때도 이 번호를 넣는다
- 카메라 serial: 0단계 launch의 `serial` 값(1주차 launch에 쓴 값 또는 SpinView)
- 렌즈 높이: 4번의 5

아래는 차량 스택 설치 문서의 요약이다.

```bash
sudo apt update && sudo apt install -y ros-humble-ackermann-msgs ros-humble-serial-driver ros-humble-urg-node \
  ros-humble-control-msgs ros-humble-test-msgs ros-humble-rosbridge-server ros-humble-sick-scan-xd \
  ros-humble-asio-cmake-module
cd ~/f1tenth_ws && source /opt/ros/humble/setup.bash && colcon build
```

f1tenth_system(humble-devel)은 `~/f1tenth_ws/src/f1tenth_system`에 받아 두었다(submodule 포함).
위 목록은 rosdep이 계산한 누락 의존성에 `asio-cmake-module`을 더한 것이다.
`asio-cmake-module`은 rosdep이 잡지 못하지만 `vesc_driver` 빌드에 필요하다.
apt가 "dpkg was interrupted"로 멈추면 먼저 `sudo dpkg --configure -a`를 실행한다. VESC·조이스틱 udev 규칙과 LiDAR 연결은
[F1TENTH 문서](https://f1tenth.readthedocs.io)를 따른다(`vesc.yaml`의 port: `/dev/sensors/vesc`).

주행 노드(`camsim_driver`) 빌드:

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
colcon build --base-paths camreal/ros2/camsim_driver --packages-select camsim_driver
```

`camsim/`나 `camreal/`을 고친 뒤에도 다시 빌드한다(설치 시 복사됨).

주행 노드는 GPU용 onnxruntime이 필요하다. 이 PC는 Jetson AI Lab 휠(onnxruntime-gpu 1.23.0, JetPack 6)을 쓴다.
아래 출력에 `CUDAExecutionProvider`가 없으면 설치한다.

```bash
python3 -c "import onnxruntime as o; print(o.get_available_providers())"
pip3 install "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/4eb/e6a8902dc7708/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl#sha256=4ebe6a8902dc7708434b2e1541b3fe629ebf434e16ab5537d1d6a622b42c622b"
```

torch는 `export`(학생 7단계)에만 필요하다. 없으면 학생 문서 설치 3번의 휠을 설치한다.

### 2. 시뮬 모델 받기 (camsim 5장 산출물)

Colab 5장 셀이 Drive `MyDrive/camsim_results/`에 `model.onnx`, `checkpoint.json`, `model.pt`를 저장한다.
차에는 **`model.onnx`와 `checkpoint.json` 두 파일**만 받는다(`model.pt`는 Colab에서 이어 학습할 때만 씀).
두 파일 모두 "링크가 있는 모든 사용자: 뷰어"로 공유한다.

```bash
cd ~/f1tenth_gym
python3 -m venv ~/camsim-transfer-venv && ~/camsim-transfer-venv/bin/python -m pip install gdown
mkdir -p data/models/sim
~/camsim-transfer-venv/bin/python -m gdown --fuzzy 'MODEL_ONNX_공유링크' -O data/models/sim/model.onnx
~/camsim-transfer-venv/bin/python -m gdown --fuzzy 'CHECKPOINT_JSON_공유링크' -O data/models/sim/checkpoint.json
```

SHA-256 대조와 BEV 크기 검사는 camreal이 읽을 때 자동으로 한다. 따로 할 필요 없다.
camsim 문서의 `trtexec` 엔진 빌드는 하지 않는다. camreal은 onnxruntime CUDA로 `model.onnx`를 바로 돌린다
(ResNet-18 약 10 ms). 옛 노트북의 `model.pt` 결과는 거부되니 최신 노트북으로 다시 받는다.
학습 설정은 `checkpoint.json`에서만 읽는다. 학생이 Colab에서 `camera.h_i2g_file`을 썼다면
그 `.npy`도 같은 폴더에 둔다.

### 3. 설정 파일

```bash
cd ~/f1tenth_gym
mkdir -p data/config
cp -n camreal/config/course.yaml data/camreal.yaml          # model 경로 확인. calibration(car.yaml)은 calibrate가 만듦
cp -n camreal/ros2/camsim_driver/config/vehicle.yaml data/config/vehicle.yaml
```

`data/config/vehicle.yaml`에서 `wheelbase_m`(실측), `steer_max_rad`, `target_speed_mps`(기본 0.5, 상한 2.0)를 채운다.
두 노드가 이 파일 하나(`/**:`)를 같이 읽는다. launch는 모르는 이름, 기본값과 형식이 다른 값(`0.33` 자리에 `1`), `drive_enabled`가 있으면 멈춘다.
이전 형식(`camsim_driver_node:`)이 남아 있으면 launch가 "이전 형식의 vehicle.yaml입니다" 오류를 낸다.
`cp -n`은 덮어쓰지 않으니 `-n` 없이 다시 복사하고 값을 채운다.
같은 wheelbase를 `~/f1tenth_ws/src/f1tenth_system/f1tenth_stack/config/vesc.yaml`의 odometry에도 넣는다.
학생은 `data/camreal.yaml`에서 `sessions`(4단계)와 `model`(8단계)만 바꾼다.
`calibrate`도 이 파일을 읽는다. `calibration`은 저장 위치, `model`의 `checkpoint.json`은 BEV 미리보기 규격과 1 m(`ahead_m`)다
(모델이 없으면 camsim 기본 설정으로 보여 주고 터미널에 그렇게 적는다).

### 4. 캘리브레이션 (1주차 ost.yaml + 바닥 마커)

학생은 1단계에서 `python3 -m camreal calibrate`로 `data/calibration/car.yaml`(옆에 BEV 미리보기 `car_bev.png`)을 만든다.
1주차 `ost.yaml`(K, D, P)에 바닥 마커 클릭으로 구한 `H_i2g`(왜곡 보정 영상 → 후륜축 지면, m)를 더한 파일이다.
노드가 읽는 항목 외에 출처(`source`: ost·마커 파일 경로와 sha256, 쓴 프레임, 만든 시각), 마커별 오차, 추정 카메라 자세(`camera_pose_estimate`)가 들어 있다.
담당자는 아래 세 가지를 준비한다.

**기준 ost.yaml (한 번).** 학생 문서에서는 이 파일이 보통 경로다(`--ost camreal/config/ost_reference_1920x1200.yaml`).
1주차 72쪽에서 해상도를 1280×720으로 바꾼 뒤 캘리브레이션한 학생 파일은 3주차 영상(1920×1200)과 해상도가 달라 `calibrate`가 거부한다.
이 파일을 커밋하기 전에는 학생 문서의 기준 파일 명령이 `기준 ost.yaml이 아직 레포에 없습니다`로 멈춘다(학생 문서는 손 들기로 안내).
커밋하기 전에 레포를 받은 차는 `cd ~/f1tenth_gym && git pull`로 받는다.
같은 카메라·렌즈 모델이고 초점 링을 고정했다는 전제다. 3주차 해상도(1920×1200)로 1주차 방식 그대로 캘리브레이션한다.
먼저 카메라 launch(`/opt/ros/humble/share/spinnaker_camera_driver/launch/driver_node.launch.py`)의 `image_width`·`image_height`를
1920·1200(offset 0)으로 두고 `ros2 topic echo /flir_camera/image_raw --once --field width`로 확인한다. 모든 차를 이 해상도로 맞춘다.

```bash
ros2 run camera_calibration cameracalibrator --size 10x7 --square 0.025 \
  image:=/flir_camera/image_raw camera:=/flir_camera --no-service-check
tar -xzf /tmp/calibrationdata.tar.gz -C /tmp ost.yaml
cp /tmp/ost.yaml ~/f1tenth_gym/camreal/config/ost_reference_1920x1200.yaml
```

GUI의 scale 슬라이더는 0에 둔 채 SAVE한다(슬라이더가 P를 바꾼다). 파일 맨 위에 만든 날짜·차·렌즈를 주석으로 적고 커밋한다.
차마다 다른 주점 차이는 차별 `H_i2g`가 대부분 흡수하고, 남는 오차는 `calibrate`의 마커 오차로 드러난다.

**바닥 마커와 주차 칸 (한 번).**

1. 차가 늘 같은 자세로 서도록 네 바퀴 바깥에 테이프로 주차 칸을 만든다. 차를 1° 틀어 세우면 2 m 앞에서 3.5 cm 어긋난다
2. 좌우 뒷바퀴 축 바로 아래를 잇는 선(후륜축 선, x = 0)과 차 중심선(y = 0)을 바닥에 표시한다
3. `mkdir -p data/calibration && cp camreal/config/markers.yaml data/calibration/markers.yaml` 후 템플릿 위치
   (x 0.6/1.0/1.5/2.0 m, y +0.4/0/−0.4 m)에 테이프 십자 12개를 붙인다. 카메라 화면에 안 들어오는 마커는 화면 안으로 옮긴다
4. 줄자로 각 십자 중심을 재서 `data/calibration/markers.yaml`의 값을 실측값으로 고친다 (x는 후륜축 선에서 앞쪽, y는 중심선에서 왼쪽 +, 단위 m)
5. 차마다 카메라 렌즈 중심 높이를 재서 1번의 차 번호·카메라 serial 옆에 라벨로 붙인다(학생 1단계 확인 2). `calibrate`가 추정한 높이와 2 cm 안으로 맞아야 한다
6. 같은 주차 칸을 쓰는 차에는 같은 `markers.yaml`을 `data/calibration/`에 복사한다

마커는 1 m 앞 waypoint 자리를 둘러싸야 한다. x ≤ 1 m 쪽과 x ≥ 1 m 쪽 각각에 왼쪽(y ≥ 0.2 m)과 오른쪽(y ≤ −0.2 m) 마커가
하나 이상 찍혀야 한다(A1·A3와 C1·C3를 찍으면 충분). 비면 `calibrate`가 "…둘러싸지 못합니다" 노랑 경고를 낸다(저장은 됨).
`markers.yaml`은 단위 m로 6개 이상, x > 0, 좌표 중복 없이, 두 줄 이상에 나눠 적어야 `calibrate`가 시작한다.
`calibrate` 실행 중에 이 파일을 고치면 저장이 거부되니 `calibrate`를 다시 실행한다.

**차마다 한 번 미리 해 보기.** 2번 모델과 3번 설정 파일을 둔 뒤 학생 문서 0~1단계를 그대로 하되, 기록 이름 `calib`는 `calib_ta`로
바꾼다(1-1의 `--output data/bags/calib_ta`, 1-2의 `calibrate calib_ta`). `calib`를 남기면 수업에서 학생의 1-1 기록이 `already exists`로 멈추고,
`calibrate calib`가 조교의 영상을 띄운다(같은 브라우저면 조교가 찍은 점까지). 이미 `calib`로 했으면 `mv data/bags/calib data/bags/calib_ta`.
1 m 근처(A·B줄) 마커 오차가 3 cm를 넘으면 마커 실측값, 주차 자세, 클릭 위치 순으로 다시 확인한다.
카메라 높이가 음수로 나오면(저장 안 됨) 마커 y 부호나 [x, y] 순서가 틀린 것이다.
여기서 저장한 `car.yaml`은 수업 중 학생의 1단계가 막혔을 때 그대로 쓰인다.

실측 캘리브레이션 전에 연결만 볼 때는 가정값을 쓴다(주행 불가). `data/camreal.yaml`의 `calibration`을 이 파일로 바꿔 쓰고,
`calibrate`를 실행하기 **전에** `data/calibration/car.yaml`로 되돌린다. `calibrate`는 `calibration`이 가리키는 파일에 저장한다(터미널의 `저장 위치`,
화면의 `저장:`). 되돌리지 않으면 실측값이 `ASSUMED_camera.yaml`에 저장되고 `car.yaml`은 생기지 않는다.

```bash
cd ~/f1tenth_gym
python3 -m camreal.tools.make_assumed_calibration --out data/calibration/ASSUMED_camera.yaml
```

카메라 마운트·ROI·해상도가 바뀌면 1단계를 다시 한다(전 파일은 `data/calibration/old/`에 남음).
노드는 시작할 때만 캘리브레이션을 읽으니 launch도 다시 실행한다.
캘리브레이션을 바꾸면 이미 `prepare`한 세션은 재사용이 거부된다. 안내대로 기존 `data/labeling/세션`을 다른 이름으로 옮기고
다시 `prepare`한다(기존 BEV는 안 바뀜).

### 5. 사전 점검 (수업 전, 시뮬 모델이 실차에서 되는지 판정)

학생 문서를 그대로 따라간다. 1단계는 4번 미리 해 보기의 결과를 쓰고, 다시 하면 기록 이름은 `calib_ta2`처럼 학생의 `calib`와 겹치지 않게 한다.
안 될 때 캘리브레이션 문제인지 모델 문제인지 가르는 것이 목적이다.
시뮬 모델은 학생과 같은 기본 설정으로 Colab 1~5장을 돌린 것을 쓴다(2번).
트랙 테이프 색·폭, 바닥색이 시뮬 기본값(노란 5 cm 테이프, 회색 바닥)과 크게 다르면 먼저 Colab 파라미터를 맞춰 다시 학습한다.
차선 폭은 기본값(`lane.follow_walls: true`)에서 테이프가 맵 벽을 따라가 구간마다 다르다. `track_width_m`(0.8 m)은 `follow_walls: false`일 때만 쓰인다.
실제 트랙처럼 폭이 일정한 차선으로 다시 학습하려면 `follow_walls: false`, `track_width_m` = 실제 차선 폭으로 둔다.

| 학생 단계 | 할 일 | 합격 기준 |
|---|---|---|
| 설치 5 | 확인 블록 | `ROS OK`, `GPU OK`, `camsim_driver`, `f1tenth_stack`, 차 번호 |
| 0 | 카메라, 1주차 파일 확인 | 영상 1920×1200. 1주차 `ost.yaml`도 1920×1200이면 1단계에 그 파일, 아니면(보통) 기준 파일 |
| 1 | 지면 캘리브레이션 | A·B줄 마커 오차 3 cm 이하, 추정 카메라 높이 ≈ 줄자 값 (±2 cm), BEV 미리보기의 초록 십자가 테이프 십자 위 |
| 2 | 직선 가운데에 똑바로 세우고 예측만 보기 | BEV에서 테이프 평행, 간격 = 실제 폭, `/waypoint` y ≈ 0 |
| 2 | 손으로 밀며 직선·코너·치우친 자세. 동시에 4단계 명령으로 `run_pilot` 기록 | 예측점이 차선 가운데 쪽 |
| 2 (선택) | 손으로 밀며 불 끄기, 손전등, 그림자 | 메모 (주행은 평소 조명에서만) |
| 3 전 | 바퀴 띄우고 주행 켬 + RB: 조향 방향(점이 왼쪽이면 바퀴도 왼쪽), RB를 떼면 정지, RB를 누른 채 LB+스틱이면 수동이 우선, T1 카메라를 Ctrl+C로 끄면 속도 0(BEV는 회색 `no image`), T3 Ctrl+C면 속도 0으로 끝남, `ros2 topic hz /joy` 약 20 Hz | 전부 정상 |
| 3 | 0.5 m/s, RB | 평소 조명에서 개입 없이 3바퀴 연속 |

판정:

- 3단계 통과 → 3주차는 주행 중심(학생 0~3단계). 4~8단계는 빼거나 과제로
- 2단계 BEV가 틀림 → 캘리브레이션 문제. 1단계부터 다시 (모델 판정 보류)
- 2단계 BEV는 맞는데 예측·주행이 틀림 → 모델 문제
  1. 시뮬을 실차에 맞춰 재학습: 트랙 파라미터 + `car.yaml`의 `camera_pose_estimate`
     (`height_m`, `pitch_deg`, `hfov_deg`, `x_m` → camsim `camera.height_m`, `camera.pitch_deg`, `camera.hfov_deg`, `camera.offset_x_m`)
  2. 그래도 안 되면 `run_pilot` bag으로 5~6단계(추출·라벨링)를 해 보고, 3주차에 실차 데이터 수집(4~7단계)을 넣는다.
     7단계(export, 시뮬 모델의 실데이터 오차)는 bag 하나로는 안 된다. `sessions`에 적힌 세션만 읽고, train·val에 서로 다른 주행 bag이 필요하다.
     하려면 다른 주행을 `run_pilot_val`로 하나 더 기록해 5~6단계를 똑같이 하고, `sessions`를 train `[run_pilot]`, val `[run_pilot_val]`로
     바꿔 export한다(수업 전에 되돌린다)

주행 안전: [ROS 문서의 실차 확인 순서](ros2/camsim_driver/README.md#실차-확인-순서)대로 바퀴를 띄운 상태부터 시작한다.
자율주행은 조이스틱 **RB를 누르고 있는 동안** 전달된다(LB는 수동 운전). LB가 아닌 다른 버튼을 눌러도 똑같이 전달되니 주행 중에는 RB만 누르게 한다.
단, **조이스틱이 연결되지 않으면(`/joy` 없음) 버튼 없이도 전달된다.** 주행 전 `ros2 topic hz /joy`를 확인한다.

## 수업 진행 (이론 → 실습)

1·2주차처럼 이론 다음에 실습한다. 학생은 코드를 보지 않는다. README에서 명령어를 복사해 붙여넣고, 브라우저에서 클릭하고, 화면을 보고 판단하는 것만 한다.
슬라이드도 README의 단계 번호(설치, 0~8)와 터미널 이름(T1 카메라, T2 차량 스택, T3 주행 노드, T4 BEV 화면, T5 그때그때)을 그대로 쓴다.

### 이론

1·2주차에 배운 것(BEV, ROS 토픽)은 다시 가르치지 않고 오늘 쓰는 곳에 연결한다.

1. 오늘의 그림: 카메라 → 왜곡 보정(1주차 K, d) → BEV(오늘 1단계) → 모델(2주차) → 1 m 앞 점 → pure pursuit → 조향
2. 복습, pure pursuit: 2주차 "왜 Waypoint?" 그림에 공식 한 줄 `δ = atan(2L·sin α / l_d)`를 더한다.
   `pure_pursuit_node`의 계산은 이것뿐이다(l_d ≈ 1 m, L = `vehicle.yaml`의 `wheelbase_m`, 속도 0.5 m/s 고정).
3. 실차에서 IPM 만들기: 1주차 체커보드 대응점과 같은 원리로, 바닥 마커 십자와 영상 점을 짝지어 H(영상 → 바닥 m)를 구한다.
   2주차의 "실차에서는 Perspective view → IPM → BEV"를 실제로 만드는 단계다. 1주차 재투영 오차(px) 대신 마커 오차(cm)와 추정 카메라 높이로 확인한다.
4. 노드 구성과 안전: `waypoint_node` → `/waypoint` → `pure_pursuit_node` → `/drive`. 조이스틱이 우선이다
   (RB를 누르는 동안 자율주행, 버튼을 다 떼면 정지, LB는 수동. LB가 아닌 다른 버튼도 RB와 같으니 RB만 누르게 한다). 영상이 끊기거나 예측점이 이상하거나 새 점이 0.25초 동안 안 오면 노드가 속도 0을 보낸다.
   사람이 멈추는 순서는 버튼에서 손 떼기 → T3 Ctrl+C → 차를 들어 올리고 전원 끄기.
5. 시뮬과 실차의 차이: 2주차 노트북 3장의 sim-to-real 표를 실차에서 확인한다. 차에 올린 기본 모델은 증강 없이 학습했다(`AUGMENT_FN = None`).
   조명 변화는 2단계 화면에서 보고, 주행(3단계)은 평소 조명에서만 한다(관찰지 소개).
6. 실차 데이터, rosbag과 라벨링: rosbag은 토픽 메시지를 받은 시각과 함께 그대로 저장한 파일이고 1단계 캘리브레이션 기록에 바로 쓴다
   (`record`, `info`, 폴더 하나 = bag 하나, 1분에 약 5 GB, train과 val은 다른 주행). 라벨링은 2주차 6쪽 "같은 대상은 같은 기준으로"를 이어서,
   BEV의 1 m 원과 차선 가운데가 만나는 점을 사람이 찍는다. A안이면 라벨링은 실습 끝 과제 안내 때 보여 줘도 된다.

슬라이드에는 성공했을 때의 화면(클릭 도구, BEV와 예측점, `rqt_graph`)을 넣는다. 명령어는 GitHub README에서 복사하게 한다. PDF에서 복사하면 줄바꿈과 따옴표가 깨지기 쉽다.

### 실습

1. 조교 차로 주행(3단계)을 먼저 보여 준다. 조 안에서 역할(명령 입력, 화면 보고 기록, 차 옮기기, 정지 담당)을 나눈다.
2. 학생 설치 확인(설치 5번)과 0~2단계. 1단계는 주차 칸에서 3초 기록할 때만 조별로 줄을 서고, 클릭은 자리에서 한다.
   1단계가 막히면 저장하지 말고 넘어간다(사전 점검 때 만든 `car.yaml`이 그대로 쓰인다). 이미 저장했으면 `ls data/calibration/old/`에서
   파일 이름의 날짜·시각을 보고 학생이 저장하기 직전 것을 `cp data/calibration/old/car-날짜-시각.yaml data/calibration/car.yaml`로 되돌린다.
3. 조교가 차마다 바퀴를 띄워 점검한다(사전 점검 표의 "3 전" 줄).
4. 학생 3단계(평소 조명에서만). 트랙은 조별로 돌아가며 쓰고, 조마다 첫 주행의 조이스틱은 조교가 잡는다. 조마다 정지 담당 한 명이 T3(주행 노드) 앞에 선다.
   기다리는 조는 조명을 바꿔 가며 2단계 화면을 관찰한다.
5. 조별로 어디서, 왜 벗어났는지 공유한다(관찰지).

사전 점검에서 3단계를 통과하지 못했으면 3단계는 실패 원인만 보고 4~6단계(기록, 라벨링)까지 한다. 7~8단계와 재주행은 다음 주에 한다.
8단계는 지금 학생이 코드를 짜는 형태라, 그 전에 실행만 하면 되는 Colab 셀로 바꿔 둔다.
시간표는 사전 점검 때 단계별로 걸린 시간을 재서 짠다.

## 운영 메모

- 원본 영상 기록은 1분에 약 5 GB다. 30~60초씩 기록하고, `prepare`가 끝난 bag은 필요하면 외부로 옮긴다.
- `calibrate`는 저장할 때 기존 `car.yaml`을 `data/calibration/old/car-날짜-시각.yaml`로 복사해 두고 새로 쓴다(`car_bev.png`도 새로 씀).
  되돌릴 때는 그 파일을 다시 `car.yaml`로 복사한다.
- 클릭 도구(`calibrate`)와 라벨 도구(`label`)는 같은 포트(8765)를 쓴다. 동시에 띄우려면 `--port`를 바꾼다.
- 두 도구는 그 컴퓨터 안(http://127.0.0.1:8765)에서만 열린다. 차에 모니터를 연결해 쓰거나, 노트북에서는
  `ssh -L 8765:127.0.0.1:8765 <차 주소>`로 접속한 채 노트북 브라우저에서 같은 주소를 연다. 주행 중 T3에서 Ctrl+C할 화면도 정지 담당 앞에 있어야 한다.
- 클릭 도구는 찍은 점을 브라우저에 영상별로 기억한다. 같은 bag으로 `calibrate`를 다시 실행하면 점이 다시 나온다.
  다른 탭에서 같은 영상의 점을 고치면 먼저 연 탭은 멈춘다(새로 고치면 됨).
- 조이스틱 우선순위는 `ackermann_mux`가 정한다(조이스틱 `teleop` 100 > 자율주행 `/drive` 10). 3주차는 기본 설정 그대로 RB를 누르는 동안 자율주행이 통과한다
  (`joy_teleop.yaml`의 `default` 명령이 버튼을 하나도 안 누르면 속도 0을 `teleop`으로 보내기 때문. 그래서 LB가 아닌 아무 버튼이나 RB와 같다). 이후 주차에 버튼 없이 자율주행하려면
  `~/f1tenth_ws/src/f1tenth_system/f1tenth_stack/config/joy_teleop.yaml`에서 `default` 블록만 지우고 다시 빌드한다. 그러면 평소엔 `/drive`가 통과하고 LB+스틱이 우선한다.
- train/val은 서로 다른 주행이어야 한다. export는 같은 bag이 두 split에 들어가면 거부한다.
- 조명 조건별 세션(`run_val_bright`, `run_val_dark` 등)을 만들면 export 결과에서 조건별 오차가 나뉜다.
- 라벨 서버는 프로젝트당 하나만 띄운다. 여러 학생이 같은 세션을 동시에 편집하지 않는다.
- 실데이터 학습은 학생 몫이다. 데이터셋은 camsim `DiskDataset` 형식이고, Jetson 메모리(약 7 GB)로는 학습이
  빠듯하니 Colab을 권한다(학생 문서 8단계).

## 담당자 도구

| 명령 | 용도 |
|---|---|
| `python3 -m camreal calibrate --image 파일 --ost 파일` | bag 없이 저장한 이미지(원본 해상도)로 지면 캘리브레이션 |
| `python3 -m camreal.tools.infer_images --input 폴더 --out 새폴더` | 실제 이미지에 예측 오버레이 (HTML) |
| `python3 -m camreal.tools.make_assumed_calibration --out 파일` | 연결 확인용 가정 캘리브레이션 |
| `python3 -m camreal.evaluate --model 폴더 --dataset 폴더 --out 새폴더` | 모델별 실데이터 오차 비교 |

테스트: `PYTHONPATH="$PWD/camreal/ros2/camsim_driver:$PYTHONPATH" python3 -m pytest -q camreal`

## 이 PC의 현재 상태 (2026-09-30)

- 브랜치 `camreal` = 정태 레포 main(`264b632`, ResNet-18 + ONNX 전달) + camreal. 이전 작업은 `week3` 브랜치에 그대로 있다.
- `data/models/sim_dev`: **개발 확인용** ResNet-18. 이 PC에서 시뮬 4000장으로 1500 step(6분) 학습했고, 시뮬 검증 오차는 평균 3.8 cm다.
  수업용은 2단계의 Colab 결과로 바꾼다(`data/camreal.yaml`의 `model`).
- `data/_old_camsim/`: 이전 camsim(`6bb2f8f`) 규격의 개발 모델 `sim_test`와 `run_train` 프로젝트. 새 규격과 맞지 않아 옮겨 뒀다.
- 캘리브레이션: `data/calibration/ASSUMED_camera.yaml`(가정값, 주행 불가).
- `data/bags/run_train`은 트랙 주행이 아니라 사무실에서 기울어진 카메라로 찍은 11초 영상이다.
  흐름 확인용일 뿐이고 라벨 연습용으로 쓸 수 없다. 새 규격으로 22프레임을 다시 추출했고, 이전 프로젝트의 연습 라벨 22장을 옮겨 왔다.
- 이전 형식 설정·프로젝트(waypoint 6개용)는 `data/_week3_backup/`에 옮겨 두었다.
- f1tenth_system: `~/f1tenth_ws`에 설치·빌드했다(11개 패키지). bringup은 뜨고, VESC·Hokuyo만 하드웨어가 없어 연결 오류가 난다.
  위 연결 검증은 임시 workspace에서 했다.
- VESC·LiDAR·조이스틱은 이 PC에 연결되어 있지 않았다.
