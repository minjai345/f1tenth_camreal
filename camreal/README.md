# 3주차 실차 실습 (camreal)

2주차에 시뮬로만 학습한 모델로 실차를 달려 보고, 실차 영상으로 학습용 데이터셋을 만듦.

| 단계 | 할 일 | 결과 |
|---|---|---|
| 설치 | 패키지 설치, 차량 스택·주행 노드 빌드, 차 번호 설정 (차마다 한 번, 맨 처음) | `ROS OK`, `GPU OK` |
| 0 | 카메라 켜기, 1주차 캘리브레이션 확인 | 영상 해상도, `ost.yaml` 해상도 |
| 1 | 지면 캘리브레이션 | `data/calibration/car.yaml` |
| 2 | 예측만 보기 | `/waypoint` 확인 |
| 3 | 시뮬 모델로 주행 | 어디서 왜 벗어나는지 메모 |
| 4 | rosbag 기록 | `data/bags/세션이름/` |
| 5 | 프레임 추출 | `data/labeling/세션이름/` |
| 6 | 라벨링 | 프레임마다 1 m 앞 점 하나 |
| 7 | 데이터셋 만들기 | `data/datasets/이름/`, `이름.zip` |
| 8 | 실데이터로 학습 (각자) | 직접 해 보기 |

설정은 `data/camreal.yaml` 하나(조교가 만들어 둠). 고칠 곳은 `sessions`(4단계)와 `model`(8단계)뿐이고 나머지는 그대로 둘 것.

```yaml
model: data/models/sim                   # 주행에 쓸 모델 (2주차 5장의 model.onnx + checkpoint.json)
calibration: data/calibration/car.yaml   # 1단계 지면 캘리브레이션이 만드는 파일
image_topic: /flir_camera/image_raw
sessions:
  train: [run_train]   # 학습용 주행 (여러 개 가능)
  val: [run_val]       # 검증용 주행 (train과 다른 주행)
```

## 터미널

이 문서는 차 컴퓨터의 브라우저에서 열 것 (휴대폰으로 열면 차 터미널에 붙여넣을 수 없음).
명령 블록 위의 **T1**~**T5**는 그 명령을 붙여넣을 터미널.

| 터미널 | 쓰는 곳 |
|---|---|
| T1 카메라 | 0단계에서 켜고 끝까지 둠 |
| T2 차량 스택 | 2단계에서 켜고 끝까지 둠 |
| T3 주행 노드 | 2·3단계의 launch |
| T4 BEV 화면 | 2·3단계의 `rqt_image_view` |
| T5 그때그때 | 기록, 클릭 도구, 확인 명령(`echo`, `hz`, `rqt_graph`, `bag info`), `prepare`, `label`, `export` |

- 터미널 열기 Ctrl+Alt+T, 붙여넣기 Ctrl+Shift+V (Ctrl+V 아님), 멈추기 Ctrl+C
- T5 명령은 하나씩. 블록 하나를 붙여넣고, 계속 출력되는 명령(`hz`, `echo`, 기록, 클릭 도구 등)은 다 봤으면 Ctrl+C로 멈춘 뒤 다음 블록
- 차에 붙은 라벨: 차 번호(설치), 카메라 serial(0단계), 렌즈 높이(1단계)

## 설치 (차마다 한 번, 맨 처음)

조교가 설치해 둔 차(수업 때는 보통 이쪽)는 맨 아래 [5. 확인](#5-확인)만 하고, 결과가 다르면 손 들기.

1주차(ROS 2 Humble, 카메라 드라이버)와 2주차(onnxruntime-gpu, torch)에 깐 것 위에 아래만 더 깔면 됨. 위에서부터 순서대로, 모두 **T5**.

### 1. 레포 받기

**T5**

```bash
cd ~
git clone https://github.com/minjai345/f1tenth_camreal.git f1tenth_gym
```

이미 `~/f1tenth_gym`이 있으면 `cd ~/f1tenth_gym && git remote -v`로 주소 확인. `minjai345/f1tenth_camreal`이면 `git pull`,
다른 주소면 `mv ~/f1tenth_gym ~/f1tenth_gym_old` 후 위 명령.

### 2. ROS 패키지

1주차의 `ros-humble-desktop`에 없는 것만.

**T5**

```bash
sudo apt update
sudo apt install -y ros-humble-ackermann-msgs ros-humble-serial-driver ros-humble-asio-cmake-module \
  ros-humble-urg-node ros-humble-control-msgs ros-humble-test-msgs ros-humble-rosbridge-server \
  ros-humble-sick-scan-xd ros-humble-joy python3-colcon-common-extensions
```

`E: dpkg was interrupted`가 나오면 `sudo dpkg --configure -a` 후 다시.

### 3. Python 패키지

2주차에 이 차에서 설치했으면 이미 있음. 확인:

**T5**

```bash
python3 -c "import onnxruntime as o; print(o.get_available_providers())"
python3 -c "import torch; print(torch.__version__)"
```

첫 줄에 `CUDAExecutionProvider`가 없으면 설치 (주행에 필수).

**T5**

```bash
pip3 install "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/4eb/e6a8902dc7708/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl#sha256=4ebe6a8902dc7708434b2e1541b3fe629ebf434e16ab5537d1d6a622b42c622b"
```

둘째 줄이 `No module named 'torch'`면 설치 (7단계 데이터셋 만들기에만 필요).

**T5**

```bash
pip3 install "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl#sha256=62a1beee9f2f147076a974d2942c90060c12771c94740830327cae705b2595fc"
```

### 4. 차량 스택, 주행 노드, 차 번호

[차량 스택 설치 문서](CAR_STACK.md)의 0~6단계를 그대로 (설치 여부 확인, f1tenth_system 받기·빌드, 주행 노드 빌드, VESC 장치 이름 등록).

그다음 차마다 ROS 도메인을 나눔 ([CAR_STACK.md](CAR_STACK.md) 7단계와 같음). 수업에서는 여러 차가 한 공유기를 써서,
도메인이 같으면 다른 차의 `/drive`와 `/joy`(조이스틱)가 이 차를 움직임. 차에 붙은 차 번호(1~101)를 넣을 것.

**T5**

```bash
read -p '차 번호(1~101): ' N && echo "export ROS_DOMAIN_ID=$N" >> ~/.bashrc
```

- `차 번호(1~101):`가 나오면 숫자만 치고 Enter (예: `3`)
- 새 터미널부터 적용됨. 열려 있던 터미널은 모두 닫고 다시 열 것
- 차마다 한 번만. 잘못 넣었으면 [차량 스택 문서의 문제 해결](CAR_STACK.md#문제-해결)

### 5. 확인

**T5** (새 터미널)

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
source ~/f1tenth_ws/install/setup.bash
source install/setup.bash
python3 -c "import rclpy, cv_bridge, rosbag2_py, ackermann_msgs.msg; print('ROS OK')"
python3 -c "import onnxruntime as o; print('GPU OK' if 'CUDAExecutionProvider' in o.get_available_providers() else 'GPU 없음: 3번 다시')"
ros2 pkg list | grep -E "f1tenth_stack|camsim_driver"
echo "차 번호: ${ROS_DOMAIN_ID:-없음 (4번 다시)}"
```

`ROS OK`, `GPU OK`, `camsim_driver`, `f1tenth_stack`, 그리고 차에 붙은 차 번호가 나오면 설치 끝.

## 0. 카메라 켜기

**T1** 카메라 켜기 (1주차와 같은 명령):

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
ros2 launch spinnaker_camera_driver driver_node.launch.py camera_type:=blackfly_s serial:="'카메라_serial'"
```

`카메라_serial` 자리에 차에 붙은 라벨의 카메라 serial을 넣을 것 (따옴표는 그대로).

**T5** 영상이 들어오는지 확인 (숫자가 나오면 Ctrl+C):

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
ros2 topic hz /flir_camera/image_raw
```

**T5** 영상 해상도와 1주차 캘리브레이션 파일의 해상도:

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
ros2 topic echo /flir_camera/image_raw --once --field width
ros2 topic echo /flir_camera/image_raw --once --field height
head -2 ~/camera_calibration/ost.yaml
```

- 영상은 `1920`, `1200`이어야 함. 아니면 손 들기 (조교가 카메라 설정을 맞춤)
- `ost.yaml`의 `image_width`, `image_height`가 영상과 같으면 1단계에서 1주차 파일을 씀
- 다르거나 파일이 없다는 오류가 나오면 1단계에서 레포의 기준 파일을 씀. 1주차 72쪽에서 해상도를 1280×720으로 바꾼 뒤
  캘리브레이션했으면 다른 게 정상이고, 보통 이쪽

## 1. 지면 캘리브레이션

1주차에는 렌즈 왜곡(K, D)만 구함. 여기서는 바닥과 영상의 관계(H_i2g)를 구해서 위에서 내려다본 영상(BEV)을 만듦.
조교가 바닥에 붙여 둔 주차 칸, 후륜축 선, 마커 십자 12개를 씀.
마커 이름은 A~D줄(가까운 줄부터) + 1~3열(왼쪽부터). 예: B1 = 1 m 앞, 왼쪽 0.4 m.

### 1-1. 기록

차를 주차 칸에 맞춰 세움 (뒷바퀴 중심이 후륜축 선 위, 차가 칸과 나란하게). 조별로 순서대로.

**T5** 3초 기록하고 Ctrl+C:

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
mkdir -p data/bags
ros2 bag record --storage sqlite3 \
  --qos-profile-overrides-path camreal/config/recording_qos.yaml \
  --output data/bags/calib /flir_camera/image_raw /flir_camera/camera_info
```

다시 기록할 때나 `already exists` 오류가 나올 때는 이름을 바꿀 것 (`calib2`, 아래 클릭 도구 명령도 `calib2`로).
처음인데 이 오류가 나오면 이 차에서 누가 먼저 `calib`로 기록한 것이니 그대로 `calib2`로.

### 1-2. 마커 클릭

**T5** 클릭 도구 실행, 기준 파일로 (0단계에서 해상도가 달랐거나 1주차 파일이 없었을 때. 보통 이쪽):

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
python3 -m camreal calibrate calib --ost camreal/config/ost_reference_1920x1200.yaml
```

**T5** 클릭 도구 실행, 1주차 파일로 (0단계에서 해상도가 같았을 때만. 위 명령 대신):

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
python3 -m camreal calibrate calib
```

차 컴퓨터의 브라우저에서 http://127.0.0.1:8765 열기 (127.0.0.1 = 지금 이 컴퓨터. 휴대폰이나 다른 컴퓨터에서는 안 열림).
화면 위쪽에 쓰는 `ost.yaml`(기준 파일 / 1주차 학생 파일)과 해상도가 나옴.

| 조작 | 동작 |
|---|---|
| 영상에서 십자 중심 클릭 | 영상 위 제목에 나온 마커(예: `B2 (x 1.000 m, y 0.000 m)`)에 점이 찍히고 다음 미작업 마커로 넘어감. 찍기 전에 이름을 매번 확인 |
| 드래그 | 찍은 점 옮기기 |
| 마우스를 영상 위에 둔 채 방향키 | 선택한 점을 1 px씩 옮기기 |
| 돋보기 | 커서 옆에 4배로 보임. 보면서 찍을 것 |
| 건너뛰기 (`S`) | 영상에 안 보이는 마커 |
| 지우기 (`Del`) | 선택한 마커의 점 지우기 |
| 목록의 줄 클릭 | 그 마커 선택 (다시 찍을 때) |
| 저장 | `data/calibration/car.yaml`에 저장. 전에 있던 파일은 `data/calibration/old/`에 남음 |

- 목록 순서대로 찍으면 4번째 점까지는 BEV와 오차가 안 나옴 (`H를 계산할 수 없습니다`가 떠도 정상. A줄 셋이 한 줄이라서).
  5번째 점부터 BEV 미리보기와 오차, 6번째부터 모든 마커의 오차가 나옴. 저장은 6개부터
- calibrate를 다시 실행해도 같은 기록이면 찍은 점이 다시 나옴 (위쪽에 `이전에 찍은 점 N개를 불러옴`)

잘 됐는지 3가지 확인 (화면의 `점검 기준`):

1. 마커 오차: 그 마커를 빼고 나머지로 구한 H가 그 마커를 얼마나 맞히는지(cm). 초록 5 cm 이하,
   노랑 5~20 cm(경고, 저장은 됨), 빨강 20 cm 초과(잘못 찍힌 점이 있음, 저장 안 됨). 가까운 A·B줄은 3 cm 이하여야 함
2. 카메라 높이: 결과의 `카메라 높이`가 차에 붙은 렌즈 높이와 2 cm 안으로 맞아야 함
3. BEV 미리보기: 초록 십자(찍은 마커를 줄자 값 위치에 그린 것)가 바닥 테이프 십자 위에 겹쳐야 함

- 3가지가 다 맞을 때만 저장. 저장되면 아래에 `저장했습니다: ...`가 나옴
- 안 맞으면 저장하지 말고 T5에서 Ctrl+C. 조교가 미리 만든 캘리브레이션으로 2단계를 하면 됨. 잘못 저장했으면 손 들기 (조교가 되돌림)
- 결과의 `카메라 높이`, `pitch`는 관찰지에 적을 것 (시뮬은 높이 0.20 m, pitch 0°로 가정)
- 끝나면 T5에서 Ctrl+C

### 1단계에서 막힐 때

- 터미널에 `영상 해상도(...)와 ost.yaml(...)의 해상도(...)가 다릅니다` 또는 `ost.yaml이 없습니다` → 메시지의 방법 중
  기준 파일(`--ost ...`)만 씀. 위의 "기준 파일로" 명령
- 빨강이 있음: 한 점을 잘못 찍으면 멀쩡한 마커도 노랑·빨강이 됨. 상태 칸에 `다시 찍기`가 뜬 마커가 있으면 그것부터,
  없으면 아래 메시지에 적힌 마커를 하나씩 다시 찍기. 여러 줄에 걸쳐 많이 찍을수록 `○○ 마커가 틀린 것으로 보입니다`라고 짚어 줄 때가 많음
- `거의 같은 곳(5 px 안)에 찍은 마커` 경고: 한 십자를 두 마커로 찍은 것. 둘 중 하나를 다시 찍기
- `둘러싸지 못합니다` 노랑 메시지: 메시지에 적힌 쪽(예: `x ≥ 1 m 왼쪽`)의 마커를 더 찍기. 영상에 안 보이면 손 들기
- 차가 칸에서 1° 틀어지면 2 m 앞에서 3.5 cm 어긋남 → 차를 다시 세우고 1-1부터 (새 이름으로 기록)
- 카메라 마운트를 건드렸으면 1단계 처음부터 다시
- `설정 파일이 없습니다`, `마커 파일이 없습니다`, `기준 ost.yaml이 아직 레포에 없습니다` → 손 들기 (조교 준비물)

## 2. 예측만 보기

노드 두 개가 나눠서 일함:

- `waypoint_node`: 카메라 영상 → BEV → 모델 → 1 m 앞 점을 `/waypoint`로 보냄
- `pure_pursuit_node`: `/waypoint`를 받아 조향각을 계산해 `/drive`로 보냄. 속도는 0.5 m/s 고정

여기서는 주행을 끄고 점만 봄.

**T2** 차량 스택 켜기:

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
source ~/f1tenth_ws/install/setup.bash
ros2 launch f1tenth_stack bringup_launch.py
```

LiDAR 오류(`Error connecting to Hokuyo`)는 무시 (이 실습에서 안 씀).

**T3** 주행 끄고 주행 노드 실행:

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch camsim_driver camsim_driver.launch.py
```

`준비됨:` 줄이 두 개(waypoint_node, pure_pursuit_node) 나오면 정상. `시작하지 못했습니다` 같은 오류가 나오면 손 들기.

**T4** 모델이 보는 화면 띄우기:

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
ros2 run rqt_image_view rqt_image_view /camsim_driver/bev
```

- 청록 원: 후륜축에서 1 m, 자홍 점: 모델 예측
- 자홍 점이 좌우 테이프 가운데에 있으면 정상
- 점이 화면 밖이면 가장자리에 빈 원으로 표시됨
- 위쪽 글자: 예측 좌표(m)와 상태. `valid`가 정상. 다른 글자면 `/waypoint`가 나가지 않고 차는 멈춤 → 손 들기
- 화면이 회색: 새 화면이 안 나오는 중 (카메라가 꺼졌으면 `no image`). 위쪽 글자는 몇 초째인지와 이유. 차는 멈춤 → 손 들기

**T5** 좌표 보기 (다 봤으면 Ctrl+C):

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
ros2 topic echo /waypoint
```

**T5** 노드 연결 보기 (다 봤으면 창 닫기):

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
rqt_graph
```

- `/waypoint`의 `point.x`, `point.y`: 후륜축 기준 앞, 왼쪽 거리 (m). 2주차 CSV의 `wp_x`, `wp_y`와 같은 좌표
- `rqt_graph`: `/flir_camera/image_raw` → `waypoint_node` → `/waypoint` → `pure_pursuit_node`
- 직선 구간 가운데에 차를 똑바로 세우면 BEV에서 좌우 테이프가 평행하고 간격이 실제 차선 폭과 같아야 하고, `point.y`가 0 근처여야 함. 아니면 1단계 다시
- 차를 손으로 밀면서 직선, 코너, 한쪽으로 치우친 자세에서 자홍 점이 차선 가운데로 가는지 볼 것
- 조명도 여기서 바꿔 볼 것(불 끄기, 손전등, 그림자): BEV와 점이 어떻게 바뀌는지 관찰지에 메모. 3단계 주행은 평소 조명에서만

## 3. 시뮬 모델로 주행

조교가 바퀴를 띄워 점검한 차만. 평소 조명에서만. 트랙은 조별 순번.

**T5** 조이스틱 신호 확인 (약 20 Hz가 안 나오면 주행 켜지 말 것. 다 봤으면 Ctrl+C):

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
ros2 topic hz /joy
```

**T3** 2단계 launch를 Ctrl+C로 끄고, 주행을 켜서 다시 실행:

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch camsim_driver camsim_driver.launch.py drive_enabled:=true
```

- `준비됨:` 줄이 두 개 나와야 함. 그 전에 launch가 끝나면 위로 올려서 `시작하지 못했습니다` 줄을 찾아 손 들기
- 조이스틱은 RB만 누르고 있기. RB를 누르고 있는 동안 모델 명령대로 달림
- LB가 아닌 다른 버튼(A·B·X·Y 등)을 눌러도 똑같이 달림. 주행 중에는 RB 말고 아무 버튼도 누르지 말 것
- LB를 누른 채 스틱: 수동 운전 (자율주행보다 우선). 스틱을 끝까지 밀면 최대 약 5 m/s라 살살
- 멈추는 순서: 1) 버튼에서 손을 모두 떼기 → 2) 그래도 가면 정지 담당이 T3에서 Ctrl+C (속도 0을 보내고 꺼짐) →
  3) 그래도 안 되면 차를 들어 올리고 전원 스위치 끄기
- 정지 담당은 주행 내내 T3 앞에 있을 것
- 조이스틱이 끊기면 버튼 없이도 달림. 그래서 `/joy` 확인이 먼저
- 노드가 알아서 속도 0을 보내는 경우: 영상이 끊기거나 오래됨, 예측점이 이상함(차 뒤쪽, 3 m보다 멂, 숫자가 아님),
  새 점이 0.25초 동안 안 옴. 멈추기 전까지 마지막 점을 향해 조금 더 감 (0.5 m/s면 약 15 cm)

### 관찰 (7단계 결과와 비교할 것)

- 트랙 어디서 벗어나는지, 그때 BEV와 예측점은 어땠는지
- 2단계에서 조명을 바꿨을 때 BEV와 예측이 어떻게 바뀌었는지
- 2주차 시뮬 BEV와 뭐가 다른지, 1단계의 카메라 높이·pitch가 시뮬 가정(0.20 m, 0°)과 얼마나 다른지

## 4. rosbag 기록

먼저 T3(주행 노드)를 Ctrl+C로 끔. T2 차량 스택은 켠 채로. 조이스틱 LB를 누른 채 스틱으로 천천히 트랙을 돌면서 기록.

**T5** 기록 (끝낼 때 Ctrl+C):

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
mkdir -p data/bags
ros2 bag record --storage sqlite3 \
  --qos-profile-overrides-path camreal/config/recording_qos.yaml \
  --output data/bags/run_train /flir_camera/image_raw /flir_camera/camera_info
```

**T5** 확인:

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
ros2 bag info data/bags/run_train
```

- 1분에 약 5 GB. 한 번에 30~60초만
- 검증용은 따로 한 번 더 주행해서 `--output data/bags/run_val`로 기록. 한 주행을 잘라 나누지 말 것
- 조명 조건별로 따로 기록하면(예: `run_val_dark`) 7단계에서 조건별 오차 비교 가능. 새 이름은 `data/camreal.yaml`의 `sessions`에 추가
- 같은 이름이 이미 있으면 `already exists` 오류. 새 이름으로

## 5. 프레임 추출

**T5**

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
python3 -m camreal prepare run_train
```

- 입력: `data/bags/run_train`
- 출력: `data/labeling/run_train/` (0.5초마다 원본 영상 `raw/`와 모델 입력 BEV `bev/`)
- 다시 실행해도 기존 이미지와 라벨은 그대로
- `run_val`도 똑같이

## 6. 라벨링

**T5** (끝나면 Ctrl+C):

```bash
cd ~/f1tenth_gym
python3 -m camreal label run_train
```

브라우저에서 http://127.0.0.1:8765 열기.
BEV의 점선 원(후륜축에서 1 m)이 좌우 테이프 가운데와 만나는 곳을 클릭.

| 조작 | 동작 |
|---|---|
| 클릭, 드래그 | 원 위에 점 찍기, 옮기기 |
| Enter | 승인하고 다음 프레임 |
| X | 제외하고 다음 프레임 (가운데가 애매할 때) |
| ←, → | 이전, 다음 프레임 |
| 다음 미작업 버튼 | 아직 안 한 프레임으로 |

`run_val`도 똑같이.

## 7. 데이터셋 만들기

**T5**

```bash
cd ~/f1tenth_gym
python3 -m camreal export week3_real
```

`sessions`의 train, val에서 승인한 프레임만 모음.

| 결과 | 내용 |
|---|---|
| `data/datasets/week3_real/train/`, `val/` | camsim과 같은 형식 (`images/*.png` + `labels.csv`) |
| `data/datasets/week3_real.zip` | Colab에 올릴 압축 파일 |
| `data/datasets/week3_real/baseline/index.html` | 시뮬 모델의 실데이터 오차 (초록 정답, 자홍 예측, 오차 큰 순) |

- 터미널에 세션별 평균, 최대 오차(cm)가 나옴. 3단계 메모와 비교
- 어떤 조건에서 오차가 컸는지, 2주차 증강으로 막을 수 있었을지 생각해 보기
- 같은 이름이 이미 있으면 `week3_real_v2`처럼 새 이름으로
- `3/3` 다음에 `No module named 'torch'`가 나오면 설치 3번의 torch를 깔고 새 이름으로 다시

## 8. (각자) 실데이터로 학습해 보기

`labels.csv`는 camsim 형식 그대로라 2주차 노트북의 `cfg`로 바로 읽힘. 실차는 위치를 몰라서 `x`, `y`, `theta`는 `nan`.

**Colab**

```python
from camsim import dataset, train
ds_train = dataset.DiskDataset("week3_real/train", cfg, "all")   # zip 푼 경로
ds_val = dataset.DiskDataset("week3_real/val", cfg, "all")
net, hist = train.train(None, cfg, steps=500, batch_size=8, lr=3e-4, dataset=ds_train, val_dataset=ds_val)
```

- `train.train`은 시뮬 가중치가 아니라 처음부터 시작함 (ResNet-18은 ImageNet 가중치에서). 시뮬 가중치에서 이어 학습하거나 시뮬 데이터와 섞는 방법은 직접 설계
- 노트북 2장 데이터 생성 셀의 `DATA_DIR`를 실데이터 폴더로 두지 말 것 (시뮬 데이터로 덮어씀)
- 학습은 Colab 추천 (차 컴퓨터 메모리 약 7 GB)

2주차 5장과 같은 형식으로 저장:

**Colab**

```python
from camsim import model, handoff
model.save(net, "model.pt", cfg)                                     # 이어 학습할 때 쓰는 파일
onnx = model.export_onnx(net, cfg, "model.onnx")                     # 차에서 돌리는 파일
handoff.export_checkpoint(onnx, "my_model", cfg, "real-finetune")    # my_model/에 model.onnx + checkpoint.json
```

두 파일을 차의 `data/models/my_model/`에 넣고 시뮬 모델과 같은 검증 데이터로 비교:

**T5**

```bash
cd ~/f1tenth_gym
python3 -m camreal.evaluate --model data/models/my_model --dataset data/datasets/week3_real --out out/my_model_eval
```

주행해 보려면 `data/camreal.yaml`의 `model`을 `data/models/my_model`로 바꾸고 2단계부터 다시.

---

- `data/`, `out/`은 Git에 안 올라감. 옮길 땐 폴더째 복사
- 명령 옵션: `python3 -m camreal --help`, `python3 -m camreal calibrate --help`
