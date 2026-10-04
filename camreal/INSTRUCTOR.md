# Camreal — 담당자 준비

학생은 [실습 문서](README.md)의 0~6단계를 따라간다. 담당자는 아래를 수업 전에 준비한다.

## 레포 구조

```text
camsim/     2주차 시뮬 실습 (정태 레포 jeongtaek1m/f1tenth_gym 그대로. 여기서 고치지 않는다)
camreal/
  __main__.py         학생 명령: prepare / label / export
  checkpoint.py       camsim 5장 산출물(model.onnx + checkpoint.json) 읽기, onnxruntime 추론
  preprocessing.py    실차 왜곡 보정 · IPM · 학습 가시 영역
  labeling/           rosbag 추출 · 브라우저 라벨링 · camsim 형식 export
  evaluate.py         실데이터 오차 (export가 시뮬 모델 기준으로 자동 실행)
  tools/              담당자 도구
  ros2/camsim_driver/ 실차 ROS 노드 (상세: ros2/camsim_driver/README.md)
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
아래는 같은 내용의 요약이다.
차가 여러 대면 차마다 `ROS_DOMAIN_ID`를 다르게 둔다(CAR_STACK.md 7단계). 같은 공유기에서 같은 ID면 다른 차의 `/drive`가
이 차를 움직이고, 주행 노드는 다른 차의 `/waypoint`·캘리브레이션 알림 때문에 속도 0에 머문다.

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

### 3. 1주차 캘리브레이션

학생이 1주차에 만든 캘리브레이션 파일을 `data/calibration/car.yaml`로 복사한다(학생 문서 맨 위).
이 파일은 [예시 형식](ros2/camsim_driver/config/calibration.example.yaml)이어야 한다.

- `K`, `D`, `new_K`는 원본 해상도(1920×1200) 기준이다.
- `H_i2g`는 **왜곡 보정된 같은 해상도 영상 → 후륜축 지면(x 전방, y 좌측, m)** 변환이다.
- `ground_frame: rear_axle`, `homography_space: undistorted_full_resolution`.

1주차 파일 형식이 다르면 샘플 하나를 받아 변환기를 추가하고, 학생 문서의 `cp`를 변환 명령으로 바꾼다.
학생이 파일을 손으로 고치게 하지 않는다.
실측 파일이 오기 전에는 가정값으로 흐름만 확인한다(주행 불가).

```bash
cd ~/f1tenth_gym
python3 -m camreal.tools.make_assumed_calibration --out data/calibration/ASSUMED_camera.yaml
```

카메라 마운트·ROI·해상도가 바뀌면 보정도 다시 한다.
실측 보정으로 바꾸면 기존 세션은 새 이름으로 다시 `prepare`한다(기존 BEV는 안 바뀜).

### 4. 설정 파일

```bash
cd ~/f1tenth_gym
cp -n camreal/config/course.yaml data/camreal.yaml          # model, calibration 경로 확인
mkdir -p data/config && cp -n camreal/ros2/camsim_driver/config/vehicle.yaml data/config/vehicle.yaml
```

`data/config/vehicle.yaml`에서 `wheelbase_m`(실측), `steer_max_rad`, `target_speed_mps`를 채운다.
같은 wheelbase를 `~/f1tenth_ws/src/f1tenth_system/f1tenth_stack/config/vesc.yaml`의 odometry에도 넣는다.
학생은 `data/camreal.yaml`에서 `sessions`와 `model`(6단계)만 바꾼다. 캘리브레이션은 경로를 고치지 않고 `data/calibration/car.yaml`로 복사해 넣는다.

### 5. pilot (수업 전 한 번)

트랙에서 차에 단 카메라로 짧게 두 번(train/val) 기록하고 학생 순서대로 끝까지 해 본다.

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
python3 -m camreal prepare run_train
python3 -m camreal label run_train          # 몇 장만 승인
python3 -m camreal export pilot             # train/val 세션에 승인 1장 이상 필요
python3 -m camreal.tools.infer_images --input data/labeling/run_train/raw --out out/pilot_sim
```

BEV에서 테이프가 평행하게 펴지고 1 m 원 위치가 실제 거리와 맞는지 확인한다. 안 맞으면 캘리브레이션부터 고친다.
주행은 [ROS 문서의 실차 확인 순서](ros2/camsim_driver/README.md#실차-확인-순서)대로 바퀴를 띄운 상태부터 시작한다.
자율주행은 조이스틱 **RB를 누르고 있는 동안** 전달된다(LB는 수동 운전).
단, **조이스틱이 연결되지 않으면(`/joy` 없음) 버튼 없이도 전달된다.** 주행 전 `ros2 topic hz /joy`를 확인한다.

## 운영 메모

- 원본 영상 기록은 1분에 약 5 GB다. 30~60초씩 기록하고, `prepare`가 끝난 bag은 필요하면 외부로 옮긴다.
- train/val은 서로 다른 주행이어야 한다. export는 같은 bag이 두 split에 들어가면 거부한다.
- 조명 조건별 세션(`run_val_bright`, `run_val_dark` 등)을 만들면 export 결과에서 조건별 오차가 나뉜다.
- 라벨 서버는 프로젝트당 하나만 띄운다. 여러 학생이 같은 세션을 동시에 편집하지 않는다.
- 실데이터 학습은 학생 몫이다. 데이터셋은 camsim `DiskDataset` 형식이고, Jetson 메모리(약 7 GB)로는 학습이
  빠듯하니 Colab을 권한다(학생 문서 6단계).

## 담당자 도구

| 명령 | 용도 |
|---|---|
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
