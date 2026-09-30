# 3주차 실차 실습 (camreal)

2주차에 시뮬로만 학습한 모델로 실차를 달려 보고, 실차 영상으로 학습용 데이터셋을 만듦.

| 단계 | 할 일 | 결과 |
|---|---|---|
| 준비 | [차량 스택 설치](CAR_STACK.md), 1주차 캘리브레이션 복사 (차마다 한 번) | 차량 스택, 주행 노드 빌드, `data/calibration/car.yaml` |
| 0 | 카메라 켜기 | `/flir_camera/image_raw` |
| 1 | 시뮬 모델로 주행 | 어디서 왜 벗어나는지 메모 |
| 2 | rosbag 기록 | `data/bags/세션이름/` |
| 3 | 프레임 추출 | `data/labeling/세션이름/` |
| 4 | 라벨링 | 프레임마다 1 m 앞 점 하나 |
| 5 | 데이터셋 만들기 | `data/datasets/이름/`, `이름.zip` |
| 6 | 실데이터로 학습 (각자) | 직접 해 보기 |

설정은 `data/camreal.yaml` 하나. 고칠 곳은 `sessions`(2단계)와 `model`(6단계)뿐이고 나머지는 그대로 둘 것.

```yaml
model: data/models/sim                   # 주행에 쓸 모델 (2주차 5장의 model.onnx + checkpoint.json)
calibration: data/calibration/car.yaml   # 1주차 카메라 캘리브레이션
image_topic: /flir_camera/image_raw
sessions:
  train: [run_train]   # 학습용 주행 (여러 개 가능)
  val: [run_val]       # 검증용 주행 (train과 다른 주행)
```

1주차 캘리브레이션 파일은 `data/calibration/car.yaml`로 복사해 둘 것 (차마다 한 번):

```bash
cd ~/f1tenth_gym
mkdir -p data/calibration
cp 1주차_캘리브레이션.yaml data/calibration/car.yaml
```

`1주차_캘리브레이션.yaml` 자리에 1주차에 만든 캘리브레이션 파일 경로를 넣을 것.

## 0. 카메라 켜기

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
ros2 launch spinnaker_camera_driver driver_node.launch.py camera_type:=blackfly_s serial:="'카메라_serial'"
```

`카메라_serial` 자리에 1주차에서 확인한 카메라 serial을 넣을 것. 새 터미널에서 영상이 들어오는지 확인:

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
ros2 topic hz /flir_camera/image_raw
```

## 1. 시뮬 모델로 주행

모델이 BEV에서 1 m 앞 점을 찍고, 차는 그 점을 따라 조향함. 속도는 0.5 m/s 고정.

### 차량 스택 켜기 (새 터미널)

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
source ~/f1tenth_ws/install/setup.bash
ros2 launch f1tenth_stack bringup_launch.py
```

### 주행 끄고 예측만 보기

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch camsim_driver camsim_driver.launch.py
```

새 터미널에서 모델이 보는 화면 띄우기:

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
ros2 run rqt_image_view rqt_image_view /camsim_driver/bev
```

- 청록 원: 후륜축에서 1 m, 자홍 점: 모델 예측
- 자홍 점이 좌우 테이프 가운데에 있으면 정상
- 점이 화면 밖이면 가장자리에 빈 원으로 표시됨
- 위쪽 상태 글자가 `valid`가 아니면 차는 정지 명령을 받음

### 주행 켜기

조교가 바퀴를 띄워 점검한 차만. 먼저 조이스틱 신호 확인 (약 20 Hz가 안 나오면 주행 켜지 말 것):

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
ros2 topic hz /joy
```

예측만 보던 노드를 Ctrl+C로 끄고, 주행을 켜서 다시 실행:

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch camsim_driver camsim_driver.launch.py drive_enabled:=true
```

- 조이스틱 RB를 누르고 있는 동안만 모델 명령대로 달림
- 멈출 땐 버튼에서 손 떼기
- 조이스틱이 끊기면 버튼 없이도 달림. 그래서 `/joy` 확인이 먼저

### 관찰 (5단계 결과와 비교할 것)

- 트랙 어디서 벗어나는지, 그때 BEV와 예측점은 어땠는지
- 조명 켜기/끄기, 손전등, 그림자, 바닥 반사에 BEV와 예측이 어떻게 바뀌는지
- 2주차 시뮬 BEV와 뭐가 다른지

## 2. rosbag 기록

1단계의 주행 노드만 끄고, 조이스틱 LB로 천천히 트랙을 돌면서 기록.

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
mkdir -p data/bags
ros2 bag record --storage sqlite3 \
  --qos-profile-overrides-path camreal/config/recording_qos.yaml \
  --output data/bags/run_train /flir_camera/image_raw /flir_camera/camera_info
```

Ctrl+C로 끝내고 확인:

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
ros2 bag info data/bags/run_train
```

- 1분에 약 5 GB. 한 번에 30~60초만
- 검증용은 따로 한 번 더 주행해서 `--output data/bags/run_val`로 기록. 한 주행을 잘라 나누지 말 것
- 조명 조건별로 따로 기록하면(예: `run_val_dark`) 5단계에서 조건별 오차 비교 가능. 새 이름은 `data/camreal.yaml`의 `sessions`에 추가

## 3. 프레임 추출

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
python3 -m camreal prepare run_train
```

- 입력: `data/bags/run_train`
- 출력: `data/labeling/run_train/` (0.5초마다 원본 영상 `raw/`와 모델 입력 BEV `bev/`)
- 다시 실행해도 기존 이미지와 라벨은 그대로
- `run_val`도 똑같이

## 4. 라벨링

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

끝나면 터미널에서 Ctrl+C. `run_val`도 똑같이.

## 5. 데이터셋 만들기

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

- 터미널에 세션별 평균, 최대 오차(cm)가 나옴. 1단계 메모와 비교
- 어떤 조건에서 오차가 컸는지, 2주차 증강으로 막을 수 있었을지 생각해 보기
- 같은 이름이 이미 있으면 `week3_real_v2`처럼 새 이름으로

## 6. (각자) 실데이터로 학습해 보기

`labels.csv`는 camsim 형식 그대로라 2주차 노트북의 `cfg`로 바로 읽힘. 실차는 위치를 몰라서 `x`, `y`, `theta`는 `nan`.

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

```python
from camsim import model, handoff
model.save(net, "model.pt", cfg)                                     # 이어 학습할 때 쓰는 파일
onnx = model.export_onnx(net, cfg, "model.onnx")                     # 차에서 돌리는 파일
handoff.export_checkpoint(onnx, "my_model", cfg, "real-finetune")    # my_model/에 model.onnx + checkpoint.json
```

두 파일을 차의 `data/models/my_model/`에 넣고 시뮬 모델과 같은 검증 데이터로 비교:

```bash
cd ~/f1tenth_gym
python3 -m camreal.evaluate --model data/models/my_model --dataset data/datasets/week3_real --out out/my_model_eval
```

주행해 보려면 `data/camreal.yaml`의 `model`을 `data/models/my_model`로 바꾸고 1단계 다시.

---

- `data/`, `out/`은 Git에 안 올라감. 옮길 땐 폴더째 복사
- 명령 옵션: `python3 -m camreal --help`
