# camsim_driver (ROS 2 Humble)

FLIR `sensor_msgs/Image` → BGR8 → 왜곡 보정 → IPM BEV(학습 가시 영역으로 자름) →
camsim 모델 → **1 m 앞 waypoint (x, y) 하나** → Pure Pursuit → `/drive`.
노드/실행 파일 이름은 `camsim_driver_node`다. waypoint가 곧 Pure Pursuit 목표점이라 lookahead 설정은 없다.
lookahead는 학습 설정 `waypoints.ahead_m`(기본 1 m)이며 바꾸려면 다시 학습한다.
속도는 모델과 별개인 고정 파라미터다.

| 출력 | 내용 |
|---|---|
| `/predicted_path` | `nav_msgs/Path`: 후륜축 (0,0) → waypoint. 원본 영상 timestamp 유지 |
| `/camsim_driver/bev` | 모델이 보는 BEV(긴 변 최대 400 px) + 1 m 원 + waypoint + 상태 글자 (기본 5 Hz) |
| `/drive` | `drive_enabled:=true`일 때만. `ackermann_mux` navigation 입력 |

## 빌드

저장소 전체가 필요하다. 설치 시 `camsim/`, `camreal/` 코드를 **복사**하므로 두 폴더를 고치면 다시 빌드한다.
Gym·pyglet·시뮬레이터는 필요 없다.

```bash
sudo apt install ros-humble-ackermann-msgs ros-humble-cv-bridge python3-colcon-common-extensions
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
colcon build --base-paths camreal/ros2/camsim_driver --packages-select camsim_driver
source install/setup.bash
```

모델은 onnxruntime(GPU)으로 돌린다. PyTorch는 필요 없다. onnxruntime-gpu는 package.xml로 설치하지 않는다([INSTRUCTOR.md](../../INSTRUCTOR.md) 1단계).

## 설정

모델·캘리브레이션·영상 토픽은 학생 설정 `data/camreal.yaml`에서 읽는다.
차량별 값은 `config/vehicle.yaml`을 `data/config/vehicle.yaml`로 복사해 채운다.

| 파라미터 | 의미 |
|---|---|
| camreal_config | 기본 `data/camreal.yaml` (저장소 루트 기준 상대 경로) |
| device | **cuda 권장**. 아래 성능 참고 |
| wheelbase_m | **필수 실측**. 기본 0은 시작 실패 |
| steer_max_rad | 실제 조향 한계 |
| target_speed_mps | 고정 목표 속도, 기본 0.5 |
| input_timeout_s, path_timeout_s | 영상·예측의 최대 나이 (0.25 s) |
| max_waypoint_m | 이보다 먼 예측은 무효 → 정지 (3.0) |
| debug_image_topic, debug_image_hz | 디버그 BEV, `''`이면 끔 |
| path_frame | 후륜축 프레임 `rear_axle` |
| drive_enabled | launch 인자 `drive_enabled:=true`가 우선. 기본 false면 `/drive` publisher 자체가 없음 |

모든 파라미터는 시작할 때 고정된다. 바꾸려면 노드를 재시작한다.
시뮬 wheelbase 0.3302 m와 f1tenth_system odometry의 0.25 m(`vesc.yaml`의 `vesc_to_odom_node.wheelbase`)는
서로 다르며 어느 쪽도 실측값이 아니다. 실제 축간 거리를 재서 두 곳을 같이 맞춘다.

캘리브레이션 형식과 처리 순서:

1. `Image`를 encoding에 따라 cv_bridge로 BGR8 변환(Bayer 8-bit 포함, 16-bit 거부).
2. 캘리브레이션의 `image_width × image_height`와 정확히 같은지 확인(자동 resize/crop 없음).
3. K, D, new_K로 왜곡 보정(같은 해상도).
4. **보정된 전체 해상도 영상** 기준 `H_i2g`로 IPM. 원점은 후륜축의 지면 투영점, x 전방/y 좌측, 미터.
5. 학습 가시 영역 밖과 관측 불가 픽셀은 학습 바닥색으로 채움.

`calibration_status: assumed`(가정 캘리브레이션)이면 `drive_enabled:=true`가 거부된다.

## 실행

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch camsim_driver camsim_driver.launch.py                      # 주행 끔: 예측·시각화만
ros2 run rqt_image_view rqt_image_view /camsim_driver/bev              # 다른 터미널
ros2 launch camsim_driver camsim_driver.launch.py drive_enabled:=true  # 주행 켬
```

다른 경로의 차량 설정은 `params_file:=/절대경로/vehicle.yaml`로 지정한다.
RViz는 Fixed Frame `rear_axle`에 Path `/predicted_path`. 다른 frame과 같이 보려면 실측한
`base_link → rear_axle` 정적 TF를 발행한다(이미 있는 TF와 중복 금지, 임의의 0 오프셋 금지).

```bash
ros2 run tf2_ros static_transform_publisher --x "$REAR_X" --y 0 --z 0 \
  --roll 0 --pitch 0 --yaw 0 --frame-id base_link --child-frame-id rear_axle
```

## f1tenth_system 연결 (humble-devel, 2026-09-28 확인)

```text
/drive ─▶ ackermann_mux (navigation, priority 10, timeout 0.2 s) ─▶ /ackermann_cmd
/teleop ─▶ ackermann_mux (joystick,  priority 100)                        │
                                                                          ▼
          ackermann_to_vesc: /commands/motor/speed  = 4614 × 속도(m/s)
                             /commands/servo/position = -1.2135 × 조향(rad) + 0.5304  ─▶ vesc_driver
```

- mux 출력 토픽은 **`/ackermann_cmd`**다. bringup의 `ackermann_cmd_out → ackermann_drive` remap은
  존재하지 않는 이름이라 효과가 없다. 확인은 `ros2 topic echo /ackermann_cmd`로 한다.
- 조이스틱(Logitech F-710): **LB = 수동 운전 deadman, RB = 자율주행(navigation) deadman**.
  이 노드는 별도 mux를 띄우지 않는다. 실제 동작(가짜 `/joy`로 확인, 2026-09-28):
  - `joy_teleop`의 `default` 명령은 **버튼이 하나도 안 눌렸을 때** 속도 0을 `/teleop`(우선순위 100)으로 보낸다.
    그 결과 `/drive`가 막힌다.
  - 그래서 RB뿐 아니라 LB가 아닌 **아무 버튼**을 눌러도 `/drive`가 통과한다.
  - **`/joy`가 없으면(조이스틱 미연결·joy 노드 종료) 버튼 없이도 `/drive`가 통과한다.**
    주행 전 `ros2 topic hz /joy`를 확인한다. 무선 조이스틱이 꺼지거나 범위를 벗어날 때의 동작은 실차에서 확인한다.
- 게인·오프셋은 `f1tenth_stack/config/vesc.yaml` 값이다. 차량마다 서보 중앙·방향을 보정한다.

## 정지 동작

`drive_enabled` 상태에서 다음 경우 제어 주기(25 Hz)마다 속도 0, 조향 0을 발행한다:
영상 끊김·오래된 timestamp, 추론 오류, NaN/Inf, 뒤쪽(x ≤ 0)이나 `max_waypoint_m`보다 먼 waypoint,
예측이 `path_timeout_s`보다 오래됨. mux timeout에 정지를 맡기지 않는다. 정상 입력이 돌아오면 자동 재개된다.
프로세스 강제 종료·OS 정지에는 노드가 정지 명령을 보낼 수 없으므로 RB를 떼는 것(deadman)과
차량 E-stop을 반드시 함께 확인한다. 속도 0은 목표 속도이지 즉시 제동을 보장하지 않는다.

## 성능 (이 Jetson, 1920×1200 bayer_rggb8, ResNet-18, BEV 380×300, 2026-09-30)

| 단계 | 시간 |
|---|---:|
| 왜곡 보정 + IPM | 19 ms |
| 모델, onnxruntime CUDA | 14 ms (프레임 사이에 GPU가 쉬면 클럭이 내려가 27 ms) |
| 모델, CPU 1스레드 / 4스레드 | 373 ms / 100 ms |

bag 재생(영상 35 Hz)으로 부하를 준 상태에서 예측은 24 Hz였다. 경로 만료는 bag이 처음으로 되감길 때만 났다.
CPU로는 주행할 수 없어서 `device: cuda`가 기본이다. onnxruntime의 TensorRT 실행은 2.8 ms지만
첫 실행 때 엔진을 만드느라 1분을 기다려야 해서 쓰지 않는다.

## 실차 확인 순서

1. 카메라: `ros2 topic hz /flir_camera/image_raw`, `ros2 topic echo /flir_camera/image_raw --once --field encoding`.
2. 주행 끔으로 노드 실행 → `/camsim_driver/bev`에서 waypoint가 차선 중앙 근처인지, 좌우 부호가 맞는지 확인.
3. `ros2 launch f1tenth_stack bringup_launch.py` → `ros2 topic info /drive --verbose`로 mux 구독 확인.
4. **바퀴를 띄우고** `drive_enabled:=true` + RB → `/ackermann_cmd`, `/commands/servo/position`의
   조향 방향(양수 = 좌회전)과 한계, 카메라를 가렸을 때 속도 0이 반복되는지 확인.
5. 넓은 공간에서 저속(0.5 m/s)으로 시작한다.

## 하드웨어 없는 검증

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
PYTHONPATH="$PWD/camreal/ros2/camsim_driver:$PYTHONPATH" python3 -m pytest -q camreal
```

2026-09-28 확인 결과:

- 테스트 통과: 전처리·예측 일치, 조향 부호·크기, 입력 끊김, 오래된 경로, 지연 worker, 비정상 출력 정지, ROS 메시지.
- 녹화 bag을 `--clock`으로 재생해 노드 → 실제 `ackermann_mux` → `ackermann_to_vesc`를 연결:
  `/drive` (0.5 m/s, −0.2206 rad) → `/ackermann_cmd` 동일 → 모터 2307 ERPM, 서보 0.7981.
  영상이 끊긴 뒤에는 모터 0, 서보 0.5304(중앙)였다.
- 2026-09-30 camsim ResNet-18·ONNX 전달로 바꾼 뒤 bag 재생 다시 확인: 노드가 CUDA로 모델을 열고 예측 24 Hz.
- 미확인: 실제 VESC·조이스틱·LiDAR(연결 없음), 실측 캘리브레이션, 물리적 정지 거리.
