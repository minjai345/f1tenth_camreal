# camsim_driver (ROS 2 Humble)

노드가 둘이다. 인식 노드가 모델 좌표를 토픽으로 내고, 주행 노드가 그 토픽을 받아 조향한다.

```text
/flir_camera/image_raw ─▶ [waypoint_node] ─▶ /waypoint  (geometry_msgs/PointStamped)
                                         ├─▶ /predicted_path (nav_msgs/Path, RViz)
                                         └─▶ /camsim_driver/bev (디버그 영상)
/waypoint ─▶ [pure_pursuit_node] ─▶ /drive (ackermann_msgs/AckermannDriveStamped, drive_enabled일 때만)
                                         ─▶ ackermann_mux
```

- `waypoint_node`: FLIR `sensor_msgs/Image` → BGR8 → 왜곡 보정 → IPM BEV(학습 가시 영역으로 자름) →
  camsim 모델 → **1 m 앞 waypoint (x, y) 하나** → `/waypoint`.
- `pure_pursuit_node`: `/waypoint` → Pure Pursuit → `/drive` (25 Hz).

waypoint가 곧 Pure Pursuit 목표점이라 lookahead 설정은 없다.
lookahead는 학습 설정 `waypoints.ahead_m`(기본 1 m)이며 바꾸려면 다시 학습한다.
속도는 모델과 별개인 고정 파라미터다.

| 토픽 | 보내는 노드 | 내용 |
|---|---|---|
| `/waypoint` | waypoint_node | `header.stamp` = 원본 영상 촬영 시각, `header.frame_id` = `rear_axle`, `point.x`·`point.y` = waypoint(m), `point.z` = 0. 유효하고 신선한 예측만 |
| `/predicted_path` | waypoint_node | `nav_msgs/Path`: 후륜축 (0,0) → waypoint. `/waypoint`와 같은 stamp, 같은 조건 |
| `/camsim_driver/bev` | waypoint_node | 모델이 보는 BEV(긴 변 최대 400 px) + 1 m 원 + waypoint + 좌표·상태 글자 (기본 5 Hz). 새 BEV가 `input_timeout_s` + 1/`debug_image_hz`(기본 0.45 s) 동안 없으면(영상 끊김·버림, 추론 오류) 마지막 화면을 회색으로 바꿔 경과 초와 이유를 적어 같은 주기로 보냄 (영상이 `input_timeout_s`보다 오래 안 오면 `no image`, 영상은 오는데 추론이 멈추면 `inference stalled`) |
| `/drive` | pure_pursuit_node | `drive_enabled:=true`일 때만. `ackermann_mux` navigation 입력 |

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

모델은 onnxruntime(GPU)으로 돌린다. PyTorch는 필요 없다. onnxruntime-gpu는 package.xml로 설치하지 않는다([CAR_STACK.md](../../CAR_STACK.md) 3번).

수업에서는 여러 차가 한 공유기를 쓴다. 차마다 `ROS_DOMAIN_ID`(차 번호 1~101)를 다르게 둔다([CAR_STACK.md](../../CAR_STACK.md) 10번).
ID가 같으면 다른 차의 `/drive`와 `/joy`(조이스틱)가 이 차에 들어오고, pure_pursuit_node는 다른 차의 `/waypoint` 때문에
속도 0에 머문다.

## 설정

모델·캘리브레이션·영상 토픽은 학생 설정 `data/camreal.yaml`에서 읽는다.
차량별 값은 `config/vehicle.yaml`을 `data/config/vehicle.yaml`로 복사해 채운다.
파일은 `/**:` → `ros__parameters:` 한 섹션이고 두 노드가 같이 읽는다. 각 노드는 자기 파라미터만 선언하고 나머지는 무시한다.
그래서 이름을 틀리면(`target_speed`) 그 값은 조용히 기본값이 된다. launch는 노드를 띄우기 전에 파일을 검사해
이전 형식, 두 노드가 모르는 이름, 기본값과 형식이 다른 값(`0.33` 자리에 `1`), `drive_enabled`가 있으면 멈춘다.
`ros2 run`으로 따로 띄우면 이 검사가 없다.

```bash
cd ~/f1tenth_gym
mkdir -p data/config && cp -n camreal/ros2/camsim_driver/config/vehicle.yaml data/config/vehicle.yaml
```

노드 분리 전 형식(`camsim_driver_node:` 섹션)의 파일이면 launch가 "이전 형식의 vehicle.yaml입니다" 오류로 멈춘다.
`cp -n`은 기존 파일을 덮어쓰지 않으므로 `-n` 없이 다시 복사하고 `wheelbase_m` 등 실측값을 옮겨 적는다.

| 파라미터 | 노드 | 의미 |
|---|---|---|
| camreal_config | 둘 다 | 기본 `data/camreal.yaml` (저장소 루트 기준 상대 경로). pure_pursuit_node는 `drive_enabled`일 때 캘리브레이션 파일의 `calibration_status`만 본다 |
| waypoint_topic | 둘 다 | 기본 `/waypoint` |
| path_frame | 둘 다 | 후륜축 프레임 `rear_axle`. 다른 `frame_id`의 `/waypoint`는 거부 |
| max_waypoint_m | 둘 다 | 이보다 먼 waypoint는 무효 → 정지 (3.0) |
| future_tolerance_s | 둘 다 | ROS 시각보다 미래인 stamp를 이만큼까지 허용 (0.02 s) |
| device | waypoint_node | **cuda 권장**. 아래 성능 참고 |
| cpu_threads | waypoint_node | onnxruntime CPU 스레드 |
| input_timeout_s | waypoint_node | 예측을 낼 때 영상의 최대 나이 (0.25 s) |
| image_qos_reliability | waypoint_node | 영상 구독 QoS. 기본 `best_effort` (KEEP_LAST 1) |
| path_topic | waypoint_node | RViz 경로, 기본 `/predicted_path` |
| debug_image_topic, debug_image_hz | waypoint_node | 디버그 BEV, `''`이면 끔 |
| wheelbase_m | pure_pursuit_node | **필수 실측**. 기본 0이면 pure_pursuit_node가 시작하지 않는다(주행 켬이면 launch도 끝남) |
| steer_max_rad | pure_pursuit_node | 실제 조향 한계 |
| target_speed_mps | pure_pursuit_node | 고정 목표 속도, 기본 0.5. 수업 상한 2.0 m/s(`params.py`의 `MAX_SPEED_MPS`)를 넘으면 시작 거부 |
| waypoint_timeout_s | pure_pursuit_node | `/waypoint`의 최대 나이 (0.25 s). 넘으면 정지 |
| control_hz, drive_topic | pure_pursuit_node | 제어 주기 25 Hz, `/drive` |
| drive_enabled | pure_pursuit_node | vehicle.yaml에 두지 않는다(launch가 거부). launch 인자 `drive_enabled:=true`나 `ros2 run`의 `-p drive_enabled:=true`로만 켠다. 기본 false면 `/drive` publisher 자체가 없음 |

모든 파라미터는 시작할 때 고정된다. 바꾸려면 노드를 재시작한다.
시뮬 wheelbase 0.3302 m와 f1tenth_system odometry의 0.25 m(`vesc.yaml`의 `vesc_to_odom_node.wheelbase`)는
서로 다르며 어느 쪽도 실측값이 아니다. 실제 축간 거리를 재서 두 곳을 같이 맞춘다.

캘리브레이션 파일(`data/calibration/car.yaml`)은 `python3 -m camreal calibrate`(학생 문서 1단계)가
`ost.yaml`(1주차 학생 파일 또는 기준 파일 `camreal/config/ost_reference_1920x1200.yaml`)과 바닥 마커 클릭으로 만든다.
형식과 처리 순서 (waypoint_node):

1. `Image`를 encoding에 따라 cv_bridge로 BGR8 변환(Bayer 8-bit 포함, 16-bit 거부).
2. 캘리브레이션의 `image_width × image_height`와 정확히 같은지 확인(자동 resize/crop 없음).
3. K, D, new_K로 왜곡 보정(같은 해상도).
4. **보정된 전체 해상도 영상** 기준 `H_i2g`로 IPM. 원점은 후륜축의 지면 투영점, x 전방/y 좌측, 미터.
5. 학습 가시 영역 밖과 관측 불가 픽셀은 학습 바닥색으로 채움.

`calibration_status: assumed`(가정 캘리브레이션)이면 `drive_enabled:=true`에서 pure_pursuit_node가 시작을 거부한다.
waypoint_node는 그대로 예측·시각화를 한다.
영상 해상도가 캘리브레이션과 다르면 waypoint_node가 오류 로그(2초에 한 번)에 두 해상도와 해결 방법 3가지를 남기고 `/waypoint`를 내지 않는다.
노드는 시작할 때만 캘리브레이션 파일을 읽는다. `calibrate`로 파일을 바꾸면 launch를 다시 실행한다.

## 실행

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch camsim_driver camsim_driver.launch.py                      # 두 노드, 주행 끔: 예측·시각화만
ros2 run rqt_image_view rqt_image_view /camsim_driver/bev              # 다른 터미널
ros2 launch camsim_driver camsim_driver.launch.py drive_enabled:=true  # 주행 켬 (pure_pursuit_node에만 전달)
```

launch가 두 노드를 같은 `params_file`로 띄운다. 다른 경로의 차량 설정은 `params_file:=/절대경로/vehicle.yaml`로 지정한다.
`drive_enabled:=true`에서 pure_pursuit_node가 끝나면(시작 실패 포함) launch 전체가 끝난다. 오류는 그 위에 있다.
그때 아직 시작 중이던 waypoint_node는 traceback 없이 끝난다(`process has finished cleanly`).
노드를 따로 띄울 때 (터미널 두 개, 둘 다 저장소 루트에서. vehicle.yaml 검사는 없다. `--params-file` 뒤의 `-p`가 이긴다):

```bash
ros2 run camsim_driver waypoint_node --ros-args --params-file data/config/vehicle.yaml
ros2 run camsim_driver pure_pursuit_node --ros-args --params-file data/config/vehicle.yaml -p drive_enabled:=false # 주행 끔
ros2 run camsim_driver pure_pursuit_node --ros-args --params-file data/config/vehicle.yaml -p drive_enabled:=true  # 주행 켬
```

연결 확인 (다른 터미널):

```bash
ros2 run rqt_graph rqt_graph   # /flir_camera/image_raw → /waypoint_node → /waypoint → /pure_pursuit_node (→ /drive)
ros2 topic hz /waypoint        # 예측 주기
ros2 topic echo /waypoint      # frame_id rear_axle, point.x ≈ 1 (m), point.y는 + 왼쪽, z = 0
ros2 topic info /drive         # 주행 끔이면 Publisher count: 0 (mux가 없으면 Unknown topic), 주행 켬이면 1
```

로그: 상태가 바뀌면 남긴다(최대 1초에 한 번). waypoint_node는 `예측: valid` / 이유(`invalid waypoint output`,
`inference result expired` 등)를, 추론이 실패하면 `추론 실패, /waypoint 없음: <원인>`을,
영상 timestamp가 잘못되면 `영상 버림: invalid|stale|future|non-increasing image timestamp`를,
pure_pursuit_node는 `제어: valid` / `waypoint timeout` / `2 publishers on /waypoint` 등을 남긴다.
`non-increasing`이 계속되면 bag이나 시계가 되감긴 것이니 두 노드를 다시 시작한다.
`/camsim_driver/bev` 위쪽 글자는 영어다(OpenCV 글꼴에 한글이 없다). 설명은 로그에 있다.
RViz는 Fixed Frame `rear_axle`에 Path `/predicted_path`(또는 PointStamped `/waypoint`). 다른 frame과 같이 보려면 실측한
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
  camsim_driver는 별도 mux를 띄우지 않는다. 실제 동작(가짜 `/joy`로 확인, 2026-09-28):
  - `joy_teleop`의 `default` 명령은 **버튼이 하나도 안 눌렸을 때** 속도 0을 `/teleop`(우선순위 100)으로 보낸다.
    그 결과 `/drive`가 막힌다.
  - 그래서 RB뿐 아니라 LB가 아닌 **아무 버튼**을 눌러도 `/drive`가 통과한다.
  - **`/joy`가 없으면(조이스틱 미연결·joy 노드 종료) 버튼 없이도 `/drive`가 통과한다.**
    주행 전 `ros2 topic hz /joy`를 확인한다. 무선 조이스틱이 꺼지거나 범위를 벗어날 때의 동작은 실차에서 확인한다.
- 게인·오프셋은 `f1tenth_stack/config/vesc.yaml` 값이다. 차량마다 서보 중앙·방향을 보정한다.

## 정지 동작

`drive_enabled` 상태에서 pure_pursuit_node는 제어 주기(25 Hz)마다 `/drive`를 낸다.
최신 `/waypoint`가 신선할 때만 목표 속도와 Pure Pursuit 조향이고, 아니면 속도 0, 조향 0이다.

- 신선: 받은 뒤 경과(monotonic) ≤ `waypoint_timeout_s`, 그리고 −`future_tolerance_s` ≤ ROS 시각 − stamp ≤ `waypoint_timeout_s`.
  ROS 시각이 멈춰도(bag `--clock` 일시정지) monotonic 기준으로 멈춘다.
- 거부: NaN/Inf, x ≤ 0, `max_waypoint_m`보다 먼 점, `frame_id` ≠ `path_frame`, stamp가 이전과 같거나 과거
  (거부한 메시지의 stamp까지 포함), 받을 때 이미 오래되었거나 미래인 stamp. 거부하면 들고 있던 waypoint도 버린다(즉시 0).
- `/waypoint` publisher가 둘 이상이면(waypoint_node를 두 번 띄움, `/waypoint`가 든 bag 재생, 같은 `ROS_DOMAIN_ID`로 같은
  네트워크에 있는 다른 차) 하나만 남을 때까지 속도 0과 경고.
  강제 종료(`kill -9`)된 publisher는 DDS에서 사라질 때까지 십수 초 동안 수에 남는다.

waypoint_node는 다음 경우 `/waypoint`를 내지 않는다. 그러면 pure_pursuit_node가 timeout으로 멈춘다:
영상 끊김, 잘못되었거나 오래된 영상 timestamp, 추론 오류(해상도 불일치 포함), NaN/Inf·뒤쪽(x ≤ 0)·`max_waypoint_m`보다 먼 예측,
추론이 끝났을 때 영상이 `input_timeout_s`보다 오래됨. waypoint_node가 죽어도 같다.
그래서 마지막 정상 영상의 촬영 시각부터 `waypoint_timeout_s` + 제어 주기(0.25 + 0.04 s) 안에 속도 0이 된다.
**그 사이에는 마지막 유효 waypoint를 향해 계속 달린다**: 0.5 m/s면 최대 약 15 cm, 2 m/s면 약 60 cm.
속도를 올리면 `waypoint_timeout_s`도 줄인다(노드 분리 전 단일 노드는 잘못된 영상 timestamp·추론 오류·무효 예측이면
다음 제어 주기에 멈췄다).
mux timeout에 정지를 맡기지 않는다. 정상 입력이 돌아오면 자동 재개된다.
영상 내용은 검사하지 않는다(모델은 어떤 영상에도 점을 낸다). 차를 세우는 것은 조이스틱 버튼에서 손 떼기와 아래 종료 동작이다.

pure_pursuit_node는 Ctrl+C(launch 종료 포함), `kill`(SIGTERM), 터미널 닫힘·SSH 끊김(SIGHUP), `Ctrl+\`(SIGQUIT)에,
그리고 자기를 띄운 launch·`ros2 run`이 죽었을 때(launch에만 SIGTERM을 보내면 launch는 노드를 두고 먼저 끝난다)에도
마지막으로 속도 0, 조향 0을 한 번 보내고 구독자(mux)가 받았다고 응답할 때까지 최대 0.5 s 기다린다(부하가 크면 바로 끝나는
프로세스의 마지막 메시지를 DDS가 버렸다). 그 뒤에 오는 Ctrl+C는 무시해 launch 로그에 `process has died`가 남지 않는다.
두 노드의 실행 파일(`scripts/`)은 다른 모듈보다 먼저 이 신호들을 받아 두므로(`spin.py`), 노드 모듈(cv2·rclpy·onnxruntime)을
import하는 중의 Ctrl+C나 launch 종료에도 traceback 없이 끝난다. Python 자체가 뜨는 첫 순간의 Ctrl+C만 `KeyboardInterrupt`를
남긴다(아직 아무것도 보내기 전이다).
pure_pursuit_node 자체의 `kill -9`, OS 정지, 전원 차단에는 정지 명령을 보낼 수 없으므로 조이스틱 버튼에서 손을 떼는 것(deadman)과
차량 E-stop을 반드시 함께 확인한다. 속도 0은 목표 속도이지 즉시 제동을 보장하지 않는다.

## 성능 (이 Jetson, 1920×1200 bayer_rggb8, ResNet-18, BEV 380×300, 2026-09-30)

노드 분리 전 측정이다. 영상 처리와 추론은 waypoint_node에 그대로 있다.

| 단계 | 시간 |
|---|---:|
| 왜곡 보정 + IPM | 19 ms |
| 모델, onnxruntime CUDA | 14 ms (프레임 사이에 GPU가 쉬면 클럭이 내려가 27 ms) |
| 모델, CPU 1스레드 / 4스레드 | 373 ms / 100 ms |

bag 재생(영상 35 Hz)으로 부하를 준 상태에서 예측은 24 Hz였다. 경로 만료는 bag이 처음으로 되감길 때만 났다.
CPU로는 주행할 수 없어서 `device: cuda`가 기본이다. onnxruntime의 TensorRT 실행은 2.8 ms지만
첫 실행 때 엔진을 만드느라 1분을 기다려야 해서 쓰지 않는다.

## 실차 확인 순서

1. 카메라: `ros2 topic hz /flir_camera/image_raw`, `ros2 topic echo /flir_camera/image_raw --once --field encoding`,
   `echo $ROS_DOMAIN_ID`(차마다 다른 번호).
2. 주행 끔으로 launch → `rqt_graph`에서 `/waypoint_node → /waypoint → /pure_pursuit_node` 연결,
   `ros2 topic hz /waypoint`, `ros2 topic echo /waypoint`(직선 가운데에 똑바로 세우면 y ≈ 0),
   `/camsim_driver/bev`에서 waypoint가 차선 중앙 근처인지, 좌우 부호가 맞는지 확인. `ros2 topic info /drive`의 Publisher count는 0.
3. `ros2 launch f1tenth_stack bringup_launch.py` → 주행 켬으로 launch → `ros2 topic info /drive --verbose`로 mux 구독 확인.
4. **바퀴를 띄우고** `drive_enabled:=true` + RB로 확인한다. 노드 분리 후 처음 달리는 차는 이 단계를 반드시 다시 한다.
   - `ros2 topic hz /joy`가 약 20 Hz. 안 나오면 주행을 켜지 않는다.
   - `/ackermann_cmd`, `/commands/servo/position`의 조향 방향(양수 = 좌회전)과 한계.
   - RB를 누른 채 LB+스틱이면 `/ackermann_cmd`가 조이스틱 값으로 바뀜(수동 우선).
   - 정지(이것이 합격 기준): RB를 떼면 멈춤, 카메라 드라이버를 Ctrl+C로 끄거나 케이블을 뽑으면 속도 0(영상 끊김),
     launch를 Ctrl+C하면 속도 0으로 끝남.
   - 조이스틱을 끄는 것은 정지가 아니다. `/joy`가 없으면 버튼 없이도 `/drive`가 통과한다.
   - LB가 아닌 버튼은 어느 것이든 RB처럼 joy_teleop의 속도 0을 풀어 `/drive`를 통과시킨다. 그래서 RB만 누른다.
   - 멈추는 순서: 버튼에서 모두 손 떼기 → launch Ctrl+C(마지막 속도 0) → 차를 들어 올리고 전원 끄기.
5. 넓은 공간에서 저속(0.5 m/s)으로 시작한다.

## 하드웨어 없는 검증

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
PYTHONPATH="$PWD/camreal/ros2/camsim_driver:$PYTHONPATH" python3 -m pytest -q camreal
```

ROS가 없으면 ROS 테스트는 건너뛴다. 연결 테스트는 `/test/...` 토픽과 차 번호(1~101) 밖의 DDS domain(215~232)을 써서
켜져 있는 차량 스택에 닿지 않는다.

2026-10-05 확인 결과 (두 노드, Docker `osrf/ros:humble-desktop-full` + Python 3.10, numpy 1.24.4, OpenCV 4.5.4).
개발 PC·Docker에서 잰 시간·주기는 Jetson과 무관하므로 적지 않는다.

- 테스트 통과: FrameMailbox·WaypointFollower(만료, 잘못된 값, 시계 정지, 최신 프레임, 실패 epoch, 지연 worker,
  frame_id 불일치, 같거나 과거·미래 stamp와 원인별 이유, 거부된 메시지까지 본 stamp 순서, NaN 조향, 조향 = camsim `pure_pursuit`,
  `reason`은 마지막 명령의 이유, BEV에 그리는 이유는 ASCII), vehicle.yaml 검사(이전 형식, 모르는 이름, 형식, `drive_enabled`),
  영상이 끊기거나 추론이 실패·정지하면 BEV가 회색이 되고 경과 초와 이유를 적음(영상이 끊기면 첫 회색 화면부터 `no image`,
  `inference failure (ValueError)`, `inference stalled`), 실행 파일(`scripts/`)이 노드 모듈을 import하는 중에 SIGINT·SIGTERM을
  받으면 traceback 없이 exit code 0(신호를 받아 두기 전에 import하는 것은 `camsim_driver.spin`뿐),
  `/waypoint` 메시지 변환, 두 노드를 한 프로세스에서 띄운 연결 시험(테이프 차선 영상 → `/waypoint` → `/drive`, 영상이 끊기면 정지,
  재개, `/waypoint` publisher가 둘이면 정지, 종료 시 마지막 0), 뒤쪽 예측이면 `/waypoint` 없이 0만, 마지막 0은 받았다는
  응답까지 기다림, 실행 파일에 SIGINT 두 번·끝날 때까지 SIGINT·처리 중에 겹친 SIGINT·SIGHUP·부모 SIGKILL이면 마지막
  `/drive`가 0이고 exit code 0, 가정 캘리브레이션 주행 거부, 시작 오류가 한국어로 파라미터 이름을 말함(`target_speed_mps`
  2.0 초과, `device` 포함), 캘리브레이션 오류와 해상도 불일치가 한국어(두 해상도와 해결 방법 3가지), launch가
  `drive_enabled`를 pure_pursuit_node에만 넘기고 주행 켬이면 그 종료로 launch를 끝냄.
- `colcon build` 후 launch(가짜 카메라 + 테스트 모델): 주행 끔이면 `/drive` publisher가 없음. 주행 켬이면 유효한 waypoint에
  `/drive` 0.5 m/s, 영상을 끊으면 속도 0, 다시 주면 재개. 달리는 중에 프로세스 그룹에 SIGINT(Ctrl+C)·SIGHUP, launch에만
  SIGTERM을 보내면 모두 마지막 `/drive`가 0이고 남은 노드 없음(SIGINT는 두 노드 `process has finished cleanly`).
  `wheelbase_m: 0.0`, vehicle.yaml 오타, 주행 켬 + 가정 캘리브레이션은 각각 한국어 오류로 멈추고, 해상도가 다른 영상이면
  waypoint_node 로그에 두 해상도와 해결 방법 3가지. 주행 켬에서 `target_speed_mps: 5.0`이면 pure_pursuit_node의 한국어 오류 뒤
  launch가 끝나고, import 중이던 waypoint_node는 traceback 없이 `process has finished cleanly`.

2026-09-28 확인 결과 (노드 분리 전 단일 노드):

- 녹화 bag을 `--clock`으로 재생해 노드 → 실제 `ackermann_mux` → `ackermann_to_vesc`를 연결:
  `/drive` (0.5 m/s, −0.2206 rad) → `/ackermann_cmd` 동일 → 모터 2307 ERPM, 서보 0.7981.
  영상이 끊긴 뒤에는 모터 0, 서보 0.5304(중앙)였다.
- 2026-09-30 camsim ResNet-18·ONNX 전달로 바꾼 뒤 bag 재생 다시 확인: 노드가 CUDA로 모델을 열고 예측 24 Hz.
- 미확인: 실제 VESC·조이스틱·LiDAR(연결 없음), 실측 캘리브레이션, 물리적 정지 거리, 두 노드로 나눈 뒤의 실차 주행.
