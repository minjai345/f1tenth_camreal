# camsim_driver (ROS 2 Humble)

노드가 둘이다. 인식 노드가 모델 좌표를 토픽으로 내고, 주행 노드가 그 토픽을 받아 조향한다.

```text
/flir_camera/image_raw ─▶ [waypoint_node] ─▶ /waypoint  (geometry_msgs/PointStamped)
                                         ├─▶ /predicted_path (nav_msgs/Path, RViz)
                                         ├─▶ /camsim_driver/bev (디버그 영상)
                                         └─▶ /camsim_driver/calibration (주행 켬인 pure_pursuit_node가 확인)
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
| `/camsim_driver/bev` | waypoint_node | 모델이 보는 BEV(긴 변 최대 400 px) + 1 m 원 + waypoint + 좌표·상태 글자 (기본 5 Hz) |
| `/camsim_driver/calibration` | waypoint_node | `std_msgs/String` `<calibration_status> <SHA-256>`: 시작할 때 읽은 캘리브레이션 파일. 한 번 보내고(transient_local) 나중에 뜬 노드도 받는다 |
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

모델은 onnxruntime(GPU)으로 돌린다. PyTorch는 필요 없다. onnxruntime-gpu는 package.xml로 설치하지 않는다([INSTRUCTOR.md](../../INSTRUCTOR.md) 1단계).

차가 여러 대면 차마다 `~/.bashrc`에 `export ROS_DOMAIN_ID=<차 번호 1~101>`을 둔다(차 밖에서 토픽을 볼 일이 없으면
`export ROS_LOCALHOST_ONLY=1`도, [CAR_STACK.md](../../CAR_STACK.md) 7단계). 같은 공유기에서 ID가 같으면 다른 차의 `/drive`가
이 차의 `ackermann_mux`로 들어가고, pure_pursuit_node는 다른 차의 `/waypoint`·캘리브레이션 알림 때문에 속도 0에 머문다.

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
| camreal_config | 둘 다 | 기본 `data/camreal.yaml` (저장소 루트 기준 상대 경로). pure_pursuit_node는 `drive_enabled`일 때 캘리브레이션 파일의 상태와 SHA-256만 본다 |
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

캘리브레이션 형식과 처리 순서 (waypoint_node):

1. `Image`를 encoding에 따라 cv_bridge로 BGR8 변환(Bayer 8-bit 포함, 16-bit 거부).
2. 캘리브레이션의 `image_width × image_height`와 정확히 같은지 확인(자동 resize/crop 없음).
3. K, D, new_K로 왜곡 보정(같은 해상도).
4. **보정된 전체 해상도 영상** 기준 `H_i2g`로 IPM. 원점은 후륜축의 지면 투영점, x 전방/y 좌측, 미터.
5. 학습 가시 영역 밖과 관측 불가 픽셀은 학습 바닥색으로 채움.

`calibration_status: assumed`(가정 캘리브레이션)이면 `drive_enabled:=true`에서 pure_pursuit_node가 시작을 거부한다.
waypoint_node는 그대로 예측·시각화를 한다.
주행 켬인 pure_pursuit_node는 waypoint_node가 `/camsim_driver/calibration`으로 알린 파일이 자기가 확인한 파일과
같고(SHA-256) assumed가 아닐 때만 달린다. 알림 전에는 속도 0(`Control: waiting for waypoint_node calibration`),
다르면 오류 로그와 속도 0이다(예: waypoint_node를 띄운 뒤 `calibrate`로 파일을 바꿈, 두 터미널의 작업 폴더가 다름).
캘리브레이션을 바꾸면 두 노드를 함께 다시 시작한다(launch 다시).

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
ros2 topic echo /camsim_driver/calibration --once   # data: measured <SHA-256>
```

로그: 상태가 바뀌면 남긴다(최대 1초에 한 번). waypoint_node는 `Waypoint: valid` / 이유(`no edges in view (edge 1.4 < 2.5)`
등. 이 값은 매 영상 바뀌므로 그동안 1초마다 남는다),
영상 timestamp가 잘못되면 `Image dropped: invalid|stale|future|non-increasing image timestamp`를,
pure_pursuit_node는 `Control: valid` / `waypoint timeout` / `2 publishers on /waypoint` 등을 남긴다.
`non-increasing`이 계속되면 bag이나 시계가 되감긴 것이니 두 노드를 다시 시작한다.
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
- waypoint_node의 캘리브레이션 알림 전이거나 파일이 다르면 속도 0 (설정 절). 같은 `ROS_DOMAIN_ID`인 다른 차의 알림도 "다름"이 된다.

waypoint_node는 다음 경우 `/waypoint`를 내지 않는다. 그러면 pure_pursuit_node가 timeout으로 멈춘다:
영상 끊김, 잘못되었거나 오래된 영상 timestamp, 추론 오류, NaN/Inf·뒤쪽(x ≤ 0)·`max_waypoint_m`보다 먼 예측,
추론이 끝났을 때 영상이 `input_timeout_s`보다 오래됨, 경계가 보이지 않는 영상(`no edges in view`, 아래. 이때는 모델을 돌리지
않는다). waypoint_node가 죽어도 같다.
그래서 마지막 정상 영상의 촬영 시각부터 `waypoint_timeout_s` + 제어 주기(0.25 + 0.04 s) 안에 속도 0이 된다.
**그 사이에는 마지막 유효 waypoint를 향해 계속 달린다**: 0.5 m/s면 최대 약 15 cm, 2 m/s면 약 60 cm.
속도를 올리면 `waypoint_timeout_s`도 줄인다(노드 분리 전 단일 노드는 잘못된 영상 timestamp·추론 오류·무효 예측이면
다음 제어 주기에 멈췄다).
mux timeout에 정지를 맡기지 않는다. 정상 입력이 돌아오면 자동 재개된다.

경계 검사(`runtime.py`의 `ImageCheck`): 모델은 검은 영상에도 점을 찍으므로, 학습 가시 영역에 테이프 폭 정도의 경계가 없으면
모델을 돌리지 않는다. 경계 = 10 cm 상자 평균 − 40 cm 상자 평균의 RMS(uint8 단계). 한 채널이라도 경계가 `MIN_EDGE`(2.5)와
그 채널의 픽셀 잡음 × `EDGE_PER_NOISE`(0.5)를 둘 다 넘으면 통과다. 렌즈를 가리면 카메라의 auto exposure·gain이 몇 초 안에 영상을
밝혀 잡음과 완만한 밝기만 남는다. 평균·표준편차로는 이것도 영상으로 보이지만 이 검사는 거른다. 값은 시뮬 렌더로 정했다
(1920×1200 가정 카메라: 트랙 약 10, 무늬 없는 손바닥·바닥 2 미만, 잡음만 있으면 잡음의 0.2~0.4배).
**정지 장치가 아니다.** 불을 꺼도 테이프가 보이면, 트랙 밖 바닥에 무늬가 있으면, 렌즈를 일부만 가리면 계속 달린다.
렌즈를 가려서 차를 세우지 않는다. 세우는 방법은 RB 떼기, launch Ctrl+C, 차량 E-stop이다.
차에서 값을 다시 정하려면 bag(가린 뒤 5초 넘게, 불 끔, 테이프 없는 바닥, 정상 트랙)을 재생해 `no edges in view (edge … < …)`
로그를 보고, 정상 트랙에서는 이 로그가 한 번도 안 나오는지 본다. 카메라 노출 설정은
`ros2 param get /flir_camera exposure_auto`, `ros2 param get /flir_camera gain_auto`로 확인한다.

pure_pursuit_node는 Ctrl+C(launch 종료 포함), `kill`(SIGTERM), 터미널 닫힘·SSH 끊김(SIGHUP), `Ctrl+\`(SIGQUIT)에,
그리고 자기를 띄운 launch·`ros2 run`이 죽었을 때(launch에만 SIGTERM을 보내면 launch는 노드를 두고 먼저 끝난다)에도
마지막으로 속도 0, 조향 0을 한 번 보내고 구독자(mux)가 받았다고 응답할 때까지 최대 0.5 s 기다린다(부하가 크면 바로 끝나는
프로세스의 마지막 메시지를 DDS가 버렸다). 그 뒤에 오는 Ctrl+C는 무시해 launch 로그에 `process has died`가 남지 않는다.
pure_pursuit_node 자체의 `kill -9`, OS 정지, 전원 차단에는 정지 명령을 보낼 수 없으므로 RB를 떼는 것(deadman)과
차량 E-stop을 반드시 함께 확인한다. 속도 0은 목표 속도이지 즉시 제동을 보장하지 않는다.

## 성능 (이 Jetson, 1920×1200 bayer_rggb8, ResNet-18, BEV 380×300, 2026-09-30)

노드 분리 전 측정이다. 영상 처리와 추론은 waypoint_node에 그대로 있다. 그 뒤에 더한 경계 검사(BEV에 상자 평균 3번)는
Jetson에서 아직 재지 않았다.

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
   - `/ackermann_cmd`, `/commands/servo/position`의 조향 방향(양수 = 좌회전)과 한계.
   - 정지(이것이 합격 기준): 카메라 드라이버를 Ctrl+C로 끄거나 케이블을 뽑으면 속도 0(영상 끊김), RB를 떼면 멈춤,
     launch를 Ctrl+C하면 속도 0으로 끝남.
   - 렌즈 가림(경계 검사 확인, 정지 방법이 아님): 손바닥·검은 천으로 **완전히** 가리고 **5초 넘게** 그대로 둔다. 그동안
     속도가 계속 0이고 `Waypoint: no edges in view`가 나와야 한다. auto exposure가 몇 초 뒤 영상을 밝히므로 처음 순간만
     보면 안 된다. 중간에 바퀴가 다시 돌면 이 카메라 설정에서는 검사가 안 듣는 것이니 메모하고 위 경계 검사 값을 bag으로 다시 정한다.
5. 넓은 공간에서 저속(0.5 m/s)으로 시작한다.

## 하드웨어 없는 검증

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
PYTHONPATH="$PWD/camreal/ros2/camsim_driver:$PYTHONPATH" python3 -m pytest -q camreal
```

ROS가 없으면 ROS 테스트는 건너뛴다. 연결 테스트는 `/test/...` 토픽과 임의의 DDS domain을 써서 켜져 있는 차량 스택에 닿지 않는다.

2026-10-04 확인 결과 (두 노드, Docker `osrf/ros:humble` + Python 3.10, numpy 1.24, OpenCV 4.5):

- 테스트 통과: FrameMailbox·WaypointFollower(만료, 잘못된 값, 시계 정지, 최신 프레임, 실패 epoch, 지연 worker,
  frame_id 불일치, 같거나 과거·미래 stamp와 원인별 이유, 거부된 메시지까지 본 stamp 순서, NaN 조향, 조향 = camsim `pure_pursuit`,
  `reason`은 마지막 명령의 이유), 경계 검사(실제 `CameraPreprocessor` + 1920×1200 가정 캘리브레이션 + Bayer 잡음:
  밝은·gain 높은·어두운 트랙은 통과, 가린 렌즈의 auto·최대 gain 잡음, 손바닥, 무늬 없는 바닥, 검은·흰 화면은 거름),
  vehicle.yaml 검사(이전 형식, 모르는 이름, 형식, `drive_enabled`), `/waypoint` 메시지 변환,
  두 노드를 한 프로세스에서 띄운 연결 시험(테이프 차선 영상 → `/waypoint` → `/drive`, 영상 중단 후 마지막 영상 stamp부터
  0.33 s 안에 정지, 재개, `/waypoint` publisher가 둘이면 정지, 종료 시 마지막 0), 검은 영상·auto gain 잡음·손바닥 그늘·뒤쪽
  예측이면 `/waypoint` 없이 0만, 캘리브레이션 알림 전·불일치·assumed면 0만, 읽는 동안 캘리브레이션이 바뀌면 시작 거부,
  마지막 0은 받았다는 응답까지 기다림, 실행 파일에 SIGINT 두 번·끝날 때까지 SIGINT·SIGHUP·부모 SIGKILL이면 마지막 `/drive`가
  0이고 exit code 0, 가정 캘리브레이션 주행 거부, 시작 오류가 한국어로 파라미터 이름을 말함(`target_speed_mps` 2.0 초과 포함),
  launch가 `drive_enabled`를 pure_pursuit_node에만 넘기고 주행 켬이면 그 종료로 launch를 끝냄.
- 부하(개발 PC 12코어에 바쁜 프로세스 24개)에서 SIGINT 두 번 시험 15회: 마지막 0 15회 모두 도착(응답 대기 없이는 15회 중 1회 놓침).
- `colcon build` 후 launch: 가짜 영상 20 Hz → `/waypoint`(rear_axle, stamp = 영상 stamp) → `/drive` 0.5 m/s.
  영상을 끊으면 마지막 영상 stamp 기준 약 0.25 s 뒤 속도 0. Ctrl+C하면 마지막 `/drive`가 0.
  각 노드는 vehicle.yaml의 다른 노드 파라미터를 무시한다(`ros2 param get`으로 확인).
  Ctrl+C, launch에만 SIGTERM, 프로세스 그룹에 SIGHUP 모두 마지막 `/drive`가 0이고 남은 노드 없음.
  가짜 영상을 검게 바꾸면 마지막 정상 waypoint stamp부터 0.28 s 뒤 속도 0, 되돌리면 재개.
  경계 검사로 바꾼 뒤 다시(주행 켬, 640×400 가짜 영상 약 15 Hz): 렌즈 가림 auto gain(평균 20, 표준편차 10)·최대 gain(40, 40)·
  손바닥 그늘·검은 영상 각 2~3초 동안 0.5 s 뒤 `/drive`가 모두 0이고 `/waypoint` 없음, 트랙 영상으로 돌아오면 재개.
  이전 판정(평균·표준편차)은 같은 영상에서 계속 달렸다(제어 64회 중 51회 0.5 m/s 등). 프로세스 그룹에 SIGINT 두 번이면 두 노드
  모두 `process has finished cleanly`(이전에는 `process has died ... exit code -2`).
  `wheelbase_m: 0.0` + 주행 켬이면 한국어 오류 뒤 launch가 끝나고, 주행 끔이면 인식만 계속. vehicle.yaml 오타는 시작 전에 거부.

2026-09-28 확인 결과 (노드 분리 전 단일 노드):

- 녹화 bag을 `--clock`으로 재생해 노드 → 실제 `ackermann_mux` → `ackermann_to_vesc`를 연결:
  `/drive` (0.5 m/s, −0.2206 rad) → `/ackermann_cmd` 동일 → 모터 2307 ERPM, 서보 0.7981.
  영상이 끊긴 뒤에는 모터 0, 서보 0.5304(중앙)였다.
- 2026-09-30 camsim ResNet-18·ONNX 전달로 바꾼 뒤 bag 재생 다시 확인: 노드가 CUDA로 모델을 열고 예측 24 Hz.
- 미확인: 실제 VESC·조이스틱·LiDAR(연결 없음), 실측 캘리브레이션, 물리적 정지 거리, 두 노드로 나눈 뒤의 실차 주행.
