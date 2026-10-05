# 차 설치 (차마다 한 번)

수업에 쓰는 차마다 한 번 함. 위에서부터 순서대로, 블록 하나씩 복사해서 실행하고 결과가 **정상**과 같으면 다음 번호로.
1주차(ROS 2 Humble, 카메라 드라이버)와 2주차(onnxruntime-gpu)를 한 차에서 시작함. 인터넷과 sudo 비밀번호가 필요함.
**실차**는 Jetson Orin Nano(JetPack 6.2.3)에서 2026-10-05에 실제로 나온 결과와 걸린 시간.

이미 설치한 차인지 모르겠으면 11번부터 해 볼 것. 모두 정상이면 5번의 첫 확인(공식 스택인지)과 4·8·9·12번만 함.
하나라도 다르면 1번부터.

| 번호 | 할 일 | 정상 |
|---|---|---|
| 1 | 레포 받기 | `~/f1tenth_gym` |
| 2 | ROS 패키지 | 오류 없이 끝남 |
| 3 | onnxruntime-gpu (주행 노드가 모델을 GPU로 돌림) | `CUDAExecutionProvider` |
| 4 | torch (7단계 데이터셋 만들기에만 필요) | `2.8.0 True` |
| 5 | 차량 스택 받기 | submodule 세 줄 |
| 6 | 차량 스택 빌드 | `11 packages finished` |
| 7 | 주행 노드 빌드 | `1 package finished` |
| 8 | VESC 장치 이름 | `/dev/sensors/vesc` |
| 9 | 조이스틱 D 모드 | `c219` |
| 10 | 차 번호 (`ROS_DOMAIN_ID`) | 새 터미널에서 차 번호 |
| 11 | 설치 확인 | `ROS OK`, `GPU OK` |
| 12 | 차량 스택 켜고 조이스틱·바퀴 확인 (바퀴를 띄우고) | LB·RB 번호, 바퀴 똑바로 |

## 1. 레포 받기

실습 코드(주행 노드, 캘리브레이션·라벨링 도구)를 `~/f1tenth_gym`에 받음.

```bash
cd ~
git clone https://github.com/minjai345/f1tenth_camreal.git f1tenth_gym
```

- 정상: `Cloning into 'f1tenth_gym'...` 다음에 오류 없이 끝남
- `already exists`가 나오면 `cd ~/f1tenth_gym && git remote -v`로 주소 확인. `minjai345/f1tenth_camreal`이면 `git pull`,
  다른 주소면 `mv ~/f1tenth_gym ~/f1tenth_gym_old` 후 위 명령
- `Username for 'https://github.com'`이 나오면 레포가 아직 비공개라는 뜻. Ctrl+C로 멈추고 공개된 뒤 다시

## 2. ROS 패키지

차량 스택 빌드와 조이스틱에 필요한 것 중 1주차의 `ros-humble-desktop`에 없는 것.

```bash
sudo apt update
sudo apt install -y ros-humble-ackermann-msgs ros-humble-serial-driver ros-humble-asio-cmake-module \
  ros-humble-urg-node ros-humble-control-msgs ros-humble-test-msgs ros-humble-rosbridge-server \
  ros-humble-sick-scan-xd ros-humble-joy python3-colcon-common-extensions
```

- 정상: 오류 없이 끝남. 이미 깔린 것은 `is already the newest version`
- `E: dpkg was interrupted`가 나오면 `sudo dpkg --configure -a` 후 다시
- 실차: 열 개 모두 이미 깔려 있었음

## 3. onnxruntime-gpu

주행 노드가 모델을 GPU로 돌릴 때 씀. 2주차에 깔았으면 이미 있음. 먼저 확인:

```bash
python3 -c "import onnxruntime as o; print(o.__version__, o.get_available_providers())"
```

- 정상: `CUDAExecutionProvider`가 들어 있음. 4번으로
- 실차: `1.23.0 ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']` (2주차에 깔려 있었음)
- `No module named 'onnxruntime'`이거나 `CUDAExecutionProvider`가 없으면 설치하고 위 확인을 다시:

```bash
pip3 install "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/4eb/e6a8902dc7708/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl#sha256=4ebe6a8902dc7708434b2e1541b3fe629ebf434e16ab5537d1d6a622b42c622b"
```

## 4. torch

7단계(데이터셋 만들기)에만 필요하고 주행에는 안 씀. 먼저 확인:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

- 정상: `2.8.0 True`. 5번으로
- `No module named 'torch'`면 설치 (실차: 내려받기 포함 2분 30초):

```bash
pip3 install "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl#sha256=62a1beee9f2f147076a974d2942c90060c12771c94740830327cae705b2595fc"
```

- 정상: 마지막 줄이 `Successfully installed ... torch-2.8.0`. 위 확인을 다시 하면 `2.8.0 True`
- `WARNING: The scripts torchfrtrace and torchrun are installed in ... which is not on PATH`는 무시 (그 명령은 안 씀)

## 5. 차량 스택 받기

F1TENTH 공식 차량 스택(f1tenth_system, humble-devel)을 `~/f1tenth_ws/src`에 받음.
VESC(모터·조향) 드라이버, 조이스틱 설정, `ackermann_mux`가 들어 있음. 먼저 이미 받았는지 확인:

```bash
git -C ~/f1tenth_ws/src/f1tenth_system remote get-url origin
```

- `https://github.com/f1tenth/f1tenth_system.git`이면 이미 받은 것. 6번으로
- `cannot change to`가 나오면 아래 명령으로 받기
- 다른 주소면 공식 스택이 아님. `mv ~/f1tenth_ws ~/f1tenth_ws_old` 후 아래 명령

```bash
mkdir -p ~/f1tenth_ws/src
cd ~/f1tenth_ws/src
git clone --recursive -b humble-devel https://github.com/f1tenth/f1tenth_system.git
git -C f1tenth_system submodule status
```

- 정상: 마지막 명령이 `ackermann_mux`, `teleop_tools`, `vesc` 세 줄을 출력하고, 줄 맨 앞이 `-`가 아님
- 실차: 5초. f1tenth_system `94cb8d7`, ackermann_mux `b3c0b08`, teleop_tools `163827a`, vesc `153998d`
- 줄 맨 앞이 `-`면 submodule을 못 받은 것: `git -C f1tenth_system submodule update --init --recursive`

## 6. 차량 스택 빌드

```bash
cd ~/f1tenth_ws
source /opt/ros/humble/setup.bash
colcon build --parallel-workers 2
```

- 정상: 마지막에 `Summary: 11 packages finished`
- 실차: 1분 13초. 함께 나오는 `6 packages had stderr output: f1tenth_stack joy_teleop key_teleop mouse_teleop vesc_ackermann vesc_driver`는
  경고(setuptools·ament 안내)라 괜찮음
- `--parallel-workers 2`는 메모리 부족 방지용

## 7. 주행 노드 빌드

3주차 주행 노드(`camsim_driver`)를 빌드함.

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
colcon build --base-paths camreal/ros2/camsim_driver --packages-select camsim_driver
```

- 정상: `Summary: 1 package finished` (실차 2초)
- `git pull`로 코드를 새로 받았으면 이 단계 다시

## 8. VESC 장치 이름

차량 스택은 VESC를 `/dev/sensors/vesc`로 찾음. 먼저 장치 규칙이 있는지 확인:

```bash
cat /etc/udev/rules.d/99-vesc.rules
```

- `sensors/vesc`가 들어간 줄이 나오면 이미 있음. 맨 아래 확인으로
- `No such file or directory`면 규칙 만들기:

```bash
echo 'KERNEL=="ttyACM[0-9]*", ACTION=="add", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5740", MODE="0666", GROUP="dialout", SYMLINK+="sensors/vesc"' | sudo tee /etc/udev/rules.d/99-vesc.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

규칙을 막 만들었으면 VESC USB를 뽑았다 다시 꽂아야 장치 이름이 생김. 배터리를 연결해 VESC가 켜지고 VESC USB가 꽂힌 상태에서 확인:

```bash
ls -l /dev/sensors/vesc
```

- 정상: `/dev/sensors/vesc -> ../ttyACM0`처럼 나옴
- 안 나오면 VESC USB를 뽑았다 다시 꽂기. 그래도 없으면 `lsusb | grep 0483:5740`으로 VESC가 보이는지 확인 (안 보이면 전원·케이블)
- 실차: 규칙은 이미 있었음
- LiDAR는 이 실습에서 안 씀

## 9. 조이스틱 D 모드

공식 차량 스택의 조이스틱 설정은 F710 뒷면 스위치가 **D**일 때에 맞춰져 있음
([F1TENTH 문서](https://github.com/f1tenth/f1tenth_doc/blob/main/getting_started/driving/drive_manual.rst)도 D로 두라고 함).
**X**면 수동 조향이 왼쪽 트리거(LT)에 붙는데, LT를 안 누른 값이 1.0이라 LB만 눌러도 바퀴가 왼쪽 끝까지 꺾임
(실차: X 상태의 `/joy`에서 셋째 축이 1.0).

1. 조이스틱 뒷면 스위치를 **D**로
2. 앞면 MODE 불이 켜져 있으면 MODE를 한 번 눌러 끔 (켜져 있으면 왼쪽 스틱과 십자키가 바뀜)
3. 확인:

```bash
lsusb | grep -i f710
```

- 정상: `046d:c219` (`F710 Gamepad [DirectInput Mode]`)
- `046d:c21f` (`F710 Wireless Gamepad [XInput Mode]`)면 아직 X. 스위치를 다시 보고 USB 수신기를 뽑았다 다시 꽂기
- 실차: `c21f`(X)로 꽂혀 있었음
- 버튼 번호는 12번에서 `/joy`로 확인

## 10. 차 번호 (`ROS_DOMAIN_ID`)

수업에서는 여러 차가 한 공유기를 씀. 도메인이 같으면 ROS 토픽이 섞여서 다른 차의 `/drive`와 `/joy`(조이스틱)가
이 차를 움직이고, 주행 노드는 `/waypoint` publisher가 둘이라며 멈춤. 차에 붙은 번호(1~101)를 넣을 것. 먼저 이미 넣었는지 확인:

```bash
grep ROS_DOMAIN_ID ~/.bashrc
```

- 아무것도 안 나오면 아래 명령
- `export ROS_DOMAIN_ID=숫자`가 나오면 이미 넣은 것. 숫자가 차 번호와 다르면 [문제 해결](#문제-해결)

```bash
read -p '차 번호(1~101): ' N && echo "export ROS_DOMAIN_ID=$N" >> ~/.bashrc
```

- `차 번호(1~101):`가 나오면 숫자만 치고 Enter (예: `3`)
- 새 터미널부터 적용됨. 열려 있던 터미널은 모두 닫고 다시 열 것
- 확인: 새 터미널에서 `echo $ROS_DOMAIN_ID`가 차 번호

노트북의 rqt·RViz로 이 차의 토픽을 볼 일이 없으면(모니터를 차에 연결해 씀) 아래도 실행. 토픽이 차 밖으로 나가지 않아
다른 차와 섞일 일이 없음. SSH로 접속해 명령을 치는 것은 괜찮음. 이것도 새 터미널부터 적용됨.

```bash
grep -q ROS_LOCALHOST_ONLY ~/.bashrc || echo 'export ROS_LOCALHOST_ONLY=1' >> ~/.bashrc
```

## 11. 설치 확인

새 터미널에서:

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
source ~/f1tenth_ws/install/setup.bash
source install/setup.bash
python3 -c "import rclpy, cv_bridge, rosbag2_py, ackermann_msgs.msg; print('ROS OK')"
python3 -c "import onnxruntime as o; print('GPU OK' if 'CUDAExecutionProvider' in o.get_available_providers() else 'GPU 없음: 차 설치 3번')"
ros2 pkg list | grep -E "f1tenth_stack|camsim_driver"
echo "차 번호: ${ROS_DOMAIN_ID:-없음 (차 설치 10번)}"
```

정상 (마지막 줄은 그 차의 번호):

```text
ROS OK
GPU OK
camsim_driver
f1tenth_stack
차 번호: 3
```

- 실차: 위 네 줄이 그대로 나옴 (차 번호는 아직 안 넣음)
- `ROS OK`가 없으면 [문제 해결](#문제-해결)의 numpy 줄과 2번, `GPU 없음`이면 3번, `f1tenth_stack`이 없으면 5·6번,
  `camsim_driver`가 없으면 1·7번, `차 번호: 없음`이면 10번

## 12. 차량 스택 켜고 조이스틱·바퀴 확인

차를 받침대에 올려 바퀴가 바닥에 닿지 않게 하고 함. 터미널 하나에서 차량 스택 켜기:

```bash
source /opt/ros/humble/setup.bash
source ~/f1tenth_ws/install/setup.bash
ros2 launch f1tenth_stack bringup_launch.py
```

- VESC가 연결 안 돼 있으면 `Failed to connect to the VESC`. 8번 확인
- LiDAR 오류(`Error connecting to Hokuyo`)는 무시

다른 터미널에서 조이스틱 버튼 확인 (Ctrl+C로 끝냄):

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /joy --field buttons
```

- 아무것도 안 누르면 `array('i', [0, 0, 0, ...])`처럼 모두 0
- LB를 누르고 있으면 다섯째 값이 1, RB를 누르고 있으면 여섯째 값이 1. 다르면 9번 확인. 그래도 다르면 이 차로 주행하지 말 것
- 아무것도 안 나오면 [문제 해결](#문제-해결)

스틱 확인 (Ctrl+C로 끝냄):

```bash
ros2 topic echo /joy --field axes
```

- 아무것도 안 만지면 모두 0 근처. 가만히 있어도 셋째 값이 1.0이면 X 모드. 9번
- 오른쪽 스틱을 왼쪽으로 밀면 셋째 값이 1 쪽(조향), 왼쪽 스틱을 위로 밀면 둘째 값이 1 쪽(속도)
- 왼쪽 스틱을 위로 밀어도 둘째 값이 안 바뀌면 MODE 불 확인(9번)

조향 방향과 바퀴 중립 (설정 파일은 `~/f1tenth_ws/src/f1tenth_system/f1tenth_stack/config/vesc.yaml`):

- LB를 누른 채 오른쪽 스틱을 왼쪽으로 밀면 앞바퀴가 왼쪽(차가 가는 방향 기준)으로 꺾여야 함. 반대면 `vesc.yaml`의
  `steering_angle_to_servo_gain` 부호 문제. 고치기 전에는 이 차로 주행하지 말 것
- 버튼을 하나도 안 누르면 조향 0이 나감. 이때 앞바퀴가 똑바로여야 함
- 틀어져 있으면 `vesc.yaml`의 `steering_angle_to_servo_offset`(공식 기본값 0.5304)을 0.01씩 바꿔 봄.
  왼쪽으로 틀어져 있으면 먼저 올려 보고 더 틀어지면 내림. 오른쪽이면 반대
- `vesc.yaml`을 고친 뒤에는 차량 스택을 Ctrl+C로 끄고 `cd ~/f1tenth_ws && colcon build --packages-select f1tenth_stack` 후 다시 켜서 봄
- 맞춘 값은 차마다 다름. 차 라벨 옆에 적어 둘 것

| 조이스틱 | 동작 |
|---|---|
| LB 누른 채 스틱 | 수동 운전 |
| RB 누르고 있는 동안 | 자율주행 명령(`/drive`) 전달 |
| 버튼에서 손 뗌 | 정지 |

- LB가 아닌 다른 버튼(A·B·X·Y 등)을 눌러도 `/drive`가 전달됨. RB만 쓸 것
- USB 수신기가 빠지거나 joy 노드가 꺼지면(`/joy`가 끊기면) 버튼 없이도 `/drive`가 전달됨. 자율주행 전에 `ros2 topic hz /joy`로
  약 20 Hz 나오는지, 위 버튼 확인에서 RB를 눌렀다 떼면 값이 바뀌는지 볼 것 (조이스틱 전원이 꺼져도 수신기가 꽂혀 있으면 `/joy`는 계속 나올 수 있음)
- 수동 최고 속도는 `vesc.yaml`과 같은 폴더의 `joy_teleop.yaml`에서 `human_control` → `drive-speed`의 `scale`(기본 5.0, 스틱 끝까지 밀면 약 5 m/s).
  바꾸면 위처럼 `f1tenth_stack`을 다시 빌드

## 문제 해결

| 증상 | 해결 |
|---|---|
| `E: dpkg was interrupted` | `sudo dpkg --configure -a` 후 2번 다시 |
| 빌드 중 `asio_cmake_module` 없음 | `sudo apt install -y ros-humble-asio-cmake-module` 후 6번 다시 |
| `Package 'f1tenth_stack' not found` | `source ~/f1tenth_ws/install/setup.bash`를 빠뜨렸거나 빌드가 중간에 멈춘 것. source 후 다시, 그래도 같으면 6번 다시 (`11 packages finished` 확인) |
| 빌드 중 `Could not find a package configuration file provided by "이름"` | 그 패키지가 없는 것. ROS 패키지 이름이면 `sudo apt install -y ros-humble-이름`(이름의 `_`는 `-`로) 후 6번 다시 |
| `package 'camsim_driver' not found` | 7번을 안 했거나 `source ~/f1tenth_gym/install/setup.bash`를 빠뜨림 |
| 빌드 중 멈춤, 메모리 부족 | 브라우저 등을 닫고 `colcon build --parallel-workers 1` |
| 3번 설치 후에도 `CUDAExecutionProvider`가 없음, `onnxruntime에 CUDA가 없습니다` | 보통 CPU용 onnxruntime이 같이 깔린 것. `pip3 uninstall -y onnxruntime onnxruntime-gpu` 후 3번 설치 다시 |
| `ROS OK` 대신 `compiled using NumPy 1.x`나 `numpy.core.multiarray failed to import` | 다른 설치가 numpy를 2.x로 올린 것. `pip3 install "numpy<2"` 후 11번 다시 |
| `No module named 'torch'` (`python3 -m camreal export`의 `3/3`에서) | 4번 후 export를 새 이름으로 다시 (예: `week3_real_v2`) |
| 4번 확인에서 `libcudss.so.0`이나 `libcusparseLt.so.0`을 못 찾는다는 오류 | 4번 휠이 아닌 다른 torch가 깔린 것(4번 휠은 두 라이브러리를 안 씀). `pip3 uninstall -y torch` 후 4번 설치 다시. 주행에는 상관없음 |
| LB만 눌러도 바퀴가 한쪽 끝까지 꺾임 | 조이스틱이 X 모드. 9번 |
| `/joy`에 아무것도 안 나옴 | 조이스틱 전원(AA 건전지)과 USB 수신기 확인, 9번. 그래도 안 되면 `sudo usermod -aG input $USER` 후 로그아웃했다 다시 로그인 |
| 다른 터미널의 토픽이 안 보임 (`ros2 topic list`에 없음) | 10번 전에 연 터미널. 닫고 새로 열 것 |
| 차 번호를 잘못 넣음, `ros2` 명령이 `invalid literal for int()`나 `ROS_DOMAIN_ID is not an integral number`로 멈춤 | `sed -i '/ROS_DOMAIN_ID/d' ~/.bashrc` 후 10번 다시, 새 터미널 |
| `/waypoint publisher가 2개입니다` | 같은 `ROS_DOMAIN_ID`를 쓰는 다른 차가 있을 수 있음. 10번 확인 |

매번 `source` 치기 싫으면 한 번만: `grep -q f1tenth_ws/install ~/.bashrc || echo 'source ~/f1tenth_ws/install/setup.bash' >> ~/.bashrc`
