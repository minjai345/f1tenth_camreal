# 차량 스택 설치 (f1tenth_system)

차를 움직이는 VESC 드라이버, 조이스틱, `ackermann_mux`와 주행 노드(`camsim_driver`)를 빌드함.
차마다 한 번. 10분 정도 걸리고 인터넷과 sudo 비밀번호가 필요함. 위에서부터 순서대로 복사해서 실행.

## 0. 이미 설치돼 있는지 확인

```bash
ls ~/f1tenth_ws/install/f1tenth_stack
```

- 폴더 내용이 나오면 차량 스택은 설치된 것. 4단계로
- `No such file or directory`가 나오면 1단계부터

## 1. 필요한 패키지 설치

```bash
sudo apt update
sudo apt install -y ros-humble-ackermann-msgs ros-humble-serial-driver ros-humble-asio-cmake-module \
  ros-humble-urg-node ros-humble-control-msgs ros-humble-test-msgs ros-humble-rosbridge-server \
  ros-humble-sick-scan-xd ros-humble-joy python3-colcon-common-extensions
```

`E: dpkg was interrupted`가 나오면 아래를 실행하고 위 명령 다시:

```bash
sudo dpkg --configure -a
```

## 2. 소스 받기

```bash
mkdir -p ~/f1tenth_ws/src
cd ~/f1tenth_ws/src
git clone --recursive -b humble-devel https://github.com/f1tenth/f1tenth_system.git
```

`--recursive`를 빼먹었으면: `cd ~/f1tenth_ws/src/f1tenth_system && git submodule update --init --recursive`

## 3. 빌드

```bash
cd ~/f1tenth_ws
source /opt/ros/humble/setup.bash
colcon build --parallel-workers 2
```

마지막에 `Summary: 11 packages finished`가 나오면 성공. `--parallel-workers 2`는 메모리 부족 방지용.

## 4. 주행 노드 빌드

```bash
cd ~/f1tenth_gym
source /opt/ros/humble/setup.bash
colcon build --base-paths camreal/ros2/camsim_driver --packages-select camsim_driver
```

- `Summary: 1 package finished`가 나오면 성공
- `git pull`로 코드를 새로 받았으면 이 단계 다시

## 5. 확인

```bash
source /opt/ros/humble/setup.bash
source ~/f1tenth_ws/install/setup.bash
ros2 pkg list | grep -E "f1tenth_stack|vesc_driver|ackermann_mux"
```

`ackermann_mux`, `f1tenth_stack`, `vesc_driver` 세 줄이 나오면 됨.

주행 노드가 GPU로 모델을 돌릴 수 있는지도 확인:

```bash
python3 -c "import onnxruntime as o; print(o.get_available_providers())"
```

`CUDAExecutionProvider`가 들어 있으면 됨. 없으면 아래 문제 해결 표.

## 6. VESC 장치 이름 등록 (차마다 한 번)

VESC를 USB로 연결한 상태에서 실행. 차량 스택은 VESC를 `/dev/sensors/vesc`로 찾음.

```bash
echo 'KERNEL=="ttyACM[0-9]*", ACTION=="add", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5740", MODE="0666", GROUP="dialout", SYMLINK="sensors/vesc"' | sudo tee /etc/udev/rules.d/99-vesc.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
ls -l /dev/sensors/vesc
```

`/dev/sensors/vesc -> ../ttyACM0`처럼 나오면 됨. 안 나오면 VESC USB를 뽑았다 다시 꽂기.
LiDAR는 이 실습에서 안 씀.

## 7. 차마다 ROS 도메인 나누기 (차마다 한 번)

수업에서는 여러 차가 한 공유기를 씀. 도메인이 같으면 ROS 토픽이 섞여서 다른 차의 `/drive`와 `/joy`(조이스틱)가
이 차를 움직이고, 주행 노드는 `/waypoint` publisher가 둘이라며 멈춤. 차에 붙은 번호(1~101)를 넣을 것.

```bash
read -p '차 번호(1~101): ' N && echo "export ROS_DOMAIN_ID=$N" >> ~/.bashrc
```

- `차 번호(1~101):`가 나오면 숫자만 치고 Enter (예: `3`)
- 새 터미널부터 적용됨. 열려 있던 터미널은 모두 닫고 다시 열 것
- 확인: 새 터미널에서 `echo $ROS_DOMAIN_ID`가 차 번호

노트북의 rqt·RViz로 이 차의 토픽을 볼 일이 없으면(모니터를 차에 연결해 씀) 아래도 실행. 토픽이 차 밖으로 나가지 않아
다른 차와 섞일 일이 없음. SSH로 접속해 명령을 치는 것은 괜찮음. 이것도 새 터미널부터 적용됨.

```bash
echo 'export ROS_LOCALHOST_ONLY=1' >> ~/.bashrc
```

## 8. 차량 스택 켜기

```bash
source /opt/ros/humble/setup.bash
source ~/f1tenth_ws/install/setup.bash
ros2 launch f1tenth_stack bringup_launch.py
```

| 조이스틱 | 동작 |
|---|---|
| LB 누른 채 스틱 | 수동 운전 |
| RB 누르고 있는 동안 | 자율주행 명령(`/drive`) 전달 |
| 버튼에서 손 뗌 | 정지 |

- 다른 버튼을 눌러도 `/drive`가 전달됨. RB만 쓸 것
- 조이스틱이 끊기면 버튼 없이도 `/drive`가 전달됨. 자율주행 전에 `ros2 topic hz /joy`로 약 20 Hz 나오는지 확인
- VESC, LiDAR가 연결 안 돼 있으면 `Failed to connect to the VESC`, `Error connecting to Hokuyo` 오류가 뜸. 나머지 노드는 정상

## 문제 해결

| 증상 | 해결 |
|---|---|
| `E: dpkg was interrupted` | `sudo dpkg --configure -a` 후 1단계 다시 |
| 빌드 중 `asio_cmake_module` 없음 | `sudo apt install -y ros-humble-asio-cmake-module` 후 3단계 다시 |
| `Package 'f1tenth_stack' not found` | 빌드가 중간에 멈춘 것. 3단계 다시 실행해서 `11 packages finished` 확인 |
| `package 'camsim_driver' not found` | 4단계를 안 했거나 `source ~/f1tenth_gym/install/setup.bash`를 빠뜨림 |
| 다른 터미널의 토픽이 안 보임 (`ros2 topic list`에 없음) | 7단계 전에 연 터미널. 닫고 새로 열 것 |
| `ros2` 명령이 `invalid literal for int()`나 `ROS_DOMAIN_ID is not an integral number`로 멈춤 | 7단계에서 숫자가 아닌 값이 들어감. `sed -i '/ROS_DOMAIN_ID/d' ~/.bashrc` 후 7단계 다시, 새 터미널 |
| `/waypoint publisher가 2개입니다` | 같은 `ROS_DOMAIN_ID`를 쓰는 다른 차가 있을 수 있음. 7단계 확인 |
| 빌드 중 멈춤, 메모리 부족 | 브라우저 등을 닫고 `colcon build --parallel-workers 1` |
| `CUDAExecutionProvider`가 없음, `onnxruntime에 CUDA가 없습니다` | `pip3 install "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/4eb/e6a8902dc7708/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl#sha256=4ebe6a8902dc7708434b2e1541b3fe629ebf434e16ab5537d1d6a622b42c622b"` |
| `No module named 'torch'` (`python3 -m camreal export`의 `3/3`에서) | torch는 데이터셋 만들기(`export`)에만 필요함. `pip3 install "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl#sha256=62a1beee9f2f147076a974d2942c90060c12771c94740830327cae705b2595fc"` 후 export를 새 이름으로 다시 (예: `week3_real_v2`) |

매번 `source` 치기 싫으면 한 번만: `echo 'source ~/f1tenth_ws/install/setup.bash' >> ~/.bashrc`
