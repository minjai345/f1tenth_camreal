# 3주차 예제 데이터

조교 차(F1TENTH, FLIR Blackfly S, 렌즈 높이 약 20 cm)로 2026-10-06 실습실 트랙을 천천히(최고 1.5 m/s) 달리며 녹화한 것.
`notebooks/week3_bag_label_train.ipynb`가 씀.

| 파일 | 내용 |
|---|---|
| `run_train2_part1_2hz/` | rosbag2 (ROS 2 Humble, sqlite3). 83초 녹화 중 차선이 잘 보이는 15초를 잘라 0.5초에 한 장만 남김(영상 28장, GitHub 파일 크기 제한 100 MB 때문). 사람 얼굴은 모자이크 |
| `car.yaml` | 바닥 캘리브레이션 (`H_i2g`: 왜곡을 편 영상 → 바닥, 원점 뒷바퀴 축 가운데, x 앞, y 왼쪽, m). 곧은 차선 위 점 4개로 구함 (2026-10-05). 대충 맞춘 값 |
| `ost.yaml` | 렌즈 캘리브레이션 (K, D, 1920×1200). 체커보드 10×7, 25 mm, 재투영 오차 0.9 px. bag의 `/flir_camera/camera_info`에도 같은 값 |

bag 토픽: `/flir_camera/image_raw` (`sensor_msgs/Image`, `bayer_rggb8`, 1920×1200), `/flir_camera/camera_info`.

녹화한 명령 (차에서, README 4단계와 같음):

```bash
ros2 bag record --storage sqlite3 \
  --qos-profile-overrides-path camreal/config/recording_qos.yaml \
  --output data/bags/run_train2 /flir_camera/image_raw /flir_camera/camera_info
```
