# 담당자용 보조 도구

학생 실행 경로는 [camreal 안내](../README.md)다. 모든 명령은 저장소 루트에서 실행하며 기존 출력 폴더를 덮어쓰지 않는다.

| 도구 | 용도 |
|---|---|
| infer_images | 실제 카메라 이미지(PNG/JPG)에 모델 예측을 그려 HTML로 보기. 주행 명령은 내지 않음 |
| make_assumed_calibration | 실측 캘리브레이션이 없을 때 흐름 확인용 가정값 생성 (주행 불가) |

```bash
cd ~/f1tenth_gym
python3 -m camreal.tools.infer_images --input data/labeling/run_train/raw --out out/run_train_sim
python3 -m camreal.tools.make_assumed_calibration --width 1920 --height 1200 \
  --out data/calibration/ASSUMED_camera.yaml
```

`infer_images`는 모델·캘리브레이션을 `data/camreal.yaml`에서 읽는다. 다른 것을 쓰려면
`--model 폴더`, `--calibration 파일`로 바꾼다. 결과는 `--out` 폴더의 `index.html`에서 본다.
BEV 밖으로 나간 예측은 가장자리에 빈 원으로 표시된다.

가정 캘리브레이션은 화각 90°, 높이 0.20 m, pitch·전방 오프셋 0, 왜곡 0을 가정한다.
`--hfov-deg`, `--camera-height-m`, `--pitch-deg`, `--offset-x-m`으로 바꿀 수 있지만 실측값은 아니다.
이미지 해상도에 맞게 만들어야 하고, 이 파일로는 ROS 노드의 주행 활성화가 거부된다.

모델별 실데이터 오차 비교는 `python3 -m camreal.evaluate`를 쓴다([학생 문서 8단계](../README.md#8-각자-실데이터로-학습해-보기)).
