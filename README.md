# f1tenth_gym — camsim fork

[f1tenth/f1tenth_gym](https://github.com/f1tenth/f1tenth_gym) 을 fork 해서 카메라 기반 waypoint 실습 `camsim/` 을 얹은 레포다.
시뮬레이터(`gym/f110_gym/`)는 업스트림 그대로고, 카메라 렌더링·데이터셋·학습·폐루프 검증은 전부 `camsim/` 에 있다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeongtaek1m/f1tenth_gym/blob/main/notebooks/camsim_lab.ipynb)

배지를 누르면 코랩에서 실습 노트북이 열린다. 첫 셀이 레포를 clone 하고 의존성을 설치하므로
로컬에 아무것도 깔 필요가 없다. 자세한 내용은 [camsim/README.md](camsim/README.md).

## 구성

    camsim/            렌더러, 데이터셋, 모델, 학습, 폐루프, 드라이브 전달
    camreal/           3주차 실차: 시뮬 모델 ROS 주행, rosbag 라벨링, 실데이터 데이터셋
    notebooks/         실습 노트북 (camsim/scripts/build_notebook.py 가 원본)
    gym/f110_gym/      업스트림 시뮬레이터 (수정 없음)
    examples/          맵과 중심선. 트랙 지오메트리의 출처

## 3주차 실차 (camreal)

camsim 5장에서 저장한 `model.onnx` + `checkpoint.json`을 실차 Jetson에서 그대로 쓴다(onnxruntime CUDA).

```text
주행: ros2 launch camsim_driver camsim_driver.launch.py [drive_enabled:=true]
기록: ros2 bag record ...
준비: python3 -m camreal prepare run_train
라벨: python3 -m camreal label run_train
데이터셋: python3 -m camreal export week3_real
```

실습 문서는 [camreal/README.md](camreal/README.md), 수업 준비 문서는 [camreal/INSTRUCTOR.md](camreal/INSTRUCTOR.md)다.
실차 데이터·모델(`data/`)과 결과(`out/`)는 Git에서 제외된다.

## 업스트림 인용

```
@inproceedings{okelly2020f1tenth,
  title={F1TENTH: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning},
  author={O'Kelly, Matthew and Zheng, Hongrui and Karthik, Dhruv and Mangharam, Rahul},
  booktitle={NeurIPS 2019 Competition and Demonstration Track},
  pages={77--89},
  year={2020},
  organization={PMLR}
}
```
