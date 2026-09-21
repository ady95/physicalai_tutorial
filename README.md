# 《로봇 없이 시작하는 Physical AI 따라하기》 예제 코드

위키독스 책 [로봇 없이 시작하는 Physical AI 따라하기](https://wikidocs.net/book/21363)의 예제 코드 저장소입니다.
실물 로봇 없이 MuJoCo 시뮬레이션, OpenCV·YOLO, Hugging Face LeRobot(ACT, SmolVLA)만으로
Physical AI의 전체 파이프라인을 완주합니다.

## 준비

- Ubuntu 22.04 LTS 이상 (Windows는 WSL2, macOS는 CPU 경로)
- Python 3.12 (LeRobot 0.6 이상이 3.12를 요구합니다. uv sync 가 자동으로 내려받습니다)
- [uv](https://docs.astral.sh/uv/) (Python 패키지·가상환경 관리자)
- NVIDIA GPU는 6부(ACT)·7부(SmolVLA) 학습 실습에 필요합니다. 없으면 Google Colab 경로를 사용합니다.

```bash
git clone https://github.com/ady95/physicalai_tutorial.git
cd physicalai_tutorial
uv sync            # .venv 생성 + 고정된 버전으로 패키지 설치
source .venv/bin/activate
```

## 폴더 구성

| 폴더 | 책의 장 | 내용 |
|---|---|---|
| `ch01/` | 1부 | AI 없는 가상 Agent, 랜덤 Agent |
| `ch02/` | 2부 | 개발환경 확인, MuJoCo 첫 실행, 블록 떨어뜨리기, 로봇팔 관절 움직이기 |
| `ch03/` | 3부 | 좌표계·Pose, FK/IK, Rule 기반 Pick and Place |
| `ch04/` | 4부 | OpenCV, YOLO, 가상 카메라, Vision + Robot 연결 |
| `ch05/` | 5부 | 강화학습: 점 환경 PPO, 로봇팔 도달 환경과 Reward 비교 |
| `ch06/` | 6부 | LeRobot 데이터셋 조회, 시연 데이터셋 기록, ACT 평가 |
| `ch07/` | 7부 | SmolVLA 평가(사전학습 / 파인튜닝), 명령 문장 변화 실험 |
| `ch08/` | 8부 | 3색 블록 데이터셋·색 지정 평가, 언어 조건 진단, 환경 변형 실험 |
| `ch09/` | 9부 (선택) | LLM Planner: 로봇 도구 계층, 도구 호출 루프, 색 지정 과제 비교 |
| `common/` | 공통 | 장면(scene), 로봇 래퍼, IK, 가상 카메라, Menagerie 다운로드 |
| `scripts_run_all.sh` | 검증 | 1~4부(WITH_RL=1 이면 5부까지) 예제를 순서대로 실행해 로그 저장 |
| `scripts_train_all.sh` | 6~7부 | 데이터셋 기록 → ACT 학습·평가 → SmolVLA 평가·파인튜닝·평가 |
| `scripts_ch08.sh` | 8부 | 환경 변형, 일반화, 3색 블록 실험 |
| `scripts_ch09.sh` | 9부 (선택) | LLM Planner 실측. 외부 LLM API 키와 요금이 필요 |
| `notebooks/` | 공통 | GPU 없는 독자를 위한 Colab 노트북 |

9부는 선택 과정입니다. GPU도 LeRobot도 PyTorch도 쓰지 않는 대신 외부 LLM API를 호출하므로, `uv pip install openai` 와 환경변수 `OPENAI_API_KEY`(필요하면 `OPENAI_MODEL`, `OPENAI_BASE_URL`)가 필요합니다.

각 폴더의 스크립트는 저장소 루트에서 실행합니다. MuJoCo 렌더링이 필요한 스크립트는 Linux + NVIDIA에서 환경변수 MUJOCO_GL=egl 이 필요합니다.

```bash
python ch01/agent_3lines.py
MUJOCO_GL=egl python ch03/pick_and_place.py
```

학습 명령(ACT, SmolVLA)은 `scripts_train_all.sh` 와 `scripts_ch08.sh` 안에 있습니다. 6부·7부 학습에는 NVIDIA GPU(VRAM 8 GB 이상)가 필요하며, 책의 부록 D에 GPU가 없을 때의 경로가 있습니다.

## GPU가 없다면 (Colab)

`notebooks/colab_act_train.ipynb` 가 데이터셋 만들기 → ACT 학습 → 평가 → SmolVLA 파인튜닝을 Colab에서 한 번에 진행합니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ady95/physicalai_tutorial/blob/main/notebooks/colab_act_train.ipynb)

## 학습 체크포인트

6~8부에서 필자가 학습한 ACT·SmolVLA 체크포인트는 [릴리스 페이지](https://github.com/ady95/physicalai_tutorial/releases/tag/v0.1-checkpoints)에 있습니다.
GPU가 없거나 학습을 건너뛰고 싶을 때 내려받아 평가 스크립트(`ch06/eval_act.py`, `ch07/eval_smolvla.py`)에 바로 넣을 수 있습니다.

## 라이선스

MIT
