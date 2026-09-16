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
| `ch05/` | 5부 | 강화학습 첫 실습, 로봇팔 RL |
| `ch06/` | 6부 | LeRobot 데이터셋, Demonstration 생성, ACT 학습·평가 |
| `ch07/` | 7부 | SmolVLA 추론·파인튜닝 |
| `ch08/` | 8부 | 최종 프로젝트 |

각 폴더의 스크립트는 저장소 루트에서 실행합니다.

```bash
python ch01/agent_3lines.py
```

## 라이선스

MIT
