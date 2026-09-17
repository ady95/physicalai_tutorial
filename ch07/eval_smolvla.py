"""7부 실습 — SmolVLA를 사용해보자 / 자연어로 가상 로봇을 움직여보자

SmolVLA(사전학습 모델 또는 우리 데이터로 파인튜닝한 체크포인트)에 카메라 이미지 + 관절각 + 자연어 명령을
넣어 시뮬레이션 로봇을 움직이고 성공률을 잽니다. 6부의 eval_act.py와 같은 루프이며,
SmolVLA가 기대하는 카메라 이름(camera1, camera2)과 명령 문장(task)만 다릅니다.

실행:
    MUJOCO_GL=egl python ch07/eval_smolvla.py --checkpoint lerobot/smolvla_base --episodes 5 --video
    MUJOCO_GL=egl python ch07/eval_smolvla.py --checkpoint outputs/train/smolvla_so101/checkpoints/last/pretrained_model
    MUJOCO_GL=egl python ch07/eval_smolvla.py --checkpoint ... --task "Move the red cube to the left." --episodes 3
"""

import argparse
import os

os.environ.setdefault("SVT_LOG", "0")
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ch06.eval_act import run_episode  # noqa: E402
from ch06.record_demos import TASK, random_cube  # noqa: E402

# SmolVLA 사전학습 모델은 카메라를 camera1, camera2, camera3이라는 이름으로 받는다
SMOLVLA_KEYS = {"top": "observation.images.camera1", "wrist": "observation.images.camera2"}


def seed_noise(seed):
    """SmolVLA는 행동을 만들 때마다 Flow Matching의 시작 noise를 무작위로 뽑는다.
    episode 시작 전에 난수 seed를 고정하면 같은 입력에 같은 행동이 나와 결과가 재현되고,
    문장만 바꾼 비교에서 noise 차이가 섞이지 않는다."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_smolvla(checkpoint, device):
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(checkpoint).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config, checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device)}})
    return policy, preprocess, postprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="lerobot/smolvla_base", help="Hub 이름 또는 체크포인트 폴더")
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1000, help="블록 위치를 뽑는 seed")
    ap.add_argument("--noise-seed", type=int, default=0, help="행동 생성 noise의 seed (episode 마다 +1)")
    ap.add_argument("--max-seconds", type=float, default=12.0)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--out", default="outputs/ch07_smolvla_eval.mp4")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    policy, preprocess, postprocess = load_smolvla(args.checkpoint, device)
    n_params = sum(p.numel() for p in policy.parameters()) / 1e6
    print(f"policy: SmolVLA ({n_params:.0f}M params) from {args.checkpoint}, device {device}, 로드 {time.time() - t0:.0f} s")
    print(f"chunk_size {policy.config.chunk_size}, n_action_steps {policy.config.n_action_steps}")
    print(f"task: \"{args.task}\"")

    rng = np.random.default_rng(args.seed)
    results, t0 = [], time.time()
    for i in range(args.episodes):
        cube = random_cube(rng)
        seed_noise(args.noise_seed + i)
        ok, ts = run_episode(policy, preprocess, postprocess, device, cube, max_seconds=args.max_seconds,
                             video=(args.video and i == 0), out=args.out,
                             image_keys=SMOLVLA_KEYS, task=args.task)
        results.append(ok)
        print(f"episode {i + 1:2d}  블록 ({cube[0]:+.3f}, {cube[1]:+.3f})  → "
              f"{'성공 (%.1f s)' % ts if ok else '실패'}   ({time.time() - t0:4.0f} s)")
    print(f"\n성공률 {sum(results)}/{len(results)} = {100 * sum(results) / len(results):.0f}%")


if __name__ == "__main__":
    main()
