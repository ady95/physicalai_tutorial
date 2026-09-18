"""7부 실습 — 명령 문장을 바꾸면 SmolVLA의 행동이 달라지는가

같은 장면에서 문장만 바꿔 4초 동안 실행하고, 손끝의 최종 위치와 이동 경로 길이를 비교합니다.
성공 여부가 아니라 "언어가 행동에 영향을 주는가"를 보는 실험입니다.

SmolVLA는 행동을 만들 때마다 무작위 noise를 쓰므로 비교를 두 가지로 나눕니다.
  - 문장 효과: noise seed를 고정하고 문장만 바꿨을 때 최종 위치가 얼마나 달라지는가
  - noise 효과: 문장을 고정하고 noise seed만 바꿨을 때 최종 위치가 얼마나 달라지는가
문장 효과가 noise 효과보다 뚜렷하게 커야 "언어가 행동을 바꾼다"고 말할 수 있습니다.

실행:
    MUJOCO_GL=egl python ch07/task_variation.py                       # 사전학습 모델
    MUJOCO_GL=egl python ch07/task_variation.py --checkpoint outputs/train/smolvla_so101/checkpoints/last/pretrained_model
"""

import argparse
import itertools
import os

os.environ.setdefault("SVT_LOG", "0")
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ch06.record_demos import FPS, IMG_H, IMG_W  # noqa: E402
from ch07.eval_smolvla import SMOLVLA_KEYS, load_smolvla, seed_noise  # noqa: E402
from common.robot import SO101Sim  # noqa: E402
from common.vision import SimCamera  # noqa: E402

TASKS = [
    "Pick up the red cube and put it in the blue box.",
    "Move the cube to the left.",
    "Do nothing.",
    "Raise your arm up high.",
]


def rollout(policy, preprocess, postprocess, device, task, seconds, top_camera):
    """같은 장면에서 task 문장으로 seconds 초 실행. (최종 손끝 위치, 경로 길이) 반환"""
    robot = SO101Sim(cube_pos=(0.24, 0.0), render=False)
    cams = {"top": SimCamera(robot.model, top_camera, IMG_W, IMG_H),
            "wrist": SimCamera(robot.model, "wrist_cam", IMG_W, IMG_H)}
    policy.reset()
    prev, path = robot.site_pose()[0].copy(), 0.0
    for _ in range(int(seconds * FPS)):
        state = torch.from_numpy(robot.data.qpos[:6].astype(np.float32))
        obs = {"observation.state": state.unsqueeze(0).to(device), "task": [task]}
        for k, cam in cams.items():
            rgb, _ = cam.capture(robot.data)
            obs[SMOLVLA_KEYS[k]] = (torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0).unsqueeze(0).to(device)
        with torch.inference_mode():
            a = postprocess(policy.select_action(preprocess(obs))).squeeze(0).cpu().numpy()
        robot.data.ctrl[:6] = a
        robot.step(int(round(1 / (FPS * robot.model.opt.timestep))))
        s = robot.site_pose()[0]
        path += np.linalg.norm(s - prev)
        prev = s.copy()
    final = robot.site_pose()[0].copy()
    for cam in cams.values():
        cam.close()
    robot.close()
    return final, path


def mean_pairwise(points):
    return float(np.mean([np.linalg.norm(a - b) for a, b in itertools.combinations(points, 2)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="lerobot/smolvla_base")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="비교할 noise seed 목록")
    ap.add_argument("--top-camera", default="top")
    args = ap.parse_args()
    np.set_printoptions(precision=2, suppress=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, preprocess, postprocess = load_smolvla(args.checkpoint, device)

    # 같은 문장·같은 seed를 두 번 실행해 결과가 같은지 먼저 확인한다
    seed_noise(args.seeds[0])
    f1, _ = rollout(policy, preprocess, postprocess, device, TASKS[0], args.seconds, args.top_camera)
    seed_noise(args.seeds[0])
    f2, _ = rollout(policy, preprocess, postprocess, device, TASKS[0], args.seconds, args.top_camera)
    print(f"재현 확인: 같은 문장·같은 seed 두 번 실행한 최종 위치 차이 {np.linalg.norm(f1 - f2) * 100:.2f} cm\n")

    final = {}
    print(f"{'seed':>4s}  {'명령':<52s} {f'{args.seconds:g}초 후 손끝 (x,y,z)':<24s} {'경로 길이':>8s}")
    for seed in args.seeds:
        for task in TASKS:
            seed_noise(seed)
            pos, path = rollout(policy, preprocess, postprocess, device, task, args.seconds, args.top_camera)
            final[(seed, task)] = pos
            print(f"{seed:4d}  " + f"\"{task}\"".ljust(52), f"{np.round(pos, 2)}".ljust(24), f"{path * 100:6.0f} cm")

    by_sentence = np.mean([mean_pairwise([final[(s, t)] for t in TASKS]) for s in args.seeds])
    by_noise = np.mean([mean_pairwise([final[(s, t)] for s in args.seeds]) for t in TASKS])
    print(f"\n문장만 바꿨을 때 (seed 고정) 최종 위치 차이 평균: {by_sentence * 100:5.1f} cm")
    print(f"noise만 바꿨을 때 (문장 고정) 최종 위치 차이 평균: {by_noise * 100:5.1f} cm")


if __name__ == "__main__":
    main()
