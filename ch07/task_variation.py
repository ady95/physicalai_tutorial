"""7부 실습 — 명령 문장을 바꾸면 SmolVLA 의 행동이 달라지는가

같은 장면에서 문장만 바꿔 4초 동안 실행하고, 손끝의 최종 위치와 이동 경로 길이, 행동의 평균·범위를 비교합니다.
성공 여부가 아니라 "언어가 행동에 영향을 주는가"를 보는 실험입니다.

실행:
    MUJOCO_GL=egl python ch07/task_variation.py                       # 사전학습 모델
    MUJOCO_GL=egl python ch07/task_variation.py --checkpoint outputs/train/smolvla_so101/checkpoints/last/pretrained_model
"""

import argparse
import os

os.environ.setdefault("SVT_LOG", "0")
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ch06.record_demos import FPS, IMG_H, IMG_W  # noqa: E402
from ch07.eval_smolvla import SMOLVLA_KEYS, load_smolvla  # noqa: E402
from common.robot import SO101Sim  # noqa: E402
from common.vision import SimCamera  # noqa: E402

TASKS = [
    "Pick up the red cube and put it in the blue box.",
    "Move the cube to the left.",
    "Do nothing.",
    "Raise your arm up high.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="lerobot/smolvla_base")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--top-camera", default="top")
    args = ap.parse_args()
    np.set_printoptions(precision=2, suppress=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, preprocess, postprocess = load_smolvla(args.checkpoint, device)
    print(f"{'명령':<52s} {'4초 후 손끝 (x,y,z)':<24s} {'경로 길이':>8s}  행동 범위(관절별)")
    for task in TASKS:
        robot = SO101Sim(cube_pos=(0.24, 0.0), render=False)
        cams = {"top": SimCamera(robot.model, args.top_camera, IMG_W, IMG_H),
                "wrist": SimCamera(robot.model, "wrist_cam", IMG_W, IMG_H)}
        policy.reset()
        prev, path, acts = robot.site_pose()[0].copy(), 0.0, []
        for _ in range(int(args.seconds * FPS)):
            state = torch.from_numpy(robot.data.qpos[:6].astype(np.float32))
            obs = {"observation.state": state.unsqueeze(0).to(device), "task": [task]}
            for k, cam in cams.items():
                rgb, _ = cam.capture(robot.data)
                obs[SMOLVLA_KEYS[k]] = (torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0).unsqueeze(0).to(device)
            with torch.inference_mode():
                a = postprocess(policy.select_action(preprocess(obs))).squeeze(0).cpu().numpy()
            acts.append(a)
            robot.data.ctrl[:6] = a
            robot.step(int(round(1 / (FPS * robot.model.opt.timestep))))
            s = robot.site_pose()[0]
            path += np.linalg.norm(s - prev)
            prev = s.copy()
        acts = np.array(acts)
        print(f"\"{task}\"".ljust(52), f"{np.round(robot.site_pose()[0], 2)}".ljust(24), f"{path * 100:6.0f} cm ", np.round(acts.max(0) - acts.min(0), 2))
        for cam in cams.values():
            cam.close()
        robot.close()


if __name__ == "__main__":
    main()
