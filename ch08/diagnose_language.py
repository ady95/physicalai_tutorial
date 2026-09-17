"""8부 프로젝트 1 진단 — 명령한 색의 블록 쪽으로 손이 가는가

집기 성공 여부와 별개로, 세 블록 중 손끝이 가장 가까이 간 블록이 명령한 색과 일치하는 비율을 잽니다.
문장이 대상 선택에 아무 영향도 주지 않으면 이 비율은 33% 근처에 머뭅니다.

SmolVLA는 행동을 만들 때마다 무작위 noise를 쓰므로, 한 배치에서 세 문장을 시킬 때 noise seed를 같게 고정합니다.
그래야 세 실행의 차이가 문장에서만 옵니다. 배치마다 seed를 바꿔 여러 배치의 결과를 모읍니다.

실행:
    MUJOCO_GL=egl python ch08/diagnose_language.py --checkpoint outputs/train/smolvla_multicolor/checkpoints/last/pretrained_model
"""

import argparse
import math
import os

os.environ.setdefault("SVT_LOG", "0")
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ch06.record_demos import FPS, IMG_H, IMG_W  # noqa: E402
from ch07.eval_smolvla import SMOLVLA_KEYS, load_smolvla, seed_noise  # noqa: E402
from ch08.record_multicolor import COLORS, TASK_FMT, random_layout  # noqa: E402
from common.robot import SO101Sim  # noqa: E402
from common.vision import SimCamera  # noqa: E402


def wilson(k, n, z=1.96):
    """성공 k / 시행 n 비율의 95% 신뢰구간 (Wilson)"""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def closest_block(policy, preprocess, postprocess, device, layout, color, seconds, top_camera):
    red, green, yellow = layout
    robot = SO101Sim(cube_pos=red, box_pos=(0.05, 0.22), render=False,
                     extra_cubes=[("green", green), ("yellow", yellow)])
    cams = {"top": SimCamera(robot.model, top_camera, IMG_W, IMG_H),
            "wrist": SimCamera(robot.model, "wrist_cam", IMG_W, IMG_H)}
    policy.reset()
    best = {c: 9.0 for c in COLORS}
    for _ in range(int(seconds * FPS)):
        state = torch.from_numpy(robot.data.qpos[:6].astype(np.float32))
        obs = {"observation.state": state.unsqueeze(0).to(device), "task": [TASK_FMT.format(color=color)]}
        for k, cam in cams.items():
            rgb, _ = cam.capture(robot.data)
            obs[SMOLVLA_KEYS[k]] = (torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0).unsqueeze(0).to(device)
        with torch.inference_mode():
            a = postprocess(policy.select_action(preprocess(obs))).squeeze(0).cpu().numpy()
        robot.data.ctrl[:6] = a
        robot.step(int(round(1 / (FPS * robot.model.opt.timestep))))
        s = robot.site_pose()[0]
        for c in COLORS:
            best[c] = min(best[c], float(np.linalg.norm(s[:2] - robot.cube_pos(c)[:2])))
    for cam in cams.values():
        cam.close()
    robot.close()
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--layouts", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=2000, help="블록 배치를 뽑는 seed")
    ap.add_argument("--noise-seed", type=int, default=0, help="행동 생성 noise의 seed (배치마다 +1)")
    ap.add_argument("--top-camera", default="top")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, preprocess, postprocess = load_smolvla(args.checkpoint, device)
    rng = np.random.default_rng(args.seed)
    hits, n, changed = 0, 0, 0
    print(f"{'layout':>6s} {'명령 색':<7s} {'가장 가까이 간 블록':<12s} {'거리(cm)':>8s}   각 블록까지 최소 거리 cm (red, green, yellow)")
    for L in range(args.layouts):
        layout = random_layout(rng)
        picked = []
        for color in COLORS:
            seed_noise(args.noise_seed + L)               # 한 배치의 세 문장은 같은 noise
            best = closest_block(policy, preprocess, postprocess, device, layout, color, args.seconds, args.top_camera)
            nearest = min(best, key=best.get)
            picked.append(nearest)
            hits += int(nearest == color)
            n += 1
            print(f"{L + 1:6d} {color:<7s} {nearest:<12s} {best[nearest] * 100:8.1f}   {[round(best[c] * 100, 1) for c in COLORS]}")
        changed += int(len(set(picked)) > 1)
    lo, hi = wilson(hits, n)
    print(f"\n명령한 색의 블록에 가장 가까이 간 비율: {hits}/{n} = {100 * hits / n:.0f}%"
          f"  (95% 신뢰구간 {100 * lo:.0f}~{100 * hi:.0f}%, 문장과 무관하면 33%)")
    print(f"문장만 바꿨을 때 가는 블록이 달라진 배치: {changed}/{args.layouts}")


if __name__ == "__main__":
    main()
