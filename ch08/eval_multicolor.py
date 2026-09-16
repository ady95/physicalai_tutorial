"""8부 프로젝트 1 — 블록 세 개 장면에서 SmolVLA 에게 색을 지정해 집게 하기

"Pick up the {color} cube ..." 명령을 바꿔 가며 SmolVLA 를 실행하고,
지정한 색의 블록이 상자에 들어갔는지(성공), 다른 색을 집었는지(오답)를 셉니다.

실행:
    MUJOCO_GL=egl python ch08/eval_multicolor.py --checkpoint outputs/train/smolvla_multicolor/checkpoints/last/pretrained_model
    MUJOCO_GL=egl python ch08/eval_multicolor.py --checkpoint ... --episodes-per-color 5 --video
"""

import argparse
import os

os.environ.setdefault("SVT_LOG", "0")
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ch03.pick_and_place import is_in_box  # noqa: E402
from ch06.record_demos import FPS, IMG_H, IMG_W  # noqa: E402
from ch07.eval_smolvla import SMOLVLA_KEYS, load_smolvla  # noqa: E402
from ch08.record_multicolor import COLORS, TASK_FMT, random_layout  # noqa: E402
from common.robot import SO101Sim  # noqa: E402
from common.vision import SimCamera  # noqa: E402


def run_episode(policy, preprocess, postprocess, device, layout, color, max_seconds=12.0, video=False, out=None,
                top_camera="top"):
    red, green, yellow = layout
    robot = SO101Sim(cube_pos=red, box_pos=(0.05, 0.22), render=video, camera="fixed",
                     extra_cubes=[("green", green), ("yellow", yellow)])
    cams = {"top": SimCamera(robot.model, top_camera, IMG_W, IMG_H),
            "wrist": SimCamera(robot.model, "wrist_cam", IMG_W, IMG_H)}
    steps_per_frame = int(round(1 / (FPS * robot.model.opt.timestep)))
    task = TASK_FMT.format(color=color)
    policy.reset()
    result = "실패"
    for i in range(int(max_seconds * FPS)):
        state = torch.from_numpy(robot.data.qpos[:6].astype(np.float32))
        obs = {"observation.state": state.unsqueeze(0).to(device), "task": [task]}
        for k, cam in cams.items():
            rgb, _ = cam.capture(robot.data)
            obs[SMOLVLA_KEYS[k]] = (torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0).unsqueeze(0).to(device)
        obs = preprocess(obs)
        with torch.inference_mode():
            action = policy.select_action(obs)
        action = postprocess(action)
        robot.data.ctrl[:6] = action.squeeze(0).cpu().numpy()
        robot.step(steps_per_frame)
        box = robot.box_pos()
        in_box = [c for c in COLORS if is_in_box(robot.cube_pos(c), box)]
        if in_box:
            result = "성공" if in_box == [color] else f"오답({','.join(in_box)})"
            robot.hold(0.5)
            break
    for cam in cams.values():
        cam.close()
    if video and out:
        robot.save_video(out)
    robot.close()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes-per-color", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--out", default="outputs/ch08_multicolor.mp4")
    ap.add_argument("--top-camera", default="top")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, preprocess, postprocess = load_smolvla(args.checkpoint, device)
    rng = np.random.default_rng(args.seed)
    stats = {c: {"성공": 0, "오답": 0, "실패": 0} for c in COLORS}
    t0 = time.time()
    for i in range(args.episodes_per_color):
        layout = random_layout(rng)                       # 같은 배치에서 세 가지 색을 차례로 시킨다
        for color in COLORS:
            res = run_episode(policy, preprocess, postprocess, device, layout, color,
                              video=(args.video and i == 0 and color == "green"), out=args.out, top_camera=args.top_camera)
            stats[color][res.split("(")[0]] += 1
            print(f"layout {i + 1:2d}  \"{TASK_FMT.format(color=color)}\"  → {res}   ({time.time() - t0:4.0f} s)")
    print()
    print(f"{'색':<8s} {'성공':>5s} {'오답':>5s} {'실패':>5s}")
    for c in COLORS:
        s = stats[c]
        print(f"{c:<8s} {s['성공']:5d} {s['오답']:5d} {s['실패']:5d}")
    n = args.episodes_per_color * 3
    print(f"\n전체 성공률 {sum(s['성공'] for s in stats.values())}/{n} = {100 * sum(s['성공'] for s in stats.values()) / n:.0f}%")


if __name__ == "__main__":
    main()
