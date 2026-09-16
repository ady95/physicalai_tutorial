"""6부 프로젝트 — 학습한 ACT 에게 Pick and Place 시키기

학습된 ACT 체크포인트를 불러와 시뮬레이션에서 여러 episode 를 실행하고 성공률을 잽니다.
    Camera + 관절각 → ACT → 행동(목표 각도 6개) → 로봇
매 1/FPS 초마다 관측을 새로 찍어 Policy 에 넣습니다 (닫힌 고리).

실행:
    MUJOCO_GL=egl python ch06/eval_act.py --checkpoint outputs/train/act_so101/checkpoints/last/pretrained_model
    MUJOCO_GL=egl python ch06/eval_act.py --checkpoint ... --episodes 20 --video
"""

import argparse
import os

os.environ.setdefault("SVT_LOG", "0")          # 영상 인코더(SVT-AV1)의 장황한 로그 끄기
import sys
import time

import mujoco
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ch03.pick_and_place import is_in_box  # noqa: E402
from ch06.record_demos import FPS, IMG_H, IMG_W, TASK, random_cube  # noqa: E402
from common.robot import SO101Sim  # noqa: E402
from common.vision import SimCamera  # noqa: E402


def run_episode(policy, preprocess, postprocess, device, cube, max_seconds=12.0, video=False, out=None):
    robot = SO101Sim(cube_pos=cube, box_pos=(0.05, 0.22), render=video, camera="fixed")
    cams = {"top": SimCamera(robot.model, "top", IMG_W, IMG_H),
            "wrist": SimCamera(robot.model, "wrist_cam", IMG_W, IMG_H)}
    steps_per_frame = int(round(1 / (FPS * robot.model.opt.timestep)))
    policy.reset()
    n_frames = int(max_seconds * FPS)
    success, t_success = False, None
    for i in range(n_frames):
        state = torch.from_numpy(robot.data.qpos[:6].astype(np.float32))
        obs = {"observation.state": state.unsqueeze(0).to(device), "task": [TASK]}
        for k, cam in cams.items():
            rgb, _ = cam.capture(robot.data)
            img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0      # (C,H,W) 0~1
            obs[f"observation.images.{k}"] = img.unsqueeze(0).to(device)
        obs = preprocess(obs)
        with torch.inference_mode():
            action = policy.select_action(obs)
        action = postprocess(action)
        robot.data.ctrl[:6] = action.squeeze(0).cpu().numpy()
        robot.step(steps_per_frame)
        if not success and is_in_box(robot.cube_pos(), robot.box_pos()):
            success, t_success = True, (i + 1) / FPS
            robot.hold(0.5)
            break
    for cam in cams.values():
        cam.close()
    if video and out:
        robot.save_video(out)
    robot.close()
    return success, t_success


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-root", default="outputs/datasets/so101_pickplace_sim")
    ap.add_argument("--repo-id", default="physicalai/so101_pickplace_sim")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--out", default="outputs/ch06_act_eval.mp4")
    args = ap.parse_args()

    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act import ACTPolicy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = ACTPolicy.from_pretrained(args.checkpoint).to(device).eval()
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    preprocess, postprocess = make_pre_post_processors(
        policy.config, args.checkpoint, dataset_stats=meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(device)}})
    print(f"policy: ACT ({sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M params), device {device}")
    print(f"chunk_size {policy.config.chunk_size}, n_action_steps {policy.config.n_action_steps}")

    rng = np.random.default_rng(args.seed)
    results, t0 = [], time.time()
    for i in range(args.episodes):
        cube = random_cube(rng)
        ok, ts = run_episode(policy, preprocess, postprocess, device, cube,
                             video=(args.video and i == 0), out=args.out)
        results.append(ok)
        print(f"episode {i + 1:2d}  블록 ({cube[0]:+.3f}, {cube[1]:+.3f})  → "
              f"{'성공 (%.1f s)' % ts if ok else '실패'}   ({time.time() - t0:4.0f} s)")
    print(f"\n성공률 {sum(results)}/{len(results)} = {100 * sum(results) / len(results):.0f}%")


if __name__ == "__main__":
    main()
