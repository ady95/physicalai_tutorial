"""8부 프로젝트 3 — 처음 보는 환경에서 테스트하기

학습 때와 다른 조명, 바닥 색, 카메라 위치, 블록 위치 범위에서 ACT 의 성공률을 잽니다.
    Training Environment ≠ Test Environment

실행:
    MUJOCO_GL=egl python ch08/robustness.py --checkpoint outputs/train/act_so101/checkpoints/last/pretrained_model
    MUJOCO_GL=egl python ch08/robustness.py --checkpoint ... --episodes 10 --only light_dim,floor_dark
"""

import argparse
import os

os.environ.setdefault("SVT_LOG", "0")
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ch06.eval_act import run_episode  # noqa: E402
from ch06.record_demos import IMG_H, IMG_W, random_cube  # noqa: E402
from common.robot import SO101Sim  # noqa: E402
from common.vision import SimCamera  # noqa: E402

# 이름: (설명, build_scene 인자)
VARIANTS = {
    "baseline":     ("학습 환경 그대로", {}),
    "light_dim":    ("조명 절반 밝기", {"light_diffuse": (0.3, 0.3, 0.3)}),
    "light_bright": ("조명 두 배 밝기", {"light_diffuse": (1.2, 1.2, 1.2)}),
    "light_side":   ("조명을 옆으로 옮김 (그림자 방향 변화)", {"light_pos": (0.6, 0.6, 0.8)}),
    "floor_dark":   ("바닥을 어둡게", {"floor_rgba": (0.4, 0.4, 0.45, 1)}),
    "floor_red":    ("바닥을 붉은 톤으로 (블록과 비슷한 색)", {"floor_rgba": (0.9, 0.6, 0.6, 1)}),
    "cam_shift":    ("위 카메라를 5 cm 옆으로", {"top_cam_pos": (0.2, 0.05, 0.8)}),
    "cam_high":     ("위 카메라를 10 cm 높게", {"top_cam_pos": (0.2, 0.0, 0.9)}),
}


def save_examples(out_dir):
    """각 변형의 top 카메라 이미지를 한 장씩 저장 (책의 그림용)."""
    os.makedirs(out_dir, exist_ok=True)
    for name, (_, kw) in VARIANTS.items():
        robot = SO101Sim(cube_pos=(0.24, 0.0), box_pos=(0.05, 0.22), render=False, **kw)
        cam = SimCamera(robot.model, "top", IMG_W, IMG_H)          # 그림은 항상 top 카메라로
        rgb, _ = cam.capture(robot.data)
        cv2.imwrite(f"{out_dir}/variant_{name}.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cam.close()
        robot.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--repo-id", default="physicalai/so101_pickplace_sim")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=3000)
    ap.add_argument("--only", default=None, help="쉼표로 변형 이름 지정")
    ap.add_argument("--top-camera", default="top")
    args = ap.parse_args()

    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act import ACTPolicy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = ACTPolicy.from_pretrained(args.checkpoint).to(device).eval()
    stats = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root).stats if args.dataset_root else None
    preprocess, postprocess = make_pre_post_processors(
        policy.config, args.checkpoint, dataset_stats=stats,
        preprocessor_overrides={"device_processor": {"device": str(device)}})

    save_examples("outputs/ch08")
    names = args.only.split(",") if args.only else list(VARIANTS)
    print(f"{'변형':<14s} {'설명':<34s} {'성공률':>8s}")
    results = {}
    t0 = time.time()
    for name in names:
        desc, kw = VARIANTS[name]
        rng = np.random.default_rng(args.seed)                # 모든 변형에 같은 블록 위치들
        ok = 0
        for i in range(args.episodes):
            cube = random_cube(rng)
            s, _ = run_episode(policy, preprocess, postprocess, device, cube, scene_kw=kw, top_camera=args.top_camera)
            ok += int(s)
        results[name] = ok / args.episodes
        print(f"{name:<14s} {desc:<34s} {ok:3d}/{args.episodes:<3d} = {100 * ok / args.episodes:4.0f}%   ({time.time() - t0:4.0f} s)")
    print("\n저장: outputs/ch08/variant_*.png")


if __name__ == "__main__":
    main()
