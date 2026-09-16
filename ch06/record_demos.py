"""6부 실습 — Demonstration Dataset 만들어보기

3부의 Rule 기반 Pick and Place 를 "시연자"로 삼아, 시뮬레이션에서 여러 episode 를 실행하며
(관측, 행동) 을 LeRobotDataset 형식으로 기록합니다. 사람이 조종 장치로 시연하는 것을
프로그램이 대신하는 것입니다.

    관측 = 관절각 6개 (observation.state) + 카메라 2대 이미지 (observation.images.top / .wrist)
    행동 = 6개 구동기의 목표 각도 (action)  ← 그 순간 컨트롤러가 명령한 값
    fps  = 20

실행:
    MUJOCO_GL=egl python ch06/record_demos.py --episodes 50
    MUJOCO_GL=egl python ch06/record_demos.py --episodes 5 --root outputs/datasets/test
"""

import argparse
import os

os.environ.setdefault("SVT_LOG", "0")          # 영상 인코더(SVT-AV1)의 장황한 로그 끄기
import shutil
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ch03.pick_and_place import APPROACH_Z, GRASP_Z, LIFT_Z, PLACE_Z, above, is_in_box  # noqa: E402
from common.robot import SO101Sim  # noqa: E402
from common.vision import SimCamera  # noqa: E402

FPS = 20
IMG_H, IMG_W = 240, 320
TASK = "Pick up the red cube and put it in the blue box."
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class RecordingRobot(SO101Sim):
    """SO101Sim 과 같지만, 1/FPS 초마다 (관측, 행동) 프레임을 기록한다."""

    def __init__(self, dataset=None, task=TASK, top_camera="top", **kw):
        super().__init__(render=False, **kw)
        self.dataset = dataset
        self.task = task
        self.cams = {"top": SimCamera(self.model, top_camera, IMG_W, IMG_H),
                     "wrist": SimCamera(self.model, "wrist_cam", IMG_W, IMG_H)}
        self.record_every = int(round(1 / (FPS * self.model.opt.timestep)))
        self.n_recorded = 0

    def observe(self):
        state = self.data.qpos[:6].astype(np.float32)             # 관절각 6개
        imgs = {k: cam.capture(self.data)[0] for k, cam in self.cams.items()}
        return state, imgs

    def step(self, n=1):
        for _ in range(n):
            if self.dataset is not None and self._step_count % self.record_every == 0:
                state, imgs = self.observe()
                action = self.data.ctrl[:6].astype(np.float32)   # 지금 명령 중인 목표 각도
                self.dataset.add_frame({
                    "observation.state": state,
                    "observation.images.top": imgs["top"],
                    "observation.images.wrist": imgs["wrist"],
                    "action": action,
                    "task": self.task,
                })
                self.n_recorded += 1
            mujoco.mj_step(self.model, self.data)
            self._step_count += 1

    def close(self):
        for cam in self.cams.values():
            cam.close()
        super().close()


def scripted_pick_and_place(robot):
    """3-6 과 같은 다섯 단계. 성공 여부를 돌려준다."""
    cube, box = robot.cube_pos(), robot.box_pos()
    robot.open_gripper(0.5)
    robot.move_to(above(cube, APPROACH_Z), seconds=1.2)
    robot.move_to(above(cube, GRASP_Z), seconds=0.8)
    robot.hold(0.2)
    robot.close_gripper(0.8)
    robot.move_to(above(cube, LIFT_Z), seconds=1.0)
    if robot.cube_pos()[2] < 0.06:
        return False
    robot.move_to(above(box, LIFT_Z), seconds=1.5)
    robot.move_to(above(box, PLACE_Z), seconds=0.8)
    robot.hold(0.2)
    robot.open_gripper(0.5)
    robot.move_to(above(box, LIFT_Z), seconds=0.8)
    robot.hold(0.3)
    return is_in_box(robot.cube_pos(), box)


def random_cube(rng):
    r, th = rng.uniform(0.17, 0.27), rng.uniform(-0.7, 0.7)
    return (float(r * np.cos(th)), float(r * np.sin(th)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--root", default="outputs/datasets/so101_pickplace_sim")
    ap.add_argument("--repo-id", default="physicalai/so101_pickplace_sim")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-failed", action="store_true", help="실패한 시연도 저장")
    ap.add_argument("--cube", type=float, nargs=2, default=None, help="블록 위치를 고정 (8-2 의 일반화 실험용)")
    ap.add_argument("--top-camera", default="top", help="위 카메라 이름 (top 또는 top_zoom)")
    args = ap.parse_args()

    from lerobot.datasets import LeRobotDataset

    if os.path.exists(args.root):
        shutil.rmtree(args.root)

    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        "observation.images.top": {"dtype": "video", "shape": (IMG_H, IMG_W, 3), "names": ["height", "width", "channels"]},
        "observation.images.wrist": {"dtype": "video", "shape": (IMG_H, IMG_W, 3), "names": ["height", "width", "channels"]},
    }
    dataset = LeRobotDataset.create(repo_id=args.repo_id, fps=FPS, features=features,
                                    root=args.root, robot_type="so101_sim", use_videos=True)

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    saved, failed = 0, 0
    for ep in range(args.episodes):
        cube = tuple(args.cube) if args.cube else random_cube(rng)
        robot = RecordingRobot(dataset=dataset, cube_pos=cube, box_pos=(0.05, 0.22), top_camera=args.top_camera)
        ok = scripted_pick_and_place(robot)
        n = robot.n_recorded
        robot.close()
        if ok or args.keep_failed:
            dataset.save_episode()
            saved += 1
        else:
            dataset.clear_episode_buffer()
            failed += 1
        print(f"episode {ep + 1:3d}/{args.episodes}  블록 ({cube[0]:+.3f}, {cube[1]:+.3f})  "
              f"{n:3d} frames  {'저장' if ok or args.keep_failed else '실패→버림'}  ({time.time() - t0:5.0f} s)")

    dataset.finalize()
    print(f"\n완료: episode {saved}개 저장, {failed}개 실패, 총 {time.time() - t0:.0f} 초")
    print(f"데이터셋 위치: {args.root}")
    print(f"총 프레임 수: {dataset.meta.total_frames}")


if __name__ == "__main__":
    main()
