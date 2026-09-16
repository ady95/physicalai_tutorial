"""3부 실습 — Inverse Kinematics 로 로봇 손을 빨간 블록 위로 보내기

블록 위치를 읽고 → 그 위(바닥에서 8 cm)를 목표로 IK 를 풀고 → 관절을 움직이고 →
다시 IK 로 손끝이 블록을 감싸는 높이(2 cm)까지 내려갑니다. 손가락은 항상 아래를 향하게 합니다.

실행:
    MUJOCO_GL=egl python ch03/ik_reach.py
    python ch03/ik_reach.py --no-video --cube 0.20 0.08
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.robot import SO101Sim  # noqa: E402

np.set_printoptions(precision=3, suppress=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", type=float, nargs=2, default=[0.24, 0.0], help="블록 x y (m)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--out", default="outputs/ch03_ik_reach.mp4")
    args = ap.parse_args()

    robot = SO101Sim(cube_pos=tuple(args.cube), render=not args.no_video, camera="side")
    cube = robot.cube_pos()
    print(f"블록 위치      : {cube}")
    print(f"손끝 시작 위치 : {robot.site_pose()[0]}   손가락 방향 {robot.finger_dir()}")
    robot.open_gripper(0.5)

    # 1) IK 만 풀어 보기 (아직 움직이지 않음)
    target = np.array([cube[0], cube[1], 0.08])        # 블록 위, 바닥에서 8 cm
    q, err = robot.ik(target)
    print(f"\n[IK] 목표 {target} → 관절각 {q}  (오차 {err:.4f} m)")

    # 2) 그 관절각으로 이동
    robot.move_arm(q, seconds=1.5)
    robot.hold(0.3)
    pos = robot.site_pose()[0]
    print(f"[이동 후] 손끝 {pos}  목표와 거리 {np.linalg.norm(pos - target) * 100:.2f} cm  손가락 방향 {robot.finger_dir()}")

    # 3) 블록 바로 위까지 내려가기
    target2 = np.array([cube[0], cube[1], 0.02])       # 손끝을 블록 중심(1.5 cm)보다 5 mm 위로
    q2, err2 = robot.move_to(target2, seconds=1.0)
    robot.hold(0.3)
    pos = robot.site_pose()[0]
    print(f"[하강 후] 손끝 {pos}  목표와 거리 {np.linalg.norm(pos - target2) * 100:.2f} cm  관절각 {q2}")

    # 4) IK 의 한계: 손이 닿지 않는 곳
    far = np.array([0.45, 0.0, 0.02])
    q3, err3 = robot.ik(far)
    print(f"\n[도달 불가 목표] {far} → 오차 {err3:.3f} m (0 에 가까워지지 않음 = 팔 길이 밖)")

    robot.save_video(args.out)
    robot.close()


if __name__ == "__main__":
    main()
