"""3부 보조 — 작업 영역 확인: 블록 위치를 바꿔 가며 집기 성공 여부를 훑는다

로봇팔이 위에서 내려 집을 수 있는 영역이 어디까지인지 실험으로 알아냅니다.
IK 오차, 집게가 멈춘 각도, 블록이 밀린 거리, 들어 올린 높이를 함께 출력합니다.

실행:
    python ch03/workspace_sweep.py            # 접근 높이 0.08, 집는 높이 0.02
    python ch03/workspace_sweep.py 0.06 0.02
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.robot import SO101Sim  # noqa: E402

np.set_printoptions(precision=3, suppress=True)
approach_z = float(sys.argv[1]) if len(sys.argv) > 1 else 0.08
grasp_z = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02


def try_pick(cx, cy):
    robot = SO101Sim(cube_pos=(cx, cy), render=False)
    cube = robot.cube_pos()
    robot.open_gripper(0.5)
    _, e1 = robot.move_to([cube[0], cube[1], approach_z], seconds=1.0)
    _, e2 = robot.move_to([cube[0], cube[1], grasp_z], seconds=0.8)
    robot.hold(0.2)
    site_before = robot.site_pose()[0]
    cube_before = robot.cube_pos()
    robot.close_gripper(0.8)
    grip = robot.gripper_opening()
    robot.move_to([cube[0], cube[1], 0.12], seconds=1.0)
    robot.hold(0.3)
    cube_after = robot.cube_pos()
    ok = cube_after[2] > 0.06
    print(f"cube=({cx:+.2f},{cy:+.2f}) ik_err approach={e1:.3f} grasp={e2:.3f} "
          f"site-cube dxy={site_before[:2]-cube_before[:2]} cube moved={cube_before[:2]-cube[:2]} "
          f"grip={grip:+.2f} lifted z={cube_after[2]:.3f} -> {'OK' if ok else 'FAIL'}")
    return ok


results = {}
for cx in [0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30]:
    for cy in [0.0, 0.08, -0.08]:
        results[(cx, cy)] = try_pick(cx, cy)
n_ok = sum(results.values())
print(f"성공 {n_ok}/{len(results)}")
