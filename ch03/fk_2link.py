"""3부 실습 — Forward Kinematics 이해하기

1) 관절 2개짜리 평면 로봇팔의 FK 를 삼각함수 몇 줄로 직접 계산합니다.
2) 같은 일을 MuJoCo 가 SO-101 에 대해 어떻게 해 주는지 확인합니다.
   관절각을 넣고 mj_forward 만 부르면 손끝 위치가 나옵니다 (물리 시뮬레이션 불필요).

실행:
    python ch03/fk_2link.py
"""

import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.scene import build_scene  # noqa: E402

np.set_printoptions(precision=3, suppress=True)


def fk_2link(theta1, theta2, l1=0.12, l2=0.14):
    """평면 2관절 로봇팔. theta 는 rad, 길이는 m. 손끝 (x, y) 를 돌려준다."""
    x1 = l1 * np.cos(theta1)                     # 첫 관절 끝(팔꿈치)
    y1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)       # 두 번째 링크는 누적 각도로
    y2 = y1 + l2 * np.sin(theta1 + theta2)
    return np.array([x2, y2]), np.array([x1, y1])


if __name__ == "__main__":
    print("=== 1) 2관절 평면 로봇팔 FK (l1=0.12, l2=0.14) ===")
    print(f"{'theta1':>8s} {'theta2':>8s}   {'팔꿈치 (x,y)':<18s} {'손끝 (x,y)':<18s} 손끝까지 거리")
    for t1, t2 in [(0, 0), (0, 90), (45, 0), (45, 45), (90, -90), (30, 120)]:
        hand, elbow = fk_2link(np.radians(t1), np.radians(t2))
        print(f"{t1:8d} {t2:8d}   {str(elbow):<18s} {str(hand):<18s} {np.linalg.norm(hand):.3f}")

    print("\n=== 2) SO-101 의 FK: 관절각 → 손끝 위치 (MuJoCo mj_forward) ===")
    model = mujoco.MjModel.from_xml_path(build_scene())
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    print(f"{'q = [pan, lift, elbow, wrist_flex, roll]':<44s} 손끝 (x, y, z)")
    for q in [[0, 0, 0, 0, 0],             # 모든 관절 0: 팔을 앞으로 쭉 뻗음
              [0, -1.5, 0.5, 0, 0],          # home(대기) 자세: 팔을 위로 세움
              [0.5, -1.5, 0.5, 0, 0],        # 대기 자세에서 pan 만 0.5 rad
              [0, -1.57, 1.57, 1.2, 0],      # 손목을 꺾어 손가락이 바닥을 향함
              [0, -1.0, 1.0, 1.2, 0],
              [0, 0.0, 0.3, 1.27, 0]]:       # 블록 위 (3-5 에서 IK 가 찾아낸 자세와 비슷)
        data.qpos[:5] = q
        mujoco.mj_forward(model, data)       # 운동학만 계산. mj_step 이 아니다
        print(f"{str(q):<44s} {data.site_xpos[site_id]}")

    print("\n=== 3) 관절 하나를 조금 바꾸면 손끝은 얼마나 움직이나 (수치 미분) ===")
    q0 = np.array([0, -1.0, 1.0, 1.2, 0.0])
    data.qpos[:5] = q0
    mujoco.mj_forward(model, data)
    p0 = data.site_xpos[site_id].copy()
    eps = 0.01
    for i, n in enumerate(names):
        q = q0.copy()
        q[i] += eps
        data.qpos[:5] = q
        mujoco.mj_forward(model, data)
        dp = (data.site_xpos[site_id] - p0) / eps
        print(f"{n:<14s} 관절 1 rad 당 손끝 이동 (x,y,z) = {dp}  (크기 {np.linalg.norm(dp):.3f} m/rad)")
