"""3부 실습 — 가상 물체의 Pose 읽기

로봇팔 + 블록 + 상자 장면에서 블록, 상자, 로봇 손끝의 Pose(위치 + 자세)를 읽습니다.
자세는 Quaternion으로 저장되어 있는데, 사람이 읽기 쉬운 Roll / Pitch / Yaw 로도 바꿔 봅니다.

실행:
    python ch03/read_pose.py
"""

import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.scene import build_scene  # noqa: E402

np.set_printoptions(precision=3, suppress=True)


def quat_to_rpy(q):
    """MuJoCo quaternion (w, x, y, z) → roll, pitch, yaw (rad)."""
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def print_body_pose(model, data, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    pos, quat = data.xpos[bid], data.xquat[bid]
    rpy = np.degrees(quat_to_rpy(quat))
    print(f"{name:<8s} pos(x,y,z)={pos}  quat(w,x,y,z)={quat}  rpy(deg)={rpy}")


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(build_scene(cube_pos=(0.25, 0.0), box_pos=(0.0, 0.25)))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)      # 로봇을 home 자세로
    mujoco.mj_forward(model, data)                   # 위치·자세 계산

    print("=== home 자세에서 각 Body의 Pose ===")
    for name in ["cube", "box", "gripper"]:
        print_body_pose(model, data, name)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    R = data.site_xmat[site_id].reshape(3, 3)
    print(f"\n손끝 site 'gripperframe' 위치 = {data.site_xpos[site_id]}")
    print(f"손끝 site의 x축(손가락 방향) = {R[:, 0]}")

    print("\n=== 블록을 45도 돌려 놓으면 Quaternion은? ===")
    # freejoint의 qpos: [x, y, z, qw, qx, qy, qz]
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    adr = model.jnt_qposadr[jid]
    yaw = np.radians(45)
    data.qpos[adr + 3: adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]   # z축 회전 quaternion
    mujoco.mj_forward(model, data)
    print_body_pose(model, data, "cube")

    print("\n=== 로봇 좌표계에서 본 블록 위치 ===")
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    R_base = data.xmat[base_id].reshape(3, 3)
    p_rel = R_base.T @ (data.xpos[cube_id] - data.xpos[base_id])
    print(f"base 위치 {data.xpos[base_id]},  블록의 base 기준 좌표 {p_rel}")
