"""2부 실습 — 가상 로봇팔 관절 하나씩 움직이기

MuJoCo Menagerie의 SO-101 로봇팔(6 DOF: 관절 5개 + 그리퍼)을 불러와
관절(joint)·구동기(actuator) 목록을 출력하고, 관절을 하나씩 차례로 움직여 봅니다.
로봇 손(gripper body)의 위치가 어떻게 바뀌는지 함께 확인합니다.

실행:
    MUJOCO_GL=egl python ch02/robot_arm_joints.py
    python ch02/robot_arm_joints.py --no-video
"""

import argparse
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.menagerie import so101_xml  # noqa: E402


def print_model_info(model):
    print(f"model: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, 0) or '(unnamed)'}")
    print(f"body {model.nbody}개, joint {model.njnt}개, actuator {model.nu}개, "
          f"qpos 크기 {model.nq}, ctrl 크기 {model.nu}")
    print()
    print("joint 목록 (관절 이름 / 종류 / 가동 범위 rad)")
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        lo, hi = model.jnt_range[j]
        print(f"  [{j}] {name:<14s} hinge  range = [{lo:+.3f}, {hi:+.3f}]")
    print()
    print("actuator 목록 (구동기 이름 → 움직이는 joint / ctrl 범위)")
    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        jid = model.actuator_trnid[a][0]
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        lo, hi = model.actuator_ctrlrange[a]
        print(f"  [{a}] {name:<14s} → {jname:<14s} ctrl = [{lo:+.3f}, {hi:+.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--out", default="outputs/ch02_robot_arm_joints.mp4")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(so101_xml("scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print_model_info(model)

    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")

    renderer = None
    frames = []
    if not args.no_video:
        renderer = mujoco.Renderer(model, height=480, width=640)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.azimuth, cam.elevation, cam.distance = 150, -25, 0.9
        cam.lookat[:] = [0.0, -0.1, 0.15]

    fps = 30
    steps_per_frame = int(round(1 / (fps * model.opt.timestep)))
    hold_steps = int(1.0 / model.opt.timestep)      # 목표를 1초 동안 유지

    def run(seconds_steps):
        for s in range(seconds_steps):
            mujoco.mj_step(model, data)
            if renderer is not None and s % steps_per_frame == 0:
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render().copy())

    print()
    print("관절을 하나씩 움직여 봅니다 (각 관절: 범위의 60% → 0 으로 복귀)")
    print(f"{'actuator':<14s} {'ctrl':>7s}   hand x      y      z   (m)")
    print(f"{'(초기 자세)':<14s} {0.0:>7.3f}   " + "  ".join(f"{v:+.3f}" for v in data.xpos[hand_id]))

    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        lo, hi = model.actuator_ctrlrange[a]
        target = 0.6 * hi if abs(hi) > abs(lo) else 0.6 * lo

        data.ctrl[:] = 0.0
        data.ctrl[a] = target                    # 이 관절에만 목표 각도 명령
        run(hold_steps)
        pos = data.xpos[hand_id]
        actual = data.qpos[model.jnt_qposadr[model.actuator_trnid[a][0]]]
        print(f"{name:<14s} {target:>7.3f}   " + "  ".join(f"{v:+.3f}" for v in pos)
              + f"   (실제 관절각 {actual:+.3f})")

        data.ctrl[:] = 0.0                       # 원위치
        run(hold_steps)

    if frames:
        import imageio

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        imageio.mimsave(args.out, frames, fps=fps)
        imageio.imwrite(args.out.replace(".mp4", "_last.png"), frames[-1])
        print(f"\n영상 저장: {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
