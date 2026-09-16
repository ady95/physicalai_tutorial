"""4부 실습 — Simulation Camera 사용하기

장면에 정의된 카메라(top, side, front)와 로봇 손목 카메라(wrist_cam)로
RGB 이미지와 Depth 이미지를 찍어 저장합니다. 카메라의 위치·시야각을 출력하고,
한 픽셀의 depth 로 월드 좌표를 되찾는 것(unprojection)을 확인합니다.

실행:
    MUJOCO_GL=egl python ch04/sim_camera.py
"""

import os
import sys

import cv2
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.robot import SO101Sim  # noqa: E402
from common.vision import SimCamera  # noqa: E402

np.set_printoptions(precision=3, suppress=True)
OUT = "outputs/ch04"


def save_depth_png(depth, path, max_m=1.0):
    d = np.clip(depth / max_m, 0, 1)
    img = (255 * (1 - d)).astype(np.uint8)              # 가까울수록 밝게
    cv2.imwrite(path, cv2.applyColorMap(img, cv2.COLORMAP_JET))


def main():
    os.makedirs(OUT, exist_ok=True)
    robot = SO101Sim(cube_pos=(0.24, 0.0), box_pos=(0.05, 0.22), render=False)
    model, data = robot.model, robot.data

    print("=== 장면의 카메라 목록 ===")
    for c in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, c)
        print(f"  [{c}] {name:<10s} pos={data.cam_xpos[c]}  fovy={model.cam_fovy[c]:.0f}deg")

    print("\n=== 카메라별 RGB / Depth 촬영 ===")
    for name in ["top", "fixed", "side", "wrist_cam"]:
        cam = SimCamera(model, camera=name, width=640, height=480)
        rgb, depth = cam.capture(data)
        cv2.imwrite(f"{OUT}/cam_{name}_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        save_depth_png(depth, f"{OUT}/cam_{name}_depth.png")
        print(f"  {name:<10s} rgb {rgb.shape} {rgb.dtype}  depth {depth.shape} {depth.dtype} "
              f"min {depth.min():.3f} m  max {depth.max():.3f} m")
        cam.close()

    # 손목 카메라는 대기 자세에서는 하늘을 본다. 블록 위로 손을 보낸 뒤 다시 찍는다
    robot.open_gripper(0.5)
    robot.move_to([0.24, 0.0, 0.08], seconds=1.2)
    cam = SimCamera(model, camera="wrist_cam", width=640, height=480)
    rgb, depth = cam.capture(data)
    cv2.imwrite(f"{OUT}/cam_wrist_cam_above_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    save_depth_png(depth, f"{OUT}/cam_wrist_cam_above_depth.png", max_m=0.3)
    print(f"  {'wrist_cam':<10s} (블록 위 8 cm 로 이동 후)  depth min {depth.min():.3f} m  max {depth.max():.3f} m")
    cam.close()
    robot.reset()

    print("\n=== top 카메라: 픽셀 → 월드 좌표 되찾기 ===")
    cam = SimCamera(model, camera="top", width=640, height=480)
    rgb, depth = cam.capture(data)
    print(f"  초점 거리 f = {cam.f:.1f} px  (fovy {model.cam_fovy[cam.cam_id]:.0f}deg, 세로 480px)")
    print(f"  카메라 위치 {data.cam_xpos[cam.cam_id]},  회전행렬의 z축 {data.cam_xmat[cam.cam_id].reshape(3,3)[:,2]} (카메라는 -z 를 본다)")

    # 이미지 중앙 픽셀 → 바닥의 어느 점인가
    u, v = 320, 240
    p = cam.pixel_to_world(data, u, v, depth[v, u])
    print(f"  중앙 픽셀 ({u},{v}) depth={depth[v, u]:.3f} m → 월드 {p}")

    # 블록 중심의 실제 위치를 이미지에 투영했다가 다시 되돌려 보기
    cube = robot.cube_pos() + [0, 0, 0.015]            # 블록 윗면 중심
    R = data.cam_xmat[cam.cam_id].reshape(3, 3)
    p_cam = R.T @ (cube - data.cam_xpos[cam.cam_id])
    u_c = cam.cx + cam.f * p_cam[0] / (-p_cam[2])
    v_c = cam.cy - cam.f * p_cam[1] / (-p_cam[2])
    print(f"  블록 윗면 중심 {cube} → 픽셀 ({u_c:.1f}, {v_c:.1f})")
    ui, vi = int(round(u_c)), int(round(v_c))
    back = cam.pixel_to_world(data, u_c, v_c, depth[vi, ui])
    print(f"  그 픽셀의 depth {depth[vi, ui]:.3f} m → 월드 {back}   (오차 {np.linalg.norm(back - cube) * 1000:.1f} mm)")

    marked = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.circle(marked, (ui, vi), 6, (0, 255, 0), 2)
    cv2.imwrite(f"{OUT}/cam_top_marked.png", marked)
    print(f"\n저장: {OUT}/cam_*_rgb.png, cam_*_depth.png, cam_top_marked.png")
    cam.close()


if __name__ == "__main__":
    main()
