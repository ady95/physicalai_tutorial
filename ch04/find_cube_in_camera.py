"""4부 실습 — 가상 카메라에서 블록 찾기

top 카메라 이미지에서 빨간 블록을 색으로 찾고(픽셀 좌표), depth로 월드 좌표를 계산해
시뮬레이터가 알고 있는 진짜 위치와 비교합니다. 블록 위치를 여러 번 바꿔 오차를 잽니다.

실행:
    MUJOCO_GL=egl python ch04/find_cube_in_camera.py
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.robot import SO101Sim  # noqa: E402
from common.vision import SimCamera, locate_object  # noqa: E402

np.set_printoptions(precision=3, suppress=True)
OUT = "outputs/ch04"


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    print(f"{'true (x, y)':<22s} {'pixel (u, v)':<18s} {'estimated (x, y, z)':<26s} 오차(mm)")
    errors = []
    for i in range(8):
        if i == 0:
            cube_xy = (0.24, 0.0)
        else:
            r, th = rng.uniform(0.17, 0.27), rng.uniform(-0.7, 0.7)
            cube_xy = (r * np.cos(th), r * np.sin(th))
        robot = SO101Sim(cube_pos=cube_xy, box_pos=(0.05, 0.22), render=False)
        cam = SimCamera(robot.model, camera="top")
        rgb, depth = cam.capture(robot.data)

        est, det = locate_object(cam, robot.data, "red", rgb, depth)
        true = robot.cube_pos()
        if est is None:
            print(f"{str(np.round(true[:2], 3)):<22s} 검출 실패")
            continue
        err = np.linalg.norm(est[:2] - true[:2]) * 1000
        errors.append(err)
        u, v = det["center"]
        print(f"{str(np.round(true[:2], 3)):<22s} ({u:6.1f}, {v:6.1f})    {str(est):<26s} {err:5.1f}")

        if i == 0:
            x, y, w, h = det["bbox"]
            marked = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.rectangle(marked, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(marked, (int(u), int(v)), 4, (0, 255, 255), -1)
            cv2.putText(marked, f"red ({est[0]:.3f}, {est[1]:.3f})", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(f"{OUT}/find_cube_top.png", marked)
            cv2.imwrite(f"{OUT}/find_cube_mask.png", det["mask"])

            # 상자도 같은 방법으로
            box_est, box_det = locate_object(cam, robot.data, "blue", rgb, depth)
            print(f"  (상자) true {robot.box_pos()[:2]}  estimated {box_est}  "
                  f"오차 {np.linalg.norm(box_est[:2] - robot.box_pos()[:2]) * 1000:.1f} mm")
        cam.close()

    e = np.array(errors)
    print(f"\n검출 {len(e)}/8,  위치 오차 평균 {e.mean():.1f} mm, 최대 {e.max():.1f} mm")
    print(f"저장: {OUT}/find_cube_top.png, find_cube_mask.png")


if __name__ == "__main__":
    main()
