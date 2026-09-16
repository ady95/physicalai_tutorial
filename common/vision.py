"""가상 카메라 도우미: RGB/Depth 렌더링, 픽셀 → 월드 좌표 변환, 색으로 물체 찾기.

4부(Vision)부터 8부까지 사용합니다.
"""

import cv2
import mujoco
import numpy as np

# HSV 범위 (OpenCV: H 0~179, S 0~255, V 0~255)
COLOR_RANGES = {
    "red": [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    "blue": [((100, 120, 70), (130, 255, 255))],
    "yellow": [((20, 120, 70), (35, 255, 255))],
    "green": [((40, 80, 70), (80, 255, 255))],
}


class SimCamera:
    """MuJoCo 카메라 하나를 RGB + Depth 로 찍는 객체."""

    def __init__(self, model, camera="top", width=640, height=480):
        self.model = model
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        self.width, self.height = width, height
        self.rgb_renderer = mujoco.Renderer(model, height=height, width=width)
        self.depth_renderer = mujoco.Renderer(model, height=height, width=width)
        self.depth_renderer.enable_depth_rendering()
        fovy = np.radians(model.cam_fovy[self.cam_id])
        self.f = (height / 2) / np.tan(fovy / 2)          # 초점 거리 (픽셀 단위)
        self.cx, self.cy = width / 2, height / 2

    def capture(self, data):
        """(rgb [H,W,3] uint8, depth [H,W] float m) 을 돌려준다."""
        self.rgb_renderer.update_scene(data, camera=self.cam_id)
        rgb = self.rgb_renderer.render().copy()
        self.depth_renderer.update_scene(data, camera=self.cam_id)
        depth = self.depth_renderer.render().copy()
        return rgb, depth

    def pixel_to_world(self, data, u, v, depth):
        """픽셀 (u, v) 와 그 픽셀의 depth(카메라 축 방향 거리) → 월드 좌표 (x, y, z)."""
        x_cam = (u - self.cx) / self.f * depth
        y_cam = -(v - self.cy) / self.f * depth            # 이미지 세로는 아래가 +, 카메라 y 는 위가 +
        z_cam = -depth                                      # MuJoCo 카메라는 -z 방향을 본다
        p_cam = np.array([x_cam, y_cam, z_cam])
        R = data.cam_xmat[self.cam_id].reshape(3, 3)
        t = data.cam_xpos[self.cam_id]
        return R @ p_cam + t

    def close(self):
        self.rgb_renderer.close()
        self.depth_renderer.close()


def color_mask(rgb, color):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in COLOR_RANGES[color]:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def find_color_center(rgb, color, min_area=20):
    """RGB 이미지에서 color 의 가장 큰 덩어리 중심 (u, v), 면적, bbox. 없으면 None."""
    mask = color_mask(rgb, color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area:
        return None
    m = cv2.moments(c)
    return {"center": (m["m10"] / m["m00"], m["m01"] / m["m00"]), "area": area,
            "bbox": cv2.boundingRect(c), "mask": mask}


def locate_object(cam, data, color, rgb=None, depth=None):
    """카메라로 찍어 color 물체의 월드 좌표를 추정한다. (world_xyz, 검출 정보) 또는 (None, None)."""
    if rgb is None:
        rgb, depth = cam.capture(data)
    det = find_color_center(rgb, color)
    if det is None:
        return None, None
    u, v = det["center"]
    # 중심 픽셀 주변 3x3 의 depth 중앙값 (가장자리 픽셀의 배경 depth 를 피함)
    ui, vi = int(round(u)), int(round(v))
    patch = depth[max(vi - 1, 0): vi + 2, max(ui - 1, 0): ui + 2]
    d = float(np.median(patch))
    return cam.pixel_to_world(data, u, v, d), det
