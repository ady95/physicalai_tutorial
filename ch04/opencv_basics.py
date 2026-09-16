"""4부 실습 — OpenCV로 카메라(이미지·영상) 다루기

실물 카메라 없이 3부에서 저장한 시뮬레이션 영상을 카메라 대신 사용합니다.
이미지 읽기/쓰기, 영상의 프레임 읽기, 해상도·FPS 확인, 색 공간(BGR/RGB/HSV) 변환을 익힙니다.

실행:
    python ch04/opencv_basics.py                       # 기본: outputs/ch03_pick_and_place.mp4
    python ch04/opencv_basics.py --video 내영상.mp4
    python ch04/opencv_basics.py --camera 0            # 웹캠이 있다면
"""

import argparse
import os

import cv2
import numpy as np

DEFAULT_VIDEO = "outputs/ch03_pick_and_place.mp4"
FALLBACK_VIDEO = "assets/sample_pick_and_place.mp4"


def open_source(args):
    if args.camera is not None:
        return cv2.VideoCapture(args.camera), f"camera {args.camera}"
    path = args.video or (DEFAULT_VIDEO if os.path.exists(DEFAULT_VIDEO) else FALLBACK_VIDEO)
    return cv2.VideoCapture(path), path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--out", default="outputs/ch04")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cap, name = open_source(args)
    if not cap.isOpened():
        raise SystemExit(f"열 수 없음: {name}  (먼저 3부의 pick_and_place.py 를 실행해 영상을 만드세요)")

    # 1) 영상 정보
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"source: {name}")
    print(f"resolution: {w} x {h},  fps: {fps:.1f},  frames: {n}  (약 {n / fps:.1f} 초)")

    # 2) 프레임 하나 읽기
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 3)          # 1/3 지점으로 이동
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("프레임을 읽지 못했습니다")
    print(f"frame: shape={frame.shape}, dtype={frame.dtype}  (높이, 너비, 채널) — OpenCV 는 BGR 순서")
    print(f"픽셀 (240, 320) 의 BGR 값 = {frame[240, 320]}")

    # 3) 색 공간 변환
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    print(f"같은 픽셀의 RGB = {rgb[240, 320]},  GRAY = {gray[240, 320]},  HSV = {hsv[240, 320]}")

    # 4) 저장
    cv2.imwrite(f"{args.out}/frame_bgr.png", frame)
    cv2.imwrite(f"{args.out}/frame_gray.png", gray)
    cv2.imwrite(f"{args.out}/frame_hsv_h.png", hsv[:, :, 0])   # Hue 채널만
    small = cv2.resize(frame, (w // 4, h // 4))
    cv2.imwrite(f"{args.out}/frame_small.png", small)
    print(f"저장: {args.out}/frame_bgr.png, frame_gray.png, frame_hsv_h.png, frame_small.png ({small.shape[1]}x{small.shape[0]})")

    # 5) 영상 전체 훑기: 프레임마다 평균 밝기
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    brightness = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        brightness.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean())
    cap.release()
    b = np.array(brightness)
    print(f"전체 {len(b)} 프레임 평균 밝기: min {b.min():.1f}, max {b.max():.1f}, mean {b.mean():.1f}")


if __name__ == "__main__":
    main()
