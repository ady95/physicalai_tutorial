"""4부 실습 — 영상에서 특정 색상의 물체 찾기

HSV 색 공간에서 "빨강" 범위의 마스크를 만들고, 그 윤곽(contour)의 중심을 구해
영상 내내 빨간 블록을 추적합니다. 결과를 표시한 영상을 저장합니다.

실행:
    python ch04/color_detect.py                        # outputs/ch03_pick_and_place.mp4
    python ch04/color_detect.py --video 내영상.mp4 --color blue
"""

import argparse
import os

import cv2
import numpy as np

# HSV 범위 (OpenCV의 H는 0~179). 빨강은 0 근처와 179 근처 두 구간에 걸쳐 있다.
COLOR_RANGES = {
    "red": [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    "blue": [((100, 120, 70), (130, 255, 255))],
    "yellow": [((20, 120, 70), (35, 255, 255))],
}


def find_color(frame_bgr, color="red", min_area=30):
    """색 마스크 → 가장 큰 덩어리의 중심 (u, v)와 면적. 없으면 None."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in COLOR_RANGES[color]:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))   # 작은 점 제거

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area:
        return None, mask
    m = cv2.moments(c)
    u, v = m["m10"] / m["m00"], m["m01"] / m["m00"]     # 중심 (가로 u, 세로 v)
    x, y, w, h = cv2.boundingRect(c)
    return {"center": (u, v), "area": area, "bbox": (x, y, w, h)}, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    ap.add_argument("--color", default="red", choices=list(COLOR_RANGES))
    ap.add_argument("--out", default="outputs/ch04_color_detect.mp4")
    args = ap.parse_args()

    path = args.video or ("outputs/ch03_pick_and_place.mp4" if os.path.exists("outputs/ch03_pick_and_place.mp4")
                          else "assets/sample_pick_and_place.mp4")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"열 수 없음: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print(f"source: {path}  ({w}x{h}, {fps:.0f} fps)")
    print(f"{'frame':>5s} {'center u':>9s} {'center v':>9s} {'area':>7s}")
    found, total, i = 0, 0, 0
    last_mask, last_frame = None, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        det, mask = find_color(frame, args.color)
        last_mask, last_frame = mask, frame
        total += 1
        if det is not None:
            found += 1
            u, v = det["center"]
            x, y, bw, bh = det["bbox"]
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.circle(frame, (int(u), int(v)), 4, (0, 255, 255), -1)
            cv2.putText(frame, f"{args.color} ({u:.0f},{v:.0f})", (x, max(y - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if i % 30 == 0:
                print(f"{i:5d} {u:9.1f} {v:9.1f} {det['area']:7.0f}")
        elif i % 30 == 0:
            print(f"{i:5d} {'-':>9s} {'-':>9s} {'-':>7s}")
        writer.write(frame)
        i += 1
    cap.release()
    writer.release()
    if last_mask is not None:
        cv2.imwrite(args.out.replace(".mp4", "_mask.png"), last_mask)
        cv2.imwrite(args.out.replace(".mp4", "_last.png"), last_frame)
    print(f"\n검출 {found}/{total} 프레임,  저장: {args.out}")


if __name__ == "__main__":
    main()
