"""4부 실습 — YOLO로 물체 검출하기

Ultralytics YOLO의 사전학습 모델(COCO 80 클래스)로 이미지에서 물체를 찾습니다.
1) 실제 사진(bus.jpg, 자동 다운로드) 2) 우리 시뮬레이션 카메라 이미지 에 각각 적용해 봅니다.

실행:
    python ch04/yolo_detect.py
    python ch04/yolo_detect.py --image 내사진.jpg --model yolo11s.pt
"""

import argparse
import os

import cv2
from ultralytics import YOLO

OUT = "outputs/ch04"


def detect(model, source, tag):
    results = model.predict(source, conf=0.25, verbose=False)
    r = results[0]
    print(f"\n[{tag}] {os.path.basename(str(source))}  이미지 크기 {r.orig_shape}, 검출 {len(r.boxes)}개, "
          f"추론 {r.speed['inference']:.1f} ms")
    print(f"  {'class':<14s} {'conf':>5s}   bbox (x1, y1, x2, y2)")
    for b in r.boxes:
        cls = model.names[int(b.cls)]
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        print(f"  {cls:<14s} {float(b.conf):5.2f}   ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
    annotated = r.plot()                      # 박스와 라벨을 그린 BGR 이미지
    path = f"{OUT}/yolo_{tag}.png"
    cv2.imwrite(path, annotated)
    print(f"  저장: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt", help="처음 실행 시 자동 다운로드 (약 5 MB)")
    ap.add_argument("--image", default=None)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    model = YOLO(args.model)
    print(f"model: {args.model}, 클래스 {len(model.names)}개 (예: {list(model.names.values())[:8]} ...)")

    if args.image:
        detect(model, args.image, "custom")
        return

    # 1) 실제 사진
    detect(model, "https://ultralytics.com/images/bus.jpg", "bus")

    # 2) 시뮬레이션 카메라 이미지 (4-6에서 저장한 것)
    for name in ["cam_fixed_rgb.png", "cam_top_rgb.png"]:
        path = f"{OUT}/{name}"
        if os.path.exists(path):
            detect(model, path, name.replace("_rgb.png", ""))
        else:
            print(f"\n{path}가 없습니다. 먼저 ch04/sim_camera.py를 실행하세요.")


if __name__ == "__main__":
    main()
