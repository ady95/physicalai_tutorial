"""9-5 프로젝트 — 8-1에서 SmolVLA가 못 한 과제를 LLM Planner로

8-1과 8-1-1에서 SmolVLA는 블록 세 개 중 문장이 가리키는 색을 고르지 못했습니다.
179 Episode · 20,000스텝으로도 사실상 0%였습니다. 같은 장면, 같은 문장, 같은 판정으로
이번에는 LLM에게 도구를 쥐여 주고 시켜 봅니다.

비교가 성립하도록 8-1과 완전히 같은 것을 씁니다.
  - 블록 배치        ch08.record_multicolor.random_layout, 같은 seed
  - 명령 문장        "Pick up the {color} cube and put it in the blue box."
  - 성공/오답 판정   ch03.pick_and_place.is_in_box

실행:
    MUJOCO_GL=egl python ch09/eval_llm_multicolor.py --layouts 20 --perception state
    MUJOCO_GL=egl python ch09/eval_llm_multicolor.py --layouts 20 --perception camera
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ch03.pick_and_place import is_in_box  # noqa: E402
from ch08.record_multicolor import COLORS, TASK_FMT, random_layout  # noqa: E402
from ch09.llm_agent import run_agent  # noqa: E402
from ch09.llm_client import LLMClient, save_trace  # noqa: E402
from ch09.robot_tools import SYSTEM_PROMPT, RobotTools  # noqa: E402
from common.robot import SO101Sim  # noqa: E402
from common.vision import SimCamera  # noqa: E402

BOX = (0.05, 0.22)
MOVED = 0.01          # 1 cm 넘게 움직인 블록을 "건드린 블록"으로 본다


def run_episode(client, layout, color, perception="state", max_steps=30, verbose=False):
    red, green, yellow = layout
    robot = SO101Sim(cube_pos=red, box_pos=BOX, render=False, camera="fixed",
                     extra_cubes=[("green", green), ("yellow", yellow)])
    cam = SimCamera(robot.model, "top", 320, 240) if perception in ("color", "camera") else None
    tools = RobotTools(robot, colors=COLORS, perception=perception, cam=cam)
    start = {c: robot.cube_pos(c).copy() for c in COLORS}

    rec = run_agent(client, tools, TASK_FMT.format(color=color), SYSTEM_PROMPT,
                    max_steps=max_steps, verbose=verbose)

    box = robot.box_pos()
    in_box = [c for c in COLORS if is_in_box(robot.cube_pos(c), box)]
    if in_box == [color]:
        rec["result"] = "성공"
    elif in_box:
        rec["result"] = f"오답({','.join(in_box)})"
    else:
        rec["result"] = "실패"

    # 8-1-1의 "언어 일치"와 같은 뜻: 문장이 가리킨 블록을 실제로 건드렸는가
    moved = {c: float(np.linalg.norm(robot.cube_pos(c) - start[c])) for c in COLORS}
    touched = max(moved, key=moved.get) if max(moved.values()) > MOVED else None
    rec["touched"] = touched
    rec["language_match"] = touched == color
    rec["moved_cm"] = {c: round(100 * v, 1) for c, v in moved.items()}

    if cam is not None:
        cam.close()
    robot.close()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", type=int, default=20, help="배치 수. 배치마다 세 색을 차례로 시킨다")
    ap.add_argument("--seed", type=int, default=2000, help="8-1의 평가와 같은 배치 seed")
    ap.add_argument("--perception", default="state", choices=["state", "color", "camera"])
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--model", default=None)
    ap.add_argument("--trace", default="outputs/traces/ch09_multicolor.json")
    args = ap.parse_args()

    client = LLMClient(model=args.model)
    print(f"모델 {client.model}  perception={args.perception}  배치 {args.layouts}개 × 색 3가지\n")

    rng = np.random.default_rng(args.seed)
    stats = {c: {"성공": 0, "오답": 0, "실패": 0} for c in COLORS}
    records, t0 = [], time.time()

    for i in range(args.layouts):
        layout = random_layout(rng)                      # 8-1과 같은 배치 생성기
        for color in COLORS:
            rec = run_episode(client, layout, color, args.perception, args.max_steps)
            if len(records) >= 6:               # trace 파일이 커지지 않도록 앞 6개만 대화 전문을 남긴다
                rec.pop("transcript", None)
            rec["layout"], rec["color"] = i, color
            records.append(rec)
            stats[color][rec["result"].split("(")[0]] += 1
            print(f"layout {i + 1:2d}  \"{TASK_FMT.format(color=color)}\"  → {rec['result']:<12s} "
                  f"건드린 블록 {rec['touched']}   ({time.time() - t0:5.0f} s)")

    n = args.layouts * 3
    n_ok = sum(s["성공"] for s in stats.values())
    n_match = sum(r["language_match"] for r in records)
    print(f"\n{'색':<8s} {'성공':>5s} {'오답':>5s} {'실패':>5s}")
    for c in COLORS:
        s = stats[c]
        print(f"{c:<8s} {s['성공']:5d} {s['오답']:5d} {s['실패']:5d}")
    print(f"\n전체 성공률 {n_ok}/{n} = {100 * n_ok / n:.0f}%")
    print(f"언어 일치   {n_match}/{n} = {100 * n_match / n:.0f}%   (8-1-1의 SmolVLA 179ep: 18/60 = 30%)")

    u = client.usage
    print(f"\nLLM 호출 {u['requests']}회  토큰 {u['input_tokens'] + u['output_tokens']:,}  대기 {u['seconds']:.0f} s")
    print(f"에피소드당  토큰 {(u['input_tokens'] + u['output_tokens']) / n:,.0f}  대기 {u['seconds'] / n:.0f} s")
    save_trace(args.trace, {"model": client.model, "perception": args.perception, "seed": args.seed,
                            "usage": u, "stats": stats, "episodes": records})


if __name__ == "__main__":
    main()
