"""9-3 실습 — LLM이 지휘하는 Pick and Place

3-6의 Rule 기반 Pick and Place와 같은 과제, 같은 블록 위치, 같은 성공 판정입니다.
다른 것은 단 하나, 다섯 단계의 순서를 사람이 적었느냐 LLM이 정하느냐입니다.

    3-6  사람이 적은 상태 기계  →  성공률 90% (10회)
    9-3  LLM이 도구를 골라 진행  →  이 스크립트로 측정

실행:
    python ch09/run_llm_pick.py --no-video                    # 기본 위치 1회
    MUJOCO_GL=egl python ch09/run_llm_pick.py                 # 영상까지 저장
    python ch09/run_llm_pick.py --no-video --trials 10        # 3-6과 같은 무작위 위치 10회
    MUJOCO_GL=egl python ch09/run_llm_pick.py --no-video --trials 10 --perception color
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ch03.pick_and_place import is_in_box  # noqa: E402
from ch09.llm_agent import run_agent  # noqa: E402
from ch09.llm_client import LLMClient, save_trace  # noqa: E402
from ch09.robot_tools import SYSTEM_PROMPT, RobotTools  # noqa: E402
from common.robot import SO101Sim  # noqa: E402
from common.vision import SimCamera  # noqa: E402

TASK = "Pick up the red cube and put it in the blue box."
BOX = (0.05, 0.22)


def run_once(client, cube, perception="state", render=False, max_steps=25, verbose=True):
    robot = SO101Sim(cube_pos=cube, box_pos=BOX, render=render, camera="fixed")
    cam = SimCamera(robot.model, "top", 320, 240) if perception in ("color", "camera") else None
    tools = RobotTools(robot, colors=["red"], perception=perception, cam=cam)

    rec = run_agent(client, tools, TASK, SYSTEM_PROMPT, max_steps=max_steps, verbose=verbose)
    rec["success"] = bool(is_in_box(robot.cube_pos(), robot.box_pos()))
    rec["cube_final"] = [round(float(v), 4) for v in robot.cube_pos()]
    rec["cube_start"] = [round(float(v), 4) for v in cube]

    if cam is not None:
        cam.close()
    return rec, robot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", type=float, nargs=2, default=[0.24, 0.0])
    ap.add_argument("--trials", type=int, default=1, help="2 이상이면 3-6과 같은 방식으로 위치를 무작위화")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perception", default="state", choices=["state", "color", "camera"],
                    help="state 정확한 좌표 / color 4부 색 검출 / camera 사진만 보고 판단")
    ap.add_argument("--max-steps", type=int, default=25)
    ap.add_argument("--model", default=None, help="비우면 환경변수 OPENAI_MODEL")
    ap.add_argument("--verbose", action="store_true", help="반복 모드에서도 도구 호출을 하나씩 출력")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--out", default="outputs/ch09_llm_pick.mp4")
    ap.add_argument("--trace", default="outputs/traces/ch09_llm_pick.json")
    args = ap.parse_args()

    client = LLMClient(model=args.model)
    print(f"모델 {client.model}  perception={args.perception}  스텝 상한 {args.max_steps}\n")

    t0 = time.time()
    records = []
    if args.trials == 1:
        rec, robot = run_once(client, tuple(args.cube), args.perception, render=not args.no_video,
                              max_steps=args.max_steps)
        print(f"\n결과: {'성공' if rec['success'] else '실패'}  (도구 호출 {rec['tool_calls']}회, 종료 {rec['stop']})")
        robot.save_video(args.out)
        robot.close()
        records.append(rec)
    else:
        rng = np.random.default_rng(args.seed)
        for i in range(args.trials):
            r, th = rng.uniform(0.17, 0.27), rng.uniform(-0.7, 0.7)   # 3-6과 같은 부채꼴 영역
            cube = (float(r * np.cos(th)), float(r * np.sin(th)))
            if args.verbose:
                print(f"--- trial {i + 1} 블록 ({cube[0]:+.3f}, {cube[1]:+.3f}) ---")
            rec, robot = run_once(client, cube, args.perception, render=False,
                                  max_steps=args.max_steps, verbose=args.verbose)
            robot.close()
            records.append(rec)
            print(f"trial {i + 1:2d}  블록 ({cube[0]:+.3f}, {cube[1]:+.3f})  도구 {rec['tool_calls']:2d}회  "
                  f"→ {'성공' if rec['success'] else '실패'}  최종 {rec['cube_final']}  ({time.time() - t0:4.0f} s)")
        n_ok = sum(r["success"] for r in records)
        print(f"\n성공률 {n_ok}/{len(records)} = {100 * n_ok / len(records):.0f}%")

    u = client.usage
    n = max(1, len(records))
    print(f"\nLLM 호출 {u['requests']}회  입력 {u['input_tokens']:,} 토큰  출력 {u['output_tokens']:,} 토큰  "
          f"대기 {u['seconds']:.0f} s")
    print(f"에피소드당 평균  호출 {u['requests'] / n:.1f}회  토큰 {(u['input_tokens'] + u['output_tokens']) / n:,.0f}  "
          f"대기 {u['seconds'] / n:.0f} s")
    save_trace(args.trace, {"model": client.model, "perception": args.perception,
                            "usage": u, "episodes": records})


if __name__ == "__main__":
    main()
