"""9부 — 관찰 → 판단 → 행동 루프를 LLM으로 닫기.

1-2에서 세 줄로 적었던 루프가 그대로입니다. policy 자리에 LLM이 들어갔을 뿐입니다.

    obs = get_scene_state()      관찰   ← 도구 호출의 결과
    action = LLM(obs)            판단   ← 다음에 부를 도구를 LLM이 고른다
    robot.execute(action)        행동   ← 3부의 IK와 관절 보간

성공 여부는 LLM의 주장(report_done)이 아니라 시뮬레이터의 좌표로 판정합니다.
"""

import json

from ch09.llm_client import strip_images

STOP_REASONS = {"done": "report_done 호출", "max_steps": "스텝 상한 도달", "no_call": "도구를 부르지 않음"}


def run_agent(client, tools, task, system_prompt, max_steps=25, verbose=True):
    """도구를 부르는 대화를 끝까지 돌린다. 판정은 하지 않고 진행 기록만 돌려준다."""
    items = [{"role": "system", "content": system_prompt},
             {"role": "user", "content": task}]
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    stop, claimed = "max_steps", None

    for step in range(max_steps):
        output, dt = client.respond(items, tools.schema())
        items += output

        calls = [o for o in output if o.get("type") == "function_call"]
        for o in output:
            if o.get("type") == "message":
                text = " ".join(c.get("text", "") for c in o.get("content", []))
                if text.strip():
                    log(f"  [{step + 1:2d}] 말: {text.strip()[:100]}")
        if not calls:
            stop = "no_call"
            break

        for c in calls:
            args = json.loads(c.get("arguments") or "{}")
            result = tools.call(c["name"], args)
            log(f"  [{step + 1:2d}] {c['name']}({_fmt(args)}) → {_fmt(result)}   ({dt:.1f}s)")
            items.append({"type": "function_call_output", "call_id": c["call_id"],
                          "output": json.dumps(result, ensure_ascii=False)})
            if tools.pending_image:                     # look() 이 찍은 사진을 다음 요청에 붙인다
                items.append({"role": "user", "content": [
                    {"type": "input_text", "text": "Here is the photo you asked for."},
                    {"type": "input_image", "image_url": tools.pending_image}]})
                tools.pending_image = None
            if c["name"] == "report_done":
                stop, claimed = "done", bool(args.get("success"))
        if stop == "done":
            break

    return {"stop": stop, "claimed_success": claimed, "steps": step + 1,
            "tool_calls": len(tools.calls), "transcript": strip_images(items)}


def _fmt(d):
    if not isinstance(d, dict):
        return str(d)
    parts = []
    for k, v in d.items():
        if isinstance(v, float):
            v = round(v, 3)
        parts.append(f"{k}={v}")
    return " ".join(parts)[:110]
