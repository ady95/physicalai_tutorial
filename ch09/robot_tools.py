"""9부 — LLM에게 쥐여 줄 로봇 도구 계층.

3부에서 만든 SO101Sim 위에 얇게 한 겹 올린 것뿐입니다. 새로 만드는 제어 코드는 없습니다.
LLM은 "무엇을 할지"만 정하고, IK와 관절 보간은 3부 코드가 그대로 합니다. 이 분리가 9-1의 핵심입니다.

  LLM              0.2 ~ 2 Hz    어느 블록을, 어떤 순서로
  RobotTools       도구 호출     목표점 검사 · 안전 범위 제한
  SO101Sim(IK)     500 Hz        관절각 계산과 보간
"""

import base64
import io

import numpy as np

from common.vision import locate_object

# 작업 영역 — 3-6의 workspace_sweep.py로 확인한 범위. 밖으로 나가는 목표는 잘라내고 LLM에게 알려 준다
R_MIN, R_MAX = 0.12, 0.30
Z_MIN, Z_MAX = 0.005, 0.30

# 이동 시간은 LLM이 아니라 도구가 정한다. 거리에 비례시켜 3-6이 쓴 속도를 넘지 않게 한다.
# 이 값을 빼고 모든 이동을 1초로 고정하면, 상자까지의 긴 대각선 이동에서 관성으로 블록을 놓친다 (9-2).
MOVE_SPEED = 0.15                 # m/s
MOVE_MIN, MOVE_MAX = 0.8, 2.0     # s
SETTLE = 0.2                      # 이동 후 흔들림이 가라앉기를 기다리는 시간 (3-6과 같음)

SYSTEM_PROMPT = """You control a 5-DOF SO-101 robot arm in a MuJoCo simulation.

Coordinates are in meters in the world frame. The arm base is at the origin.
The gripper approaches from above: its fingers always point straight down.

Facts you need:
- A cube is 3 cm wide and its center sits at z = 0.015 when resting on the table.
- To grasp a cube, put the gripper at the cube's x, y and z = 0.02, then close it.
  Closing at a higher z grabs nothing; lower z collides with the table.
- Approach from z = 0.08 above the target before going down. Lift to z = 0.12 before moving sideways,
  otherwise the cube is dragged across the table or knocked over.
- Release above the box at about z = 0.06.
- Reachable area: distance from origin between 0.12 and 0.30 m. Targets outside are clamped.

Reading the result of a move:
- position_error is how far the gripper ended from the point you asked for, in meters.
  Below 0.02 is fine. Above that, the point is hard to reach with the fingers pointing down.
- tilted means the arm could not keep the fingers exactly vertical. This is normal near the edge
  of the workspace and does NOT mean the grasp failed. Ignore it unless you are about to grasp.
- Never re-send a move to the position the gripper actually reached. That is where it already is,
  and repeating it pushes the arm further off. If a point is not reachable, pick a different one,
  usually a lower z, or just carry on with the task.

To check whether you are holding the cube, call get_scene_state and look at the cube's z.
A cube on the table sits at z = 0.015; a lifted cube is clearly higher.

Work one step at a time. When the task is finished or you cannot continue, call report_done."""


def _xyz(v):
    return [round(float(x), 4) for x in v]


class RobotTools:
    """도구 이름 → 실제 동작. perception 이 LLM에게 보이는 세상을 정한다.

    state   시뮬레이터의 정확한 좌표를 그대로 알려준다 (3부와 같은 조건)
    color   4부의 색 검출로 추정한 좌표를 알려준다 (오차가 있다)
    camera  좌표를 주지 않는다. LLM이 look()으로 사진을 보고 스스로 판단한다
    """

    def __init__(self, robot, colors=("red",), perception="state", cam=None):
        self.robot = robot
        self.colors = list(colors)
        self.perception = perception
        self.cam = cam
        self.calls = []                 # (name, args, result) 기록
        self.pending_image = None       # look() 이 찍어 둔 사진, 다음 요청에 붙는다

    # ---------- 도구 정의 ----------
    def schema(self):
        tools = [
            {"type": "function", "name": "get_scene_state",
             "description": "Current gripper pose, gripper open/closed, and what is known about the objects.",
             "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
             "strict": True},
            {"type": "function", "name": "move_gripper_to",
             "description": "Move the gripper to a world position (meters). Fingers stay pointing down.",
             "parameters": {"type": "object",
                            "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}},
                            "required": ["x", "y", "z"], "additionalProperties": False},
             "strict": True},
            {"type": "function", "name": "set_gripper",
             "description": "Open or close the gripper.",
             "parameters": {"type": "object",
                            "properties": {"state": {"type": "string", "enum": ["open", "closed"]}},
                            "required": ["state"], "additionalProperties": False},
             "strict": True},
            {"type": "function", "name": "report_done",
             "description": "Call when the task is complete or you cannot continue.",
             "parameters": {"type": "object",
                            "properties": {"success": {"type": "boolean"}, "note": {"type": "string"}},
                            "required": ["success", "note"], "additionalProperties": False},
             "strict": True},
        ]
        if self.perception == "camera":
            tools.insert(1, {"type": "function", "name": "look",
                             "description": "Take a photo with the overhead camera and look at it.",
                             "parameters": {"type": "object", "properties": {}, "required": [],
                                            "additionalProperties": False},
                             "strict": True})
        return tools

    # ---------- 도구 실행 ----------
    def call(self, name, args):
        fn = getattr(self, f"_t_{name}", None)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            result = fn(**args)
        except TypeError as e:
            result = {"error": f"bad arguments: {e}"}
        self.calls.append({"name": name, "args": args, "result": result})
        return result

    def _t_get_scene_state(self):
        pos, _ = self.robot.site_pose()
        state = {"gripper_xyz": _xyz(pos),
                 "gripper": "open" if self.robot.gripper_opening() > 0.5 else "closed",
                 "box_xyz": _xyz(self.robot.box_pos())}
        if self.perception == "state":
            state["objects"] = {c: _xyz(self.robot.cube_pos(c)) for c in self.colors}
        elif self.perception == "color":
            seen = {}
            for c in self.colors:
                p, _ = locate_object(self.cam, self.robot.data, c)
                if p is not None:
                    seen[c] = _xyz(p)
            state["objects"] = seen
            state["note"] = "positions estimated from the camera, accurate to about 1 cm"
        else:
            state["note"] = "object positions are not available; call look() to see the scene"
        return state

    def _t_look(self):
        rgb, _ = self.cam.capture(self.robot.data)
        self.pending_image = encode_png(rgb)
        h, w = rgb.shape[:2]
        return {"image": "attached below", "width": w, "height": h,
                "note": "overhead camera looking straight down at the table"}

    def _t_move_gripper_to(self, x, y, z):
        target = np.array([float(x), float(y), float(z)])
        target, clamped = clamp_target(target)
        here, _ = self.robot.site_pose()
        seconds = float(np.clip(np.linalg.norm(target - here) / MOVE_SPEED, MOVE_MIN, MOVE_MAX))
        _, err = self.robot.move_to(target, seconds=seconds)
        self.robot.hold(SETTLE)
        pos, _ = self.robot.site_pose()
        # solve_ik 의 err 는 위치 오차와 방향 오차를 함께 묶은 값이라 그대로 주면 오해를 부른다 (9-2).
        # position_error 는 실제로 측정한 값이고,
        # tilted 는 그 IK 잔차가 큰지로 판단하는 대리 지표다. 손끝 회전을 직접 잰 각도가 아니라
        # 위치 오차가 커도 True 가 될 수 있다. 정확히 재려면 site_pose() 의 손가락 축과
        # 수직 방향 사이의 각도를 쓰면 되지만, 그러면 LLM에게 보이는 값이 달라져
        # 9-3 의 측정과 비교할 수 없게 되므로 원고의 실측과 같은 정의를 유지한다 (4차 검수 #29).
        out = {"gripper_xyz": _xyz(pos),
               "position_error": round(float(np.linalg.norm(np.asarray(pos) - target)), 4),
               "tilted": bool(err > 0.1),
               "seconds": round(seconds, 2)}
        if clamped:
            out["warning"] = f"target was outside the reachable area, clamped to {_xyz(target)}"
        return out

    def _t_set_gripper(self, state):
        if state == "open":
            self.robot.open_gripper(0.5)
        else:
            self.robot.close_gripper(0.8)
        self.robot.hold(SETTLE)
        return {"gripper": state, "opening_rad": round(self.robot.gripper_opening(), 3)}

    def _t_report_done(self, success, note):
        return {"acknowledged": True, "claimed_success": bool(success), "note": note}


def clamp_target(p):
    """작업 영역 밖의 목표를 잘라낸다. (잘라낸 목표, 잘렸는지)"""
    q = p.copy()
    r = float(np.linalg.norm(q[:2]))
    if r < 1e-6:
        q[0], r = R_MIN, R_MIN
    if not R_MIN <= r <= R_MAX:
        q[:2] *= np.clip(r, R_MIN, R_MAX) / r
    q[2] = np.clip(q[2], Z_MIN, Z_MAX)
    return q, bool(not np.allclose(q, p))


def encode_png(rgb):
    """RGB 배열 → data: URL. Responses API의 input_image 가 받는 형식."""
    import imageio.v3 as iio

    buf = io.BytesIO(iio.imwrite("<bytes>", rgb, extension=".png"))
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
