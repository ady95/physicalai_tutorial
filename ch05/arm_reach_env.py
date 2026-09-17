"""5부 — SO-101 로봇팔을 Gymnasium 환경으로 감싸기: 손끝을 블록 위로 보내는 과제

    observation = [관절각 5, 손끝 위치 3, 목표(블록 위 2 cm) 위치 3, 목표-손끝 차이 3]   (14)
    action      = 관절 5개의 목표 각도 변화량 (각 -1 ~ 1, 실제로는 0.05 rad 배)
    한 스텝     = 물리 시뮬레이션 10 스텝 (0.02 초)
    episode     = 최대 100 스텝 (2 초). 손끝이 목표 2 cm 안에 들어오면 성공

reward_mode:
    "sparse"  : 성공 시 +10 만
    "dense"   : 매 스텝 -거리
    "shaped"  : 매 스텝 -거리, 손끝이 목표에 가까워진 만큼 +, 성공 시 +10
"""

import os
import sys

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.robot import ARM_JOINTS, GRIPPER_OPEN, SO101Sim  # noqa: E402

TARGET_ABOVE = 0.02       # 블록 중심 위 높이 (3-6의 GRASP_Z와 같은 값)


class ArmReachEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, reward_mode="dense", max_steps=100, substeps=10, seed=None):
        super().__init__()
        self.reward_mode = reward_mode
        self.max_steps = max_steps
        self.substeps = substeps
        self.robot = SO101Sim(render=False)
        self.model, self.data = self.robot.model, self.robot.data
        lo = self.model.jnt_range[[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS], 0]
        hi = self.model.jnt_range[[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS], 1]
        self.ctrl_lo, self.ctrl_hi = lo, hi
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.cube_jnt_adr = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")]
        self.t = 0
        self.prev_dist = None

    # ----- 도우미 -----
    def _target(self):
        c = self.robot.cube_pos()
        return np.array([c[0], c[1], TARGET_ABOVE])

    def _obs(self):
        site = self.robot.site_pose()[0]
        tgt = self._target()
        return np.concatenate([self.robot.q_arm, site, tgt, tgt - site]).astype(np.float32)

    def _dist(self):
        return float(np.linalg.norm(self._target() - self.robot.site_pose()[0]))

    # ----- Gymnasium API -----
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot.reset()
        # 블록을 작업 영역(반지름 0.17~0.27, ±40도) 안 무작위 위치로
        r, th = self.np_random.uniform(0.17, 0.27), self.np_random.uniform(-0.7, 0.7)
        self.data.qpos[self.cube_jnt_adr: self.cube_jnt_adr + 3] = [r * np.cos(th), r * np.sin(th), 0.015]
        self.data.ctrl[self.robot.grip_act] = GRIPPER_OPEN
        mujoco.mj_forward(self.model, self.data)
        self.t = 0
        self.prev_dist = self._dist()
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        ctrl = self.data.ctrl[self.robot.arm_act] + 0.05 * action          # 목표 각도를 조금씩 바꾼다
        self.data.ctrl[self.robot.arm_act] = np.clip(ctrl, self.ctrl_lo, self.ctrl_hi)
        self.robot.step(self.substeps)          # 물리 10 스텝 (렌더링이 켜져 있으면 영상 프레임도 기록)
        self.t += 1

        dist = self._dist()
        success = dist < 0.02
        if self.reward_mode == "sparse":
            reward = 10.0 if success else 0.0
        elif self.reward_mode == "dense":
            reward = -dist
        else:  # shaped
            reward = -dist + 10.0 * (self.prev_dist - dist) + (10.0 if success else 0.0)
        self.prev_dist = dist

        terminated = success
        truncated = self.t >= self.max_steps
        return self._obs(), float(reward), terminated, truncated, {"dist": dist, "success": success}
