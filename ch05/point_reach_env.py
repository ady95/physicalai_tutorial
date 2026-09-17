"""5부 — 가장 단순한 연속 행동 환경: 점(point) Agent가 목표 지점까지 이동하기

Gymnasium 인터페이스(reset / step / observation_space / action_space)를 갖춘 환경입니다.
1부의 GridWorld와 같은 문제지만 위치와 행동이 연속값이고, 강화학습 라이브러리가
그대로 쓸 수 있는 형식입니다.

    observation = [agent_x, agent_y, goal_x, goal_y]      (각 -1 ~ 1)
    action      = [dx, dy]                                (각 -1 ~ 1, 실제 이동은 0.1 배)
    reward      = -거리 (매 스텝)  +10 (목표 5 cm 안에 도달)
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PointReachEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps=100, reward_mode="dense"):
        super().__init__()
        self.max_steps = max_steps
        self.reward_mode = reward_mode
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.agent = np.zeros(2, dtype=np.float32)
        self.goal = np.zeros(2, dtype=np.float32)
        self.t = 0

    def _obs(self):
        return np.concatenate([self.agent, self.goal]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent = self.np_random.uniform(-1, 1, size=2).astype(np.float32)
        self.goal = self.np_random.uniform(-1, 1, size=2).astype(np.float32)
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        self.agent = np.clip(self.agent + 0.1 * action, -1, 1)
        self.t += 1
        dist = float(np.linalg.norm(self.agent - self.goal))
        reached = dist < 0.05

        if self.reward_mode == "dense":
            reward = -dist + (10.0 if reached else 0.0)
        else:                                     # "sparse": 도달했을 때만 보상
            reward = 10.0 if reached else 0.0

        terminated = reached
        truncated = self.t >= self.max_steps
        return self._obs(), reward, terminated, truncated, {"dist": dist, "success": reached}
