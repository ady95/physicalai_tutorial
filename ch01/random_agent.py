"""1부 실습 2 — 랜덤하게 움직이는 Agent 만들기

2차원 격자 세계(GridWorld)에서 Observation / State / Action / Reward / Episode 를
코드로 확인합니다. 랜덤 Policy와 규칙 기반 Policy를 여러 Episode 돌려 비교합니다.

실행:
    python ch01/random_agent.py
"""

import random

# 행동(Action) 정의: 상하좌우 4가지
ACTIONS = {
    0: (0, -1),   # 위
    1: (0, +1),   # 아래
    2: (-1, 0),   # 왼쪽
    3: (+1, 0),   # 오른쪽
}
ACTION_NAMES = {0: "위", 1: "아래", 2: "왼쪽", 3: "오른쪽"}


class GridWorld:
    """size x size 격자 세계. Agent가 goal 칸에 도달하면 Episode가 끝납니다."""

    def __init__(self, size=5, max_steps=50, seed=None):
        self.size = size
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        """새 Episode 시작. Agent와 Goal 위치를 무작위로 놓습니다."""
        self.agent = (0, 0)
        self.goal = (self.size - 1, self.size - 1)
        self.t = 0
        return self.observe()

    def observe(self):
        return {"agent": self.agent, "goal": self.goal}

    def step(self, action):
        """행동을 적용하고 (observation, reward, done) 을 돌려줍니다."""
        dx, dy = ACTIONS[action]
        x = min(self.size - 1, max(0, self.agent[0] + dx))
        y = min(self.size - 1, max(0, self.agent[1] + dy))
        self.agent = (x, y)
        self.t += 1

        if self.agent == self.goal:
            reward, done = +10.0, True       # 목표 도달
        elif self.t >= self.max_steps:
            reward, done = -1.0, True        # 시간 초과
        else:
            reward, done = -0.1, False       # 한 스텝 쓸 때마다 작은 벌점

        return self.observe(), reward, done

    def render(self):
        rows = []
        for y in range(self.size):
            row = ""
            for x in range(self.size):
                if (x, y) == self.agent:
                    row += "A "
                elif (x, y) == self.goal:
                    row += "G "
                else:
                    row += ". "
            rows.append(row)
        return "\n".join(rows)


def random_policy(observation):
    """무엇을 보든 아무 행동이나 고릅니다."""
    return random.choice(list(ACTIONS.keys()))


def rule_policy(observation):
    """목표 쪽으로 한 칸 움직이는 규칙."""
    (ax, ay), (gx, gy) = observation["agent"], observation["goal"]
    if gx > ax:
        return 3   # 오른쪽
    if gx < ax:
        return 2   # 왼쪽
    if gy > ay:
        return 1   # 아래
    return 0       # 위


def run_episode(env, policy, verbose=False):
    """Episode 하나를 끝까지 돌리고 (총 보상, 스텝 수, 성공 여부)를 돌려줍니다."""
    observation = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action = policy(observation)
        observation, reward, done = env.step(action)
        total_reward += reward
        if verbose:
            print(f"t={env.t:2d} action={ACTION_NAMES[action]:<3s} reward={reward:+.1f}")
            print(env.render())
            print()
    success = env.agent == env.goal
    return total_reward, env.t, success


def evaluate(policy, name, episodes=100, seed=0):
    random.seed(seed)
    env = GridWorld(size=5, max_steps=50)
    rewards, steps, successes = [], [], 0
    for _ in range(episodes):
        r, t, ok = run_episode(env, policy)
        rewards.append(r)
        steps.append(t)
        successes += int(ok)
    print(f"[{name}] {episodes} episodes | 성공률 {successes / episodes * 100:5.1f}% "
          f"| 평균 스텝 {sum(steps) / episodes:5.1f} | 평균 보상 {sum(rewards) / episodes:+6.2f}")


if __name__ == "__main__":
    print("=== 랜덤 Policy, Episode 1개 자세히 보기 ===")
    random.seed(0)
    env = GridWorld(size=5, max_steps=50)
    total, t, ok = run_episode(env, random_policy, verbose=False)
    print(f"총 보상 {total:+.1f}, 스텝 {t}, 성공 {ok}")
    print(env.render())
    print()

    print("=== 100 Episode 통계 비교 ===")
    evaluate(random_policy, "random_policy")
    evaluate(rule_policy, "rule_policy")
