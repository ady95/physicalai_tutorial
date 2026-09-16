"""1부 실습 1 — AI 없는 가상 Agent 만들기: 세 줄의 코드

책 전체의 출발점입니다.

    observation = env.observe()
    action = policy(observation)
    env.step(action)

가장 단순한 1차원 세계(LineWorld)에서 Agent가 목표 지점까지 이동합니다.
AI는 없습니다. policy는 사람이 짠 규칙(if문)입니다.

실행:
    python ch01/agent_3lines.py
"""


class LineWorld:
    """0부터 9까지의 눈금이 있는 1차원 세계.

    Agent는 position에 서 있고, goal 위치로 가는 것이 목표입니다.
    """

    def __init__(self, start=0, goal=7):
        self.position = start
        self.goal = goal
        self.t = 0  # 시간(스텝) 카운터

    def observe(self):
        """Agent가 세계로부터 받는 관측(Observation)."""
        return {"position": self.position, "goal": self.goal}

    def step(self, action):
        """행동(Action)을 세계에 적용합니다. action은 -1(왼쪽) 또는 +1(오른쪽)."""
        self.position = max(0, min(9, self.position + action))
        self.t += 1

    def done(self):
        return self.position == self.goal

    def render(self):
        """텍스트로 세계를 그립니다. A = Agent, G = Goal."""
        cells = []
        for i in range(10):
            if i == self.position:
                cells.append("A")
            elif i == self.goal:
                cells.append("G")
            else:
                cells.append(".")
        return "".join(cells)


def policy(observation):
    """규칙 기반 Policy: 목표가 오른쪽이면 +1, 왼쪽이면 -1."""
    if observation["goal"] > observation["position"]:
        return +1
    return -1


if __name__ == "__main__":
    env = LineWorld(start=0, goal=7)
    print(f"t={env.t}  {env.render()}")

    while not env.done():
        observation = env.observe()      # 1. 관측
        action = policy(observation)     # 2. 판단
        env.step(action)                 # 3. 행동
        print(f"t={env.t}  {env.render()}  action={action:+d}")

    print(f"목표 도달! 총 {env.t} 스텝")
