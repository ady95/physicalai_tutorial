"""5부 실습 — 처음으로 Reinforcement Learning 해보기

PointReachEnv 에서 (1) 랜덤 Policy 를 평가하고 (2) PPO 로 학습한 뒤 (3) 다시 평가합니다.
학습 곡선(episode 보상)을 그림으로 저장합니다.

실행:
    python ch05/train_point_reach.py                    # dense reward, 100k steps
    python ch05/train_point_reach.py --reward sparse    # sparse reward 로 비교
    python ch05/train_point_reach.py --steps 300000
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ch05.point_reach_env import PointReachEnv  # noqa: E402

OUT = "outputs/ch05"


def evaluate(env, policy_fn, episodes=100, seed=0):
    """policy_fn(obs) -> action. 성공률, 평균 스텝, 평균 보상을 돌려준다."""
    succ, steps, rewards = 0, [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        total, done = 0.0, False
        while not done:
            obs, r, term, trunc, info = env.step(policy_fn(obs))
            total += r
            done = term or trunc
        succ += int(info["success"])
        steps.append(env.t)
        rewards.append(total)
    return succ / episodes, float(np.mean(steps)), float(np.mean(rewards))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", default="dense", choices=["dense", "sparse"])
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    env = PointReachEnv(reward_mode=args.reward)

    print(f"=== 환경: PointReachEnv (reward={args.reward}) ===")
    print(f"observation_space {env.observation_space}")
    print(f"action_space      {env.action_space}")

    # 1) 랜덤 Policy
    rng = np.random.default_rng(args.seed)
    rate, st, rw = evaluate(env, lambda o: rng.uniform(-1, 1, 2))
    print(f"\n[학습 전: 랜덤 Policy]  성공률 {rate * 100:5.1f}%  평균 스텝 {st:5.1f}  평균 보상 {rw:+8.2f}")

    # 2) PPO 학습
    train_env = Monitor(PointReachEnv(reward_mode=args.reward))
    model = PPO("MlpPolicy", train_env, seed=args.seed, verbose=0,
                n_steps=1024, batch_size=256, learning_rate=3e-4, gamma=0.98)
    print(f"\n[PPO 학습] {args.steps:,} steps ...")
    t0 = time.time()
    model.learn(total_timesteps=args.steps)
    elapsed = time.time() - t0
    print(f"학습 시간 {elapsed:.0f} 초")
    model.save(f"{OUT}/ppo_point_reach_{args.reward}")

    # 3) 학습된 Policy 평가
    policy = lambda o: model.predict(o, deterministic=True)[0]   # noqa: E731
    rate, st, rw = evaluate(env, policy)
    print(f"[학습 후: PPO Policy]   성공률 {rate * 100:5.1f}%  평균 스텝 {st:5.1f}  평균 보상 {rw:+8.2f}")

    # 4) 학습 곡선 저장
    ep_rewards = train_env.get_episode_rewards()
    ep_lengths = train_env.get_episode_lengths()
    if ep_rewards:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        w = 20
        smooth = np.convolve(ep_rewards, np.ones(w) / w, mode="valid")
        x = np.cumsum(ep_lengths)[w - 1:]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot(x, smooth)
        ax.set_xlabel("timesteps")
        ax.set_ylabel(f"episode reward (avg of {w})")
        ax.set_title(f"PPO on PointReachEnv ({args.reward})")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = f"{OUT}/point_reach_curve_{args.reward}.png"
        fig.savefig(path, dpi=120)
        print(f"학습 곡선 저장: {path}  (episode {len(ep_rewards)}개)")
        n = len(ep_rewards)
        for frac in [0.05, 0.25, 0.5, 0.75, 1.0]:
            k = max(1, int(n * frac))
            seg = ep_rewards[max(0, k - 50): k]
            print(f"  진행 {int(frac * 100):3d}%  최근 50 episode 평균 보상 {np.mean(seg):+7.2f}")


if __name__ == "__main__":
    main()
