"""5부 실습 — 로봇팔을 강화학습으로 움직여보자

ArmReachEnv 에서 PPO 로 "손끝을 블록 위로" 보내는 Policy 를 학습합니다.
Reward 설계(sparse / dense / shaped)에 따라 결과가 어떻게 달라지는지 비교하는 것이 목적입니다.

실행:
    python ch05/train_arm_reach.py --reward dense --steps 300000
    python ch05/train_arm_reach.py --reward sparse --steps 300000
    MUJOCO_GL=egl python ch05/train_arm_reach.py --eval-only --reward dense   # 학습된 모델로 영상 저장
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ch05.arm_reach_env import ArmReachEnv  # noqa: E402

OUT = "outputs/ch05"


def evaluate(env, policy_fn, episodes=50, seed=1000):
    succ, dists, steps = 0, [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(policy_fn(obs))
            done = term or trunc
        succ += int(info["success"])
        dists.append(info["dist"])
        steps.append(env.t)
    return succ / episodes, float(np.mean(dists)), float(np.mean(steps))


def make_env(reward_mode):
    def _f():
        from stable_baselines3.common.monitor import Monitor
        return Monitor(ArmReachEnv(reward_mode=reward_mode))
    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", default="dense", choices=["sparse", "dense", "shaped"])
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv

    model_path = f"{OUT}/ppo_arm_reach_{args.reward}"
    eval_env = ArmReachEnv(reward_mode=args.reward)

    if not args.eval_only:
        print(f"=== ArmReachEnv (reward={args.reward}) ===")
        print(f"observation {eval_env.observation_space.shape}, action {eval_env.action_space.shape}")
        rng = np.random.default_rng(args.seed)
        rate, d, st = evaluate(eval_env, lambda o: rng.uniform(-1, 1, 5), episodes=20)
        print(f"[학습 전: 랜덤 Policy]  성공률 {rate * 100:5.1f}%  최종 거리 {d * 100:5.1f} cm  평균 스텝 {st:5.1f}")

        vec = SubprocVecEnv([make_env(args.reward) for _ in range(args.n_envs)])
        model = PPO("MlpPolicy", vec, seed=args.seed, verbose=0, n_steps=512, batch_size=512,
                    learning_rate=3e-4, gamma=0.98, ent_coef=0.0, device="cpu")
        print(f"\n[PPO 학습] {args.steps:,} steps, {args.n_envs} envs ...")
        t0 = time.time()
        # 진행 상황을 25% 마다 출력
        chunk = args.steps // 4
        for k in range(4):
            model.learn(total_timesteps=chunk, reset_num_timesteps=(k == 0))
            rate, d, st = evaluate(eval_env, lambda o: model.predict(o, deterministic=True)[0], episodes=20)
            print(f"  {int((k + 1) * 25):3d}%  ({time.time() - t0:5.0f} s)  성공률 {rate * 100:5.1f}%  "
                  f"최종 거리 {d * 100:5.1f} cm  평균 스텝 {st:5.1f}")
        print(f"학습 시간 {time.time() - t0:.0f} 초")
        model.save(model_path)
        vec.close()
    else:
        model = PPO.load(model_path)

    rate, d, st = evaluate(eval_env, lambda o: model.predict(o, deterministic=True)[0], episodes=50)
    print(f"\n[학습 후: PPO Policy, 50 episodes]  성공률 {rate * 100:5.1f}%  최종 거리 {d * 100:5.1f} cm  평균 스텝 {st:5.1f}")

    if args.eval_only:
        # 영상 저장: 렌더링이 켜진 로봇으로 한 episode
        from common.robot import SO101Sim
        env = ArmReachEnv(reward_mode=args.reward)
        env.robot = SO101Sim(render=True, camera="fixed")
        env.model, env.data = env.robot.model, env.robot.data
        obs, _ = env.reset(seed=7)
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0])
            done = term or trunc
        env.robot.hold(0.5)
        print(f"영상용 episode: 성공 {info['success']}, 최종 거리 {info['dist'] * 100:.1f} cm, 스텝 {env.t}")
        env.robot.save_video(f"{OUT}/arm_reach_{args.reward}.mp4")
        env.robot.close()


if __name__ == "__main__":
    main()
