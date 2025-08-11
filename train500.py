#!/usr/bin/env python
"""
train.py

使用 Stable-Baselines3 的 SAC 算法训练 KukaK120Env 环境（env2.py），
- 关闭渲染并采用向量化环境并行采样（这里设置为 8 个并行环境），
- 添加评估回调、自定义成功率回调和奖励回调，
- 训练结束后绘制并保存成功率和奖励随回合变化的曲线图，
- 并将训练好的模型保存在 trained_models 文件夹中。
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

# 导入环境类（确保 envs/env1.py 存在并正确实现 Gymnasium 接口）
from envs.env500 import KukaK120Env


# 自定义成功率回调，统计每 1000 个回合中成功回合的比例（定义成功为回合奖励 >= 1.0）
class SuccessRateCallback(BaseCallback):
    def __init__(self, check_freq=1000, verbose=0):
        super(SuccessRateCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_successes = 0
        self.episode_count = 0
        self.success_rates = []

    def _on_step(self) -> bool:
        # Monitor 包装器会在 info 中包含 "episode" 信息
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                if info["episode"].get("is_success",False):  # 成功定义：回合奖励 >= 10.0
                    self.episode_successes += 1
                self.episode_count += 1
                if self.episode_count >= self.check_freq:
                    success_rate = (self.episode_successes / self.episode_count) * 100
                    self.success_rates.append(success_rate)
                    if self.verbose > 0:
                        print(f"Success rate over last {self.episode_count} episodes: {success_rate:.2f}%")
                    self.episode_successes = 0
                    self.episode_count = 0
        return True

    def _on_training_end(self) -> None:
        if self.success_rates:
            plt.figure()
            plt.plot(self.success_rates, marker='o')
            plt.xlabel(f"Block (each block = {self.check_freq} episodes)")
            plt.ylabel("Success Rate (%)")
            plt.title("Success Rate Curve")
            plt.grid(True)
            save_path = os.path.join(os.path.dirname(__file__), "trained_models500", "success_rate.png")
            plt.savefig(save_path)
            # plt.show()
            print("Success rate curve saved to:", save_path)


# 自定义奖励回调，记录每个回合的总奖励并绘制奖励曲线图
class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True

    def _on_training_end(self) -> None:
        if self.episode_rewards:
            plt.figure()
            plt.plot(self.episode_rewards, marker='o')
            plt.xlabel("Episode")
            plt.ylabel("Episode Reward")
            plt.title("Reward Curve")
            plt.grid(True)
            save_path = os.path.join(os.path.dirname(__file__), "trained_models500", "reward_curve.png")
            plt.savefig(save_path)
            # plt.show()
            print("Reward curve saved to:", save_path)


def main():
    # 使用 Monitor 包装环境以记录每个回合的统计数据
    def make_env():
        env = KukaK120Env(render=False)
        return Monitor(env)

    # 创建向量化环境，使用 8 个并行环境（关闭渲染以加速训练）
    num_envs = 1
    vec_env = make_vec_env(make_env, n_envs=num_envs)
    # vec_env = make_vec_env(lambda: KukaK120Env(render=True), n_envs=1)

    # 创建保存模型和日志的目录
    model_dir = os.path.join(os.path.dirname(__file__), "trained_models500")
    os.makedirs(model_dir, exist_ok=True)
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # 创建评估回调（使用单个环境评估）
    eval_env = Monitor(KukaK120Env(render=False))
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=model_dir,
                                 log_path=logs_dir,
                                 eval_freq=10000,
                                 deterministic=True)

    # 创建自定义回调
    success_callback = SuccessRateCallback(check_freq=100, verbose=1)
    reward_callback = RewardCallback(verbose=1)

    # 创建 SAC 模型。默认超参数（actor/critic lr=3e-4, gamma=0.99, buffer_size=1e6, batch_size=256, 等）由库提供。
    # 这里指定 device="cuda" 如果 GPU 可用，否则使用 "cpu"
    device = "cuda"  # 根据你的硬件情况，如 RTX 4060 可用，则设置为 "cuda"
    model = SAC("MlpPolicy", vec_env, ent_coef="auto_1.0", learning_rate=3e-4, batch_size=256, tensorboard_log="./logs/", verbose=1, device=device)

    total_timesteps = 5000000  # 训练总步数，可根据需要调整
    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, success_callback, reward_callback])

    # 保存最终模型
    model_path = os.path.join(model_dir, "SAC_KukaK120_model.zip")
    model.save(model_path)
    print("Model saved to:", model_path)

    vec_env.close()

if __name__ == "__main__":
    main()
