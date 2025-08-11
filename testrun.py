import os
import time
import numpy as np
from stable_baselines3 import SAC
from envs.env500 import KukaK120Env

# 加载模型路径
MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_models500", "best_model.zip")

# 创建渲染环境
env = KukaK120Env(render=True)

# 加载训练好的模型
model = SAC.load(MODEL_PATH, env=env)

# 重置环境
obs, info = env.reset()

# 运行模型进行可视化验证
for episode in range(1):  # 运行 10 个回合
    obs, info = env.reset()
    done = False
    episode_reward = 0
    step_count = 0

    while not done:
        # 模型预测动作
        action, _states = model.predict(obs, deterministic=True)

        # 执行动作
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        step_count += 1

        # 控制渲染速度
        time.sleep(0.05)  # 可以调整这个时间控制可视化速度

        # 如果达到最大步数或者成功结束
        if done or truncated:
            print(f"Episode {episode + 1}: Total Reward = {episode_reward:.2f}, Steps = {step_count}")
            break

# 关闭环境
env.close()
