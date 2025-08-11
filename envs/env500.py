#!/usr/bin/env python
"""
env1.py

Gymnasium 环境：
  - 加载已有的机器人 URDF（路径为 ../urdf/kr120r2500pro.urdf）；
  - 环境内生成一个大圆柱体（障碍物）和一个红色盒子（目标），
    其中红盒子每回合随机生成，位置范围：x=1.5, y∈[-0.2,0.2], z∈[0.4,0.8]；
  - 状态空间 = [
         joint_angles (6维),                # 各受控关节的当前角度
         end_effector_pos (3维),            # 末端执行器的位置
         target_pos - end_effector_pos (3维),# 目标与末端的相对位置
         obstacle_pos (3维) + obstacle_vel (3维)  # 障碍物中心位置及其线速度
      ] 共18维；
  - 动作空间 = 关节角度增量（6维向量，单位：rad，范围 ±0.1）。
  - 奖励：直接以末端与目标的绝对距离作为奖励，即 reward = -d_box - 0.01，
      如果末端与目标距离小于0.02，则额外奖励100并结束回合。
"""

import os
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '..')
sys.path.append(os.path.abspath(parent_dir))
import time
import math
import gymnasium as gym  # 使用 gymnasium
import numpy as np
from gymnasium import spaces
import pybullet as p
import pybullet_data
import random

# 从 utils 模块导入碰撞检测及最小距离计算函数
from utils import check_collision, compute_min_distance

class KukaK120Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render=True):
        super(KukaK120Env, self).__init__()
        self.render_mode = render
        if self.render_mode:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        # 设置 PyBullet 搜索路径、重力与步长
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.timeStep = 1.0 / 240.0  # 每步仿真时间约0.00417秒（240Hz仿真频率）
        p.setTimeStep(self.timeStep)
        self.maxSteps = 500
        self.currentStep = 0

        # 机器人 URDF
        self.robot_urdf = os.path.join(os.path.dirname(__file__), "..", "urdf", "kr120r2500pro.urdf")
        self.robotStartPos = [0, 0, 0]
        self.robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

        # 障碍物（大圆柱体），放远一些避免初始碰撞
        self.cylinderStartPos = [1.2, 0, 0.6]
        # self.cylinderStartPos = [10.0, 10.0, 0.6]
        self.cylinder_radius = 0.2
        self.cylinder_height = 1.2

        # 目标（红盒子）的位置，每回合随机，x固定为1.5, y∈[-0.2, 0.2], z∈[0.4, 0.8]
        self.box_half_extents = [0.01, 0.01, 0.01]
        self.boxStartPos = [1.8, 0, 1.0]  # 在 reset() 中更新

        # 逆运动学参数：假定末端执行器的 link index 为 6；目标朝向固定
        self.endEffectorLinkIndex = 6
        # self.robotEndOrientation = p.getQuaternionFromEuler([1.57, 0, 1.57])

        # 动作空间：关节角度增量（6维，单位：rad，范围 ±0.1）
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(6,), dtype=np.float32)
        # 状态空间：18维，由6维关节角度 + 3维末端位置 + 3维(目标-末端) + 3维障碍物位置 + 3维障碍物线速度组成
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        # 加载机器人（固定底座）
        self.robot = p.loadURDF(self.robot_urdf, self.robotStartPos, self.robotStartOrientation, useFixedBase=True)
        p.changeVisualShape(self.robot, -1, rgbaColor=[0.2, 0.2, 0.2, 1])
        num_joints = p.getNumJoints(self.robot)
        for i in range(num_joints):
            if i == 5:
                p.changeVisualShape(self.robot, i, rgbaColor=[0.2, 0.2, 0.2, 1])
            else:
                p.changeVisualShape(self.robot, i, rgbaColor=[0.8, 0.4, 0.0, 1.0])
        p.resetBasePositionAndOrientation(self.robot, self.robotStartPos, self.robotStartOrientation)

        # 生成障碍物（大圆柱体）
        cylCol = p.createCollisionShape(p.GEOM_CYLINDER, radius=self.cylinder_radius, height=self.cylinder_height)
        cylVis = p.createVisualShape(p.GEOM_CYLINDER, radius=self.cylinder_radius, length=self.cylinder_height,
                                     rgbaColor=[0.3, 0.3, 0.3, 1])
        self.cylinderId = p.createMultiBody(baseMass=0,
                                            baseCollisionShapeIndex=cylCol,
                                            baseVisualShapeIndex=cylVis,
                                            basePosition=self.cylinderStartPos)

        # 随机生成目标（红盒子）位置
        # box_y = np.random.uniform(-0.2, 0.2)
        # box_z = np.random.uniform(1.2, 1.3)
        # self.boxStartPos = [1.6, box_y, box_z]
        box_y = random.uniform(0.4, 0.6)
        box_z = random.uniform(1.0, 1.2)
        self.boxStartPos = np.array([1.8, box_y, box_z])
        boxCol = p.createCollisionShape(p.GEOM_BOX, halfExtents=self.box_half_extents)
        boxVis = p.createVisualShape(p.GEOM_BOX, halfExtents=self.box_half_extents, rgbaColor=[1, 0, 0, 1])
        self.boxId = p.createMultiBody(baseMass=0,
                                       baseCollisionShapeIndex=boxCol,
                                       baseVisualShapeIndex=boxVis,
                                       basePosition=self.boxStartPos)

        # 获取机器人所有关节，选择 revolute 或 prismatic 的前6个作为可控关节
        controlled = []
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot, i)
            joint_type = joint_info[2]
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                controlled.append(i)
        self.controlled_joints = controlled[:6]
        # print("controlled_joints:",self.controlled_joints)

        # 设置稳定的初始关节角度（根据机器人实际情况调整）
        initial_joint_positions = [0.0, -0.5, 0.3, 0.0, 0.0, 0.0]
        for idx, joint in enumerate(controlled[:6]):
            p.resetJointState(self.robot, joint, initial_joint_positions[idx])

        # 记录初始末端与目标的距离，用于奖励判断
        ee_state = p.getLinkState(self.robot, self.endEffectorLinkIndex)
        ee_pos = np.array(ee_state[4])
        self.initial_distance = np.linalg.norm(ee_pos - np.array(self.boxStartPos))
        self.currentStep = 0
        return self._get_obs(), {}

    def _get_obs(self):
        # 1. 获取6维关节角度
        joint_angles = []
        for j in self.controlled_joints:
            state = p.getJointState(self.robot, j)
            joint_angles.append(state[0])
        joint_angles = np.array(joint_angles)  # 6维
        # print("joint_angles:",joint_angles)

        # 2. 获取末端执行器的位置（3维）
        # ee_state = p.getLinkState(self.robot, self.endEffectorLinkIndex, computeLinkVelocity=1)
        ee_state = p.getLinkState(self.robot, self.endEffectorLinkIndex)
        end_effector_pos = np.array(ee_state[4])

        # 3. 目标与末端的相对位置（3维）
        target_pos = np.array(self.boxStartPos)
        relative_target = target_pos - end_effector_pos

        # 4. 获取障碍物中心位置（3维）和障碍物线速度（3维）
        obstacle_pos, _ = p.getBasePositionAndOrientation(self.cylinderId)
        obstacle_pos = np.array(obstacle_pos)
        obstacle_vel, _ = p.getBaseVelocity(self.cylinderId)
        obstacle_vel = np.array(obstacle_vel)

        # 拼接状态：6 + 3 + 3 + 3 + 3 = 18维
        state = np.concatenate([joint_angles, end_effector_pos, relative_target, obstacle_pos, obstacle_vel])
        # state[np.abs(state) < 1e-4] = 0.0
        return np.round(state, decimals=4)

    def step(self, action):
        # 对每个可控关节更新目标角度：目标角度 = 当前角度 + 动作增量
        for i, joint in enumerate(self.controlled_joints):
            current_angle = p.getJointState(self.robot, joint)[0]
            target_angle = current_angle + action[i]
            p.setJointMotorControl2(bodyUniqueId=self.robot,
                                    jointIndex=joint,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=target_angle,
                                    force=10000,
                                    positionGain=0.02,
                                    velocityGain=0.5)
        p.stepSimulation()
        time.sleep(self.timeStep)
        self.currentStep += 1

        obs = self._get_obs()
        ee_state = p.getLinkState(self.robot, self.endEffectorLinkIndex)
        ee_pos = np.array(ee_state[4])
        d_box = np.linalg.norm(ee_pos - np.array(self.boxStartPos))
        # print(f"Step {self.currentStep}: End-effector to target distance = {d_box:.4f}")

        # 奖励基于末端与目标的绝对距离
        reward = -0.01 - 0.5 * d_box
        reward = np.round(reward, decimals=4)
        reward += 0.1 * (self.initial_distance - d_box)  # 鼓励距离的减小

        terminated = False
        info = {}
        if d_box < 0.02:
            reward += 50.0
            terminated = True
            info["is_success"] = True
        else:
            info["is_success"] = False

        if d_box > 2.0 * self.initial_distance:
            reward -= 10.0
            terminated = True

        collision_flag = check_collision(self.robot, [self.cylinderId])
        if collision_flag:
            reward -= 10.0
            terminated = True
        else:
            min_dist = compute_min_distance(self.robot, [self.cylinderId])
            if min_dist < 0.1:
                reward -= 10 * (0.2 - min_dist)

        truncated = False
        if self.currentStep >= self.maxSteps:
            truncated = True
            reward -= 1.0

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        # 在 GUI 模式下，PyBullet 自动渲染
        pass

    def close(self):
        p.disconnect()

    # 自己添加的方法！！！
    def get_end_effector_pos(self):
        ee_state = p.getLinkState(self.robot, self.endEffectorLinkIndex)
        return np.array(ee_state[4])  # 返回末端位置 (x, y, z)


if __name__ == "__main__":
    env = KukaK120Env(render=True)
    obs, info = env.reset()
    print("Initial observation:", obs)
    for j in range(200):
        action = np.random.uniform(-0.1, 0.1, size=(6,))
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step: {j+1}, Reward: {reward}")
        print("Observation:", obs)
        if terminated or truncated:
            break
    env.close()
