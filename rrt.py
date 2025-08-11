#!/usr/bin/env python
"""
完整代码：单环境下的 RRT 规划与运动（带关节角度范围限制）
"""

import os
import time
import random
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


####################################
# 辅助函数
####################################
def clip_joint_angles(joint_angles, joint_limits):
    """
    对每个关节角度进行裁剪，使其落在对应的 (lower, upper) 范围内。

    参数:
        joint_angles: 关节角度列表
        joint_limits: 列表，每个元素为 (lower, upper)
    返回:
        裁剪后的关节角度列表
    """
    clipped = []
    for angle, (low, high) in zip(joint_angles, joint_limits):
        clipped.append(np.clip(angle, low, high))
    return clipped


def set_robot_configuration(client, robot, joint_angles):
    """
    将机器人 robot 中所有非固定关节设置为 joint_angles 给定的值。
    注意：这里假定 p.calculateInverseKinematics 返回的关节角度顺序与非固定关节顺序一致。
    """
    num_joints = p.getNumJoints(robot, physicsClientId=client)
    idx = 0
    for i in range(num_joints):
        info = p.getJointInfo(robot, i, physicsClientId=client)
        if info[2] != p.JOINT_FIXED:
            p.resetJointState(robot, i, joint_angles[idx], physicsClientId=client)
            idx += 1


def check_collision(client, robot, obstacles, threshold=0.0):
    """
    对 robot 与 obstacles（障碍物 id 列表）进行碰撞检测，
    若任一障碍物与 robot 最近距离小于 threshold，则返回 True。
    """
    for obs in obstacles:
        pts = p.getClosestPoints(robot, obs, distance=threshold, physicsClientId=client)
        if len(pts) > 0:
            return True
    return False


def get_end_effector_pos(client, robot, end_effector_link_index):
    """
    返回机器人 robot 指定 link 的世界坐标位置。
    """
    state = p.getLinkState(robot, end_effector_link_index, physicsClientId=client)
    return np.array(state[4])


def get_controlled_joints(robot, client):
    """
    返回机器人 robot 中所有非固定关节的索引（作为受控关节）。
    """
    controlled = []
    num_joints = p.getNumJoints(robot, physicsClientId=client)
    for i in range(num_joints):
        info = p.getJointInfo(robot, i, physicsClientId=client)
        if info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            controlled.append(i)
    return controlled[:6]


####################################
# RRT 规划相关类
####################################
class Node:
    def __init__(self, position, parent=None, config=None):
        self.position = np.array(position)  # 末端执行器位置（3D）
        self.parent = parent  # 父节点
        self.config = config  # 对应机器人完整关节配置


class RRTPlanner:
    def __init__(self, client, robot, obstacles, end_effector_link_index,
                 x_limits, y_limits, z_limits, step_size=0.05, max_iter=1000,
                 joint_limits=None):
        """
        参数:
            client: PyBullet 客户端 id（本代码中只用一个环境）
            robot: 机器人 id
            obstacles: 障碍物 id 列表
            end_effector_link_index: 末端执行器的 link 索引
            x_limits, y_limits, z_limits: 末端工作空间采样范围
            step_size: 每次扩展的步长（单位：m）
            max_iter: 最大迭代次数
            joint_limits: 对非固定关节的角度限制（列表形式，每个元素为 (lower, upper)）
        """
        self.client = client
        self.robot = robot
        self.obstacles = obstacles
        self.end_effector_link_index = end_effector_link_index
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.z_limits = z_limits
        self.step_size = step_size
        self.max_iter = max_iter
        self.joint_limits = joint_limits

    def sample_random_point(self):
        x = random.uniform(self.x_limits[0], self.x_limits[1])
        y = random.uniform(self.y_limits[0], self.y_limits[1])
        z = random.uniform(self.z_limits[0], self.z_limits[1])
        return np.array([x, y, z])

    def nearest_node(self, tree, point):
        dists = [np.linalg.norm(node.position - point) for node in tree]
        return tree[np.argmin(dists)]

    def steer(self, from_node, to_point):
        direction = to_point - from_node.position
        dist = np.linalg.norm(direction)
        if dist < self.step_size:
            return to_point
        else:
            return from_node.position + (direction / dist) * self.step_size

    def is_collision_free(self, from_pos, to_pos, num_steps=20):
        """
        沿 from_pos 到 to_pos 的直线插值，
        对每个插值点利用 IK 计算机器人配置，并检测是否与障碍物碰撞。
        """
        targetOrn = [0, 0, 0, 1]
        for i in range(num_steps + 1):
            t = i / num_steps
            interp_point = from_pos + t * (to_pos - from_pos)
            ik_solution = p.calculateInverseKinematics(self.robot, self.end_effector_link_index,
                                                       interp_point.tolist(),
                                                       # targetOrientation=targetOrn,
                                                       maxNumIterations=100,  # 增加最大迭代次数
                                                       residualThreshold=0.001,  # 降低残差阈值
                                                       physicsClientId=self.client)
            if self.joint_limits is not None:
                ik_solution = clip_joint_angles(ik_solution, self.joint_limits)
            set_robot_configuration(self.client, self.robot, ik_solution)
            # 验证IK解后的实际位置
            actual_pos = get_end_effector_pos(self.client, self.robot, self.end_effector_link_index)
            if np.linalg.norm(actual_pos - interp_point) > 0.02:  # 允许2厘米误差
                print(f"IK误差过大: {np.linalg.norm(actual_pos - interp_point) > 0.02} at {interp_point}")
                return False
            if check_collision(self.client, self.robot, self.obstacles, threshold=0.0):
                print("碰撞了！")
                return False
        return True

    def plan(self, start_pos, goal_pos, goal_threshold=0.1):
        """
        从 start_pos 到 goal_pos 进行 RRT 路径规划，返回由 Node 构成的路径列表。
        如果起点与目标距离小于 goal_threshold，则直接返回平凡路径。
        """
        targetOrn = [0, 0, 0, 1]
        if np.linalg.norm(goal_pos - start_pos) < goal_threshold:
            config = p.calculateInverseKinematics(self.robot, self.end_effector_link_index,
                                                  goal_pos.tolist(),
                                                  # targetOrientation=targetOrn,
                                                  physicsClientId=self.client)
            if self.joint_limits is not None:
                config = clip_joint_angles(config, self.joint_limits)
            start_node = Node(start_pos, None, config)
            goal_node = Node(goal_pos, start_node, config)
            return [start_node, goal_node]

        start_config = p.calculateInverseKinematics(self.robot, self.end_effector_link_index,
                                                    start_pos.tolist(),
                                                    # targetOrientation=targetOrn,
                                                    physicsClientId=self.client)
        if self.joint_limits is not None:
            start_config = clip_joint_angles(start_config, self.joint_limits)
        start_node = Node(start_pos, None, start_config)
        tree = [start_node]

        for i in range(self.max_iter):
            if random.random() < 0.2:
                sample = goal_pos
            else:
                sample = self.sample_random_point()
            nearest = self.nearest_node(tree, sample)
            new_pos = self.steer(nearest, sample)

            new_config = p.calculateInverseKinematics(self.robot, self.end_effector_link_index,
                                                      new_pos.tolist(),
                                                      # targetOrientation=targetOrn,
                                                      physicsClientId=self.client)
            set_robot_configuration(self.client, self.robot, new_config)
            actual_pos = get_end_effector_pos(self.client, self.robot, self.end_effector_link_index)
            if np.linalg.norm(actual_pos - new_pos) > 0.02:
                continue  # 跳过不符合精度要求的节点

            if self.is_collision_free(nearest.position, new_pos):

                if self.joint_limits is not None:
                    new_config = clip_joint_angles(new_config, self.joint_limits)
                new_node = Node(new_pos, parent=nearest, config=new_config)
                tree.append(new_node)
                if np.linalg.norm(new_node.position - goal_pos) < goal_threshold:
                    if self.is_collision_free(new_node.position, goal_pos):
                        goal_config = p.calculateInverseKinematics(self.robot, self.end_effector_link_index,
                                                                   goal_pos.tolist(),
                                                                   # targetOrientation=targetOrn,
                                                                   physicsClientId=self.client)
                        if self.joint_limits is not None:
                            goal_config = clip_joint_angles(goal_config, self.joint_limits)
                        goal_node = Node(goal_pos, parent=new_node, config=goal_config)
                        tree.append(goal_node)
                        return self.extract_path(goal_node)
        return None

    def extract_path(self, node):
        path = []
        while node is not None:
            path.append(node)
            node = node.parent
        path.reverse()
        return path


####################################
# 主函数：环境搭建、规划与执行
####################################
def main():
    # 1. 连接单一 GUI 环境
    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.resetSimulation(physicsClientId=client)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(1.0 / 240.0, physicsClientId=client)
    p.loadURDF("plane.urdf", physicsClientId=client)

    # 2. 加载机器人（固定底座），请确保 URDF 路径正确
    current_dir = os.path.dirname(os.path.realpath(__file__))
    robot_urdf = os.path.join(current_dir, "urdf", "kr120r2500pro.urdf")
    robot = p.loadURDF(robot_urdf,
                       [0, 0, 0],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       useFixedBase=True,
                       physicsClientId=client)

    # 3. 加载障碍物 —— 圆柱体，并保存其 id
    cylinder_radius = 0.2
    cylinder_height = 1.2
    cylinder_start_pos = [1.2, 0, 0.6]
    cyl_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=cylinder_radius, height=cylinder_height,
                                     physicsClientId=client)
    cyl_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=cylinder_radius, length=cylinder_height,
                                  rgbaColor=[0.3, 0.3, 0.3, 1],
                                  physicsClientId=client)
    cylinder_id = p.createMultiBody(baseMass=0,
                                    baseCollisionShapeIndex=cyl_col,
                                    baseVisualShapeIndex=cyl_vis,
                                    basePosition=cylinder_start_pos,
                                    physicsClientId=client)

    # 4. 加载目标 —— 红色盒子，目标位置随机（例如：x 固定 1.6，y∈[-0.2,0.2]，z∈[1.2,1.3]）
    box_half_extents = [0.01, 0.01, 0.01]
    # target_y = random.uniform(0.4, 0.6)
    # # target_z = random.uniform(0.6, 0.95)
    # target_z = random.uniform(1.0, 1.2)
    target_y = 0.5
    target_z = 1.0
    target_pos = np.array([1.8, target_y, target_z])
    box_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=box_half_extents, physicsClientId=client)
    box_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=box_half_extents,
                                  rgbaColor=[1, 0, 0, 1],
                                  physicsClientId=client)
    p.createMultiBody(baseMass=0,
                      baseCollisionShapeIndex=box_col,
                      baseVisualShapeIndex=box_vis,
                      basePosition=target_pos.tolist(),
                      physicsClientId=client)

    # 5. 初始化机器人关节角度（使用你提供的初始角度）
    initial_joint_positions = [0.0, -0.5, 0.3, 0.0, 0.0, 0.0]
    controlled_joints = get_controlled_joints(robot, client)
    for idx, joint in enumerate(controlled_joints[:len(initial_joint_positions)]):
        p.resetJointState(robot, joint, initial_joint_positions[idx], physicsClientId=client)

    # 6. 设定末端执行器 link 索引（根据 URDF 调整，这里假定为 6）
    end_effector_link_index = 6

    start_ee_pos = get_end_effector_pos(client, robot, end_effector_link_index)
    print("规划起点末端位置:", start_ee_pos)
    print("目标位置:", target_pos)

    # 7. 定义每个非固定关节的角度限制（单位：弧度），这里对前 6 个关节进行限制
    joint_limits = [
        (-3.14, 3.14),  # 第一关节
        (-2.0, 2.0),  # 第二关节
        (-1.5, 1.5),  # 第三关节
        (-3.14, 3.14),  # 第四关节
        (-3.14, 3.14),  # 第五关节
        (-3.14, 3.14)  # 第六关节
    ]
    # 如果非固定关节数超过 6，则为其余关节设置默认限制
    num_non_fixed = len(get_controlled_joints(robot, client))
    if num_non_fixed > len(joint_limits):
        for i in range(num_non_fixed - len(joint_limits)):
            joint_limits.append((-3.14, 3.14))

    # 8. 创建 RRT 规划器（工作空间采样范围可根据实际情况调整）
    x_limits = [0.5, 2.0]
    y_limits = [-1.0, 1.0]
    z_limits = [0.3, 2.0]
    start_planning_time = time.time()
    planner = RRTPlanner(client, robot, [cylinder_id], end_effector_link_index,
                         x_limits, y_limits, z_limits, step_size=0.05, max_iter=5000,
                         joint_limits=joint_limits)

    print("开始 RRT 路径规划……")
    path_nodes = planner.plan(start_ee_pos, target_pos, goal_threshold=0.05)
    # 记录规划结束时间
    end_planning_time = time.time()
    planning_time = end_planning_time - start_planning_time

    if path_nodes is None:
        print("未找到路径！请调整采样范围、目标位置或阈值。")
        return
    print("规划成功，路径节点数:", len(path_nodes))

    # 9. 提取规划路径的末端位置与对应的关节配置
    path_positions = [node.position for node in path_nodes]
    path_configs = [node.config for node in path_nodes]

    # 在 OpenGL 界面中以绿色显示规划路径
    for i in range(len(path_positions) - 1):
        p.addUserDebugLine(path_positions[i].tolist(),
                           path_positions[i + 1].tolist(),
                           [0, 1, 0],
                           lineWidth=2,
                           lifeTime=0,
                           physicsClientId=client)

    # 由于规划过程中调用了 IK，可能改变了机器人状态；此处将机器人重置为初始状态
    for idx, joint in enumerate(controlled_joints[:len(initial_joint_positions)]):
        p.resetJointState(robot, joint, initial_joint_positions[idx], physicsClientId=client)

    # 10. 执行规划路径（在关节空间内插值控制）
    trajectory = []  # 记录末端执行器实际位置
    suan_t = []
    num_interp_steps = 10  # 每段插值步数
    prev_ee_pos = get_end_effector_pos(client, robot, end_effector_link_index)
    num_joints = p.getNumJoints(robot, physicsClientId=client)

    # 记录机械臂执行开始时间
    start_execution_time = time.time()
    for i in range(len(path_configs) - 1):
        q_start = np.array(path_configs[i])
        q_end = np.array(path_configs[i + 1])
        for t in np.linspace(0, 1, num_interp_steps):
            q_interp = (1 - t) * q_start + t * q_end
            joint_idx = 0
            for j in range(num_joints):
                info = p.getJointInfo(robot, j, physicsClientId=client)
                if info[2] != p.JOINT_FIXED:
                    p.setJointMotorControl2(bodyUniqueId=robot,
                                            jointIndex=j,
                                            controlMode=p.POSITION_CONTROL,
                                            targetPosition=q_interp[joint_idx],
                                            force=10000,
                                            positionGain=0.02,
                                            velocityGain=0.5,
                                            physicsClientId=client)
                    joint_idx += 1
            for _ in range(10):
                p.stepSimulation(physicsClientId=client)
                tt = time.time() - start_execution_time
                current_ee_pos = get_end_effector_pos(client, robot, end_effector_link_index)
                trajectory.append(current_ee_pos)
                suan_t.append(tt)
                p.addUserDebugLine(prev_ee_pos.tolist(), current_ee_pos.tolist(),
                                   [0, 0, 1],
                                   lineWidth=2,
                                   lifeTime=1,
                                   physicsClientId=client)
                prev_ee_pos = current_ee_pos
                # time.sleep(1.0 / 240.0)

    # 在目标处停留一段时间
    for _ in range(240):
        p.stepSimulation(physicsClientId=client)
        tt = time.time() - start_execution_time
        suan_t.append(tt)
        current_ee_pos = get_end_effector_pos(client, robot, end_effector_link_index)
        trajectory.append(current_ee_pos)
        p.addUserDebugLine(prev_ee_pos.tolist(), current_ee_pos.tolist(),
                           [0, 0, 1],
                           lineWidth=2,
                           lifeTime=1,
                           physicsClientId=client)
        prev_ee_pos = current_ee_pos
        time.sleep(1.0 / 240.0)

    # 记录机械臂执行结束时间
    end_execution_time = time.time()
    execution_time = end_execution_time - start_execution_time

    # 11. 断开 PyBullet
    p.disconnect(client)

    # 12. 计算路径粗糙度和路径长度
    trajectory = np.array(trajectory)
    # print("trajectory:",trajectory)

    # 计算路径粗糙度（二阶差分平方的平均值）
    # 计算一阶速度
    epsilon = 1e-5
    dt = np.diff(suan_t)
    print("dt",dt)
    # 将 dt 中的零值替换为 epsilon
    dt = np.where(dt == 0, epsilon, dt)
    dt_mid = (dt[:-1] + dt[1:]) / 2  # (T-2,)

    # **确保 dt_mid 形状为 (T-2, 1)，匹配 trajectory**
    dt_mid = dt_mid[:, None]  # 变成列向量 (T-2, 1)
    acceleration = (trajectory[2:] - 2 * trajectory[1:-1] + trajectory[:-2]) / dt_mid**2

    # 计算粗糙度
    roughness = np.mean(np.sum(acceleration ** 2, axis=1))

    # 计算路径长度（相邻点的欧氏距离总和）
    distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
    path_length = np.sum(distances)

    # 输出结果
    print(f"路径粗糙度 (Roughness): {roughness:.4f}")
    print(f"路径长度 (Path Length): {path_length:.4f} m")
    print(f"规划时间 (Planning Time): {planning_time:.4f} s")
    print(f"机械臂执行时间 (Execution Time): {execution_time:.4f} s")

    # 13. 利用 matplotlib 绘制末端执行器三维轨迹图
    # trajectory = np.array(trajectory)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color='blue', label="Траектория робота")
    ax.scatter(start_ee_pos[0], start_ee_pos[1], start_ee_pos[2], c='black', marker='o', label="Начальная точка")
    ax.scatter(target_pos[0], target_pos[1], target_pos[2], c='green', marker='o', label="Целевая точка")
    ax.set_xlabel("X(m)")
    ax.set_ylabel("Y(m)")
    ax.set_zlabel("Z(m)")
    # ax.legend()
    ax.legend(loc='upper right')
    ax.set_title("Совместные траектории человека и робота")
    plt.show()


if __name__ == "__main__":
    main()
