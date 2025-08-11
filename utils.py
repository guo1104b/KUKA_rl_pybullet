"""
utils.py

提供两个函数：
  - check_collision(robot, obj_ids): 检测机器人与给定物体列表中是否有碰撞；
  - compute_min_distance(robot, obj_ids, distance_threshold=10.0): 计算机器人与给定物体之间的最小距离。
"""

import pybullet as p

def check_collision(robot, obj_ids):
    """
    检查机器人与列表中任一物体是否发生碰撞。
    若任一对之间存在接触，返回 True；否则返回 False。
    """
    for obj_id in obj_ids:
        pts = p.getContactPoints(bodyA=robot, bodyB=obj_id)
        if len(pts) > 0:
            return True
    return False

def compute_min_distance(robot, obj_ids, distance_threshold=1.0):
    """
    计算机器人与列表中物体之间的最小距离。
    调用 p.getClosestPoints() 在 distance_threshold 范围内搜索，
    返回找到的最小距离；如果未找到，则返回 distance_threshold。
    """
    min_dist = distance_threshold
    for obj_id in obj_ids:
        pts = p.getClosestPoints(bodyA=robot, bodyB=obj_id, distance=distance_threshold)
        for pt in pts:
            d = pt[8]  # 第9个元素为距离
            if d < min_dist:
                min_dist = d
    return min_dist
