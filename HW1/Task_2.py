import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '3-map/map.npy')


### START CODE HERE ###

from heapdict import heapdict
from math import sqrt
from scipy.ndimage import distance_transform_edt


def octile_distance(a, b):
    # 启发式函数h，计算Octile距离
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return (sqrt(2) - 1) * min(dx, dy) + max(dx, dy)

def reconstruct_path(parent, current):
    # 重建路径
    path = [current]
    while current in parent:
        current = parent[current]
        path.append(current)
    path.reverse()
    return path

###  END CODE HERE  ###


def Improved_A_star(world_map, start_pos, goal_pos):
    """
    Given map of the world, start position of the robot and the position of the goal, 
    plan a path from start position to the goal using A* algorithm.

    Arguments:
    world_map -- A 120*120 array indicating current map, where 0 indicating traversable and 1 indicating obstacles.
    start_pos -- A 2D vector indicating the current position of the robot.
    goal_pos -- A 2D vector indicating the position of the goal.

    Return:
    path -- A N*2 array representing the planned path by A* algorithm.
    """

    ### START CODE HERE ###

    # Timing and statistics
    t0 = time.perf_counter() # 统计运行时间
    expansions = 0 # 统计被加入closed_set的节点数量

    max_x, max_y = world_map.shape
    start = tuple(start_pos)
    goal = tuple(goal_pos)

    # 起点、终点检查是否越界/障碍物
    def _in_bounds(n):
        return 0 <= n[0] < max_x and 0 <= n[1] < max_y

    if not _in_bounds(start) or not _in_bounds(goal):
        raise ValueError("Start or goal position is out of map bounds.")
    if world_map[start[0]][start[1]] == 1 or world_map[goal[0]][goal[1]] == 1:
        raise ValueError("Start or goal position is occupied by an obstacle.")

    # 计算距障距离图
    free_mask = (world_map == 0) # 1 for free cells, 0 for obstacles
    # 对自由格计算其到最近障碍的欧氏距离，越大越安全
    dist_map = distance_transform_edt(free_mask)

    # 权重
    w_obs = 3.0 # 距障惩罚权重，惩罚贴边
    w_turn = 0.5 # 转弯惩罚权重，惩罚拐弯
    safety_buffer = 3.0 # 安全缓冲

    def obstacle_penalty(n):
        d = dist_map[n[0], n[1]] # 当前格到最近障碍的距离
        if d <= safety_buffer:
            # 把这个格当作不可通行
            return np.inf  
        # 否则，距离越近，距障惩罚越大
        return w_obs / (d + 1.0)

    # 初始化open_set和closed_set
    open_set = heapdict()
    open_set[start] = (octile_distance(start, goal), 0) # 起始点的f值即为h值

    closed_set = set()

    parent = {} # 记录最优前驱，用于最终回溯路径
    g_score = {start: 0.0} # 记录从起点到节点的当前最短代价
    last_dir = {start: None}  # 记录进入节点的方向，用于转弯惩罚
    counter = 1 # 用于在f值相同时，按插入顺序处理节点

    # 直走代价 1，对角代价 √2
    neighbors_delta = [
        (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, sqrt(2)), (1, -1, sqrt(2)), (-1, 1, sqrt(2)), (-1, -1, sqrt(2)),
    ]

    while open_set:
        # 从 open_set 取出 f 最小的节点 current
        current, _ = open_set.popitem()

        # 如果 current 已在 closed_set 中，跳过
        if current in closed_set:
            continue
        # 否则, 将 current 加入 closed_set
        closed_set.add(current)
        expansions += 1

        # 如果 current == goal，回溯起点到终点的输出路径
        if current == goal:
            path = reconstruct_path(parent, current)
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            steps = max(len(path) - 1, 0)
            print(f"Improved A* computational time: {elapsed_ms:.2f} ms | expanded nodes: {expansions} | path steps: {steps}")
            return np.array(path, dtype=int)

        # 遍历 current 的邻居节点
        for dx, dy, step_cost in neighbors_delta:
            # 检查是否越界/障碍物
            neighbor = (current[0] + dx, current[1] + dy)
            if not _in_bounds(neighbor):
                continue
            if world_map[neighbor[0]][neighbor[1]] == 1:
                continue

            # 计算距障惩罚
            obs_cost = obstacle_penalty(neighbor)
            if np.isinf(obs_cost):
                # 到最近障碍的距离过近，当作不可通行
                continue  

            # 计算转弯代价
            prev_dir = last_dir.get(current, None)
            turn_cost = 0.0 if prev_dir is None or prev_dir == (dx, dy) else w_turn

            # 计算从起点经过 current 到该 neighbor 的临时代价tentative_g
            tentative_g = g_score[current] + step_cost + turn_cost + obs_cost

            # 如果该路径更优，更新 neighbor 的 g 值、f 值和前驱节点
            if tentative_g < g_score.get(neighbor, np.inf):
                parent[neighbor] = current # 更新前驱节点
                g_score[neighbor] = tentative_g # 更新 g 值
                last_dir[neighbor] = (dx, dy) # 更新进入方向
                f_score = tentative_g + octile_distance(neighbor, goal) # 计算 f 值
                open_set[neighbor] = (f_score, counter) # 加入 open_set
                counter += 1

    path = np.empty((0, 2), dtype=int)
    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000.0
    print(f"Improved A* computational time: {elapsed_ms:.2f} ms | expanded nodes: {expansions} | path steps: 0 (no path)")

    ###  END CODE HERE  ###
    return path





if __name__ == '__main__':

    # Get the map of the world representing in a 120*120 array, where 0 indicating traversable and 1 indicating obstacles.
    map = np.load(MAP_PATH)

    # Define goal position of the exploration
    goal_pos = [100, 100]

    # Define start position of the robot.
    start_pos = [10, 10]

    # Plan a path based on map from start position of the robot to the goal.
    path = Improved_A_star(map, start_pos, goal_pos)

    # Visualize the map and path.
    obstacles_x, obstacles_y = [], []
    for i in range(120):
        for j in range(120):
            if map[i][j] == 1:
                obstacles_x.append(i)
                obstacles_y.append(j)

    path_x, path_y = [], []
    for path_node in path:
        path_x.append(path_node[0])
        path_y.append(path_node[1])

    plt.plot(path_x, path_y, "-r")
    plt.plot(start_pos[0], start_pos[1], "xr")
    plt.plot(goal_pos[0], goal_pos[1], "xb")
    plt.plot(obstacles_x, obstacles_y, ".k")
    plt.grid(True)
    plt.axis("equal")
    # plt.show()
    plt.savefig('HW1/Figures/Task_2.png', bbox_inches='tight')