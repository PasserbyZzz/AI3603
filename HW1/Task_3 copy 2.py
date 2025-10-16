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
from scipy.interpolate import make_interp_spline

def octile_distance(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return (sqrt(2) - 1) * min(dx, dy) + max(dx, dy)

def reconstruct_path(parent, current):
    path = [current]
    while current in parent:
        current = parent[current]
        path.append(current)
    path.reverse()
    return path

def smooth_path_polyfit(path, degree=3, num_points=200):
    """
    对路径进行多项式插值平滑（全局）
    path: N*2 array
    degree: 多项式阶数
    num_points: 平滑后采样点数
    return: 平滑后的路径，num_points*2 array
    """
    path = np.array(path)
    if len(path) < degree + 1:
        return path  # 点太少无法拟合
    t = np.arange(len(path))
    t_new = np.linspace(0, len(path) - 1, num_points)
    poly_x = np.polyfit(t, path[:, 0], degree)
    poly_y = np.polyfit(t, path[:, 1], degree)
    x_new = np.polyval(poly_x, t_new)
    y_new = np.polyval(poly_y, t_new)
    return np.stack([x_new, y_new], axis=1)

# 分段多项式插值平滑
def segmental_smooth_path_polyfit(path, degree=3, segment_len=15, points_per_segment=20):
    """
    将路径分段，每段做多项式插值平滑，最后拼接
    path: N*2 array
    degree: 多项式阶数
    segment_len: 每段原始点数
    points_per_segment: 每段平滑采样点数
    return: 平滑后的路径
    """
    path = np.array(path)
    N = len(path)
    if N < degree + 1:
        return path
    segments = []
    for i in range(0, N-1, segment_len):
        seg = path[i : min(i+segment_len+1, N)]
        if len(seg) < degree + 1:
            # 剩余点太少，直接拼接
            segments.append(seg)
            break
        t = np.arange(len(seg))
        t_new = np.linspace(0, len(seg)-1, points_per_segment)
        poly_x = np.polyfit(t, seg[:,0], degree)
        poly_y = np.polyfit(t, seg[:,1], degree)
        x_new = np.polyval(poly_x, t_new)
        y_new = np.polyval(poly_y, t_new)
        smooth_seg = np.stack([x_new, y_new], axis=1)
        if i > 0:
            # 避免段首点重复
            smooth_seg = smooth_seg[1:]
        segments.append(smooth_seg)
    return np.vstack(segments)

###  END CODE HERE  ###


def Self_driving_path_planner(world_map, start_pos, goal_pos):
    """
    Given map of the world, start position of the robot and the position of the goal, 
    plan a path from start position to the goal using A* algorithm, and smooth the path using polynomial interpolation.

    Arguments:
    world_map -- A 120*120 array indicating current map, where 0 indicating traversable and 1 indicating obstacles.
    start_pos -- A 2D vector indicating the current position of the robot.
    goal_pos -- A 2D vector indicating the position of the goal.

    Return:
    path -- A N*2 array representing the planned and smoothed path.
    """

    # --- A*部分 ---
    t0 = time.perf_counter()
    expansions = 0
    max_x, max_y = world_map.shape
    start = tuple(start_pos)
    goal = tuple(goal_pos)

    def _in_bounds(n):
        return 0 <= n[0] < max_x and 0 <= n[1] < max_y

    if not _in_bounds(start) or not _in_bounds(goal):
        raise ValueError("Start or goal position is out of map bounds.")
    if world_map[start[0]][start[1]] == 1 or world_map[goal[0]][goal[1]] == 1:
        raise ValueError("Start or goal position is occupied by an obstacle.")

    free_mask = (world_map == 0)
    dist_map = distance_transform_edt(free_mask)

    w_obs = 3.0
    w_turn = 0.5
    safety_buffer = 3.0

    def obstacle_penalty(n):
        d = dist_map[n[0], n[1]]
        if d <= safety_buffer:
            return np.inf
        return w_obs / (d + 1.0)

    open_set = heapdict()
    open_set[start] = (octile_distance(start, goal), 0)
    closed_set = set()
    parent = {}
    g_score = {start: 0.0}
    last_dir = {start: None}
    counter = 1
    neighbors_delta = [
        (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, sqrt(2)), (1, -1, sqrt(2)), (-1, 1, sqrt(2)), (-1, -1, sqrt(2)),
    ]

    while open_set:
        current, _ = open_set.popitem()

        if current in closed_set:
            continue
        closed_set.add(current)
        expansions += 1

        if current == goal:
            path = reconstruct_path(parent, current)
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            steps = max(len(path) - 1, 0)
            print(f"A* time: {elapsed_ms:.2f} ms | expanded: {expansions} | steps: {steps}")
            path = np.array(path, dtype=int)
            # 分段多项式插值平滑
            path = segmental_smooth_path_polyfit(path, degree=3, segment_len=15, points_per_segment=20)
            return path
        
        for dx, dy, step_cost in neighbors_delta:
            neighbor = (current[0] + dx, current[1] + dy)
            if not _in_bounds(neighbor):
                continue
            if world_map[neighbor[0]][neighbor[1]] == 1:
                continue
            obs_cost = obstacle_penalty(neighbor)
            if np.isinf(obs_cost):
                continue
            prev_dir = last_dir.get(current, None)
            turn_cost = 0.0 if prev_dir is None or prev_dir == (dx, dy) else w_turn
            tentative_g = g_score[current] + step_cost + turn_cost + obs_cost
            if tentative_g < g_score.get(neighbor, np.inf):
                parent[neighbor] = current
                g_score[neighbor] = tentative_g
                last_dir[neighbor] = (dx, dy)
                f_score = tentative_g + octile_distance(neighbor, goal)
                open_set[neighbor] = (f_score, counter)
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
    path = Self_driving_path_planner(map, start_pos, goal_pos)

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
    plt.show()
    # plt.savefig('HW1/Figures/Task_3.png', bbox_inches='tight')