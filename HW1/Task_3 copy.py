import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '3-map/map.npy')


### START CODE HERE ###

from easydubins.dubin_path import dubins_path, get_curve
from heapdict import heapdict
import math

def wrap_angle(theta):
    # wrap to [0, 2π)
    twopi = 2.0 * math.pi
    return theta % twopi

def angle_to_deg(theta):
    return math.degrees(theta) % 360.0

def deg_to_rad(deg):
    return math.radians(deg)

def heuristic_dubins(curr, goal, radius):
    # curr, goal: (x,y,theta_rad)
    try:
        mode, lengths, _ = dubins_path(curr, goal, radius)
        if mode is None or lengths is None or any(l is None for l in lengths):
            # fallback to Euclidean
            return math.hypot(goal[0]-curr[0], goal[1]-curr[1])
        return abs(lengths[0]) + abs(lengths[1]) + abs(lengths[2])
    except Exception:
        # conservative fallback
        return math.hypot(goal[0]-curr[0], goal[1]-curr[1])

def in_bounds(world_map, x, y):
    h, w = world_map.shape
    return 0 <= x < h and 0 <= y < w

def is_free(world_map, x, y):
    if not in_bounds(world_map, x, y):
        return False
    return world_map[int(round(x))][int(round(y))] == 0

def collision_along_segment(world_map, pts):
    # pts: list of [x,y,heading_deg] or (x,y,theta)
    # 加入距障硬约束
    from scipy.ndimage import distance_transform_edt
    # 这里假设 dist_map 已在 hybrid_a_star 中计算并传入全局
    global dist_map, safety_buffer
    for p in pts:
        x = p[0]
        y = p[1]
        if not is_free(world_map, x, y):
            return True
        # 距障硬约束
        d_obs = dist_map[int(round(x)), int(round(y))]
        if d_obs <= safety_buffer:
            return True
    return False

def path_length(points):
    if not points or len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(len(points)-1):
        x1, y1 = points[i][0], points[i][1]
        x2, y2 = points[i+1][0], points[i+1][1]
        total += math.hypot(x2-x1, y2-y1)
    return total

def dubins_shortcut_smooth(world_map, path, radius, step, iterations=80, min_gap=8, seed=0):
    """Iteratively try to replace sub-segments with Dubins connections to shorten/smooth.
    path: list of [x,y,heading_deg]
    returns: new path list of same format
    """
    if not path or len(path) < min_gap + 2:
        return path
    rng = np.random.default_rng(seed)
    best = list(path)
    best_len = path_length(best)
    n = len(best)
    for _ in range(iterations):
        if n < min_gap + 2:
            break
        i = int(rng.integers(0, n - 1 - min_gap))
        j = int(rng.integers(i + min_gap, n - 1))
        pi = best[i]
        pj = best[j]
        # Build dubins connection between pi and pj using get_curve (degree inputs)
        try:
            cand_mid = get_curve(pi[0], pi[1], float(pi[2]), pj[0], pj[1], float(pj[2]),
                                 radius, max_line_distance=max(0.2, step/2.0))
        except Exception:
            continue
        if not cand_mid:
            continue
        # collision check on candidate mid segment
        if collision_along_segment(world_map, cand_mid):
            continue
        # Compose new candidate full path: include exact endpoints to preserve heading anchors
        new_seg = [pi] + cand_mid + [pj]
        cand = best[:i] + new_seg + best[j+1:]
        new_len = path_length(cand)
        if new_len + 1e-6 < best_len:
            best = cand
            best_len = new_len
            n = len(best)
    return best

def motion_primitives(x, y, theta, step, radius):
    # Returns list of (nx, ny, ntheta, cost)
    actions = []
    # straight
    nx = x + step * math.cos(theta)
    ny = y + step * math.sin(theta)
    ntheta = wrap_angle(theta)
    actions.append((nx, ny, ntheta, step))
    # left arc: delta heading = + step / radius
    dth = step / radius
    th2 = wrap_angle(theta + dth)
    nx_l = x + radius * (math.sin(th2) - math.sin(theta))
    ny_l = y - radius * (math.cos(th2) - math.cos(theta))
    actions.append((nx_l, ny_l, th2, step))
    # right arc: delta heading = - step / radius
    th3 = wrap_angle(theta - dth)
    nx_r = x + radius * (math.sin(th3) - math.sin(theta))
    ny_r = y - radius * (math.cos(th3) - math.cos(theta))
    actions.append((nx_r, ny_r, th3, step))
    return actions

def sample_motion(world_map, x, y, theta, nx, ny, ntheta, step_samples=5):
    # Sample along straight or arc using linear interpolation in configuration space
    pts = []
    for i in range(1, step_samples+1):
        t = i / float(step_samples)
        sx = (1-t)*x + t*nx
        sy = (1-t)*y + t*ny
        pts.append((sx, sy))
    # convert to projection-like points [x,y,heading_deg] for collision usage
    vis_pts = [[px, py, 0.0] for (px, py) in pts]
    return vis_pts

def analytic_expand_if_possible(world_map, node, goal, radius, step):
    # node = (x,y,theta), goal=(gx,gy,gtheta)
    sx, sy, sth = node
    gx, gy, gth = goal
    sdeg = angle_to_deg(sth)
    gdeg = angle_to_deg(gth)
    # step used as max_line_distance for sampling density
    try:
        curve = get_curve(sx, sy, sdeg, gx, gy, gdeg, radius, max_line_distance=step/2.0)
        if not curve:
            return None
        if collision_along_segment(world_map, curve):
            return None
        return curve  # list of [x,y,heading_deg]
    except Exception:
        return None
    
def hybrid_a_star(world_map, start_xy, goal_xy,
                  start_heading_deg=0.0, goal_heading_deg=0.0,
                  step=1.0, radius=5.0, theta_bins=72):
    # State: (x,y,theta_rad)
    start = (float(start_xy[0]), float(start_xy[1]), deg_to_rad(start_heading_deg))
    goal = (float(goal_xy[0]), float(goal_xy[1]), deg_to_rad(goal_heading_deg))


    h, w = world_map.shape
    if not is_free(world_map, start[0], start[1]) or not is_free(world_map, goal[0], goal[1]):
        raise ValueError("Start or goal is in obstacle/bounds.")

    # --- 距障惩罚参数 ---
    from scipy.ndimage import distance_transform_edt
    free_mask = (world_map == 0)
    global dist_map, safety_buffer
    dist_map = distance_transform_edt(free_mask)
    w_obs = 3.0
    w_turn = 0.5
    safety_buffer = 3.0

    # Closed set with discretization of theta
    def theta_to_bin(theta):
        b = int(round((wrap_angle(theta) / (2*math.pi)) * theta_bins)) % theta_bins
        return b

    closed = set()
    parents = {}
    g_cost = {}

    start_key = (int(round(start[0])), int(round(start[1])), theta_to_bin(start[2]))
    g_cost[start_key] = 0.0
    pq = heapdict()
    # heuristic using dubins to fixed goal heading
    f0 = heuristic_dubins(start, goal, radius)
    pq[start_key] = f0

    state_map = {start_key: start}  # key: (ix,iy,itheta) -> (x,y,theta)
    last_dir = {start_key: None}    # 记录进入方向
    expansions = 0
    best_goal_curve = None

    while pq:
        ckey, _ = pq.popitem()
        current = state_map[ckey]
        cx, cy, cth = current
        if ckey in closed:
            continue
        closed.add(ckey)
        expansions += 1

        # goal proximity check and analytic expansion
        dist_to_goal = math.hypot(goal[0]-cx, goal[1]-cy)
        if dist_to_goal < 5.0 or expansions % 50 == 0:
            curve = analytic_expand_if_possible(world_map, current, goal, radius, step)
            if curve is not None:
                best_goal_curve = curve
                # reconstruct path to current then append curve
                path = []
                # backtrack parents
                tmp = ckey
                while tmp in parents:
                    px, py, pth = parents[tmp][0]
                    path.append([px, py, angle_to_deg(pth)])
                    tmp = parents[tmp][1]
                path.append([start[0], start[1], angle_to_deg(start[2])])
                path.reverse()
                # append curve (already includes current onwards)
                path.extend(best_goal_curve)
                return path, expansions

        # expand motion primitives
        for nx, ny, nth, cost_inc in motion_primitives(cx, cy, cth, step, radius):
            if not in_bounds(world_map, nx, ny):
                continue
            seg_pts = sample_motion(world_map, cx, cy, cth, nx, ny, nth, step_samples=5)
            if collision_along_segment(world_map, seg_pts):
                continue
            nkey = (int(round(nx)), int(round(ny)), theta_to_bin(nth))

            # 距障惩罚
            d_obs = dist_map[int(round(nx)), int(round(ny))]
            if d_obs <= safety_buffer:
                continue  # 视为碰撞
            obs_cost = w_obs / (d_obs + 1.0)

            # 转弯惩罚
            prev_dir = last_dir.get(ckey, None)
            cur_dir = (nx - cx, ny - cy)
            turn_cost = 0.0
            if prev_dir is not None:
                dot = prev_dir[0]*cur_dir[0] + prev_dir[1]*cur_dir[1]
                norm1 = math.hypot(*prev_dir)
                norm2 = math.hypot(*cur_dir)
                if norm1 > 1e-6 and norm2 > 1e-6:
                    cos_angle = dot / (norm1 * norm2)
                    if cos_angle < 0.99:  # 夹角大于约8°
                        turn_cost = w_turn

            tentative_g = g_cost[ckey] + cost_inc + obs_cost + turn_cost
            if nkey not in g_cost or tentative_g < g_cost[nkey]:
                g_cost[nkey] = tentative_g
                parents[nkey] = ((nx, ny, nth), ckey)
                last_dir[nkey] = cur_dir
                hval = heuristic_dubins((nx, ny, nth), goal, radius)
                pq[nkey] = tentative_g + hval
                state_map[nkey] = (nx, ny, nth)

    # If failed, return empty
    return [], expansions

###  END CODE HERE  ###


def Self_driving_path_planner(world_map, start_pos, goal_pos):
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

    # Assumptions: default headings (deg). You can change these as needed.
    start_heading_deg = 0.0
    goal_heading_deg = 0.0
    step = 1.0
    radius = 5.0
    t0 = time.time()
    raw_path, expansions = hybrid_a_star(world_map, start_pos, goal_pos,
                                     start_heading_deg=start_heading_deg,
                                     goal_heading_deg=goal_heading_deg,
                                     step=step, radius=radius, theta_bins=72)
    dt = (time.time() - t0) * 1000.0
    print(f"Hybrid A* expansions: {expansions}, time: {dt:.1f} ms, path points: {len(raw_path)}")

    # Post smoothing with Dubins shortcut
    if raw_path:
        pre_len = path_length(raw_path)
        t1 = time.time()
        path = dubins_shortcut_smooth(world_map, raw_path, radius=radius, step=step,
                                        iterations=120, min_gap=8, seed=42)
        dt2 = (time.time() - t1) * 1000.0
        post_len = path_length(path)
        print(f"Dubins smoothing: {pre_len:.1f} -> {post_len:.1f} (Δ {pre_len-post_len:.1f}), time: {dt2:.1f} ms")
        return path

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