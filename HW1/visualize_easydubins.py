import math
import numpy as np
import matplotlib.pyplot as plt
from easydubins.dubin_path import dubins_path, get_projection, get_curve

def main():
    # 起点/终点（注意：get_curve 接受 heading 为度）
    s_x, s_y, s_head_deg = 10.0, 10.0, 0.0
    e_x, e_y, e_head_deg = 50.0, 50.0, 180.0
    radius = 5.0
    max_line_dist = 0.5

    # 使用 get_curve（包装函数，会设置采样步长，输入 heading 为度）
    pts = get_curve(s_x, s_y, s_head_deg, e_x, e_y, e_head_deg, radius, max_line_dist)
    pts = np.array(pts) if pts else np.empty((0, 3))

    # 也可以直接调用 dubins_path + get_projection（注意这里 start/end 的 heading 要用弧度）
    start_rad = (s_x, s_y, math.radians(s_head_deg))
    end_rad = (e_x, e_y, math.radians(e_head_deg))
    solution = dubins_path(start_rad, end_rad, radius)
    proj = get_projection(start_rad, end_rad, solution)
    proj = np.array(proj) if proj else np.empty((0, 3))

    print(f"get_curve points: {pts.shape[0]}")
    print(f"get_projection points: {proj.shape[0]}")

    # 绘图
    plt.figure(figsize=(8, 8))
    if pts.size:
        plt.plot(pts[:, 0], pts[:, 1], '-r', label='get_curve samples')
        # headings in pts are in degrees; convert to vector (dx, dy)
        headings = pts[:, 2].astype(float)
        dx = np.cos(np.radians(90.0 - headings))
        dy = np.sin(np.radians(90.0 - headings))
        plt.quiver(pts[:, 0], pts[:, 1], dx, dy, color='r', scale=10, width=0.003)
    if proj.size:
        plt.plot(proj[:, 0], proj[:, 1], '--g', label='get_projection samples')
        headings_p = proj[:, 2].astype(float)
        dxp = np.cos(np.radians(90.0 - headings_p))
        dyp = np.sin(np.radians(90.0 - headings_p))
        plt.quiver(proj[:, 0], proj[:, 1], dxp, dyp, color='g', scale=10, width=0.002)

    plt.scatter([s_x], [s_y], c='g', marker='o', label='start')
    plt.scatter([e_x], [e_y], c='r', marker='x', label='end')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title('easydubins: path samples with heading arrows')
    plt.show()


if __name__ == '__main__':
    main()