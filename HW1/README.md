# `easydubins`

## `dubins_path(start, end, radius)`

输入：
- start: tuple (s_x, s_y, s_yaw) — s_yaw 必须是弧度（radians）
- end: tuple (e_x, e_y, e_yaw) — e_yaw 必须是弧度
- radius: float — 转弯半径（同坐标单位）

输出：
- mode: 例如 ['L','S','L']（字符串或 list，其中小写表示反向）
- lengths: list/tuple [len1, len2, len3]（单位与坐标同）
- radii: list/tuple [r, r, r]（通常三者相同）

## `get_projection(start, end, solution)`

输入：
- start: (x, y, yaw_rad)
- end: (x, y, yaw_rad)
- solution: 返回自 dubins_path 的三元 (mode, lengths, radii)

输出：
- 一个 list，元素为 [x, y, heading]，其中 heading 是以度为单位（这是 split_line/tangent_angle 里使用的角度制）

## `get_curve(s_x, s_y, s_head, e_x, e_y, e_head, radius, max_line_distan`

输入：
- s_x, s_y: 起点坐标（float）
- s_head: 起点朝向（degrees） —— 注意：以度为单位！
- e_x, e_y: 终点坐标（float）
- e_head: 终点朝向（degrees）
- radius: 转弯半径（float）
- max_line_distance: 最大采样间距（float）——例如 0.5（同坐标单位）

输出：
- 直接返回 get_projection(...) 的结果：list of [x, y, heading_deg]