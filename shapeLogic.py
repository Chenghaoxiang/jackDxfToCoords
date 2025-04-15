import math

import ezdxf
import numpy as np
import pandas as pd
import bisect
from ezdxf.math import Vec2
from scipy.interpolate import interp1d,BSpline


def shape_circle(msp,data,pointNum):
    for entity in msp:
        if entity.dxftype == "CIRCLE":
            # 提取圆参数
            cx, cy, cz = entity.dxf.center  # 中心坐标（绝对位置）
            radius = entity.dxf.radius  # 圆的半径
            # 计算每个点的角度（弧度）
            angle_step = math.radians(360.0 / pointNum)

            # 生成等分点坐标
            for i in range(pointNum):
                theta = i * angle_step
                x = cx + radius * math.cos(theta)
                y = cy + radius * math.sin(theta)

                # 将点坐标添加到data列表
                data.append({
                    "角度(度)": theta,
                    "距离": round(radius, 6),  # 恒定半径
                    "X坐标": round(x, 6),
                    "Y坐标": round(y, 6),
                })
            # 第一个圆输出完就可以了
            break

def shape_rectangle(msp,data,pointNum):
    rectangle_datas = []
    entity = None
    for entitys in msp:
        print(f"dxftype = {entitys.dxftype},closed={entitys.closed},len={len(entitys)}")
        if entitys.dxftype() == 'LWPOLYLINE' and entitys.closed and len(entitys) == 4:
            entity = entitys
            break

    if entity is None:
        data = []
    else:
        # 获取顶点坐标（转换为二维点）
        vertices = [(v[0], v[1]) for v in entity.vertices()]

        # 计算包围盒参数
        x_coords = [x for x, _ in vertices]
        y_coords = [y for _, y in vertices]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        center = ((x_min + x_max) / 2, (y_min + y_max) / 2)

        # 角度等分参数
        angle_step = 360.0 / pointNum

        # 遍历每个角度
        for i in range(pointNum):
            theta_deg = i * angle_step
            theta = math.radians(theta_deg)

            # 射线方向单位向量
            dx = math.cos(theta)
            dy = math.sin(theta)

            min_t = float('inf')
            intersect = None

            # 检查四边交点（轴对齐版）
            # 右边（x_max）
            if abs(dx) > 1e-9:
                t = (x_max - center[0]) / dx
                if t > 0:
                    y = center[1] + dy * t
                    if y_min <= y <= y_max and t < min_t:
                        min_t = t
                        intersect = (x_max, y)

            # 左边（x_min）
            if abs(dx) > 1e-9:
                t = (x_min - center[0]) / dx
                if t > 0:
                    y = center[1] + dy * t
                    if y_min <= y <= y_max and t < min_t:
                        min_t = t
                        intersect = (x_min, y)

            # 上边（y_max）
            if abs(dy) > 1e-9:
                t = (y_max - center[1]) / dy
                if t > 0:
                    x = center[0] + dx * t
                    if x_min <= x <= x_max and t < min_t:
                        min_t = t
                        intersect = (x, y_max)

            # 下边（y_min）
            if abs(dy) > 1e-9:
                t = (y_min - center[1]) / dy
                if t > 0:
                    x = center[0] + dx * t
                    if x_min <= x <= x_max and t < min_t:
                        min_t = t
                        intersect = (x, y_min)

            if intersect:
                # data.append((round(, 2), round(, 2)))
                x = intersect[0]
                y = intersect[1]
                angle = math.atan2(y, x) % 360
                data.append({
                    "角度(度)": angle,
                    "距离": math.sqrt(x ** 2 + y ** 2),
                    "X坐标": round(x, 6),
                    "Y坐标": round(y, 6),
                })
                # print(datas)
    data.sort(key = lambda item:item["角度(度)"])
    print(data)
        # 现在 data 包含每个角度下的坐标和距离信息

def shape_ellipse(data,entity,pointNum):
    # 提取椭圆参数
    cx, cy, cz = entity.dxf.center  # 中心坐标
    dx, dy, dz = entity.dxf.major_axis  # 长轴向量（相对于中心）
    a = math.hypot(dx, dy)  # 长轴长度
    phi = math.atan2(dy, dx)  # 旋转角度（弧度）
    ratio = entity.dxf.ratio  # 短轴比例
    b = a * ratio  # 短轴长度

    # 按角度均匀生成点
    for i in range(pointNum):
        theta_deg = i * 360.0 / pointNum  # 当前角度（度）
        theta_rad = math.radians(theta_deg)
        theta_rel = theta_rad - phi  # 相对于椭圆旋转的角度

        # 计算极径（椭圆在该方向上的半径）
        denominator = math.sqrt((b * math.cos(theta_rel)) ** 2 + (a * math.sin(theta_rel)) ** 2)
        r = (a * b) / denominator  # 极径公式

        # 计算笛卡尔坐标
        x = cx + r * math.cos(theta_rad)
        y = cy + r * math.sin(theta_rad)

        data.append({
            "角度(度)": round(theta_deg, 6),  # 保留6位小数
            "距离": round(r, 6),
            "X坐标": round(x, 6),
            "Y坐标": round(y, 6),
        })
    data.sort(key=lambda item:item["角度(度)"])

def shape_line(data,entity,pointNum):
    start_point = entity.dxf.start  # 获取起始点
    end_point = entity.dxf.end  # 获取终止点

    # 计算起点和终点之间的坐标
    start_x, start_y, start_z = start_point
    end_x, end_y, end_z = end_point

    # 计算每个点需要的增量
    num_points = pointNum
    delta_x = (end_x - start_x) / (num_points - 1)
    delta_y = (end_y - start_y) / (num_points - 1)
    #二维不需要z轴坐标
    #delta_z = (end_z - start_z) / (num_points - 1)

    # 生成360个点
    for i in range(num_points):
        result_x = start_x + i * delta_x
        result_y = start_y + i * delta_y
        #result_z = start_z + i * delta_z
        angle = math.atan2(result_y , result_x)       #求的是对于零点的坐标
        deg = math.degrees(angle)
        if deg < 0:
            deg += 360
        data.append({
            "角度(度)": round(deg, 6),  # 保留6位小数
            "距离": round(math.sqrt(result_x ** 2 + result_y **2), 6),
            "x轴坐标": round(result_x,6),
            "y轴坐标": round(result_y,6),
        })
    data.sort(key=lambda item:item["角度(度)"])

def shape_arc(data, msp,pointNum):

    arcs = []
    for entity in msp:
        if entity.dxftype() == "ARC":
            arcs.append(entity)
    arc_num = len(arcs)
    if arc_num == 0:
        return []  # 没有ARC时返回空列表

    base = pointNum // arc_num
    rem = pointNum % arc_num
    points_info = []

    for i, arc in enumerate(arcs):
        if i < rem:
            n = base + 1
        else:
            n = base

        if n <= 0:
            continue

        center = arc.dxf.center  # 圆心 (x, y, z)
        radius = arc.dxf.radius
        start_angle = arc.dxf.start_angle
        end_angle = arc.dxf.end_angle

        # 计算总角度delta
        delta = (end_angle - start_angle) % 360
        if delta == 0:
            delta = 360  # 处理完整的圆

        # 生成角度列表
        if n == 1:
            angles = [(start_angle + delta / 2) % 360]
        else:
            angle_step = delta / (n - 1)
            angles = [(start_angle + i * angle_step) % 360 for i in range(n)]

        # 计算每个点的坐标和角度、半径
        for theta_deg in angles:
            theta_rad = math.radians(theta_deg)
            x = center[0] + radius * math.cos(theta_rad)
            y = center[1] + radius * math.sin(theta_rad)
            z = center[2]  # 保持z坐标与圆心一致

            angle = math.atan2(y, x)  # 求的是对于零点的坐标
            deg = math.degrees(angle)
            if deg < 0:
                deg += 360

            # 将结果存储到 data 列表中
            data.append({
                "圆心角度": round(theta_deg,6),
                "圆心距离": round(radius,6),
                "零点角度":deg,
                "零点距离":math.sqrt(x ** 2 + y ** 2),
                "x坐标": round(x,6),
                "y坐标": round(y,6),
            })


def shape_polyline(data,msp,pointNum):
    # 收集所有多段线实体
    polylines = []
    for entity in msp:
        if entity.dxftype() in ['LWPOLYLINE', 'POLYLINE']:
            polylines.append(entity)

    if not polylines:
        return []

    # 计算每个多段线的长度
    pl_lengths = []
    for pl in polylines:
        vertices = []
        is_closed = False

        if pl.dxftype() == 'LWPOLYLINE':
            # 处理轻量多段线
            lw_points = pl.get_points('xy')
            vertices = [(p[0], p[1]) for p in lw_points]
            is_closed = pl.closed
        else:
            # 处理旧式多段线
            vertices = []
            for vertex in pl.vertices():
                loc = vertex.dxf.location
                vertices.append((loc.x, loc.y))
            is_closed = pl.is_closed

        # 计算多段线长度
        length = 0.0
        if len(vertices) < 2:
            pl_lengths.append(0.0)
            continue

        prev = vertices[0]
        for curr in vertices[1:]:
            dx = curr[0] - prev[0]
            dy = curr[1] - prev[1]
            length += math.hypot(dx, dy)
            prev = curr

        # 处理闭合线段
        if is_closed and len(vertices) >= 2:
            dx = vertices[0][0] - vertices[-1][0]
            dy = vertices[0][1] - vertices[-1][1]
            length += math.hypot(dx, dy)

        pl_lengths.append(length)

    total_length = sum(pl_lengths)
    if total_length <= 0:
        return []

    # 分配点数
    fractions = [(length / total_length) * pointNum for length in pl_lengths]
    allocations = [math.floor(f) for f in fractions]
    remainders = [f - a for f, a in zip(fractions, allocations)]
    remaining = pointNum - sum(allocations)

    # 分配剩余的点数
    indexed_remainders = sorted(enumerate(remainders), key=lambda x: -x[1])
    for i in range(remaining):
        if i < len(indexed_remainders):
            idx = indexed_remainders[i][0]
            allocations[idx] += 1

    # 收集所有点
    all_points = []

    for pl, pl_len, num_points in zip(polylines, pl_lengths, allocations):
        if num_points <= 0:
            continue

        # 提取顶点和闭合状态
        if pl.dxftype() == 'LWPOLYLINE':
            vertices = [(p[0], p[1]) for p in pl.get_points('xy')]
            is_closed = pl.closed
        else:
            vertices = [(v.dxf.location.x, v.dxf.location.y) for v in pl.vertices()]
            is_closed = pl.is_closed

        if len(vertices) < 1:
            continue

        # 构建线段和累积长度
        segments = []
        cumulative = [0.0]
        current_len = 0.0
        prev = vertices[0]

        for curr in vertices[1:]:
            dx = curr[0] - prev[0]
            dy = curr[1] - prev[1]
            seg_len = math.hypot(dx, dy)
            segments.append((prev, curr, seg_len))
            current_len += seg_len
            cumulative.append(current_len)
            prev = curr

        # 处理闭合线段
        if is_closed and len(vertices) >= 2:
            dx = vertices[0][0] - vertices[-1][0]
            dy = vertices[0][1] - vertices[-1][1]
            seg_len = math.hypot(dx, dy)
            segments.append((vertices[-1], vertices[0], seg_len))
            current_len += seg_len
            cumulative.append(current_len)

        total_len = current_len
        if total_len <= 0:
            continue

        # 计算步长
        if is_closed:
            step = total_len / num_points
        else:
            step = total_len / (num_points - 1) if num_points > 1 else 0

        for i in range(num_points):
            if is_closed:
                t = i * step
            else:
                t = i * step if num_points > 1 else 0

            if t > total_len:
                t = total_len

            # 确定所在线段
            idx = bisect.bisect_left(cumulative, t) - 1
            idx = max(0, min(idx, len(segments) - 1))

            seg_start, seg_end, seg_len = segments[idx]
            t_in_seg = t - cumulative[idx]

            if seg_len == 0:
                x, y = seg_start
            else:
                ratio = t_in_seg / seg_len
                ratio = max(0.0, min(1.0, ratio))
                dx = seg_end[0] - seg_start[0]
                dy = seg_end[1] - seg_start[1]
                x = seg_start[0] + dx * ratio
                y = seg_start[1] + dy * ratio

            angle = math.atan2(y, x)  # 求的是对于零点的坐标
            deg = math.degrees(angle)
            if deg < 0:
                deg += 360

            data.append({
                "角度": deg,
                "距离": round(math.sqrt(x ** 2 + y ** 2), 6),
                "x坐标": round(x,6),
                "y坐标": round(y,6),
            })


def compute_arc_length(v1, v2, bulge):
    dx = v2[0] - v1[0]
    dy = v2[1] - v1[1]
    d = math.hypot(dx, dy)
    theta = 4 * math.atan(abs(bulge))
    if theta == 0:
        return d  # 直线情况，但bulge不为0时理论上不会出现
    arc_length = (d * theta) / (2 * math.sin(theta / 2))
    return arc_length


def get_arc_point(S, E, bulge, local_s, seg_length):
    if seg_length == 0:
        return (S[0], S[1])

    dx = E[0] - S[0]
    dy = E[1] - S[1]
    d = math.hypot(dx, dy)
    if d == 0:
        return (S[0], S[1])

    Mx = (S[0] + E[0]) / 2
    My = (S[1] + E[1]) / 2
    h_x = (-dy * bulge) / 2
    h_y = (dx * bulge) / 2
    Ox = Mx + h_x
    Oy = My + h_y

    r = math.hypot(S[0] - Ox, S[1] - Oy)
    if r == 0:
        return (S[0], S[1])

    angle_S = math.atan2(S[1] - Oy, S[0] - Ox)
    theta = 4 * math.atan(bulge)

    delta_theta = (local_s / seg_length) * theta
    current_angle = angle_S + delta_theta

    x = Ox + r * math.cos(current_angle)
    y = Oy + r * math.sin(current_angle)
    return (x, y)


def sample_lwpolyline(pline, num_points):
    vertices = list(pline.get_points('xyb'))
    is_closed = pline.closed
    segments = []
    total_length = 0.0

    for i in range(len(vertices) - 1):
        v1 = vertices[i]
        v2 = vertices[i + 1]
        bulge = v1[2]
        start = (v1[0], v1[1])
        end = (v2[0], v2[1])
        if bulge == 0:
            seg_type = 'line'
            length = math.hypot(end[0] - start[0], end[1] - start[1])
        else:
            seg_type = 'arc'
            length = compute_arc_length(v1, v2, bulge)
        segments.append({
            'type': seg_type,
            'start': start,
            'end': end,
            'bulge': bulge,
            'start_length': total_length,
            'end_length': total_length + length,
            'length': length
        })
        total_length += length

    if is_closed and len(vertices) > 1:
        v1 = vertices[-1]
        v2 = vertices[0]
        bulge = v1[2]
        start = (v1[0], v1[1])
        end = (v2[0], v2[1])
        if bulge == 0:
            seg_type = 'line'
            length = math.hypot(end[0] - start[0], end[1] - start[1])
        else:
            seg_type = 'arc'
            length = compute_arc_length(v1, v2, bulge)
        segments.append({
            'type': seg_type,
            'start': start,
            'end': end,
            'bulge': bulge,
            'start_length': total_length,
            'end_length': total_length + length,
            'length': length
        })
        total_length += length

    if total_length == 0 or num_points == 0:
        return []

    if num_points == 1:
        s_list = [total_length * 0.5]
    else:
        s_list = [i * (total_length / (num_points - 1)) for i in range(num_points)]

    sampled_points = []
    for s in s_list:
        if s >= total_length:
            last_seg = segments[-1]
            sampled_points.append(last_seg['end'])
            continue
        for seg in segments:
            if seg['start_length'] <= s < seg['end_length']:
                local_s = s - seg['start_length']
                if seg['type'] == 'line':
                    t = local_s / seg['length'] if seg['length'] != 0 else 0.0
                    x = seg['start'][0] + t * (seg['end'][0] - seg['start'][0])
                    y = seg['start'][1] + t * (seg['end'][1] - seg['start'][1])
                else:
                    x, y = get_arc_point(seg['start'], seg['end'], seg['bulge'], local_s, seg['length'])
                sampled_points.append((x, y))
                break
        else:
            if s == total_length:
                last_seg = segments[-1]
                sampled_points.append(last_seg['end'])
    return sampled_points

def shape_lwpolyline(msp, data,pointNum):
    lwpolylines = [entity for entity in msp if entity.dxftype() == 'LWPOLYLINE']
    polyline_lengths = []
    total_length = 0.0

    for pline in lwpolylines:
        length = 0.0
        vertices = list(pline.get_points('xyb'))
        for i in range(len(vertices) - 1):
            v1 = vertices[i]
            v2 = vertices[i + 1]
            bulge = v1[2]
            if bulge == 0:
                dx = v2[0] - v1[0]
                dy = v2[1] - v1[1]
                segment_length = math.hypot(dx, dy)
            else:
                segment_length = compute_arc_length(v1, v2, bulge)
            length += segment_length
        if pline.closed and len(vertices) > 1:
            v1 = vertices[-1]
            v2 = vertices[0]
            bulge = v1[2]
            if bulge == 0:
                dx = v2[0] - v1[0]
                dy = v2[1] - v1[1]
                segment_length = math.hypot(dx, dy)
            else:
                segment_length = compute_arc_length(v1, v2, bulge)
            length += segment_length
        polyline_lengths.append((pline, length))
        total_length += length

    if total_length == 0 or not lwpolylines:
        return []

    points = []
    for pline, length in polyline_lengths:
        ratio = length / total_length
        num = int(round(pointNum * ratio))
        if num <= 0:
            continue
        sampled = sample_lwpolyline(pline, num)
        points.extend(sampled)

    if len(points) > pointNum:
        points = points[:pointNum]

    for point in points:
        x, y = point[0], point[1]
        angle = math.atan2(y, x)  # 求的是对于零点的坐标
        deg = math.degrees(angle)
        if deg < 0:
            deg += 360

        data.append({
            "零点角度":deg,
            "零点距离":round(math.sqrt(x ** 2 + y ** 2),6),
            "x坐标": round(x, 6),
            "y坐标": round(y, 6),
        })


def my_angle(x,y):
    # 计算线段的长度
    length = math.hypot(x, y)
    # 计算角度（弧度）
    angle_rad = math.atan2(y, x)
    # 转换为角度
    angle_deg = math.degrees(angle_rad)
    # 调整到 [0, 360) 范围
    if angle_deg < 0:
        angle_deg += 360.0
    return angle_deg, length


def spline_data(ctrl_pts,degree,num):
    # 假设为3次开放均匀B样条
    #degree = 3
    n = len(ctrl_pts)
    knots = np.zeros(n + degree + 1)
    knots[:degree + 1] = 0
    knots[-degree - 1:] = n - degree  # 19
    knots[degree + 1:-degree - 1] = np.arange(1, n - degree)

    # 创建B样条
    spline = BSpline(knots, ctrl_pts, degree)

    # 弧长均匀采样
    t_dense = np.linspace(knots[degree], knots[-degree - 1], 1000)
    points_dense = spline(t_dense)
    diffs = np.diff(points_dense, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    cumulative_length = np.insert(np.cumsum(distances), 0, 0)
    total_length = cumulative_length[-1]

    s_positions = np.linspace(0, total_length, num)
    t_of_s = interp1d(cumulative_length, t_dense, kind='linear')(s_positions)
    uniform_points = spline(t_of_s)
    return uniform_points

def shape_spline(dxf_path, data,pointNum):
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    splines = msp.query('SPLINE')
    num = len(splines)

    # 遍历每个SPLINE并处理控制点
    control_points_list = []
    for spline in splines:
        # 获取控制点列表，每个点为(x, y, z)元组
        points = spline.control_points
        # 转换为NumPy数组
        ctrl_pts = np.array(points)
        degree = spline.dxf.degree
        uniform_points = spline_data(ctrl_pts, degree,int(pointNum / num))
        for uniform_point in uniform_points:
            x = uniform_point[0]
            y = uniform_point[1]
            angle_deg, length = my_angle(x, y)
            data.append({
                "角度": angle_deg,
                "距离": length,
                "x坐标": x,
                "y坐标": y,
            })

    data.sort(key=lambda item: item["角度"])

def shape_composite(msp, data):
    points = []

    # 获取 LINE 实体中的点
    def composite_line(line, num_points=360):
        # 计算线段上的等间距点
        x1, y1, _ = line.dxf.start
        x2, y2, _ = line.dxf.end
        x_vals = np.linspace(x1, x2, num_points)
        y_vals = np.linspace(y1, y2, num_points)
        return list(zip(x_vals, y_vals))

    # 获取 ARC 实体中的点
    def composite_arc(arc, num_points=360):
        # 计算圆弧上的等间距点
        cx, cy, _ = arc.dxf.center  # 圆心
        radius = arc.dxf.radius  # 半径
        start_angle = arc.dxf.start_angle  # 起始角度
        end_angle = arc.dxf.end_angle  # 结束角度

        # 角度范围
        angles = np.linspace(start_angle, end_angle, num_points)

        # 计算每个角度对应的坐标
        x_vals = cx + radius * np.cos(np.radians(angles))
        y_vals = cy + radius * np.sin(np.radians(angles))
        return list(zip(x_vals, y_vals))

    # 获取 CIRCLE 实体中的点
    def composite_circle(circle, num_points=360):
        # 计算圆上的等间距点
        cx, cy, _ = circle.dxf.center
        radius = circle.dxf.radius
        angles = np.linspace(0, 360, num_points)
        x_vals = cx + radius * np.cos(np.radians(angles))
        y_vals = cy + radius * np.sin(np.radians(angles))
        return list(zip(x_vals, y_vals))

    # 获取 SPLINE 实体中的点
    def composite_spline(spline, num_points=360):
        # 获取样条的控制点
        control_points = spline.control_points
        # 通过控制点插值生成均匀的采样点（简化处理，实际应用可能需要更精确的样条插值方法）
        x_vals = np.linspace(control_points[0][0], control_points[-1][0], num_points)
        y_vals = np.linspace(control_points[0][1], control_points[-1][1], num_points)
        return list(zip(x_vals, y_vals))

    # 获取 POLYLINE 或 LWPOLYLINE 实体中的点
    def composite_polyline(polyline, num_points=360):
        points = []
        if polyline.is_closed:  # 如果是封闭的多段线，确保首尾相连
            polyline.append(polyline[0])

        # 遍历每个线段
        for i in range(len(polyline) - 1):
            start_point = polyline[i]
            end_point = polyline[i + 1]
            x_vals = np.linspace(start_point[0], end_point[0], num_points)
            y_vals = np.linspace(start_point[1], end_point[1], num_points)
            points.extend(zip(x_vals, y_vals))
        return points

    # 遍历所有图形实体
    for entity in msp:
        if entity.dxftype() == 'LINE':
            points.extend(composite_line(entity))
        elif entity.dxftype() == 'ARC':
            points.extend(composite_arc(entity))
        elif entity.dxftype() == 'CIRCLE':
            points.extend(composite_circle(entity))
        elif entity.dxftype() == 'SPLINE':
            points.extend(composite_spline(entity))
        elif entity.dxftype() == 'LWPOLYLINE' or entity.dxftype() == 'POLYLINE':
            points.extend(composite_polyline(entity))

    # 输出提取的点，最多取 360 个
    for point in points[:360]:
        data.append({
            "x坐标": round(point[0], 6),
            "y坐标": round(point[1], 6),
        })