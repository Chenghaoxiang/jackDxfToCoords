import math

import ezdxf
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d,BSpline


def shape_circle(data,entity):
    # 提取圆参数
    cx, cy, cz = entity.dxf.center  # 中心坐标（绝对位置）
    radius = entity.dxf.radius  # 圆的半径

    # 生成每个角度的数据
    for theta_deg in range(0, 360):
        theta_rad = math.radians(theta_deg)

        # 直接计算笛卡尔坐标（圆无需旋转或比例变换）
        x = cx + radius * math.cos(theta_rad)
        y = cy + radius * math.sin(theta_rad)

        data.append({
            "角度(度)": theta_deg,
            "距离": round(radius, 6),  # 恒定半径
            "X坐标": round(x, 6),
            "Y坐标": round(y, 6),
        })

def shape_rectangle(data,entity):
    points = list(entity.vertices())
    # 计算每条边的长度
    lengths = [
        math.hypot(points[1][0] - points[0][0], points[1][1] - points[0][1]),  # 边1
        math.hypot(points[2][0] - points[1][0], points[2][1] - points[1][1]),  # 边2
        math.hypot(points[3][0] - points[2][0], points[3][1] - points[2][1]),  # 边3
        math.hypot(points[0][0] - points[3][0], points[0][1] - points[3][1]),  # 边4
    ]

    # 找出最长边和最短边
    long_edge_length = max(lengths)
    short_edge_length = min(lengths)

    # 根据长短边的长度赋值
    if lengths[0] == long_edge_length or lengths[2] == long_edge_length:  # 边1或边3是长边
        a = long_edge_length / 2  # 长边
        b = short_edge_length / 2  # 短边
    else:  # 边2或边4是长边
        a = long_edge_length / 2  # 长边
        b = short_edge_length / 2  # 短边

    # 计算中心坐标
    cx = sum(p[0] for p in points) / 4
    cy = sum(p[1] for p in points) / 4

    # 计算旋转角度(phi)
    dx = points[1][0] - points[0][0]
    dy = points[1][1] - points[0][1]
    phi = math.atan2(dy, dx)

    # 生成每个角度的数据（360个点）
    data = []
    for theta_deg in range(0, 360):
        theta_rad = math.radians(theta_deg)
        theta_rel = theta_rad - phi  # 转换为局部坐标系角度

        # 计算局部坐标系中的极径（矩形边界交点）
        cos_theta = math.cos(theta_rel)
        sin_theta = math.sin(theta_rel)

        if abs(cos_theta) < 1e-9:  # 避免除以零
            r_local = b / abs(sin_theta) if sin_theta != 0 else 0
        elif abs(sin_theta) < 1e-9:
            r_local = a / abs(cos_theta) if cos_theta != 0 else 0
        else:
            r_x = a / abs(cos_theta)  # x方向最大延伸
            r_y = b / abs(sin_theta)  # y方向最大延伸
            r_local = min(r_x, r_y)  # 取较小值确保在边界内

        # 计算局部坐标并旋转回全局坐标系
        x_local = r_local * cos_theta
        y_local = r_local * sin_theta
        x_rot = x_local * math.cos(phi) - y_local * math.sin(phi)
        y_rot = x_local * math.sin(phi) + y_local * math.cos(phi)

        # 将结果转换为全局坐标
        x = cx + x_rot
        y = cy + y_rot

        # 添加结果到数据列表
        data.append({
            "角度(度)": theta_deg,
            "距离": round(math.hypot(x - cx, y - cy), 6),
            "X坐标": round(x, 6),
            "Y坐标": round(y, 6),
        })
        # 现在 data 包含每个角度下的坐标和距离信息

def shape_ellipse(data,entity):
    # 提取椭圆参数
    cx, cy, cz = entity.dxf.center  # 中心坐标
    dx, dy, dz = entity.dxf.major_axis  # 长轴向量（相对于中心）
    a = math.hypot(dx, dy)  # 长轴长度
    phi = math.atan2(dy, dx)  # 旋转角度（弧度）
    ratio = entity.dxf.ratio  # 短轴比例
    b = a * ratio  # 短轴长度

    # 生成每个角度的数据
    # data = []
    for theta_deg in range(0, 360):
        theta_rad = math.radians(theta_deg)
        theta_rel = theta_rad - phi  # 相对于椭圆旋转的角度

        # 计算极径（椭圆在该方向上的半径）
        denominator = math.sqrt((b * math.cos(theta_rel)) ** 2 + (a * math.sin(theta_rel)) ** 2)
        r = (a * b) / denominator  # 极径公式

        # 计算笛卡尔坐标
        x = cx + r * math.cos(theta_rad)
        y = cy + r * math.sin(theta_rad)

        data.append({
            "角度(度)": theta_deg,
            "距离": round(r, 6),
            "X坐标": round(x, 6),  # 保留6位小数
            "Y坐标": round(y, 6),
        })

def shape_line(data,entity):
    start_point = entity.dxf.start  # 获取起始点
    end_point = entity.dxf.end  # 获取终止点

    # 计算起点和终点之间的坐标
    start_x, start_y, start_z = start_point
    end_x, end_y, end_z = end_point

    # 计算每个点需要的增量
    num_points = 360
    delta_x = (end_x - start_x) / (num_points - 1)
    delta_y = (end_y - start_y) / (num_points - 1)
    #二维不需要z轴坐标
    #delta_z = (end_z - start_z) / (num_points - 1)

    # 生成360个点
    for i in range(num_points):
        result_x = start_x + i * delta_x
        result_y = start_y + i * delta_y
        #result_z = start_z + i * delta_z
        data.append({
            "x轴坐标": round(result_x,6),
            "y轴坐标": round(result_y,6),
        })

def shape_arc(data, entity):
    #获取圆心的坐标
    center_x = entity.dxf.center_x
    center_y = entity.dxf.center_y
    center_z = entity.dxf.center_z

    #获取半径
    radius = entity.dxf.radius

    #获取起始角度和结束角度
    start_angle = entity.dxf.start_angle
    end_angle = entity.dxf.end_angle

    # 计算每个点需要的增量
    num_points = 360

    # 计算角度增量
    angle_increment = (end_angle - start_angle) / (num_points - 1)

    # 计算每个点的坐标和角度
    for i in range(num_points):
        # 计算当前角度
        current_angle = start_angle + i * angle_increment

        # 将角度转换为弧度
        radians = math.radians(current_angle)

        # 计算 x 和 y 坐标
        x = center_x + radius * math.cos(radians)
        y = center_y + radius * math.sin(radians)

        # 将结果存储到 data 列表中
        data.append({
            "角度": round(current_angle,6),
            "距离": round(radius,6),
            "x坐标": round(x,6),
            "y坐标": round(y,6),
        })

#多边形polyline的辅助函数，计算角度
def calculate_angle(center_x, center_y, point):
    point_x, point_y = point
    radians = math.atan2(point_y - center_y, point_x - center_x)
    degrees = math.degrees(radians)  # 转换为度
    return degrees

def shape_polyline(data,entity):
    #计算多边形的中心点
    x_coords = []
    y_coords = []

    for vertex in entity.vertices():
        x_coords.append(vertex.dxf.x)
        y_coords.append(vertex.dxf.y)

    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)

    points = []

    # 计算多边形的弧长并均匀分配
    length = 0
    segments = []

    # 计算每条边的长度
    vertices = list(entity.vertices())
    vertex_count = len(vertices)

    for i in range(vertex_count):
        current_vertex = vertices[i]
        next_vertex = vertices[(i + 1) % vertex_count]  # 保持循环访问

        edge_length = math.sqrt((next_vertex.dxf.x - current_vertex.dxf.x) ** 2 +
                                (next_vertex.dxf.y - current_vertex.dxf.y) ** 2)
        segments.append(edge_length)
        length += edge_length

    # 计算360个均匀分布的点(按周长去)
    point_count = 360
    segment_proportion = length / point_count

    current_length = 0
    segment_index = 0

    for i in range(point_count):
        while current_length + segments[segment_index] < (i + 1) * segment_proportion:
            current_length += segments[segment_index]
            segment_index = (segment_index + 1) % vertex_count

        # 计算插值点
        ratio = ((i + 1) * segment_proportion - current_length) / segments[segment_index]
        vertex_a = vertices[segment_index]
        next_vertex = vertices[(segment_index + 1) % vertex_count]

        p_x = (1 - ratio) * vertex_a.dxf.x + ratio * next_vertex.dxf.x
        p_y = (1 - ratio) * vertex_a.dxf.y + ratio * next_vertex.dxf.y
        points.append((p_x, p_y))

    # 计算每个点与中心点的角度
    for point in points:
        #各个点对于多边形中心点的角度
        angle = calculate_angle(center_x,center_y, point)
        #各个点对于多边形中心点的距离
        distance = math.sqrt((point[0] - center_x) ** 2 + (point[1] - center_y) ** 2)
        data.append({
            "角度": angle,
            "距离": round(angle, 6),
            "x坐标": round(point[0]),
            "y坐标": round(point[1]),
        })

def shape_lwpolyline(msp, data):
    # 提取 LWPOLYLINE 顶点数据
    vertices = []
    for lwpolyline in msp.query('LWPOLYLINE'):  # 查找所有 LWPOLYLINE 实体
        for vertex in lwpolyline.vertices():  # 获取每个顶点的坐标
            vertices.append((vertex[0], vertex[1]))

    # 计算两个点之间的角度（与中心点的角度）
    def get_angle(center, point):
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        angle = math.degrees(math.atan2(dy, dx))  # 计算角度
        return angle

    # 计算两个点之间的距离
    def get_distance(center, point):
        return math.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)

    # 获取所有的 LWPOLYLINE 或 POLYLINE
    polylines = msp.query('LWPOLYLINE')

    # 遍历所有多段线
    for polyline in polylines:
        points = []

        # 获取顶点坐标
        for vertex in polyline.vertices():
            points.append((vertex[0], vertex[1]))  # 获取 (x, y) 坐标

        # 计算中心点（质心）
        center_x = sum(p[0] for p in points) / len(points)
        center_y = sum(p[1] for p in points) / len(points)
        center = (center_x, center_y)

        # 如果多段线是开口的，那么角度计算会基于每个点与中心点的角度
        for point in points:
            angle = get_angle(center, point)
            distance = get_distance(center, point)
            data.append({
                "角度": angle,
                "距离": round(distance, 6),
                "x坐标": round(point[0], 6),
                "y坐标": round(point[1], 6),
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


def spline_data(ctrl_pts,degree):
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

    s_positions = np.linspace(0, total_length, 90)
    t_of_s = interp1d(cumulative_length, t_dense, kind='linear')(s_positions)
    uniform_points = spline(t_of_s)
    return uniform_points

def shape_spline(dxf_path, data):
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    splines = msp.query('SPLINE')
    print("test331")

    datas = []

    # 遍历每个SPLINE并处理控制点
    control_points_list = []
    for spline in splines:
        # 获取控制点列表，每个点为(x, y, z)元组
        points = spline.control_points
        # 转换为NumPy数组
        ctrl_pts = np.array(points)
        degree = spline.dxf.degree
        uniform_points = spline_data(ctrl_pts, degree)
        for uniform_point in uniform_points:
            x = uniform_point[0]
            y = uniform_point[1]
            angle_deg, length = my_angle(x, y)
            datas.append({
                "角度": angle_deg,
                "距离": length,
                "x坐标": x,
                "y坐标": y,
            })

    data = sorted(data, key=lambda item: item["角度"])

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