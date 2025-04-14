import math

import ezdxf
from collections import defaultdict

from ezdxf import DXFValueError
from ezdxf.math import Vec3


def get_acad_version(version_code):
    """版本代码转换函数（包含AutoCAD 2018）"""
    versions = {
        'AC1009': 'AutoCAD R12',
        'AC1012': 'AutoCAD R13',
        'AC1014': 'AutoCAD R14',
        'AC1015': 'AutoCAD 2000',
        'AC1018': 'AutoCAD 2004',
        'AC1021': 'AutoCAD 2007',
        'AC1024': 'AutoCAD 2010',
        'AC1027': 'AutoCAD 2013',
        'AC1032': 'AutoCAD 2018',   # 新增2018版本对应
        'AC1061': 'AutoCAD 2023'    # 新增2023版本对应
    }
    return versions.get(version_code, f"未知版本 ({version_code})")  # 处理未知版本

def analyze_dxf(file_path):
    """核心分析函数"""
    try:
        doc = ezdxf.readfile(file_path)
    except Exception as e:
        return f"错误: {str(e)}"

    splines = []
    circles = []
    ellipsis = []
    lines = []
    arcs = []
    polylines = []
    lwpolylines = []

    version = get_acad_version(doc.dxfversion)
    # 基础信息
    info = [
        f"[基础信息]",
        f"AutoCAD版本: {version}",
        f"图层数量: {len(doc.layers)}",
        f"用户块定义: {len([b for b in doc.blocks if not b.name.startswith('*')])}",
        f"模型空间实体: {len(doc.modelspace())}"
    ]

    # 详细信息
    details = ["\n[图层信息]"]
    details.extend(f"{layer.dxf.name}: 颜色({layer.dxf.color}), 线型({layer.dxf.linetype})"
                   for layer in doc.layers)

    # 实体类型统计
    counter = defaultdict(int)
    for entity in doc.modelspace():
        counter[entity.dxftype()] += 1
        dxf_type = entity.dxftype()
        if dxf_type == 'SPLINE':
            try:
                splines.append({
                    'degree': entity.dxf.degree,
                    'control': len(entity.control_points),
                    'knots': len(entity.knots),
                    'fit': len(entity.fit_points),
                    'periodic': '是' if entity.closed else '否',
                })
            except AttributeError as e:
                print(e)
                pass  # 处理不完整的SPLINE实体
        elif dxf_type == 'CIRCLE':
            try:
                circles.append({
                    'center': entity.dxf.center,  # 圆心坐标 (x,y,z)[position=3][position=16]
                    'radius': entity.dxf.radius,  # 圆半径[position=3][position=16]
                    'layer': entity.dxf.layer  # 所在图层[position=11][position=14]
                })
            except AttributeError as e:
                print(e)
                pass  # 处理不完整的CIRCLE实体
        elif dxf_type == 'ELLIPSE':
            try:
                ellipsis.append({
                    'center': entity.dxf.center,  # 圆心坐标 (x,y,z)
                    'major_axis': entity.dxf.major_axis, #长轴矢量(x,y)
                    'ratio': entity.dxf.ratio, #短长轴比例
                    'start_param': entity.dxf.start_param,  # 起始参数(弧度)
                    'end_param': entity.dxf.end_param,  # 结束参数(弧度)
                    'layer': entity.dxf.layer  # 所在图层
                })
            except AttributeError as e:
                print(e)
                pass
        elif dxf_type == 'LINE':
            try:
                lines.append({
                    'start': entity.dxf.start,
                    'end': entity.dxf.end,
                    'length': entity.dxf.end.distance(entity.dxf.start),  # 新增线段长度计算
                    'layer': entity.dxf.layer,
                })
            except AttributeError as e:
                print(e)
                pass
        elif dxf_type == 'ARC':
            try:
                arcs.append({
                    'center': entity.dxf.center,
                    'radius': entity.dxf.radius,
                    'start_angle': entity.dxf.start_angle,  # 单位：度
                    'end_angle': entity.dxf.end_angle,
                    'layer': entity.dxf.layer,
                    'angle_diff': (entity.dxf.end_angle - entity.dxf.start_angle) % 360,  # 角度差标准化
                    'length': entity.dxf.radius * math.radians(  # 弧长计算公式
                        abs((entity.dxf.end_angle - entity.dxf.start_angle) % 360))
                })
            except AttributeError as e:
                print(e)
                pass
        elif dxf_type == 'POLYLINE':
            try:
                vertex_list = [vertex.dxf.location for vertex in entity.vertices]
                total_length = sum(
                    Vec3(v1).distance(Vec3(v2))
                    for v1, v2 in zip(vertex_list, vertex_list[1:])
                ) if len(vertex_list) > 1 else 0.0

                polylines.append({
                    'vertices': vertex_list,
                    'is_closed': entity.is_closed,
                    'total_length': round(total_length, 3),  # 保持3位小数精度
                    'elevation': round(entity.dxf.elevation, 3),
                    'thickness': round(entity.dxf.thickness, 3),
                    'width': round(entity.dxf.const_width, 3) if hasattr(entity.dxf, 'const_width') else 0.0,
                    'layer': entity.dxf.layer
                })
            except (AttributeError, KeyError) as e:  # 增加KeyError容错
                print(f"Polyline解析错误: {str(e)}")
                pass

        elif dxf_type == 'LWPOLYLINE':
            try:
                # 获取带高程值的顶点坐标（三维化处理）
                points = [
                    (x, y, entity.dxf.elevation)
                    for x, y in entity.get_points('xy')
                ]

                points = [
                    (
                        float(round(x, 6)),  # 转换x坐标
                        float(round(y, 6)),  # 转换y坐标
                        float(round(entity.dxf.elevation, 6))  # 转换z坐标
                    )
                    for x, y in entity.get_points('xy')
                ]

                # 计算总长度（兼容空顶点情况）
                total_length = sum(
                    Vec3(p1).distance(Vec3(p2))
                    for p1, p2 in zip(points, points[1:])
                ) if len(points) > 1 else 0.0

                lwpolylines.append({
                    'vertices': points,
                    'is_closed': entity.closed,
                    'total_length': round(total_length, 6),
                    'elevation': round(entity.dxf.elevation, 6),
                    'width': round(entity.dxf.const_width, 6) if entity.dxf.hasattr('const_width') else 0.0,
                    'layer': entity.dxf.layer,
                    'thickness': round(entity.dxf.thickness, 6)  # 继承POLYLINE属性
                })
            except (AttributeError, DXFValueError) as e:
                print(f"LWPOLYLINE解析异常: {str(e)}")
                pass

    entity_stats = ["\n[实体类型统计]"]
    entity_stats.extend(f"{etype}: {count}" for etype, count in sorted(counter.items()))

    line_info = []
    if lines:
        line_info.append("\n[Line参数详情]")
        for idx, line in enumerate(lines, 1):
            # 格式化三维坐标为(x,y,z)，保留3位小数
            start = tuple(round(coord, 6) for coord in line['start'])
            end = tuple(round(coord, 6) for coord in line['end'])
            line_details = [
                f"Line {idx}:",
                f"- 起点坐标: {start[0:2]}",  # 兼容二维显示模式
                f"- 终点坐标: {end[0:2]}",
                f"- 长度: {line['length']:.3f} units\n"  # 新增几何特征计算
                f"- 所在图层: {line['layer']}\n",  # 图层属性
            ]
            line_info.extend(line_details)
    else:
        lines.clear()

    arc_info = []
    if arcs:
        arc_info.append("\n[Arc参数详情]")
        for idx, arc in enumerate(arcs, 1):
            # 三维坐标格式化处理
            center = tuple(round(coord, 3) for coord in arc['center'])
            arc_details = [
                f"Arc {idx}:",
                f"- 投影中心(圆心): {center[0:2]}",  # 二维坐标简化显示
                f"- 半径值: {arc['radius']:.3f} units",
                f"- 起始角: {arc['start_angle']:.1f}°",  # 角度制显示
                f"- 终止角: {arc['end_angle']:.1f}°",
                f"- 包角: {arc['angle_diff']:.1f}°",  # 实际扫掠角度
                f"- 弧长: {arc['length']:.3f} units",  # 几何特征计算
                f"- 所在图层: {arc['layer']}"  # 图层信息
            ]
            arc_info.extend(arc_details)
    else:
        arcs.clear()

    ellipsis_info = []
    if ellipsis:
        ellipsis_info.append("\n[Ellipse参数详情]")
        for idx, ellipse in enumerate(ellipsis, 1):
            # 格式化圆心坐标为(x,y,z)，保留3位小数
            center = tuple(round(coord, 3) for coord in ellipse['center'])
            major_axis = tuple(round(coord, 6) for coord in ellipse['major_axis'])
            ellipse_details = [
                f"ellipse {idx}:",
                f"- 圆心坐标: {center[0:2]}",  # 三维坐标参考
                f"- 长轴矢量: {major_axis[0:2]}",     # 显示矢量方向
                f"- 短长轴比例: {ellipse['ratio']:.3f}", # 保留3位精度
                f"- 起始参数: {ellipse['start_param']:.3f} rad", # 弧度值
                f"- 结束参数: {ellipse['end_param']:.3f} rad",
                f"- 所在图层: {ellipse['layer']}\n"       # 图层名称参考
            ]
            ellipsis_info.extend(ellipse_details)
    else:
        ellipsis.clear()

    circle_info = []
    if circles:
        circle_info.append("\n[Circle参数详情]")
        for idx, circle in enumerate(circles, 1):
            # 格式化圆心坐标为(x,y,z)，保留3位小数
            center = tuple(round(coord, 3) for coord in circle['center'])
            circle_details = [
                f"Circle {idx}:",
                f"- 圆心坐标: {center[0:2]}",  # 三维坐标参考
                f"- 半径: {circle['radius']:.3f}",  # 保留3位精度
                f"- 所在图层: {circle['layer']}\n"  # 图层名称参考
            ]
            circle_info.extend(circle_details)
    else:
        circles.clear()

    polyline_info = []
    if polylines:
        polyline_info.append("\n[Polyline参数详情]")
        for idx, pline in enumerate(polylines, 1):
            # 顶点坐标格式化处理
            vertices = [tuple(round(c, 3) for c in vertex) for vertex in pline['vertices']]
            vertex_sample = ""
            if vertices:
                if len(vertices) > 2:
                    vertex_sample = f"{vertices[0][0:2]} → {vertices[1][0:2]} ... → {vertices[-1][0:2]}"
                else:
                    vertex_sample = " → ".join(str(v[0:2]) for v in vertices)

            pline_details = [
                f"Polyline {idx}:",
                f"- 闭合状态: {'是' if pline['is_closed'] else '否'}",
                f"- 顶点数量: {len(vertices)}",
                f"- 顶点序列: {vertex_sample}" if vertex_sample else "",
                f"- 总长度: {pline['total_length']:.3f} units",
                f"- 标高: {pline['elevation']:.3f}",
                f"- 厚度: {pline['thickness']:.3f}",
                f"- 线宽: {pline['width']:.3f}" if pline['width'] else "",
                f"- 所在图层: {pline['layer']}\n"
            ]
            polyline_info.extend([d for d in pline_details if d])  # 过滤空字符串
    else:
        polylines.clear()

    lwpolyline_info = []
    if lwpolylines:
        lwpolyline_info.append("\n[LwPolyline参数详情]")
        for idx, pline in enumerate(lwpolylines, 1):
            # 顶点坐标格式化（自动附加Z值）
            vertices = [tuple(round(c, 3) for c in vertex) for vertex in pline['vertices']]

            # 顶点序列智能简写
            vertex_sample = ""
            if vertices:
                if len(vertices) > 3:
                    # 显示前2个和最后1个顶点（二维投影）
                    sample = [f"{vertices[0][0:2]}", f"{vertices[1][0:2]}", "...", f"{vertices[-1][0:2]}"]
                    vertex_sample = " → ".join(sample)
                else:
                    vertex_sample = " → ".join(
                        f"({x:.3f}, {y:.3f})"  # 直接使用格式化字符串
                        for (x, y, _) in vertices  # 解包三维坐标
                    )

            pline_details = [
                f"LwPolyline {idx}:",
                f"- 闭合状态: {'是' if pline['is_closed'] else '否'}",
                f"- 顶点数量: {len(vertices)}",
                f"- 路径示意: {vertex_sample}" if vertex_sample else "",
                f"- 空间长度: {pline['total_length']:.3f} units",
                f"- 基准高程(共享z轴高度): {pline['elevation']:.3f}",
                f"- 全局线宽: {pline['width']:.3f}" if pline['width'] > 0 else "",
                f"- 拉伸厚度: {pline['thickness']:.3f}" if pline['thickness'] != 0 else "",
                f"- 所在图层: {pline['layer']}\n"
            ]
            # 过滤空条目并追加
            lwpolyline_info.extend([detail for detail in pline_details if detail])
    else:
        lwpolylines.clear()

    # Spline参数详情
    spline_info = []
    if splines:
        spline_info.append("\n[Spline参数详情]")
        for idx, spline in enumerate(splines, 1):
            spline_details = [
                f"Spline {idx}:",
                f"- 阶数: {spline['degree']}",
                f"- 控制点: {spline['control']}",
                f"- 节点数: {spline['knots']}",
                f"- 拟合点: {spline['fit']}",
                f"- 周期性: {spline['periodic']}\n",
            ]
            spline_info.extend(spline_details)
    else:
        splines.clear()
        #spline_info.append("\n[无Spline实体]")

    result = []
    result.extend(info+entity_stats)
    if ellipsis:
        result.extend(ellipsis_info)
    if circles:
        result.extend(circle_info)
    if splines:
        result.extend(spline_info)
    if lines:
        result.extend(line_info)
    if arcs:
        result.extend(arc_info)
    if polylines:
        result.extend(polyline_info)
    if lwpolylines:
        result.extend(lwpolyline_info)

    result.extend(details)
    return "\n".join(result)