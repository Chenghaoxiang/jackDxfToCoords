import ezdxf
from collections import defaultdict

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
        'AC1032': 'AutoCAD 2018',  # 新增2018版本对应
        'AC1061': 'AutoCAD 2023'
    }
    return versions.get(version_code, f"未知版本 ({version_code})")  # 处理未知版本

def analyze_dxf(file_path):
    """核心分析函数"""
    try:
        doc = ezdxf.readfile(file_path)
    except Exception as e:
        return f"错误: {str(e)}"

    splines = []

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
                print()
            except Exception as e:
                print(e)

    entity_stats = ["\n[实体类型统计]"]
    entity_stats.extend(f"{etype}: {count}" for etype, count in sorted(counter.items()))

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
                f"- 周期性: {spline['periodic']}",
            ]
            spline_info.extend(spline_details)
    else:
        splines.clear()
        #spline_info.append("\n[无Spline实体]")

    return "\n".join(info + entity_stats + spline_info +details)