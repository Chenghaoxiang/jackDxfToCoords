import math
import ezdxf
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline, interp1d
import matplotlib.pyplot as plt


def my_angle(x, y):
    length = math.hypot(x, y)
    angle_rad = math.atan2(y, x)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360.0
    return angle_deg, length


def spline_data(ctrl_pts, degree):
    # 生成B样条数据
    n = len(ctrl_pts)
    knots = np.zeros(n + degree + 1)
    knots[:degree + 1] = 0
    knots[-degree - 1:] = n - degree
    knots[degree + 1:-degree - 1] = np.arange(1, n - degree)

    # 创建B样条
    spline = BSpline(knots, ctrl_pts[:, :2], degree)  # 只取前两列（x,y）

    # 密集采样用于绘制曲线
    t_dense = np.linspace(knots[degree], knots[-degree - 1], 1000)
    points_dense = spline(t_dense)

    # 弧长均匀采样（测试点）
    diffs = np.diff(points_dense, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    cumulative_length = np.insert(np.cumsum(distances), 0, 0)
    total_length = cumulative_length[-1]

    s_positions = np.linspace(0, total_length, 90)
    t_of_s = interp1d(cumulative_length, t_dense, kind='linear')(s_positions)
    uniform_points = spline(t_of_s)

    return points_dense, uniform_points


# 读取DXF文件
doc = ezdxf.readfile("resource/B01A.dxf")
msp = doc.modelspace()
splines = msp.query('SPLINE')

# 初始化数据结构
data = []
control_points_list = []  # 存储所有控制点
spline_curves_list = []  # 存储所有样条曲线坐标
test_points_list = []  # 存储所有测试点

# 处理每个SPLINE
for spline in splines:
    # 获取控制点（可能包含z坐标）
    ctrl_pts = np.array(spline.control_points)
    control_points_list.append(ctrl_pts[:, :2])  # 只保留x,y坐标
    degree = spline.dxf.degree
    # 生成样条数据
    points_dense, uniform_points = spline_data(ctrl_pts, degree)
    spline_curves_list.append(points_dense)
    test_points_list.append(uniform_points)

    # 保存测试点数据到Excel
    for point in uniform_points:
        angle_deg, length = my_angle(point[0], point[1])
        data.append({
            "角度": angle_deg,
            "距离": length,
            "x坐标": point[0],
            "y坐标": point[1]
        })

# 数据排序保存
datas = sorted(data, key=lambda x: x["角度"])
pd.DataFrame(datas).to_excel("out_file/test2.xlsx", index=False)

# 绘制验证图形
plt.figure(figsize=(10, 6))
for idx, (ctrl_pts, curve, test_points) in enumerate(zip(
        control_points_list, spline_curves_list, test_points_list
)):
    # 绘制控制多边形
    plt.plot(ctrl_pts[:, 0], ctrl_pts[:, 1], 'ko--',
             markersize=5, linewidth=1, label='Control Polygon' if idx == 0 else "")

    # 绘制样条曲线
    plt.plot(curve[:, 0], curve[:, 1], 'k-',
             linewidth=2, label='Spline Curve' if idx == 0 else "")

    # 绘制测试点
    plt.scatter(test_points[:, 0], test_points[:, 1],
                color='red', s=15, label='Test Points' if idx == 0 else "")

# 图例和标签
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
plt.legend(unique_labels.values(), unique_labels.keys())

plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Spline Verification Plot")
plt.grid(True, linestyle='--', alpha=0.7)
plt.axis('equal')  # 保持坐标轴比例一致
plt.savefig("out_file/spline_verification.png", dpi=300)
plt.show()