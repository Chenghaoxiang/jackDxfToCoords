import math

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline, interp1d
import matplotlib.pyplot as plt


def compute_angle(x,y):
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

# 输入控制点（示例格式）
ctrl_pts = np.array([
    [12.49840886, -10.00268878],
    [12.75579993, -9.48433996],
    [13.27140654,- 8.445981951],
    [13.81808722,- 6.792210359],
    [14.19149422,- 5.091549049],
    [14.37128495,- 3.356511632],
    [14.36059563,- 1.611260739],
    [14.15462647,0.152460827],
    [13.68618906,1.896914643],
    [12.92154011,3.568452595],
    [11.87849425,5.08340037],
    [10.60653113,6.383161781],
    [9.141371545,7.422061382],
    [7.514005672,8.201030036],
    [5.756225599,8.732255909],
    [3.920575439,9.072485984],
    [1.829321182,9.287927116],
    [-0.509885459,9.327336061],
    [-3.399744629,9.054320252],
    [-6.576892845,8.484965442],
    [-9.46989324,7.821976616],
    [-11.6893696,6.966688207],
    [-13.19384612,6.004386389],
    [-14.50246454,4.801172025],
    [-15.5188042,3.536985543],
    [-16.02356214,2.645695044],
    [-16.23734534,2.268201359],
    [-16.23734534, 2.268201359]
])

# 假设为3次开放均匀B样条
degree = 3
n = len(ctrl_pts)
knots = np.zeros(n + degree + 1)
knots[:degree+1] = 0
knots[-degree-1:] = n - degree  # 19
knots[degree+1:-degree-1] = np.arange(1, n - degree)

# 创建B样条
spline = BSpline(knots, ctrl_pts, degree)

# 弧长均匀采样
t_dense = np.linspace(knots[degree], knots[-degree-1], 1000)
points_dense = spline(t_dense)
diffs = np.diff(points_dense, axis=0)
distances = np.linalg.norm(diffs, axis=1)
cumulative_length = np.insert(np.cumsum(distances), 0, 0)
total_length = cumulative_length[-1]

s_positions = np.linspace(0, total_length, 90)
t_of_s = interp1d(cumulative_length, t_dense, kind='linear')(s_positions)
uniform_points = spline(t_of_s)

data = []
datas = []
for uniform_point in uniform_points:
    x  = uniform_point[0]
    y  = uniform_point[1]
    angle_deg, length = my_angle(x,y)
    data.append({
        "角度": angle_deg,
        "距离": length,
        "x坐标": x,
        "y坐标": y,
    })

datas = sorted(data,key=lambda item:item["角度"])
df = pd.DataFrame(datas)
df.to_excel("out_file/test2.xlsx", index=False)
print("数据已保存到 out_file/test2.xlsx")


# 输出结果
print(uniform_points)
# ------------------------------
# 4. 绘图
# ------------------------------
# plt.figure(figsize=(10, 6))
#
# # 绘制原始B样条曲线（密集点模拟曲线）
# plt.plot(points_dense[:, 0], points_dense[:, 1], 'b--', lw=1, alpha=0.5, label='B样条曲线')
#
# # 绘制控制点（红色圆点）
# plt.scatter(ctrl_pts[:, 0], ctrl_pts[:, 1], c='red', s=50, label='控制点', zorder=3)
#
# # 绘制均匀分布的90个点（绿色叉号）
# plt.scatter(uniform_points[:, 0], uniform_points[:, 1], c='green', marker='x', s=40, label='均匀采样点', zorder=2)
#
# # 添加图例和标签
# plt.legend()
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('B')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.axis('equal')  # 保持坐标轴比例一致
#
# # 显示图像
# plt.show()