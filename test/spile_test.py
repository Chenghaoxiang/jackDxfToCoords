import math

import ezdxf
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline, interp1d
import matplotlib.pyplot as plt


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

def shape_spline(dxf_path):
    # 输入控制点（示例格式）
    # doc = ezdxf.readfile("resource/B01A.dxf")
    # print(dxf_path)
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    splines = msp.query('SPLINE')

    data = []
    datas = []

    # 遍历每个SPLINE并处理控制点
    control_points_list = []
    for spline in splines:
        # 获取控制点列表，每个点为(x, y, z)元组
        points = spline.control_points
        # 转换为NumPy数组
        ctrl_pts = np.array(points)

        degree = spline.dxf.degree

        uniform_points = spline_data(ctrl_pts,degree)
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
    df = pd.DataFrame(data)
    df.to_excel("../out_file/test2.xlsx", index=False)
    print("数据已保存到 ../out_file/test2.xlsx")


    # 输出结果
    print(datas)

if __name__ == '__main__':
    shape_spline("../resource/B01A.dxf")

