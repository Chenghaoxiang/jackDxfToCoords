import math
from enum import Enum

import pandas as pd
import ezdxf

from shapeLogic import shape_circle, shape_rectangle, shape_ellipse, shape_line, shape_arc, shape_polyline, \
    shape_composite, shape_spline, shape_lwpolyline


class SHAPE(Enum):
    CIRCLE = 1          #圆形
    RECTANGLE = 2       #矩形
    ELLIPSE = 3         #椭圆
    LINE = 4            #线条
    ARC = 5             #弧形
    POLYLINE = 6        #多边形
    LWPOLYLINE = 7      #多段线
    SPLINE = 8
    composite = 9       #复合线段，不一定闭合，就是由line或者arc这些拼接来的
    POINT = 9           #点
    TEXT = 10
    DIMENSION = 11      #尺寸标注



def export_shape_polar_to_excel(shape,dxf_path, output_excel,min_val,max_val,pointNum):
    # 读取 DXF 文件
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # 查找第一个实体
    entity = None
    for entitys in msp:
        if entitys.dxftype() == shape:
            entity = entitys
            break
        elif shape == SHAPE.composite.name:
            entity = 1 #非空就行跳过下面的判断
            break

    if not entity:
        print("未找到对应的图形实体")
        for entitys in msp:
            print(entitys.dxftype())
            print(0-SHAPE.__members__.get(entitys.dxftype()).value)
            return 0-SHAPE.__members__.get(entitys.dxftype()).value
        return 0

    data = []

    match shape:
        case SHAPE.CIRCLE.name:
            shape_circle(data,entity)
        case SHAPE.RECTANGLE.name:
            shape_rectangle(data, entity)
        case SHAPE.ELLIPSE.name:
            shape_ellipse(data,entity)
        case SHAPE.LINE.name:
            shape_line(data,entity)
        case SHAPE.ARC.name:
            shape_arc(data, entity)
        case SHAPE.POLYLINE.name:
            shape_polyline(data, entity)
        case SHAPE.LWPOLYLINE.name:
            shape_lwpolyline(msp, data)
        case SHAPE.SPLINE.name:
            shape_spline(dxf_path,data)
        case SHAPE.composite.name:
            shape_composite(msp, data)

    # 创建 DataFrame 并保存为 Excel
    # df = pd.DataFrame(data)
    # df.to_excel(output_excel, index=False)
    # print(f"数据已保存到 {output_excel}")
    return 1

# 直接使用
if __name__ == '__main__':
    filepath = "D:/PythonProject/jackDxfToCoords/resource/rectangle.dxf"
    output_file = "out_file/out_RECTANGLE.xlsx"
    shape = SHAPE.RECTANGLE.name

    export_shape_polar_to_excel(shape, filepath, output_file)


