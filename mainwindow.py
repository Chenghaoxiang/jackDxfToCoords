import os
import re
import sys
from math import degrees
from pathlib import Path

import ezdxf
from PyQt5.QtCore import Qt, QSize, QPropertyAnimation, QEasingCurve, QUrl
from PyQt5.QtGui import (QIcon, QFont, QColor,
                         QPainter, QLinearGradient,
                         QImage, QPixmap, QDesktopServices)
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout,
                             QFormLayout, QHBoxLayout, QLabel,
                             QComboBox, QLineEdit, QPushButton,
                             QFileDialog, QGraphicsDropShadowEffect,
                             QSizePolicy, QMessageBox, QScrollArea)

from logicBeind import export_shape_polar_to_excel, SHAPE
from viewFile import analyze_dxf


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setFont(QFont("Segoe UI", 10))
        self.setWindowTitle("dxf文件极坐标生成工具")
        self.setWindowIcon(QIcon("_internal/resource/logo.svg"))
        self.resize(600, 400)
        self.initUI()
        self.setup_animations()

    def initUI(self):
        # 主布局设置
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(30)

        # 标题栏
        title = QLabel("智能文件导出系统")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-size: 26px;
                font-weight: 600;
                color: #2C3E50;
                padding-bottom: 12px;
                border-bottom: 3px solid #3498DB;
            }
        """)
        main_layout.addWidget(title)

        # 表单布局
        form_layout = QFormLayout()
        form_layout.setVerticalSpacing(25)
        form_layout.setLabelAlignment(Qt.AlignRight)
        form_layout.setFormAlignment(Qt.AlignHCenter | Qt.AlignTop)

        # 图形类别选择
        self.combo_graph = AnimatedComboBox()
        self.combo_graph.setIconSize(QSize(24, 24))
        self.btn_reset = GlowButton("重置", icon_path="_internal/resource/reset.svg")
        animation = QPropertyAnimation(self.btn_reset, b"geometry")
        animation.setDuration(300)  # 动画时长 300ms
        animation.setEasingCurve(QEasingCurve.OutQuad)  # 缓动效果
        # btn_excel.clicked.connect(lambda:)
        form_layout.addRow(self.create_file_row(FormLabel("图形类型："), self.combo_graph, self.btn_reset))

        self.edit_dxf = HighlightLineEdit()
        if getattr(sys, 'frozen', False):
            app_dir = Path(sys.executable).parent
            # pdf_path = f"{app_dir}/resource/1.pdf"
        else:
            app_dir = Path(__file__).parent

        # 添加第二个输入行: 取点个数和查看文件信息
        # self.numberOfDegrees = HighlightLineEdit()
        self.numberOfPoints = HighlightLineEdit()
        self.btn_viewFile = FloatButton(QIcon("_internal/resource/file_data.svg"), " 查看文件信息", self)
        self.btn_viewFile.clicked.connect(self.show_result)

        # 将水平布局添加到表单布局中
        form_layout.addRow(self.create_file_row(FormLabel("取点个数:"), self.numberOfPoints, self.btn_viewFile))

        # DXF文件选择
        self.edit_dxf = HighlightLineEdit()
        self.edit_dxf.setText(f"{app_dir}\\resource\\circle.dxf")
        btn_dxf = GlowButton("浏览", icon_path="_internal/resource/folder_icon.svg")
        animation = QPropertyAnimation(btn_dxf, b"geometry")
        animation.setDuration(300)  # 动画时长 300ms
        animation.setEasingCurve(QEasingCurve.OutQuad)  # 缓动效果
        btn_dxf.clicked.connect(lambda: self.select_file(self.edit_dxf, "打开DXF文件", "*.dxf"))
        form_layout.addRow(self.create_file_row(FormLabel("DXF位置："), self.edit_dxf, btn_dxf))

        # Excel导出位置
        self.edit_excel = HighlightLineEdit()
        self.edit_excel.setText(f"{app_dir}\\out_file")
        btn_excel = GlowButton("浏览", icon_path="_internal/resource/folder_icon.svg")
        btn_excel.clicked.connect(lambda: self.save_directory(self.edit_excel))
        form_layout.addRow(self.create_file_row(FormLabel("导出位置："), self.edit_excel, btn_excel))

        main_layout.addLayout(form_layout)

        # 生成按钮
        self.btn_generate = FloatButton(QIcon("_internal/resource/rocket_icon.svg"), " 开始生成", self)
        self.btn_generate.clicked.connect(self.generate)
        main_layout.addWidget(self.btn_generate, 0, Qt.AlignCenter)

        self.setLayout(main_layout)
        self.apply_shadows()

    def create_file_row(self, label, edit, btn):
        container = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)
        layout.addWidget(label)
        layout.addWidget(edit)
        layout.addWidget(btn)
        container.setLayout(layout)
        return container

    def apply_shadows(self):
        def add_shadow(widget, radius=12, offset=4):
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(radius)
            shadow.setColor(QColor(0, 0, 0, 25))
            shadow.setOffset(offset, offset)
            widget.setGraphicsEffect(shadow)

        add_shadow(self.combo_graph)
        add_shadow(self.btn_generate, 21, 6)
        add_shadow(self.edit_dxf)
        add_shadow(self.edit_excel)

    def setup_animations(self):
        # 窗口入场动画
        self.animation = QPropertyAnimation(self, b"windowOpacity")
        self.animation.setDuration(800)
        self.animation.setStartValue(0)
        self.animation.setEndValue(1)
        self.animation.start()

    def select_file(self, edit, title, file_type):
        path, _ = QFileDialog.getOpenFileName(self, title, "", f"Files ({file_type})")
        if path:
            edit.setText(path)

    def save_directory(self, line_edit):
        directory = QFileDialog.getExistingDirectory(
            self, "选择保存目录", "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if directory:
            line_edit.setText(directory)

    def generate(self):
        try:
            # id是下拉框选择对象的序列号+1，变成枚举类型SHAPE的value
            id = self.combo_graph.currentIndex() + 1
            shape = next((name for name, member in SHAPE.__members__.items() if member.value == id), None)

            filepath = self.edit_dxf.text()
            out_dir = self.edit_excel.text()
            output_file = out_dir + f"\\{shape}.xlsx"
            print(output_file)

            degreesName = self.numberOfDegrees.text()
            pointName = self.numberOfPoints.text()

            input_str = degreesName
            match = re.match(r'^(\d+)-(\d+)$', str(input_str).strip())

            if match:
                min_val = int(match.group(1))  # 0
                max_val = int(match.group(2))  # 180
                if min_val > max_val or min_val == max_val:
                    QMessageBox.critical(
                        self,
                        f'{min_val} > {max_val} 或 {min_val} == {max_val}',
                        '错误信息：请正确填写输入角度范围\n正确填写示例: "0-180"',
                        QMessageBox.Cancel
                    )
            else:
                QMessageBox.critical(
                    self,
                    '正则提取失败',
                    '错误信息：请正确填写输入角度\n正确填写示例: "0-180"',
                    QMessageBox.Cancel
                )
                return 0

            pointNum = int(pointName.strip())

            print(degreesName)
            print(pointName)

            returnId = export_shape_polar_to_excel(shape, filepath, output_file,min_val,max_val,pointNum)

            if returnId < 0:
                ShapeErrorid = -returnId;
                ShapeErrorName = SHAPE(ShapeErrorid);
                QMessageBox.critical(
                    self,
                    '操作失败',
                    f'并不是选择的{shape}图形，而是{ShapeErrorName}图形\n请选择{ShapeErrorName}图形,如果图形复杂请选择复合线段',
                    QMessageBox.Ok
                )

            if returnId == 1:
                reply = QMessageBox.information(
                    self,
                    '操作成功',
                    f'文件已成功生成！\n文件位置:{filepath}',
                    QMessageBox.Open,
                    QMessageBox.Close
                )
                if reply == QMessageBox.Open:
                    QDesktopServices.openUrl(QUrl.fromLocalFile(out_dir))
        except Exception as e:
            print(e)
            # 生成失败提示
            QMessageBox.critical(
                self,
                '操作失败',
                f'错误信息：{str(e)}',
                QMessageBox.Ok
            )

    def show_result(self):
        """按钮点击处理函数"""
        filepath = self.edit_dxf.text()
        result = analyze_dxf(filepath)

        msg = QMessageBox(self)
        msg.setWindowTitle("DXF分析结果")
        # 创建滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)  # 允许内容自适应大小

        # 创建内部标签并设置长文本
        content = QLabel(result)
        content.setWordWrap(True)  # 启用自动换行
        scroll.setWidget(content)

        # 设置滚动区域的最小尺寸（避免窗口过小）
        scroll.setMinimumSize(400, 300)  # 可根据需求调整

        # 将滚动区域添加到QMessageBox
        msg.layout().addWidget(scroll, 0, 1)  # 替换原QLabel位置
        msg.exec_()

# 自定义控件类
class FormLabel(QLabel):
    def __init__(self, text):
        super().__init__(text)
        self.setStyleSheet("""
            QLabel {
                color: #7F8C8D;
                font-size: 14px;
                padding-right: 20px;
                min-width: 120px;
            }
        """)
        self.setAlignment(Qt.AlignRight | Qt.AlignVCenter)


class HighlightLineEdit(QLineEdit):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(42)
        self.setStyleSheet("""
            QLineEdit {
                background: #FFFFFF;
                border: 2px solid #E0E6ED;
                border-radius: 8px;
                padding: 12px 15px;
                font-size: 14px;
                selection-background-color: #3498DB50;
            }
            QLineEdit:focus {
                border-color: #3498DB;
                box-shadow: 0 2px 12px rgba(52, 152, 219, 0.15);
            }
        """)


class GlowButton(QPushButton):
    def __init__(self, text, icon_path=None):
        super().__init__(text)
        self.setMinimumSize(90, 40)
        if icon_path:
            self.setIcon(QIcon(icon_path))
            self.setIconSize(QSize(18, 18))
        self.setStyleSheet("""
            QPushButton {
                background: #3498DB;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                transition: all 0.3s;
            }
            QPushButton:hover {
                background: #2980B9;
                box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
            }
            QPushButton:pressed {
                transform: translateY(2px);
            }
        """)


class FloatButton(QPushButton):
    def __init__(self, icon, text, parent):
        super().__init__(icon, text, parent)
        self.setIconSize(QSize(24, 24))
        self.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3498DB, stop:1 #38A3A5);
                color: white;
                border-radius: 10px;
                padding: 15px 35px;
                font-size: 16px;
                font-weight: 500;
            }
        """)
        self.setup_animation()

    def setup_animation(self):
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(1000)
        self.animation.setEasingCurve(QEasingCurve.OutInQuad)
        self.hover_animation = QPropertyAnimation(self, b"geometry")
        self.hover_animation.setDuration(300)

    def enterEvent(self, event):
        self.hover_animation.stop()
        self.hover_animation.setStartValue(self.geometry())
        self.hover_animation.setEndValue(self.geometry().adjusted(-4, -4, 4, 4))
        self.hover_animation.start()

    def leaveEvent(self, event):
        self.hover_animation.stop()
        self.hover_animation.setStartValue(self.geometry())
        self.hover_animation.setEndValue(self.geometry().adjusted(4, 4, -4, -4))
        self.hover_animation.start()


class AnimatedComboBox(QComboBox):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(45)
        self.setStyleSheet(
            r"""
            QComboBox {
                background: white;
                border: 2px solid #E0E6ED;
                border-radius: 8px;
                padding: 10px 15px;
                font-size: 14px;
                min-width: 200px;
                qproperty-iconSize: 20px;
            }
            QComboBox::drop-down {
                width: 35px;
                border-left: 2px solid #E0E6ED;
                subcontrol-origin: padding;
                subcontrol-position: right center;
            }

            QComboBox::down-arrow {
                image: url("_internal/resource/arrow_down.svg");
                width: 16px;
                height: 16px;
            }

            QComboBox::down-arrow:open {
                image: url("_internal/resource/arrow_up.svg");
            }

            QComboBox:hover {
                border-color: #3498DB;
            }
            QComboBox QAbstractItemView {
                border: 2px solid #E0E6ED;
                selection-background-color: #3498DB;
                padding: 8px;
                outline: 0px;  /* 移除焦点虚线框 */
            }
        """)

        # 添加下拉项图标（示例）
        self.addItem(QIcon("_internal/resource/circle.svg"), "圆形(CIRCLE)")
        self.addItem(QIcon("_internal/resource/rectangle.svg"), "矩形(RECTANGLE)")
        self.addItem(QIcon("_internal/resource/ellipse.svg"), "椭圆(ELLIPSE)")

        self.addItem(QIcon("_internal/resource/line_icon.svg"), "线条(LINE)")
        self.addItem(QIcon("_internal/resource/arc_icon.svg"), "圆弧(ARC)")
        self.addItem(QIcon("_internal/resource/polyline_icon.svg"), "多边形(POLYLINE)")
        self.addItem(QIcon("_internal/resource/lwpolyline_icon.svg"), "多段线(LWPOLYLINE)")
        self.addItem(QIcon("_internal/resource/spline_icon.svg"), "样条线-光滑曲线(SPLINE)")
        self.addItem(QIcon("_internal/resource/composite_icon.svg"), "复合线段(composite)")


if __name__ == '__main__':
    app = QApplication([])
    window = MyWindow()
    window.show()
    app.exec_()