# -*- coding: utf-8 -*-
# author: Ye Yang
# Data Analysis GUI for MT
# 指数计算器可以不选择文件直接运行，最后存储的时候再选择位置

import os
import sys
from pathlib import Path

from PySide6.QtCore import QFileInfo
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QMainWindow,
                               QPushButton, QSizePolicy, QStatusBar, QVBoxLayout, QWidget, QFileDialog, QLabel, QLineEdit, QMessageBox)

import MtFe
import MtFc
import MtDy
import ExponentialCalculator  # 添加新模块导入

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("MT Data Analysis")

        self.centralwidget = QWidget()
        self.layoutcentralwiget = QVBoxLayout(self.centralwidget)

        self.functionswidget = QWidget(self.centralwidget)
        self.layoutfunctionswidget = QHBoxLayout(self.functionswidget)

        self.functionselectwidget = QWidget(self.functionswidget)
        self.layoutfunctionsselectwidget = QHBoxLayout(self.functionselectwidget)

        self.functionselectable = QLabel(self.functionselectwidget, text="Function:")
        self.functionselectable.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.layoutfunctionsselectwidget.addWidget(self.functionselectable)

        # 在init方法中修改
        self.functionselectgroup = QComboBox(self.functionselectwidget)
        self.functionselectgroup.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.functionselectgroup.addItems(["Force-Extension Analysis", "Force Calibration", 
                                        "Kinetics Analysis", "Exponential Calculator"])  # 添加新选项
        self.layoutfunctionsselectwidget.addWidget(self.functionselectgroup)

        self.layoutfunctionswidget.addWidget(self.functionselectwidget)
        self.layoutcentralwiget.addWidget(self.functionswidget)

        self.fileselectwidget = QWidget(self.centralwidget)
        self.horizontalLayout = QHBoxLayout(self.fileselectwidget)

        self.file_path_name = QLineEdit(self.fileselectwidget, text="Select Files or Input File Path")
        self.file_path_name.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.horizontalLayout.addWidget(self.file_path_name)

        self.select_files = QPushButton(self.fileselectwidget, text="Select Files")
        self.select_files.clicked.connect(self.on_read_files_button_clicked)
        self.horizontalLayout.addWidget(self.select_files)

        self.layoutcentralwiget.addWidget(self.fileselectwidget)

        self.runwidget = QWidget(self.centralwidget)
        self.layoutrunwidget = QHBoxLayout(self.runwidget)

        self.runbutton = QPushButton(self.runwidget, text="Run")
        self.runbutton.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.runbutton.clicked.connect(self.on_run_button_clicked)
        self.layoutrunwidget.addWidget(self.runbutton)

        self.layoutcentralwiget.addWidget(self.runwidget)

        self.setCentralWidget(self.centralwidget)

        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("By: Ye, Y(2023).")

        self.last_directory = "/"

    def on_read_files_button_clicked(self):
        self.file_name, self.file_type = QFileDialog.getOpenFileName(None, "Select one file", self.last_directory, "TDMS Files (*.tdms)")
        if self.file_name:
            self.file_path_name.setText(self.file_name)
            self.last_directory = os.path.dirname(self.file_name)

    def get_file_info(self):
        self.file_name = self.file_path_name.text()
        self.file_info = QFileInfo(self.file_name)
        self.Data_Saved_Path = self.file_info.absolutePath()
        self.base_name = Path(self.file_name).stem
        return {'file_name': self.file_name, 'file_type': self.file_type, 'Data_Saved_Path': self.Data_Saved_Path, 'base_name': self.base_name, 'self.file_info': self.file_info}

    def Force_Extension_Analysis(self):
        data_for_figure = self.get_file_info()
        data_for_figure['calculate_force'] = True  # 需要计算力
        self.FE_Analysis_Window = MtFe.FigureView(data_for_figure)
        self.FE_Analysis_Window.show()

    def force_calibration(self):
        data_for_figure = self.get_file_info()
        data_for_figure['calculate_force'] = False  # 不需要计算力
        self.Calibration_Window = MtFc.ForceCalibration(data_for_figure)
        self.Calibration_Window.show()

    def knetics_analysis(self):
        data_for_figure = self.get_file_info()
        data_for_figure['calculate_force'] = True  # 需要计算力
        self.Kinetic_Window = MtDy.KineticsAnalysis(data_for_figure)
        self.Kinetic_Window.show()

    def exponential_calculator(self):
        # 创建默认数据字典，提供默认保存路径
        data_for_calculator = {
            'Data_Saved_Path': os.path.expanduser("~/Documents"),  # 默认保存到用户文档目录
            'file_name': "",
            'base_name': "exponential_calculation",
            'file_type': "",
            'self.file_info': None
        }
        
        # 如果已有文件信息，尝试使用它，但不强制要求文件必须存在
        file_path = self.file_path_name.text()
        if file_path and file_path != "Select Files or Input File Path":
            try:
                file_info = self.get_file_info()
                if os.path.exists(file_info.get('file_name', '')):
                    data_for_calculator = file_info
                elif os.path.isdir(os.path.dirname(file_path)):
                    # 如果至少目录存在，可以使用这个目录
                    data_for_calculator['Data_Saved_Path'] = os.path.dirname(file_path)
            except:
                # 如果获取文件信息失败，继续使用默认值
                pass
        
        self.Exp_Calc_Window = ExponentialCalculator.ExponentialCalculator(data_for_calculator)
        self.Exp_Calc_Window.show()

    def on_run_button_clicked(self):
        current_function = self.functionselectgroup.currentText()
        
        # 为指数计算器提供特殊处理
        if current_function == "Exponential Calculator":
            self.exponential_calculator()
            return
            
        # 其他功能仍然需要文件检查
        if not os.path.exists(self.file_path_name.text()):
            QMessageBox.warning(self, "Warning", "Please select a valid file!", QMessageBox.Ok, QMessageBox.Ok)
            return

        functions = {
            "Force-Extension Analysis": self.Force_Extension_Analysis,
            "Force Calibration": self.force_calibration,
            "Kinetics Analysis": self.knetics_analysis,
        }

        try:
            functions[current_function]()
        except KeyError:
            QMessageBox.warning(self, "Warning", "Please select a valid function!", QMessageBox.Ok, QMessageBox.Ok)

            
if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())
