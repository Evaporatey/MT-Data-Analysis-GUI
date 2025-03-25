# author: Ye Yang
# MT Dynamics Analysis with HMM
# unicode: utf-8

# 标准库导入
import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# 第三方库导入
import matplotlib.pyplot as plt
import nptdms as nt
import numpy as np
import openpyxl
import pandas as pd
import scipy
from hmmlearn import hmm
from scipy.signal import find_peaks, peak_prominences

# PySide6导入
from PySide6.QtCore import QRect, QFileInfo, Qt
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, 
    QMenu, QMenuBar, QMessageBox, QPushButton, QRadioButton, QSizePolicy, 
    QSpinBox, QStatusBar, QVBoxLayout, QWidget, QTabWidget, QFormLayout
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

# 本地模块导入
from tdms_reader import read_tdms_chunk, read_tdms_file


class KineticsAnalysis(QWidget):
    def __init__(self, data_for_figure):
        super().__init__()
        
        # 初始化数据属性
        self._initialize_data_attributes()
        
        # 继承传递的数据
        self._load_inherited_data(data_for_figure)
        
        # 加载TDMS数据
        self._load_tdms_data()
        
        # 创建UI界面
        self._setup_ui()
        
        # 创建图形对象
        self._setup_figure()
        
        # 首次绘图
        self.plotfig()
    
    def _initialize_data_attributes(self):
        """初始化所有数据属性"""
        self.bead_data_list = None
        self.time_data = None
        self.time_data_list = None
        self.time_point_end = None
        self.time_point_end_data = None
        self.time_point_start_data = None
        self.time_point_start = None
        self.bead_data = None
        self.base_line = None
        self.bead_array = None
        self.bin_data = None
        self.bin_data_diff = None
        self.extension_start_point = None
        self.extension_end_point = None
        self.time_start = None
        self.time_end = None
        self.above_time_segment = None
        self.above_time_data = []
        self.below_time_segment = None
        self.below_time_data = []
        self.current_data_segment = None
        self.force_regions = []
        self.region_highlight_rectangle = None
        self.dynamic_annotations = []
        self.min_region_duration = 1.0  # 默认最小区域持续时间（秒）
    
    def _load_inherited_data(self, data_for_figure):
        """加载从父窗口传递的数据"""
        self.Data_Saved_Path = data_for_figure['Data_Saved_Path']
        self.file_name = data_for_figure['file_name']
        self.file_type = data_for_figure['file_type']
        self.file_info = data_for_figure['self.file_info']
        self.base_name = data_for_figure['base_name']
    
    def _load_tdms_data(self):
        """加载并处理TDMS数据"""
        # 获取TDMS数据并转换为DataFrame
        self.tdms_data_frame = read_tdms_file(self.file_name, need_force=True)
        self.tdms_data_store = self.tdms_data_frame
        
        # 获取磁铁移动状态数据
        beads_list = self.tdms_data_frame.columns.values.tolist()
        str_magnet_move = str(beads_list[-1])
        magnet_move_state = self.tdms_data_store[str(str_magnet_move)].values.tolist()
        
        # 处理磁铁移动数据
        new_magnet_move_state = [i for i in magnet_move_state if not math.isnan(i)]
        self.new_int_magnet_move_state = [int(i) for i in new_magnet_move_state]
        self.num_of_state = len(self.new_int_magnet_move_state) / 2
        self.final_slice_magnet_move_state = [
            self.new_int_magnet_move_state[i:i+2] 
            for i in range(0, len(self.new_int_magnet_move_state), 2)
        ]
        self.num_of_sliced_data = len(self.final_slice_magnet_move_state)
        self.beads_list = beads_list
    
    def _create_widget(self, parent, objectName="", sizePolicy=None):
        """创建标准widget的辅助方法"""
        widget = QWidget(parent)
        if objectName:
            widget.setObjectName(objectName)
        if sizePolicy:
            widget.setSizePolicy(*sizePolicy)
        return widget
    
    def _create_layout(self, parent, layout_type=QHBoxLayout, objectName="", margins=None):
        """创建布局的辅助方法"""
        layout = layout_type(parent)
        if objectName:
            layout.setObjectName(objectName)
        if margins is not None:
            layout.setContentsMargins(*margins)
        return layout
    
    def _create_label(self, parent, text="", objectName="", fixed_size=True):
        """创建标签的辅助方法"""
        label = QLabel(parent)
        if objectName:
            label.setObjectName(objectName)
        label.setText(text)
        if fixed_size:
            label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        return label
    
    def _create_combobox(self, parent, items=None, objectName="", fixed_size=True, 
                         enabled=True, connection=None):
        """创建下拉选择框的辅助方法"""
        combobox = QComboBox(parent)
        if objectName:
            combobox.setObjectName(objectName)
        if fixed_size:
            combobox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        if items:
            combobox.addItems(items)
        combobox.setEnabled(enabled)
        if connection:
            combobox.currentTextChanged.connect(connection)
        return combobox
    
    def _create_checkbox(self, parent, text="", objectName="", checked=False, 
                         fixed_size=True, connection=None):
        """创建复选框的辅助方法"""
        checkbox = QCheckBox(parent)
        if objectName:
            checkbox.setObjectName(objectName)
        checkbox.setText(text)
        checkbox.setChecked(checked)
        if fixed_size:
            checkbox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        if connection:
            checkbox.stateChanged.connect(connection)
        return checkbox
    
    def _create_button(self, parent, text="", objectName="", fixed_size=True, connection=None):
        """创建按钮的辅助方法"""
        button = QPushButton(parent)
        if objectName:
            button.setObjectName(objectName)
        button.setText(text)
        if fixed_size:
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        if connection:
            button.clicked.connect(connection)
        return button
    
    def _setup_ui(self):
        """设置UI界面"""
        # 设置窗口属性
        self.setWindowTitle("MT Dynamics Analysis")
        self.resize(1600, 900)  # 更合理的窗口大小
        
        # 创建主界面布局为水平布局
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        
        # 创建左侧图形面板（可伸缩）
        self.figure_panel = QWidget()
        self.figure_layout = QVBoxLayout(self.figure_panel)
        self.figure_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建右侧控制面板（宽度固定）
        self.control_panel = QWidget()
        self.control_panel.setFixedWidth(300)  # 控制面板宽度固定
        self.control_layout = QVBoxLayout(self.control_panel)
        
        # 创建标签页控件
        self.tab_widget = QTabWidget()
        
        # 创建各个功能标签页
        self._create_basic_tab()      # 基本控制标签页
        self._create_analysis_tab()   # 分析控制标签页
        self._create_advanced_tab()   # 高级控制标签页
        
        # 添加标签页控件到控制面板
        self.control_layout.addWidget(self.tab_widget)
        
        # 添加数据信息显示区域
        self._create_data_info_panel()
        
        # 将面板添加到主布局
        self.main_layout.addWidget(self.figure_panel, 3)  # 图形面板占据更多空间
        self.main_layout.addWidget(self.control_panel, 1)
    
    def _create_basic_tab(self):
        """创建基本控制标签页"""
        basic_tab = QWidget()
        basic_layout = QVBoxLayout(basic_tab)
        
        # 添加数据选择组
        data_group = QGroupBox("Data Selection")
        data_layout = QVBoxLayout(data_group)
        
        # Y轴选择
        y_layout = QHBoxLayout()
        y_label = QLabel("Y Axis (Bead):")
        self.y_axis_box = QComboBox()
        y_items = [str(self.beads_list[i]) for i in range(4, (len(self.beads_list) - 1))]
        self.y_axis_box.addItems(y_items)
        self.y_axis_box.currentTextChanged.connect(self.plotfig)
        y_layout.addWidget(y_label)
        y_layout.addWidget(self.y_axis_box)
        data_layout.addLayout(y_layout)
        
        # 滤波器选择
        filter_layout = QHBoxLayout()
        filter_label = QLabel("Filter Type:")
        filter_items = ["Moving Average", "Median Filter"]
        self.filterselection = QComboBox()
        self.filterselection.addItems(filter_items)
        self.filterselection.currentTextChanged.connect(self.plotfig)
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.filterselection)
        data_layout.addLayout(filter_layout)
        
        # 数据分段
        slice_layout = QHBoxLayout()
        self.sliced_data_box = QCheckBox("Segment Data")
        self.sliced_data_box.stateChanged.connect(self.plotfig)
        slice_label = QLabel("Select Segment:")
        self.chose_sliced_data_box = QComboBox()
        slice_items = [str(i) for i in range(1, self.num_of_sliced_data + 1)]
        self.chose_sliced_data_box.addItems(slice_items)
        self.chose_sliced_data_box.currentTextChanged.connect(self.plotfig)
        self.chose_sliced_data_box.setEnabled(False)  # 默认禁用
        slice_layout.addWidget(self.sliced_data_box)
        slice_layout.addWidget(slice_label)
        slice_layout.addWidget(self.chose_sliced_data_box)
        data_layout.addLayout(slice_layout)
        
        basic_layout.addWidget(data_group)
        
        # 添加滤波设置组
        filter_group = QGroupBox("Filter Settings")
        filter_layout = QVBoxLayout(filter_group)
        
        # 启用滤波
        self.check_fitted_data_box = QCheckBox("Use Filtered Data")
        self.check_fitted_data_box.setChecked(True)
        self.check_fitted_data_box.stateChanged.connect(self.plotfig)
        filter_layout.addWidget(self.check_fitted_data_box)
        
        # 内核大小
        kernel_layout = QHBoxLayout()
        kernel_label = QLabel("Filter Kernel Size:")
        self.kernel_size_box = QSpinBox()
        self.kernel_size_box.setRange(1, 1000)
        self.kernel_size_box.setValue(51)
        self.kernel_size_box.setSingleStep(2)
        self.kernel_size_box.valueChanged.connect(self.plotfig)
        kernel_layout.addWidget(kernel_label)
        kernel_layout.addWidget(self.kernel_size_box)
        filter_layout.addLayout(kernel_layout)
        
        basic_layout.addWidget(filter_group)
        
        # 添加最小区域持续时间设置
        region_group = QGroupBox("Region Settings")
        region_layout = QVBoxLayout(region_group)
        
        duration_layout = QHBoxLayout()
        duration_label = QLabel("Min Region Duration (sec):")
        self.min_duration_spin = QSpinBox()
        self.min_duration_spin.setRange(0, 60)
        self.min_duration_spin.setValue(1)
        self.min_duration_spin.setSingleStep(1)
        self.min_duration_spin.valueChanged.connect(self.on_min_duration_changed)
        duration_layout.addWidget(duration_label)
        duration_layout.addWidget(self.min_duration_spin)
        region_layout.addLayout(duration_layout)
        
        basic_layout.addWidget(region_group)
        
        # 添加基本数据操作按钮
        button_group = QGroupBox("Data Operations")
        button_layout = QVBoxLayout(button_group)
        
        # 保存选中数据按钮
        self.SaveSelectedDataButton = QPushButton("Save Selected Data")
        self.SaveSelectedDataButton.clicked.connect(self.save_selected_data)
        button_layout.addWidget(self.SaveSelectedDataButton)
        
        basic_layout.addWidget(button_group)
        basic_layout.addStretch(1)  # 添加弹性空间
        
        self.tab_widget.addTab(basic_tab, "Basic")
    
    def _create_analysis_tab(self):
        """创建分析控制标签页"""
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)
        
        # 添加区域选择组
        region_group = QGroupBox("Force Region Selection")
        region_layout = QVBoxLayout(region_group)
        
        region_select_layout = QHBoxLayout()
        region_label = QLabel("Select Region:")
        self.region_selector = QComboBox()
        self.region_selector.currentTextChanged.connect(self.select_force_region)
        region_select_layout.addWidget(region_label)
        region_select_layout.addWidget(self.region_selector)
        region_layout.addLayout(region_select_layout)
        
        analysis_layout.addWidget(region_group)
        
        # 添加动力学预分析组
        dynamics_group = QGroupBox("Dynamics Pre-Analysis")
        dynamics_layout = QVBoxLayout(dynamics_group)
        
        # 敏感度设置
        sensitivity_layout = QHBoxLayout()
        sensitivity_label = QLabel("Detection Sensitivity:")
        self.sensitivity_combo = QComboBox()
        self.sensitivity_combo.addItems(["Low", "Medium", "High", "Very High"])
        self.sensitivity_combo.setCurrentIndex(1)  # 默认选中中
        sensitivity_layout.addWidget(sensitivity_label)
        sensitivity_layout.addWidget(self.sensitivity_combo)
        dynamics_layout.addLayout(sensitivity_layout)
        
        # 预分析按钮
        self.preanalyze_button = QPushButton("Pre-Analyze Dynamics")
        self.preanalyze_button.clicked.connect(self.preanalyze_dynamics)
        dynamics_layout.addWidget(self.preanalyze_button)
        
        analysis_layout.addWidget(dynamics_group)
        
        # 添加动力学分析组
        kinetics_group = QGroupBox("Kinetics Analysis")
        kinetics_layout = QVBoxLayout(kinetics_group)
        
        # 动力学分析按钮
        self.kinetics_analysis_calculation_button = QPushButton("HMM Kinetics Analysis")
        self.kinetics_analysis_calculation_button.clicked.connect(self.kinetics_analysis_calculation)
        kinetics_layout.addWidget(self.kinetics_analysis_calculation_button)
        
        analysis_layout.addWidget(kinetics_group)
        
        # 添加数据保存组
        save_group = QGroupBox("Data Saving")
        save_layout = QVBoxLayout(save_group)
        
        # 保存时间数据按钮
        self.save_time_data_button = QPushButton("Save Time Data")
        self.save_time_data_button.clicked.connect(self.save_time_data)
        save_layout.addWidget(self.save_time_data_button)
        
        analysis_layout.addWidget(save_group)
        
        analysis_layout.addStretch(1)  # 添加弹性空间
        
        self.tab_widget.addTab(analysis_tab, "Analysis")
    
    def _create_advanced_tab(self):
        """创建高级控制标签页"""
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout(advanced_tab)
        
        # 这里可以添加更多高级设置
        # 目前暂时保留为空，以便后续扩展
        
        advanced_layout.addStretch(1)  # 添加弹性空间
        
        self.tab_widget.addTab(advanced_tab, "Advanced")
    
    def _create_data_info_panel(self):
        """创建数据信息显示面板"""
        info_group = QGroupBox("Data Information")
        info_layout = QFormLayout(info_group)
        
        # T1 (时间点1)
        self.force_data_info = QLabel("0")
        info_layout.addRow("T1:", self.force_data_info)
        
        # T2 (时间点2)
        self.extension_data_info = QLabel("0")
        info_layout.addRow("T2:", self.extension_data_info)
        
        # 力数据
        self.force_data_display = QLabel("0")
        info_layout.addRow("Force (pN):", self.force_data_display)
        
        # 添加到控制面板
        self.control_layout.addWidget(info_group)
    
    def on_min_duration_changed(self, value):
        """当最小区域持续时间改变时重新绘图"""
        self.min_region_duration = value
        self.plotfig()
    
    def _setup_figure(self):
        """设置matplotlib图形和事件处理"""
        self.fig = plt.figure(figsize=(10, 8))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        # 添加到图形面板
        self.figure_layout.addWidget(self.canvas, 1)
        self.figure_layout.addWidget(self.toolbar, 0)
        
        # 设置事件处理
        self._setup_wheel_events()
        self._setup_zoom_events()
        self._setup_pan_events()
        self._setup_point_selection_events()
    
    def _setup_wheel_events(self):
        """设置滚轮事件处理"""
        def wheelEvent(event):
            if self.y_axis_box.underMouse():
                if event.angleDelta().y() > 0:
                    self.y_axis_box.setCurrentIndex((self.y_axis_box.currentIndex() - 1) % self.y_axis_box.count())
                else:
                    self.y_axis_box.setCurrentIndex((self.y_axis_box.currentIndex() + 1) % self.y_axis_box.count())
            elif self.chose_sliced_data_box.underMouse():
                if event.angleDelta().y() > 0:
                    self.chose_sliced_data_box.setCurrentIndex(
                        (self.chose_sliced_data_box.currentIndex() - 1) % self.chose_sliced_data_box.count())
                else:
                    self.chose_sliced_data_box.setCurrentIndex(
                        (self.chose_sliced_data_box.currentIndex() + 1) % self.chose_sliced_data_box.count())
            elif self.kernel_size_box.underMouse():
                if event.angleDelta().y() > 0:
                    self.kernel_size_box.setValue(self.kernel_size_box.value() - 2)
                else:
                    self.kernel_size_box.setValue(self.kernel_size_box.value() + 2)
            else:
                super(KineticsAnalysis, self).wheelEvent(event)

        self.y_axis_box.wheelEvent = wheelEvent
        self.chose_sliced_data_box.wheelEvent = wheelEvent
    
    def _setup_zoom_events(self):
        """设置缩放事件处理"""
        def zoom_event(event):
            if event.inaxes is None:
                return
                
            axtemp = event.inaxes
            x_min, x_max = axtemp.get_xlim()
            y_min, y_max = axtemp.get_ylim()
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_zoom = x_range / 10
            y_zoom = y_range / 10

            if event.button == 'up':
                axtemp.set(xlim=(x_min + x_zoom, x_max - x_zoom),
                          ylim=(y_min + y_zoom, y_max - y_zoom))
            elif event.button == 'down':
                axtemp.set(xlim=(x_min - x_zoom, x_max + x_zoom),
                          ylim=(y_min - y_zoom, y_max + y_zoom))
                
            # 更新显示的力值标签位置
            self._update_dynamic_force_labels()
            self.canvas.draw_idle()

        self.canvas.mpl_connect('scroll_event', zoom_event)
    
    def _setup_pan_events(self):
        """设置平移事件处理"""
        self.lastx = 0
        self.lasty = 0
        self.press = False

        def on_press(event):
            if event.inaxes is not None and event.button == 3:
                self.lastx = event.xdata
                self.lasty = event.ydata
                self.press = True

        def on_move(event):
            if not self.press or event.inaxes is None:
                return
                
            axtemp = event.inaxes
            x = event.xdata - self.lastx
            y = event.ydata - self.lasty

            x_min, x_max = axtemp.get_xlim()
            y_min, y_max = axtemp.get_ylim()

            axtemp.set(xlim=(x_min - x, x_max - x), 
                      ylim=(y_min - y, y_max - y))
                      
            # 更新显示的力值标签位置
            self._update_dynamic_force_labels()
            self.canvas.draw_idle()

        def on_release(event):
            self.press = False

        self.canvas.mpl_connect('button_press_event', on_press)
        self.canvas.mpl_connect('button_release_event', on_release)
        self.canvas.mpl_connect('motion_notify_event', on_move)
    
    def _setup_point_selection_events(self):
        """设置点选择事件处理"""
        self.points_for_save = []

        def get_point(event):
            if event.inaxes is None or event.button != 2:
                return
                
            # 如果已有两个点，清空重新开始
            if len(self.points_for_save) == 2:
                self.points_for_save.clear()
                self.force_data_info.setText("0")
                self.extension_data_info.setText("0")
                self.force_data_display.setText("0")

            # 保存第一个点
            if len(self.points_for_save) == 0:
                self.x1 = event.xdata
                self.y1 = event.ydata
                self.points_for_save.append((self.x1, self.y1))
                self.force_data_info.setText(f"{self.x1:.4f}")
            # 保存第二个点并更新数据
            else:
                self.x2 = event.xdata
                self.y2 = event.ydata
                self.points_for_save.append((self.x2, self.y2))
                self.extension_data_info.setText(f"{self.x2:.4f}")
                
                # 更新时间点和力值显示
                self._update_time_points()
                self._update_force_display()
                
                # 在图表上标记选择的区域
                if hasattr(self, 'fig') and hasattr(self.fig, 'axes') and len(self.fig.axes) > 0:
                    ax = self.fig.axes[0]
                    ymin, ymax = ax.get_ylim()
                    # 添加垂直线标记选择的时间点
                    ax.axvline(x=self.x1, color='g', linestyle='--', alpha=0.5)
                    ax.axvline(x=self.x2, color='g', linestyle='--', alpha=0.5)
                    # 更新图表
                    self.canvas.draw_idle()

        # 连接事件处理函数
        self.canvas.mpl_connect('button_press_event', get_point)
        
    def _update_dynamic_force_labels(self):
        """更新图表上的动态力值标签位置以适应缩放"""
        if not hasattr(self, 'fig') or not hasattr(self.fig, 'axes') or len(self.fig.axes) == 0:
            return
            
        ax = self.fig.axes[0]
        
        # 如果有力值区域标签，更新它们的位置
        if hasattr(self, 'force_region_labels') and self.force_region_labels:
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            
            for region_idx, label in enumerate(self.force_region_labels):
                if region_idx < len(self.force_regions):
                    region = self.force_regions[region_idx]
                    # 计算新的标签位置 - 在视图中央
                    label_x = (region['start_time'] + region['end_time']) / 2
                    
                    # 如果标签在当前视图外，则隐藏
                    if label_x < x_min or label_x > x_max:
                        label.set_visible(False)
                    else:
                        label.set_visible(True)
                        # 重新定位在视图顶部
                        label_y = y_max - (y_max - y_min) * 0.05
                        label.set_position((label_x, label_y))
                        label.set_text(f"{region['mean_force']:.2f} pN")

    def plotfig(self):
        """绘制并更新图形显示"""
        # 准备数据
        self.currentfilterselection = self.filterselection.currentText()
        self.chosen_bead = self.y_axis_box.currentText()
        
        # X轴固定为时间
        self.x_axis = self.tdms_data_store['time s']
        self.y_axis = self.tdms_data_store[str(self.chosen_bead)]
        self.y_axis_2 = self.tdms_data_store['force pN']  # 保留力数据用于分析，但不绘制曲线
        self.xx = self.x_axis.values
        self.yy = self.y_axis.values
        self.yy_2 = self.y_axis_2.values

        # 清除当前图形 - 修复坐标轴重叠问题
        self.fig.clear()
        
        # 初始化动态标签列表
        self.force_region_labels = []
        
        # 根据UI状态更新控件可用性
        self.chose_sliced_data_box.setEnabled(self.sliced_data_box.isChecked())
        self.kernel_size_box.setEnabled(self.check_fitted_data_box.isChecked())
        
        # 获取数据段(原始或切片)
        axis_x, axis_y, axis_y_2 = self._get_plot_data()
        
        # 创建图形和子图
        ax = self.fig.add_subplot(111)
        
        # 绘制原始数据
        ax.plot(axis_x, axis_y, color='darkgrey', label='Raw Data')
        
        # 如果启用了拟合,应用滤波器并绘制拟合数据
        if self.check_fitted_data_box.isChecked():
            set_size = self.kernel_size_box.value()
            axis_x_fitted, y_fitted = self._apply_filter(axis_x, axis_y, set_size)
            ax.plot(axis_x_fitted, y_fitted, color='red', label='Filtered Data')
        
        # 添加力值区域划分并更新区域选择下拉框
        self._highlight_force_regions(ax, axis_x, axis_y_2)
        self._update_region_selector()
        
        # 添加图例
        ax.legend(loc='upper left')
        
        # 设置标签
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Extension (nm)')
        
        # 更新画布
        self.canvas.draw_idle()

    def _highlight_force_regions(self, ax, time_data, force_data, threshold=0.5):
        """根据力值变化添加背景色块以区分不同区域
        
        Args:
            ax: matplotlib轴对象
            time_data: 时间数据
            force_data: 力数据
            threshold: 力值变化阈值，用于检测力值显著变化
        """
        if len(time_data) < 2 or len(force_data) < 2:
            return
        
        # 清除之前的标签
        self.force_region_labels = []
        
        # 重置区域信息
        self.force_regions = []
        
        # 计算力值的梯度
        force_gradient = np.abs(np.gradient(force_data))
        
        # 获取力值变化显著的点
        # 使用 threshold 乘以力梯度的标准差，使检测更加稳健
        change_points = np.where(force_gradient > threshold * np.std(force_gradient))[0]
        
        # 合并临近的变化点，避免过度分段
        if len(change_points) > 0:
            merged_points = [change_points[0]]
            for point in change_points[1:]:
                if point - merged_points[-1] > 10:  # 至少间隔10个点
                    merged_points.append(point)
            change_points = np.array(merged_points)
        
        # 添加开始和结束点
        all_points = np.concatenate(([0], change_points, [len(time_data) - 1]))
        
        # 使用交替的背景色
        colors = ['lightyellow', 'lightblue', 'lightgreen', 'lightpink', 'lightgrey']
        
        # 为每个区域添加背景色和力值标注
        for i in range(len(all_points) - 1):
            start_idx = all_points[i]
            end_idx = all_points[i+1]
            
            # 确保区段长度足够
            if end_idx - start_idx < 10:
                continue
            
            # 计算区域持续时间
            region_duration = time_data[end_idx] - time_data[start_idx]
            
            # 如果区域持续时间小于最小值，跳过此区域
            if region_duration < self.min_region_duration:
                continue
            
            # 计算区域内的平均力值
            mean_force = np.mean(force_data[start_idx:end_idx])
            
            # 添加背景色块
            color = colors[i % len(colors)]
            ax.axvspan(time_data[start_idx], time_data[end_idx], 
                      alpha=0.3, color=color)
            
            # 添加力值标签 (放在区域顶部)
            if end_idx - start_idx > 20:  # 只有区域足够大才添加标签
                label_x = (time_data[start_idx] + time_data[end_idx]) / 2
                y_lim = ax.get_ylim()
                label_y = y_lim[1] - (y_lim[1] - y_lim[0]) * 0.05  # 放在顶部附近
                label = ax.text(label_x, label_y, f"{mean_force:.2f} pN", 
                       ha='center', va='top', fontsize=9,
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                self.force_region_labels.append(label)
            
            # 记录区域信息，便于后续选择
            self.force_regions.append({
                'start_time': time_data[start_idx],
                'end_time': time_data[end_idx],
                'mean_force': mean_force,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'duration': region_duration
            })

    def _update_region_selector(self):
        """更新区域选择下拉框"""
        # 清空当前项
        self.region_selector.clear()
        
        # 如果没有区域，显示提示并返回
        if not self.force_regions:
            self.region_selector.addItem("No regions detected")
            return
            
        # 添加每个区域到下拉框
        for i, region in enumerate(self.force_regions):
            # 显示区域编号、时间范围和平均力值
            self.region_selector.addItem(
                f"Region {i+1}: {region['start_time']:.2f}-{region['end_time']:.2f}s ({region['mean_force']:.2f} pN)"
            )

    def _get_plot_data(self):
        """根据UI设置获取绘图数据"""
        if self.sliced_data_box.isChecked():
            # 获取所选切片
            selected_sliced_data_num = int(self.chose_sliced_data_box.currentText()) - 1
            section_of_selected_data = self.final_slice_magnet_move_state[selected_sliced_data_num]
            
            # 计算切片起止点
            start_point = section_of_selected_data[0]
            if start_point != 0:
                start_point = start_point - 1
            end_point = section_of_selected_data[1] - 1
            
            # 切片数据
            axis_x = self.xx[start_point:end_point]
            axis_y = self.yy[start_point:end_point]
            axis_y_2 = self.yy_2[start_point:end_point]
        else:
            # 使用全部数据
            axis_x = self.xx
            axis_y = self.yy
            axis_y_2 = self.yy_2
            
        return axis_x, axis_y, axis_y_2

    def _apply_filter(self, x_data, y_data, kernel_size):
        """应用滤波器到数据"""
        if self.currentfilterselection == 'Median Filter':
            y_fitted = scipy.signal.medfilt(y_data, kernel_size=kernel_size)
            return x_data, y_fitted
        elif self.currentfilterselection == 'Moving Average':
            # 使用更高效的numpy卷积
            kernel = np.ones(kernel_size) / kernel_size
            y_fitted = np.convolve(y_data, kernel, mode='valid')
            
            # 返回相应的x轴数据
            valid_x = x_data[kernel_size-1:]
            
            return valid_x, y_fitted
        
        return x_data, y_data  # 默认不应用滤波

    def _prepare_time_data(self):
        """Helper method to prepare data for saving time intervals"""
        self.x1 = float(self.force_data_info.text())
        self.x2 = float(self.extension_data_info.text())
        
        # Find the closest time points in data
        self.time_point_start = min(self.xx, key=lambda x: abs(x - self.x1))
        self.time_point_start_data = self.xx.tolist().index(self.time_point_start)
        self.time_point_end = min(self.xx, key=lambda x: abs(x - self.x2))
        self.time_point_end_data = self.xx.tolist().index(self.time_point_end)

        # Extract relevant data
        time_range = slice(self.time_point_start_data, self.time_point_end_data)
        self.time_data = self.tdms_data_store['time s'].values[time_range]
        self.bead_data = self.tdms_data_store[str(self.y_axis_box.currentText())].values[time_range]
        self.force_data = self.tdms_data_store['force pN'].values[time_range]
        
        # Calculate force and time interval
        self.force_calculate = np.mean(self.force_data)
        return self.x2 - self.x1

    def _save_time_data(self, is_open_time):
        """Save time data to Excel file
        
        Args:
            is_open_time (bool): True for open time, False for close time
        """
        time_interval = self._prepare_time_data()
        
        # Configure columns based on data type
        if is_open_time:
            time_col, force_col, count_col = 1, 2, 3
            other_count_col = 6
            time_type = "open"
        else:
            time_col, force_col, count_col = 4, 5, 6
            other_count_col = 3
            time_type = "close"
        
        # Prepare file and worksheet
        self.current_bead_name = self.y_axis_box.currentText()
        xlsx_file_path = os.path.join(self.Data_Saved_Path, f"{self.base_name}_Dy.xlsx")
        worksheet_name = f"{self.current_bead_name}_{self.chose_sliced_data_box.currentText()}"
        
        # Create or open workbook
        if os.path.exists(xlsx_file_path):
            workbook = openpyxl.load_workbook(xlsx_file_path)
        else:
            workbook = openpyxl.Workbook()
            workbook.active.title = worksheet_name
        
        # Create or get worksheet
        if worksheet_name not in workbook.sheetnames:
            worksheet = workbook.create_sheet(worksheet_name)
            worksheet.append([
                'Time Interval (on)', 'Force (on, PN)', 'Counts(on)', 
                'Time Interval (off)', 'Force (off, PN)', 'Counts(off)'
            ])
            worksheet.cell(2, count_col, 0)
            worksheet.cell(2, other_count_col, 0)
        else:
            worksheet = workbook[worksheet_name]
        
        # Update counter
        n_counts = worksheet.cell(2, count_col).value or 0
        n_counts += 1
        
        # Write data
        worksheet.cell(1 + n_counts, time_col, time_interval)
        worksheet.cell(1 + n_counts, force_col, self.force_calculate)
        worksheet.cell(2, count_col, n_counts)
        
        # Save workbook
        workbook.save(xlsx_file_path)
        
        # 显示保存成功消息
        QMessageBox.information(self, "Save Successful", 
                               f"{time_type.capitalize()} time data saved to: {xlsx_file_path}")

    def save_time_data(self):
        """合并后的时间数据保存功能"""
        # 检查是否有选择的时间点
        if len(self.points_for_save) != 2:
            QMessageBox.warning(self, "Selection Error", "Please select two time points to define time interval first")
            return
        
        # 创建选择对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Time Type")
        dialog.resize(300, 150)
        
        layout = QVBoxLayout(dialog)
        
        # 创建单选按钮组
        group_box = QGroupBox("Select Time Type:")
        radio_layout = QVBoxLayout()
        
        open_radio = QRadioButton("Open Time")
        open_radio.setChecked(True)
        close_radio = QRadioButton("Close Time")
        
        radio_layout.addWidget(open_radio)
        radio_layout.addWidget(close_radio)
        group_box.setLayout(radio_layout)
        layout.addWidget(group_box)
        
        # 添加按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        # 显示对话框并处理结果
        if dialog.exec():
            is_open_time = open_radio.isChecked()
            self._save_time_data(is_open_time)
    
    def select_force_region(self):
        """选择特定力值区域进行分析"""
        if not hasattr(self, 'force_regions') or not self.force_regions:
            return
            
        selected_idx = self.region_selector.currentIndex()
        if (selected_idx < 0 or selected_idx >= len(self.force_regions)):
            return
            
        region = self.force_regions[selected_idx]
        
        # 更新时间点选择
        self.force_data_info.setText(f"{region['start_time']:.4f}")
        self.extension_data_info.setText(f"{region['end_time']:.4f}")
        
        # 更新图表选择
        ax = self.fig.axes[0]
        if hasattr(self, 'region_highlight_rectangle') and self.region_highlight_rectangle:
            try:
                self.region_highlight_rectangle.remove()
            except:
                pass
        
        # 添加高亮矩形
        y_min, y_max = ax.get_ylim()
        self.region_highlight_rectangle = ax.axvspan(region['start_time'], region['end_time'], 
                                                    alpha=0.4, color='yellow', zorder=-10)
        
        # 更新图表
        self.canvas.draw_idle()
        
        # 自动调用时间点更新函数
        self._update_time_points()
        self._update_force_display()
        
    def preanalyze_dynamics(self):
        """优化的预分析动力学功能"""
        if len(self.points_for_save) != 2:
            QMessageBox.warning(self, "Selection Error", "Please select two time points to define analysis region first")
            return
            
        # 获取已选时间点
        self.x1 = float(self.force_data_info.text())
        self.x2 = float(self.extension_data_info.text())
        
        # 确保时间点的顺序正确
        if self.x1 > self.x2:
            self.x1, self.x2 = self.x2, self.x1
            
        # 找到最接近的数据点索引
        self.time_point_start = min(self.xx, key=lambda x: abs(x - self.x1))
        self.time_point_start_data = self.xx.tolist().index(self.time_point_start)
        self.time_point_end = min(self.xx, key=lambda x: abs(x - self.x2))
        self.time_point_end_data = self.xx.tolist().index(self.time_point_end)
        
        # 获取数据
        time_range = slice(self.time_point_start_data, self.time_point_end_data)
        time_data = self.tdms_data_store['time s'].values[time_range]
        bead_data = self.tdms_data_store[str(self.y_axis_box.currentText())].values[time_range]
        force_data = self.tdms_data_store['force pN'].values[time_range]
        
        # 检查数据长度
        if len(bead_data) < 50:
            QMessageBox.warning(self, "Insufficient Data", "Selected region has too few data points for analysis")
            return
            
        # 应用滤波器去噪
        kernel_size = self.kernel_size_box.value()
        if self.currentfilterselection == 'Median Filter':
            filtered_data = scipy.signal.medfilt(bead_data, kernel_size=kernel_size)
        else:
            kernel = np.ones(kernel_size) / kernel_size
            filtered_data = np.convolve(bead_data, kernel, mode='same')
        
        # 获取用户选择的灵敏度
        sensitivity_level = self.sensitivity_combo.currentText()
        
        # 根据灵敏度级别设置参数
        if sensitivity_level == "Low":
            threshold_factor = 2.0
            min_prominence = 2.0
            min_distance = kernel_size
        elif sensitivity_level == "Medium":
            threshold_factor = 1.5
            min_prominence = 1.5
            min_distance = int(kernel_size * 0.8)
        elif sensitivity_level == "High":
            threshold_factor = 1.0
            min_prominence = 1.0
            min_distance = int(kernel_size * 0.5)
        else:  # Very High
            threshold_factor = 0.8
            min_prominence = 0.8
            min_distance = int(kernel_size * 0.3)
            
        # 计算标准差和信号变化
        signal_std = np.std(filtered_data)
        
        # 多维度检测状态变化点
        # 1. 基于梯度的变化点检测
        signal_diff = np.abs(np.diff(filtered_data))
        peak_threshold = threshold_factor * signal_std
        peaks_diff, _ = find_peaks(signal_diff, height=peak_threshold, distance=min_distance)
        
        # 2. 基于信号峰值突变的检测
        peaks_raw, _ = find_peaks(filtered_data, distance=min_distance)
        prominences = peak_prominences(filtered_data, peaks_raw)[0]
        peaks_prominence = peaks_raw[prominences > min_prominence * signal_std]
        
        # 3. 基于信号突变的谷值检测
        neg_filtered_data = -filtered_data
        valleys_raw, _ = find_peaks(neg_filtered_data, distance=min_distance)
        valley_prominences = peak_prominences(neg_filtered_data, valleys_raw)[0]
        valleys_prominence = valleys_raw[valley_prominences > min_prominence * signal_std]
        
        # 合并不同方法检测到的点
        all_peaks = np.unique(np.concatenate([peaks_diff, peaks_prominence, valleys_prominence]))
        
        # 过滤掉太接近的点
        if len(all_peaks) > 1:
            filtered_peaks = [all_peaks[0]]
            for peak in all_peaks[1:]:
                if peak - filtered_peaks[-1] >= min_distance:
                    filtered_peaks.append(peak)
            all_peaks = np.array(filtered_peaks)
        
        # 如果没有检测到变化点
        if len(all_peaks) == 0:
            QMessageBox.information(self, "Pre-Analysis Results", 
                                   "No significant dynamics changes detected in the selected region.\n"
                                   "Possible reasons:\n"
                                   "1. Small data fluctuations, no significant state transitions\n"
                                   "2. Try increasing detection sensitivity\n"
                                   "3. Inappropriate filter parameters, try adjusting filter type or kernel size\n"
                                   "4. Try zooming in to see more detailed regions")
            return
            
        # 清除之前的动态标注
        if hasattr(self, 'dynamics_annotations'):
            for ann in self.dynamics_annotations:
                try:
                    ann.remove()
                except:
                    pass
        
        # 获取图表轴对象
        ax = self.fig.axes[0]
        
        # 创建新的标注
        self.dynamics_annotations = []
        
        # 为检测到的变化点添加标记
        for peak_idx in all_peaks:
            if peak_idx < len(time_data):
                # 添加垂直线标记变化点
                line = ax.axvline(time_data[peak_idx], color='magenta', linestyle='-', 
                                 alpha=0.7, linewidth=1)
                self.dynamics_annotations.append(line)
                
                # 计算状态差值
                if peak_idx > 5 and peak_idx < len(filtered_data) - 5:
                    before_state = np.mean(filtered_data[max(0, peak_idx-5):peak_idx])
                    after_state = np.mean(filtered_data[peak_idx:min(len(filtered_data), peak_idx+5)])
                    state_diff = abs(after_state - before_state)
                    
                    # 添加标注文本
                    text = ax.text(time_data[peak_idx], filtered_data[peak_idx], 
                                   f"Δ{state_diff:.2f}nm", color='magenta',
                                   fontsize=8, ha='left', va='bottom',
                                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                    self.dynamics_annotations.append(text)
        
        # 绘制滤波后的曲线用于参考
        filter_line = ax.plot(time_data, filtered_data, 'orange', alpha=0.5, linewidth=1, label='Filtered Signal')
        self.dynamics_annotations.append(filter_line[0])
        
        # 更新画布
        self.canvas.draw_idle()
        
        # 显示统计信息
        QMessageBox.information(self, "Pre-Analysis Results", 
                               f"Detected {len(all_peaks)} possible state transitions.\n"
                               f"Average Force: {np.mean(force_data):.2f} pN\n"
                               f"Signal Std Dev: {signal_std:.2f} nm\n"
                               f"Detection Threshold: {peak_threshold:.2f} nm\n"
                               f"Current Sensitivity: {sensitivity_level}\n\n"
                               "Note: Pink lines mark potential state transition locations")

    def kinetics_analysis_calculation(self):
        self.x1 = float(self.force_data_info.text())
        self.x2 = float(self.extension_data_info.text())
        self.time_point_start = min(self.xx, key=lambda x: abs(x - self.x1))
        self.time_point_start_data = self.xx.tolist().index(self.time_point_start)
        self.time_point_end = min(self.xx, key=lambda x: abs(x - self.x2))
        self.time_point_end_data = self.xx.tolist().index(self.time_point_end)

        self.time_data_list = self.tdms_data_store['time s'].values
        self.time_data = self.time_data_list[self.time_point_start_data:self.time_point_end_data]
        self.bead_data_list = self.tdms_data_store[str(self.y_axis_box.currentText())].values
        self.bead_data = self.bead_data_list[self.time_point_start_data:self.time_point_end_data]
        self.force_data_list = self.tdms_data_store['force pN'].values
        self.force_data = self.force_data_list[self.time_point_start_data:self.time_point_end_data]

        self.fit_bead_data = scipy.signal.medfilt(self.bead_data, kernel_size=3)

        self.base_line = np.mean(np.array(self.fit_bead_data))
        self.force_calculate = np.mean(np.array(self.force_data))

        # Create HMM analysis dialog
        import matplotlib.pyplot as plt

        class StateSelectionDialog(QDialog):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setWindowTitle("HMM State Selection")
                self.resize(300, 150)
                
                layout = QVBoxLayout(self)
                
                # Create group box for radio buttons
                groupBox = QGroupBox("Select number of states:")
                radioLayout = QVBoxLayout()
                
                # Create radio buttons
                self.radio2 = QRadioButton("2 States")
                self.radio2.setChecked(True)
                self.radio3 = QRadioButton("3 States")
                
                radioLayout.addWidget(self.radio2)
                radioLayout.addWidget(self.radio3)
                groupBox.setLayout(radioLayout)
                layout.addWidget(groupBox)
                
                # Add buttons
                buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
                buttonBox.accepted.connect(self.accept)
                buttonBox.rejected.connect(self.reject)
                layout.addWidget(buttonBox)
            
            def getSelectedStates(self):
                if self.radio2.isChecked():
                    return 2
                else:
                    return 3

        # Show dialog to select number of states
        dialog = StateSelectionDialog(self)
        if dialog.exec():
            n_states = dialog.getSelectedStates()
            
            # Prepare data for HMM (reshape to 2D array)
            extension_data_for_hmm = np.array(self.bead_data).reshape(-1, 1)
            
            # Create and train HMM model
            model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100)
            model.fit(extension_data_for_hmm)
            
            # Get state sequence
            hidden_states = model.predict(extension_data_for_hmm)
            
            # Get state means for square wave generation
            state_means = []
            for i in range(n_states):
                mask = (hidden_states == i)
                if np.any(mask):
                    mean = np.mean(self.bead_data[mask])
                    state_means.append((i, mean))
            
            # Sort states by mean values
            state_means.sort(key=lambda x: x[1])
            state_map = {old_state: new_state for new_state, (old_state, _) in enumerate(state_means)}
            
            # Create square wave representation
            square_wave = np.zeros_like(self.bead_data)
            for i in range(n_states):
                mask = (hidden_states == i)
                square_wave[mask] = state_means[state_map[i]][1]
            
            # Get transition matrix
            transition_matrix = model.transmat_
            
            # Create HMM result window
            class HMMResultWindow(QMainWindow):
                def __init__(self, parent, time_data, extension_data, square_wave, hidden_states, 
                             transition_matrix, n_states, force_calculate, bead_name):
                    super().__init__()
                    self.parent = parent
                    self.time_data = time_data
                    self.extension_data = extension_data
                    self.square_wave = square_wave
                    self.hidden_states = hidden_states
                    self.transition_matrix = transition_matrix
                    self.n_states = n_states
                    self.force = force_calculate
                    self.bead_name = bead_name
                    
                    self.setWindowTitle(f"HMM Analysis Results - Force: {self.force:.2f} pN")
                    self.resize(800, 600)
                    
                    # Create central widget and layout
                    central_widget = QWidget()
                    layout = QVBoxLayout(central_widget)
                    
                    # Create figure and plot
                    self.fig = plt.figure(figsize=(10, 8))
                    self.ax = self.fig.add_subplot(111)
                    
                    # Plot original extension data
                    self.ax.plot(time_data, extension_data, 'gray', alpha=0.5, label='Original Data')
                    
                    # Plot square wave representation with state labels
                    self.ax.plot(time_data, square_wave, 'r-', linewidth=2, label='State Model')
                    
                    # Add annotation for each state level
                    state_values = []
                    colors = ['blue', 'green', 'purple']  # Different colors for different states
                    
                    # Map original states to physical meaning based on position
                    if n_states == 2:
                        state_names = ["Close", "Open"]
                    else:
                        state_names = ["Close", "Intermediate", "Open"]
                    
                    # Sort states by their mean values (lowest to highest)
                    sorted_states = sorted([(i, state_map[i], mean_val) for i, mean_val in state_means], key=lambda x: x[2])
                    
                    for idx, (orig_state, mapped_state, mean_val) in enumerate(sorted_states):
                        state_values.append((mapped_state + 1, mean_val))
                        color = colors[idx % len(colors)]
                        
                        # Add horizontal line for state level
                        self.ax.axhline(y=mean_val, color=color, linestyle='--', alpha=0.7)
                        
                        # Add text label showing both state number and physical meaning
                        label_text = f"State {mapped_state + 1}: {state_names[idx]}"
                        label_text += f" ({mean_val:.2f} nm)"
                        
                        self.ax.text(time_data[0] + (time_data[-1] - time_data[0]) * 0.05, mean_val, label_text, 
                                color=color, va='center', ha='left', fontsize=10, 
                                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
                    
                    self.ax.set_xlabel('Time (s)')
                    self.ax.set_ylabel('Extension (nm)')
                    self.ax.set_title(f'HMM State Analysis - {n_states} states, Force: {self.force:.2f} pN')
                    self.ax.legend()
                    
                    # Add transition probabilities as text
                    textstr = "Transition Probabilities:\n"
                    
                    # Create a mapping from original state to meaningful state name
                    state_name_map = {}
                    for idx, (orig_state, mapped_state, _) in enumerate(sorted_states):
                        state_name_map[orig_state] = state_names[idx]
                    
                    for i in range(n_states):
                        for j in range(n_states):
                            mapped_i = state_map.get(i, i)
                            mapped_j = state_map.get(j, j)
                            name_i = state_name_map.get(i, f"State {mapped_i+1}")
                            name_j = state_name_map.get(j, f"State {mapped_j+1}")
                            textstr += f"P({name_i} → {name_j}) = {transition_matrix[i, j]:.3f}\n"
                    
                    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                    self.ax.text(0.05, 0.95, textstr, transform=self.ax.transAxes, fontsize=9,
                            verticalalignment='top', bbox=props)
                    
                    # Create canvas
                    self.canvas = FigureCanvasQTAgg(self.fig)
                    layout.addWidget(self.canvas)
                    
                    # Add save button
                    save_button = QPushButton("Save Results")
                    save_button.clicked.connect(self.save_results)
                    
                    button_layout = QHBoxLayout()
                    button_layout.addWidget(save_button)
                    button_layout.addStretch(1)
                    layout.addLayout(button_layout)
                    
                    self.setCentralWidget(central_widget)
                
                def save_results(self):
                    # Create a dialog to select what to save
                    save_dialog = QDialog(self)
                    save_dialog.setWindowTitle("Save Options")
                    save_dialog.resize(300, 250)
                    
                    layout = QVBoxLayout(save_dialog)
                    
                    save_original = QCheckBox("Save Original Data")
                    save_original.setChecked(True)
                    save_fitted = QCheckBox("Save Fitted Curve")
                    save_fitted.setChecked(True)
                    save_transitions = QCheckBox("Save Transition Probabilities")
                    save_transitions.setChecked(True)
                    save_figure = QCheckBox("Save Figure Image")
                    save_figure.setChecked(True)
                    
                    # Add figure format selection
                    figure_format_layout = QHBoxLayout()
                    figure_format_label = QLabel("Figure Format:")
                    self.figure_format = QComboBox()
                    self.figure_format.addItems(["PNG", "PDF", "SVG"])
                    figure_format_layout.addWidget(figure_format_label)
                    figure_format_layout.addWidget(self.figure_format)
                    
                    layout.addWidget(save_original)
                    layout.addWidget(save_fitted)
                    layout.addWidget(save_transitions)
                    layout.addWidget(save_figure)
                    layout.addLayout(figure_format_layout)
                    
                    buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
                    buttonBox.accepted.connect(save_dialog.accept)
                    buttonBox.rejected.connect(save_dialog.reject)
                    layout.addWidget(buttonBox)
                    
                    if save_dialog.exec():
                        # Get base file path
                        base_filename = f"{self.parent.base_name}_{self.bead_name}_HMM_{self.n_states}states_{self.force:.2f}pN"
                        file_path = os.path.join(self.parent.Data_Saved_Path, f"{base_filename}.xlsx")
                        
                        # Save figure if option selected
                        if save_figure.isChecked():
                            figure_format = self.figure_format.currentText().lower()
                            figure_path = os.path.join(self.parent.Data_Saved_Path, f"{base_filename}.{figure_format}")
                            self.fig.savefig(figure_path, format=figure_format, dpi=300, bbox_inches='tight')
                        
                        # Only create Excel file if any of the data options are selected
                        if save_original.isChecked() or save_fitted.isChecked() or save_transitions.isChecked():
                            # Create workbook and worksheet
                            wb = openpyxl.Workbook()
                            ws = wb.active
                            ws.title = f"HMM Analysis {self.force:.2f}pN"
                            
                            # Add headers
                            headers = ["Time (s)"]
                            if save_original.isChecked():
                                headers.append("Original Extension (nm)")
                            if save_fitted.isChecked():
                                headers.append("Fitted State (nm)")
                                
                            # Add transition probability headers
                            if save_transitions.isChecked():
                                for i in range(self.n_states):
                                    for j in range(self.n_states):
                                        mapped_i = state_map.get(i, i)
                                        mapped_j = state_map.get(j, j)
                                        headers.append(f"P(State {mapped_i+1} → {mapped_j+1})")
                            
                            ws.append(headers)
                            
                            # Add data rows
                            for i in range(len(self.time_data)):
                                row = [self.time_data[i]]
                                if save_original.isChecked():
                                    row.append(self.extension_data[i])
                                if save_fitted.isChecked():
                                    row.append(self.square_wave[i])
                                
                                # Add transition probabilities in adjacent columns
                                # Only add them in first row since they don't change
                                if save_transitions.isChecked() and i == 0:
                                    for i_state in range(self.n_states):
                                        for j_state in range(self.n_states):
                                            row.append(self.transition_matrix[i_state, j_state])
                                elif save_transitions.isChecked():
                                    # Add empty cells for other rows
                                    for _ in range(self.n_states * self.n_states):
                                        row.append("")
                                
                                ws.append(row)
                            
                            # Save the workbook
                            wb.save(file_path)
                        
                        # Show success message with information about saved files
                        msg = "Saved files:\n"
                        if save_original.isChecked() or save_fitted.isChecked() or save_transitions.isChecked():
                            msg += f"- Data: {file_path}\n"
                        if save_figure.isChecked():
                            msg += f"- Figure: {figure_path}"
                        
                        QMessageBox.information(self, "Save Complete", msg)
                
            # Show HMM results window
            self.hmm_window = HMMResultWindow(self, self.time_data, self.bead_data, square_wave, 
                                             hidden_states, transition_matrix, n_states,
                                             self.force_calculate, self.y_axis_box.currentText())
            self.hmm_window.show()

    def save_selected_data(self):
        # Extract time points from UI and find corresponding indices
        self.x1 = float(self.force_data_info.text())
        self.x2 = float(self.extension_data_info.text())
        self.time_point_start = min(self.xx, key=lambda x: abs(x - self.x1))
        self.time_point_start_data = self.xx.tolist().index(self.time_point_start)
        self.time_point_end = min(self.xx, key=lambda x: abs(x - self.x2))
        self.time_point_end_data = self.xx.tolist().index(self.time_point_end)

        # Extract data for the selected time range
        time_range = slice(self.time_point_start_data, self.time_point_end_data)
        self.time_data = self.tdms_data_store['time s'].values[time_range]
        self.bead_data = self.tdms_data_store[str(self.y_axis_box.currentText())].values[time_range]
        self.force_data = self.tdms_data_store['force pN'].values[time_range]

        # Apply appropriate filter
        kernel_size = self.kernel_size_box.value()
        if self.currentfilterselection == 'Median Filter':
            self.fit_bead_data = scipy.signal.medfilt(self.bead_data, kernel_size=kernel_size)
        elif self.currentfilterselection == 'Moving Average':
            self.fit_bead_data = np.convolve(self.bead_data, np.ones((kernel_size,)) / kernel_size, mode='valid')
            offset = kernel_size - 1
            self.time_data = self.time_data[offset:]
            self.bead_data = self.bead_data[offset:]
            self.force_data = self.force_data[offset:]

        # Calculate average force
        self.force_calculate = np.mean(self.force_data)

        # Prepare file paths
        self.current_bead_name = self.y_axis_box.currentText()
        base_filename = f"{self.base_name}_{self.current_bead_name}_{self.force_calculate:.2f}pN"
        xlsx_file_path = os.path.join(self.Data_Saved_Path, f"{base_filename}_SelectedData.xlsx")
        image_file_path = os.path.join(self.Data_Saved_Path, f"{base_filename}_SelectedData.png")
        worksheet_name = f"{self.current_bead_name}_{self.force_calculate:.2f}pN"

        # Create and save the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.time_data, self.bead_data, 'gray', alpha=0.5, label='Raw Data')
        ax.plot(self.time_data, self.fit_bead_data, 'r-', linewidth=2, 
                label=f'Filtered Data ({self.currentfilterselection}, kernel={kernel_size})')
        
        # 添加力值区域标注
        self._highlight_force_regions_in_save_plot(ax, self.time_data, self.force_data)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Extension (nm)')
        ax.set_title(f'Selected Data - {self.current_bead_name}, Force: {self.force_calculate:.2f} pN')
        ax.legend(loc='upper left')
        fig.savefig(image_file_path, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory

        # Create or load workbook
        workbook = openpyxl.load_workbook(xlsx_file_path) if os.path.exists(xlsx_file_path) else openpyxl.Workbook()
        
        if worksheet_name not in workbook.sheetnames:
            # Use active sheet for new workbook or create new sheet
            if len(workbook.sheetnames) == 1 and workbook.active.title == 'Sheet':
                worksheet = workbook.active
                worksheet.title = worksheet_name
            else:
                worksheet = workbook.create_sheet(worksheet_name)
            
            # Add headers and data
            worksheet.append(['Time (s)', 'Extension (nm)', 'Fit Extension (nm)', 'Force (pN)'])
            for i in range(len(self.time_data)):
                worksheet.cell(i + 2, 1, self.time_data[i])
                worksheet.cell(i + 2, 2, self.bead_data[i])
                worksheet.cell(i + 2, 3, self.fit_bead_data[i])
                worksheet.cell(i + 2, 4, self.force_data[i])
            
            # Save workbook
            workbook.save(xlsx_file_path)
            
        # Show confirmation message
        QMessageBox.information(self, "Save Complete", 
                               f"Data saved to:\n- Excel: {xlsx_file_path}\n- Image: {image_file_path}")

    def _highlight_force_regions_in_save_plot(self, ax, time_data, force_data, threshold=0.3):
        """为保存的图片标注不同的力值区域
        
        Args:
            ax: matplotlib轴对象
            time_data: 时间数据
            force_data: 力数据
            threshold: 力值变化阈值，用于检测力值显著变化
        """
        if len(time_data) < 2 or len(force_data) < 2:
            return
        
        # 计算力值的梯度
        force_gradient = np.abs(np.gradient(force_data))
        
        # 获取力值变化显著的点
        # 使用自适应阈值，确保小区域的变化也能检测到
        change_points = np.where(force_gradient > threshold * np.std(force_gradient))[0]
        
        # 合并临近的变化点，避免过度分段
        if len(change_points) > 0:
            merged_points = [change_points[0]]
            for point in change_points[1:]:
                if point - merged_points[-1] > 5:  # 更小的间隔以捕获更多的变化
                    merged_points.append(point)
            change_points = np.array(merged_points)
        
        # 添加开始和结束点
        all_points = np.concatenate(([0], change_points, [len(time_data) - 1]))
        
        # 使用交替的背景色
        colors = ['lightyellow', 'lightblue', 'lightgreen', 'lightpink', 'lightgrey']
        
        # 为每个区域添加背景色和力值标注
        for i in range(len(all_points) - 1):
            start_idx = all_points[i]
            end_idx = all_points[i+1]
            
            # 确保区段长度足够
            if end_idx - start_idx < 5:
                continue
            
            # 计算区域内的平均力值
            mean_force = np.mean(force_data[start_idx:end_idx])
            
            # 添加背景色块
            color = colors[i % len(colors)]
            ax.axvspan(time_data[start_idx], time_data[end_idx], 
                      alpha=0.3, color=color)
            
            # 添加更加明显的力值标签 (放在区域中央)
            if end_idx - start_idx > 10:  # 只有区域足够大才添加标签
                label_x = (time_data[start_idx] + time_data[end_idx]) / 2
                y_lim = ax.get_ylim()
                label_y = y_lim[1] - (y_lim[1] - y_lim[0]) * 0.1  # 放在顶部附近
                
                # 添加带边框的白色背景标签
                ax.text(label_x, label_y, f"{mean_force:.2f} pN", 
                       ha='center', va='center', fontsize=10, fontweight='bold',
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5'))

    def _update_time_points(self):
        """更新选中的时间点数据"""
        try:
            # 从UI获取选择的时间点
            self.x1 = float(self.force_data_info.text())
            self.x2 = float(self.extension_data_info.text())
            
            # 找到最接近的数据点
            self.time_point_start = min(self.xx, key=lambda x: abs(x - self.x1))
            self.time_point_start_data = self.xx.tolist().index(self.time_point_start)
            self.time_point_end = min(self.xx, key=lambda x: abs(x - self.x2))
            self.time_point_end_data = self.xx.tolist().index(self.time_point_end)
            
            # 确保开始点在结束点之前
            if (self.time_point_start_data > self.time_point_end_data):
                self.time_point_start_data, self.time_point_end_data = self.time_point_end_data, self.time_point_start_data
            
            # 提取相应的数据段
            time_range = slice(self.time_point_start_data, self.time_point_end_data)
            self.time_data = self.tdms_data_store['time s'].values[time_range]
            self.bead_data = self.tdms_data_store[str(self.y_axis_box.currentText())].values[time_range]
            self.force_data = self.tdms_data_store['force pN'].values[time_range]
        except Exception as e:
            print(f"更新时间点时出错: {e}")

    def _update_force_display(self):
        """计算并显示选定区域的平均力值"""
        try:
            if hasattr(self, 'force_data') and len(self.force_data) > 0:
                # 计算平均力值
                self.force_calculate = np.mean(self.force_data)
                # 更新UI显示
                self.force_data_display.setText(f"{self.force_calculate:.2f} pN")
        except Exception as e:
            print(f"更新力值显示时出错: {e}")

