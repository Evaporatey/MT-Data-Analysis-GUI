# -*- coding: utf-8 -*-
# author: Ye Yang
# Data Analysis GUI for MT
# 指数计算器可以不选择文件直接运行，最后存储的时候再选择位置

import os
import sys
import traceback  # Import traceback for detailed error reporting
from pathlib import Path

from PySide6.QtCore import QFileInfo, Qt  # Import Qt for alignment
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QMainWindow,
                               QPushButton, QSizePolicy, QStatusBar, QVBoxLayout, QWidget, QFileDialog, QLabel, QLineEdit, QMessageBox)

import MtFe
import MtFc
import MtDy
import ExponentialCalculator  # 添加新模块导入

class MainWindow(QMainWindow):

    # Placeholder texts
    PLACEHOLDER_SELECT_FILE = "Select Files or Input File Path"
    PLACEHOLDER_SAVE_DIR = "Optional: Specify Save Directory"

    def __init__(self):
        super().__init__()

        self.setWindowTitle("MT Data Analysis")

        # --- Central Widget and Main Layout ---
        self.centralwidget = QWidget()
        self.setCentralWidget(self.centralwidget)
        self.layoutcentralwiget = QVBoxLayout(self.centralwidget)  # Main vertical layout

        # --- Function Selection ---
        function_layout = QHBoxLayout()  # Horizontal layout for function selection
        self.functionselectable = QLabel("Function:", self.centralwidget)
        self.functionselectable.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        function_layout.addWidget(self.functionselectable)

        self.functionselectgroup = QComboBox(self.centralwidget)
        self.functionselectgroup.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.functionselectgroup.addItems(["Force-Extension Analysis", "Force Calibration",
                                           "Kinetics Analysis", "Exponential Calculator"])
        self.functionselectgroup.currentIndexChanged.connect(self.on_function_changed)  # Connect signal
        function_layout.addWidget(self.functionselectgroup)
        self.layoutcentralwiget.addLayout(function_layout)  # Add function layout to main layout

        # --- File/Directory Selection ---
        file_layout = QHBoxLayout()  # Horizontal layout for file selection
        self.file_path_name = QLineEdit(self.centralwidget)  # Placeholder set in on_function_changed
        self.file_path_name.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        file_layout.addWidget(self.file_path_name)

        self.select_files = QPushButton(self.centralwidget)  # Text set in on_function_changed
        self.select_files.clicked.connect(self.on_select_path_button_clicked)  # Connect to unified handler
        file_layout.addWidget(self.select_files)
        self.layoutcentralwiget.addLayout(file_layout)  # Add file layout to main layout

        # --- Run Button ---
        self.runbutton = QPushButton("Run", self.centralwidget)
        self.runbutton.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.runbutton.clicked.connect(self.on_run_button_clicked)
        self.layoutcentralwiget.addWidget(self.runbutton, 0, Qt.AlignCenter)  # Add button centered

        # --- Status Bar ---
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("By: Ye, Y(2023).")

        # --- Initialize Variables ---
        self.last_directory = os.path.expanduser("~")  # Default to home directory
        self.file_name = ""  # Initialize file_name
        self.file_type = ""  # Initialize file_type

        # --- Set Initial UI State ---
        self.on_function_changed()  # Call once to set initial state based on default selection

    def on_function_changed(self):
        """Updates UI elements when the selected function changes."""
        current_function = self.functionselectgroup.currentText()
        if current_function == "Exponential Calculator":
            self.file_path_name.setPlaceholderText(self.PLACEHOLDER_SAVE_DIR)
            if self.file_path_name.text() and not os.path.isdir(self.file_path_name.text()):
                self.file_path_name.setText("")
            self.select_files.setText("Select Directory")
        else:
            self.file_path_name.setPlaceholderText(self.PLACEHOLDER_SELECT_FILE)
            self.select_files.setText("Select Files")

    def on_select_path_button_clicked(self):
        """Handles clicks on the 'Select Files' / 'Select Directory' button."""
        current_function = self.functionselectgroup.currentText()
        if current_function == "Exponential Calculator":
            self._select_save_directory()
        else:
            self._select_data_file()

    def _select_data_file(self):
        """Opens a dialog to select a data file."""
        selected_file, selected_filter = QFileDialog.getOpenFileName(
            self,
            "Select one file",
            self.last_directory,
            "TDMS Files (*.tdms);;All Files (*)"
        )
        if selected_file:
            self.file_name = selected_file
            self.file_type = selected_filter
            self.file_path_name.setText(self.file_name)
            self.last_directory = os.path.dirname(self.file_name)

    def _select_save_directory(self):
        """Opens a dialog to select a save directory."""
        selected_directory = QFileDialog.getExistingDirectory(
            self,
            "Select Save Directory",
            self.last_directory
        )
        if selected_directory:
            self.file_path_name.setText(selected_directory)
            self.last_directory = selected_directory

    def get_file_info(self):
        """Gets file information, updating last_directory if path is manually entered."""
        current_path_text = self.file_path_name.text()

        if not self.file_name or self.file_name != current_path_text:
            if os.path.isfile(current_path_text):
                self.file_name = current_path_text
                self.file_type = ""
                self.last_directory = os.path.dirname(self.file_name)
            else:
                self.file_name = None

        if not self.file_name or not os.path.isfile(self.file_name):
            return None

        file_info = QFileInfo(self.file_name)
        data_saved_path = file_info.absolutePath()
        base_name = Path(self.file_name).stem
        return {
            'file_name': self.file_name,
            'file_type': self.file_type,
            'Data_Saved_Path': data_saved_path,
            'base_name': base_name,
            'self.file_info': file_info
        }

    def Force_Extension_Analysis(self):
        file_details = self.get_file_info()
        if file_details is None:
            QMessageBox.warning(self, "Warning", "Invalid or non-existent file path specified.", QMessageBox.Ok)
            return
        file_details['calculate_force'] = True
        self.FE_Analysis_Window = MtFe.FigureView(file_details)
        self.FE_Analysis_Window.show()

    def force_calibration(self):
        file_details = self.get_file_info()
        if file_details is None:
            QMessageBox.warning(self, "Warning", "Invalid or non-existent file path specified.", QMessageBox.Ok)
            return
        file_details['calculate_force'] = False
        self.Calibration_Window = MtFc.ForceCalibration(file_details)
        self.Calibration_Window.show()

    def knetics_analysis(self):
        file_details = self.get_file_info()
        if file_details is None:
            QMessageBox.warning(self, "Warning", "Invalid or non-existent file path specified.", QMessageBox.Ok)
            return
        file_details['calculate_force'] = True
        self.Kinetic_Window = MtDy.KineticsAnalysis(file_details)
        self.Kinetic_Window.show()

    def exponential_calculator(self):
        """Runs the Exponential Calculator, using the path in QLineEdit as save dir if valid."""
        save_path = os.path.expanduser("~")
        path_text = self.file_path_name.text()

        if path_text and path_text != self.PLACEHOLDER_SAVE_DIR and os.path.isdir(path_text):
            save_path = path_text
            self.last_directory = save_path

        data_for_calculator = {
            'Data_Saved_Path': save_path,
            'file_name': "",
            'base_name': "exponential_calculation",
            'file_type': "",
            'self.file_info': None
        }

        self.Exp_Calc_Window = ExponentialCalculator.ExponentialCalculator(data_for_calculator)
        self.Exp_Calc_Window.show()

    def on_run_button_clicked(self):
        current_function = self.functionselectgroup.currentText()

        if current_function == "Exponential Calculator":
            try:
                self.exponential_calculator()
            except Exception as e:
                error_details = traceback.format_exc()
                QMessageBox.critical(self, "Error", f"An error occurred while running {current_function}:\n{e}\n\nDetails:\n{error_details}", QMessageBox.Ok)
            return

        file_details = self.get_file_info()
        if file_details is None:
            if self.file_path_name.text() and self.file_path_name.text() != self.PLACEHOLDER_SELECT_FILE:
                QMessageBox.warning(self, "Warning", "The specified file path is invalid or does not exist. Please select a valid file.", QMessageBox.Ok)
            else:
                QMessageBox.warning(self, "Warning", "Please select or enter a valid file path first!", QMessageBox.Ok)
            return

        functions = {
            "Force-Extension Analysis": self.Force_Extension_Analysis,
            "Force Calibration": self.force_calibration,
            "Kinetics Analysis": self.knetics_analysis,
        }

        try:
            selected_method = functions.get(current_function)
            if selected_method:
                selected_method()
            else:
                QMessageBox.warning(self, "Warning", f"Function '{current_function}' is not implemented correctly.", QMessageBox.Ok)
        except Exception as e:
            error_details = traceback.format_exc()
            QMessageBox.critical(self, "Error", f"An error occurred while running {current_function}:\n{e}\n\nDetails:\n{error_details}", QMessageBox.Ok)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())
