# author: Ye Yang
# For exponential equation solving and plotting
# help to calculate the X value corresponding to the given Y value in MT measurement
# x is not accurate when y is too large

# Force Clibration: F=43.89994*exp(-0.76672*H)+41.72197*exp(-0.76717*H)-0.16789 (Date: 2023-05-26)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from scipy.optimize import fsolve
from force_models import get_default_parameters  # Import default parameters function

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QRadioButton,
                              QLineEdit, QPushButton, QGroupBox, QLabel, QButtonGroup,
                              QMessageBox, QFileDialog, QSizePolicy, QDoubleSpinBox)
from PySide6.QtCore import Qt

class ExponentialCalculator(QWidget):
    def __init__(self, data_for_figure):
        super().__init__()
        
        # Get file information in a safer way
        self.Data_Saved_Path = data_for_figure.get('Data_Saved_Path', os.path.expanduser("~/Documents"))
        self.file_name = data_for_figure.get('file_name', "")
        self.base_name = data_for_figure.get('base_name', "exponential_calculation")
        
        # Initialize UI
        self._init_ui()
        
    def _init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("Exponential Equation Calculator")
        self.resize(1200, 800)
        
        # Create main layout
        main_layout = QHBoxLayout(self)
        
        # Left parameter panel
        parameter_panel = QWidget()
        parameter_panel.setFixedWidth(400)
        parameter_layout = QVBoxLayout(parameter_panel)
        
        # Add equation type selection group
        equation_group = QGroupBox("Equation Type")
        equation_layout = QVBoxLayout(equation_group)
        
        self.equation_button_group = QButtonGroup(equation_group)
        self.single_exp_radio = QRadioButton("Single Exponential: y = a * exp(b * x) + c")
        self.double_exp_radio = QRadioButton("Double Exponential: y = a1 * exp(b1 * x) + a2 * exp(b2 * x) + c")
        self.single_exp_radio.setChecked(True)
        
        self.equation_button_group.addButton(self.single_exp_radio)
        self.equation_button_group.addButton(self.double_exp_radio)
        
        equation_layout.addWidget(self.single_exp_radio)
        equation_layout.addWidget(self.double_exp_radio)
        
        parameter_layout.addWidget(equation_group)
        
        # Get default force calculation parameters
        default_params = get_default_parameters()
        # Get default single exponential parameters
        default_single_params = get_default_parameters('single_exp')
        
        self.single_exp_group = QGroupBox("Single Exponential Parameters")
        single_exp_layout = QFormLayout(self.single_exp_group)
        
        self.a_input = QLineEdit(str(default_single_params['a']))
        self.b_input = QLineEdit(str(default_single_params['b']))
        self.c_input = QLineEdit(str(default_single_params['c']))
        
        single_exp_layout.addRow("a:", self.a_input)
        single_exp_layout.addRow("b:", self.b_input)
        single_exp_layout.addRow("c:", self.c_input)
        
        parameter_layout.addWidget(self.single_exp_group)
        
        # Double exponential parameters group
        self.double_exp_group = QGroupBox("Double Exponential Parameters")
        double_exp_layout = QFormLayout(self.double_exp_group)
        
        self.a1_input = QLineEdit(str(default_params['a1']))
        self.b1_input = QLineEdit(str(default_params['b1']))
        self.a2_input = QLineEdit(str(default_params['a2']))
        self.b2_input = QLineEdit(str(default_params['b2']))
        self.c2_input = QLineEdit(str(default_params['c']))
        
        double_exp_layout.addRow("a1:", self.a1_input)
        double_exp_layout.addRow("b1:", self.b1_input)
        double_exp_layout.addRow("a2:", self.a2_input)
        double_exp_layout.addRow("b2:", self.b2_input)
        double_exp_layout.addRow("c:", self.c2_input)
        
        parameter_layout.addWidget(self.double_exp_group)
        self.double_exp_group.setVisible(False)
        
        # Connect radio button signals
        self.single_exp_radio.toggled.connect(self._toggle_parameter_groups)
        
        # Y value range settings
        y_range_group = QGroupBox("Y Value Range")
        y_range_layout = QFormLayout(y_range_group)
        
        self.y_start_input = QLineEdit("0.0")
        self.y_end_input = QLineEdit("10.0")
        self.y_step_input = QLineEdit("0.5")
        
        y_range_layout.addRow("Start Y:", self.y_start_input)
        y_range_layout.addRow("End Y:", self.y_end_input)
        y_range_layout.addRow("Y Step:", self.y_step_input)
        
        parameter_layout.addWidget(y_range_group)
        
        # Initial X value estimation
        x_estimate_group = QGroupBox("X Value Estimation Range")
        x_estimate_layout = QFormLayout(x_estimate_group)
        
        self.x_min_input = QLineEdit("0.0")
        self.x_max_input = QLineEdit("10.0")
        
        x_estimate_layout.addRow("Min X:", self.x_min_input)
        x_estimate_layout.addRow("Max X:", self.x_max_input)
        
        parameter_layout.addWidget(x_estimate_group)
        
        # Calculate buttons
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)
        
        self.calculate_button = QPushButton("Calculate")
        self.calculate_button.clicked.connect(self.calculate)
        self.save_button = QPushButton("Save Results")
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setEnabled(False)
        
        buttons_layout.addWidget(self.calculate_button)
        buttons_layout.addWidget(self.save_button)
        
        parameter_layout.addWidget(buttons_widget)
        
        parameter_layout.addStretch(1)  # Add elastic space
        
        # Right result display panel
        results_panel = QWidget()
        results_layout = QVBoxLayout(results_panel)
        
        # Chart display
        self.fig = plt.figure(figsize=(6, 5))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        results_layout.addWidget(self.canvas)
        results_layout.addWidget(self.toolbar)
        
        # Results table label
        self.results_label = QLabel("Results will be displayed here")
        results_layout.addWidget(self.results_label)
        
        # Add panels to main layout
        main_layout.addWidget(parameter_panel)
        main_layout.addWidget(results_panel, 1)  # Right side takes more space
        
    def _toggle_parameter_groups(self, checked):
        """Toggle parameter group visibility"""
        self.single_exp_group.setVisible(self.single_exp_radio.isChecked())
        self.double_exp_group.setVisible(self.double_exp_radio.isChecked())
    
    def calculate(self):
        """Execute calculation"""
        try:
            # Get Y value range
            y_start = float(self.y_start_input.text())
            y_end = float(self.y_end_input.text())
            y_step = float(self.y_step_input.text())
            
            # Generate Y value array
            y_values = np.arange(y_start, y_end + y_step/2, y_step)  # Include end value
            
            # Get X value estimation range
            x_min = float(self.x_min_input.text())
            x_max = float(self.x_max_input.text())
            
            # Store results
            self.results = {"Y": y_values, "X": []}
            
            # Calculate using selected equation
            if self.single_exp_radio.isChecked():
                # Single exponential
                a = float(self.a_input.text())
                b = float(self.b_input.text())
                c = float(self.c_input.text())
                
                # Define equation (y - f(x))
                def equation(x, y_target):
                    return a * np.exp(b * x) + c - y_target
                
                # Clear chart
                self.fig.clear()
                ax = self.fig.add_subplot(111)
                
                # Plot equation curve
                x_plot = np.linspace(x_min, x_max, 1000)
                y_plot = a * np.exp(b * x_plot) + c
                ax.plot(x_plot, y_plot, 'b-', label=f'y = {a}·exp({b}·x) + {c}')
                
                # Calculate X for each Y value
                for y_target in y_values:
                    try:
                        # Try to solve between x_min and x_max
                        x0 = (x_min + x_max) / 2  # Initial guess
                        x_result = fsolve(equation, x0, args=(y_target,))[0]
                        
                        # Verify solution is within range and converged
                        # Use relative error for validation, fairer for large values
                        error = abs(equation(x_result, y_target))
                        relative_error = error / max(1.0, abs(y_target))
                        
                        if x_min <= x_result <= x_max and relative_error < 1e-6:
                            self.results["X"].append(x_result)
                            # Mark point on the chart
                            ax.plot(x_result, y_target, 'ro', markersize=5)
                        else:
                            self.results["X"].append(np.nan)  # No solution or out of range
                    except Exception as e:
                        print(f"Error calculating for Y={y_target}: {str(e)}")
                        self.results["X"].append(np.nan)  # Calculation error
            
            else:
                # Double exponential
                a1 = float(self.a1_input.text())
                b1 = float(self.b1_input.text())
                a2 = float(self.a2_input.text())
                b2 = float(self.b2_input.text())
                c = float(self.c2_input.text())
                
                # Define equation (y - f(x))
                def equation(x, y_target):
                    return a1 * np.exp(b1 * x) + a2 * np.exp(b2 * x) + c - y_target
                
                # Clear chart
                self.fig.clear()
                ax = self.fig.add_subplot(111)
                
                # Plot equation curve
                x_plot = np.linspace(x_min, x_max, 1000)
                y_plot = a1 * np.exp(b1 * x_plot) + a2 * np.exp(b2 * x_plot) + c
                ax.plot(x_plot, y_plot, 'b-', label=f'y = {a1}·exp({b1}·x) + {a2}·exp({b2}·x) + {c}')
                
                # Calculate X for each Y value
                for y_target in y_values:
                    solved = False
                    # Try multiple different initial guesses
                    guess_points = [
                        (x_min + x_max) / 2,  # Midpoint
                        x_min,                # Min boundary
                        x_max,                # Max boundary
                        x_min + (x_max - x_min) / 4,  # 1/4 point
                        x_min + 3 * (x_max - x_min) / 4  # 3/4 point
                    ]
                    
                    for x0 in guess_points:
                        try:
                            # Add more control parameters to improve convergence
                            x_result = fsolve(equation, x0, args=(y_target,), 
                                             full_output=True, 
                                             xtol=1.49012e-8, 
                                             maxfev=1000)
                            
                            # Check if solution converged (second element of return value is info dict)
                            if x_result[2] == 1:  # Check convergence status
                                x_val = x_result[0][0]  # Get solution
                                
                                # Verify solution is reasonable
                                error = abs(equation(x_val, y_target))
                                relative_error = error / max(1.0, abs(y_target))
                                
                                # Relax boundary check tolerance
                                boundary_tolerance = 0.01 * (x_max - x_min)
                                if (x_min - boundary_tolerance <= x_val <= x_max + boundary_tolerance and 
                                    relative_error < 1e-4):  # Relax convergence condition
                                    x_val = max(x_min, min(x_max, x_val))  # Constrain near-boundary solutions
                                    self.results["X"].append(x_val)
                                    # Mark point on chart
                                    ax.plot(x_val, y_target, 'ro', markersize=5)
                                    solved = True
                                    break
                        except Exception as e:
                            print(f"Error trying from starting point {x0} for Y={y_target}: {str(e)}")
                            continue
                    
                    if not solved:
                        print(f"Could not solve for X value at Y={y_target}")
                        self.results["X"].append(np.nan)  # No solution or out of range
            
            # Set chart properties
            ax.set_xlabel('X Value')
            ax.set_ylabel('Y Value')
            ax.set_title('Equation Curve and Solutions')
            ax.grid(True)
            ax.legend()
            
            # Display limited results
            result_text = "Y Value   |   X Value\n" + "-" * 20 + "\n"
            for i, (y, x) in enumerate(zip(y_values, self.results["X"])):
                if i < 10:  # Only display first 10 results
                    if np.isnan(x):
                        result_text += f"{y:.3f} | No solution\n"
                    else:
                        result_text += f"{y:.3f} | {x:.3f}\n"
                elif i == 10:
                    result_text += "...(more results omitted)...\n"
            
            self.results_label.setText(result_text)
            self.canvas.draw()
            
            # Enable save button
            self.save_button.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", f"Error during calculation: {str(e)}")
    
    def save_results(self):
        """Save results to Excel file"""
        try:
            # Check if results exist
            if not hasattr(self, 'results'):
                QMessageBox.warning(self, "No Results", "Please calculate results first")
                return
                
            # Default filename
            if self.base_name:
                default_filename = os.path.join(self.Data_Saved_Path, f"{self.base_name}_exp_calc.xlsx")
            else:
                default_filename = os.path.join(self.Data_Saved_Path, "exponential_calculation.xlsx")
        
            # Get save path
            file_name, _ = QFileDialog.getSaveFileName(self, 
                                                      "Save Results", 
                                                      default_filename, 
                                                      "Excel Files (*.xlsx)")
            
            if not file_name:
                return  # User canceled
                
            # Create DataFrame and save
            df = pd.DataFrame({
                "Y Value": self.results["Y"],
                "X Value": self.results["X"]
            })
            
            # Add equation parameter information
            if self.single_exp_radio.isChecked():
                a = float(self.a_input.text())
                b = float(self.b_input.text())
                c = float(self.c_input.text())
                info = pd.DataFrame({
                    "Equation Type": ["Single Exponential"],
                    "Equation": [f"y = {a} * exp({b} * x) + {c}"],
                    "a": [a],
                    "b": [b],
                    "c": [c]
                })
            else:
                a1 = float(self.a1_input.text())
                b1 = float(self.b1_input.text())
                a2 = float(self.a2_input.text())
                b2 = float(self.b2_input.text())
                c = float(self.c2_input.text())
                info = pd.DataFrame({
                    "Equation Type": ["Double Exponential"],
                    "Equation": [f"y = {a1} * exp({b1} * x) + {a2} * exp({b2} * x) + {c}"],
                    "a1": [a1],
                    "b1": [b1],
                    "a2": [a2],
                    "b2": [b2],
                    "c": [c]
                })
            
            # Use ExcelWriter to write multiple sheets and set numeric format
            with pd.ExcelWriter(file_name) as writer:
                df.to_excel(writer, sheet_name="Results", index=False)
                # Set numeric format to 3 decimal places
                workbook = writer.book
                worksheet = writer.sheets["Results"]
                
                # Set result display format to 3 decimal places
                for i in range(len(df)):
                    cell_y = worksheet.cell(i+2, 1)  # +2 because Excel starts at 1 and has header
                    cell_x = worksheet.cell(i+2, 2)
                    cell_y.number_format = '0.000'
                    cell_x.number_format = '0.000'
                
                info.to_excel(writer, sheet_name="Equation Info", index=False)
            
            # Save chart
            fig_path = file_name.replace('.xlsx', '.png')
            self.fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            
            QMessageBox.information(self, "Save Successful", f"Results saved to\n{file_name}\nChart saved to\n{fig_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Error during save: {str(e)}")