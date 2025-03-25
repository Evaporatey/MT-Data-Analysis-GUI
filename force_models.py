# author: Ye Yang
# manage force calculation functions
# -*- coding: utf-8 -*-

import numpy as np

# 力校准默认参数 (2023-05-26)
DEFAULT_FORCE_PARAMS = {
    'a1': 43.89994,
    'b1': -0.76672,
    'a2': 41.72197,
    'b2': -0.76717,
    'c': -0.16789
}

# 单指数模型默认参数
DEFAULT_SINGLE_EXP_PARAMS = {
    'a': 85.0,  # 近似于a1+a2
    'b': -0.77,  # 近似于b1和b2的平均值
    'c': -0.17   # 近似于c
}

def calculate_force(height, params=None, model='double_exp'):
    """
    根据磁铁高度计算力
    
    Args:
        height: 磁铁高度，单位mm (可以是数组或单个值)
        params: 参数字典，默认使用各模型的默认参数
        model: 使用的模型，'double_exp'(双指数)或'single_exp'(单指数)
        
    Returns:
        力，单位pN
    """
    if model == 'single_exp':
        if params is None:
            params = DEFAULT_SINGLE_EXP_PARAMS
        force = params['a'] * np.exp(params['b'] * height) + params['c']
    else:  # 默认使用双指数模型
        if params is None:
            params = DEFAULT_FORCE_PARAMS
        force = (params['a1'] * np.exp(params['b1'] * height) + 
                params['a2'] * np.exp(params['b2'] * height) + 
                params['c'])
    
    return force

def get_default_parameters(model='double_exp'):
    """
    返回默认参数的副本
    
    Args:
        model: 'double_exp'(双指数)或'single_exp'(单指数)
        
    Returns:
        默认参数字典
    """
    if model == 'single_exp':
        return DEFAULT_SINGLE_EXP_PARAMS.copy()
    else:
        return DEFAULT_FORCE_PARAMS.copy()
