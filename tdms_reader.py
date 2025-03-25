# -*- coding: utf-8 -*-
# author: Ye Yang
# Load TDMS files using multiple processes from MT data


import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from nptdms import TdmsFile
from force_models import calculate_force  # 导入集中管理的力计算函数


def read_tdms_chunk(file_path, chunk_start, chunk_end):
    """Read a chunk of a TDMS file."""
    with TdmsFile.open(file_path) as tdms_file:
        data = {}
        for group in tdms_file.groups():
            group_name = group.name
            data[group_name] = {}
            for channel in group.channels():
                channel_data = channel[chunk_start:chunk_end]
                data[group_name][channel.name] = channel_data

    return data


def read_tdms_file(file_path, num_workers=12, need_force=True, force_model='double_exp'):
    """Read a TDMS file using multiple processes."""
    file_size = os.path.getsize(file_path)
    chunk_size = file_size // num_workers
    start_indices = [i * chunk_size for i in range(num_workers)]
    end_indices = [(i + 1) * chunk_size for i in range(num_workers)]
    end_indices[-1] = file_size  # Ensure the last chunk includes the end of the file

    all_data = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for data in executor.map(read_tdms_chunk, [file_path] * num_workers, start_indices, end_indices):
            for group_name, channels in data.items():
                if group_name not in all_data:
                    all_data[group_name] = {}
                for channel_name, channel_data in channels.items():
                    if channel_name not in all_data[group_name]:
                        all_data[group_name][channel_name] = []
                    all_data[group_name][channel_name].extend(channel_data)

    df_list_dic = all_data.get('Measured', {})

    max_len = max(len(v) for v in df_list_dic.values())
    for k in df_list_dic:
        df_list_dic[k] += [np.nan] * (max_len - len(df_list_dic[k]))

    df_list = pd.DataFrame.from_dict(df_list_dic)

    # 只在选择了Force-Extension Analysis和Kinetics Analysis时计算力
    if need_force:
        # Calculate the force from the mag height instead of the force from tdms file
        mag_height_array = df_list['mag height mm'].values
        force_pn_array = calculate_force(mag_height_array, model=force_model)  # 添加模型选择参数
        df_list['force pN'] = force_pn_array

    return df_list
