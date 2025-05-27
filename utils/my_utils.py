# -*- coding: utf-8 -*-
# @Time : 2024/3/15 19:20
# @Author : Xiao Han
# @E-mail : hahahenha@gmail.com
# @Site : 
# @project: GLM
# @File : my_utils.py
# @Software: PyCharm
from random import shuffle

from time import sleep
import shutil
import os
from collections import defaultdict
import math
import scipy.sparse as sp
import torch
import numpy as np
import re

def add_to_file(lock_file_path, file_path, content):
    cnt = 0
    while(True):
        if lock_file_check(file_path =lock_file_path):
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(content)
            lock_remove(file_path=lock_file_path)
            flag = True
            break
        sleep(5)
        cnt += 1
        if cnt % 1000 == 999:
            print('Write to file error! please check the lock file:', lock_file_path)


def lock_file_check(file_path):
    # Check if the file exists
    if os.path.exists(file_path):
        return False
    else:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Create the file
        with open(file_path, 'w') as file:
            pass  # Creating an empty file
        return True

def lock_remove(file_path):
    try:
        os.remove(file_path)
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"An error occurred while trying to delete the file {file_path}: {e}")
        return False


def find_jpg_files(directory_path):
    """
    Check the specified directory for any .jpg files.
    If found, returns a list of the names of these files.
    """
    # Initialize a list to hold the names of jpg files
    jpg_file_names = []

    # Loop through each file in the directory
    for file in os.listdir(directory_path):
        # Check if the file is a jpg file
        if file.endswith(".jpg"):
            jpg_file_names.append(file)
    
    return jpg_file_names

def clear_directory(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.remove(item_path)  # 删除文件或链接
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  # 删除文件夹及其所有内容
    # print(f"All files and folders in '{directory}' have been deleted.")

def substring_after_char(s, char):
    # 找到特定字符的索引
    index = s.find(char)
    
    # 如果找到了字符，则截取从开始到该字符的子字符串
    if index != -1:
        sub_s = s[index:]
        if len(sub_s) == 1:
            return 'None'
        else:
            return sub_s[1:]
    else:
        # 如果未找到字符，返回原始字符串
        return s

def substring_before_char(s, char):
    # 找到特定字符的索引
    index = s.find(char)
    
    # 如果找到了字符，则截取从开始到该字符的子字符串
    if index != -1:
        return s[:index + 1]
    else:
        # 如果未找到字符，返回原始字符串
        return s

def extract_after_substring(s, substring):
    # 检查子串是否在字符串中
    index = s.find(substring)
    if index != -1:
        # 找到子串，截取子串之后的所有字符
        return s[index + len(substring):]
    else:
        # 子串不在字符串中，返回原字符串
        return s

def remove_trailing_substring(s, substring='.'):
    num = int(-1 * len(substring))
    if s.endswith(substring):
        return s[:num]
    else:
        return s

def dict_sort_by_value(myDict):
    # 按值从大到小排序，相同值的记录打乱顺序
    # 先按value分组
    groups = defaultdict(list)
    for key, value in myDict.items():
        groups[value].append(key)

    # 对每个分组内的元素打乱顺序
    for group in groups.values():
        shuffle(group)
    # 将分组按value从大到小排序并展开
    sorted_myDict = sum([[(key, value) for key in groups[value]] for value in sorted(groups.keys(), reverse=True)], [])

    return sorted_myDict

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """将scipy稀疏矩阵转换为torch稀疏张量"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def extract_number_before_percent(s):
    """
    从含有 '%' 的字符串中提取 '%' 前的数字（允许浮点数）。

    参数:
    s (str): 输入的字符串。

    返回:
    str: '%' 前的数字字符串。如果没有找到合适的数字，返回 None。
    """
    # 使用正则表达式匹配 '%' 前的数字（包括小数点）
    match = re.search(r'[\d.]+(?=%)', str(s))
    if match:
        return float(match.group())
    else:
        return 0