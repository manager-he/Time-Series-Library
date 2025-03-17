import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_Stock_day

# 设置参数
args = {
    'augmentation_ratio': 0.0  # 禁用数据增强以简化测试
}

# 初始化数据集
root_path = '../stock_data/train_set'
flag = 'train'
size = [30, 15, 15]  # [seq_len, label_len, pred_len]
features = 'M'  # 使用所有特征
data_path = '碧桂园_stock_filtered_with_news_grade.xlsx'
target = 'rank'
scale = True
timeenc = 1  # 使用 time_features 生成时间特征
freq = 'd'  # 每日频率

dataset = Dataset_Stock_day(
    args=args,
    root_path=root_path,
    flag=flag,
    size=size,
    features=features,
    data_path=data_path,
    target=target,
    scale=scale,
    timeenc=timeenc,
    freq=freq
)

# 1. 数据加载与预处理
print("数据加载与预处理:")
print(f"数据集长度: {len(dataset)}")

# 2. 边界计算
print("\n边界计算:")
print(f"训练集边界: {dataset.border1s[0]} 到 {dataset.border2s[0]}")
print(f"验证集边界: {dataset.border1s[1]} 到 {dataset.border2s[1]}")
print(f"测试集边界: {dataset.border1s[2]} 到 {dataset.border2s[2]}")

# 3. 时间特征生成
print("\n时间特征生成:")
print(f"时间特征形状: {dataset.data_stamp.shape}")

# 4. 数据切片
print("\n数据切片:")
index = 0  # 选择第一个样本进行测试
seq_x, seq_y, seq_x_mark, seq_y_mark = dataset.__getitem__(index)
print(f"seq_x 形状: {seq_x.shape}")
print(f"seq_y 形状: {seq_y.shape}")
print(f"seq_x_mark 形状: {seq_x_mark.shape}")
print(f"seq_y_mark 形状: {seq_y_mark.shape}")

# 5. 数据增强
print("\n数据增强:")
if args['augmentation_ratio'] > 0:
    print("数据增强已启用，检查增强后的数据")
else:
    print("数据增强未启用")

# 6. 数据长度
print("\n数据长度:")
print(f"数据集长度: {len(dataset)}")

# 7. 使用 DataLoader 进行批量加载
print("\n使用 DataLoader 进行批量加载:")
batch_size = 4
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for batch in dataloader:
    seq_x_batch, seq_y_batch, seq_x_mark_batch, seq_y_mark_batch = batch
    print(f"Batch seq_x 形状: {seq_x_batch.shape}")
    print(f"Batch seq_y 形状: {seq_y_batch.shape}")
    print(f"Batch seq_x_mark 形状: {seq_x_mark_batch.shape}")
    print(f"Batch seq_y_mark 形状: {seq_y_mark_batch.shape}")
    break  # 只查看一个批次