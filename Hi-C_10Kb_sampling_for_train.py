import os.path as osp
import random
import math
import random
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch import nn

import cv2
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE, SGConv
import time
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

KERNEL_SIZE = 10
STRIDE = 5
WINDOW = 100
WINDOW_STRIDE = 50
BRINK = 2
THRESHOLD = 2
# DIAGONAL = 5


torch.set_default_tensor_type(torch.DoubleTensor)
dev = torch.device("cuda")


def get_point_line_distance(point, line):
    point_x = point[0]
    point_y = point[1]
    line_s_x = line[0][0]
    line_s_y = line[0][1]
    line_e_x = line[1][0]
    line_e_y = line[1][1]
#若直线与y轴平行，则距离为点的x坐标与直线上任意一点的x坐标差值的绝对值
    if line_e_x - line_s_x == 0:
        return math.fabs(point_x - line_s_x)
#若直线与x轴平行，则距离为点的y坐标与直线上任意一点的y坐标差值的绝对值
    if line_e_y - line_s_y == 0:
        return math.fabs(point_y - line_s_y)
#斜率
    k = (line_e_y - line_s_y) / (line_e_x - line_s_x)
#截距
    b = line_s_y - k * line_s_x
#带入公式得到距离dis
    dis = math.fabs(k * point_x - point_y + b) / math.pow(k * k + 1, 0.5)
    return dis


def canny_continuous_filter(matrix, value):
    # 岛屿算法
    # value = THRESHOLD

    def dfs(i, j, len):
        nonlocal count
        if not 0 <= i < len or not 0 <= j < len or m[i][j] == 0:
            return False
        count += 1
        m[i][j] = 0
        dfs(i + 1, j, len)
        dfs(i, j + 1, len)
        dfs(i - 1, j, len)
        dfs(i, j - 1, len)
        if count >= value:
            return True
        return False

    m = torch.clone(matrix)
    len = m.size(0)
    for i in range(len):
        for j in range(len):
            if m[i][j] == 255:
                count = 0
                tmp = dfs(i, j, len)
                if tmp:
                    return tmp
    return False


def subgraph_create_sample(hic, sprite, start_row_index, start_col_index):
    hic_matrix = hic
    # 边缘检测获得二值检测图
    # 压缩至0-255
    img = hic_matrix.numpy().astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    canny = cv2.Canny(img, 75, 150)    # 去对角线
    canny2 = canny
    canny = canny.astype(np.float32)
    # 滑动窗口获得样本(torch函数)

    stride = STRIDE
    kernel_size = KERNEL_SIZE
    canny_torch = torch.from_numpy(canny)
    canny_torch = canny_torch.view(1, 1, 100, 100)
    slide_window = torch.nn.functional.unfold(canny_torch, kernel_size=kernel_size, dilation=1, stride=stride)
    B, C_kh_kw, L = slide_window.size()
    slide_window = slide_window.permute(0, 2, 1)
    slide_window = slide_window.view(B, L, -1, kernel_size, kernel_size)
    slide_window = slide_window.squeeze(0)
    slide_window = slide_window.squeeze(1)
    # silde_window_size n*n*kernel_size*kernel_size
    row_num = int(math.sqrt(L))
    slide_window = slide_window.view(row_num, row_num, kernel_size, kernel_size)
    # 对获得的样本进行筛选，返回可以选取的坐标
    img_file = ""#canny_保存
    numble = np.random.rand()

    if not os.path.exists(img_file):
        os.mkdir(img_file)
    index = [[], []]
    for i in range(row_num):
        for j in range(row_num):
            brink = BRINK
            value = THRESHOLD
            if start_row_index + i * stride + kernel_size < start_col_index + j * stride and slide_window[i][j].sum() >= brink * 255 and canny_continuous_filter(slide_window[i][j], value):
                dis = np.array([get_point_line_distance((start_row_index + i * stride, start_col_index + j * stride), [[0, 0], [1, 1]])])
                if dis > 30:
                    index[0].append(i)
                    index[1].append(j)
                    cv2.imwrite(img_file + "" + str(numble) + ".png", canny2)
                elif 30 > dis > 20:
                    value = 2 * value
                    brink = 2 * brink
                    if slide_window[i][j].sum() >= brink * 255 and canny_continuous_filter(slide_window[i][j], value):
                        index[0].append(i)
                        index[1].append(j)
                        cv2.imwrite(img_file + "" + str(numble) + ".png", canny2)
                elif dis < 20:
                    value = 4 * value
                    brink = 4 * brink
                    if slide_window[i][j].sum() >= brink * 255 and canny_continuous_filter(slide_window[i][j], value):
                        index[0].append(i)
                        index[1].append(j)
                        cv2.imwrite(img_file + "" + str(numble) + ".png", canny2)

    sample_num = len(index[0])
    index = torch.Tensor(index) * stride
    index = index.int()
    index[0] += start_row_index
    index[1] += start_col_index

    return index


# 将节点特征的坐标转换为矩阵
def feature_index_trans_matrix(node_feature, feature_index, sample_num, kernel_size):
    feature = torch.zeros(sample_num, kernel_size * 2, node_feature.size(0))
    feature_index = feature_index.int().numpy()
    for i in range(sample_num):
        row_feature = node_feature[feature_index[0][i]: feature_index[0][i] + kernel_size]
        col_feature = node_feature[feature_index[1][i]: feature_index[1][i] + kernel_size]
        feature[i] = torch.cat((col_feature, row_feature), 0)
    return feature


# 将样本拼接成对称矩阵
def sample_combine(hic_sample, sprite_sample, sample_num, kernel_size):
    l = kernel_size
    hic_combine_sample = torch.zeros(sample_num, l * 2, l * 2)
    sprite_combine_sample = torch.zeros(sample_num, l * 2, l * 2)
    for i in range(sample_num):
        hic_combine_sample[i][l:l * 2, 0:l] = hic_sample[i]
        hic_combine_sample[i][0:l, l:l * 2] = hic_sample[i].T
        sprite_combine_sample[i][l:l * 2, 0:l] = sprite_sample[i]
        sprite_combine_sample[i][0:l, l:l * 2] = sprite_sample[i].T
    return hic_combine_sample, sprite_combine_sample


def all_create_sample():
    # 将整个图分成多个子图
    print("划分样本")
    x_sample = torch.ones(0, KERNEL_SIZE * 2, KERNEL_SIZE * 2)
    y_sample = torch.ones(0, KERNEL_SIZE * 2, KERNEL_SIZE * 2)
    all_feature = torch.ones(0, KERNEL_SIZE * 2, node_num)
    sample_index = torch.ones(2, 0)
    # hic划分子图
    hic = hic_adjacent_matrix.view(1, 1, node_num, node_num)
    hic_window = torch.nn.functional.unfold(hic, kernel_size=WINDOW, dilation=1, stride=WINDOW_STRIDE)
    B, C_kh_kw, L = hic_window.size()
    hic_window = hic_window.permute(0, 2, 1)
    hic_window = hic_window.view(B, L, -1, WINDOW, WINDOW)
    hic_window = hic_window.squeeze(0)
    hic_window = hic_window.squeeze(1)
    row_num = int(math.sqrt(L))
    hic_window = hic_window.view(row_num, row_num, WINDOW, WINDOW)
    # sprite划分子图
    sprite = sprite_adjacent_matrix.view(1, 1, node_num, node_num)
    sprite_window = torch.nn.functional.unfold(sprite, kernel_size=WINDOW, dilation=1, stride=WINDOW_STRIDE)
    B, C_kh_kw, L = sprite_window.size()
    sprite_window = sprite_window.permute(0, 2, 1)
    sprite_window = sprite_window.view(B, L, -1, WINDOW, WINDOW)
    sprite_window = sprite_window.squeeze(0)
    sprite_window = sprite_window.squeeze(1)
    row_num = int(math.sqrt(L))
    sprite_window = sprite_window.view(row_num, row_num, WINDOW, WINDOW)
    # 将每个图继续划分小子图，收集符合条件子图左上角坐标
    slide_index = [[], []]
    print(row_num)
    for i in range(row_num):
        for j in range(row_num):
            if i > j:
                continue
            node_feature_index = subgraph_create_sample(hic_window[i][j], sprite_window[i][j], i * WINDOW_STRIDE, j * WINDOW_STRIDE)
            sample_index = torch.cat((sample_index, node_feature_index), 1)

            zero_list = [i] * node_feature_index.size(1)
            one_list = [j] * node_feature_index.size(1)
            slide_index[0].extend(zero_list)
            slide_index[1].extend(one_list)

        print('{}划分样本进度：{}%, 样本数：{}'.format(chr, int(i * 100 / row_num), sample_index.size(1)))

    # index去重
    sample_index_np = pd.DataFrame(sample_index.long().t().numpy())
    dupes = sample_index_np.duplicated()
    slide_index_np = np.array(slide_index).T
    final_slide_index = slide_index_np[~dupes]
    slide_index_torch = torch.from_numpy(final_slide_index.T).long()

    sample_index = torch.tensor(np.array(pd.DataFrame(sample_index.t().numpy()).drop_duplicates())).t().long()
    sample_num = sample_index.size(1)
    # index转换为对应子图
    hic_sample = torch.zeros(sample_num, KERNEL_SIZE, KERNEL_SIZE)
    sprite_sample = torch.zeros(sample_num, KERNEL_SIZE, KERNEL_SIZE)
    print("拼接")
    for i in range(sample_num):
        hic_sample[i] = hic_adjacent_matrix[sample_index[0][i]: sample_index[0][i] + KERNEL_SIZE,
                        sample_index[1][i]: sample_index[1][i] + KERNEL_SIZE]
        sprite_sample[i] = sprite_adjacent_matrix[sample_index[0][i]: sample_index[0][i] + KERNEL_SIZE,
                           sample_index[1][i]: sample_index[1][i] + KERNEL_SIZE]
    # print(node_feature.size())
    # exit(0)
    all_feature = feature_index_trans_matrix(node_feature, sample_index, sample_num, KERNEL_SIZE)
    print(node_feature.size())
    x_sample, y_sample = sample_combine(hic_sample, sprite_sample, sample_num, KERNEL_SIZE)
    print("样本处理完成")
    print('样本数量: {}'.format(x_sample.size(0)))
    return x_sample, y_sample, all_feature, sample_index, slide_index_torch



for chr in range(1, 24):
    v = #设置序号
    pwd = os.getcwd()
    name = pwd + '\\' + 'output\\version' + str(v) + '\\chr' + str(chr)
    if not os.path.exists(name):
        os.makedirs(name)

    print("读取数据")
    df_hic = pd.read_csv("" + str(chr) + ".matrix", sep='\t',header=0, index_col=0) 
    df_sprite = pd.read_csv("" + str(chr) +".matrix", sep='\t', header=0, index_col=0)
    print("数据处理")
    sprite_adjacent_matrix = torch.tensor(np.array(df_sprite.values.astype(float)))
    hic_adjacent_matrix = torch.tensor(np.array(df_hic.values.astype(float)))

    node_num = sprite_adjacent_matrix.size(0)
    node_feature = torch.eye(sprite_adjacent_matrix.size(0))

    torch.save(hic_adjacent_matrix, "./output/version" + str(v) + "/chr" + str(chr) + "/hic_no_10kb.pt")
    torch.save(sprite_adjacent_matrix, "./output/version" + str(v) + "/chr" + str(chr) + "/sprite_no_10kb.pt")
    print("读取完毕")

    x_sample, y_sample, feature, sample_index, slide_index = all_create_sample()
    sample_index = sample_index.t()
    slide_index = slide_index.t()
    perm = torch.randperm(x_sample.size(0))
    x_sample_train, y_sample_train = x_sample[perm], y_sample[perm]
    feature_train = feature[perm]
    sample_index_train = sample_index[perm]
    slide_index_train = slide_index[perm]

    torch.save(x_sample_train, "./output/version" + str(v) + "/chr" + str(chr) + "/x_sample_10kb.pt")
    torch.save(y_sample_train, "./output/version" + str(v) + "/chr" + str(chr) + "/y_sample_10kb.pt")
    torch.save(feature_train, "./output/version" + str(v) + "/chr" + str(chr) + "/feature_10kb.pt")
    torch.save(sample_index_train, "./output/version" + str(v) + "/chr" + str(chr) + "/sample_index_10kb.pt")
    torch.save(slide_index_train, "./output/version" + str(v) + "/chr" + str(chr) + "/slide_index_10kb.pt")
