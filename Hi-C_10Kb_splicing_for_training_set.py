import pandas as pd
import torch
import numpy as np
import random
import math
import openpyxl
import matplotlib.pyplot as plt
import matplotlib
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE, SGConv, ARMAConv
from utils import *
from skimage.metrics import structural_similarity as ssim
from sklearn.preprocessing import  minmax_scale
import os
from matplotlib.patches import Rectangle
import pandas  as pd


INPUT = 128
WINDOW_SIZE = 10
# matplotlib.use('Agg')
dev = torch.device("cuda")
torch.set_default_tensor_type(torch.FloatTensor)

def sum_row_col(input_matrix):
    return torch.cat((torch.sum(input_matrix, dim=0), torch.sum(input_matrix, dim=1)))

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

class InnerProductDecoder(torch.nn.Module):
    def __init__(self):
        super(InnerProductDecoder, self).__init__()
        self.align = torch.nn.Linear(128, 128)

    def forward(self, z):
        adj = torch.matmul(self.align(z), z.t())
        return adj

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = ARMAConv(in_channels, 2 * out_channels, num_stacks=3, num_layers=3)
        self.conv2 = ARMAConv(2 * out_channels, out_channels, num_stacks=3, num_layers=3)

    def forward(self, x, edge_index, edge_weight):
        output = F.relu(self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight))
        #print(x.size(), edge_index, edge_weight.size())
        return F.relu(self.conv2(x=output, edge_index=edge_index, edge_weight=edge_weight))

class dnn(torch.nn.Module):
    def __init__(self):
        super(dnn, self).__init__()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.encoder = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(in_channels=INPUT, out_channels=4 * INPUT, kernel_size=(1,)),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose1d(in_channels=4 * INPUT, out_channels=8 * INPUT, kernel_size=(1,)),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose1d(in_channels=8 * INPUT, out_channels=16 * INPUT, kernel_size=(1,)),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose1d(in_channels=16 * INPUT, out_channels=64 * INPUT, kernel_size=(1,)),
            torch.nn.ReLU()
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=64 * INPUT, out_channels=16 * INPUT, kernel_size=(1,)),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=16 * INPUT, out_channels=8 * INPUT, kernel_size=(1,)),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=8 * INPUT, out_channels=4 * INPUT, kernel_size=(1,)),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=4 * INPUT, out_channels=INPUT, kernel_size=(1,)),
            torch.nn.ReLU()
        )

    def forward(self, input):
        # i = self.dropout(input)
        encode = self.encoder(input)
        # encode = self.dropout(encode)
        decode = self.decoder(encode)
        return decode


class GAE_DNN(GAE):
    def __init__(self, encoder, decoder, dnn):
        super(GAE_DNN, self).__init__(encoder, decoder)
        self.dnner = dnn
        GAE_DNN.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)
        reset(self.dnner)


def get_weight(point):
    hic8 = hic[point[0]+1:point[0]+WINDOW_SIZE-1, point[1]+1:point[1]+WINDOW_SIZE-1].sum()
    hic10 = hic[point[0]:point[0]+WINDOW_SIZE, point[1]:point[1]+WINDOW_SIZE].sum()
    hic12 = hic[point[0]-1:point[0]+WINDOW_SIZE+1, point[1]-1:point[1]+WINDOW_SIZE+1].sum()
    regress10 = regress.sum()
    regress8 = regress[1:WINDOW_SIZE-1, 1:WINDOW_SIZE-1].sum()
    hicin = hic10 - hic8
    hicout = hic12 - hic10
    spritein = regress10 - regress8
    return hicin/spritein

def mat_traverse(Z):
    rows, cols = Z.shape
    for i in range(rows):
        for j in range(cols):
            contact = Z[i, j]
            if contact < 0:
                Z[i, j] = 0
            elif contact > 0:
                contact = np.around(contact, decimals=-1)
                Z[i, j] = contact
    return Z

def pearson_calculate(x, y):
    """
    --------------------------------
    描述 :   根据行和列和求两个矩阵的相似性
    --------------------------------
    """

    a = sum_row_col(x).numpy()
    b = sum_row_col(y).numpy()
    c = np.corrcoef(a, b)[0][1]
    rangex = np.max(np.array(x))-np.min(np.array(x))
    rangey = np.max(np.array(y))-np.min(np.array(y))
    range = max(rangex, rangey)
    d = ssim(np.array(x), np.array(y), data_range=range)
    return c, d


pearson_list = np.zeros([23, 2])
for chr in range(1, 24):
    v = #设置序号
    hic = torch.load("./output/version" + str(v) + "/chr" + str(chr) + "/hic_no_10kb.pt")
    sprite = torch.load("./output/version" + str(v) + "/chr" + str(chr) + "/sprite_no_10kb.pt")
    mul = hic.sum() / sprite.sum()
    x = torch.load("./output/version" + str(v) + "/chr" + str(chr) + "/x_sample_10kb.pt")
    y = torch.load("./output/version" + str(v) + "/chr" + str(chr) + "/y_sample_10kb.pt")
    mul1 = x.sum() / y.sum()
    feature = torch.load("./output/version" + str(v) + "/chr" + str(chr) + "/feature_10kb.pt")
    index = torch.load("./output/version" + str(v) + "/chr" + str(chr) + "/sample_index_10kb.pt")
    hic_promax = hic.clone().detach()
    hic_promax_1 = hic.clone().detach()

    model = torch.load("./output/version" + str(v) + "/chr" + str(chr) + "/epoch_300_conv1d_ARMA_10kb_out_128.pkl")
    model.eval()

    ones = torch.ones(WINDOW_SIZE, WINDOW_SIZE)
    zeros = torch.zeros(WINDOW_SIZE, WINDOW_SIZE)
    count = torch.zeros(feature.size(2), feature.size(2))
    value = torch.zeros(feature.size(2), feature.size(2))

    for i in range(sample_num):
        # print('regress_Z获取进度 {:.3f}%'.format(i*100/sample_num))
        node_feature = feature[i]
        edge_index = torch.nonzero(x[i]).long().t()
        edge_weight = x[i][edge_index[0], edge_index[1]]

        regress = model.module.encoder(node_feature.float().to(dev), edge_index.long().to(dev), edge_weight.float().to(dev))
        regress = model.module.dnner(regress.view(1, regress.size(0), INPUT).permute(0, 2, 1)).permute(0, 2, 1).view(regress.size(0), INPUT)
        regress = model.module.decoder(regress).detach().cpu()[WINDOW_SIZE:WINDOW_SIZE*2, 0:WINDOW_SIZE]

        xsample = x[i][WINDOW_SIZE:WINDOW_SIZE*2, 0:WINDOW_SIZE]
        ysample = y[i][WINDOW_SIZE:WINDOW_SIZE*2, 0:WINDOW_SIZE]

        hic_promax[index[i][0]:index[i][0] + WINDOW_SIZE, index[i][1]:index[i][1] + WINDOW_SIZE] = zeros
        mulp = get_weight(index[i])
        value[index[i][0]:index[i][0] + WINDOW_SIZE, index[i][1]:index[i][1] + WINDOW_SIZE] += regress * mulp
        count[index[i][0]:index[i][0] + WINDOW_SIZE, index[i][1]:index[i][1] + WINDOW_SIZE] += ones

    count[count == 0] += 1
    count = count ** -1
    value = value * count
    hic_promax += value
    p_predict_y, s_predict_y = pearson_calculate(hic_promax_1, hic_promax)

    pearson_list[chr][0] = p_predict_y
    pearson_list[chr][1] = s_predict_y

    Z = hic_print.numpy()
    Z = np.triu(Z)
    Z += Z.T - np.diag(Z.diagonal())
    ZZ = mat_traverse(Z)
    np.savetxt("./output/version" + str(v) + "/chr" + str(chr) + '/his.txt', delimiter="\t", fmt='%.6f', X=Z)
    np.savetxt("./output/version" + str(v) + "/chr" + str(chr) + '/his_int.txt', delimiter="\t", fmt='%i', X=ZZ)

    print("结束")
    print("chr" + str(chr))
np.savetxt("./output/version" + str(V) + "/pearson_list.txt", fmt='%f', X=pearson_list)
