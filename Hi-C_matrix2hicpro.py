import numpy as np
import os
import scipy.sparse as ss

for k in range(1, 24):
    matrix = np.loadtxt("Hi-C.txt")#载入Hi-C大矩阵
    mat = np.zeros((len(matrix.nonzero()[0]), 3))
    cnt = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            if matrix[i][j] != 0:
                mat[cnt][0] = i + 1
                mat[cnt][1] = j + 1
                mat[cnt][2] = matrix[i][j]
                cnt += 1
    #np.savetxt("Hi-C.hicpro", mat, fmt=['%d', '%d', '%f'], delimiter='\t')
    np.savetxt("Hi-C.hicpro", mat, fmt=['%d', '%d', '%i'], delimiter='\t')
