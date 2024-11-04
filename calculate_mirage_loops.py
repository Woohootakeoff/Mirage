import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from functools import reduce

bin_size = 10000 #分辨率
bin_size1 = 10000
bin_size2 = 10000

def find_duplicate_values(a):
    duplicates = set()
    for i in range(len(a)):
        if a.count(a[i]) > 1:
            duplicates.add(a[i])
    return duplicates

def create_new_list(a, duplicates):
    new_list = []
    for value in duplicates:
        for i in range(len(a)):
            if a[i] == value or abs(a[i] - value) <= bin_size1:
                new_list.append(a[i])
    return new_list

def find_multiway_contact_loci(lista):
    duplicate_values = find_duplicate_values([row[0] for row in lista])
    unique_values = create_new_list([row[0] for row in lista], duplicate_values)
    func = lambda x, y: x if y in x else x + [y]
    unique_values = reduce(func, [[], ] + unique_values)
    listb = []
    perm_all = []
    for value in unique_values:
        group = [item[1] for item in lista if item[0] == value]
        perms = list(permutations(group, 2))
        for perm in perms:
            if perm[0] < perm[1]:
                perm_all.append(perm)
                lower_bound_1 = perm[0] - bin_size2
                upper_bound_1 = perm[0] + bin_size2
                lower_bound_2 = perm[1] - bin_size2
                upper_bound_2 = perm[1] + bin_size2
                for item in lista:
                    if lower_bound_1 <= item[0] <= upper_bound_1 and lower_bound_2 <= item[1] <= upper_bound_2:
                        listb.append([item[0], item[1]])
    return listb, perm_all

def find_dup_bin2(data):
    duplicates = {}
    for item in data:
        key = item[0]
        value = item[1]
        if key in duplicates:
            duplicates[key].append(value)
        else:
            duplicates[key] = [value]
    combinations = []
    for key, value in duplicates.items():
        if isinstance(value, list):
            for i in range(len(value) - 1):
                for j in range(i + 1, len(value)):
                    combinations.append((value[i], value[j]))
        else:
            continue
    return combinations

for i in range(1, 24):
    print("chr" + str(i))
    spritedata = pd.read_csv("reshape.bedgraph", sep='\t', header=None, index_col=False)#bedgraph文件提取出来的bin1_start, bin2_start
    spritedata = spritedata.values.tolist()
    testdata = pd.read_csv("reshape.bedgraph", sep='\t', header=None, index_col=False)
    #testdata2 = pd.read_csv("reshape.bedgraph", sep='\t', header=None, index_col=False)
    testdata = testdata.values.tolist()
    hicdata = pd.read_csv("reshape.bedgraph", sep='\t', header=None, index_col=False)
    #hicdata2 = pd.read_csv("reshape.bedgraph", sep='\t', header=None, index_col=False)
    hicdata = hicdata.values.tolist()

    print("sprite")
    sprite_loci, sprite_perm_all = find_multiway_contact_loci(spritedata)
    print("test")
    test_loci, test_perm_all = find_multiway_contact_loci(testdata)
    print("hic")
    hic_loci, hic_perm_all = find_multiway_contact_loci(hicdata)

    df_hic = pd.read_csv("Hi-C.matrix", sep='\t', header=0, index_col=0)
    x_len = len(df_hic)
    df_sprite = pd.read_csv("sprite.matrix", sep='\t', header=0, index_col=0)
    sprite_len = len(df_sprite)
    df_test = pd.read_csv("Mirage.matrix", sep='\t', header=0, index_col=0)
    test_len = len(df_test)

    for tmp in hic_perm_all:
        tmp_bin1 = tmp[0]
        tmp_bin2 = tmp[1]
        list_sample = []
        for coord_x in range(x_len):
            for coord_y in range(x_len):
                if int(tmp_bin1/bin_size - 4) <= coord_x <= int(tmp_bin1/bin_size + 5) and int(tmp_bin2/bin_size - 4) <= coord_y <= int(tmp_bin2/bin_size + 5):
                    list_sample.append(df_hic.iloc[coord_x][coord_y])
        list_sample = np.array(list_sample)
        list_sample = list_sample.reshape(10, 10)
        target = list_sample[5][5]
        result = np.percentile(list_sample, 90)
        if target >= result:
            #np.savetxt("Hi-C_100window_" + str(tmp_bin1) + "-" + str(tmp_bin2) + ".matrix", list_sample, fmt="%i")
            print("Hi-C_fake_loop:", tmp)
    for sprite_tmp in sprite_perm_all:
        sprite_tmp_bin1 = sprite_tmp[0]
        sprite_tmp_bin2 = sprite_tmp[1]
        sprite_list_sample = []
        for coord_x in range(sprite_len):
            for coord_y in range(sprite_len):
                if int(sprite_tmp_bin1/bin_size - 4) <= coord_x <= int(sprite_tmp_bin1/bin_size + 5) and \
                        int(sprite_tmp_bin2/bin_size - 4) <= coord_y <= int(sprite_tmp_bin2/bin_size + 5):
                    sprite_list_sample.append(df_sprite.iloc[coord_x][coord_y])
        sprite_list_sample = np.array(sprite_list_sample)
        sprite_list_sample = sprite_list_sample.reshape(10, 10)
        sprite_target = sprite_list_sample[5][5]
        sprite_result = np.percentile(sprite_list_sample, 90)
        if sprite_target >= sprite_result:
            #np.savetxt("sprite_100window_" + str(sprite_tmp_bin1) + "-" + str(sprite_tmp_bin2) + ".matrix", sprite_list_sample, fmt="%i")
            print("sprite_fake_loop:", sprite_tmp)
    for test_tmp in test_perm_all:
        test_tmp_bin1 = test_tmp[0]
        test_tmp_bin2 = test_tmp[1]
        test_list_sample = []
        for coord_x in range(test_len):
            for coord_y in range(test_len):
                if int(test_tmp_bin1/bin_size - 4) <= coord_x <= int(test_tmp_bin1/bin_size + 5) and \
                        int(test_tmp_bin2/bin_size - 4) <= coord_y <= int(test_tmp_bin2/bin_size + 5):
                    test_list_sample.append(df_test.iloc[coord_x][coord_y])
        test_list_sample = np.array(test_list_sample)
        test_list_sample = test_list_sample.reshape(10, 10)
        test_target = test_list_sample[5][5]
        test_result = np.percentile(test_list_sample, 90)
        if test_target >= test_result:
            #np.savetxt("Mirage_100window_" + str(test_tmp_bin1) + "-" + str(test_tmp_bin2) + ".matrix", test_list_sample, fmt="%i")
            print("his_fake_loop:", test_tmp)
