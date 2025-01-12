# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

# --------------------
# HodgeRanking算法实现|
# --------------------


Task_Num = 70

# 对某一个Y值：Y1，它可能没有比较1，3 但是一旦比较过了8，9，那么Y1[8][9] = 1，Y1[9][8] = 2
def get_pair_matrix_Y(raw_data, abondoned=[]):
    Voter_Num = raw_data.shape[0]
    Y_raw = np.zeros([Voter_Num, Task_Num, Task_Num])
    for index, voters in raw_data.iterrows():
        if voters['Q13'] != 'task A and B are equally difficult':
            # print("failed")
            continue
        task_A_list, task_B_list, choice_list = voters['task_A_id_list'].split(','), voters['task_B_id_list'].split(','), \
                                                voters['choices'].split(',')
        for question_index in range(min(len(task_A_list), 50)):
            # print(question_index, len(task_A_list), len(task_B_list), len(choice_list))
            task_A, task_B, choice = int(task_A_list[question_index]), int(task_B_list[question_index]), int(
                choice_list[question_index])
            if task_A in abondoned or task_B in abondoned:
                continue
            if Y_raw[index][task_A][task_B] == 0:
                Y_raw[index][task_A][task_B] = choice
                Y_raw[index][task_B][task_A] = inverse_choice(choice)
            elif Y_raw[index][task_A][task_B] == choice:
                continue
            else:
                Y_raw[index][task_A][task_B] = 0
                Y_raw[index][task_B][task_A] = 0
    return Y_raw

# choice转换，用于解决taskA taskB与taskB taskA的问题
def inverse_choice(choice):
    choice_new = 0
    if choice == 1:
        choice_new = 2
    elif choice == 2:
        choice_new = 1
    elif choice == 3:
        choice_new = 3
    return choice_new

def get_average_Y(Y_raw):
    # raw_Y是get_pair_matrix_Y(raw_data)输出的统计矩阵，尺寸是(Voter_Num, Task_Num, Task_Num)
    # 经过转置，Y_T的尺寸变为(Task_Num, Task_Num, Voter_Num)
    Y_raw_T = np.transpose(Y_raw, (1, 2, 0))
    # 这里使用了hodge-rank论文P12页关于average pair ranking的第3种计算方式（Binary comparison）
    Y_3_binary_comparison = np.zeros([Task_Num, Task_Num])
    # 第4种计算方式（Logarithmic odds ratio）
    Y_4_log_ratio = np.zeros([Task_Num, Task_Num])

    for i in range(Task_Num):
        for j in range(i + 1, Task_Num):
            Cnt_i_bigger_than_j = np.sum(np.equal(Y_raw_T[i][j], 1))
            Cnt_i_smaller_than_j = np.sum(np.equal(Y_raw_T[i][j], 2))
            Cnt_i_qual_j = np.sum(np.equal(Y_raw_T[i][j], 3))

            # 这里没有考虑i和j没有被比较过的情形。
            if Cnt_i_bigger_than_j + Cnt_i_smaller_than_j == 0:
                continue
                # print(f"Cnt_i_bigger_than_j + Cnt_i_smaller_than_j ==0 for i={i} and j={j}")
            else:
                Pr_i_bigger_than_j = Cnt_i_bigger_than_j / (Cnt_i_bigger_than_j + Cnt_i_smaller_than_j)
                Pr_j_bigger_than_i = Cnt_i_smaller_than_j / (Cnt_i_bigger_than_j + Cnt_i_smaller_than_j)
                Y_3_binary_comparison[i][j] = Pr_j_bigger_than_i - Pr_i_bigger_than_j
                Y_3_binary_comparison[j][i] = -Y_3_binary_comparison[i][j] # Y是一个反对称矩阵

    return Y_3_binary_comparison, Y_4_log_ratio

# w_{i,j}^{α}取值为0或1
# W_{i,j}取值为w_{i,j}^{α}之和
def get_weight(Y_raw):
    W = np.zeros([Task_Num, Task_Num])
    Y_T = np.transpose(Y_raw, (1, 2, 0))
    Voter_Num = Y_T.shape[2]
    for i in range(Task_Num):
        for j in range(i + 1, Task_Num):
            cnt_0 = np.sum(np.equal(Y_T[i][j], 0))
            # if cnt_0 == Voter_Num:
            #     print(f"cnt_0 == Voter_Num for task {i} and task {j}")
            W[i][j] = Voter_Num - cnt_0
            # 考虑到在Y_3计算中，舍弃了task[i] == task[j]的情况，所以权重W也应舍弃相应的取值
            cnt_3 = np.sum(np.equal(Y_T[i][j], 3))
            W[i][j] = W[i][j] - cnt_3
            W[j][i] = W[i][j]  # W是对称的非负值矩阵
    return W


# -div 是 grad的逆运算
# 基于定义(grad s, Y) = (s, -div Y) 可以求出div算子的计算公式
def get_div_Y(Y, W):
    div = np.zeros(Task_Num)
    for i in range(Task_Num):
        div[i] = np.dot(W[i], Y[i])
    return div


## 这里其实是拉普拉斯算子，是div与grad的复合运算
## 基于div和grad的定义，可以求出复合算子的计算公式，可以证明论文中的公式有误
def get_delta_0(W):
    delta = np.zeros([Task_Num, Task_Num])
    for i in range(Task_Num):
        for j in range(Task_Num):
            if i == j:
                delta[i][j] = np.sum(W[i])  # 求取dii行的和，而不是矩阵的迹
            elif W[i][j] != 0:
                delta[i][j] = -W[i][j]
            else:
                delta[i][j] = 0
    return delta


# 将序关系s向量转换为两两比较的pairwise ranking matrix
def grad(s):
    M = np.zeros([len(s), len(s)])
    for i in range(len(s)):
        for j in range(len(s)):
            M[i][j] = s[j] - s[i]
    return M


# 根据hodge分解定理推出的定理3（论文P25页），求得最小二乘解
def get_hodge_solution(Y, W):
    delta = get_delta_0(W)
    inverse_delta = np.linalg.pinv(delta)
    div = get_div_Y(Y, W)
    s = -np.dot(inverse_delta, div)
    residual = Y - grad(s)
    return s, residual


# 计算不一致性指数，包括两个指数：
# cyclicity ratio: Cp, mesure of global inconsistency
# relative curl: Cr, quantifies the local inconsistency (in edges and triangles)
def get_inconsistency_index(Y, s, R, W):
    Cp = get_weighted_l2_X(R, W) / get_weighted_l2_X(Y, W)
    Cr = np.zeros([Task_Num, Task_Num, Task_Num])
    for i in range(Task_Num):
        for j in range(Task_Num):
            for k in range(Task_Num):
                Cr[i][j][k] = (Y[i][j] + Y[j][k] + Y[k][i]) / (3 * (s[j] - s[i]))
    return Cp, Cr


def get_weighted_l2_X_Y(X, Y, W):
    ans = 0
    for i in range(Task_Num):
        for j in range(Task_Num):
            ans += W[i][j] * (X[i][j] - Y[i][j]) * (X[i][j] - Y[i][j])
    return ans


def get_weighted_l2_X(X, W):
    ans = 0
    for i in range(Task_Num):
        for j in range(Task_Num):
            ans += W[i][j] * X[i][j] * X[i][j]
    return ans


# 按照从大到小的顺序排序
# 获取原始数组中元素在排序后的位置（也就是序号）
# 例如，arr = [3, 1, 4, 2]，排序后为[4, 3, 2, 1]输出结果为[(0, 1), (1, 3), (2, 0), (3, 2)]
# (0,1)表示原来index是0的元素，在排序后索引位置为1
def get_sort_index_list(arr):
    sorted_indices = sorted(range(len(arr)), key=lambda x: arr[x])
    result = list(zip(sorted_indices, [i for i in range(len(arr))]))
    return result

