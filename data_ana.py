import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import *
import seaborn as sns
from argparse import ArgumentParser

args = ArgumentParser()
args.add_argument('--exp_id', type=int, default=0)



def check_data(data):
    failed = []
    passed = []
    for row in data.iterrows():
        if row[0] in [0, 1]: continue
        if row[1]['Q13'] != 'task A and B are equally difficult' or row[1]['Q13'] is None:
            failed.append(row[1]['Q3'])
        else:
            passed.append(row[1]['Q3'])
    print(f"Failure check done, failed: {len(failed)}")
    return failed, passed

def read_data(data):
    dic = {}
    task_A_list = []
    task_B_list = []
    choices_list = []
    for row in data.iterrows():
        if row[0] in [0, 1]: continue
        if row[1]['Q13'] != 'task A and B are equally difficult' or row[1]['Q13'] is None: continue
        task_A_list = [int(val) for val in row[1]['task_A_id_list'].split(',')]
        task_B_list = [int(val) for val in row[1]['task_B_id_list'].split(',')]
        choices_list = [int(val) for val in row[1]['choices'].split(',')]
        # print(len(task_A_list), len(task_B_list), len(choices_list))
        for a, b, c in zip(task_A_list, task_B_list, choices_list):
            switch = False
            if a<b:
                task_pair = (a, b)
            else:
                task_pair = (b, a)
                switch = True
            if task_pair not in dic:
                dic[task_pair] = [0, 0, 0, 0]
            
            if not switch:
                dic[task_pair][c-1] += 1
            
            else:
                if c == 0:
                    dic[task_pair][1] += 1
                elif c == 1:
                    dic[task_pair][0] += 1
                else:
                    dic[task_pair][2] += 1 

            dic[task_pair][3] += 1
    return dic

def read_valid_data(data):
    dic = {}
    task_A_list = []
    task_B_list = []
    choices_list = []
    for row in data.iterrows():
        if row[0] in [0, 1]: continue
        if row[1]['Q13'] != 'task A and B are equally difficult' or row[1]['Q13'] is None: continue
        task_A_list = [int(val) for val in row[1]['task_A_id_list'].split(',')]
        task_B_list = [int(val) for val in row[1]['task_B_id_list'].split(',')]
        choices_list = [int(val) for val in row[1]['choices'].split(',')]

        # print(len(task_A_list), len(task_B_list), len(choices_list))
        for a, b, c in zip(task_A_list, task_B_list, choices_list):
            task_pair = (a, b)
            # print(task_pair, c)
            if task_pair not in dic:
                dic[task_pair] = [0, 0, 0, 0]
            if c == 0:
                dic[task_pair][1] += 1
            elif c == 1:
                dic[task_pair][0] += 1
            else:
                dic[task_pair][2] += 1 

            dic[task_pair][3] += 1
    return dic

# A method to check the comparison times(If all the task pairs are compared)
def check_task_pairs(dic):
    # return: task_pairs with comparison times, uncompared task_pairs
    un_compared = []
    for i in range(69):
        for j in range(i, 70):
            if i == j:
                continue
            if (i, j) not in dic:
                un_compared.append((i, j))

    # average, max compare times:
    compare_times = []
    for key, val in dic.items():
        compare_times.append(val[3])
    print('average compare times:', np.mean(compare_times))
    print('max compare times:', np.max(compare_times))
    print('uncompared task pairs:', len(un_compared))
    return compare_times, un_compared

def count_compare_times(dic, un_compared):
    compare_times = {}
    compare_times[0] = un_compared
    for key, val in dic.items():
        if val[3] not in compare_times:
            compare_times[val[3]] = [key]
        else:
            compare_times[val[3]] += [key]

    # remove the key 37

    compare_times.pop(37)
    compare_times.pop(0)
    # plot the resuls
    sns.set(style="darkgrid")
    plt.figure(figsize=(10, 5))
    plt.bar(compare_times.keys(), [len(val) for val in compare_times.values()])
    # mark the value on the bar
    for a, b in zip(compare_times.keys(), [len(val) for val in compare_times.values()]):
        plt.text(a, b, '%d' % b, ha='center', va='bottom', fontsize=10)
    plt.xlabel('compare times', fontsize=18)
    plt.ylabel('task pairs', fontsize=18)
    save_path = 'figs/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.tight_layout()
    plt.savefig(save_path + 'compare_times_distribution.png')
    plt.show()
    return compare_times


if __name__ == '__main__':  
    path = 'data/qs/combined_2.18.csv'
    data = load_csv(path) 
    dic = read_data(data) 
    compare_times, un_compared = check_task_pairs(dic)
    count_compare_times(dic, un_compared)

 