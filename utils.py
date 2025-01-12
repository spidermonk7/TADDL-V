import pandas as pd
import numpy as np


a = {
        'Human Label':
        [[0, 3, 4, 6, 8, 10, 12, 13, 16, 36], 
        [1, 15, 17, 20, 22, 23, 26, 27, 28, 33, 47, 51, 52, 53, 56, 59, 66], 
        [2, 5, 9, 11, 14, 18, 19, 25, 30, 34, 40, 41, 42, 44, 45, 46, 54, 60, 63, 65, 67],
        [7, 24, 29, 32, 37, 39, 48, 49, 62, 64, 68], 
        [21, 31, 35, 38, 43, 50, 55, 57, 58, 61, 69]
        ]
        }

def load_xlsx(path):
    data = pd.read_excel(path)
    return data


def load_csv(path):
    return pd.read_csv(path, encoding='utf-8')


def read_result(level_path='./results/AGI-V70_difficulty_levels.txt', grade_path='./results/AGI-V70_difficulty_grades.txt'):
    levels = []
    grades = []
    with open(level_path, 'r') as f:
        for line in f:
            level = line.split(': ')[-1].strip()
            levels.append(eval(level))
            
    with open(grade_path, 'r') as f:
        for line in f:
            grad = line.split(': ')[-1].strip()
            grades.append(eval(grad))
    return levels, grades
    

def extract_task_ability_pair(data, TA_model='GPT4o'):
    dic = {}
    spliter = ' '
    ability_col = f'Required Ability({TA_model})'
    assert TA_model in ['GPT4o', 'GPT3.5', 'GT', 'GPT4']
    if TA_model == 'GPT4o' or TA_model=='GPT3.5':
        spliter = ','
        
    for row in data.iterrows():
        if type(row[1][ability_col]) != str:
            dic[row[1]['index']] = [row[1][ability_col]]
        else:
            dic[row[1]['index']] = [int(val) for val in row[1][ability_col].split(spliter)]

    return dic

def calculate_s(masses, t_a_dic):
    s = np.zeros(len(t_a_dic))
    for task in t_a_dic:
        for ability in t_a_dic[task]:
            s[task] += masses[ability-1]
        s[task] = s[task]/len(t_a_dic[task])
    s = {i: s[i] for i in range(len(s))}
    return s
