# load tasks from different benches
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *



# Analysis functions
def sample_from_behavior_1k(num, save_path='data/existing_benches/behavior-1k.csv'):
    path = 'data/existing_benches/BEHAVIOR-1K Tasks.csv'
    data = load_csv(path)
    data['Task Name'].sample(num).to_csv(save_path)
    return 

def calculate_difficulties(dic_ab, data):
    dic = {}
    for task, abilities in dic_ab.items():
        dic[task] = sum([data[ability-1] for ability in abilities])/len(abilities)

    return dic

def assess_our_bench(ab_mass):
    path = 'data/source/AK_marked_v4.xlsx'
    data = load_xlsx(path)
    TA_models = 'GPT4o'
    extract_task_ability_pair(data, TA_models)
    dic = extract_task_ability_pair(data, TA_models)
    
    diff = calculate_difficulties(dic, ab_mass)
    
    return diff
    
def assess_sample_bench(ab_mass):
    s, _ = read_result()
    s = np.array(s)
    s -= s.min()
    s += 1e-4
    diff = {}
    for id, si in enumerate(s):
        diff[id] = si
        
    return diff
          
def assess_behavior(ab_mass):
    path1k = 'data/existing_benches/behavior-1k.csv'
    path100 = 'data/existing_benches/behavior-100.xlsx'
    
    data1k = load_csv(path1k)
    data100 = load_xlsx(path100)
    
    TA_model = 'GPT4o'
    dic1k = extract_task_ability_pair(data1k, TA_model)
    dic100 = extract_task_ability_pair(data100, TA_model)
    
    diff_1k = calculate_difficulties(dic1k, ab_mass)
    diff_100 = calculate_difficulties(dic100, ab_mass)
    
    return diff_1k, diff_100

def assess_hiphy_bench(ab_mass):
    path = 'data/existing_benches/Hi-Phy.xlsx'
    data = load_xlsx(path)
    
    TA_models = 'GPT4o'
    extract_task_ability_pair(data, TA_models)
    dic = extract_task_ability_pair(data, TA_models)
    
    diff = calculate_difficulties(dic, ab_mass)
    
    return diff
# ---------------------------------------------------------------

# functions for plotting
def generate_ass_dic_array():
    ab_mass = np.load(f"results/FA_GPT4o.npy")
    dic_1k, dic_100 = assess_behavior(ab_mass)
    dic_our = assess_our_bench(ab_mass)
    dic_sample = assess_sample_bench(ab_mass)
    dic_hiphy = assess_hiphy_bench(ab_mass)
    dic = {}
    # dic['AGI-V70(sample)'] = [val for val in dic_sample.values()]
    dic['AGI-V70'] = [val for val in dic_our.values()]
    dic['BEHAVIOR-1k'] = [val for val in dic_1k.values()]
    dic['BEHAVIOR-100'] = [val for val in dic_100.values()]
    dic['Hi-Phy'] =[val for val in dic_hiphy.values()]
    
    return dic

def plot_advanced_data(input_data):
    sns.set(style="darkgrid")
    keys = list(input_data.keys())
    data = [input_data[key] for key in keys]
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create box plot with colored boxes and black edges
    boxplot = sns.boxplot(data=data, ax=ax, width=0.5, showcaps=True, showfliers=False, 
                          boxprops=dict(edgecolor='black', linewidth=1.5),
                          whiskerprops=dict(color='black', linewidth=1.5),
                          capprops=dict(color='black', linewidth=1.5),
                          medianprops=dict(color='black', linewidth=2),
                          color='white'
                          )

    # Manually color the boxes
    for patch in boxplot.artists:
        patch.set_facecolor('white')
        patch.set_edgecolor('black')


    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, fontsize=17)
    
    ax.set_yticks(np.arange(0, 1.6, 0.2))
    ax.set_yticklabels([f'{val:.1f}' for val in np.arange(0, 1.6, 0.2)], fontsize=17)

    ax.set_ylabel('Difficulties', fontsize=20)
    ax.set_ylabel('Proportions', fontsize=20)
    ax.set_ylim(0, 1.08)
    plt.tight_layout()
    plt.savefig('figs/advanced_data.png')
    # plt.show()

def plot_ability_distributions(distributions):
    sns.set(style="darkgrid")
    keys = list(distributions.keys())
    values = list(distributions.values())

    fig, axs = plt.subplots(1, len(keys), figsize=(18, 6), sharey=True)
    for i, (key, val) in enumerate(distributions.items()):
        ax = axs[i]
        indices = np.arange(len(val))

        # Bar plot
        bars = ax.bar(indices, val, alpha=0.7, label=key, color='white')
        for patch in bars.patches:
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)

        # Connect the top of each bar with a line
        ax.plot(indices, val, color='black', marker='o', linestyle='-', linewidth=2, markersize=5)

        # Calculate and display the variance
        variance = np.var(val)
        # ax.text(0.5, 0.9, f'Variance: {variance:.4f}', transform=ax.transAxes, ha='center', va='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.7))

        ax.set_xticks(indices)
        # ax.set_xticklabels([f'Ability {j+1}' for j in range(len(val))])
        # ax.set_xlabel('Abilities')
        ax.set_yticks(np.arange(0, 0.51, 0.15))
        ax.set_yticklabels([f'{val:.1f}' for val in np.arange(0, 0.51, 0.15)], fontsize=17)

        
        # set title at the bottom of each subfig
        ax.set_title(key, fontsize=20)
        
        if i == 0:
            ax.set_ylabel('Proportions', fontsize=16)
    plt.tight_layout()
    plt.savefig('figs/ability_distributions.png')
    # plt.show()

def ability_distributions():
    path_100 = 'data/existing_benches/behavior-100.xlsx'
    path_1k = 'data/existing_benches/behavior-1k.csv'
    path_our = 'data/source/AK_marked_v4.xlsx'
    path_hiphy = 'data/existing_benches/Hi-Phy.xlsx'
    
    data_100 = load_xlsx(path_100)
    data_1k = load_csv(path_1k)
    data_our = load_xlsx(path_our)
    data_hiphy = load_xlsx(path_hiphy)
    
    abilities_1k = extract_task_ability_pair(data_1k)
    abilities_100 = extract_task_ability_pair(data_100)
    abilities_our = extract_task_ability_pair(data_our)
    abilities_hiphy = extract_task_ability_pair(data_hiphy)
    
    abdis_1k = np.zeros(5)
    abdis_100 = np.zeros(5)
    abdis_our = np.zeros(5)
    abdis_hiphy = np.zeros(5)
    
    
    for key, val in abilities_1k.items():
        for v in val:
            abdis_1k[v-1] += 1
            
    for key, val in abilities_100.items():
        for v in val:
            abdis_100[v-1] += 1
            
    for key, val in abilities_our.items():
        for v in val:
            abdis_our[v-1] += 1
            
    for key, val in abilities_hiphy.items():
        for v in val:
            abdis_hiphy[v-1] += 1

    # normalize the distribution
    abdis_1k = abdis_1k / abdis_1k.sum()
    abdis_100 = abdis_100 / abdis_100.sum()
    abdis_our = abdis_our / abdis_our.sum()
    abdis_hiphy = abdis_hiphy / abdis_hiphy.sum()
    
    dic = {'AGI-V70': abdis_our, 'BEHAVIOR-1k': abdis_1k, 'BEHAVIOR-100': abdis_100, 'Hi-Phy': abdis_hiphy}
    
    return dic
# ---------------------------------------------------------------


# get figure in paper: Figure 5b and 5c
def get_Figure_5b():
    distributions = ability_distributions()
    plot_ability_distributions(distributions)

def get_Figure_5c():
    # dic = generate_ass_dic()
    dic = generate_ass_dic_array()
    
    # plot_data(dic)
    plot_advanced_data(dic)


if __name__ == '__main__':
    get_Figure_5b()
    get_Figure_5c()
    
