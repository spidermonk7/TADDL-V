from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from hodge import *
from argparse import ArgumentParser
import os
import pandas as pd
from utils import *
from scipy.stats import gaussian_kde,linregress
from matplotlib.colors import LinearSegmentedColormap
from asses_benches import get_Figure_5b, get_Figure_5c
from data_ana import read_data, read_valid_data
import seaborn as sns
from scipy.optimize import minimize
from scipy.interpolate import make_interp_spline


# set the random seed to 0
def load_config():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--data_file_path', type=str, default='./data/qs/combined_2.18.csv')
    arg_parser.add_argument('--print_details', action='store_true')
    arg_parser.add_argument('--exp_id', type=int, default=0)
    arg_parser.add_argument('--TA_model', type=str, default='modified')
    
    args = arg_parser.parse_args()
    
    assert args.TA_model in ['GPT4o', 'Human_Label', 'GPT3.5', 'GPT4', 'modified']
    
    config = {}
    config['data_file_path'] = args.data_file_path
    config['print_details'] = args.print_details
    config['exp_id'] = args.exp_id
    config['TA_model'] = args.TA_model
    
    

    return config




# Exp[11]: Select the proper cluster number for Task Difficulty Stratification
def get_propoer_cluster_num():
    def elbow_method(X, cluster_nums):
        # try to find a better cluster number
        range_clusters = range(1, cluster_nums)
        inertias = []
        for n_clusters in range_clusters:
            cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
            inertias.append(cluster.inertia_)
                    
        sns.set(style='darkgrid')
        plt.figure(figsize=(10, 6))
        plt.plot(range_clusters, inertias, marker='o')
        
        # mark the k=5
        plt.axvline(x=5, color='r', linestyle='--') 
        plt.xlabel('Number of clusters', fontsize=16)
        plt.ylabel('Inertia', fontsize=16)
    
        if 'figs' not in os.listdir():
            os.makedirs('figs')
        plt.tight_layout()
        plt.savefig('figs/elbow_method.png')
        plt.show()    

    levels, grades = read_result()
    X = np.array(levels).reshape(-1, 1)
    elbow_method(X, 10)

# Run the hodge method to get the difficulty levels 
def run_hodge(data_file_path, print_details=False):
    raw_data_df = pd.read_csv(data_file_path)
    Y_raw = get_pair_matrix_Y(raw_data_df)
    Y_3, Y_4 = get_average_Y(Y_raw)
    W = get_weight(Y_raw)
    Y = Y_3  # 选择用Y_3作为Y_raw的平均值

    # 最小二乘求解
    s, residual = get_hodge_solution(Y, W)
    clusters = None
    if print_details:
        Cp, Cr = get_inconsistency_index(Y, s, residual, W)
        print(f"inconsistency index, Cp: {Cp}")
   
    n_clusters=5
    X = s.reshape(-1, 1)
    # carry out cluster method
    cluster = KMeans(n_clusters=n_clusters,random_state=0).fit(X)
    clusters = [np.where(cluster.labels_==i)[0].tolist() for i in range(n_clusters)]

    # sort the clusters based on the mean of dimension 0
    clusters = sorted(clusters, key=lambda x: np.mean(s[x]))
    
    # save the cluster result to a txt file
    save_path = 'results/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_cluster_path = save_path + 'AGI-V70_difficulty_grades.txt'
    save_difficulty_path = save_path + 'AGI-V70_difficulty_levels.txt'
    with open(save_difficulty_path, 'w') as f:
        for i in range(70):
            f.write(f"{s[i] - min(s) + 1e-4}\n")
    with open(save_cluster_path, 'w') as f:
        for id, c in enumerate(clusters):
            f.write(f"Level {id+1}: {c}\n")

    return s, clusters

# Exp[0]: Get and save the difficulty levels & difficulty grades
def get_difficulty_levels(data_file_path='./data/qs/combined_2.18.csv', print_details=False):
    if print_details:
        print(f"{'='*20} start {'='*20}")
        print(f"reading data from {data_file_path}")
    s, clusters = run_hodge(data_file_path, print_details)
    if print_details:
        for i in range(70):
            print(f"task {i}: {s[i]}")
        print(f"{'-'*20} clusters {'-'*20}")
        if clusters is not None:
            for id, c in enumerate(clusters):
                print(f"Level {id+1}: {c}")
        else:
            print("No clusters found, try set --save_cluster_path to generate and save the cluster map")
        print(f"{'='*20} end {'='*20}")
    return s, clusters

# Exp[1]: Solve the Weighted Average Masses(F(A)) of Ability Sets(Fig2 in the paper)
def solve_FA(TA_model):
    if TA_model == 'Human_Label':
        TA_model = 'GT'
    path = 'data/source/AK_marked_v4.xlsx'
    data = load_xlsx(path)
    # Preperation for solving Eq(7)
    dic = extract_task_ability_pair(data, TA_model=TA_model)
    # s = [-0.6672563874156494, -0.31292387300189234, -0.3747768685453051, -0.5932192896097728, -0.7818293148136716, -0.02886179159615257, -0.6490937158949027, -0.2045583925727929, -0.2952467800424811, 0.5852645447578632, -0.5890594325064331, -0.1365415165807757, -0.21376197747935938, -0.3629847320754949, -0.2891692879577607, -0.1918461300804619, -0.6780938715140045, -0.21764550409761171, 0.11744072985735314, -0.11416079428341515, 0.11624459108826216, -0.022189341374330738, -0.4811964269630615, -0.24809494591937606, -0.13847951332665123, -0.1927882762658033, -0.10666384978543528, 0.535904880394268, 0.20526455106846372, -0.37799363262764824, -0.19404290460012585, 0.3180860654977551, 0.11459142985578243, -0.0621094924379673, -0.3423531836998627, 0.042642763083779844, -0.24158304367757316, -0.03374048325206536, 0.16104833870902016, 0.32960554044363, -0.014450526909995683, -0.2772822392450799, 0.13485003457149833, 0.5638745042864997, 0.1881744668048225, -0.14674471987254925, 0.39189810181512436, 0.26063457674705154, 0.41810876150084486, 0.14630276611672854, 0.5101560204583445, 0.47747962649834974, 0.3005072446882672, 0.43466522040159433, 0.5078018522668665, -0.2766349434812061, -0.0703220846778818, 0.19174126747124035, 0.23372489245246036, -0.34657016733312007, -0.11252057104303362, 0.30586765668253396, 0.7425201825520881, -0.2751066563452655, 0.4545082403049548, 0.24950860741854375, 0.35125151765789975, 0.32470077862257063, 0.3317263230134376, 0.6158005858180643]
    s, _ = read_result()
    s = np.array(s)
    s -= s.min()
    s += 1e-4
    
    
    def loss_function(x, s=s, dic=dic):
        loss = 0
        for i in range(70):
            ability = []
            for j in [1, 2, 3, 4, 5]:
                if j in dic[i]:
                    ability.append(x[j-1])
            # ability_weight = np.mean(ability)
            ability_weight = np.sum(ability)
            loss += np.square(ability_weight - s[i])

        return loss
    
    history = []
    def callback(x):
        f_val = loss_function(x)
        history.append(f_val)
        print(f_val, ',')

    # solve Eq(7)
    initial_guess = [0, 0, 0, 0, 0]
    # Define bounds for each parameter (x1, x2, ..., x5)
    bounds = [(0, None), (0, None), (0, None), (0, None), (0, None)]
    result = minimize(loss_function, initial_guess, method='SLSQP', callback=callback,)
    mass = result.x
    print(f"Using {TA_model} as TA model, we can solve Equation (7), the results are:\n {result.x}")
    if TA_model == 'GT':
        np.save('results/FA_Human_Label.npy', result.x)
    else:
        np.save(f'results/FA_{TA_model}.npy', result.x)

    # ----------------- Plotting Result-----------------
    sns.set(style="darkgrid")
    categories = ['Feature Perception', 'Object Perception', 'Spatial Vision', 'Sequential Vision', 'Visual Reasoning']
    values = result.x
    colors = ['#d7191c', '#fdae61', '#ffffbf', '#abdda4', '#2b83ba']
    colors.reverse()
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(categories, values, color=colors)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 2), ha='center', va='bottom', fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.patch.set_facecolor('none')
    ax.set_ylabel('F(A)', fontsize=12)
    # plt.ylim(0, 1.2)
    # Save and show
    plt.tight_layout()
    plt.savefig(f'figs/FA_{TA_model}.png')
    plt.show()
    plt.close()
    # ----------------- Plotting Loss-----------------
    epochs = list(range(1, len(history)+1))
    sns.set(style="darkgrid")
    plt.figure(figsize=(8, 6))
    plt.title('Loss Curve', fontsize=18)
    plt.plot(epochs[2:], history[2:], marker='D', label='GPT4o',)
    plt.xlabel('Epochs')
    plt.legend()
    plt.gca().patch.set_alpha(1)  # Ensure the plot background is not transparent
    plt.gcf().set_facecolor('none')  # Set the figure background to transparent
    plt.savefig(f'figs/{TA_model}_loss.png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()

# Exp[2]: Calculate the relative differences between the difficulty levels of the tasks
def calculate_relative_difference():
    # load_masses
    if not os.path.exists('results/FA_Human_Label.npy'):
        solve_FA('Human_Label')
        print(f"the masses of Human_Label are not found, we have calculated it and saved it to results/FA_Human_Label.npy")
    if not os.path.exists('results/FA_GPT4o.npy'):
        solve_FA('GPT4o')
        print(f"the masses of GPT4o are not found, we have calculated it and saved it to results/FA_GPT4o.npy")
    
    masses_Human = np.load('results/FA_Human_Label.npy')
    masses_GPT4o = np.load('results/FA_GPT4o.npy')
    
    path = 'data/source/AK_marked_v4.xlsx'
    data = load_xlsx(path)
    
    dic_Human = extract_task_ability_pair(data, TA_model='GT')
    dic_GPT4o = extract_task_ability_pair(data, TA_model='GPT4o')
    
    s_cal_human = calculate_s(masses_Human, dic_Human)
    s_cal_GPT4o = calculate_s(masses_GPT4o, dic_GPT4o)

    s_cal_human = np.array([s_cal_human[i] for i in range(70)])
    s_cal_GPT4o = np.array([s_cal_GPT4o[i] for i in range(70)])


    # save s_cal_GPT4o
    with open('results/AGI-V70_difficulty_grades_GPT4o.txt', 'w') as f:
        for i in range(70):
            f.write(f"{s_cal_GPT4o[i]}\n")

    with open('results/AGI-V70_difficulty_grades_Human_Label.txt', 'w') as f:
        for i in range(70):
            f.write(f"{s_cal_human[i]}\n")
    
    s, _ = read_result()
    s = np.array(s) - min(s)
    
    delta_s_Human = np.abs(s - s_cal_human)/(s + s_cal_human)
    delta_s_GPT4o = np.abs(s - s_cal_GPT4o)/(s + s_cal_GPT4o)


    # sort the x idx based on the s from read_result: from small to big
    idx = np.argsort(s)
    delta_s_Human = delta_s_Human[idx]
    delta_s_GPT4o = delta_s_GPT4o[idx]

    
    return delta_s_Human, delta_s_GPT4o, idx

def relative_difference(delta_s_Human=None, delta_s_GPT4o=None, idx = None):    
    delta_s_Human = np.where(np.isfinite(delta_s_Human), delta_s_Human, np.nan)  # Replace inf with NaN for mean calculation
    delta_s_GPT4o = np.where(np.isfinite(delta_s_GPT4o), delta_s_GPT4o, np.nan)  # Replace inf with NaN for mean calculation    
    
    # Plot the delta values
    plt.figure(figsize=(20, 6))
    sns.set(style="darkgrid")
    plt.plot(delta_s_Human, label='Human Label', marker='o', linestyle='-', color='green')
    plt.axhline(y=np.nanmean(delta_s_Human), color='purple', linestyle='--', label=f'Mean(HL): {np.nanmean(delta_s_Human):.4f}')
    
    plt.plot(delta_s_GPT4o, label='GPT4o', marker='o', linestyle='-', color='blue')
    plt.axhline(y=np.nanmean(delta_s_GPT4o), color='red', linestyle='--', label=f'Mean(GPT4o): {np.nanmean(delta_s_GPT4o):.4f}')

    # plt.xlabel('Task Index')
    # plt.xlabel('Task Indices', fontsize=20, fontweight='bold')
    plt.ylabel('Relative Difference', fontsize=30, fontweight='bold')
    plt.yticks(fontsize=15)

    # set x ticks to be the given argument idx
    print(f"the is {idx}")
    plt.xticks(ticks=[j for j in range(len(idx))], labels=[str(i+1) for i in idx], fontsize=15, fontweight='bold', rotation = 90)
    # plt.xticks(fontsize=15)
    plt.legend(fontsize=18)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f'figs/relative_difference_Delta.png')
    plt.show()

def get_Figure_4a():
    delta_s_Human, delta_s_GPT4o, idx = calculate_relative_difference()    
    relative_difference(delta_s_Human=delta_s_Human, delta_s_GPT4o=delta_s_GPT4o, idx = idx)

# Exp[3]: Calculate the heatmap
def heat_scatter_map_single(name='GPT4o'):
    mass = np.load(f'results/FA_{name}.npy')
    if name == 'Human_Label':
        name = 'GT'
    cmap = LinearSegmentedColormap.from_list('blue_green_yellow', ['blue', 'green', 'yellow'])
    data_dic = {}
    path = 'data/source/AK_marked_v4.xlsx'
    data = load_xlsx(path)
    

    s = calculate_s(mass, extract_task_ability_pair(data, TA_model=name))
    data_dic[name] = (read_result()[0], np.array([s[i] for i in range(70)]))

    df = pd.DataFrame({
        's': data_dic[name][0],
        's_prime': data_dic[name][1]
    })

    # Calculate the slope and intercept of the regression line
    slope, intercept, r_value, p_value, std_err = linregress(df['s'], df['s_prime'])
    fig = plt.figure(figsize=(6, 6))
    
    values = np.vstack([data_dic[name][0], data_dic[name][1]])
    kernel = gaussian_kde(values)
    rate = 0.15
    xlim = [min(data_dic[name][0]) - rate * np.ptp(data_dic[name][0]), max(data_dic[name][0]) + rate * np.ptp(data_dic[name][0])]
    ylim = [min(data_dic[name][1]) - (rate + 0.1) * np.ptp(data_dic[name][1]), max(data_dic[name][1]) + (rate + 0.1) * np.ptp(data_dic[name][1])]

    X, Y = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
    Z = np.reshape(kernel(np.vstack([X.ravel(), Y.ravel()])).T, X.shape)
    
    x_vals = np.array(xlim)
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color='darkorange', linewidth=2, label=f'Regression Line (k={slope:.2f})')
    ax = sns.scatterplot(x='s', y='s_prime', data=df, 
                        color='skyblue', edgecolor='black', s=70, alpha=1, label=name)
    
    # Plot the smoothed density heatmap
    plt.imshow(Z, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], origin='lower', cmap=cmap, aspect='auto')
    plt.title(name, fontsize=16, fontweight='bold')
    plt.xlabel('Sampled difficulty level s', fontsize=16, fontweight='bold')
    plt.ylabel('Calculated difficulty level s\'', fontsize=16, fontweight='bold')
    plt.legend(fontsize=13)
    
    # set the y limit levels with larger fontsize
    plt.xticks(np.arange(0,1.75, step=0.5), fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(xlim)
    plt.ylim(ylim)
    # Set the aspect ratio based on the data ranges
    ax_aspect = np.ptp(xlim) / np.ptp(ylim)
    plt.gca().set_aspect(ax_aspect)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if name == 'GT':
        plt.savefig(f'figs/scatter_diagram_heat_single_Human_Label.png', bbox_inches='tight', dpi=300)
    else:
        plt.savefig(f'figs/scatter_diagram_heat_single_{name}.png', bbox_inches='tight', dpi=300)
    plt.show()

def get_Figure_4b():
    TA_model = 'GPT4o'
    heat_scatter_map_single(TA_model)
    TA_model = 'Human_Label'
    heat_scatter_map_single(TA_model)    

# Exp[4]: Get the level wise ability distributions(Fig5(a) in the paper)
def level_wise_ability(TA_model='GPT4o'):
    path = 'data/source/AK_marked_v4.xlsx'
    data = load_xlsx(path)
    
    # levels based on sampled value
    _, grades = read_result()

    t_a_pair = extract_task_ability_pair(data, TA_model=TA_model)
    t_a_pair_human = extract_task_ability_pair(data, TA_model='GT')
      
    dic_level_ability = {}
    for lid, level in enumerate(grades):
        for t in level:
            for a in t_a_pair[t]:
                if lid not in dic_level_ability:
                    dic_level_ability[lid] = [0, 0, 0, 0, 0]
                dic_level_ability[lid][a-1] += 1
                
    dic_level_ability_human = {}
    for lid, level in enumerate(grades):
        for t in level:
            for a in t_a_pair_human[t]:
                if lid not in dic_level_ability_human:
                    dic_level_ability_human[lid] = [0, 0, 0, 0, 0]
                dic_level_ability_human[lid][a-1] += 1
                
    # Normalize the distribution
    for a in dic_level_ability:
        total = sum(dic_level_ability[a])
        dic_level_ability[a] = [val/total for val in dic_level_ability[a]]

    for a in dic_level_ability_human:
        total = sum(dic_level_ability_human[a])
        dic_level_ability_human[a] = [val/total for val in dic_level_ability_human[a]]
        
    data_hodge = dic_level_ability
    data_human = dic_level_ability_human
    
    # Possible values for abilities: ability1-5
    abilities = ['A_F', 'A_O', 'A_S', 'A_T', 'A_R']
    abilities = ['', '','','','']
    x = np.arange(len(abilities))  # Indices for abilities

    # Set the style using Seaborn
    sns.set(style="darkgrid")

    # Colors for models
    color_hodge = '#d7191c'  # Red
    color_human = '#2b83ba'  # Blue

    # Create the plot
    fig, axes = plt.subplots(1, 5, figsize=(25, 3))
    axes = axes.flatten()

    for key in data_hodge:
        values_hodge = data_hodge[key]
        values_human = data_human[key]
        
        ax = axes[key]
        
        # Generate smooth curves
        x_smooth = np.linspace(x.min(), x.max(), 300)
        spl = make_interp_spline(x, values_hodge, k=1)  
        spl_human = make_interp_spline(x, values_human, k=1)
        
        y_smooth = spl(x_smooth)
        y_smooth_human = spl_human(x_smooth)

        # Plot smooth curves
        ax.plot(x_smooth, y_smooth, label=f'GPT-4o', color=color_hodge, lw=2)
        ax.plot(x_smooth, y_smooth_human, label=f'Human Label', color=color_human, lw=2)
        
        # Fill area under curves
        ax.fill_between(x_smooth, y_smooth, alpha=0.3, color=color_hodge)
        ax.fill_between(x_smooth, y_smooth_human, alpha=0.3, color=color_human)
        
        # Customize ticks and labels
        ax.set_xticks(x)
        ax.set_xticklabels(abilities, rotation=0, ha='center', fontsize=10)
        ax.set_yticks(np.arange(0, 0.61, 0.2))
        ax.set_yticklabels([f'{val:.1f}' for val in np.arange(0, 0.61, 0.2)], fontsize=15)
        if key == 0:
            ax.set_ylabel('Proportions', fontsize=15)
        
        # ax.set_title(f'Level {key+1}', fontsize=18)
        if key!=0:
            ax.set_yticklabels([])
        if key == 4:
            ax.legend(fontsize=18)
        # ax.legend()
    plt.tight_layout(pad=3, w_pad=0.2)
    
    # Save the plot
    plt.savefig(f'figs/smoothed_levelwise_ab_distributions_Both{TA_model}.png')
    plt.show()


    return dic_level_ability

# Exp[5]: Get the benchmark wise ability distributions(Fig5(b) in the paper)
# (Imported from asses_benches.py)

# Exp[6]: Get the benchmark difficulty assessment(Fig5(c) in the paper)
# (Imported from asses_benches.py)

# Exp[7]: Calculate the normal rate
def calculate_inverse_pairs(dic, task='valid'):
    s, _ = read_result()
    s = np.array(s)
    s -= s.min()
    s += 1e-4
    # s = load_s()
    TA_models = ['GT', 'GPT3.5', 'GPT4', 'GPT4o']
    dataGT = np.load('results/FA_Human_Label.npy')
    dataGPT3_5 = np.load('results/FA_GPT3.5.npy')
    dataGPT4 = np.load('results/FA_GPT4.npy')
    dataGPT4o = np.load('results/FA_GPT4o.npy')
    
    abilities = [dataGT, dataGPT3_5, dataGPT4, dataGPT4o]
    if task == 'valid':
        path = 'data/source/vision_tasks.xlsx'
        data = load_xlsx(path)

        for TA_model,data_ in zip(TA_models, abilities):
            dic_ab = extract_task_ability_pair(data, TA_model)
            # cal = calculate_difficulties(dic_ab, data_)
            cal = calculate_s(data_, dic_ab)
      
            lis = cal.values()
            inverse = {}
            for id, value in enumerate(lis):
                for tasks in range(70):
                    if (id, tasks) not in dic.keys():
                        raise ValueError(f'pair {(id, tasks)} not in dic')
                    if (value < s[tasks] and dic[(id, tasks)][0] > dic[(id, tasks)][1]) or (value > s[tasks] and dic[(id, tasks)][0] < dic[(id, tasks)][1]):
                        # print(f"value {value} < s {s[tasks]} and dic[({id, tasks})][0]:{dic[(id, tasks)][0]} > dic[({id, tasks})][1]: {dic[(id, tasks)][1]}")
                        """
                        1. 前者判断，按我们的方式，id比task难，但是实际上人类认为id比task容易
                        2. 后者判断，按我们的方式，id比task容易，但是实际上人类认为id比task难
                        """
                        # print(f'pair {(id, tasks)} is inverse')
                        if id not in inverse:
                            inverse[id] = 1
                        else:
                            inverse[id] += 1
            inverse = sum(inverse.values())/700
            print(f'----------------- {TA_model} normal rate: {round(1-inverse, 4)} -----------------')
    elif task == 'origin':
        path = 'data/source/AK_marked_v4.xlsx'
        data = load_xlsx(path)

        for TA_model,data_ in zip(TA_models, abilities):
            dic_ab = extract_task_ability_pair(data, TA_model)
            # cal = calculate_difficulties(dic_ab, data_)
            cal = calculate_s(data_, dic_ab)
            lis = cal.values()
            inverse = {}
            for id, value in enumerate(lis):
                for tasks in range(id+1, 70):
                    if (id, tasks) not in dic.keys():
                        print(dic.keys())
                        raise ValueError(f'pair {(id, tasks)} not in dic')
                    if (value < s[tasks] and dic[(id, tasks)][0] > dic[(id, tasks)][1]) or (value > s[tasks] and dic[(id, tasks)][0] < dic[(id, tasks)][1]):
                        # print(f"value {value} < s {s[tasks]} and dic[({id, tasks})][0]:{dic[(id, tasks)][0]} > dic[({id, tasks})][1]: {dic[(id, tasks)][1]}")
                        """
                        1. 前者判断，按我们的方式，id比task难，但是实际上人类认为id比task容易
                        2. 后者判断，按我们的方式，id比task容易，但是实际上人类认为id比task难
                        """
                        # print(f'pair {(id, tasks)} is inverse')
                        if id not in inverse:
                            inverse[id] = 1
                        else:
                            inverse[id] += 1
            inverse = sum(inverse.values())/(60*70/2)
            print(f'----------------- {TA_model} normal rate: {round(1-inverse, 4)} -----------------')

def normal_rate():
    print('[origin assessment]')
    path = 'data/qs/combined_2.18.csv'
    data = load_csv(path)
    dic = read_data(data)
    calculate_inverse_pairs(dic, task='origin')
    print('\n')
    print('[validation assessment]')
    path = 'data/qs/validation.csv'
    data_valid = load_csv(path)
    dic_valid = read_valid_data(data_valid)
    calculate_inverse_pairs(dic_valid, task='valid')

# Exp[8-1]: Consistency Analysis 1->Intrenal consistency of survey data
def consistency_check():
    path = 'data/qs/combined_2.18.csv'
    data = load_csv(path) 
    dic = read_data(data)  
    
    # check consistency:
    consistency_dic = {
        'strongly consistent': 0,
        'consistent': 0,
        'weakly consistent': 0,
        'inconsistent': 0,

    }
    for key in dic:
        if dic[key][0] == 0 and dic[key][1] !=0 or dic[key][0] != 0 and dic[key][1] == 0 or dic[key][0] == 0 and dic[key][1] == 0:
            consistency_dic['strongly consistent'] += 1

        elif dic[key][0] + dic[key][2] >= dic[key][3]*0.8 or dic[key][1] + dic[key][2] >= dic[key][3]*0.8:
            consistency_dic['consistent'] += 1

        elif dic[key][0] + dic[key][2] >= dic[key][3]*0.6 or dic[key][1] + dic[key][2] >= dic[key][3]*0.6:
            consistency_dic['weakly consistent'] += 1

        else:
            consistency_dic['inconsistent'] += 1

    # normalize the consistency_dic
    total = sum([val for val in consistency_dic.values()])
    for key in consistency_dic:
        consistency_dic[key] /= total
        
    print(consistency_dic)
    # plot the consistency with pie
    plt.figure()
    plt.pie([val for val in consistency_dic.values()], labels=[key for key in consistency_dic.keys()], autopct='%1.1f%%')
    plt.title('Consistency of task pairs')
    plt.savefig('figs/consistency.png')


# Exp[8-2]: Consistency Analysis 2->Level-wise consistency of survey data
def level_wise_consistency():
    path = 'data/qs/combined_2.18.csv'
    data = load_csv(path)
    dic = read_data(data)
    result = np.zeros((5, 5))
    _, hodge_res = read_result()
    for i in range(4):
        for j in range(i+1, 5):
            inversed = 0
            low_level = hodge_res[i]
            high_level = hodge_res[j]
            for task1 in low_level:
                for task2 in high_level:
                    if task1 < task2:
                        if dic[(task1, task2)][0] > dic[(task1, task2)][1]:
                            inversed += 1
                    elif task1>task2:
                        if dic[(task2, task1)][0] < dic[(task2, task1)][1]:
                            inversed += 1
            inversed/=(len(low_level)*len(high_level))
            result[i, j] = 1 - inversed

    matrix = result
    # Define the custom colormap
    colors = [(1, 0.9, 0.9), (0.9, 0, 0)]  # RGB tuples for pink to red
    n_bins = 20  # Discretizes the interpolation into 100 steps
    custom_cmap = LinearSegmentedColormap.from_list(name='custom_red', colors=colors, N=n_bins)

    # Mask zeros to distinguish them clearly (optional)
    masked_matrix = np.ma.masked_where(matrix == 0, matrix)

    # Mask zeros to distinguish them clearly (optional)
    masked_matrix = np.ma.masked_where(matrix == 0, matrix)
    
    # Plotting the matrix with the custom colormap and normalization
    plt.imshow(masked_matrix, cmap=custom_cmap, interpolation='nearest')

    # Adding color bar
    plt.colorbar(label='Value')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i == j:
                plt.gca().add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, color=(0.93,0.93,0.93), lw=0))
            if i<j:
                plt.gca().add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1, color=(0.93,0.93,0.93), lw=0))

    # Adding text annotations for each cell
    for (i, j), val in np.ndenumerate(matrix):
        if val != 0:  # Skip the zero values for annotations
            plt.text(j, i, f'{val:.2f}', ha='center', va='center', color='black' if val < 0.75 else 'white')
        if i >= j:
            plt.text(j, i, f'----', ha='center', va='center', color='black' if val < 0.75 else 'white')
            # set a light corlor for the diagonal

    # Remove the axis lines (spines) but keep the labels
    ax = plt.gca()
    plt.xticks(np.arange(matrix.shape[1]), labels=np.arange(1, matrix.shape[1] + 1))
    plt.yticks(np.arange(matrix.shape[0]), labels=np.arange(1, matrix.shape[0] + 1))


    # Display the plot
    plt.title("Level-wise data consistency comparison")
    plt.savefig('figs/level_wise_consistency.png')
    plt.show()

# Exp[8-3]: Consistency Analysis 3->Statistical Consistency, Correlations between the survey data and Human Expert Label
def check_correlation():
    """
    --------------------------------------------
    Check the correlation ratio between the human label and the HodgeRank+Cluster
    --------------------------------------------
    args:
    (1) a: dict, the data of the human label and the HodgeRank+Cluster

    return:
    None

    """
    a['HodgeRank+Cluster'] = read_result()[1]
    for key in a:
        temp = []
        for i in range(5):
            temp += a[key][i]
        a[key] = temp
        
    dataframe = pd.DataFrame(a)
    column_list = ['Human Label', 'HodgeRank+Cluster']

    # 绘制特征之间的散点图
    sns.pairplot(dataframe[column_list], kind="scatter")

    plt.savefig('figs/scatter_plot.png')

    # 计算特征之间的相关性并绘制热力图
    fig, axes = plt.subplots(1, 3, figsize=(25, 7))

    # Pearson Correlation Heatmap
    correlation_matrix = dataframe[column_list].corr(method='pearson')
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, annot_kws={'size': 20}, ax=axes[0], cbar=False)
    axes[0].set_title('Pearson Correlation', fontsize=18)
    axes[0].set_yticklabels(column_list, fontsize=18)
    axes[0].set_xticklabels(column_list, fontsize=15)
    axes[1].set_xticklabels(column_list, fontsize=15)
    axes[2].set_xticklabels(column_list, fontsize=15)



    # Kendall Correlation Heatmap
    correlation_matrix = dataframe[column_list].corr(method='kendall')
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, annot_kws={'size': 20}, ax=axes[1], yticklabels=False, cbar=False)
    axes[1].set_title('Kendall Correlation', fontsize=18)

    # Spearman Correlation Heatmap
    correlation_matrix = dataframe[column_list].corr(method='spearman')
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, annot_kws={'size': 20}, ax=axes[2], yticklabels=False)
    axes[2].set_title('Spearman Correlation', fontsize=18)

    # Add color bar to the rightmost subplot
    cbar = axes[2].collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)

    plt.tight_layout(pad=2)
    plt.savefig('figs/correlation_heatmap_combined.png')
    plt.show()




if __name__ == '__main__':
    config = load_config()
    exp_dic = {
        0: 'get_difficulty_levels',
        1: 'solve_FA',
        2: 'calculate_relative_difference',
        3: 'heatmap',
        4: 'level_wise_ability',
        5: 'bench_wise_ability',
        6: 'bench_difficulty_assessment',
        7: 'normal rate',
        8: 'Internal consistency',
        9: 'Level-wise consistency',
        10: 'Correlations',
        11: 'get_propoer_cluster_num'
    }
    
    if exp_dic[config['exp_id']] == 'get_difficulty_levels':
        get_difficulty_levels(config['data_file_path'], config['print_details'])
        
    elif exp_dic[config['exp_id']] == 'solve_FA':
        solve_FA(config['TA_model'])
    
    elif exp_dic[config['exp_id']] == 'get_propoer_cluster_num':
        get_propoer_cluster_num()
    
    elif exp_dic[config['exp_id']] == 'calculate_relative_difference':
        get_Figure_4a()
        
    elif exp_dic[config['exp_id']] == 'heatmap':
        get_Figure_4b()
    
    elif exp_dic[config['exp_id']] == 'level_wise_ability':
        level_wise_ability(TA_model=config['TA_model'])
    
    elif exp_dic[config['exp_id']] == 'bench_wise_ability':
        get_Figure_5b()
    
    elif exp_dic[config['exp_id']] == 'bench_difficulty_assessment':
        get_Figure_5c()
        
    elif exp_dic[config['exp_id']] == 'normal rate':
        normal_rate()
    
    elif exp_dic[config['exp_id']] == 'Internal consistency':
        consistency_check()
    
    elif exp_dic[config['exp_id']] == 'Level-wise consistency':
        level_wise_consistency()
    
    elif exp_dic[config['exp_id']] == 'Correlations':
        check_correlation()
    
    else:
        print("No such experiment")