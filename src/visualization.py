import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import re
import ast
import math
import logging

from IPython.core.pylabtools import figsize
from astropy.units.quantity_helper.function_helpers import quantile
from docutils.nodes import inline
from matplotlib.lines import lineStyles
from scipy.stats import entropy
from sympy.abc import alpha

from Federated_Learning.src.objective_func import L, contrained_OPT_obj

sns.set_style('darkgrid')
sns.set_context('poster')

def visualize_obj_func():
    # Define theta range
    theta = np.linspace(-2*np.pi, 2*np.pi, 400)
    x = np.linspace(-4,4, 200)
    y = np.linspace(0, 2, 400)
    # Plotting
    plt.figure(figsize=(16, 9))
    plt.plot(theta, (np.cos(theta-1)+np.sin(3*(theta)))/2 + 1, label=r'$L(\theta)$', linewidth=4)
    plt.plot(x, np.abs(x)/2, label=r'$G(\theta)$', linewidth=3, linestyle='--')
    plt.scatter(-2.57, 0.062, marker='*', color='red', s=280, label=r'$\theta_{good}^*$', zorder=4)
    plt.arrow(x=-2*math.pi, y=0 ,dx=4*math.pi, dy=0, width=0.008, head_width=0.1, head_length=0.2, color='black')
    plt.arrow(x=0, y=0, dx=0, dy=2.0, width=0.05, head_width=0.35, head_length=0.06, color='black')
    # plt.title('Objective Function $L(\\theta)$ with Components $L_1(\\theta)$ and $L_2(\\theta)$')
    # plt.xlabel('r$\theta$')
    # plt.ylabel('Function Value')
    plt.xticks([])
    plt.yticks([])
    # plt.axis('off')
    plt.legend(prop={'size': 20})
    plt.grid(True, linestyle='--', alpha=1.0)
    plt.savefig('../results/objective_function.png', bbox_inches='tight')
    plt.show()


def particles_position_vis_1d(t, images_save_path, particle_positions_x_axis, consensus_point_x_axis):
    x_axis = np.linspace(-2*np.pi, 2*np.pi, num=1000)
    obj_function_value = L(x_axis)
    plt.figure(figsize=(8,6))

    plt.plot(x_axis, obj_function_value, alpha=0.6)

    particle_positions_x_axis = particle_positions_x_axis[np.where((particle_positions_x_axis<=2*np.pi) & (particle_positions_x_axis>=-2*np.pi))]
    particle_positions_y_axis = L(particle_positions_x_axis)
    plt.scatter(particle_positions_x_axis, particle_positions_y_axis, label=r'$I^{\beta}[\rho_t^N]$', color='darkorange', s=40)

    consensus_point_y_axis = L(consensus_point_x_axis)
    plt.scatter(consensus_point_x_axis, consensus_point_y_axis, label='m_t',
                marker='v', s=60, color='darkred')
    plt.legend(loc='upper right')
    plt.title(f'Particles Positions at time T={t}', fontsize=16)
    images_save_name = os.path.join(images_save_path, f'img_{t}.png')
    plt.savefig(images_save_name,transparent=False,facecolor='white')


def vis_contour_plot():
    x = np.linspace(-4,4, 100)
    y = np.linspace(-4,4, 100)

    X, Y = np.meshgrid(x, y)

    Z = np.sin(0.5*(X-0.1)**2) + np.cos((Y-np.pi))
    # theta = np.concatenate([X,Y])
    # Z = constrained_opt_obj.G(theta)

    # Draw contour plot
    plt.figure(figsize=(16, 9))
    plt.contourf(X, Y, Z, cmap='viridis')
    # plt.colorbar()
    plt.scatter(-2.97, 0, marker='*', color='red', s=300)
    plt.title(r'$L(\theta) = sin(\frac{1}{2}(\theta_1-0.1)^2) + cos(\theta_2-\pi)$', fontsize=40)
    # plt.xlabel('x')
    # plt.ylabel('y')
    plt.xticks(np.arange(-4,4.1, step=1))
    plt.yticks(np.arange(-3,3.1, step=1))
    plt.grid(True, linestyle='--', alpha=0.5, linewidth=0.7)
    plt.savefig('../results/objective_function_2d.png', bbox_inches='tight')
    plt.show()

# vis_contour_plot()

def parse_selection_time_and_reward(file_path):
    # Pattern to identify lines with rewards and times
    pattern_selection_time = r'Selection time of agent (\d+): \[([^\]]+)\]'
    pattern_selection_reward = r'Estimated selection reward of agent (\d+): \[([^\]]+)\]'

    # Dictionary to hold validation losses
    selection_time_by_agent = {}
    selection_reward_by_agent = {}

    current_round = -1
    with open(file_path, 'r') as file:
        text = file.read()
        # Extract all rewards
        for match in re.finditer(pattern_selection_reward, text):
            agent_id = int(match.group(1))
            if agent_id not in selection_reward_by_agent:
                selection_reward_by_agent[agent_id] = []
            reward_str = match.group(2)
            # Convert the reward string into a numpy array
            reward_array = np.array([float(x) for x in reward_str.split()])
            selection_reward_by_agent[agent_id].append(reward_array)

        # Extract all selection times
        for match in re.finditer(pattern_selection_time, text):
            agent_id = int(match.group(1))
            if agent_id not in selection_time_by_agent:
                selection_time_by_agent[agent_id] = []
            time_str = match.group(2)
            # Convert the time string into a numpy array
            time_array = np.array([float(x) for x in time_str.split()])
            selection_time_by_agent[agent_id].append(time_array)

    selection_reward_by_round = {}
    for agent_id in selection_reward_by_agent:
        selection_reward_agent_i = selection_reward_by_agent[agent_id]
        for comm_round, selection_reward in enumerate(selection_reward_agent_i):
            if comm_round not in selection_reward_by_round:
                selection_reward_by_round[comm_round] = []
            selection_reward_by_round[comm_round].append(selection_reward)

    selection_time_final_round = []
    for agent_id in selection_time_by_agent:
        selection_time_final_round.append(selection_time_by_agent[agent_id][-1])

    return selection_time_final_round, selection_reward_by_round, selection_reward_by_agent

def parse_max_diff_loss(file_path):
    # Pattern to identify lines with max_diff loss
    pattern_max_diff = r'Maximum difference class-wise loss evaluating on dataset from agent (\d+): \[([^\]]+)\]'

    # Dictionary to hold validation losses
    max_diff_by_agent = {}

    current_round = -1
    with open(file_path, 'r') as file:
        text = file.read()
        # Extract all rewards
        for match in re.finditer(pattern_max_diff, text):
            agent_id = int(match.group(1))
            if agent_id not in max_diff_by_agent:
                max_diff_by_agent[agent_id] = []
            max_diff_str = match.group(2)
            # Convert the reward string into a numpy array
            max_diff_array = np.array([float(x) for x in max_diff_str.split(',')])
            max_diff_by_agent[agent_id].append(max_diff_array)

    max_diff_by_round = {}
    for agent_id in max_diff_by_agent:
        max_diff_agent_i = max_diff_by_agent[agent_id]
        for comm_round, max_diff in enumerate(max_diff_agent_i):
            if comm_round not in max_diff_by_round:
                max_diff_by_round[comm_round] = []
            max_diff_by_round[comm_round].append(max_diff)

    return max_diff_by_agent, max_diff_by_round


def vis_selection_reward(selection_reward, benign_agent_list, malicious_agent_list):
    g1_b_g1_b_reward = []
    g1_b_g1_m_reward = []
    g1_b_g2_b_reward = []
    g1_b_g2_m_reward = []

    g2_b_g2_b_reward = []
    g2_b_g2_m_reward = []
    g2_b_g1_b_reward = []
    g2_b_g1_m_reward = []

    g1_b_g1_b_std = []
    g1_b_g1_m_std = []

    g2_b_g2_b_std = []
    g2_b_g2_m_std = []

    g1_b_idx = np.array(benign_agent_list[0])
    g1_m_idx = np.array(malicious_agent_list[0])
    g2_b_idx = np.array(benign_agent_list[1])
    g2_m_idx = np.array(malicious_agent_list[1])

    for comm_round in selection_reward:
        selection_reward_round_i = np.array(selection_reward[comm_round])

        g1_b_select_reward = selection_reward_round_i[g1_b_idx]

        g1_b_select_reward = np.mean(g1_b_select_reward, axis=0)
        g1_b_g1_b_reward.append(np.mean(g1_b_select_reward[g1_b_idx]))
        g1_b_g1_m_reward.append(np.mean(g1_b_select_reward[g1_m_idx]))
        g1_b_g2_b_reward.append(np.mean(g1_b_select_reward[g2_b_idx]))
        g1_b_g2_m_reward.append(np.mean(g1_b_select_reward[g2_m_idx]))

        g1_b_g1_b_std.append(np.std(g1_b_select_reward[g1_b_idx]))
        g1_b_g1_m_std.append(np.std(g1_b_select_reward[g1_m_idx]))

        g2_b_select_reward = selection_reward_round_i[g2_b_idx]

        g2_b_select_reward = np.mean(g2_b_select_reward, axis=0)
        g2_b_g2_b_reward.append(np.mean(g2_b_select_reward[g2_b_idx]))
        g2_b_g2_m_reward.append(np.mean(g2_b_select_reward[g2_m_idx]))
        g2_b_g1_b_reward.append(np.mean(g2_b_select_reward[g1_b_idx]))
        g2_b_g1_m_reward.append(np.mean(g2_b_select_reward[g1_m_idx]))

        g2_b_g2_b_std.append(np.std(g2_b_select_reward[g2_b_idx]))
        g2_b_g2_m_std.append(np.std(g2_b_select_reward[g2_m_idx]))

    comm_round = np.arange(len(g1_b_g1_b_reward))

    g1_b_g1_b_reward_err0 = np.array(g1_b_g1_b_reward) - np.array(g1_b_g1_b_std)
    g1_b_g1_b_reward_err1 = np.array(g1_b_g1_b_reward) + np.array(g1_b_g1_b_std)
    g1_b_g1_m_reward_err0 = np.array(g1_b_g1_m_reward) - np.array(g1_b_g1_m_std)
    g1_b_g1_m_reward_err1 = np.array(g1_b_g1_m_reward) + np.array(g1_b_g1_m_std)

    plt.figure(figsize=(16,10))
    plt.plot(comm_round, g1_b_g1_b_reward, label='g1_b_g1_b', c='tab:green')
    plt.plot(comm_round, g1_b_g1_m_reward, label='g1_b_g1_m', c='tab:red')
    plt.fill_between(comm_round, g1_b_g1_b_reward_err0, g1_b_g1_b_reward_err1, color='tab:green', alpha=0.2)
    plt.fill_between(comm_round, g1_b_g1_m_reward_err0, g1_b_g1_m_reward_err1, color='tab:red', alpha=0.2)
    plt.plot(comm_round, g1_b_g2_b_reward, label='g1_b_g2_b')
    plt.plot(comm_round, g1_b_g2_m_reward, label='g1_b_g2_m')

    plt.ylabel('Selection Reward', fontsize=30)
    plt.xlabel('Communication Round', fontsize=30)
    plt.title('Selection Reward of Benign Agents from Group 1', fontsize=30)

    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('../results/visualization/FedCBO_emnist/select_time_and_reward/Avg_results/selection_reward_g1_b_Temp_0.25.png', bbox_inches='tight')
    plt.show()

    g2_b_g2_b_reward_err0 = np.array(g2_b_g2_b_reward) - np.array(g2_b_g2_b_std)
    g2_b_g2_b_reward_err1 = np.array(g2_b_g2_b_reward) + np.array(g2_b_g2_b_std)
    g2_b_g2_m_reward_err0 = np.array(g2_b_g2_m_reward) - np.array(g2_b_g2_m_std)
    g2_b_g2_m_reward_err1 = np.array(g2_b_g2_m_reward) + np.array(g2_b_g2_m_std)

    plt.figure(figsize=(16, 10))
    plt.plot(comm_round, g2_b_g2_b_reward, label='g2_b_g2_b', c='tab:green')
    plt.plot(comm_round, g2_b_g2_m_reward, label='g2_b_g2_m', c='tab:red')
    plt.fill_between(comm_round, g2_b_g2_b_reward_err0, g2_b_g2_b_reward_err1, color='tab:green', alpha=0.2)
    plt.fill_between(comm_round, g2_b_g2_m_reward_err0, g2_b_g2_m_reward_err1, color='tab:red', alpha=0.2)
    plt.plot(comm_round, g2_b_g1_b_reward, label='g2_b_g1_b')
    plt.plot(comm_round, g2_b_g1_m_reward, label='g2_b_g1_m')

    plt.ylabel('Selection Reward', fontsize=30)
    plt.xlabel('Communication Round', fontsize=30)
    plt.title('Selection Reward of Benign Agents from Group 2', fontsize=30)

    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('../results/visualization/FedCBO_emnist/select_time_and_reward/Avg_results/selection_reward_g2_b_Temp_0.25.png', bbox_inches='tight')
    plt.show()


def vis_selection_reward_agentwise(agent_idx, cluster_idx, selection_reward, benign_agent_list, malicious_agent_list):
    g1_b_reward = []
    g1_m_reward = []
    g2_b_reward = []
    g2_m_reward = []

    if cluster_idx == 0:
        g1_b_std = []
        g1_m_std = []
    elif cluster_idx == 1:
        g2_b_std = []
        g2_m_std = []

    g1_b_idx = np.array(benign_agent_list[0])
    g1_m_idx = np.array(malicious_agent_list[0])
    g2_b_idx = np.array(benign_agent_list[1])
    g2_m_idx = np.array(malicious_agent_list[1])

    for comm_round in selection_reward:
        selection_reward_round_i = np.array(selection_reward[comm_round])
        agent_selection_reward_round_i = selection_reward_round_i[agent_idx]

        g1_b_reward.append(np.mean(agent_selection_reward_round_i[g1_b_idx]))
        g1_m_reward.append(np.mean(agent_selection_reward_round_i[g1_m_idx]))
        g2_b_reward.append(np.mean(agent_selection_reward_round_i[g2_b_idx]))
        g2_m_reward.append(np.mean(agent_selection_reward_round_i[g2_m_idx]))

        if cluster_idx == 0:
            g1_b_std.append(np.std(agent_selection_reward_round_i[g1_b_idx]))
            g1_m_std.append(np.std(agent_selection_reward_round_i[g1_m_idx]))
        elif cluster_idx == 1:
            g2_b_std.append(np.std(agent_selection_reward_round_i[g2_b_idx]))
            g2_m_std.append(np.std(agent_selection_reward_round_i[g2_m_idx]))

    comm_round = np.arange(len(g1_b_reward))

    plt.figure(figsize=(16, 10))

    if cluster_idx == 0:
        g1_b_reward_err0 = np.array(g1_b_reward) - 0.5*np.array(g1_b_std)
        g1_b_reward_err1 = np.array(g1_b_reward) + 0.5*np.array(g1_b_std)
        g1_m_reward_err0 = np.array(g1_m_reward) - 0.5*np.array(g1_m_std)
        g1_m_reward_err1 = np.array(g1_m_reward) + 0.5*np.array(g1_m_std)

        plt.plot(comm_round, g1_b_reward, label='g1_b', c='tab:green')
        plt.plot(comm_round, g1_m_reward, label='g1_m', c='tab:red')
        plt.fill_between(comm_round, g1_b_reward_err0, g1_b_reward_err1, color='tab:green', alpha=0.2)
        plt.fill_between(comm_round, g1_m_reward_err0, g1_m_reward_err1, color='tab:red', alpha=0.2)
        plt.plot(comm_round, g2_b_reward, label='g2_b')
        plt.plot(comm_round, g2_m_reward, label='g2_m')
    elif cluster_idx == 1:
        g2_b_reward_err0 = np.array(g2_b_reward) - 0.5*np.array(g2_b_std)
        g2_b_reward_err1 = np.array(g2_b_reward) + 0.5*np.array(g2_b_std)
        g2_m_reward_err0 = np.array(g2_m_reward) - 0.5*np.array(g2_m_std)
        g2_m_reward_err1 = np.array(g2_m_reward) + 0.5*np.array(g2_m_std)

        plt.plot(comm_round, g2_b_reward, label='g2_b', c='tab:green')
        plt.plot(comm_round, g2_m_reward, label='g2_m', c='tab:red')
        plt.fill_between(comm_round, g2_b_reward_err0, g2_b_reward_err1, color='tab:green', alpha=0.2)
        plt.fill_between(comm_round, g2_m_reward_err0, g2_m_reward_err1, color='tab:red', alpha=0.2)
        plt.plot(comm_round, g1_b_reward, label='g1_b')
        plt.plot(comm_round, g1_m_reward, label='g1_m')

    plt.ylabel('Selection Reward', fontsize=30)
    plt.xlabel('Communication Round', fontsize=30)
    plt.title('Selection Reward of Agent {}'.format(agent_idx), fontsize=30)

    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('../results/visualization/FedCBO_emnist/select_time_and_reward/agentwise_results/selection_reward_of_Agent_{}_Temp_0.1.png'.format(agent_idx), bbox_inches='tight')
    plt.show()

def vis_loss(selection_reward, temp, benign_agent_list, malicious_agent_list):
    g1_b_g1_b_loss = []
    g1_b_g1_m_loss = []
    g1_b_g2_b_loss = []
    g1_b_g2_m_loss = []

    g2_b_g2_b_loss = []
    g2_b_g2_m_loss = []
    g2_b_g1_b_loss = []
    g2_b_g1_m_loss = []

    g1_b_g1_b_std = []
    g1_b_g1_m_std = []

    g2_b_g2_b_std = []
    g2_b_g2_m_std = []

    g1_b_idx = np.array(benign_agent_list[0])
    g1_m_idx = np.array(malicious_agent_list[0])
    g2_b_idx = np.array(benign_agent_list[1])
    g2_m_idx = np.array(malicious_agent_list[1])

    start_epoch = 0
    for comm_round in selection_reward:
        selection_reward_round_i = np.array(selection_reward[comm_round])

        g1_b_selection_reward_round_i = selection_reward_round_i[g1_b_idx]

        if 0.0 in g1_b_selection_reward_round_i:
            start_epoch += 1
            continue

        g1_loss_round_i = -temp * np.log(g1_b_selection_reward_round_i)

        g1_b_loss = np.mean(g1_loss_round_i, axis=0)
        g1_b_g1_b_loss.append(np.mean(g1_b_loss[g1_b_idx]))
        g1_b_g1_m_loss.append(np.mean(g1_b_loss[g1_m_idx]))
        g1_b_g2_b_loss.append(np.mean(g1_b_loss[g2_b_idx]))
        g1_b_g2_m_loss.append(np.mean(g1_b_loss[g2_m_idx]))

        g1_b_g1_b_std.append(np.std(g1_b_loss[g1_b_idx]))
        g1_b_g1_m_std.append(np.std(g1_b_loss[g1_m_idx]))

        g2_b_selection_reward_round_i = selection_reward_round_i[g2_b_idx]

        g2_loss_round_i = -temp * (np.log(g2_b_selection_reward_round_i))

        g2_b_loss = np.mean(g2_loss_round_i, axis=0)
        g2_b_g2_b_loss.append(np.mean(g2_b_loss[g2_b_idx]))
        g2_b_g2_m_loss.append(np.mean(g2_b_loss[g2_m_idx]))
        g2_b_g1_b_loss.append(np.mean(g2_b_loss[g1_b_idx]))
        g2_b_g1_m_loss.append(np.mean(g2_b_loss[g1_m_idx]))

        g2_b_g2_b_std.append(np.std(g2_b_loss[g2_b_idx]))
        g2_b_g2_m_std.append(np.std(g2_b_loss[g2_m_idx]))

    comm_round = np.arange(len(g1_b_g1_b_loss)) + start_epoch


    g1_b_g1_b_reward_err0 = np.array(g1_b_g1_b_loss) - np.array(g1_b_g1_b_std)
    g1_b_g1_b_reward_err1 = np.array(g1_b_g1_b_loss) + np.array(g1_b_g1_b_std)
    g1_b_g1_m_reward_err0 = np.array(g1_b_g1_m_loss) - np.array(g1_b_g1_m_std)
    g1_b_g1_m_reward_err1 = np.array(g1_b_g1_m_loss) + np.array(g1_b_g1_m_std)

    plt.figure(figsize=(16, 10))
    plt.plot(comm_round, g1_b_g1_b_loss, label='g1_b_g1_b', c='tab:green')
    plt.plot(comm_round, g1_b_g1_m_loss, label='g1_b_g1_m', c='tab:red')
    plt.fill_between(comm_round, g1_b_g1_b_reward_err0, g1_b_g1_b_reward_err1, color='tab:green', alpha=0.2)
    plt.fill_between(comm_round, g1_b_g1_m_reward_err0, g1_b_g1_m_reward_err1, color='tab:red', alpha=0.2)
    # plt.plot(comm_round, g1_b_g2_b_loss, label='g1_b_g2_b')
    # plt.plot(comm_round, g1_b_g2_m_loss, label='g1_b_g2_m')

    plt.ylabel('Average Evaluation Loss', fontsize=30)
    plt.xlabel('Communication Round', fontsize=30)
    # plt.xlim(xmin=4)

    plt.title('Average Loss on Group 1 Benign Agents\' Datasets', fontsize=30, fontweight = 'bold', pad=20)

    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(
        '../results/visualization/FedCBO_emnist/avg_loss/Avg_results/avg_loss_g1_b_Temp_0.25.png',
        bbox_inches='tight')
    plt.show()

    g2_b_g2_b_reward_err0 = np.array(g2_b_g2_b_loss) - np.array(g2_b_g2_b_std)
    g2_b_g2_b_reward_err1 = np.array(g2_b_g2_b_loss) + np.array(g2_b_g2_b_std)
    g2_b_g2_m_reward_err0 = np.array(g2_b_g2_m_loss) - np.array(g2_b_g2_m_std)
    g2_b_g2_m_reward_err1 = np.array(g2_b_g2_m_loss) + np.array(g2_b_g2_m_std)

    plt.figure(figsize=(16, 10))
    plt.plot(comm_round, g2_b_g2_b_loss, label='g2_b_g2_b', c='tab:green')
    plt.plot(comm_round, g2_b_g2_m_loss, label='g2_b_g2_m', c='tab:red')
    plt.fill_between(comm_round, g2_b_g2_b_reward_err0, g2_b_g2_b_reward_err1, color='tab:green', alpha=0.2)
    plt.fill_between(comm_round, g2_b_g2_m_reward_err0, g2_b_g2_m_reward_err1, color='tab:red', alpha=0.2)
    # plt.plot(comm_round, g2_b_g1_b_loss, label='g2_b_g1_b')
    # plt.plot(comm_round, g2_b_g1_m_loss, label='g2_b_g1_m')

    plt.ylabel('Average Evaluation Loss', fontsize=30)
    plt.xlabel('Communication Round', fontsize=30)
    # plt.xlim(xmin=4)
    plt.title('Average Loss on Group 2 Benign Agents\' Datasets', fontsize=30, fontweight = 'bold', pad=20)

    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(
        '../results/visualization/FedCBO_emnist/avg_loss/Avg_results/avg_loss_g2_b_Temp_0.25.png',
        bbox_inches='tight')
    plt.show()

def vis_loss_agentwise(agent_idx, cluster_idx, selection_reward, temp, benign_agent_list, malicious_agent_list):

    g1_b_loss = []
    g1_m_loss = []
    g2_b_loss = []
    g2_m_loss = []

    if cluster_idx == 0:
        g1_b_std = []
        g1_m_std = []
    elif cluster_idx == 1:
        g2_b_std = []
        g2_m_std = []

    g1_b_idx = np.array(benign_agent_list[0])
    g1_m_idx = np.array(malicious_agent_list[0])
    g2_b_idx = np.array(benign_agent_list[1])
    g2_m_idx = np.array(malicious_agent_list[1])

    start_epoch = 0
    for comm_round in selection_reward:
        selection_reward_round_i = np.array(selection_reward[comm_round])

        agent_i_selection_reward_round_i = selection_reward_round_i[agent_idx]

        if 0.0 in agent_i_selection_reward_round_i:
            start_epoch += 1
            continue

        agent_i_loss_round_i = -temp * np.log(agent_i_selection_reward_round_i)

        g1_b_loss.append(np.mean(agent_i_loss_round_i[g1_b_idx]))
        g1_m_loss.append(np.mean(agent_i_loss_round_i[g1_m_idx]))
        g2_b_loss.append(np.mean(agent_i_loss_round_i[g2_b_idx]))
        g2_m_loss.append(np.mean(agent_i_loss_round_i[g2_m_idx]))

        if cluster_idx == 0:
            g1_b_std.append(np.std(agent_i_loss_round_i[g1_b_idx]))
            g1_m_std.append(np.std(agent_i_loss_round_i[g1_m_idx]))
        elif cluster_idx == 1:
            g2_b_std.append(np.std(agent_i_loss_round_i[g2_b_idx]))
            g2_m_std.append(np.std(agent_i_loss_round_i[g2_m_idx]))


    comm_round = np.arange(len(g1_b_loss)) + start_epoch


    plt.figure(figsize=(16, 10))

    if cluster_idx == 0:
        g1_b_reward_err0 = np.array(g1_b_loss) - 0.5*np.array(g1_b_std)
        g1_b_reward_err1 = np.array(g1_b_loss) + 0.5*np.array(g1_b_std)
        g1_m_reward_err0 = np.array(g1_m_loss) - 0.5*np.array(g1_m_std)
        g1_m_reward_err1 = np.array(g1_m_loss) + 0.5*np.array(g1_m_std)

        plt.plot(comm_round, g1_b_loss, label='g1_b', c='tab:green')
        plt.plot(comm_round, g1_m_loss, label='g1_m', c='tab:red')
        plt.fill_between(comm_round, g1_b_reward_err0, g1_b_reward_err1, color='tab:green', alpha=0.2)
        plt.fill_between(comm_round, g1_m_reward_err0, g1_m_reward_err1, color='tab:red', alpha=0.2)
        # plt.plot(comm_round, g1_b_g2_b_loss, label='g1_b_g2_b')
        # plt.plot(comm_round, g1_b_g2_m_loss, label='g1_b_g2_m')

        plt.ylabel('Average Evaluation Loss', fontsize=30)
        plt.xlabel('Communication Round', fontsize=30)
        # plt.xlim(xmin=4)

        plt.title('Evaluation Loss of Benign and Malicious Agents on Agent {}\'s dataset'.format(agent_idx), fontsize=25, fontweight = 'bold', pad=20)

        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(
            '../results/visualization/FedCBO_emnist/avg_loss/agentwise_results/avg_loss_on_agent_{}_dataset.png'.format(agent_idx),
            bbox_inches='tight')
        plt.show()

    elif cluster_idx == 1:
        g2_b_reward_err0 = np.array(g2_b_loss) - 0.5*np.array(g2_b_std)
        g2_b_reward_err1 = np.array(g2_b_loss) + 0.5*np.array(g2_b_std)
        g2_m_reward_err0 = np.array(g2_m_loss) - 0.5*np.array(g2_m_std)
        g2_m_reward_err1 = np.array(g2_m_loss) + 0.5*np.array(g2_m_std)

        plt.plot(comm_round, g2_b_loss, label='g2_b', c='tab:green')
        plt.plot(comm_round, g2_m_loss, label='g2_m', c='tab:red')
        plt.fill_between(comm_round, g2_b_reward_err0, g2_b_reward_err1, color='tab:green', alpha=0.2)
        plt.fill_between(comm_round, g2_m_reward_err0, g2_m_reward_err1, color='tab:red', alpha=0.2)
        # plt.plot(comm_round, g1_b_g2_b_loss, label='g1_b_g2_b')
        # plt.plot(comm_round, g1_b_g2_m_loss, label='g1_b_g2_m')

        plt.ylabel('Average Evaluation Loss', fontsize=30)
        plt.xlabel('Communication Round', fontsize=30)
        # plt.xlim(xmin=4)

        plt.title('Evaluation Loss of Benign and Malicious Agents on Agent {}\'s dataset'.format(agent_idx), fontsize=25,
                  fontweight='bold', pad=20)

        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(
            '../results/visualization/FedCBO_emnist/avg_loss/agentwise_results/avg_loss_on_agent_{}_dataset.png'.format(
                agent_idx),
            bbox_inches='tight')
        plt.show()


def vis_max_diff(max_diff_by_round, benign_agent_list, malicious_agent_list):
    g1_b_g1_b_MD = []
    g1_b_g1_m_MD = []
    g1_b_g2_b_MD = []
    g1_b_g2_m_MD = []

    g2_b_g2_b_MD = []
    g2_b_g2_m_MD = []
    g2_b_g1_b_MD = []
    g2_b_g1_m_MD = []

    g1_b_g1_b_std = []
    g1_b_g1_m_std = []

    g2_b_g2_b_std = []
    g2_b_g2_m_std = []

    g1_b_idx = np.array(benign_agent_list[0])
    g1_m_idx = np.array(malicious_agent_list[0])
    g2_b_idx = np.array(benign_agent_list[1])
    g2_m_idx = np.array(malicious_agent_list[1])

    g2_b_ordered_idx = np.arange(len(g1_b_idx), len(g1_b_idx) + len(g2_b_idx))

    for comm_round in max_diff_by_round:
        max_diff_round_i = np.array(max_diff_by_round[comm_round])

        g1_b_MD = np.mean(max_diff_round_i[g1_b_idx], axis=0)
        g1_b_g1_b_MD.append(np.mean(g1_b_MD[g1_b_idx]))
        g1_b_g1_m_MD.append(np.mean(g1_b_MD[g1_m_idx]))
        g1_b_g2_b_MD.append(np.mean(g1_b_MD[g2_b_idx]))
        g1_b_g2_m_MD.append(np.mean(g1_b_MD[g2_m_idx]))

        g1_b_g1_b_std.append(np.std(g1_b_MD[g1_b_idx]))
        g1_b_g1_m_std.append(np.std(g1_b_MD[g1_m_idx]))


        g2_b_MD = np.mean(max_diff_round_i[g2_b_ordered_idx], axis=0)
        g2_b_g2_b_MD.append(np.mean(g2_b_MD[g2_b_idx]))
        g2_b_g2_m_MD.append(np.mean(g2_b_MD[g2_m_idx]))
        g2_b_g1_b_MD.append(np.mean(g2_b_MD[g1_b_idx]))
        g2_b_g1_m_MD.append(np.mean(g2_b_MD[g1_m_idx]))

        g2_b_g2_b_std.append(np.std(g2_b_MD[g2_b_idx]))
        g2_b_g2_m_std.append(np.std(g2_b_MD[g2_m_idx]))

    comm_round = np.arange(len(g1_b_g1_b_MD))


    g1_b_g1_b_reward_err0 = np.array(g1_b_g1_b_MD) - np.array(g1_b_g1_b_std)
    g1_b_g1_b_reward_err1 = np.array(g1_b_g1_b_MD) + np.array(g1_b_g1_b_std)
    g1_b_g1_m_reward_err0 = np.array(g1_b_g1_m_MD) - np.array(g1_b_g1_m_std)
    g1_b_g1_m_reward_err1 = np.array(g1_b_g1_m_MD) + np.array(g1_b_g1_m_std)

    plt.figure(figsize=(16, 10))
    plt.plot(comm_round, g1_b_g1_b_MD, label='g1_b_g1_b', c='tab:green')
    plt.plot(comm_round, g1_b_g1_m_MD, label='g1_b_g1_m', c='tab:red')
    plt.fill_between(comm_round, g1_b_g1_b_reward_err0, g1_b_g1_b_reward_err1, color='tab:green', alpha=0.2)
    plt.fill_between(comm_round, g1_b_g1_m_reward_err0, g1_b_g1_m_reward_err1, color='tab:red', alpha=0.2)
    plt.plot(comm_round, g1_b_g2_b_MD, label='g1_b_g2_b')
    plt.plot(comm_round, g1_b_g2_m_MD, label='g1_b_g2_m')

    plt.ylabel('Average Maximum Difference Class-wise Loss', fontsize=22)
    plt.xlabel('Communication Round', fontsize=30)
    # plt.xlim(xmin=4)

    plt.title('Maximum Difference Class-wise Loss on Group 1 Benign Agents\' Datasets', fontsize=25, fontweight = 'bold', pad=20)

    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(
        '../results/visualization/FedCB2O_emnist/Avg_results/avg_max_diff_g1_b_all_agents_alpha_2_Gfunc_30.png',
        bbox_inches='tight')
    plt.show()


    g2_b_g2_b_MD_err0 = np.array(g2_b_g2_b_MD) - np.array(g2_b_g2_b_std)
    g2_b_g2_b_MD_err1 = np.array(g2_b_g2_b_MD) + np.array(g2_b_g2_b_std)
    g2_b_g2_m_MD_err0 = np.array(g2_b_g2_m_MD) - np.array(g2_b_g2_m_std)
    g2_b_g2_m_MD_err1 = np.array(g2_b_g2_m_MD) + np.array(g2_b_g2_m_std)

    plt.figure(figsize=(16, 10))
    plt.plot(comm_round, g2_b_g2_b_MD, label='g2_b_g2_b', c='tab:green')
    plt.plot(comm_round, g2_b_g2_m_MD, label='g2_b_g2_m', c='tab:red')
    plt.fill_between(comm_round, g2_b_g2_b_MD_err0, g2_b_g2_b_MD_err1, color='tab:green', alpha=0.2)
    plt.fill_between(comm_round, g2_b_g2_m_MD_err0, g2_b_g2_m_MD_err1, color='tab:red', alpha=0.2)
    plt.plot(comm_round, g2_b_g1_b_MD, label='g2_b_g1_b')
    plt.plot(comm_round, g2_b_g1_m_MD, label='g2_b_g1_m')

    plt.ylabel('Average Maximum Difference Class-wise Loss', fontsize=22)
    plt.xlabel('Communication Round', fontsize=30)
    # plt.xlim(xmin=4)

    plt.title('Maximum Difference Class-wise Loss on Group 2 Benign Agents\' Datasets', fontsize=25, fontweight='bold',
              pad=20)

    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(
        '../results/visualization/FedCB2O_emnist/Avg_results/avg_max_diff_g2_b_all_agents_alpha_2_Gfunc_30.png',
        bbox_inches='tight')
    plt.show()



def vis_selection_time(selection_time, benign_agent_list, malicious_agent_list):
    avg_b_g1_b_g1 = 0
    avg_b_g1_b_g2 = 0
    avg_b_g1_m_g1 = 0
    avg_b_g1_m_g2 = 0

    avg_b_g2_b_g2 = 0
    avg_b_g2_b_g1 = 0
    avg_b_g2_m_g2 = 0
    avg_b_g2_m_g1 = 0

    benign_agent_idx_g1 = benign_agent_list[0]
    benign_agent_idx_g2 = benign_agent_list[1]
    mali_agent_idx_g1 = malicious_agent_list[0]
    mali_agent_idx_g2 = malicious_agent_list[1]

    for agent_id in benign_agent_idx_g1:
        selection_time_agent_i = selection_time[agent_id]

        avg_b_g1_b_g1 += np.mean(selection_time_agent_i[benign_agent_idx_g1])
        avg_b_g1_b_g2 += np.mean(selection_time_agent_i[benign_agent_idx_g2])
        avg_b_g1_m_g1 += np.mean(selection_time_agent_i[mali_agent_idx_g1])
        avg_b_g1_m_g2 += np.mean(selection_time_agent_i[mali_agent_idx_g2])

    avg_b_g1_b_g1 = int(avg_b_g1_b_g1 / len(benign_agent_idx_g1))
    avg_b_g1_b_g2 = int(avg_b_g1_b_g2 / len(benign_agent_idx_g1))
    avg_b_g1_m_g1 = int(avg_b_g1_m_g1 / len(benign_agent_idx_g1))
    avg_b_g1_m_g2 = int(avg_b_g1_m_g2 / len(benign_agent_idx_g1))

    for agent_id in benign_agent_idx_g2:
        selection_time_agent_i = selection_time[agent_id]

        avg_b_g2_b_g2 += np.mean(selection_time_agent_i[benign_agent_idx_g2])
        avg_b_g2_b_g1 += np.mean(selection_time_agent_i[benign_agent_idx_g1])
        avg_b_g2_m_g2 += np.mean(selection_time_agent_i[mali_agent_idx_g2])
        avg_b_g2_m_g1 += np.mean(selection_time_agent_i[mali_agent_idx_g1])

    avg_b_g2_b_g2 = int(avg_b_g2_b_g2 / len(benign_agent_idx_g2))
    avg_b_g2_b_g1 = int(avg_b_g2_b_g1 / len(benign_agent_idx_g2))
    avg_b_g2_m_g2 = int(avg_b_g2_m_g2 / len(benign_agent_idx_g2))
    avg_b_g2_m_g1 = int(avg_b_g2_m_g1 / len(benign_agent_idx_g1))

    categories_g1 = ['g1_b_g1_b', 'g1_b_g1_m', 'g1_b_g2_b', 'g1_b_g2_m']
    categories_g2 = ['g2_b_g2_b', 'g2_b_g2_m', 'g2_b_g1_b', 'g2_b_g1_m']
    avg_select_time_g1 = [avg_b_g1_b_g1, avg_b_g1_m_g1, avg_b_g1_b_g2, avg_b_g1_m_g2]
    avg_select_time_g2 = [avg_b_g2_b_g2, avg_b_g2_m_g2, avg_b_g2_b_g1, avg_b_g2_m_g1]

    color_label = ['tab:green', 'tab:red', 'tab:blue', 'tab:blue']

    fig, ax = plt.subplots(figsize=(16, 10))

    ax.bar(categories_g1, avg_select_time_g1, label=categories_g1, color=color_label)

    ax.set_ylabel('Average Selection Time', fontsize=30)
    ax.set_title('Average Selection Time of Benign Agents from Group 1', fontsize=30, fontweight = 'bold', pad=25)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('../results/visualization/FedCBO_cifar10/select_time_and_reward/Avg_results/avg_select_time_g1_b_Temp_0.25.png', bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(16, 10))

    ax.bar(categories_g2, avg_select_time_g2, label=categories_g2, color=color_label)

    ax.set_ylabel('Average Selection Time', fontsize=30)
    ax.set_title('Average Selection Time of Benign Agents from Group 2', fontsize=30, fontweight = 'bold', pad=25)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('../results/visualization/FedCBO_cifar10/select_time_and_reward/Avg_results/avg_select_time_g2_b_Temp_0.25.png', bbox_inches='tight')
    plt.show()


def parse_validation_losses(file_path):
    # Pattern to identify lines with validation losses
    pattern_selection_time = re.compile(r'Selection time of agent (\d+): \[([^\]]+)\]')
    pattern_selection_reward = re.compile(r'Estimated selection reward of agent (\d+): \[([^\]]+)\]')

    # Dictionary to hold validation losses
    selection_time_by_agent = {}
    selection_time_by_round = {}
    selection_reward_by_agent = {}
    selection_reward_by_round = {}

    current_round = -1
    with open(file_path, 'r') as file:
        for line in file:
            if "Communication round" in line:
                current_round = int(line.split(':')[1].strip())
                selection_time_by_round[current_round] = []
                selection_reward_by_round[current_round] = []

            # Check if the line contains selection time information
            match_selection_time = pattern_selection_time.search(line)
            if match_selection_time:
                agent_id = int(match_selection_time.group(1))
                selection_time = [float(x) for x in match_selection_time.group(2).split()]
                if agent_id not in selection_time_by_agent:
                    selection_time_by_agent[agent_id] = []
                selection_time_by_agent[agent_id].append(selection_time)
                selection_time_by_round[current_round].append(selection_time)

            # Check if the line contains selection reward information
            match_selection_reward = pattern_selection_reward.search(line)
            if match_selection_reward:
                agent_id = int(match_selection_reward.group(1))
                selection_reward = list(map(float, match_selection_reward.group(2).split(',')))
                if agent_id not in selection_reward_by_agent:
                    selection_reward_by_agent[agent_id] = []
                selection_reward_by_agent[agent_id].append(selection_reward)
                selection_reward_by_round[current_round].append(selection_reward)

    return selection_time_by_agent, selection_time_by_round, selection_reward_by_agent, selection_reward_by_round

def parse_classwise_losses(file_path):
    # Pattern to identify lines with class-wise losses
    classwise_loss_pattern = re.compile(r'The classwise loss of dataset from agent (\d+): (\[\[.*?\]\])')

    # Dictionary to hold class-wise losses and entropy
    results_classwise_loss_by_agent = {}
    results_classwise_loss_by_round = {}

    current_round = -1
    with open(file_path, 'r') as file:
        for line in file:
            # print(line)
            if "Communication round:" in line:
                current_round = int(line.split(':')[1].strip())
                results_classwise_loss_by_round[current_round] = []
            # Check if the line contains class-wise loss information
            match = classwise_loss_pattern.search(line)
            if match:
                agent_id = int(match.group(1))
                classwise_loss = ast.literal_eval(match.group(2))
                if agent_id not in results_classwise_loss_by_agent:
                    results_classwise_loss_by_agent[agent_id] = []
                results_classwise_loss_by_agent[agent_id].append(classwise_loss)
                results_classwise_loss_by_round[current_round].append(classwise_loss)


    return results_classwise_loss_by_agent, results_classwise_loss_by_round

def compute_average_loss_per_round(results_by_round):
    average_losses_per_round = {}
    for round_number, losses in results_by_round.items():
        # Flatten the list of loss arrays to compute the mean across all agents
        average_losses_per_round[round_number] = np.mean(losses, axis=0)
    return average_losses_per_round


def visualize_avg_loss_overtime(avg_loss, tag='average', agent_idx=None, save_path=None):
    if tag == 'average':
        # Split the array into benign and malicious agents and compute the mean among each group
        n_benign = 7 # Number of benign agents
        avg_benign_agents_loss = np.mean(avg_loss[:, :n_benign], axis=1)
        avg_malicious_agents_loss = np.mean(avg_loss[:, n_benign:], axis=1)

        plt.figure(figsize=(12,8))
        plt.plot(avg_benign_agents_loss, label='Benign Agents')
        plt.plot(avg_malicious_agents_loss, label='Malicious Agents')

        plt.title('Average Loss for Benign and Malicious Agents over Time')
        plt.xlabel('Communication Round')
        plt.ylabel('Average Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()
    elif tag == 'average_agentwise':
        plt.figure(figsize=(12, 8))
        for idx in range(10):
            plt.plot(avg_loss[:, idx], label='Agent {}'.format(idx+1))

        plt.title('Average Loss for Benign and Malicious Agents over Time')
        plt.xlabel('Communication Round')
        plt.ylabel('Average Loss')
        plt.legend(loc='upper right', prop={'size': 15})
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()
    elif tag == 'agentwise':
        plt.figure(figsize=(12, 8))
        for idx in range(10):
            plt.plot(avg_loss[:, idx], label='Agent {}'.format(idx))

        plt.title('Loss on Dataset {} for Benign and Malicious Agents over Time'.format(agent_idx))
        plt.xlabel('Communication Round')
        plt.ylabel('Loss on Dataset from Agent {}'.format(agent_idx))
        plt.legend(loc='upper right', prop={'size': 15})
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()
    elif tag == 'agentwise_average':
        n_benign = 7  # Number of benign agents
        avg_benign_agents_loss = np.mean(avg_loss[:, :n_benign], axis=1)
        avg_malicious_agents_loss = np.mean(avg_loss[:, n_benign:], axis=1)

        plt.figure(figsize=(12, 8))
        plt.plot(avg_benign_agents_loss, label='Benign Agents')
        plt.plot(avg_malicious_agents_loss, label='Malicious Agents')

        plt.title('Loss on Dataset {} for Benign and Malicious Agents over Time'.format(agent_idx))
        plt.xlabel('Communication Round')
        plt.ylabel('Loss on Dataset from Agent {}'.format(agent_idx))
        plt.legend(loc='upper right', prop={'size': 15})
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()






def visualize_losses(losses, t, tag='average', agent_idx=None):
    # Splitting the array into benign and malicious agents
    n_benign = 7  # Number of benign agents
    benign_losses = losses[:n_benign]
    malicious_losses = losses[n_benign:]

    # Indices for plotting
    indices = np.arange(len(losses))
    benign_indices = indices[:n_benign]
    malicious_indices = indices[n_benign:]

    # Creating the plot with lines connecting the points
    plt.figure(figsize=(12, 8))
    for agent_idx in range(10):
        plt.plot(losses[:, agent_idx], label='{}'.format(agent_idx+1))
        plt.legend(loc='upper right', prop={'size': 15})
    # plt.plot(benign_indices, benign_losses, 'o-', color='blue',
    #          label='Benign Agents')  # Connect benign points with a line
    # plt.plot(malicious_indices, malicious_losses, 'o-', color='red',
    #          label='Malicious Agents')  # Connect malicious points with a line

    # Adding plot labels and title
    plt.xlabel('Agent Index')
    plt.ylabel('Loss')
    plt.title('Losses of Benign and Malicious Agents at Round {}'.format(t))
    # plt.legend()

    # Show the plot with grid
    plt.grid(True, linestyle='--', alpha=0.7)
    # if tag == 'average':
    #     plt.savefig('../results/visualization/avg_loss_emnist/averaged_loss_round_{}.png'.format(t), bbox_inches='tight')
    # elif tag == 'agentwise':
    #     plt.savefig('../results/visualization/avg_loss_emnist/agent_{}_loss_round_{}.png'.format(agent_idx, t), bbox_inches='tight')
    plt.show()

def visualize_classwise_loss(loss_matrix, t, tag='average', agent_idx=None):
    loss_matrix = np.array(loss_matrix)
    # Agent groupings
    benign_cluster_1 = np.arange(0, 7)  # Agents 1-7
    malicious_cluster_1 = np.arange(7, 10)  # Agents 8-10
    # benign_cluster_2 = np.arange(10, 17)  # Agents 11-17
    # malicious_cluster_2 = np.arange(17, 20)  # Agents 18-20

    # Step 2: Compute average losses for each group
    avg_loss_benign_1 = np.mean(loss_matrix[benign_cluster_1], axis=0)
    avg_loss_malicious_1 = np.mean(loss_matrix[malicious_cluster_1], axis=0)
    # avg_loss_benign_2 = np.mean(loss_matrix[benign_cluster_2], axis=0)
    # avg_loss_malicious_2 = np.mean(loss_matrix[malicious_cluster_2], axis=0)

    # Step 3: Plotting the results
    plt.figure(figsize=(12, 8))
    plt.plot(avg_loss_benign_1, marker='o', label='Benign Agents in Cluster 1')
    plt.plot(avg_loss_malicious_1, marker='o', label='Malicious Agents in Cluster 1')
    # plt.plot(avg_loss_benign_2, marker='o', label='Benign Agents in Cluster 2')
    # plt.plot(avg_loss_malicious_2, marker='o', label='Malicious Agents in Cluster 2')

    plt.title('Average Classwise Loss for Agent Groups at Round {}'.format(t))
    plt.xlabel('Class Index')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    if tag == 'average':
        plt.savefig('../results/visualization/classwise_loss_emnist/avg_classwise_loss_round_{}.png'.format(t), bbox_inches='tight')
    elif tag == 'agentwise':
        plt.savefig('../results/visualization/classwise_loss_emnist/agent_{}_classwise_loss_round_{}.png'.format(agent_idx, t), bbox_inches='tight')
    plt.show()

def visualize_classwise_loss_agentwise(loss_matrix, t, agent_idx):
    plt.figure(figsize=(16, 10))
    for idx in range(10):
        plt.plot(loss_matrix[idx, :], label='{}'.format(idx + 1))

    plt.title('Class-wise Loss of for dataset of agent {} at t={}'.format(agent_idx+1, t))
    plt.xlabel('Classes')
    plt.ylabel('Class-wise Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.show()

def visualize_stat_classwise_loss(loss_matrix, vis_object='max_diff', vis_type='average', agent_idx=1, save_path=None):
    # Agent groupings
    benign_agents = np.arange(0, 7)
    benign_agents_removed = np.delete(benign_agents, np.where(benign_agents == agent_idx))
    malicious_agents = np.arange(7, 10)

    if vis_object == 'max_diff':
        title = 'Max_Diff on Dataset {} for Benign and Malicious Agents over Time'.format(agent_idx)
        xlabel = 'Communication Round'
        ylabel = 'Maximum Difference'

    elif vis_object == 'negative_entropy':
        xlabel = 'Communication Round'
        ylabel = 'Negative Entropy'

    elif vis_object == 'cross_entropy':
        title = 'Cross Entropy on Dataset {} for Benign and Malicious Agents over Time'.format(agent_idx)
        xlabel = 'Communication Round'
        ylabel = 'Cross Entropy'

    plt.figure(figsize=(12, 8))
    if vis_type == 'agentwise_average':
        avg_malicious = np.mean(loss_matrix[:, malicious_agents], axis=1)
        if vis_object == 'max_diff':
            avg_benign = np.mean(loss_matrix[:, benign_agents_removed], axis=1)
            # save_path = './results/visualization/emnist/max_diff/dataset_{}_avg_max_diff_over_time.png'.format(agent_idx)
        elif vis_object == 'negative_entropy':
            avg_benign = np.mean(loss_matrix[:, benign_agents], axis=1)
            title = 'Negative Entropy on Dataset {} for Benign and Malicious Agents over Time'.format(agent_idx)
            # save_path = './results/visualization/emnist/negative_entropy/dataset_{}_avg_negative_entropy_over_time.png'.format(agent_idx)
        elif vis_object == 'cross_entropy':
            avg_benign = np.mean(loss_matrix[:, benign_agents_removed], axis=1)
            # save_path = './results/visualization/emnist/cross_entropy/dataset_{}_avg_max_diff_over_time.png'.format(agent_idx)

        plt.plot(avg_benign, label='Benign Agents (Avg)')
        plt.plot(avg_malicious, label='Malicious Agents (Avg)')

    elif vis_type == 'agentwise':
        for idx in range(10):
            plt.plot(loss_matrix[:, idx], label='Agent {}'.format(idx+1))
        # if vis_object == 'max_diff':
        #     save_path = './results/visualization/emnist/max_diff/dataset_{}_agentwise_max_diff_over_time.png'.format(agent_idx)
        if vis_object == 'negative_entropy':
            title = 'Negative Entropy on Dataset {} for Benign and Malicious Agents over Time'.format(agent_idx)
            # save_path = './results/visualization/emnist/negative_entropy/negative_entropy_on_dataset_{}_over_time.png'.format(agent_idx)
        # elif vis_object == 'cross_entropy':
        #     save_path = './results/visualization/emnist/cross_entropy/dataset_{}_agentwise_cross_entropy_over_time.png'.format(agent_idx)
    elif vis_type == 'average':
        avg_benign = np.mean(loss_matrix[:, benign_agents], axis=1)
        avg_malicious = np.mean(loss_matrix[:, malicious_agents], axis=1)
        if vis_object == 'negative_entropy':
            title = 'Average Negative Entropy for Benign and Malicious Agents over Time'
            # save_path = './results/visualization/emnist/negative_entropy/avg_negative_entropy_over_time.png'

            plt.plot(avg_benign, label='Benign Agents (Avg)')
            plt.plot(avg_malicious, label='Malicious Agents (Avg)')
    elif vis_type == 'average_agentwise':
        for idx in range(10):
            plt.plot(loss_matrix[:, idx], label='Agent {}'.format(idx+1))
        if vis_object == 'negative_entropy':
            title = 'Average Negative Entropy for Benign and Malicious Agents over Time'
            # save_path = './results/visualization/emnist/negative_entropy/average_agentwise_negative_entropy_over_time.png'

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(prop={'size': 15})
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()




    # # Compute average entropy for each group
    # avg_entropy_benign = np.mean(entropy_matrix[:, benign_agents], axis=1)
    # avg_entropy_mali = np.mean(entropy_matrix[:, malicious_agents], axis=1)
    #
    # # Plotting the results
    # plt.figure(figsize=(16, 10))
    # if tag == 'average':
    #     plt.plot(avg_entropy_benign, marker='o', label='Benign Agents (Avg)')
    #     plt.plot(avg_entropy_mali, marker='o', label='Malicious Agents (Avg)')
    #     plt.legend()
    # elif tag == 'agentwise':
    #     for agent_idx in range(10):
    #         plt.plot(entropy_matrix[:, agent_idx], label='{}'.format(agent_idx+1))
    #         plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    #
    # plt.title('Average Entropy of Agents')
    # plt.xlabel('Communication Rounds')
    # plt.ylabel('Average Entropy')
    # plt.grid(True, linestyle='--', alpha=0.7)
    # # if tag == 'average':
    # #     plt.savefig('../results/visualization/avg_entropy_two_type_agents.png')
    # # elif tag == 'agentwise':
    # #     plt.savefig('../results/visualization/avg_entropy_each_agent.png')
    # plt.show()

def vis_constrained_set(t, images_save_path, particle_positions, consensus_point):
    def function_f(x1, x2):
        return x1 ** 2 + x2 ** 2

    # Create a meshgrid for the visualization
    x1 = np.linspace(-3, 3, 400)
    x2 = np.linspace(-3, 3, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = function_f(X1, X2)

    # Define the ellipse equation parameters
    a = np.sqrt(2)  # Semi-major axis
    b = 1  # Semi-minor axis

    # Parameterize the ellipse
    theta = np.linspace(0, 2 * np.pi, 100)
    x1_ellipse = a * np.cos(theta) - 1  # Center at (-1, 0)
    x2_ellipse = b * np.sin(theta)

    particle_positions = particle_positions[np.where((particle_positions[:, 0] <= 3) & (particle_positions[:, 0] >= -3) & (particle_positions[:, 1] <= 3) & (particle_positions[:, 1] >= -3))]

    # Plotting the contour map with the correctly formatted label for the ellipse
    plt.figure(figsize=(12, 10))
    contour = plt.contourf(X1, X2, Z, 20, cmap='GnBu')
    plt.plot(x1_ellipse, x2_ellipse, color='black')  # Adding the ellipse
    plt.scatter(0.41, 0, c='red', marker='*', s=180, zorder=3)
    plt.scatter(particle_positions[:, 0], particle_positions[:, 1], c='darkorange', s=60)
    plt.scatter(consensus_point[0], consensus_point[1], c='darkred', marker='v', s=100, zorder=4)
    # plt.xlabel('$x_1$')
    # plt.ylabel('$x_2$')
    plt.title(f'Particles Positions at time T={t}', fontsize=24)
    images_save_name = os.path.join(images_save_path, f'img_{t}.png')
    plt.savefig(images_save_name, transparent=False, facecolor='white', bbox_inches='tight')
    # plt.show()

def landscape_vis():
    # Define the function
    def L(x, y):
        return ((x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2) / 100

    def G(x, y):
        return x ** 2 + y ** 2

    # Generate values for x and y
    x = np.linspace(-6, 6, 400)
    y = np.linspace(-6, 6, 400)
    X, Y = np.meshgrid(x, y)
    Z = L(X, Y)

    # Compute g(x, y) over the same grid
    G_values = G(X, Y)
    custom_levels_g = np.concatenate((
        np.linspace(3, 9, 2),  # Dense near the origin
        np.linspace(18, 30, 2)  # Less dense further out
    ))

    # Create the figure and two subplots
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(21, 9), sharex=True, sharey=True)

    ### Subplot 1 ###
    ax1.scatter(x=2.5, y=2.0, marker='*', c='red', s=200, label=r'$\theta_{good}^*$', zorder=3)
    ax1.scatter(x=-2.805, y=3.1313, marker='X', c='tab:blue', s=150, zorder=3)
    ax1.scatter(x=-3.7793, y=-3.2832, marker='X', c='tab:blue', s=150, zorder=3)
    ax1.scatter(x=3.5844, y=-2.2, marker='X', c='tab:blue', s=150, zorder=3)
    contour1 = ax1.contourf(X, Y, Z, levels=50, cmap=plt.cm.viridis.reversed(), zorder=1)

    # Contour for G(x, y)
    contour_g1 = ax1.contour(X, Y, G_values, levels=custom_levels_g, colors='white',
                             linewidths=2, linestyles="dashed", alpha=0.7, zorder=2)
    for collection in contour_g1.collections:
        collection.set_dashes([(0, (12, 11))])

    specific_contour_levels = [0, 2]
    # ax1.clabel(contour_g1, inline=True, fontsize=16, fmt=r'$G(\theta)$', colors='black',
    #            levels=[contour_g1.levels[i] for i in specific_contour_levels])
    ax1.legend(prop={'size': 20})
    # ax1.text(-5.5, -5.5, r'$L(\theta)$', fontsize=20, color='black', va='bottom', ha='left')
    ax1.set_axis_off()

    ### Subplot 2 ###
    np.random.seed(2)  # For reproducibility
    random_points_x = np.random.uniform(-5, 5, 6)
    random_points_y = np.random.uniform(-5, 5, 6)
    points = np.array(list(zip(random_points_x, random_points_y)) + [[2.4, 1.4], [3.8, -2.0], [-2.4, 3.1], [-3.5, -3.9]])

    L_values = L(points[:, 0], points[:, 1])
    beta_quantile = np.quantile(L_values, 0.4)
    selected_points = np.where(L_values <= beta_quantile)[0]
    remaning_points = np.where(L_values > beta_quantile)[0]

    G_values_selected = G(points[selected_points, 0], points[selected_points, 1])
    weights = np.exp(-0.3 * G_values_selected)
    consensus_point = np.average(points[selected_points, :], weights=weights, axis=0)

    ax2.scatter(x=2.5, y=2.0, marker='*', c='red', s=200, label=r'$\theta_{good}^*$', zorder=3)
    ax2.scatter(x=-2.805, y=3.1313, marker='X', c='tab:blue', s=150, zorder=3)
    ax2.scatter(x=-3.7793, y=-3.2832, marker='X', c='tab:blue', s=150, zorder=3)
    ax2.scatter(x=3.5844, y=-2.2, marker='X', c='tab:blue', s=150, zorder=3)
    ax2.scatter(x=points[selected_points, 0], y=points[selected_points, 1], marker='o', c='darkorange', s=150, zorder=3)
    ax2.scatter(x=points[remaning_points, 0], y=points[remaning_points, 1], marker='o',
                facecolors='none', edgecolors='darkorange', s=150, linewidths=2, zorder=3)
    ax2.scatter(x=consensus_point[0], y=consensus_point[1], marker='D', c='forestgreen', s=150, zorder=3, label='CSP')
    ax2.contourf(X, Y, Z, levels=50, cmap=plt.cm.viridis.reversed(), zorder=1)

    # Contour for G(x, y)
    contour_g2 = ax2.contour(X, Y, G_values, levels=custom_levels_g, colors='white',
                             linewidths=2, linestyles="dashed", alpha=0.7, zorder=2)
    for collection in contour_g2.collections:
        collection.set_dashes([(0, (12, 11))])

    ax2.legend(prop={'size': 20})
    ax2.set_axis_off()

    plt.savefig('../results/bilevel_opt_CB2O_poster.png', transparent=False, facecolor='white', bbox_inches='tight', dpi=300)
    plt.show()



if __name__ == '__main__':
    landscape_vis()
    # visualize_obj_func()
    # file_path = '../results/visualization/FedCB2O_emnist/max_diff_alpha_2_Gfunc_T_30.txt'
    # max_diff_by_agent, max_diff_by_round = parse_max_diff_loss(file_path=file_path)
    # 
    # # Total number of agents in each cluster
    # total_agents_per_cluster = 50
    #
    # # Number of benign and malicious agents in each cluster
    # benign_agents_per_cluster = 35
    # malicious_agents_per_cluster = total_agents_per_cluster - benign_agents_per_cluster
    #
    # # Creating indices for benign and malicious agents for each cluster
    # benign_agents_cluster_1 = np.arange(0, benign_agents_per_cluster)
    # malicious_agents_cluster_1 = np.arange(benign_agents_per_cluster, total_agents_per_cluster)
    #
    # benign_agents_cluster_2 = np.arange(total_agents_per_cluster, total_agents_per_cluster + benign_agents_per_cluster)
    # malicious_agents_cluster_2 = np.arange(total_agents_per_cluster + benign_agents_per_cluster,
    #                                        total_agents_per_cluster * 2)
    #
    # benign_agent_list = np.stack((benign_agents_cluster_1, benign_agents_cluster_2))
    # mali_agent_list = np.stack((malicious_agents_cluster_1, malicious_agents_cluster_2))
    #
    # vis_max_diff(max_diff_by_round=max_diff_by_round, benign_agent_list=benign_agent_list, malicious_agent_list=mali_agent_list)
    #
    # vis_loss(selection_reward=selection_reward_by_round, temp=0.25, benign_agent_list=benign_agent_list, malicious_agent_list=mali_agent_list)
    #
    # for agent_idx in np.array([7,9,19]):
    #     vis_loss_agentwise(agent_idx=agent_idx, cluster_idx=0, selection_reward=selection_reward_by_round, temp=0.25, benign_agent_list=benign_agent_list, malicious_agent_list=mali_agent_list)
    #
    # for agent_idx in np.array([53,65,79]):
    #     vis_loss_agentwise(agent_idx=agent_idx, cluster_idx=1, selection_reward=selection_reward_by_round, temp=0.25, benign_agent_list=benign_agent_list, malicious_agent_list=mali_agent_list)

    # for agent_idx in np.array([53,65,79]):
    #     vis_selection_reward_agentwise(agent_idx=agent_idx, cluster_idx=1, selection_reward=selection_reward_by_round, benign_agent_list=benign_agent_list, malicious_agent_list=mali_agent_list)

    # vis_selection_reward(selection_reward=selection_reward_by_round, benign_agent_list=benign_agent_list, malicious_agent_list=mali_agent_list)

    # vis_selection_time(selection_time=selection_time_final_round, benign_agent_list=benign_agent_list, malicious_agent_list=mali_agent_list)

    # landscape_vis()
    # vis_constrained_set()
    # constrained_opt_obj = contrained_OPT_obj()
    # vis_contour_plot()
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--experiment_name', type=str, default='test')
    # parser.add_argument('--result_path', type=str, default='./results/visualization')
    # parser.add_argument('--agent_idx', type=int, default=1)
    # parser.add_argument('--data_name', type=str, default='emnist')
    # parser.add_argument('--alg', type=str, default='FedCBO')
    # parser.add_argument('--vis_type', type=str, default='average', choices=['average', 'average_agentwise', 'agentwise', 'agentwise_average', ])
    # parser.add_argument('--vis_object', type=str, default='avg_loss', choices=['avg_loss', 'classwise_loss', 'negative_entropy', 'max_diff', 'cross_entropy'])
    # parser.add_argument('--file_path', type=str)
    #
    # args = parser.parse_args()
    #
    # logging.basicConfig(level=logging.INFO)
    # logging.info("Running training for {}".format(args.experiment_name))
    #
    # output_path = os.path.join('./results/visualization', '{}_{}'.format(args.alg, args.data_name))
    # output_path = os.path.join(output_path, args.vis_object)
    #
    # # Parse the file
    # if args.vis_object == 'avg_loss':
    #     avg_loss_by_agent, avg_loss_by_round = parse_validation_losses(args.file_path)
    #     avg_loss_per_round = compute_average_loss_per_round(results_by_round=avg_loss_by_round)
    #
    #     if args.vis_type == 'average':
    #         avg_loss_per_round = np.array(list(avg_loss_per_round.values()))
    #         save_path = os.path.join(output_path, 'avg_loss_over_time.png')
    #         visualize_avg_loss_overtime(avg_loss=avg_loss_per_round, tag='average', save_path=save_path)
    #     elif args.vis_type == 'average_agentwise':
    #         avg_loss_per_round = np.array(list(avg_loss_per_round.values()))
    #         save_path = os.path.join(output_path, 'avg_loss_agentwise_over_time.png')
    #         visualize_avg_loss_overtime(avg_loss=avg_loss_per_round, tag='average_agentwise', save_path=save_path)
    #     elif args.vis_type == 'agentwise':
    #         avg_loss_agent = np.array(avg_loss_by_agent[args.agent_idx])
    #         save_path = os.path.join(output_path, 'avg_loss_on_dataset_{}_over_time.png'.format(args.agent_idx))
    #         visualize_avg_loss_overtime(avg_loss=avg_loss_agent, tag='agentwise', agent_idx=args.agent_idx, save_path=save_path)
    #     elif args.vis_type == 'agentwise_average':
    #         avg_loss_agent = np.array(avg_loss_by_agent[args.agent_idx])
    #         save_path = os.path.join(output_path, 'avg_loss_agentwise_on_dataset_{}_over_time.png'.format(args.agent_idx))
    #         visualize_avg_loss_overtime(avg_loss=avg_loss_agent, tag='agentwise_average', agent_idx=args.agent_idx, save_path=save_path)
    #
    # elif args.vis_object == 'max_diff':
    #     if args.vis_type == 'agentwise':
    #         save_path = os.path.join(output_path, 'dataset_{}_agentwise_max_diff_over_time.png'.format(args.agent_idx))
    #     elif args.vis_type == 'agentwise_average':
    #         save_path = os.path.join(output_path, 'dataset_{}_avg_max_diff_over_time.png'.format(args.agent_idx))
    #     classwise_loss_by_agent, classwise_loss_by_round = parse_classwise_losses(args.file_path)
    #     classwise_loss_agent = np.array(classwise_loss_by_agent[args.agent_idx])
    #     classwise_loss_agent_self_eval = classwise_loss_agent[:, args.agent_idx-1, :]
    #     classwise_loss_agent_self_eval = np.tile(classwise_loss_agent_self_eval[:, np.newaxis, :], (1,10,1))
    #     max_diff = np.max(classwise_loss_agent - classwise_loss_agent_self_eval, axis=2)
    #     visualize_stat_classwise_loss(loss_matrix=max_diff, vis_object='max_diff', vis_type=args.vis_type, agent_idx=args.agent_idx, save_path=save_path)
    #
    # elif args.vis_object == 'negative_entropy':
    #     if args.vis_type == 'average':
    #         save_path = os.path.join(output_path, 'average_negative_entropy_over_time.png')
    #     elif args.vis_type == 'average_agentwise':
    #         save_path = os.path.join(output_path, 'average_agentwise_negative_entropy_over_time.png')
    #     elif args.vis_type == 'agentwise':
    #         save_path = os.path.join(output_path, 'dataset_{}_negative_entropy_over_time.png'.format(args.agent_idx))
    #     elif args.vis_type == 'agentwise_average':
    #         save_path = os.path.join(output_path, 'dataset_{}_agentwise_negative_entropy_over_time.png'.format(args.agent_idx))
    #     classwise_loss_by_agent, classwise_loss_by_round = parse_classwise_losses(args.file_path)
    #     if args.vis_type in ['average', 'average_agentwise']:
    #         average_classwise_loss_per_round = compute_average_loss_per_round(results_by_round=classwise_loss_by_round)
    #         agent_avg_entropy = -entropy(list(average_classwise_loss_per_round.values()), axis=2) # Rounds X Num_agents
    #         visualize_stat_classwise_loss(loss_matrix=agent_avg_entropy, vis_object=args.vis_object, vis_type=args.vis_type, save_path=save_path)
    #     elif args.vis_type in ['agentwise', 'agentwise_average']:
    #         classwise_loss_agent = np.array(classwise_loss_by_agent[args.agent_idx])
    #         agent_entropy = -entropy(classwise_loss_agent, axis=2)  # Rounds X Num_agents
    #         visualize_stat_classwise_loss(loss_matrix=agent_entropy, vis_object=args.vis_object, vis_type=args.vis_type, agent_idx=args.agent_idx, save_path=save_path)
    #
    # elif args.vis_object == 'cross_entropy':
    #     if args.vis_type == 'agentwise':
    #         save_path = os.path.join(output_path, 'dataset_{}_agentwise_cross_entropy_over_time.png'.format(args.agent_idx))
    #     elif args.vis_type == 'agentwise_average':
    #         save_path = os.path.join(output_path, 'dataset_{}_avg_cross_entropy_over_time.png'.format(args.agent_idx))
    #     classwise_loss_by_agent, classwise_loss_by_round = parse_classwise_losses(args.file_path)
    #     classwise_loss_agent = np.array(classwise_loss_by_agent[args.agent_idx])
    #     classwise_loss_agent_self_eval = classwise_loss_agent[:, args.agent_idx - 1, :]
    #     classwise_loss_agent_self_eval = np.tile(classwise_loss_agent_self_eval[:, np.newaxis, :], (1, 10, 1))
    #     agent_cross_entropy = entropy(classwise_loss_agent, classwise_loss_agent_self_eval, axis=2)
    #     visualize_stat_classwise_loss(loss_matrix=agent_cross_entropy, vis_object=args.vis_object, vis_type=args.vis_type, agent_idx=args.agent_idx, save_path=save_path)


    # avg_loss_agent_1 = np.array(avg_loss_by_agent[1])
    # print('Shape of avg_loss_agent_1:', avg_loss_agent_1.shape)
    # visualize_losses(losses=avg_loss_agent_1, t=0)
    # for t in range(0, 101, 10):
    #     visualize_losses(losses=avg_loss_by_agent[5][t], t=t, tag='agentwise', agent_idx=5)
    #     visualize_losses(losses=avg_loss_per_round[t], t=t, tag='average')

    # file_path = '../results/visualization/loss_FedCBO_emnist_small_scale_alpha_5.txt'
    # # file_path = '../results/visualization/different_losses_Bilevel_FedCBO_small_scale.txt'
    #
    # # Parse the file
    # classwise_loss_by_agent, classwise_loss_by_round = parse_classwise_losses(file_path)
    # classwise_loss_agent_1 = np.array(classwise_loss_by_agent[1])
    # classwise_loss_agent_1_self_evaluation = classwise_loss_agent_1[:, 0, :]
    # classwise_loss_agent_1_self_evaluation = np.tile(classwise_loss_agent_1_self_evaluation[:, np.newaxis, :], (1, 10, 1))
    # max_diff = np.max(classwise_loss_agent_1-classwise_loss_agent_1_self_evaluation, axis=2)
    # print(max_diff.shape)
    # visualize_entropy(entropy_matrix=max_diff, tag='agentwise')
    # cross_entropy_agent1_dataset = entropy(classwise_loss_agent_1, classwise_loss_agent_1_self_evaluation, axis=2)
    # print(cross_entropy_agent1_dataset.shape)
    # visualize_entropy(entropy_matrix=cross_entropy_agent1_dataset, tag='agentwise')
    # for t in range(0, 101, 10):
    #     visualize_classwise_loss_agentwise(classwise_loss_agent_1[t], t=t, agent_idx=0)
    # avg_entropy_agent1_dataset = entropy(classwise_loss_agent_1, axis=2)
    # visualize_entropy(entropy_matrix=avg_entropy_agent1_dataset, tag='average')
    # average_classwise_loss_per_round = compute_average_loss_per_round(results_by_round=classwise_loss_by_round)
    # agent_avg_entropy = entropy(list(average_classwise_loss_per_round.values()), axis=2) # Rounds X Num_agents
    # visualize_entropy(entropy_matrix=agent_avg_entropy, tag='agentwise')

    # for t in range(0, 101, 10):
    #     # visualize_classwise_loss(loss_matrix=classwise_loss_by_agent[3][t], t=t, tag='agentwise', agent_idx=3)
    #     visualize_classwise_loss(loss_matrix=average_classwise_loss_per_round[t], t=t, tag='average')

    # for t in range(0, 201, 20):
    #     visualize_classwise_loss(loss_matrix=average_classwise_loss_per_round[t], t=t, tag='average')

    # print('avg classwise loss at round 0: ', average_classwise_loss_per_round[0].shape)

    # file_path = '../results/visualization/loss_FedCBO_alpha_10_prop_source_class_0.5_mali_1500.txt'
    #
    # # Parse the file
    # losses_by_agent, losses_by_round = parse_validation_losses(file_path)
    # average_losses_per_round = compute_average_loss_per_round(results_by_round=losses_by_round)
    # for t in range(0, 201,20):
    #     visualize_losses(losses=losses_by_agent[5][t], t=t, tag='agentwise', agent_idx=5)













