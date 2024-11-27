import re
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
sns.set_context("poster")

def extract_training_results_from_file(file_path):
    # Open and read the content of the file
    with open(file_path, 'r') as file:
        data = file.read()

    # Step 2: Use regular expressions to extract the relevant data
    # Extract communication rounds
    comm_rounds = re.findall(r'Communication round:\s(\d+)', data)
    # Extract corresponding 'Average acc of benign agents'
    benign_acc = re.findall(r'Average acc of benign agents:\s([\d\.]+)', data)

    # Step 3: Convert extracted data to appropriate data types
    comm_rounds = np.array([int(round) for round in comm_rounds])
    benign_acc = np.array([float(acc) for acc in benign_acc])

    return comm_rounds, benign_acc

def process_multiple_files(common_path, file_name_lists):
    all_results = {}

    # Loop through each file name
    for file_name in file_name_lists:
        file_path = os.path.join(common_path, file_name)

        # Extract the data from each file
        comm_rounds, benign_acc = extract_training_results_from_file(file_path)

        # Get the file name without the directory and extension
        file_name = os.path.basename(file_name).replace('.txt', '')

        # Store the results in a dictionary, with the file name as the key
        all_results[file_name] = {
            "CommRounds": comm_rounds,
            "AvgAccBenign": benign_acc
        }

    return all_results

def process_keys(dict_keys):
    new_file_names = []

    for key in dict_keys:
        if 'agent_selection_prob' in key:
            # Extract the number after 'alpha_'
            alpha_value = re.search(r'alpha_(\d+)', key)
            if alpha_value:
                # Create the new name in the format "alpha_{number}"
                new_name = f"PrAS_Alpha{alpha_value.group(1)}"
                new_file_names.append(new_name)
        elif 'random_agent_selection' in key:
            # For keys containing 'random_agent_selection', use "RandAgentSelect"
            new_file_names.append('RaAS')

    return new_file_names


def visualize_acc(results):
    # Create the plot
    plt.figure(figsize=(12, 8))
    labels = process_keys(results.keys())
    for info, label in zip(results.values(), labels):
        plt.plot(info['CommRounds'], info['AvgAccBenign'], label=label)
    # plt.title("Average Accuracy of Benign Agents Across Communication Rounds")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

if __name__ == "__main__":
    common_path = '../results/visualization/FedCBO_cifar10'
    file_name_lists = ['result_agent_selection_prob_alpha_1.txt', 'result_agent_selection_prob_alpha_10.txt', 'result_agent_selection_prob_alpha_1.txt', 'result_random_agent_selection.txt']
    results = process_multiple_files(common_path, file_name_lists)
    visualize_acc(results=results)
