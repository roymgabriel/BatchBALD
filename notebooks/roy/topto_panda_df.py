import blackhc.notebook
import numpy as np
import os
import re
import pandas as pd
os.getcwd()
import al_notebook.results_loader as rl
stores = rl.load_experiment_results('EMORY_COVID/binary')
stores.keys()

store = {}

def append_dataframe_label(file_path, acquision_function, seed):
    # Read the content of the .py file
    with open(file_path, 'r') as file:
        file_content = file.read()

    # Execute the content of the .py file to get the 'store' dictionary
    exec(file_content)

    # Initialize an empty list to store the chosen_targets from each iteration
    all_chosen_targets = []

    value_dictionary = pd.DataFrame(columns=['acquision_function', 'seed', 'iteration_number', 'class_0%', 'class_1%', 'train_time', 'batch_time'])
    temp_iter = 0
    
    # Loop through each iteration and extract the 'chosen_targets'
    for iteration in store['iterations']:
        #all_chosen_targets.append(iteration['chosen_targets'])
        percentage_one = np.sum(iteration['chosen_targets'])*100/np.size(iteration['chosen_targets'])
        percentage_zero = 100 - percentage_one
        train_time = iteration['train_model_elapsed_time']
        batch_time = iteration['batch_acquisition_elapsed_time']
        value_dictionary = value_dictionary._append({
            'acquision_function': acquision_function,
            'seed': seed,
            'iteration_number': temp_iter,
            'class_0%': percentage_zero,
            'class_1%': percentage_one,
            'train_time': train_time,
            'batch_time': batch_time
        }, ignore_index = True)
        temp_iter += 1
        
    # Convert the list to a numpy array for stacking
    # stacked_chosen_targets = np.array(all_chosen_targets)

    # Print the stacked chosen targets
    return value_dictionary #[percentage_one,percentage_zero,train_time,batch_time]

# print(append_dataframe_label('/Users/toptotoro_air/Desktop/Georgia-Tech/professor_adibi/with_roy/BatchBALD/laaos_results/EMORY_COVID/binary/covid_full_resnet_binary_scratch_entropy_58.py', 1, 2))

binary_dictionary = pd.DataFrame(columns=['acquision_function', 'seed', 'iteration_number', 'class_0%', 'class_1%', 'train_time', 'batch_time'])


directory_path = './laaos_results/EMORY_COVID/binary/'

# Loop through each file in the directory
for filename in os.listdir(directory_path):
    # Check if the file is a .py file
    if filename.endswith('.py'):
        # Construct the full file path
        file_path = os.path.join(directory_path, filename)
        if "_entropy_" in filename:
            pattern = r'_entropy_(\d+)\.py'
            match = re.search(pattern, filename)
            number = match.group(1)
            # binary_dictionary.join(append_dataframe_label(file_path = file_path, acquision_function = "_entropy_", seed = number))
            binary_dictionary = pd.concat([binary_dictionary,append_dataframe_label(file_path = file_path, acquision_function = "_entropy_", seed = number)], ignore_index = True )
            
        elif "_lc_" in filename:
            pattern = r'_lc_(\d+)\.py'
            match = re.search(pattern, filename)
            number = match.group(1)
            binary_dictionary = pd.concat([binary_dictionary,append_dataframe_label(file_path = file_path, acquision_function = "_lc_", seed = number)], ignore_index = True )

        elif "_meanstd_" in filename:
            pattern = r'_meanstd_(\d+)\.py'
            match = re.search(pattern, filename)
            number = match.group(1)
            binary_dictionary = pd.concat([binary_dictionary,append_dataframe_label(file_path = file_path, acquision_function = "_meanstd_", seed = number)], ignore_index = True )

        elif "_ms_" in filename:
            pattern = r'_ms_(\d+)\.py'
            match = re.search(pattern, filename)
            number = match.group(1)
            binary_dictionary = pd.concat([binary_dictionary,append_dataframe_label(file_path = file_path, acquision_function = "_ms_", seed = number)], ignore_index = True )

        elif "_multibald_" in filename:
            pattern = r'_multibald_(\d+)\.py'
            match = re.search(pattern, filename)
            number = match.group(1)
            binary_dictionary = pd.concat([binary_dictionary,append_dataframe_label(file_path = file_path, acquision_function = "_multibald_", seed = number)], ignore_index = True )

        elif "_random_" in filename:
            pattern = r'_random_(\d+)\.py'
            match = re.search(pattern, filename)
            number = match.group(1)
            binary_dictionary = pd.concat([binary_dictionary,append_dataframe_label(file_path = file_path, acquision_function = "_random_", seed = number)], ignore_index = True )

        elif "_vr_" in filename:
            pattern = r'_vr_(\d+)\.py'
            match = re.search(pattern, filename)
            number = match.group(1)
            binary_dictionary = pd.concat([binary_dictionary,append_dataframe_label(file_path = file_path, acquision_function = "_vr_", seed = number)], ignore_index = True )

        else:
            pass


binary_dictionary
# binary_dictionary[binary_dictionary['seed'] == '58']