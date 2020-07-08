from MDSimsEval.rmsf_baseline_models import bootstrap_dataset, ResidueMajority, \
    AggregatedResidues
from MDSimsEval.utils import create_analysis_actor_dict

from tqdm import tqdm
from scipy import stats
import numpy as np
import pandas as pd
import pickle


# Read the data
analysis_actors_dict = create_analysis_actor_dict('../datasets/New_AI_MD')


def calculate_accuracy(model, ligands_dict):
    # We define a function that will help us calculate the accuracy of our feature selection
    acc = 0
    for which_agon in ligands_dict['Agonists']:
        label = model.predict(which_agon)
        if label == 1:
            acc += 1

    for which_antagon in ligands_dict['Antagonists']:
        label = model.predict(which_antagon)
        if label == 0:
            acc += 1

    return acc / (len(ligands_dict['Agonists']) + len(ligands_dict['Antagonists'])) * 100


# IMPORTANT: For any RMSF analysis always initialize rmsf_cache as an empty dict and pass it as an argument to the
# rmsf methods
rmsf_cache = {}

# Windows we will evaluate our feature selection on
windows = [[0, 2500], [0, 1250], [1250, 2500], [0, 500], [500, 1000], [1000, 1500], [1500, 2000], [2000, 2500]]

# Create the bootstrap samples
train_dicts, validation_dicts = bootstrap_dataset(analysis_actors_dict, samples=3, sample_size=20)

total_metrics = {}  # We will save our accuracies of each window on this dict

for start, stop in windows:
    accs = []
    model = AggregatedResidues(start, stop, rmsf_cache, method=np.mean)
    # model = ResidueMajority(start, stop, rmsf_cache, np.mean)

    # The loop is slow at each 1st iteration but speeds due to rmsf_cache
    for train_dict, validation_dict in tqdm(list(zip(train_dicts, validation_dicts)), desc=f'Window {start} - {stop}'):
        model.fit(train_dict, residues=np.arange(290))

        accs.append([calculate_accuracy(model, train_dict), calculate_accuracy(model, validation_dict)])

    accs = np.array(accs)  # Transform to numpy array for the mean, sem below
    mean_accs = np.mean(accs, axis=0)

    # Calculating the Standard Error of the Mean gives us an indication of the fluctuations of the accuracies
    # High sem suggests that we need to increase the number of bootstrapped samples
    sem_accs = stats.sem(accs, axis=0)

    # Save the results on the dictionary that will be transformed to a DataFrame
    total_metrics[f'{start} - {stop}'] = [mean_accs[0], mean_accs[1], sem_accs[0], sem_accs[1]]

# Save the results using pickle for future use
with open('cache/baseline_models/aggregated_acc_all_res_mean.pkl', 'wb') as handle:
    pickle.dump(total_metrics, handle)

print(pd.DataFrame.from_dict(total_metrics, orient='index',
                             columns=['Mean Train Accuracy', 'Mean Test Accuracy',
                                      'Stde Train Accuracy', 'Stde Test Accuracy']).round(decimals=2))
