from MDSimsEval.rmsf_bootstrapped_analysis import bootstrapped_residue_analysis, find_top
from MDSimsEval.utils import create_analysis_actor_dict

from scipy import stats
import pandas as pd

# Parameters to be set
outer_samples_numb = 500
sample_size = 20  # e.g. if set to 20 each sample contains 20 unique agonists and 20 unique antagonists
top_residues_numb = 10

analysis_actors_dict = create_analysis_actor_dict('path_to_data_directory/')

# IMPORTANT: For any RMSF analysis always initialize rmsf_cache as an empty dict and pass it as an argument to the
# rmsf methods
rmsf_cache = {}

windows = [[1, 2500], [1, 1250], [1251, 2500], [1, 500], [501, 1000], [1001, 1500], [1501, 2000], [2001, 2500]]

important_residues = {}
for start, stop in windows:
    res = bootstrapped_residue_analysis(analysis_actors_dict, start, stop, stats.ks_2samp, threshold=0.05,
                                        samples_numb=outer_samples_numb,
                                        sample_size=sample_size, rmsf_cache=rmsf_cache)
    try:
        # The lines below aggregate the results in order to end up with a sorted list of the most important
        # residues
        flat_res = [residue for iteration_residues in res for residue in iteration_residues]
        res_freqs, __ = find_top(flat_res, top_residues_numb)
        important_residues[f'{start}-{stop}'] = [res_freq[0] for res_freq in res_freqs]
    except IndexError:
        print(f'Not enough significant residues found - Window {start}-{stop}')
        continue

# Pandas transforms the dictionary to an interpretable tabular form
residues_df = pd.DataFrame(important_residues)
print(residues_df)

