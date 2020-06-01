from MDSimsEval.rmsf_analysis import reset_rmsf_calculations
from MDSimsEval.rmsf_analysis import get_avg_rmsf_per_residue

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import random
from tqdm import tqdm
import pandas as pd


def initialize_pools(ligands_list, total_ligands=20, set_ligands=12):
    """
    Inputs a list of ``MDSimsEval.AnalysisActor`` objects and returns initial input and replacement
    sets of the given size.

    Args:
        ligands_List: List of ``MDSimsEval.AnalysisActor`` objects
        total_ligands(int): How many ligands will be used in the experiment (input + replacement sets sizes)
        set_ligands(int): How many ligands will be on the initial input set. The rest of them will be on the
        replacement set

    Returns:
        | Tuple of (List of ``MDSimsEval.AnalysisActor`` that will be our initial input set,
        |           List of ``MDSimsEval.AnalysisActor`` that will be our initial replacement set)

    """
    ligands_chosen = random.sample(ligands_list, total_ligands)

    chosen_random_ligands = random.sample(list(np.arange(len(ligands_chosen))), set_ligands)
    mask_set_ligands = np.array([which_lig in chosen_random_ligands
                                 for which_lig in np.arange(total_ligands)])

    return list(np.array(ligands_chosen)[mask_set_ligands]), list(np.array(ligands_chosen)[~mask_set_ligands])


def replacement_swap(input_set, replacement_set, numb_of_replacements=1):
    """
    Performs given number of swaps between the input_set and the replacement_set.The swaps are inplace.

    Args:
        input_set: List of ``MDSimsEval.AnalysisActor`` that is our input set
        replacement_set: List of ``MDSimsEval.AnalysisActor`` that is our replacement set

    """
    to_be_replaced_indexes = random.sample(list(np.arange(len(input_set))), numb_of_replacements)
    to_be_inputted_indexes = random.sample(list(np.arange(len(replacement_set))), numb_of_replacements)

    for replaced, inputted in zip(to_be_replaced_indexes, to_be_inputted_indexes):
        input_set[replaced], replacement_set[inputted] = replacement_set[inputted], input_set[replaced]

    return None


def minimal_stat_test(agonists, antagonists, stat_test, start, stop, threshold=0.05, cache=None):
    """
    Inputs a list of agonists and a list of antagonists and finds the most significant residues. We do not return
    the p_value but only the residue ids.

    .. note::

         RMSF calculations are cached to avoid recalculating them. In order to use the caching mechanism we give as an
         argument an empty dictionary ``{}``.

    Args:
        agonists: List of ``MDSimsEval.AnalysisActor`` agonists
        antagonists: List of ``MDSimsEval.AnalysisActor`` antagonists
        stat_test (scipy.stats): A statistical test method with the interface of scipy.stats methods
        start(int): The starting frame of the calculations
        stop(int): The stopping frame of the calculations
        threshold (float): The p-value threshold of the accepted and returned residues
        cache: Dictionary with key ``ligand_name_start_stop`` and value the RMSF run result. If set to ``None`` no cache
               will be kept


    """
    reset_rmsf_calculations({'Agonists': agonists, 'Antagonists': antagonists}, start=start, stop=stop, cache=cache)

    stacked_agonists_rmsf = np.array([get_avg_rmsf_per_residue(ligand) for ligand in agonists])
    stacked_antagonists_rmsf = np.array([get_avg_rmsf_per_residue(ligand) for ligand in antagonists])

    # Get the p_value of each residue
    p_values = []
    for agon_res_rmsf, antagon_res_rmsf in zip(stacked_agonists_rmsf.T, stacked_antagonists_rmsf.T):
        p_values.append(stat_test(agon_res_rmsf, antagon_res_rmsf)[1])

    # Select the p_values that pass the threshold
    enumed_pvalues = np.array(list(enumerate(p_values)))
    enumed_pvalues = enumed_pvalues[enumed_pvalues[:, 1] <= threshold]

    return set(enumed_pvalues[:, 0])


def sensitivity_calc(sign_residues_per_iter):
    """
    | Inputs the output of ``bootstrapped_residue_analysis`` and calculates the sensitivity of each residue.
    | The returned sensitivity of each residue is calculated by calculating ``residue_appearances / iterations``.
    | A sensitivity of 1 is ideal meaning that the residue was significant to all the iterations.
    |

    Args:
        sign_residues_per_iter: A list of sets containing the residue ids of the significant residues on each iteration

    Returns:
        A dictionary of ``ResidueId(key), Sensitivity(value)`` for all the residues that appeared at least on one
        iteration

    """
    sens_dict = {}
    for which_iter in sign_residues_per_iter:
        for which_res in which_iter:
            try:
                sens_dict[which_res] += 1
            except KeyError:
                sens_dict[which_res] = 1

    # Get the sensitivity by calculating residue_appearances / total_iterations
    sens_dict = {k: v / len(sign_residues_per_iter) for k, v in sens_dict.items()}

    return sens_dict


# def bootstrapped_residue_analysis(analysis_actors_dict, windows, stat_test=stats.ks_2samp, threshold=0.05,
#                                   input_size=12, replacement_size=8, replacements=1, iterations=5):
#     """
#     | This is the main method of finding the most significant RMSF residues in a general enough way in order to avoid
#       overfitting.
#     | To do that we follow the below bootstrapping method:
#
#     1. We pick ``input_size + replacement_size`` agonists and ``input_size + replacement_size`` antagonists
#     2. We randomly pick ``input_size`` agonists and ``input_size`` antagonists that will be our input set.
#     3. We repeat for a given number of iterations:
#         1. Find the union of the most significant residues on specific windows
#         2. Save the most significant residues
#         3. Replace ``replacement`` agonists and ``replacement`` antagonists with random ligands
#            from their respective replacement pools
#     4. Return which residues where significant on each iteration
#
#     Args:
#         analysis_actors_dict: ``{ "Agonists": List[AnalysisActor.class], "Antagonists": List[AnalysisActor.class] }``
#         windows: List of (start, stop) tuples of each window
#         stat_test (scipy.stats): A statistical test method with the interface of scipy.stats methods
#         threshold (float): The p-value threshold of the accepted and returned residues
#         input_size (int): The size of set for each class on which we will find the significant residues
#         replacement_size (int): The size of the replacement set for each class
#         replacements(int): How many replacements will occur on each iteration
#         iterations(int): How many times we will repeat the finding of the significant residues
#
#     Returns:
#         A list of sets containing which residues were statistically significant on each iteration,
#         ``[{12, 17, 53}, {17, 62}, ..., {53, 17}]``
#
#     """
#     # Create our initial input and replacement set for both agonist and antagonists
#     inp_set_agon, rep_set_agon = initialize_pools(analysis_actors_dict['Agonists'],
#                                                   total_ligands=input_size + replacement_size, set_ligands=input_size)
#     inp_set_antagon, rep_set_antagon = initialize_pools(analysis_actors_dict['Antagonists'],
#                                                         total_ligands=input_size + replacement_size,
#                                                         set_ligands=input_size)
#
#     significant_residues_per_iter = []
#     rmsf_cache = {}    # Memoization of the RMSF calculations
#
#     for i in tqdm(range(iterations), desc='Iterations'):
#         iteration_residues = []
#         for start, stop in windows:
#             sign_residues = minimal_stat_test(inp_set_agon, inp_set_antagon, stat_test, start, stop, threshold, rmsf_cache)
#             iteration_residues.append(sign_residues)
#
#         # Extract the union of the significant residues from all the windows
#         significant_residues_per_iter.append(set().union(*iteration_residues))
#
#         # Perform the replacement of our bootstrap method
#         replacement_swap(inp_set_agon, rep_set_agon, replacements)
#         replacement_swap(inp_set_antagon, rep_set_antagon, replacements)
#
#     # Calculate the sensitivity of each significant residue
#     return significant_residues_per_iter


def bootstrapped_residue_analysis(analysis_actors_dict, start, stop, stat_test, threshold, samples_numb, sample_size):
    """
    Generate ``samples_numb`` random samples  that each one will have ``sample_size`` agonists and ``sample_size``
    antagonists. Then perform a statistical test on the RMSFs of each residue of the agonists vs the RMSFs of each
    residue of the antagonists. If the difference is significant save the residue as important for the specific
    iteration.

    Finally, return the significant residues on each iteration

    Args:
        analysis_actors_dict: ``{ "Agonists": List[AnalysisActor.class], "Antagonists": List[AnalysisActor.class] }``
        start(int): The starting frame of the calculations
        stop(int): The stopping frame of the calculations
        stat_test (scipy.stats): A statistical test method with the interface of scipy.stats methods
        threshold (float): The p-value threshold of the accepted and returned residues
        samples_numb (int): How many random samples will be generated
        sample_size (int): How many ligands each sample will have of each class. Eg if ``sample_size=10`` then each
                           sample will have 10 agonists and 10 antagonists

    Returns:
         A list of ``samples_numb`` sets. Each sets contains the ``ResidueId`` of the residues that were significant on
         the specific iteration

    """
    # Create the samples
    samples = []
    for i in range(samples_numb):
        samples.append(
            {'Agonists': random.sample(analysis_actors_dict['Agonists'], sample_size),
             'Antagonists': random.sample(analysis_actors_dict['Antagonists'], sample_size)}
        )

    rmsf_cache = {}
    residues_per_sample = []

    for sample in tqdm(samples, desc='Sample'):
        sign_residues = minimal_stat_test(sample['Agonists'], sample['Antagonists'], stat_test, start, stop, threshold,
                                          rmsf_cache)

        residues_per_sample.append(set(sign_residues))

    return residues_per_sample


def create_correlation_df(analysis_actors_dict, residue_ids, method, start, stop, rmsf_cache=None):
    """
    Creates a ``numb_of_ligands x numb_of_ligands`` dataframe which has the pair correlations calculated
    on the rmsf of the given ``residue_ids``.

    The result is not in a readable format and could be passed in ``MDSimsEval.utils.render_corr_df``.

    Args:
        analysis_actors_dict: ``{ "Agonists": List[AnalysisActor.class], "Antagonists": List[AnalysisActor.class] }``
        residue_ids: List of residue ids that we want the correlation on
        Eg the top-k, high-k, most statistically significant.
        method (str): pearson, kendall, spearman
        start(int): The starting frame of the calculations
        stop(int): The stopping frame of the calculations
        rmsf_cache: Dictionary with key ``ligand_name_start_stop`` and value the RMSF run result. If set to ``None`` no
                    cache will be kept

    Returns:
        A ``pd.DataFrame`` which has the pair correlations of all the ligands

    """
    reset_rmsf_calculations(analysis_actors_dict, start, stop, rmsf_cache)

    # Calculating the RMSFs of each residue instead of each atom
    residue_rmsfs_agon = np.array([get_avg_rmsf_per_residue(ligand) for ligand in analysis_actors_dict['Agonists']])
    residue_rmsfs_antagon = np.array(
        [get_avg_rmsf_per_residue(ligand) for ligand in analysis_actors_dict['Antagonists']])

    # We need the number of total residues to create the mask below
    residues_numb = len(residue_rmsfs_agon[0])

    # Create a True, False mask of the given residues
    top_mask = [res_id in residue_ids for res_id in np.arange(residues_numb)]

    # Creating a DataFrame which will have as columns the ligand names and as rows the residues
    rmsf_array = np.array([res_rmsf[top_mask] for res_rmsf in np.vstack((residue_rmsfs_agon, residue_rmsfs_antagon))])

    # Use only the first 5 chars of the ligand name for better visual results
    ligand_names = [ligand.drug_name[:5]
                    for ligand in analysis_actors_dict['Agonists'] + analysis_actors_dict['Antagonists']]
    rmsf_df = pd.DataFrame(rmsf_array.T, columns=ligand_names)

    # Create the correlation dataframe
    corr = rmsf_df.corr(method=method)

    return corr


def find_top(flat_res, top_k):
    # Find on how many iterations or samples each residue appears as significant
    res_frequencies = {}
    for residue in flat_res:
        if residue in res_frequencies:
            res_frequencies[residue] += 1
        else:
            res_frequencies[residue] = 1

    res_frequencies = [[res_id, freq] for res_id, freq in res_frequencies.items()]

    # Keep only the top residues that have the biggest frequencies
    res_frequencies = sorted(res_frequencies, key=lambda x: x[1], reverse=True)[:top_k]

    top_residues = [[int(res_freq[0]), res_freq[1]] for res_freq in res_frequencies]

    threshold = res_frequencies[-1][1]

    return top_residues, threshold


def create_sign_mask_array(bootstrapped_results, top_k, residue_numb=290):
    masks = np.zeros((len(bootstrapped_results), residue_numb))
    threshes = []

    for index, significant_residues_per_iteration in enumerate(bootstrapped_results):
        flat_res = [residue
                    for iteration_residues in significant_residues_per_iteration
                    for residue in iteration_residues]

        top_res_freq, thresh = find_top(flat_res, top_k)

        for res, freq in top_res_freq:
            masks[index][res] = freq

        threshes.append(thresh)

    return masks, threshes


def plot_hists_summary(bootstrapped_results, residue_numb, dir_path, top_k=50):
    """
    Plots a histogram which summarizes which residues were found significant and on how many samples on
    each window. The colors go from black (1st window) to yellow (5th window). In case of a residue being important
    on more than 1 window, the bars are stacked in chronological order (from the earlier windows to the later ones).
    The height of the bars shows the number of samples the residue was statistically important.

    On the legend the word **thresh** specifies the number of samples a residue must be found significant in, in order
    to be included in the ``top-k`` most significant residues.

    .. warning ::
        Currently the function expects five windows of 500 frames each (0 - 500, 500 - 1000, ..., 2000 - 2500).

    Example:
        ::

            from MDSimsEval.utils import create_analysis_actor_dict
            from MDSimsEval.rmsf_bootstrapped_analysis import bootstrapped_residue_analysis
            from MDSimsEval.rmsf_bootstrapped_analysis import plot_hists_summary

            from scipy import stats
            import numpy as np
            import random

            # Read the data
            analysis_actors_dict = create_analysis_actor_dict('path_to_dataset_roor_folder')

            # Do not use all the ligands so as to have a validation set
            agonists_train = random.sample(analysis_actors_dict['Agonists'], 20)
            antagonists_train = random.sample(analysis_actors_dict['Antagonists'], 20)

            bootstrapped_results = []
            for start in np.arange(0, 2500, 500):
                res = bootstrapped_residue_analysis({"Agonists": agonists_train, "Antagonists": antagonists_train},
                                                    start, start + 500, stats.ks_2samp, threshold=0.05, samples_numb=1000,
                                                    sample_size=10)
                bootstrapped_results.append(res)

            # Here it is suggested to save the bootstrapped_results on disk using pickle so as to avoid
            # recalculating them

            plot_hists_summary(bootstrapped_results, residue_numb=290, dir_path='path_to_save_dir/', top_k=50)

    .. figure:: ../_static/multi_color_hists_.png
        :width: 700
        :align: center
        :height: 200px
        :alt: rmsf multi colored hists missing

        Output plot of the above script, click for higher resolution

    Args:
        bootstrapped_results: A list of `bootstrapped_residue_analysis` results for each window
        residue_numb (int): The total number of residues in the RMSF selection
        dir_path (str): The path of the directory the plot will be saved (must end with a ``/``)
        top_k (int): How many residues to include in order of significance

    """
    masks, threshes = create_sign_mask_array(bootstrapped_results, top_k, residue_numb)
    masks_offsets = np.sum(masks, axis=0)

    fig = plt.figure(figsize=(40, 7))
    ax = fig.add_subplot(111)

    plt.xlabel("Residue Id", fontsize=28)
    plt.ylabel("Number of Appearances", fontsize=28)
    plt.title(f"Multicolored Histogram of Significant Residues", fontsize=28)
    plt.xlim(0, 290)
    plt.xticks(np.arange(0, residue_numb, 15), fontsize=28)
    plt.ylim(0, 400)
    plt.yticks(np.arange(0, 401, 50), fontsize=28)

    # A list that we save the colors of the important residues on their respective windows
    window_colors_labels = [['black', f'Important on 1 - 500, Thresh: {threshes[0]}'],
                            ['darkslateblue', f'Important on 501 - 1000, Thresh: {threshes[1]}'],
                            ['green', f'Important on 1001 - 1500, Thresh: {threshes[2]}'],
                            ['red', f'Important on 1501 - 2000, Thresh: {threshes[3]}'],
                            ['yellow', f'Important on 2001 - 2500, Thresh: {threshes[4]}']]

    for res_id in np.arange(residue_numb):
        if masks_offsets[res_id] == 0:
            continue

        # Increase the current offset so as if an importance bar is already plotted, to plot on top of it
        current_offset = 0

        for iterations_on_window, color_label in zip(masks[:, res_id], window_colors_labels):
            if iterations_on_window > 0:
                plt.bar(res_id, iterations_on_window, bottom=current_offset, color=color_label[0])

                current_offset += iterations_on_window

    for color, label in window_colors_labels:
        plt.bar(-1, 0, color=color, label=label)

    plt.legend(prop={'size': 18}, markerscale=2, ncol=2, loc=(0.45, 0.75))
    ax.yaxis.grid()

    plt.savefig(f'{dir_path}multi_color_hists_.png', format='png')
