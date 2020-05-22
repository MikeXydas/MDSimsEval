from MDSimsEval.rmsf_analysis import reset_rmsf_calculations
from MDSimsEval.rmsf_analysis import get_avg_rmsf_per_residue

import numpy as np
from scipy import stats
import random
from tqdm import tqdm


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


def minimal_stat_test(agonists, antagonists, stat_test, start, stop, threshold=0.05):
    """
    Inputs a list of agonists and a list of antagonists and finds the most significant residues. We do not return
    the p_value but only the residue ids.

    .. todo::

         RMSF calcualtions get repeated. I should in the near feature create a caching mechanism.

    Args:
        agonists: List of ``MDSimsEval.AnalysisActor`` agonists
        antagonists: List of ``MDSimsEval.AnalysisActor`` antagonists
        stat_test (scipy.stats): A statistical test method with the interface of scipy.stats methods
        start(int): The starting frame of the calculations
        stop(int): The stopping frame of the calculations
        threshold (float): The p-value threshold of the accepted and returned residues


    """
    reset_rmsf_calculations({'Agonists': agonists, 'Antagonists': antagonists}, start=start, stop=stop)

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


def bootstrapped_residue_analysis(analysis_actors_dict, windows, stat_test=stats.ks_2samp, threshold=0.05,
                                  input_size=12, replacement_size=8, replacements=1, iterations=5):
    """
    | This is the main method of finding the most significant RMSF residues in a general enough way in order to avoid
      overfitting.
    | To do that we follow the below bootstrapping method:

    1. We pick ``input_size + replacement_size`` agonists and ``input_size + replacement_size`` antagonists
    2. We randomly pick ``input_size`` agonists and ``input_size`` antagonists that will be our input set.
    3. We repeat for a given number of iterations:
        1. Find the union of the most significant residues on specific windows
        2. Save the most significant residues
        3. Replace ``replacement`` agonists and ``replacement`` antagonists with random ligands
           from their respective replacement pools
    4. Return the sensitivity of the results

    | The returned sensitivity of each residue is calculated by calculating ``residue_appearances / iterations``.
    |

    Args:
        analysis_actors_dict: ``{ "Agonists": List[AnalysisActor.class], "Antagonists": List[AnalysisActor.class] }``
        windows: List of (start, stop) tuples of each window
        stat_test (scipy.stats): A statistical test method with the interface of scipy.stats methods
        threshold (float): The p-value threshold of the accepted and returned residues
        input_size (int): The size of set for each class on which we will find the significant residues
        replacement_size (int): The size of the replacement set for each class
        replacements(int): How many replacements will occur on each iteration
        iterations(int): How many times we will repeat the finding of the significant residues

    Returns:
        A dictionary of ``ResidueId, Sensitivity`` for all the residues that appeared at least on one iteration

    """
    # Create our initial input and replacement set for both agonist and antagonists
    inp_set_agon, rep_set_agon = initialize_pools(analysis_actors_dict['Agonists'],
                                                  total_ligands=input_size + replacement_size, set_ligands=input_size)
    inp_set_antagon, rep_set_antagon = initialize_pools(analysis_actors_dict['Antagonists'],
                                                        total_ligands=input_size + replacement_size,
                                                        set_ligands=input_size)

    significant_residues_per_iter = []

    for i in tqdm(range(iterations), desc='Iterations'):
        iteration_residues = []
        for start, stop in windows:
            sign_residues = minimal_stat_test(inp_set_agon, inp_set_antagon, stat_test, start, stop, threshold)
            iteration_residues.append(sign_residues)

        # Extract the union of the significant residues from all the windows
        significant_residues_per_iter.append(set().union(*iteration_residues))

        # Perform the replacement of our bootstrap method
        replacement_swap(inp_set_agon, rep_set_agon, replacements)
        replacement_swap(inp_set_antagon, rep_set_antagon, replacements)

    # Calculate the sensitivity of each significant residue
    return sensitivity_calc(significant_residues_per_iter)
