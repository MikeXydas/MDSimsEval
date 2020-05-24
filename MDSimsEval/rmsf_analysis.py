import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import imgkit
from numpy.core._multiarray_umath import ndarray

from scipy import stats

from MDAnalysis.analysis.rms import RMSF


def reset_rmsf_calculations(analysis_actors_dict, start, stop):
    """
    Resets the RMSF calculations of all the ligands in the ``analysis_actors_dict`` and recalculates them on the
    given window.

    This function could be part of the ``AnalysisActor`` class but in order to keep the class simple I have avoided it.

    Args:
        analysis_actors_dict: ``{ "Agonists": List[AnalysisActor.class], "Antagonists": List[AnalysisActor.class] }``
        start(int): The starting frame of the calculations
        stop(int): The stopping frame of the calculations

    """
    # Null the calculations of the RMSF for each ligand
    for ligand in analysis_actors_dict['Agonists'] + analysis_actors_dict['Antagonists']:
        ligand.rmsf_res = None

    # Recalculate on the given window
    for ligand in analysis_actors_dict['Agonists'] + analysis_actors_dict['Antagonists']:
        ligand.rmsf_res = RMSF(ligand.uni.select_atoms('protein')).run(start=start, stop=stop)


def get_avg_rmsf_per_residue(ligand):
    """
    Having the series of resnumbs eg [1, 1, 1, 2, 2, ..., 290, 290] and their respective
    RMSF create buckets (each bucket represents a residue) and calculate the average
    RMSF of each residue

    Args:
        ligand (AnalysisActor.class): The AnalysisActor object on which we have calculated the RMSF

    Returns:
        ndarray[#unique_resnumbs]: The average RMSF of each residue

    """
    bucket = 0
    selected_atoms = ligand.uni.select_atoms('protein').resnums

    total_rmsf = np.zeros(len(np.unique(selected_atoms)))  # Holds the sum of RMSFs of each residue
    total_atoms = np.zeros(len(np.unique(selected_atoms)))  # Holds the number of atoms of each residue

    first_time = True

    for i in range(len(selected_atoms)):
        if not first_time and selected_atoms[i] != selected_atoms[i - 1]:
            bucket += 1  # Changed residue -> go to next bucket
        elif first_time:
            first_time = False

        total_rmsf[bucket] += ligand.rmsf_res.rmsf[i]
        total_atoms[bucket] += 1

    avg_rmsf_per_residue = total_rmsf / total_atoms  # Calculate the mean

    return avg_rmsf_per_residue


def return_top_k(input_arr, analysis_actors_dict, k=10):
    """
    Returns a DataFrame of the top 10 values of the input array

    Args:
        input_arr (ndarray): A vector of the values we want to extract the top-k
        analysis_actors_dict: ``{ "Agonists": List[AnalysisActor.class], "Antagonists": List[AnalysisActor.class] }``
        k (int): The top-k residues that will be returned

    Returns:
        A ``pd.DataFrame['ResidueId', 'RMSF', 'Res Name']`` of the top-k residues

    """
    ind = np.argpartition(input_arr, -k)[-k:]
    ind = ind[np.argsort(input_arr[ind])]
    arr = np.flip(ind)  # Flip the top10 indexes since we want descending order
    arr = np.stack(
        (arr, np.around(np.array(input_arr[arr]), decimals=5)))  # Stack the RMSF values with the Residues Ids

    # Get the residues names of the ids (they are 1-based indexed on the atom selection)
    res_names = [analysis_actors_dict['Agonists'][0].uni.select_atoms(f'resid {int(res_id + 1)}')[0].resname
                 for res_id in arr[0]]

    arr = np.vstack((arr, res_names))

    ret_df = pd.DataFrame(arr.T, columns=['ResidueId', 'RMSF', 'Res Name'])
    ret_df.ResidueId = pd.to_numeric(ret_df.ResidueId)
    ret_df.RMSF = pd.to_numeric(ret_df.RMSF)

    return ret_df


def create_bar_plots_avg_stde(analysis_actors_dict, dir_path, top=50, start=0, stop=2500):
    """
    The method will create two plots (one for the agonists and one for the antagonists). The plot will have the
    avg rmsf per residue and the standard error of the mean. The color of the bars of the top-k
    ``|agon_rmsf_avg - antagon_rmsf_avg|`` RMSF residues are plotted in a different color
    so as to be easily distinguished.

    .. figure:: ../_static/rmsf_barplots.png
        :width: 550
        :align: center
        :height: 500px
        :alt: rmsf corr figure missing

        Barplots of average RMSF per residue, click for higher resolution.

    Args:
        analysis_actors_dict: ``{ "Agonists": List[AnalysisActor.class], "Antagonists": List[AnalysisActor.class] }``
        dir_path (str): The path of the directory the plot will be saved (must end with a ``/``)
        top(int): The top-k residues that will be plotted with a different color
        start(int): The starting frame of the calculations
        stop(int): The stopping frame of the calculations

    """

    reset_rmsf_calculations(analysis_actors_dict, start, stop)

    # Calculate avg RMSF per residue
    residue_rmsfs_agon = np.array([get_avg_rmsf_per_residue(ligand) for ligand in analysis_actors_dict['Agonists']])
    residue_rmsfs_antagon = np.array(
        [get_avg_rmsf_per_residue(ligand) for ligand in analysis_actors_dict['Antagonists']])

    # Average per residue
    residue_rmsfs_agon_avg = np.mean(residue_rmsfs_agon, axis=0)
    residue_rmsfs_antagon_avg = np.mean(residue_rmsfs_antagon, axis=0)

    # Standard error of the mean per residue
    residue_rmsfs_agon_sem = stats.sem(residue_rmsfs_agon)
    residue_rmsfs_antagon_sem = stats.sem(residue_rmsfs_antagon)

    # Get the indexes of the top-k absolute difference of agonists - antagonists RMSF
    top_k_indexes = return_top_k(np.abs(residue_rmsfs_agon_avg - residue_rmsfs_antagon_avg), analysis_actors_dict,
                                 k=top)
    max_diff = max(top_k_indexes.RMSF)
    min_diff = min(top_k_indexes.RMSF)

    # Get the top-k indexes plotted with different colors
    # Mask is a list of booleans where True means the residue is included in the top-k
    mask_top_k = np.array([(ind in list(top_k_indexes.ResidueId)) for ind in range(len(residue_rmsfs_agon_avg))])

    fig = plt.figure(figsize=(42, 20))
    plt.suptitle(f"Range of Top-{top} abs(difference): {min_diff} <= diff <= {max_diff}",
                 fontsize=35, y=0.95)

    # Plotting the agonists
    ax = fig.add_subplot(121)

    plt.bar(np.arange(len(residue_rmsfs_agon_avg))[~mask_top_k], residue_rmsfs_agon_avg[~mask_top_k],
            label="Agonists Avg RMSF per Residue")
    plt.bar(np.arange(len(residue_rmsfs_agon_avg))[~mask_top_k], residue_rmsfs_agon_sem[~mask_top_k],
            label="Agonists Stde RMSF per Residue", color="green")
    plt.bar(np.arange(len(residue_rmsfs_agon_avg))[mask_top_k], residue_rmsfs_agon_avg[mask_top_k],
            label=f"Top-{top} Agonists Avg RMSF per Residue", color="red")
    plt.bar(np.arange(len(residue_rmsfs_agon_avg))[mask_top_k], residue_rmsfs_agon_sem[mask_top_k],
            label=f"Top-{top} Agonists Stde RMSF per Residue", color="orange")

    plt.xlabel("Residue Id", fontsize=30)
    plt.xticks(np.arange(0, len(residue_rmsfs_agon_avg), 50), fontsize=25)
    plt.yticks(np.arange(0, 6.5, 0.5), fontsize=25)
    ax.set_ylim(0, 6)
    ax.set_xlim(0, 290)
    ax.yaxis.grid(linewidth=2)
    plt.ylabel("RMSF", fontsize=36)
    plt.title(f"Agonists Average, Stde RMSF | Frames {start} - {stop}", fontsize=32)
    plt.legend(prop={'size': 20}, markerscale=3, loc=1)

    # Plotting the antagonists
    ax = fig.add_subplot(122)

    plt.bar(np.arange(len(residue_rmsfs_antagon_avg))[~mask_top_k], residue_rmsfs_antagon_avg[~mask_top_k],
            label="Antagonists Avg RMSF per Residue")
    plt.bar(np.arange(len(residue_rmsfs_antagon_avg))[~mask_top_k], residue_rmsfs_antagon_sem[~mask_top_k],
            label="Antagonists Stde RMSF per Residue", color="green")
    plt.bar(np.arange(len(residue_rmsfs_antagon_avg))[mask_top_k], residue_rmsfs_antagon_avg[mask_top_k],
            label=f"Top-{top} Antagonists Avg RMSF per Residue", color="red")
    plt.bar(np.arange(len(residue_rmsfs_antagon_avg))[mask_top_k], residue_rmsfs_antagon_sem[mask_top_k],
            label=f"Top-{top} Antagonists Stde RMSF per Residue", color="orange")

    plt.xlabel("Residue Id", fontsize=30)
    plt.xticks(np.arange(0, len(residue_rmsfs_agon_avg), 50), fontsize=25)
    plt.yticks(np.arange(0, 6.5, 0.5), fontsize=25)
    ax.set_ylim(0, 6)
    ax.set_xlim(0, 290)
    ax.yaxis.grid()
    plt.ylabel("RMSF", fontsize=36)
    plt.title(f"Antagonists Average, Stde RMSF | Frames {start} - {stop}", fontsize=32)
    plt.legend(prop={'size': 20}, markerscale=3, loc=1)

    plt.savefig(f'{dir_path}rmsf_avg_stde_top_k_{start}_{stop}.png', format='png')

    return None


def corr_matrix(analysis_actors_dict, dir_path, method='pearson', top=290, start=0, stop=2500):
    """
    Creates a correlation matrix of the RMSF which has ``#agonists + #antagonists x #agonists + #antagonists`` dimensions.
    The correlation values are calculated on the RMSF values of the ``top-k`` residues. On the output file the ligand
    names have only their first 5 characters for visual reasons.

    Warning:
        We must NOT have ligand names that have the same first 5 characters.

    |

    .. figure:: ../_static/rmsf_corr.png
        :width: 600px
        :align: center
        :height: 250px
        :alt: rmsf corr figure missing

        Correlation heatmap, click for higher resolution.

    .. note::

         In order to save the as ``.png`` you must install ``wkhtmltopdf`` via ``sudo apt-get install wkhtmltopdf`` on
         your machine. Else the output will be in ``.html`` and can be viewed using any browser.

    Args:

        analysis_actors_dict: ``{ "Agonists": List[AnalysisActor.class], "Antagonists": List[AnalysisActor.class] }``
        dir_path (str): The path of the directory the plot will be saved (must end with a ``/``)
        method (str): pearson, kendall, spearman
        top(int): The top-k residues that will be used for the correlation calculations
        start(int): The starting frame of the calculations
        stop(int): The stopping frame of the calculations

    """
    reset_rmsf_calculations(analysis_actors_dict, start, stop)

    # Calculate RMSF per residue
    residue_rmsfs_agon = np.array(
        [get_avg_rmsf_per_residue(ligand) for ligand in analysis_actors_dict['Agonists']])
    residue_rmsfs_antagon = np.array(
        [get_avg_rmsf_per_residue(ligand) for ligand in analysis_actors_dict['Antagonists']])

    if top == 290:
        # We want to calculate the correlation on all the residues of each ligand
        mask = [True for i in range(len(residue_rmsfs_agon[0]))]
    else:
        # We want to calculate the correlation on the top-k differentiating residues of each ligand
        residue_rmsfs_agon_avg = np.mean(residue_rmsfs_agon, axis=0)
        residue_rmsfs_antagon_avg = np.mean(residue_rmsfs_antagon, axis=0)

        # Get the indexes of the top-k absolute difference of agonists - antagonists RMSF
        top_k_indexes = return_top_k(np.abs(residue_rmsfs_agon_avg - residue_rmsfs_antagon_avg), analysis_actors_dict,
                                     k=top)
        mask = np.array([(ind in list(top_k_indexes.ResidueId)) for ind in range(len(residue_rmsfs_agon_avg))])

    # Creating dataframe which will have as columns the ligand names and as rows the residues
    rmsf_array = np.array([res_rmsf[mask] for res_rmsf in np.vstack((residue_rmsfs_agon, residue_rmsfs_antagon))])

    # Use only the first 5 chars of the ligand name for better visual result
    ligand_names = [ligand.drug_name[:5]
                    for ligand in analysis_actors_dict['Agonists'] + analysis_actors_dict['Antagonists']]
    rmsf_df = pd.DataFrame(rmsf_array.T, columns=ligand_names)

    # Creating the correlation matrix adding a heatmap for easier visualization

    corr = rmsf_df.corr(method=method)
    html_render = corr.style.background_gradient(cmap='coolwarm', axis=None).set_precision(2).render()

    # Saving the correlation matrix
    try:
        # If wkhtmltopdf is installed save the results as a .png
        imgkit.from_string(html_render, f"{dir_path}rmsf_{method}_map_top{top}_{start}_{stop}.png")
    except IOError:
        # Save the html of the correlation map which can be rendered by a browser
        with open(f"{dir_path}rmsf_{method}_map_top{top}_{start}_{stop}.html", "w") as text_file:
            text_file.write(html_render)

    return None


def stat_test_residues(analysis_actors_dict, stat_test=stats.ttest_ind, threshold=0.05, start=-1, stop=-1):
    """
    Finds the most differentiating residues based on a statistical test on their RMSF. If ``start==-1`` and ``stop==-1``
    then we do not recalculate RMSF on the given window.

    | For example on the T-test we have:
    | Null Hypothesis: The RMSFs of a specific residue have identical average (expected) values

    Warning:
        I suggest firstly using the :ref:`bootstrapped_rmsf_analysis` which provides a more time consuming but also more
        generalized way of finding the most significant residues.

    Args:
        analysis_actors_dict: : ``{ "Agonists": List[AnalysisActor.class], "Antagonists": List[AnalysisActor.class] }``
        stat_test (scipy.stats): A statistical test method with the interface of scipy.stats methods
        threshold (float): The p-value threshold of the accepted and returned residues
        start(int): The starting frame of the calculations
        stop(int): The stopping frame of the calculations

    Returns:
        A list of tuples of the residue ids (0-indexed) that had below the threshold p-value, the p_value.
        Eg ``[(10, 0.03), (5, 0.042), ...]``

    """
    # Recalculate the RMSF calculations on the input window if we need a reset
    if start != -1 and stop != -1:
        reset_rmsf_calculations(analysis_actors_dict, start=start, stop=stop)

    stacked_agons_rmsf: ndarray = np.array(
        [get_avg_rmsf_per_residue(ligand) for ligand in analysis_actors_dict['Agonists']])
    stacked_antagons_rmsf = np.array(
        [get_avg_rmsf_per_residue(ligand) for ligand in analysis_actors_dict['Antagonists']])

    # Get the p_value of each residue
    p_values = []
    for agon_res_rmsf, antagon_res_rmsf in zip(stacked_agons_rmsf.T, stacked_antagons_rmsf.T):
        p_values.append(stat_test(agon_res_rmsf, antagon_res_rmsf)[1])

    # Select the p_values that pass the threshold
    enumed_pvalues = np.array(list(enumerate(p_values)))
    enumed_pvalues = enumed_pvalues[enumed_pvalues[:, 1] <= threshold]

    # Transform the ndarray to a list of tuples
    enumed_pvalues = [(int(pair[0]), pair[1]) for pair in enumed_pvalues]

    # Return in ascending order of p_values
    return sorted(enumed_pvalues, key=lambda x: x[1])