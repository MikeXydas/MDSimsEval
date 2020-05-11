import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy import stats

from MDAnalysis.analysis.rms import RMSF


def get_avg_rmsf_per_residue(drug):
    """
    Having the series of resnumbs eg [1, 1, 1, 2, 2, ..., 291, 291] and their respective
    RMSF create buckets (each bucket represents a residue) and calculate the average
    RMSF of each residue

    Args:
        drug (AnalysisActor.class): The AnalysisActor object on which we have calculated the RMSF

    Returns:
        np.array[#unique_resnumbs]: The average RMSF of each residue
    """
    bucket = 0
    selected_atoms = drug.uni.select_atoms('protein').resnums

    total_rmsf = np.zeros(len(np.unique(selected_atoms)))  # Holds the sum of RMSFs of each residue
    total_atoms = np.zeros(len(np.unique(selected_atoms)))  # Holds the number of atoms of each residue

    first_time = True

    for i in range(len(selected_atoms)):
        if not first_time and selected_atoms[i] != selected_atoms[i - 1]:
            bucket += 1  # Changed residue -> go to next bucket
        elif first_time:
            first_time = False

        total_rmsf[bucket] += drug.rmsf_res.rmsf[i]
        total_atoms[bucket] += 1

    avg_rmsf_per_residue = total_rmsf / total_atoms  # Calculate the mean

    return avg_rmsf_per_residue


def return_top_k(input_arr, analysis_actors_dict, k=10):
    """
    Returns a DataFrame of the top 10 values of the input array

    Args:
        input_arr (ndarray): A vector of the values we want to extract the top-k
        analysis_actors_dict: Dict(
                                "Agonists": List[AnalysisActor.class]
                                "Antagonists": List[AnalysisActor.class]
                              )
        k (int): The top-k residues that will be returned

    Returns:
        pd.DataFrame[Residue Id, RMSF]: A pandas dataframe, on the 1st column are the indexes of the top-k values
                                        and on the 2nd column the value
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

    ret_df = pd.DataFrame(arr.T, columns=['ResidueId', 'RMSF', "Res Name"])
    ret_df.ResidueId = pd.to_numeric(ret_df.ResidueId).astype(np.int64)
    return ret_df


def create_bar_plots_avg_stde(analysis_actors_dict, dir_path, top=50, start=0, stop=2500):
    """
    The method will create two plots (one for the agonists and one for the antagonists). The plot will have the
    avg rmsf per residue and the standard error of the mean. The color of the bars of the top-k
    |agon_rmsf_avg - antagon_rmsf_avg| RMSF residues are plotted in a different color easily distinguished.

    Args:
        analysis_actors_dict: Dict(
                                "Agonists": List[AnalysisActor.class]
                                "Antagonists": List[AnalysisActor.class]
                              )
        dir_path (str): The path of the directory the plot will be saved
        top(int): The top-k residues that will be plotted with a different color
        start(int): The starting frame of the calculations
        stop(int): The stopping frame of the calculations
    """

    # Reset the calculations of the RMSF for each ligand
    for ligand in analysis_actors_dict['Agonists'] + analysis_actors_dict['Antagonists']:
        ligand.rmsf_res = None

    # Recalculate on the given window
    for ligand in analysis_actors_dict['Agonists'] + analysis_actors_dict['Antagonists']:
        ligand.rmsf_res = RMSF(ligand.uni.select_atoms('protein')).run(start=start, stop=stop)

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
    top_k_indexes = return_top_k(np.abs(residue_rmsfs_agon_avg - residue_rmsfs_antagon_avg), analysis_actors_dict, k=top)

    # Get the top-k indexes plotted with different colors
    # Mask is a list of booleans where True means the residue is included in the top-k
    mask_top_k = np.array([(ind in list(top_k_indexes.ResidueId)) for ind in range(len(residue_rmsfs_agon_avg))])

    fig = plt.figure(figsize=(18, 40))

    # Plotting the agonists
    ax = fig.add_subplot(211)

    plt.bar(np.arange(len(residue_rmsfs_agon_avg))[~mask_top_k], residue_rmsfs_agon_avg[~mask_top_k],
            label="Agonists Avg RMSF per Residue")
    plt.bar(np.arange(len(residue_rmsfs_agon_avg))[~mask_top_k], residue_rmsfs_agon_sem[~mask_top_k],
            label="Agonists Stde RMSF per Residue")
    plt.bar(np.arange(len(residue_rmsfs_agon_avg))[mask_top_k], residue_rmsfs_agon_avg[mask_top_k],
            label=f"Top-{top} Agonists Avg RMSF per Residue")
    plt.bar(np.arange(len(residue_rmsfs_agon_avg))[mask_top_k], residue_rmsfs_agon_sem[mask_top_k],
            label=f"Top-{top} Agonists Stde RMSF per Residue")

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
    ax = fig.add_subplot(212)

    plt.bar(np.arange(len(residue_rmsfs_antagon_avg))[~mask_top_k], residue_rmsfs_antagon_avg[~mask_top_k],
            label="Antagonists Avg RMSF per Residue")
    plt.bar(np.arange(len(residue_rmsfs_antagon_avg))[~mask_top_k], residue_rmsfs_antagon_sem[~mask_top_k],
            label="Antagonists Stde RMSF per Residue")
    plt.bar(np.arange(len(residue_rmsfs_antagon_avg))[mask_top_k], residue_rmsfs_antagon_avg[mask_top_k],
            label=f"Top-{top} Antagonists Avg RMSF per Residue")
    plt.bar(np.arange(len(residue_rmsfs_antagon_avg))[mask_top_k], residue_rmsfs_antagon_sem[mask_top_k],
            label=f"Top-{top} Antagonists Stde RMSF per Residue")

    plt.xlabel("Residue Id", fontsize=30)
    plt.xticks(np.arange(0, len(residue_rmsfs_agon_avg), 50), fontsize=25)
    plt.yticks(np.arange(0, 6.5, 0.5), fontsize=25)
    ax.set_ylim(0, 6)
    ax.set_xlim(0, 290)
    ax.yaxis.grid(linewidth=2)
    plt.ylabel("RMSF", fontsize=36)
    plt.title(f"Antagonists Average, Stde RMSF | Frames {start} - {stop}", fontsize=32)
    plt.legend(prop={'size': 20}, markerscale=3, loc=1)

    plt.savefig(f'{dir_path}rmsf_avg_stde_top_k_{start}_{stop}.png', format='png')