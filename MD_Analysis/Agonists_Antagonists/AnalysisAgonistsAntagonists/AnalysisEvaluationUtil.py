import numpy as np
import pandas as pd

from tqdm import tqdm
import matplotlib.pyplot as plt

def calculate_average_cols_rg(analysis_actors_dict):
    """
    Calculates the mean of Rg of each frame for all the agonists.
    Then does the same for the antagonists.

    Args:
        analysis_actors_dict: Dict(
                                "Agonists": List[DirectoryPath (str), AnalysisActor.class]
                                "Antagonists": List[DirectoryPath (str), AnalysisActor.class]
                              )

    Returns:
        Tuple(np.array[#frames], np.array[#frames])
    """
    stacked_agonists = analysis_actors_dict['Agonists'][0][1].rg_res
    for which_drug in analysis_actors_dict['Agonists'][1:]:
        stacked_agonists = np.vstack((stacked_agonists, which_drug[1].rg_res))
    avg_agonists_cols = np.mean(stacked_agonists, axis=0)

    stacked_antagonists = analysis_actors_dict['Antagonists'][0][1].rg_res
    for which_drug in analysis_actors_dict['Antagonists'][1:]:
        stacked_antagonists = np.vstack((stacked_antagonists, which_drug[1].rg_res))
    avg_antagonists_cols = np.mean(stacked_antagonists, axis=0)

    return avg_agonists_cols, avg_antagonists_cols


def calculate_average_cols_sasa(analysis_actors_dict):
    """
    Calculates the mean of SASA of each frame for all the agonists.
    Then does the same for the antagonists.

    Args:
        analysis_actors_dict: Dict(
                                "Agonists": List[DirectoryPath (str), AnalysisActor.class]
                                "Antagonists": List[DirectoryPath (str), AnalysisActor.class]
                              )

    Returns:
        Tuple(np.array[#frames], np.array[#frames])
    """
    stacked_agonists = analysis_actors_dict['Agonists'][0][1].sasa_res[1]
    for which_drug in analysis_actors_dict['Agonists'][1:]:
        stacked_agonists = np.vstack((stacked_agonists, which_drug[1].sasa_res[1]))
    avg_agonists_cols = np.mean(stacked_agonists, axis=0)

    stacked_antagonists = analysis_actors_dict['Antagonists'][0][1].sasa_res[1]
    for which_drug in analysis_actors_dict['Antagonists'][1:]:
        stacked_antagonists = np.vstack((stacked_antagonists, which_drug[1].sasa_res[1]))
    avg_antagonists_cols = np.mean(stacked_antagonists, axis=0)

    return avg_agonists_cols, avg_antagonists_cols


def get_avg_rmsf_per_residue(drug):
    """
    Having the series of resnumbs eg [1, 1, 1, 2, 2, ..., 291, 291] and their respective
    RMSF crete buckets (each bucket represents a residue) and calculate the average
    RMSF of each residue

    Args:
        drug (AnalysisActor.class): The AnalysisActor object on which we have calculated the RMSF

    Returns:
        np.array[#unique_resnumbs]: The average RMSF of each residue
    """
    bucket = 0
    total_rmsf = np.zeros(len(np.unique(drug.uni.atoms.resnums)))  # Holds the sum of RMSFs of each residue
    total_atoms = np.zeros(len(np.unique(drug.uni.atoms.resnums)))  # Holds the number of atoms of each residue
    first_time = True
    for i in range(len(drug.uni.atoms.resnums)):
        if not first_time and drug.uni.atoms.resnums[i] != drug.uni.atoms.resnums[i - 1]:
            bucket += 1  # Changed residue -> go to next bucket
        elif first_time:
            first_time = False

        total_rmsf[bucket] += drug.rmsf_res.rmsf[i]
        total_atoms[bucket] += 1

    avg_rmsf_per_residue = total_rmsf / total_atoms  # Calculate the mean

    return avg_rmsf_per_residue


def calculate_average_cols_rmsf(analysis_actors_dict):
    """
    Calculates the mean of the mean residue RMSF for the agonists.
    Then does the same for the antagonists.

    Args:
        analysis_actors_dict: Dict(
                                "Agonists": List[DirectoryPath (str), AnalysisActor.class]
                                "Antagonists": List[DirectoryPath (str), AnalysisActor.class]
                              )

    Returns:
        Tuple(np.array[#frames], np.array[#frames])
    """
    # Agonists Iteration
    stacked_agonists = get_avg_rmsf_per_residue(analysis_actors_dict['Agonists'][0][1])
    for which_drug in analysis_actors_dict['Agonists'][1:]:
        stacked_agonists = np.vstack((stacked_agonists, get_avg_rmsf_per_residue(which_drug[1])))
    avg_agonists_cols = np.mean(stacked_agonists, axis=0)

    # Antagonists Iteration
    stacked_antagonists = get_avg_rmsf_per_residue(analysis_actors_dict['Antagonists'][0][1])
    for which_drug in analysis_actors_dict['Antagonists'][1:]:
        stacked_antagonists = np.vstack((stacked_antagonists, get_avg_rmsf_per_residue(which_drug[1])))
    avg_antagonists_cols = np.mean(stacked_antagonists, axis=0)

    return avg_agonists_cols, avg_antagonists_cols


def return_top_k(analysis_actors_dict, input_arr, k=10):
    '''
    Returns a DataFrame of the top 10 values of the input array

    Args:
        analysis_actors_dict: Dict(
                                "Agonists": List[DirectoryPath (str), AnalysisActor.class]
                                "Antagonists": List[DirectoryPath (str), AnalysisActor.class]
                              )
        input_arr (ndarray): A vector of the values we want to extract the top-k
        k (int): The top-k residues that will be displayed


    Returns:
        pd.DataFrame[Residue Id, RMSF]: A pandas dataframe, on the 1st column are the indexes of the top-k values
                                        and on the 2nd column the value
    '''
    ind = np.argpartition(input_arr, -k)[-k:]
    ind = ind[np.argsort(input_arr[ind])]
    arr = np.flip(ind)  # Flip the top10 indexes since we want descending order
    arr = np.stack(
        (arr, np.around(np.array(input_arr[arr]), decimals=5)))  # Stack the RMSF values with the Residues Ids

    # Get the residues names of the ids (they are 1-based indexed on the atom selection)
    res_names = [analysis_actors_dict['Agonists'][0][1].uni.select_atoms(f'resid {int(res_id + 1)}')[0].resname
                 for res_id in arr[0]]

    arr = np.vstack((arr, res_names))
    return pd.DataFrame(arr.T, columns=['Residue Id', 'RMSF', "Res Name"])  # Convert to dataframe for prettier print


def populate_variance_showcase_df(analysis_actors_dict, drug_type, inp_df):
    '''
    Creates a DataFrame having for each drug the number of PCs needed in order to have 50%, 75% and 95% variance

    Args:
         analysis_actors_dict: Dict(
                                "Agonists": List[DirectoryPath (str), AnalysisActor.class]
                                "Antagonists": List[DirectoryPath (str), AnalysisActor.class]
                              )
        drug_type (str): 'Agonists' or 'Antagonists'
        inp_df (pd.DataFrame): The column names initialized empty Dataframe or the output of a call of this method

    Returns:
        pd.DataFrame: The inp_df in which we appended info about the drug type we specified
    '''
    for which_drug in analysis_actors_dict[drug_type]:
        pca_var_row = pd.DataFrame([[
            which_drug[1].drug_name,
            drug_type,
            np.where(which_drug[1].pca_res.cumulated_variance > 0.5)[0][0] + 1,  # We +1 since the np.where will return
            np.where(which_drug[1].pca_res.cumulated_variance > 0.75)[0][0] + 1,  # the 0 based index of the PC
            np.where(which_drug[1].pca_res.cumulated_variance > 0.95)[0][0] + 1]
        ], columns=['Drug Name', 'Type', '50% Variance', '75% Variance', '95% Variance'])
        inp_df = inp_df.append(pca_var_row, ignore_index=True)

    return inp_df


def project_pca_on_2d(analysis_actors_dict, drug_type, plot_rows=8, plot_cols=3):
    """
    Plots the 2d projection on the first two PCs of the atom space. The colorbar expresses the progression
    of the frames (color0 -> frame0, color1 -> last_frame).
    The plot is shown inside the function but if needed it can be easily changed to return it.

    Args:
        analysis_actors_dict: Dict(
                                "Agonists": List[DirectoryPath (str), AnalysisActor.class]
                                "Antagonists": List[DirectoryPath (str), AnalysisActor.class]
                              )
        drug_type (str): 'Agonists' or 'Antagonists'
        plot_rows (int): How many rows we will have on the final plot
        plot_cols (int): How many columns we will have on the final plot

    """
    fig = plt.figure(figsize=(18, 40))
    plot_index = 1

    for which_drug in tqdm(analysis_actors_dict[drug_type], desc="Projecting " + drug_type):
        atomgroup = which_drug[1].uni.select_atoms('name CA')  # Select the atoms pca was performed on
        pca_space_2D = which_drug[1].pca_res.transform(atomgroup, 2)  # Do the transformation on the selected atoms
        step = 1  # Frames we are skipping for computational reasons (if step == 1 then no frame is skipped)

        # Scatter Plotting
        ax = fig.add_subplot(plot_rows, plot_cols, plot_index)
        plt.scatter(pca_space_2D[::step, 0], pca_space_2D[::step, 1],
                    c=np.arange(len(pca_space_2D) / step) / (len(pca_space_2D) / step), marker='o')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(
            f'{which_drug[1].drug_name} | Structural Motion Variance: {np.around(which_drug[1].pca_res.cumulated_variance[1], decimals=3)}')
        plt.colorbar()  # Add the colorbar which goes from color0 to color1 as frames progress
        plot_index += 1

    fig.suptitle(f'PCA 2D Projection of {drug_type} as frames progress', fontsize=26, y=1.03)
    plt.tight_layout()
    plt.show()


def calculate_average_cols_hbonds(analysis_actors_dict):
    """
    Calculates the mean of hydrogen bonds of each frame for all the agonists.
    Then does the same for the antagonists.

    Args:
        analysis_actors_dict: Dict(
                                "Agonists": List[DirectoryPath (str), AnalysisActor.class]
                                "Antagonists": List[DirectoryPath (str), AnalysisActor.class]
                              )

    Returns:
        Tuple(np.array[#frames], np.array[#frames])
    """
    stacked_agonists = [len(frame_bonds) for frame_bonds in analysis_actors_dict['Agonists'][0][1].hbonds]
    for which_drug in analysis_actors_dict['Agonists'][1:]:
        stacked_agonists = np.vstack((stacked_agonists, [len(frame_bonds) for frame_bonds in which_drug[1].hbonds]))
    avg_agonists_cols = np.mean(stacked_agonists, axis=0)

    stacked_antagonists = [len(frame_bonds) for frame_bonds in analysis_actors_dict['Antagonists'][0][1].hbonds]
    for which_drug in analysis_actors_dict['Antagonists'][1:]:
        stacked_antagonists = np.vstack(
            (stacked_antagonists, [len(frame_bonds) for frame_bonds in which_drug[1].hbonds]))
    avg_antagonists_cols = np.mean(stacked_antagonists, axis=0)

    return avg_agonists_cols, avg_antagonists_cols


def calculate_average_cols_sbridges(analysis_actors_dict):
    """
    Calculates the mean of salt bridges ratio of each frame for all the agonists.
    Then does the same for the antagonists.

    Args:
        analysis_actors_dict: Dict(
                                "Agonists": List[DirectoryPath (str), AnalysisActor.class]
                                "Antagonists": List[DirectoryPath (str), AnalysisActor.class]
                              )

    Returns:
        Tuple(np.array[#frames], np.array[#frames])
    """
    stacked_agonists = analysis_actors_dict['Agonists'][0][1].salt_bridges.timeseries[:, 1]
    for which_drug in analysis_actors_dict['Agonists'][1:]:
        stacked_agonists = np.vstack((stacked_agonists, which_drug[1].salt_bridges.timeseries[:, 1]))
    avg_agonists_cols = np.mean(stacked_agonists, axis=0)

    stacked_antagonists = analysis_actors_dict['Antagonists'][0][1].salt_bridges.timeseries[:, 1]
    for which_drug in analysis_actors_dict['Antagonists'][1:]:
        stacked_antagonists = np.vstack((stacked_antagonists, which_drug[1].salt_bridges.timeseries[:, 1]))
    avg_antagonists_cols = np.mean(stacked_antagonists, axis=0)

    return avg_agonists_cols, avg_antagonists_cols