import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from tqdm import tqdm


def scree_plot(analysis_actors_dict, dir_path, pcs_on_scree_plot=50, variance_ratio_line=0.75):
    """
    Creates a plot with the scree plots for each ligand and saves it on the specified ``dir_path``. With blue color is
    class 1 and with orange color class 2.

    Args:
        analysis_actors_dict: ``{ "Agonists": List[AnalysisActor.class], "Antagonists": List[AnalysisActor.class] }``
        dir_path (str): The path of the directory the plot will be saved (must end with a ``/``)
        pcs_on_scree_plot(int): The number of the first PCs that will be used on the scree plots
        variance_ratio_line(float): Float from 0.0 to 1.0 which specifies the variance ratio that a vertical line will
        be plotted

    """
    # Get the dimensions of the final plot
    plot_cols = 3
    plot_rows = math.ceil(len(analysis_actors_dict['Agonists']) + len(analysis_actors_dict['Antagonists']) / plot_cols)

    fig = plt.figure(figsize=(18, 6 * plot_rows))
    plot_index = 1

    # Agonists Iteration
    for which_ligand in analysis_actors_dict['Agonists']:
        ax = fig.add_subplot(plot_rows, plot_cols, plot_index)
        plt.axvline(x=np.where(np.cumsum(which_ligand.pca_res.explained_variance_ratio_) > variance_ratio_line)[0][0],
                    ls='--', c='grey', label=f"Reached {int(variance_ratio_line * 100)}% variance")
        plt.plot(np.arange(len(which_ligand.pca_res.explained_variance_[:pcs_on_scree_plot])),
                 which_ligand.pca_res.explained_variance_[:pcs_on_scree_plot], label="Variance Ratio")
        plt.ylabel("Variance")
        plt.xlabel("#PC")
        plt.title(which_ligand.drug_name)
        plt.legend()
        plot_index += 1

    # Antagonists Iteration
    for which_ligand in analysis_actors_dict['Antagonists']:
        ax = fig.add_subplot(plot_rows, plot_cols, plot_index)
        plt.axvline(x=np.where(np.cumsum(which_ligand.pca_res.explained_variance_ratio_) > variance_ratio_line)[0][0],
                    ls='--', c='grey', label=f"Reached {int(variance_ratio_line * 100)}% variance")
        plt.plot(np.arange(len(which_ligand.pca_res.explained_variance_[:pcs_on_scree_plot])),
                 which_ligand.pca_res.explained_variance_[:pcs_on_scree_plot], label="Variance", color='orange')
        plt.ylabel("Variance")
        plt.xlabel("#PC")
        plt.title(which_ligand.drug_name)
        plt.legend()
        plot_index += 1

    fig.suptitle('PCA Scree Plots\nAgonists: Blue\nAntagonists: Orange', fontsize=26, y=0.93)

    plt.savefig(f'{dir_path}pca_scree_plots.png', format='png')


def populate_variance_showcase_df(analysis_actors_dict, drug_type):
    """
    Creates a DataFrame having for each drug the number of PCs needed in order to have 50%, 75% and 95% variance

    Args:
        analysis_actors_dict: ``{ "Agonists": List[AnalysisActor.class], "Antagonists": List[AnalysisActor.class] }``
        drug_type (str): The class name ('Agonists' or 'Antagonists')

    Returns:
        pd.DataFrame: A DataFrame with columns ``['Drug Name', 'Type', '50% Variance', '75% Variance', '95% Variance']``
    """
    inp_df = pd.DataFrame(columns=['Drug Name', 'Type', '50% Variance', '75% Variance', '95% Variance'])
    for which_ligand in analysis_actors_dict[drug_type]:
        pca_var_row = pd.DataFrame([[
            which_ligand.drug_name,
            drug_type,
            np.where(np.cumsum(which_ligand.pca_res.explained_variance_ratio_) > 0.5)[0][0] + 1,
            # We +1 since the np.where will return
            np.where(np.cumsum(which_ligand.pca_res.explained_variance_ratio_) > 0.75)[0][0] + 1,
            # the 0 based index of the PC
            np.where(np.cumsum(which_ligand.pca_res.explained_variance_ratio_) > 0.95)[0][0] + 1]
        ], columns=['Drug Name', 'Type', '50% Variance', '75% Variance', '95% Variance'])
        inp_df = inp_df.append(pca_var_row, ignore_index=True)

    return inp_df


def project_pca_on_2d(analysis_actors_dict, drug_type, dir_path):
    """
    Plots the 2d projection on the first two PCs of the atom space. The colorbar expresses the progression
    of the frames (color0 -> frame0, color1 -> last_frame).
    The plot is shown inside the function but if need can be easily be changed to return it.

    Args:
        analysis_actors_dict: ``{ "Agonists": List[AnalysisActor.class], "Antagonists": List[AnalysisActor.class] }``
        drug_type (str): 'Agonists' or 'Antagonists'
        dir_path (str): The path of the directory the plot will be saved (must end with a ``/``)

    """
    cols = 3
    rows = math.ceil(len(analysis_actors_dict[drug_type]) / cols)

    fig = plt.figure(figsize=(18, 25))
    plot_index = 1

    for which_ligand in tqdm(analysis_actors_dict[drug_type], desc="Projecting " + drug_type):
        pca_space_2D = which_ligand.pca_res.transform(
            which_ligand.pca_xyz)  # Transform on the atom selection that PCA was fitted
        step = 1  # Frames we are skipping for computational reasons (if step == 1 then no frame is skipped)

        # Scatter Plotting
        ax = fig.add_subplot(rows, cols, plot_index)
        plt.scatter(pca_space_2D[::step, 0], pca_space_2D[::step, 1],
                    c=np.arange(len(pca_space_2D) / step) / (len(pca_space_2D) / step), marker='o')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        explained_variance_2PC = which_ligand.pca_res.explained_variance_ratio_[0] + \
                                 which_ligand.pca_res.explained_variance_ratio_[1]
        plt.title(f'{which_ligand.drug_name} | Structural Motion Variance: {explained_variance_2PC}')
        plt.colorbar()  # Add the colorbar which goes from color0 to color1 as frames progress
        plot_index += 1

    fig.suptitle(f'PCA 2D Projection of {drug_type} as frames progress', fontsize=26, y=1.03)
    plt.tight_layout()

    plt.savefig(f'{dir_path}pca_{drug_type}_2d_projection.png', format='png')

    return None


def sort_residues_by_loadings(ligand, variance_explained=0.5):
    """
    Having as an input **a ligand** find the loadings of each residue and return them in descending order.
    The method combines first k PCs where k is defined by the variance_explained argument.

    Args:
        ligand(AnalysisActor.class): An AnalysisActor object in which PCA is calculated
        variance_explained (float): Defines which PCs will be combined to calcualte the final loadings

    Returns:
        pd.DataFrame where ResidueId is the index and each row contains the loadings of the residue
    """
    pca_res = ligand.get_pca()

    # How many pcs we need to cover variance_explained
    pcs_numb = np.where(np.cumsum(pca_res.explained_variance_ratio_) > variance_explained)[0][0] + 1

    # Calculate loadings using loadings = eigenvectors @ sqrt(eigenvalues)
    loadings = np.abs(pca_res.components_[:pcs_numb, :]).T @ np.sqrt(pca_res.explained_variance_[:pcs_numb])

    # Go from 3 * #residues columns to #residues columns, combining the 3 axes
    residue_loading = np.add.reduceat(loadings, range(0, len(loadings), 3))

    return pd.DataFrame(enumerate(residue_loading), columns=['ResidueId', ligand.drug_name]).set_index('ResidueId')


def loadings_heatmap(analysis_actors_dict, dir_path, explained_variance=0.75):
    """
    | Creates a heatmap of the loadings of the residues for all the ligands. The blue line separates Class 1 fromClass 2
    |

    .. figure:: ../_static/pca_loadings_heatmap.png
        :width: 550
        :align: center
        :height: 500px
        :alt: pca loadings heatmap missing

        PCA Loadings Heatmap, click for higher resolution.

    Args:
        analysis_actors_dict: ``{ "Agonists": List[AnalysisActor.class], "Antagonists": List[AnalysisActor.class] }``
        dir_path (str): The path of the directory the plot will be saved (must end with a ``/``)
        explained_variance(float 0.0 - 1.0): Defines the number of PCs that will be used for the loadings calculation

    """
    loadings_df = sort_residues_by_loadings(analysis_actors_dict['Agonists'][0], explained_variance)

    # Join all the loadings of each ligand
    for which_ligand in analysis_actors_dict['Agonists'][1:]:
        loadings_df = loadings_df.join(sort_residues_by_loadings(which_ligand, explained_variance))
    for which_ligand in analysis_actors_dict['Antagonists'][1:]:
        loadings_df = loadings_df.join(sort_residues_by_loadings(which_ligand, explained_variance))

    fig, ax = plt.subplots(figsize=(20, 15))

    sns.heatmap(loadings_df)  # Seaborn heatmap of the loadings
    plt.axvline(len(analysis_actors_dict['Agonists']))  # Vertical line spearating agonists from antagonists

    ax.axis('tight')
    ax.set(xticks=np.arange(len(loadings_df.columns)), xticklabels=loadings_df.columns,
           yticks=np.arange(0, len(loadings_df.index), 10), yticklabels=np.arange(0, len(loadings_df.index), 10))
    plt.xticks(rotation=45)

    plt.xlabel('Ligand', fontsize=18)
    plt.ylabel('Residue Id', fontsize=18)
    plt.title(f"Heatmap of Loadings of each ligand | Explained Variance: {int(explained_variance * 100)}%", fontsize=18)
    plt.tight_layout()

    plt.savefig(f'{dir_path}pca_loadings_heatmap.png', format='png')

    return None
