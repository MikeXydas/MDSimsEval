import MDAnalysis
import MDAnalysis.analysis.distances as dist_analysis

import os
import logging
import math

from AnalysisAgonistsAntagonists.AnalysisActor import AnalysisActor
from AnalysisAgonistsAntagonists.AnalysisEvaluationUtil import *
from sklearn.cluster import KMeans

# This is the AnalysisPipeline_5HT2A.ipynb notebook converted on a directory
# The AnalysisAgonistsAntagonists package contains the AnalysisActor class and the util functions used for plotting
# This main is mostly plots and output in the same order as in the order of the notebook

# Metadata that should be set accordingly
root_directory = '../../datasets/New_AI_MD/'

dir_list = os.listdir(root_directory)
if 'Agonists' not in dir_list: logging.error('Agonists directory not found')
if 'Antagonists' not in dir_list: logging.error('Antagonists directory not found')

analysis_actors_dict = {"Agonists": [], "Antagonists": []}

# Iterating through the directories tree in order to fill the analysis_actors_dict
# A warning is thrown when reading the Lorcaserin file
for which_dir in ['Agonists', 'Antagonists']:
    simulations = os.listdir(root_directory + which_dir + '/')
    for which_sim in tqdm(simulations, desc=which_dir):
        files = os.listdir(root_directory + which_dir + '/' + which_sim + '/')
        top = ""
        traj = ""
        sasa_filepath = ""
        for file in files:
            if file[-4:] == ".xtc":
                traj = root_directory + which_dir + '/' + which_sim + '/' + file
            elif file[-4:] == ".pdb":
                top = root_directory + which_dir + '/' + which_sim + '/' + file
            elif file == 'sasa.xvg':
                # Currently SASA is calculated using GROMACS before running this notebook
                sasa_filepath = root_directory + which_dir + '/' + which_sim + '/' + file
        if traj == "" or top == "":
            logging.error("Failed to find topology or trajectory file in: "
                          + root_directory + which_dir + '/' + which_sim)
        else:
            analysis_actors_dict[which_dir].append([root_directory + which_dir + '/' + which_sim + '/',
                                                    AnalysisActor(top, traj, which_sim, sasa_file=sasa_filepath)])

# Agonists
for which_actor in tqdm(analysis_actors_dict['Agonists'], desc="Agonists Calculations"):
    which_actor[1].perform_analysis(metrics=[])

# Antagonists
for which_actor in tqdm(analysis_actors_dict['Antagonists'], desc="Antagonists Calculations"):
    which_actor[1].perform_analysis(metrics=[])

# General info about future plots
total_plots = len(analysis_actors_dict['Agonists']) + len(analysis_actors_dict['Antagonists'])
plot_cols = 3
plot_rows = math.ceil(total_plots / plot_cols)

# Radius of Gyration
# We create plot of the Rg as frames progress for each drug
fig = plt.figure(figsize=(18, 40))
plot_index = 1

# Separate Plots
# Agonists Iteration
for which_drug in analysis_actors_dict['Agonists']:
    ax = fig.add_subplot(plot_rows, plot_cols, plot_index)
    plt.plot(np.arange(which_drug[1].get_frames_number()), which_drug[1].rg_res, label="Rg")
    plt.xlabel("Frames")
    plt.ylabel("Rg")
    plt.title(which_drug[1].drug_name)
    plot_index += 1

# Antagonists Iteration
for which_drug in analysis_actors_dict['Antagonists']:
    ax = fig.add_subplot(plot_rows, plot_cols, plot_index)
    plt.plot(np.arange(which_drug[1].get_frames_number()), which_drug[1].rg_res, label="Rg", color='orange')
    plt.xlabel("Frames")
    plt.ylabel("Rg")
    plt.title(which_drug[1].drug_name)
    plot_index += 1

fig.suptitle('Radius of Gyration\nAgonists: Blue\nAntagonists: Orange', fontsize=26, y=0.93)

plt.show()

# Frame Averaged Plots
agon_rg_avg, antagon_rg_avg = calculate_average_cols_rg(analysis_actors_dict)

fig = plt.figure(figsize=(10, 9))
ax = plt.subplot(111)
ax.plot(np.arange(agon_rg_avg.shape[0]), agon_rg_avg, label="Agonists Rg")
ax.plot(np.arange(agon_rg_avg.shape[0]), antagon_rg_avg, label="Antagonists Rg")
ax.set_ylabel('Rg')
ax.set_xlabel('Frame')

ax.legend()
plt.title("Average Radius of Gyration")
plt.show()

# SASA
# Separate Plots
# We create plot of the SASA as frames progress for each drug
fig = plt.figure(figsize=(18, 40))
plot_index = 1

# Agonists Iteration
for which_drug in analysis_actors_dict['Agonists']:
    ax = fig.add_subplot(plot_rows, plot_cols, plot_index)
    plt.plot(np.arange(which_drug[1].get_frames_number()), which_drug[1].sasa_res[1], label="SASA")
    plt.ylabel("SASA")
    plt.title(which_drug[1].drug_name)
    plot_index += 1

# Antagonists Iteration
for which_drug in analysis_actors_dict['Antagonists']:
    ax = fig.add_subplot(plot_rows, plot_cols, plot_index)
    plt.plot(np.arange(which_drug[1].get_frames_number()), which_drug[1].sasa_res[1], label="SASA", color='orange')
    plt.ylabel("SASA")
    plt.title(which_drug[1].drug_name)
    plot_index += 1

fig.suptitle('Solvent Accessible Surface Area\nAgonists: Blue\nAntagonists: Orange', fontsize=26, y=0.93)

plt.show()

# Frame Averaged Plots
agonists_sasa_avg, antagonists_sasa_avg = calculate_average_cols_sasa(analysis_actors_dict)

fig = plt.figure(figsize=(10, 9))
ax = plt.subplot(111)
ax.plot(np.arange(agonists_sasa_avg.shape[0]), agonists_sasa_avg, label="Agonists SASA")
ax.plot(np.arange(agonists_sasa_avg.shape[0]), antagonists_sasa_avg, label="Antagonists SASA")
ax.legend()
ax.set_ylabel('SASA')
ax.set_xlabel('Frame')
plt.title("Average Solvent Accessible Surface Area")
plt.show()

# RMSF
# Separate Plots
# We create plot of the RMSF as residue id progresses for each drug
fig = plt.figure(figsize=(18, 40))
plot_index = 1

# Agonists Iteration
for which_drug in analysis_actors_dict['Agonists']:
    ax = fig.add_subplot(plot_rows, plot_cols, plot_index)
    plt.plot(which_drug[1].uni.atoms.resnums, which_drug[1].rmsf_res.rmsf, label="RMSF")
    #     plt.plot(np.unique(which_drug[1].uni.atoms.resnums), get_avg_rmsf_per_residue(which_drug[1]), label="SASA")
    plt.xlabel("Residue Id")
    plt.ylabel("RMSF")
    plt.title(which_drug[1].drug_name)
    plot_index += 1

# Antagonists Iteration
for which_drug in analysis_actors_dict['Antagonists']:
    ax = fig.add_subplot(plot_rows, plot_cols, plot_index)
    plt.plot(which_drug[1].uni.atoms.resnums, which_drug[1].rmsf_res.rmsf, label="RMSF", color='orange')
    plt.xlabel("Residue Id")
    plt.ylabel("RMSF")
    plt.title(which_drug[1].drug_name)
    plot_index += 1

fig.suptitle('Root Mean Square Fluctuation\nAgonists: Blue\nAntagonists: Orange', fontsize=26, y=0.93)

plt.show()

# Residue Averaged Plots
agonists_rmsf_residue_avg, antagonists_rmsf_residue_avg = calculate_average_cols_rmsf(analysis_actors_dict)

fig = plt.figure(figsize=(10, 9))
x = np.arange(agonists_rmsf_residue_avg.shape[0])
ax = plt.subplot(111)
ax.plot(np.arange(agonists_rmsf_residue_avg.shape[0]), agonists_rmsf_residue_avg, label="Agonists RMSF")
ax.plot(np.arange(agonists_rmsf_residue_avg.shape[0]), antagonists_rmsf_residue_avg, label="Antagonists RMSF")
ax.legend()
ax.set_ylabel('RMSF')
ax.set_xlabel('Residue Id')
plt.title("Average RMSF per Residue")
fig.tight_layout()
plt.show()

# Top RMSF Residues
print("Printing Top-10 Average RMSF Residues of Agonists")
print(return_top_k(analysis_actors_dict, agonists_rmsf_residue_avg))

print("\nPrinting Top-10 Average RMSF Residues of Abs(Agonists_RMSF - Antagonists_RMSF)")
print(return_top_k(analysis_actors_dict, np.abs(agonists_rmsf_residue_avg - antagonists_rmsf_residue_avg)))

# PCA
# Scree Plots
# We create scree plot the top 50 PCs
pcs_on_scree_plot = 50  # Change the number of PCs used on the scree plot

fig = plt.figure(figsize=(18, 40))
plot_index = 1

# Agonists Iteration
for which_drug in analysis_actors_dict['Agonists']:
    ax = fig.add_subplot(plot_rows, plot_cols, plot_index)
    plt.plot(np.arange(len(which_drug[1].pca_res.variance[:pcs_on_scree_plot])),
             which_drug[1].pca_res.variance[:pcs_on_scree_plot],
             label="Variance")
    plt.ylabel("Variance")
    plt.title(which_drug[1].drug_name)
    plot_index += 1

# Antagonists Iteration
for which_drug in analysis_actors_dict['Antagonists']:
    ax = fig.add_subplot(plot_rows, plot_cols, plot_index)
    plt.plot(np.arange(len(which_drug[1].pca_res.variance[:pcs_on_scree_plot])),
             which_drug[1].pca_res.variance[:pcs_on_scree_plot],
             label="Variance", color='orange')
    plt.ylabel("Variance")
    plt.title(which_drug[1].drug_name)
    plot_index += 1

fig.suptitle('PCA Scree Plots\nAgonists: Blue\nAntagonists: Orange', fontsize=26, y=0.93)

plt.show()

# Variance showcase
# Get the PCs that give us 50%, 75%, 95% variance for each drug
pca_df = pd.DataFrame(columns=['Drug Name', 'Type', '50% Variance', '75% Variance', '95% Variance'])
pca_df = populate_variance_showcase_df(analysis_actors_dict, 'Agonists', pca_df)
pca_df = populate_variance_showcase_df(analysis_actors_dict, 'Antagonists', pca_df)
print(pca_df)

# 2D Projection
project_pca_on_2d(analysis_actors_dict, 'Agonists')  # Plot the agonists projection
project_pca_on_2d(analysis_actors_dict, 'Antagonists')  # Plot the antagonists projection

# K-means on PC1
# Initialize the array
atom_group = analysis_actors_dict['Agonists'][0][1].uni.select_atoms('name CA')  # Select the atoms pca was performed on
projections_1d_array = analysis_actors_dict['Agonists'][0][1].pca_res.transform(atom_group,
                                                                                1).T  # Do the transformation on the selected atoms

# Agonists Iteration
for which_drug in tqdm(analysis_actors_dict['Agonists'][1:], desc="Projecting 1D " + "Agonists"):
    atom_group = which_drug[1].uni.select_atoms('name CA')  # Select the atoms pca was performed on
    pca_space_1D = which_drug[1].pca_res.transform(atom_group, 1)  # Do the transformation on the selected atoms
    projections_1d_array = np.vstack((projections_1d_array, pca_space_1D.T))

# Antagonists Iteration
for which_drug in tqdm(analysis_actors_dict['Antagonists'], desc="Projecting 1D " + "Antagonists"):
    atom_group = which_drug[1].uni.select_atoms('name CA')  # Select the atoms pca was performed on
    pca_space_1D = which_drug[1].pca_res.transform(atom_group, 1)  # Do the transformation on the selected atoms
    projections_1d_array = np.vstack((projections_1d_array, pca_space_1D.T))

kmeans = KMeans(n_clusters=2).fit(projections_1d_array)  # fit k-means with the k-means++ centers initialization
drug_types = np.full((1, len(analysis_actors_dict['Agonists'])), "Agonist")
drug_types = np.hstack((drug_types, np.full((1, len(analysis_actors_dict['Antagonists'])), "Antagonist")))

names_agonists = [which_drug[1].drug_name for which_drug in analysis_actors_dict['Agonists']]
names_antagonists = [which_drug[1].drug_name for which_drug in analysis_actors_dict['Antagonists']]
all_names = names_agonists + names_antagonists
kmeans_results = {
    "Drug_Names": np.array(all_names),
    "Drug_Types": drug_types[0],
    "2 Clusters": KMeans(n_clusters=2).fit(projections_1d_array).labels_,
    "3 Clusters": KMeans(n_clusters=3).fit(projections_1d_array).labels_,
    "4 Clusters": KMeans(n_clusters=4).fit(projections_1d_array).labels_,
    "6 Clusters": KMeans(n_clusters=6).fit(projections_1d_array).labels_
}
kmeans_df = pd.DataFrame(kmeans_results)
print(kmeans_df)

# Hydrogen Bonds
# Separate Plots
# We create a plot of the hydrogen bonds progression
fig = plt.figure(figsize=(18, 40))
plot_index = 1

# Agonists Iteration
for which_drug in analysis_actors_dict['Agonists']:
    ax = fig.add_subplot(plot_rows, plot_cols, plot_index)
    bonds_numb = [len(frame_bonds) for frame_bonds in
                  which_drug[1].hbonds]  # Calculate the number of bonds on each frame
    plt.plot(np.arange(len(bonds_numb)), bonds_numb, label="Hydrogen Bonds", color='blue')
    plt.ylabel("H Bonds")
    plt.title(which_drug[1].drug_name)
    plot_index += 1

# Antagonists Iteration
for which_drug in analysis_actors_dict['Antagonists']:
    ax = fig.add_subplot(plot_rows, plot_cols, plot_index)
    bonds_numb = [len(frame_bonds) for frame_bonds in
                  which_drug[1].hbonds]  # Calculate the number of bonds on each frame
    plt.plot(np.arange(len(bonds_numb)), bonds_numb, label="Hydrogen Bonds", color='orange')
    plt.ylabel("H Bonds")
    plt.title(which_drug[1].drug_name)
    plot_index += 1

fig.suptitle('Hydrogen Bonds Plots\nAgonists: Blue\nAntagonists: Orange', fontsize=26, y=0.93)

plt.show()

# Frame Averaged Plots
agonists_hbonds_avg, antagonists_hbonds_avg = calculate_average_cols_hbonds(analysis_actors_dict)

fig = plt.figure(figsize=(10, 9))
ax = plt.subplot(111)
ax.plot(np.arange(agonists_hbonds_avg.shape[0]), agonists_hbonds_avg, label="Agonists H Bonds", color='blue')
ax.plot(np.arange(agonists_hbonds_avg.shape[0]), antagonists_hbonds_avg, label="Antagonists H Bonds", color='orange')
ax.legend()
ax.set_ylabel('H Bonds')
ax.set_xlabel('Frame')
plt.title("Average NUmber of Hydrogen Bonds")
plt.show()

# Salt Bridges
# Separate Plots
# We create a plot of the salt bridges progression
fig = plt.figure(figsize=(18, 40))
plot_index = 1

# Agonists Iteration
for which_drug in analysis_actors_dict['Agonists']:
    ax = fig.add_subplot(plot_rows, plot_cols, plot_index)
    plt.plot(which_drug[1].salt_bridges.timeseries[:, 0], which_drug[1].salt_bridges.timeseries[:, 1],
             label="Salt Bridges", color='blue')
    plt.ylabel("Fraction of SBs")
    plt.title(which_drug[1].drug_name)
    plot_index += 1

# Antagonists Iteration
for which_drug in analysis_actors_dict['Antagonists']:
    ax = fig.add_subplot(plot_rows, plot_cols, plot_index)
    plt.plot(which_drug[1].salt_bridges.timeseries[:, 0], which_drug[1].salt_bridges.timeseries[:, 1],
             label="Salt Bridges", color='orange')
    plt.ylabel("Fraction of SBs")
    plt.title(which_drug[1].drug_name)
    plot_index += 1

fig.suptitle('Salt Bridges Plots\nAgonists: Blue\nAntagonists: Orange', fontsize=26, y=0.93)

plt.show()

# Frame Averaged Plot
agonists_sbridges_avg, antagonists_sbridges_avg = calculate_average_cols_sbridges(analysis_actors_dict)

fig = plt.figure(figsize=(10, 9))
ax = plt.subplot(111)
ax.plot(np.arange(agonists_sbridges_avg.shape[0]), agonists_sbridges_avg, label="Agonists Salt Bridges", color='blue')
ax.plot(np.arange(agonists_sbridges_avg.shape[0]), antagonists_sbridges_avg, label="Antagonists Salt Bridges",
        color='orange')
ax.legend()
ax.set_ylabel('S Bridges Fraction')
ax.set_xlabel('Frame')
plt.title("Average Salt Bridges Fraction'")
plt.show()

# Independent Experiments
# Distance between R3.50 (Arg105) and E6.30 (Glu209) of 5HT2A
# Read the Ergotamine simulation
erg_u = MDAnalysis.Universe("../../datasets/New_AI_MD/Agonists/Ergotamine/5HT2A_Ergotamine_1st_com_gro.pdb",
                            "../../datasets/New_AI_MD/Agonists/Ergotamine/5TH2A_Ergotamine_500ns_2500frames.xtc")

# Calculate the distance between Arg105 and Glu209 as frames progress
dists_saved = [dist_analysis.dist(erg_u.select_atoms('bynum 1711'), erg_u.select_atoms('bynum 3388'))[2][0]
               for ts in erg_u.trajectory]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(np.arange(len(dists_saved)), dists_saved, label="Ergotamine")
plt.ylabel("Distance")
plt.xlabel("Frame")
plt.title("Ergotamine Arg105, Glu209 Distance")
plt.show()

# The above distance for every drug (analysis_Actor_dict must be created)
fig = plt.figure(figsize=(18, 40))
plot_index = 1

# Agonists Iteration
for which_drug in tqdm(analysis_actors_dict['Agonists'], desc="Agonists Distance"):
    ax = fig.add_subplot(plot_rows, plot_cols, plot_index)
    dists_saved = [
        dist_analysis.dist(which_drug[1].uni.select_atoms('bynum 1711'), erg_u.select_atoms('bynum 3388'))[2][0]
        for ts in which_drug[1].uni.trajectory]
    plt.plot(np.arange(len(dists_saved)), dists_saved, color='blue')
    plt.ylabel("Distance")
    plt.title(which_drug[1].drug_name)
    plot_index += 1

# Antagonists Iteration
for which_drug in tqdm(analysis_actors_dict['Antagonists'], desc="Antagonists Distance"):
    ax = fig.add_subplot(plot_rows, plot_cols, plot_index)
    dists_saved = [
        dist_analysis.dist(which_drug[1].uni.select_atoms('bynum 1711'), erg_u.select_atoms('bynum 3388'))[2][0]
        for ts in which_drug[1].uni.trajectory]
    plt.plot(np.arange(len(dists_saved)), dists_saved, color='orange')
    plt.ylabel("Distance")
    plt.title(which_drug[1].drug_name)
    plot_index += 1

fig.suptitle('Arg105 and Glu209 Distances\nAgonists: Blue\nAntagonists: Orange', fontsize=26, y=0.93)
plt.show()
