# -*- coding: utf-8 -*-
"""
Rg and SASA are two correlated metrics which focus on describing how the protein expands or shrinks as frames pass.

The functions in this module are based on frame aggregating techniques per class. For example we will take the ``Rg`` of
all the agonists on frame x and find the average. We will do that for every frame and end up with an average description
of increases or decreases in ``Rg``.

"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def calculate_average_cols_rg(analysis_actors_dict):
    """
    Calculates the mean of Rg of each frame for both classes.

    Args:
        analysis_actors_dict: ``{ "Agonists": List[AnalysisActor.class], "Antagonists": List[AnalysisActor.class] }``

    Returns:
        Tuple(np.array[#frames], np.array[#frames])

    """

    stacked_agonists = analysis_actors_dict['Agonists'][0].rg_res
    for which_ligand in analysis_actors_dict['Agonists'][1:]:
        try:
            stacked_agonists = np.vstack((stacked_agonists, which_ligand.rg_res))
        except ValueError:
            logging.warning(f"Ligand {which_ligand.drug_name} not having the proper amount of frames.")
    avg_agonists_cols = np.mean(stacked_agonists, axis=0)

    stacked_antagonists = analysis_actors_dict['Antagonists'][0].rg_res
    for which_ligand in analysis_actors_dict['Antagonists'][1:]:
        try:
            stacked_antagonists = np.vstack((stacked_antagonists, which_ligand.rg_res))
        except ValueError:
            logging.warning(f"Ligand {which_ligand.drug_name} not having the proper amount of frames.")
    avg_antagonists_cols = np.mean(stacked_antagonists, axis=0)

    return avg_agonists_cols, avg_antagonists_cols


def summarize_rg(analysis_actors_dict, dir_path, rolling_window=100):
    """
    Creates a plot summarizing how the ``Rg`` behaves.

    |

    .. figure:: ../_static/rg_averaged_mean_std.png
        :width: 500px
        :align: center
        :height: 550px
        :alt: summarizing info rg missing

        Summarizing info of Rg for each class (Agonists, Antagonists)

    Args:
        analysis_actors_dict: ``{ "Agonists": List[AnalysisActor.class], "Antagonists": List[AnalysisActor.class] }``
        dir_path (str): The path of the directory the plot will be saved (must end with a ``/``)
        rolling_window (int): The size of the window for the rolling avg and std

    """
    agon_rg_avg, antagon_rg_avg = calculate_average_cols_rg(analysis_actors_dict)

    fig = plt.figure(figsize=(10, 14))
    gs = GridSpec(3, 2, figure=fig)

    # Plot frame averaged scatter plots of agonists vs antagonists
    ax = fig.add_subplot(gs[0:2, :])
    ax.scatter(np.arange(agon_rg_avg.shape[0]), agon_rg_avg, label="Agonists Rg")
    ax.scatter(np.arange(agon_rg_avg.shape[0]), antagon_rg_avg, label="Antagonists Rg")
    ax.set_ylabel('Rg')
    ax.set_xlabel('Frame')
    ax.legend()
    plt.title("Average Radius of Gyration")

    # Calculate and plot the moving average of the series
    ax = fig.add_subplot(gs[2,0])
    moving_average_window = rolling_window
    moving_average_agon = [np.mean(agon_rg_avg[i:i + moving_average_window]) for i in range(len(agon_rg_avg) - moving_average_window)]
    moving_average_antagon = [np.mean(antagon_rg_avg[i:i + moving_average_window]) for i in range(len(antagon_rg_avg) - moving_average_window)]
    ax.plot(moving_average_agon, label="Agonists")
    ax.plot(moving_average_antagon, label="Antagonists")
    ax.set_ylabel('Rg')
    ax.set_xlabel('Starting Frame')
    ax.legend()
    plt.title(f"Moving average of Rg of window size: {moving_average_window}")

    #  Calculate and plot the moving standard deviation of the series
    ax = fig.add_subplot(gs[2,1])
    std_window = rolling_window
    stds_agon = [np.std(agon_rg_avg[i:i + std_window]) for i in range(len(agon_rg_avg) - std_window)]
    stds_antagon = [np.std(antagon_rg_avg[i:i + std_window]) for i in range(len(antagon_rg_avg) - std_window)]
    ax.plot(stds_agon, label="Agonists")
    ax.plot(stds_antagon, label="Antagonists")
    ax.set_ylabel('Std')
    ax.set_xlabel('Starting Frame')
    ax.legend()
    plt.title(f"Moving std of Rg of window size: {std_window}")

    plt.tight_layout()
    plt.savefig(f"{dir_path}rg_averaged_mean_std.png")

    return None

def calculate_average_cols_sasa(analysis_actors_dict):
    """
    Calculates the mean of SASA of each frame for both classes.

    Args:
        analysis_actors_dict: ``{ "Agonists": List[AnalysisActor.class], "Antagonists": List[AnalysisActor.class] }``

    Returns:
        Tuple(np.array[#frames], np.array[#frames])

    """

    stacked_agonists = analysis_actors_dict['Agonists'][0].sasa_res[1]
    for which_ligand in analysis_actors_dict['Agonists'][1:]:
        try:
            stacked_agonists = np.vstack((stacked_agonists, which_ligand.sasa_res[1]))
        except ValueError:
            logging.warning(f"Ligand {which_ligand.drug_name} not having the proper amount of frames.")
    avg_agonists_cols = np.mean(stacked_agonists, axis=0)

    stacked_antagonists = analysis_actors_dict['Antagonists'][0].sasa_res[1]
    for which_ligand in analysis_actors_dict['Antagonists'][1:]:
        try:
            stacked_antagonists = np.vstack((stacked_antagonists, which_ligand.sasa_res[1]))
        except ValueError:
            logging.warning(f"Ligand {which_ligand.drug_name} not having the proper amount of frames.")
    avg_antagonists_cols = np.mean(stacked_antagonists, axis=0)

    return avg_agonists_cols, avg_antagonists_cols


def summarize_sasa(analysis_actors_dict, dir_path, rolling_window=100):
    """
    Creates a plot summarizing how the ``SASA`` behaves.

    Args:
        analysis_actors_dict: ``{ "Agonists": List[AnalysisActor.class], "Antagonists": List[AnalysisActor.class] }``
        dir_path (str): The path of the directory the plot will be saved (must end with a ``/``)
        rolling_window (int): The size of the window for the rolling avg and std

    """
    agon_sasa_avg, antagon_sasa_avg = calculate_average_cols_sasa(analysis_actors_dict)

    fig = plt.figure(figsize=(10, 14))
    gs = GridSpec(3, 2, figure=fig)

    # Plot frame averaged scatter plots of agonists vs antagonists
    ax = fig.add_subplot(gs[0:2, :])
    ax.scatter(np.arange(agon_sasa_avg.shape[0]), agon_sasa_avg, label="Agonists SASA")
    ax.scatter(np.arange(antagon_sasa_avg.shape[0]), antagon_sasa_avg, label="Antagonists SASA")
    ax.set_ylabel('SASA (A^2)')
    ax.set_xlabel('Frame')
    ax.legend()
    plt.title("Average Solvent Accessible Surface Area")

    # Calculate and plot the moving average of the series
    ax = fig.add_subplot(gs[2, 0])
    moving_average_window = rolling_window
    moving_average_agon = [np.mean(agon_sasa_avg[i:i + moving_average_window]) for i in
                           range(len(agon_sasa_avg) - moving_average_window)]
    moving_average_antagon = [np.mean(antagon_sasa_avg[i:i + moving_average_window]) for i in
                              range(len(antagon_sasa_avg) - moving_average_window)]
    ax.plot(moving_average_agon, label="Agonists")
    ax.plot(moving_average_antagon, label="Antagonists")
    ax.set_ylabel('SASA')
    ax.set_xlabel('Starting Frame')
    ax.legend()
    plt.title(f"Moving average of SASA of window size: {moving_average_window}")

    #  Calculate and plot the moving standard deviation of the series
    ax = fig.add_subplot(gs[2, 1])
    std_window = rolling_window
    stds_agon = [np.std(agon_sasa_avg[i:i + std_window]) for i in range(len(agon_sasa_avg) - std_window)]
    stds_antagon = [np.std(antagon_sasa_avg[i:i + std_window]) for i in range(len(antagon_sasa_avg) - std_window)]
    ax.plot(stds_agon, label="Agonists")
    ax.plot(stds_antagon, label="Antagonists")
    ax.set_ylabel('Std')
    ax.set_xlabel('Starting Frame')
    ax.legend()
    plt.title(f"Moving std of SASA of window size: {std_window}")
    plt.tight_layout()

    plt.savefig(f'{dir_path}sasa_averaged_mean_std.png')

    return None


def stack_rgs_sasa(analysis_actors_dict):
    rg_stacked = np.array(analysis_actors_dict['Agonists'][0].get_radius_of_gyration())
    for which_ligand in analysis_actors_dict['Agonists'][1:] + analysis_actors_dict['Antagonists']:
        rg_stacked = np.vstack((rg_stacked, np.array(which_ligand.get_radius_of_gyration())))

    sasa_stacked = np.array(analysis_actors_dict['Agonists'][0].get_sasa()[1])
    for which_ligand in analysis_actors_dict['Agonists'][1:] + analysis_actors_dict['Antagonists']:
        sasa_stacked = np.vstack((sasa_stacked, np.array(which_ligand.get_sasa()[1])))

    return rg_stacked, sasa_stacked


def rg_sasa_mean_mean_plot(analysis_actors_dict, dir_path, start=0, stop=2500):
    """
    Creates a mean of Rg - mean of SASA plot on the specified window with annotations on the plots.

    |

    .. figure:: ../_static/rg_sasa_mean.png
        :width: 600px
        :align: center
        :height: 450px
        :alt: rg sasa mean figure missing

        Mean Rg - Mean SASA Plot with ligand name annotations

    Args:
        analysis_actors_dict: ``{ "Agonists": List[AnalysisActor.class], "Antagonists": List[AnalysisActor.class] }``
        dir_path (str): The path of the directory the plot will be saved (must end with a ``/``)
        start(int): The starting frame of the calculations
        stop(int): The stopping frame of the calculations

    """
    rg_stacked, sasa_stacked = stack_rgs_sasa(analysis_actors_dict)

    fig = plt.figure(figsize=(18, 15))
    ax = fig.add_subplot(1, 1, 1)

    # Get the dot points coordinates
    agonists_rg_mean = np.mean(rg_stacked[:len(analysis_actors_dict['Agonists']), start:stop + 1], axis=1)
    antagonists_rg_mean = np.mean(rg_stacked[len(analysis_actors_dict['Agonists']):, start:stop + 1], axis=1)

    agonists_sasa_mean = np.mean(sasa_stacked[:len(analysis_actors_dict['Agonists']), start:stop + 1], axis=1)
    antagonists_sasa_mean = np.mean(sasa_stacked[len(analysis_actors_dict['Agonists']):, start:stop + 1], axis=1)

    plt.scatter(agonists_rg_mean, agonists_sasa_mean, label="Agonists", color="blue", s=150)
    plt.scatter(antagonists_rg_mean, antagonists_sasa_mean, label="Antagonists", color="orange", s=150)

    # Annotate the plot by putting the name of the ligand on each dot
    for which_ligand in range(len(analysis_actors_dict['Agonists'])):
        ax.annotate(analysis_actors_dict['Agonists'][which_ligand].drug_name,
                    (agonists_rg_mean[which_ligand] + 0.002,
                     agonists_sasa_mean[which_ligand] + 0.15),
                    fontsize=18)

    for which_ligand in range(len(analysis_actors_dict['Antagonists'])):
        ax.annotate(analysis_actors_dict['Antagonists'][which_ligand].drug_name,
                    (antagonists_rg_mean[which_ligand] + 0.002,
                     antagonists_sasa_mean[which_ligand] + 0.15),
                    fontsize=18)

    plt.xlabel("Mean Rg", fontsize=24)
    plt.xticks(np.arange(20.8, 21.5, 0.1), fontsize=18)
    plt.yticks(np.arange(148, 168, 2), fontsize=18)
    plt.ylabel("Mean SASA", fontsize=24)
    plt.title(f"Rg Mean on time - SASA Mean on time  | Frames: {start} - {stop}", fontsize=28)
    plt.legend(prop={'size': 26}, markerscale=3)
    plt.grid()

    plt.savefig(f'{dir_path}rg_sasa_mean.png', format='png')

    return None


def rg_sasa_std_std_plot(analysis_actors_dict, dir_path, start=0, stop=2500):
    """
    Creates a std of Rg - std of SASA plot on the specified window with annotations on the plots.

    Args:
        analysis_actors_dict: ``{ "Agonists": List[AnalysisActor.class], "Antagonists": List[AnalysisActor.class] }``
        dir_path (str): The path of the directory the plot will be saved (must end with a ``/``)
        start(int): The starting frame of the calculations
        stop(int): The stopping frame of the calculations

    """
    rg_stacked, sasa_stacked = stack_rgs_sasa(analysis_actors_dict)

    fig = plt.figure(figsize=(18, 15))
    ax = fig.add_subplot(1, 1, 1)

    # Get the dot points coordinates
    agonists_rg_std = np.std(rg_stacked[:len(analysis_actors_dict['Agonists']), start:stop + 1], axis=1)
    antagonists_rg_std = np.std(rg_stacked[len(analysis_actors_dict['Agonists']):, start:stop + 1], axis=1)

    agonists_sasa_std = np.std(sasa_stacked[:len(analysis_actors_dict['Agonists']), start:stop + 1], axis=1)
    antagonists_sasa_std = np.std(sasa_stacked[len(analysis_actors_dict['Agonists']):, start:stop + 1], axis=1)

    plt.scatter(agonists_rg_std, agonists_sasa_std, label="Agonists", color="blue", s=150)
    plt.scatter(antagonists_rg_std, antagonists_sasa_std, label="Antagonists", color="orange", s=150)

    # Annotate the plot by putting the name of the ligand on each dot
    for which_ligand in range(len(analysis_actors_dict['Agonists'])):
        ax.annotate(analysis_actors_dict['Agonists'][which_ligand].drug_name,
                    (agonists_rg_std[which_ligand] + 0.0004,
                     agonists_sasa_std[which_ligand] + 0.02),
                     fontsize=18)

    for which_ligand in range(len(analysis_actors_dict['Antagonists'])):
        ax.annotate(analysis_actors_dict['Antagonists'][which_ligand].drug_name,
                    (antagonists_rg_std[which_ligand] + 0.0004,
                     antagonists_sasa_std[which_ligand] + 0.02),
                    fontsize=18)

    ax.set_ylim(1.75, 4.25)
    ax.set_xlim(0.05, 0.13)

    plt.xlabel("Std Rg", fontsize=24)
    plt.xticks(np.arange(0.05, 0.14, 0.01), fontsize=18)
    plt.yticks(np.arange(1.75, 4.5, 0.25), fontsize=18)
    plt.ylabel("Std SASA", fontsize=24)
    plt.title(f"Rg Std on time - SASA Std on time  | Frames: {start} - {end}", fontsize=28)
    plt.legend(prop={'size': 26}, markerscale=3)
    plt.grid(linewidth=2)

    plt.savefig(f'{dir_path}rg_sasa_std.png', format='png')

    return None
