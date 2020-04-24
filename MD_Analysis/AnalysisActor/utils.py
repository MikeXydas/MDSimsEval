import os
import logging

from tqdm import tqdm

from .AnalysisActorClass import AnalysisActor


# ## Reading the trajectories
#
# Emphasis must be given on reading the trajectory files in an organized and optimal way.
# The current approach is:
#
# 1. Input: path which points to a directory that contains to subdirectories with names **"agonists", "antagonists"**
# 2. Extract the file paths of trajectory.xtc, topology.pdb and sasa.xvg (if available)
# 3. Create the dictionary:
# ```python
# {
#     "Agonists": List[AnalysisActor.class]
#     "Antagonists": List[AnalysisActor.class]
# }
# ```
#
# **The trajectory and topology file are expected to have a file ending of .xtc and .pdb respectively,
# although we can easily expand it to more formats**
#
def create_analysis_actor_dict(root_directory):
    dir_list = os.listdir(root_directory)

    # Check that the root_directory contains the Agonists, Antagonists folders
    if 'Agonists' not in dir_list:
        logging.error('Agonists directory not found')
    if 'Antagonists' not in dir_list:
        logging.error('Antagonists directory not found')

    analysis_actors_dict = {"Agonists": [], "Antagonists": []}

    # Iterating through the directories tree in order to fill the analysis_actors_dict
    # A warning is thrown when reading the Lorcaserin file
    for which_dir in ['Agonists', 'Antagonists']:
        simulations = tqdm(os.listdir(root_directory + which_dir + '/'))
        for which_sim in simulations:
            simulations.set_description(f"{which_dir} | {which_sim}")    # Updating the tqdm progress bar description

            # Get the MD info files
            files = os.listdir(root_directory + which_dir + '/' + which_sim + '/')

            # Initialize file paths as empty
            top = ""
            traj = ""
            sasa_filepath = ""
            salts_directory = ""

            # Iterate through the files
            for file in files:
                if file[-4:] == ".xtc" and file[:3] != 'rms':
                    # Trajectory file
                    traj = root_directory + which_dir + '/' + which_sim + '/' + file
                elif file[-4:] == ".pdb":
                    # Topology file
                    top = root_directory + which_dir + '/' + which_sim + '/' + file
                elif file == 'sasa.xvg':
                    # Currently SASA is calculated using GROMACS command, gmx sasa
                    sasa_filepath = root_directory + which_dir + '/' + which_sim + '/' + file
                elif file == 'salts':
                    # Currently salt bridges are calculated using VMD extension
                    salts_directory = root_directory + which_dir + '/' + which_sim + '/' + file + '/'

            if traj == "" or top == "":
                logging.error("Failed to find topology or trajectory file in: " + root_directory + which_dir + '/'
                                                                                + which_sim + '/' + file)
            else:
                # Everything ok, construct a new AnalysisActor
                analysis_actors_dict[which_dir].append(
                                                        AnalysisActor(top, traj, which_sim,
                                                                      sasa_file=sasa_filepath,
                                                                      salts_directory=salts_directory)
                                                      )
    return analysis_actors_dict
