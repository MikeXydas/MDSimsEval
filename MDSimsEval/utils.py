import os
import logging
import imgkit

from tqdm import tqdm

from .AnalysisActorClass import AnalysisActor


def create_analysis_actor_dict(root_directory):
    """
    Reads the simulations (topologies, trajectories and sasa.xvg, salts if available) and stores them in a dictionary
    structure. The dictionary structure called ``analysis_actors_dict`` is the core structure that our functions
    take as an argument.

     Example:
        ::

            from MDSimsEval.utils import create_analysis_actor_dict
            analysis_actors_dict = create_analysis_actor_dict('path_to_data_directory/')

            analysis_actors_dict['Agonists'][0].info()
            Out:
                <<< Info of 5-MeOT >>>
                Number of Frames: 2500
                Number of Atoms: 4743
                Number of Residues: 291

    Args:
        root_directory(str): The path of the input directory having the expected structure on the documentation

    Returns:
        analysis_actors_dict::

                            Dict(
                                "Agonists": List[AnalysisActor.class]
                                "Antagonists": List[AnalysisActor.class]
                              )

    """
    # The input directory path must end with a "/"
    if not root_directory.endswith("/"):
        root_directory = root_directory + "/"

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
            simulations.set_description(f"{which_dir} | {which_sim}")  # Updating the tqdm progress bar description

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
    # Alphabetical sorting on ligand name
    analysis_actors_dict['Agonists'] = sorted(analysis_actors_dict['Agonists'], key=lambda x: x.drug_name)
    analysis_actors_dict['Antagonists'] = sorted(analysis_actors_dict['Antagonists'], key=lambda x: x.drug_name)

    return analysis_actors_dict


def render_corr_df(corr_df, filepath, reversed=False):
    """
    Renders and saves a correlation heatmap which is visually interpretable.

    .. figure:: ../_static/rmsf_corr.png
        :width: 600px
        :align: center
        :height: 250px
        :alt: rmsf corr figure missing

        Correlation heatmap, click for higher resolution.

    .. note::

         In order to save the image as ``.png`` you must install ``wkhtmltopdf`` via ``sudo apt-get install
         wkhtmltopdf`` on your machine. Else the output will be in a ``.html`` file and can be viewed using any browser.

    Args:
        corr_df(pd.DataFrame): A DataFrame of pairwise correlations
        filepath(str): The full path the heatmap save location eg. ``/dir1/dir2/name_of_file``
        reversed(boolean): If you want the coolwarm colormap to be inversed

    """
    colormap = 'coolwarm_r' if reversed else 'coolwarm'

    html_render = corr_df.style.background_gradient(cmap=colormap, axis=None).set_precision(2).render()

    # Saving the correlation matrix
    try:
        # If wkhtmltopdf is installed save the results as a .png
        imgkit.from_string(html_render, f"{filepath}.png")
    except IOError:
        # Save the html of the correlation map, which can be rendered by a browser
        with open(f"{filepath}.html", "w") as text_file:
            text_file.write(html_render)


def complement_residues(ligand_list_full, ligand_list_selected):
    """
    Given a list of ligands and a list of ligands selected from the previous list, return the ligands included in
    the first but not the second list.

    Args:
        ligand_list_full: A list of ``List[AnalysisActorClass]`` containing the full set of ligands
        ligand_list_selected: A list of ``List[AnalysisActorClass]`` containing a subset of the ``ligand_list_full``

    Returns:
        A list of ``List[AnalysisActorClass]`` containing the complement of the two sets above

    """
    drug_names_selected = set([ligand.drug_name for ligand in ligand_list_selected])
    returned_list = [ligand for ligand in ligand_list_full if ligand.drug_name not in drug_names_selected]

    return returned_list
