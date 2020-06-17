import MDAnalysis
from MDAnalysis.analysis.rms import RMSF
import mdtraj as md_traj

from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

import re
import os
import logging


class AnalysisActor:
    """
    The AnalysisActor is the core object which inputs a single topology and trajectory and performs the analysis.

    The class will be responsible for reading, running calculations and storing the results. The role of the class
    is mainly "structural" since it is the base input of all of our analysis functions. In the future a caching
    mechanism should be added so as to not have to perform multiple times the same calculations and instead read
    them from disk.

    Note:
        **Solvent Accessible Surface Area (SASA) and Salt bridges** are calculated outside of the package.
        More info on :ref:`reading_salt_sasa`.

    Example:
        Example on calculating RMSF and PCA on the whole ``analysis_actors_dict``
        ::

            from MDSimsEval.utils import create_analysis_actor_dict
            analysis_actors_dict = create_analysis_actor_dict('path_to_data_directory/')

            for ligand in analysis_actors_dict['Agonists'] + analysis_actors_dict['Antagonists']:
                ligand.perform_analysis(metrics=["RMSF", "PCA"])
    Args:
        topology (str): The topology filepath (.pdb, .gro etc)
        trajectory (str): The trajectory filepath (.xtc etc)
        drug_name (str): The name of the agonist or antagonist
        sasa_file (str, optional): Full path of the sasa.xvg file generated by Gromacs
        salts_directory(str, optional): Full path of the salts directory generated by the vmd extension
        
    Attributes:
        uni: The universe of atoms created by MDAnalysis tool
                (`MDAnalysis.core.universe <https://www.mdanalysis.org/docs/documentation_pages/core/universe.html>`_)
        mdtraj: The trajectory of atoms created by MDTraj tool
                (`mdtraj.Trajectory <http://mdtraj.org/latest/api/generated/mdtraj.Trajectory.html#mdtraj.Trajectory>`_)
        drug_name (str): The ligand name initialized in the construction of the object
        rg_res (:obj:`list` of double): Radius of gyration of each frame (`MDAnalysisRg <https://www.mdanalysis.org/MDAnalysisTutorial/trajectories.html#trajectory-analysis>`_)
        rmsf_res (:obj:`list` of double): RMSF of each atom of the **protein** (`MDAnalysisRMSF <https://www.mdanalysis.org/pmda/api/rmsf.html>`_)
        pca_res : Object containing eigenvectors and eigenvalues of CA atoms covariance matrix
                (`MDAnalysis.analysis.pca.PCA <https://www.mdanalysis.org/docs/documentation_pages/analysis/pca.html#MDAnalysis.analysis.pca.PCA>`_)
        pca_xyz (ndarray[#atoms_selected * 3, #frames]): The coordinates of the atoms selected for PCA
        
        sasa_res (ndarray[2, #frames]):  The calculation currently is performed outside of the pipeline
                                          using `gromacs <http://manual.gromacs.org/documentation/5.1/onlinehelp/gmx-sasa.html>`_.
                                          
        hbonds (ndarray[#frames, #hbonds_of_frame, 3]):    A list containing the atom indices involved in
                                                            each of the identified hydrogen bonds at each frame. 
                                                            Each element in the list is an array where each row contains 
                                                            three integer indices, (d_i, h_i, a_i), 
                                                            such that d_i is the index of the donor atom, 
                                                            h_i the index of the hydrogen atom, 
                                                            and a_i the index of the acceptor atom involved 
                                                            in a hydrogen bond which occurs in that frame.
                                                            (`MDtraj <http://mdtraj.org/1.9.3/api/generated/mdtraj.wernet_nilsson.html?highlight=wernet#mdtraj.wernet_nilsson>`_)
                                                            
        salt_bridges_df (pd.DataFrame[#NumberOfSaltBridges, 2 + DistancesPerFrame]):
                                    A pd.DataFrame with the first two columns ["Residue1", "Residue2"] being the 
                                    residue names that form the salt bridge and the next #frames column the distance
                                    between these two residues (`VMD plugin <https://www.ks.uiuc.edu/Research/vmd/plugins/saltbr/>`_)
                                    
        salt_bridges_over_time (ndarray[#frames]): A vector of size #frames containing the number of active salt
                                                    bridges per frame
          
    """

    def __init__(self, topology, trajectory, drug_name, sasa_file="", salts_directory=""):
        self.uni = MDAnalysis.Universe(topology, trajectory)
        # self.mdtraj = md_traj.load(trajectory, top=topology) Removed due to memory constraints, PCA will not work
        # self.mdtraj = self.mdtraj.superpose(self.mdtraj)    # Align to the first frame to avoid simulation anomalies
        self.drug_name = drug_name
        self.rg_res = None
        self.rmsf_res = None
        self.pca_res = None
        self.pca_xyz = None
        self.hbonds = None

        # The following are currently calculated with external modules (GROMACS, VMD)
        if salts_directory != "":
            self.salt_bridges_df = self.__read_salt_bridges_files(salts_directory)
            self.salt_bridges_over_time = self.__find_salt_bridges_per_frame(self.salt_bridges_df)
        else:
            self.salt_bridges_df = np.array([-1])
            self.salt_bridges_over_time = np.array([-1])
        self.sasa_res = self.__read_sasa_file(sasa_file) if sasa_file != "" else np.arange(1)

    def __read_sasa_file(self, sasa_filepath):
        """
        Private method that helps us parse the sasa.xvg file generated by
        `gmx sasa -f trajectory.xtc -s topology.tpr -o sasa.xvg`
        and and stores them into a numpy array

        Args:
            sasa_filepath (str): Filepath of the .xvg file to read

        Returns:
            np.array[2, #frames], eg [[0, 1, 2, ... , n], [SASA0, SASA1, SASA2, ..., SASAn]]
        """
        sasa_list = []
        p = re.compile('\s+([0-9\.]*)\s+([0-9\.]*)\\n')    # RegEx to extract the frame, SASA pair of each frame

        with open(sasa_filepath) as fp:
            lines = fp.readlines()
            for line in lines:
                if line[0] != '#' and line[0] != '@':    # Ignore the GROMACS comment lines
                    m = p.match(line)
                    sasa_list.append([m.group(1), m.group(2)])    # Group1: Frame, Group2: SASA

        sasa_list_arr = np.array(sasa_list).T.astype('float64')    # Cast to numpy array and transpose
        return sasa_list_arr

    def __read_salt_bridges_files(self, salts_directory):
        """
        Private method that will read the salt bridges distances from the files outputted by vmd
        and will parse them into a df.

        Args:
            salts_directory (str): The path of the directory containing the distances of each
            salt bridge as outputted by the vmd extension (https://www.ks.uiuc.edu/Research/vmd/plugins/saltbr/)

        Returns:
            pd.DataFrame: Rows-> Total number of salt bridges appearing,
                          Columns -> [Residue1, Residue2, DistFrame1, DistFrame2, ...]
        """
        # Get the files of salt bridges distances (one file per salt bridge)
        salt_files = os.listdir(salts_directory)
        # Initialize our array that will contain [res1, res2, DistFrame1, ...] for each salt bridge
        distances_array = np.array([0])

        for salt_file in salt_files:
            if salt_file != 'summary.txt':    # Ignore the log file
                # Get the residue names that have the salt bridge
                res_names = re.match(r'saltbr-(\w+)-(\w+).dat', salt_file).groups()

                # Read the distances of one bridge
                dists_of_bridge = np.array(pd.read_csv(filepath_or_buffer=salts_directory + salt_file, sep=' '
                                                       , header=None)[1])
                # Stack the residue names on the front
                row = np.hstack((np.array(res_names, dtype=object), dists_of_bridge))

                if distances_array.shape == (1,):
                    # First salt bridge -> initialize distances_array
                    distances_array = row
                else:
                    # Stack a new row of residue names, distances
                    distances_array = np.vstack((distances_array, row))

        # Create the column names of our DataFrame
        column_names = np.hstack((["Residue1", "Residue2"], np.arange(len(distances_array[0]) - 2)))

        # Cast to DataFrame and return
        return pd.DataFrame(distances_array, columns=column_names)

    def __find_salt_bridges_per_frame(self, salt_df, distance_criterion=4):
        """
        __read_salt_bridges_files must be called before calling this function. This function will return the number
        of occurring salt bridges per frame base on a distance criterion.

        Args:
            distance_criterion (float): Maximum distance in Angstroms that a bridge is occurring

        Returns:
            nd.array[#frames]: for each frame has the number of occurring salt bridges
        """
        distances_df = np.array(salt_df)[:, 2:]    # Ignore the residue names
        bridges_per_frame = [len(np.where(which_frame <= distance_criterion)[0]) for which_frame in distances_df.T]

        return bridges_per_frame

    def info(self):
        """ Prints basic info of the simulation """
        print(f'\n<<< Info of {self.drug_name} >>>')
        print(f'\tNumber of Frames: {len(self.uni.trajectory)}')
        print(f'\tNumber of Atoms: {len(self.uni.atoms)}')
        print(f'\tNumber of Residues: {len(self.uni.residues)}')

    def get_frames_number(self):
        """ Returns the number of frames of the trajectory """
        return len(self.uni.trajectory)

    def perform_analysis(self, metrics=()):
        """
        Runs the analysis methods for calculating the metrics specified by metrics argument
        
        Args:
            metrics (List[str]): |  A list of the metrics to be calculated. Available:
                                 |  Empty List []: All of the available metrics will be calculated (default)
                                 |  'Rg': Radius of Gyration
                                 |  'RMSF': Root Mean Square Fluctuations
                                 |  'SASA': Solvent Accessible Surface Area
                                 |  'PCA': Principal Component Analysis
                                 |  'Hbonds': Hydrogen Bonds
                                 |  'Salt': Calculate number of salt bridges
        """
        
        # Calculate Radius of Gyration as time progresses
        if "Rg" in metrics or len(metrics) == 0:
            self.rg_res = []
            for frame in self.uni.trajectory:
                self.rg_res.append(self.uni.atoms.radius_of_gyration())
                
        # Calculate Solvent Accessible Surface Area 
        if "SASA" in metrics or len(metrics) == 0:
            # self.sasa_res = md_traj.shrake_rupley(self.mdtraj) # Cannot calculate on my laptop
            if self.sasa_res.shape == (1,):
                logging.warning(f'No sasa.xvg generated by GROMACS was found in the {self.drug_name} directory, '
                                f'SASA is not calculated')
            
        # Calculate Root Mean Square Fluctuation
        if "RMSF" in metrics or len(metrics) == 0:
            self.rmsf_res = RMSF(self.uni.select_atoms('protein')).run()

        # Perform PCA on the CA atoms
        if "PCA" in metrics or len(metrics) == 0:
            calpha_index = self.mdtraj.topology.select("name CA")
            ca_traj = self.mdtraj.atom_slice(calpha_index)
            xyz = ca_traj.xyz.reshape((ca_traj.xyz.shape[0], -1))    # Create the 3m x n input matrix
            self.pca_res = PCA()
            self.pca_res.fit(xyz)
            self.pca_xyz = xyz    # Save the coordinates on which we fitted PCA
            
        # Calculate the Hydrogen Bonds using wernet_nilsson from MDTraj
        if "Hbonds" in metrics or len(metrics) == 0:
            self.hbonds = md_traj.wernet_nilsson(self.mdtraj, periodic=False)
            
        # Claculate the Salt Bridges
        if "Salt" in metrics or len(metrics) == 0:
            if self.salt_bridges_df.shape == (1,):
                logging.warning('No salts_directory was provided when initializing this object, '
                                'salt bridges are not calculated')
            
    ''' 
    Getters of attributes
        Raises: TypeError when get is attempted on not calculated metric
    '''
    def get_radius_of_gyration(self):
        if self.rg_res is None:
            raise TypeError("Radius of Gyration was not calculated, check AnalysisActor.class docstring for more info")
        else:
            return self.rg_res
        
    def get_rmsf(self):
        if self.rmsf_res is None:
            raise TypeError("RMSF was not calculated, check AnalysisActor.class docstring for more info")
        else:
            return self.rmsf_res
        
    def get_pca(self):
        if self.pca_res is None:
            raise TypeError("PCA was not calculated, check AnalysisActor.class docstring for more info")
        else:
            return self.pca_res
        
    def get_hbonds(self):
        if self.hbonds is None:
            raise TypeError("Hydrogen bonds were not calculated, check AnalysisActor.class docstring for more info")
        else:
            return self.hbonds
    
    def get_salt_bridges_df(self):
        if self.salt_bridges_df.shape == (1,):
            raise TypeError("Salt bridges DataFrame was not calculated, check AnalysisActor.class "
                            "docstring for more info")
        else:
            return self.salt_bridges_df
        
    def get_salt_bridges_over_time(self):
        if self.salt_bridges_over_time[0] == -1:
            raise TypeError("Salt bridges over time were not calculated, check AnalysisActor.class "
                            "docstring for more info")
        else:
            return self.salt_bridges_over_time
        
    def get_sasa(self):
        if self.sasa_res.shape == (1,):
            raise TypeError("SASA was not calculated, check AnalysisActor.class docstring for more info")
        else:
            return self.sasa_res
