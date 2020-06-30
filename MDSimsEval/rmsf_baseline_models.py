import collections

from MDSimsEval.rmsf_analysis import reset_rmsf_calculations
from MDSimsEval.rmsf_analysis import get_avg_rmsf_per_residue
from MDSimsEval.rmsf_bootstrapped_analysis import find_rmsf_of_residues
from MDSimsEval.utils import complement_residues

import numpy as np
import random


class BaselineClassifierResidueMajority:
    """
    Used for evaluating residue selections based on their RMSF.

    This is a simple baseline model able to quantify how good our residue selection is.
    Given a k - k training set of agonists - antagonists,for an unknown ligand
    we iterate through the residues. If the residue is closer to the median/average of the
    training agonists (of the RMSF values of the specific residue) then the residue votes that
    the ligand is an agonist. Else it votes that the ligand is an antagonist.
    |
    | At then end we see which class had the most votes.

    Attributes:
        start (int): The starting frame of the window
        stop (int): The last frame of the window
        method (func): Function that we will use for summarizing each residue, eg ``np.mean``, ``np.median``
        agonist_residue_baseline (List[float]): A list of the aggregated RMSF value of the agonists of each residue
        antagonist_residue_baseline (List[float]): A list of the aggregated RMSF value of the antagonists of each
                                                    residue
        selected_residues (List[boolean]): A list of size total_residues, where True on the indexes of the residue ids
                                            selected

    """

    def __init__(self, start, stop, rmsf_cache, method=np.mean):
        self.start = start
        self.stop = stop
        self.method = method
        self.rmsf_cache = rmsf_cache
        self.selected_residues = None
        self.agonist_residue_baseline = None
        self.antagonist_residue_baseline = None

    def fit(self, train_analysis_actors, residues):
        """
        The function that initializes the aggregated RMSF value for each residue for each class.

        Args:
            train_analysis_actors: ``{ "Agonists": List[AnalysisActor.class], "Antagonists": List[AnalysisActor.class] }``
            residues: A list of residue ids that the model will use. For all the residues give
                      ``np.arange(290)`` . Can also be a dictionary with residue ids as keys and values a
                      list of ``[start, stop]``. This argument allows us to use a residue in more than
                      one window. This is used in the residue cherry picking part of the thesis.

                      Example:
                        ::

                            residues = [10, 11, 27, 52, 83]  # This will use the window saved as an attribute when we created the model object
                            or
                            residues = {
                                         115: [[0, 500], [2000, 2500]],
                                         117: [[2000, 2500]],
                                         81: [[2000, 2500]],
                                         78: [[1000, 1500], [1500, 2000]],
                                         254: [[0, 500], [1500, 2000]],
                                      }

        """
        if isinstance(residues, list):
            reset_rmsf_calculations(train_analysis_actors, self.start, self.stop, self.rmsf_cache)

            # Create a mask of the residues selected
            self.selected_residues = [which_res in residues for which_res in np.arange(290)]

            # Calculating baseline agonist RMSF value
            stacked_agonists = get_avg_rmsf_per_residue(train_analysis_actors['Agonists'][0])[self.selected_residues]
            for which_ligand in train_analysis_actors['Agonists'][1:]:
                stacked_agonists = np.vstack(
                    (stacked_agonists, get_avg_rmsf_per_residue(which_ligand)[self.selected_residues]))

            # Calculating baseline antagonist RMSF value
            stacked_antagonists = get_avg_rmsf_per_residue(train_analysis_actors['Antagonists'][0])[
                self.selected_residues]
            for which_ligand in train_analysis_actors['Antagonists'][1:]:
                stacked_antagonists = np.vstack(
                    (stacked_antagonists, get_avg_rmsf_per_residue(which_ligand)[self.selected_residues]))

        elif isinstance(residues, collections.Mapping):
            # Case that we are given a dictionary of residue_ids: [[start1, stop1], [start2, stop2]]
            self.selected_residues = residues
            rmsf_array = []
            for res, windows in residues.items():
                for window in windows:
                    rmsf_array.append(
                        find_rmsf_of_residues(train_analysis_actors, [res], window[0], window[1], self.rmsf_cache))

            rmsf_array = np.array(rmsf_array).reshape(len(rmsf_array), len(rmsf_array[0])).T

            stacked_agonists = rmsf_array[:len(train_analysis_actors['Agonists']), :]
            stacked_antagonists = rmsf_array[len(train_analysis_actors['Agonists']):, :]
        else:
            raise ValueError('residues argument expecting a list or a mapping (dictionary)')

        self.agonist_residue_baseline = self.method(stacked_agonists, axis=0)
        self.antagonist_residue_baseline = self.method(stacked_antagonists, axis=0)

    def predict(self, ligand):
        """
        Performs the majority voting and returns the predicted class.

        Args:
            ligand (AnalysisActorClass): The ligand we want to predict its class

        Returns:
            The class label, 1 for Agonist, 0 for Antagonist.

        """
        # We do a trick and create a Dict of ligands so as to use reset_rmsf_calculations
        reset_rmsf_calculations({'Agonists': [ligand], 'Antagonists': []}, self.start, self.stop, self.rmsf_cache)

        if isinstance(self.selected_residues, list):
            rmsf_values = get_avg_rmsf_per_residue(ligand)[self.selected_residues]
        elif isinstance(self.selected_residues, collections.Mapping):
            rmsf_array = []
            for res, windows in self.selected_residues.items():
                for window in windows:
                    rmsf_array.append(
                        find_rmsf_of_residues({'Agonists': [ligand], 'Antagonists': [ligand]},
                                              [res], window[0], window[1], self.rmsf_cache))

            rmsf_values = np.array(rmsf_array).reshape(len(rmsf_array), len(rmsf_array[0])).T[0]
        else:
            raise ValueError('UNEXPECTED: selected residues is missing or is not of type list or mapping (dictionary)')

        agon_distances = np.abs(self.agonist_residue_baseline - rmsf_values)
        antagon_distances = np.abs(self.antagonist_residue_baseline - rmsf_values)

        # Perform the majority voting
        if np.sum(agon_distances < antagon_distances) > len(rmsf_values) / 2:
            return 1  # Case agonist
        else:
            return 0  # Case antagonist


class BaselineClassifierAggregatedResidues:
    """
    This simple model fits on the training data by calculating the average RMSF value of the agonist and the
    antagonist sets. The RMSF is calculated on the residues given and aggregated to one value. This means that for
    the agonists we calculate the average/median value for each residue and then we aggregate again ending with a
    scalar value.

    Attributes:
        start (int): The starting frame of the window
        stop (int): The last frame of the window
        method (func): Function that we will use for summarizing each residue, eg `np.mean`, `np.median`
        agonist_baseline (float): The aggregated RMSF value of the agonist class
        antagonist_baseline (float): The aggregated RMSF value of the antagonist class
        selected_residues (List[boolean]): A list of size total_residues, where True on the indexes of the residue ids selected

    """

    def __init__(self, start, stop, rmsf_cache, method=np.mean):
        self.start = start
        self.stop = stop
        self.method = method
        self.rmsf_cache = rmsf_cache
        self.selected_residues = None
        self.agonist_baseline = None
        self.antagonist_baseline = None

    def fit(self, train_analysis_actors, residues):
        """
        The function that initializes the aggregated RMSF value for each class.

        Args:
            train_analysis_actors: ``{ "Agonists": List[AnalysisActor.class], "Antagonists": List[AnalysisActor.class] }``
            residues: A list of residue ids that the model will use. For all the residues give ``np.arange(290)`` . Can
                      also be a dictionary with residue ids as keys and values a list of ``[start, stop]``. This argument
                      allows us to use a residue in more than one window. This is used in the residue cherry
                      picking part of the thesis.

                      Example:
                        ::

                            residues = [10, 11, 27, 52, 83]  # This will use the window saved as an attribute when we created the model object
                            or
                            residues = {
                                         115: [[0, 500], [2000, 2500]],
                                         117: [[2000, 2500]],
                                         81: [[2000, 2500]],
                                         78: [[1000, 1500], [1500, 2000]],
                                         254: [[0, 500], [1500, 2000]],
                                      }

        """
        if isinstance(residues, list):
            reset_rmsf_calculations(train_analysis_actors, self.start, self.stop, self.rmsf_cache)

            # Create a mask of the residues selected
            self.selected_residues = [which_res in residues for which_res in np.arange(290)]

            # Calculating baseline agonist RMSF value
            stacked_agonists = get_avg_rmsf_per_residue(train_analysis_actors['Agonists'][0])[self.selected_residues]
            for which_ligand in train_analysis_actors['Agonists'][1:]:
                stacked_agonists = np.vstack(
                    (stacked_agonists, get_avg_rmsf_per_residue(which_ligand)[self.selected_residues]))

            # Calculating baseline antagonist RMSF value
            stacked_antagonists = get_avg_rmsf_per_residue(train_analysis_actors['Antagonists'][0])[
                self.selected_residues]
            for which_ligand in train_analysis_actors['Antagonists'][1:]:
                stacked_antagonists = np.vstack(
                    (stacked_antagonists, get_avg_rmsf_per_residue(which_ligand)[self.selected_residues]))

        elif isinstance(residues, collections.Mapping):
            # Case that we are given a dictionary of residue_ids: [[start1, stop1], [start2, stop2]]
            self.selected_residues = residues
            rmsf_array = []
            for res, windows in residues.items():
                for window in windows:
                    rmsf_array.append(
                        find_rmsf_of_residues(train_analysis_actors, [res], window[0], window[1], self.rmsf_cache))

            rmsf_array = np.array(rmsf_array).reshape(len(rmsf_array), len(rmsf_array[0])).T

            stacked_agonists = rmsf_array[:len(train_analysis_actors['Agonists']), :]
            stacked_antagonists = rmsf_array[len(train_analysis_actors['Agonists']):, :]
        else:
            raise ValueError('residues argument expecting a list or a mapping (dictionary)')

        self.agonist_baseline = self.method(stacked_agonists)
        self.antagonist_baseline = self.method(stacked_antagonists)

    def predict(self, ligand):
        """
        Checks the distance of the unknown ligand from the agonist and antagonist averages and returns
        as a label the class that is closest.

        Args:
            ligand (AnalysisActorClass): The ligand we want to predict its class

        Returns:
            The class label, 1 for Agonist, 0 for Antagonist.

        """
        # We do a trick and create a Dict of ligands so as to use reset_rmsf_calculations
        if isinstance(self.selected_residues, list):
            reset_rmsf_calculations({'Agonists': [ligand], 'Antagonists': []}, self.start, self.stop, self.rmsf_cache)

            rmsf_value = self.method(get_avg_rmsf_per_residue(ligand)[self.selected_residues])

        elif isinstance(self.selected_residues, collections.Mapping):
            rmsf_array = []
            for res, windows in self.selected_residues.items():
                for window in windows:
                    rmsf_array.append(
                        find_rmsf_of_residues({'Agonists': [ligand], 'Antagonists': [ligand]},
                                              [res], window[0], window[1], self.rmsf_cache))

            rmsf_array = np.array(rmsf_array).reshape(len(rmsf_array), len(rmsf_array[0])).T[0]
            rmsf_value = self.method(rmsf_array)
        else:
            raise ValueError('UNEXPECTED: selected residues is missing or is not of type list or mapping (dictionary)')

        agon_distance = np.abs(self.agonist_baseline - rmsf_value)
        antagon_distance = np.abs(self.antagonist_baseline - rmsf_value)

        # Perform the majority voting
        if agon_distance < antagon_distance:
            return 1  # Case agonist
        else:
            return 0  # Case antagonist


def bootstrap_dataset(analysis_actors_dict, samples, sample_size):
    """
    Creates a given number of bootstrapped samples of the Agonist - Antagonist dataset.
    | Also the remaining ligands are returned as a validation set.
    | Eg if sample_size = 20 and on each class we have 27 ligands, then we create
    a dict of 20 - 20 unique ligands and the remaining 7 ligands are returned as a validation set.

    Args:
        analysis_actors_dict: ``{ "Agonists": List[AnalysisActor.class], "Antagonists": List[AnalysisActor.class] }``
        samples (int): Number of bootstrapped samples generated
        sample_size (int): How many ligands of each class the training set will have

    Returns:
        A tuple of (``train_dicts``, ``test_dicts``)

    """
    train_agonists_samples = [random.sample(analysis_actors_dict['Agonists'], k=sample_size)
                              for i in np.arange(samples)]
    test_agonists_samples = [complement_residues(analysis_actors_dict['Agonists'], selected_ligands)
                             for selected_ligands in train_agonists_samples]

    train_antagonists_samples = [random.sample(analysis_actors_dict['Antagonists'], k=sample_size)
                                 for i in np.arange(samples)]
    test_antagonists_samples = [complement_residues(analysis_actors_dict['Antagonists'], selected_ligands)
                                for selected_ligands in train_antagonists_samples]

    # Merge the agonists and the antagonists to a dictionary which is the expected input
    # of classify_on_baseline method above
    train_actor_dicts = [{'Agonists': agonist_sample, 'Antagonists': antagonist_sample}
                         for agonist_sample, antagonist_sample in
                         zip(train_agonists_samples, train_antagonists_samples)]

    test_actor_dicts = [{'Agonists': agonist_sample, 'Antagonists': antagonist_sample}
                        for agonist_sample, antagonist_sample in zip(test_agonists_samples, test_antagonists_samples)]

    return train_actor_dicts, test_actor_dicts