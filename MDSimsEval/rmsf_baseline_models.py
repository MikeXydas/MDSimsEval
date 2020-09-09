import collections

from sklearn.manifold import MDS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

from MDSimsEval.rmsf_analysis import reset_rmsf_calculations
from MDSimsEval.rmsf_analysis import get_avg_rmsf_per_residue
from MDSimsEval.rmsf_bootstrapped_analysis import find_rmsf_of_residues
from MDSimsEval.utils import complement_residues

from scipy import stats
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import random


class BaselineClassifier(ABC):
    """
    An abstract class that the baseline classifiers models extend from. The subclasses must implement the abstract
    methods ``fit`` and ``predict``.

    The class provides some helper methods in order to stack the RMSF of the selected residues for training and
    getting the RMSF of the unknown ligand we want to predict its class.

    If you want to extend the method we suggest reading the other classes in this model. This will help in order
    to better understand how ``fit`` and ``predict`` are implemented and how we use the helper methods in this class.

    """
    def __init__(self, start, stop, rmsf_cache, method):
        self.start = start
        self.stop = stop
        self.method = method
        self.rmsf_cache = rmsf_cache
        self.selected_residues = None

    def read_list_stack_rmsfs(self, train_analysis_actors, residues):
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

        return stacked_agonists, stacked_antagonists

    def read_dict_stack_rmsfs(self, train_analysis_actors, residues):
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

        return stacked_agonists, stacked_antagonists

    def read_unknown_list_residues(self, ligand):
        reset_rmsf_calculations({'Agonists': [ligand], 'Antagonists': []}, self.start, self.stop, self.rmsf_cache)

        return get_avg_rmsf_per_residue(ligand)[self.selected_residues]

    def read_unknown_dict_residues(self, ligand):
        rmsf_array = []
        for res, windows in self.selected_residues.items():
            for window in windows:
                rmsf_array.append(
                    find_rmsf_of_residues({'Agonists': [ligand], 'Antagonists': [ligand]},
                                          [res], window[0], window[1], self.rmsf_cache))

        rmsf_array = np.array(rmsf_array).reshape(len(rmsf_array), len(rmsf_array[0])).T[0]

        return rmsf_array

    @abstractmethod
    def fit(self, train_analysis_actors, residues):
        pass

    @abstractmethod
    def predict(self, ligand):
        pass


class AggregatedResidues(BaselineClassifier):
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
        selected_residues (List[boolean]): A list of size total_residues, where True on the indexes of the residue ids
                                           selected

    """
    def __init__(self, start, stop, rmsf_cache, method=np.mean):
        super().__init__(start, stop, rmsf_cache, method)
        self.agonist_baseline = None
        self.antagonist_baseline = None

    def fit(self, train_analysis_actors, residues):
        """
        The function that initializes the aggregated RMSF value for each class.

        Args:
            train_analysis_actors: ``{ "Agonists": List[AnalysisActor.class], "Antagonists": List[AnalysisActor.class] }``
            residues: A list of residue ids that the model will use. For all the residues give ``np.arange(290)`` . Can
                      also be a dictionary with residue ids as keys and values a list of ``[start, stop]``. This
                      argument allows us to use a residue in more than one window. This is used in the residue cherry
                      picking part of the thesis.

                      Example:
                        ::

                            # This will use the window saved as an attribute when we created the model object
                            residues = [10, 11, 27, 52, 83]
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
            stacked_agonists, stacked_antagonists = self.read_list_stack_rmsfs(train_analysis_actors, residues)
        elif isinstance(residues, collections.Mapping):
            stacked_agonists, stacked_antagonists = self.read_dict_stack_rmsfs(train_analysis_actors, residues)
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
        if isinstance(self.selected_residues, list):
            rmsf_value = self.method(self.read_unknown_list_residues(ligand))
        elif isinstance(self.selected_residues, collections.Mapping):
            rmsf_value = self.method(self.read_unknown_dict_residues(ligand))
        else:
            raise ValueError('UNEXPECTED: selected residues is missing or is not of type list or mapping (dictionary)')

        agon_distance = np.abs(self.agonist_baseline - rmsf_value)
        antagon_distance = np.abs(self.antagonist_baseline - rmsf_value)

        if agon_distance < antagon_distance:
            return 1  # Case agonist
        else:
            return 0  # Case antagonist


class ResidueMajority(BaselineClassifier):
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
        super().__init__(start, stop, rmsf_cache, method)
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

                            # This will use the window saved as an attribute when we created the model object
                            residues = [10, 11, 27, 52, 83]
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
            stacked_agonists, stacked_antagonists = self.read_list_stack_rmsfs(train_analysis_actors, residues)
        elif isinstance(residues, collections.Mapping):
            stacked_agonists, stacked_antagonists = self.read_dict_stack_rmsfs(train_analysis_actors, residues)
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
        if isinstance(self.selected_residues, list):
            rmsf_values = self.read_unknown_list_residues(ligand)
        elif isinstance(self.selected_residues, collections.Mapping):
            rmsf_values = self.read_unknown_dict_residues(ligand)
        else:
            raise ValueError('UNEXPECTED: selected residues is missing or is not of type list or mapping (dictionary)')

        agon_distances = np.abs(self.agonist_residue_baseline - rmsf_values)
        antagon_distances = np.abs(self.antagonist_residue_baseline - rmsf_values)

        # Perform the majority voting
        if np.sum(agon_distances < antagon_distances) > len(rmsf_values) / 2:
            return 1  # Case agonist
        else:
            return 0  # Case antagonist


class KSDistance(BaselineClassifier):
    """
    Used for evaluating residue selections based on their RMSF.

    We calculate for each class the average/median RMSF of each residue. Then when we receive an unknown ligand
    we use the `k-s test <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html>`_
    which returns the "distance" of the distributions of the unknown ligand and the class. We calculate the distance
    for the other class too and classify on the class with the smallest distance.

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
        super().__init__(start, stop, rmsf_cache, method)
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

                            # This will use the window saved as an attribute when we created the model object
                            residues = [10, 11, 27, 52, 83]
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
            stacked_agonists, stacked_antagonists = self.read_list_stack_rmsfs(train_analysis_actors, residues)
        elif isinstance(residues, collections.Mapping):
            stacked_agonists, stacked_antagonists = self.read_dict_stack_rmsfs(train_analysis_actors, residues)
        else:
            raise ValueError('residues argument expecting a list or a mapping (dictionary)')

        self.agonist_residue_baseline = self.method(stacked_agonists, axis=0)
        self.antagonist_residue_baseline = self.method(stacked_antagonists, axis=0)

    def predict(self, ligand):
        """
        Performs the K-S test. If we have a mismatch of outcomes (one class accepts it, the other one rejects it) then
        we classify as the one that accepted. Else, we classify using the distance of K-S.

        Args:
            ligand (AnalysisActorClass): The ligand we want to predict its class

        Returns:
            The class label, 1 for Agonist, 0 for Antagonist
        """
        if isinstance(self.selected_residues, list):
            rmsf_values = self.read_unknown_list_residues(ligand)
        elif isinstance(self.selected_residues, collections.Mapping):
            rmsf_values = self.read_unknown_dict_residues(ligand)
        else:
            raise ValueError('UNEXPECTED: selected residues is missing or is not of type list or mapping (dictionary)')

        # We perform the K-S test which returns (distance, p_value) and keep the distance only
        agon_distance, agon_p_value = stats.ks_2samp(self.agonist_residue_baseline, rmsf_values)
        antagon_distances, antagon_p_value = stats.ks_2samp(self.antagonist_residue_baseline, rmsf_values)

        # # See if the test rejects the null hypothesis for one class and rejects it for the other
        # if antagon_p_value <= 0.05 < agon_p_value:
        #     return 1, "A-R"  # Case agonist
        # elif agon_p_value <= 0.05 < antagon_p_value:
        #     return 0, "A-R"  # Case antagonist
        #
        # # Decide if we had a reject - reject or accept - accept
        # if antagon_p_value < 0.05:
        #     ret_str = "R-R"
        # else:
        #     ret_str = "A-A"
        #
        # # Return the class that has the lowest K-S distance
        # if agon_distance < antagon_distances:
        #     return 1, ret_str  # Case agonist
        # else:
        #     return 0, ret_str  # Case antagonist

        # Return the class that has the lowest K-S distance
        if agon_distance < antagon_distances:
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


class MDStoKNN(BaselineClassifier):
    """
    This is a model that classifies ligands based on the K nearest neighbors on a MDS 2D projection.

    We first calculate the pairwise distances of **all** the ligands creating an ``agons_numb`` + ``antagons_numb`` *x*
    ``agons_numb`` + ``antagons_numb`` matrix. We perform a non-linear projection using MDS transforming the matrix to a
    ``agons_numb`` + ``antagons_numb`` *x* 2 shape.

    We provide the indexes of the ligands that the labels are known. These will be considered our train set. We fit a
    KNN model on them. We then provide the index of the ligand we want to predict. For example we may have 20 agonists,
    20 antagonists and we want to predict an unknown ligand. The transformed shape of our 2D projection will be
    20 + 20 + 1 *x* 2. The indexes [0, 1, ..., 39] will for the train set and will be passed on the fit of the KNN. We
    then pass the index 40 in the ``predict`` method in order to predict the unknown ligand.

    .. note::
        This model is more complex than the others which have a straight forward approach, similar to the ``sklearn``
        models. The main idea is that the model knows all the data points a priori in order to create the 2D mapping. We
        then reveal to the model the labels of the known ligands in order to predict the unknown ligands.

    Args:
        start (int): The starting frame of the window
        stop (int): The last frame of the window
        metric: A method used to calculate the pairwise distance of the ligands. Possible metrics are K-S distance and
                Spearman's r
        neigh (KNeighborsClassifier): The KNN model used for predicting the unknown ligands

    """
    def __init__(self, start, stop, rmsf_cache, metric, neighbors):
        super().__init__(start, stop, rmsf_cache, None)
        self.pairwise_distances = None
        self.metric_method = metric
        self.neigh = KNeighborsClassifier(n_neighbors=neighbors)

    def create_pairwise_distances(self, analysis_actors_dict, residues):
        """
        Creates the pairwise distance matrix of all the input ligands that will be projected on a 2D manifold using MDS

        Args:
            analysis_actors_dict: ``{ "Agonists": List[AnalysisActor.class], "Antagonists": List[AnalysisActor.class] }``
            residues: A list of residue ids that the model will use. For all the residues give
                      ``np.arange(290)``.

        Returns:
            A DataFrame of shape ``agons_numb`` + ``antagons_numb`` *x* ``agons_numb`` + ``antagons_numb``

        """
        reset_rmsf_calculations(analysis_actors_dict, self.start, self.stop, self.rmsf_cache)

        # Create a mask of the residues selected
        selected_residues = [which_res in residues for which_res in np.arange(290)]

        ligand_rmsf = collections.OrderedDict()
        for which_ligand in analysis_actors_dict['Agonists'] + analysis_actors_dict['Antagonists']:
            ligand_rmsf[which_ligand.drug_name] = get_avg_rmsf_per_residue(which_ligand)[selected_residues]

        distances_df = pd.DataFrame(np.zeros((len(ligand_rmsf), len(ligand_rmsf))), columns=list(ligand_rmsf),
                                    index=list(ligand_rmsf))

        for ind_ligand, ind_rmsf in ligand_rmsf.items():
            for col_ligand, col_rmsf in ligand_rmsf.items():
                distances_df.at[ind_ligand, col_ligand] = self.metric_method(ind_rmsf, col_rmsf)

        return distances_df

    def fit(self, analysis_actors_dict, residues):
        """
        Create the pairwise distance matrix and perform MDS to transform it to a 2D matrix.

        Args:
            analysis_actors_dict: ``{ "Agonists": List[AnalysisActor.class], "Antagonists": List[AnalysisActor.class] }``
            residues: A list of residue ids that the model will use. For all the residues give
                      ``np.arange(290)``.

        """
        self.pairwise_distances = self.create_pairwise_distances(analysis_actors_dict, residues)

        # Perform the MDS dimensionality reduction
        # mds = MDS(n_components=2, dissimilarity='precomputed')
        mds = PCA(n_components=2)
        self.pairwise_distances = mds.fit_transform(self.pairwise_distances)

    def choose_known_ligands(self, agonist_inds, antagonists_inds):
        """
        Give the indexes of the agonists and antagonists ligands that will be form the train set and fits the KNN
        model using the transformed pairwise distances.

        Args:
            agonist_inds: The indexes of the train agonists
            antagonists_inds: The indexes of the train antagonists

        """
        x_train = np.array(self.pairwise_distances)[agonist_inds + antagonists_inds]
        y_train = np.concatenate([np.ones(len(agonist_inds)), np.zeros(len(antagonists_inds))])

        self.neigh.fit(x_train, y_train)

    def predict(self, ligand_ind):
        """
        Given an index of the unknown ligand predict it using the KNN fitted KNN model

        Args:
             ligand_ind: The index of the unknown ligand

        Returns:
            The class label, 1 for Agonist, 0 for Antagonist

        """
        pred_x = self.pairwise_distances[ligand_ind]

        return self.neigh.predict([pred_x])
