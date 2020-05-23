.. Prerequisites

Prerequisites
=================================================

Installation
############

| **OS**: Developed on Ubuntu18.04 but any linux distribution should work
| **Python**: Developed on 3.6 but any version >=3.6 should work

**Package Only**

The steps for **installing the package only** are:

1. Clone the repository
2. ``cd MDSimsEval``
3. ``pip install .``

Then you can import any method or class
::

    from MDSimsEval.utils import create_analysis_actor_dict  # Example import

.. note::
   **Optional**: The ``rmsf_analysis.corr_matrix`` function in order to provide as ouput a .png requires ``wkhtmltopdf``
   to be installed via ``sudo apt-get install wkhtmltopdf``. More on the `imgkit package <https://github.com/kamalkraj/imgkit>`_.
   If not installed the output will be an ``.html`` file which can be opened with any browser.

**Development Environment**

If you are looking for adding functionality or changing the code you will need more packages (eg for the docs).

1. Clone the repository
2. Install the ``requirements.txt``

 - Conda Env: ``conda install --file requirements.txt``
 - VirtualEnv: ``pip install -r requirements.txt``

3. Copy the `MDSimsEval` directory to your project

Data
####

| As you may have read in the homepage this is not a package for analysis of a single MD simulation. All the
  methods need a collection of MD simulations preferably of two classes. The goal of the package is to provide
  insight of which features differentiate these two classes.
| Also the simulations must have the **same number of frames and preferably performed on the same protein**

| To make it more clear I will present my use case as an example:
| I was provided with a number of 2500 frames simulations on the `5HT2A receptor <https://en.wikipedia.org/wiki/5-HT2A_receptor>`_
 using different ligands. N of the ligands were agonists and M were antagonists. My goal was to find which features
 help us differentiate an agonist from an antagonist.

The features I analyzed are:
 - Radius of Gyration
 - Solvent-Accessible Surface Area
 - Root Mean Square Flactuation
 - PCA Loadings and 2D Projections

**Continue with** :doc:`reading the data <ReadingData>`.
