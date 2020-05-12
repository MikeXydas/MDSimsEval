.. MD_Feature_Extraction documentation

MD_Feature_Extraction Documentation
=================================================

**MD_Feature_Extraction** is a package I created as part of my undergraduate thesis that in a flexible way calculates useful
metrics from Molecular Dynamics (MD) simulations, stores them and provides a number of methods for useful visualizations.

**This package is use case specific**.The prerequisite of using this package is having **a number
of MD simulations of two classes**. In the case of my thesis I had a number of Agonist and Antagonist
simulations. My goal was to extract known features of MD simulations and evaluate them on their ability
to differentiate an agonist from an antagonist.

If your goal is simple analysis of MD I suggest these free solutions:
  * `MDAnalysis <https://www.mdanalysis.org/>`_
  * `MDtraj <http://mdtraj.org/1.9.3/>`_
  * `GROMACS <http://www.gromacs.org/>`_
  * `VMD <https://www.ks.uiuc.edu/Research/vmd/>`_

The above solutions are actually used for some of my calculations.

.. Contents
.. ========

.. toctree::
   :maxdepth: 4
   :numbered:

   ./pages/Prerequisites
   ./pages/AnalysisActor


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
