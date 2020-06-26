.. MDSimsEval documentation

MDSimsEval Documentation
=================================================

**The main goal of this package is to create a way of summarizing, describing and evaluating collections of
MD simulations of two different classes**

**MDSimsEval** is a package I created as part of my undergraduate thesis that in a flexible way calculates useful
metrics for **a collection of Molecular Dynamics (MD) Simulations**, stores them and provides a number of methods
for evaluating their discriminative ability.

**This package is use case specific**.The prerequisite of using this package is having a number
of MD simulations of two classes. In the case of my thesis I had a number of Agonist and Antagonist
simulations on the 5-HT2A receptor. My goal was to extract known features of MD simulations and evaluate them on their ability
to differentiate an agonist from an antagonist.

If your goal is simple analysis of MD I suggest these free solutions:
  * `MDAnalysis <https://www.mdanalysis.org/>`_
  * `MDtraj <http://mdtraj.org/1.9.3/>`_
  * `GROMACS <http://www.gromacs.org/>`_
  * `VMD <https://www.ks.uiuc.edu/Research/vmd/>`_

The above solutions are actually used for some of my calculations.

**In order to get started you should read** :doc:`Prerequisites <pages/Prerequisites>`,
:doc:`Reading <pages/ReadingData>`, :doc:`AnalysisActor <pages/AnalysisActor>` in this order.

.. Contents
.. ========

.. toctree::
   :maxdepth: 4
   :numbered:
   :hidden:

   ./pages/Prerequisites
   ./pages/ReadingData
   ./pages/AnalysisActor
   ./pages/RgSasaAnalysis
   ./pages/RmsfAnalysis
   ./pages/RMSFBaselineModels
   ./pages/PcaAnalysis
   ./pages/Utils
   ./pages/Examples


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
