.. Reading data documentation

Reading the simulations
=================================================
Emphasis must be given on reading the trajectory files in an organized and optimal way.

Directory Structure
###################

Our reader function expects a specific directory structure.

::

    input_directory/
      Agonists/
        Ligand1_/
          trajectory_.xtc
          topology_.pdb
          sasa.xvg *
          salts/ *
        Ligand2_
           .
           .
           .
      Antagonists/
         .
         .
         .

On the above structure everything followed by an underscore ``_`` can have a different name.

.. note::
   ``sasa.xvg`` and ``salts/`` are explained `here <aa>`_

As an example this was my actual input directory named ``New_AI_MD``:

.. image:: ../_static/file_structure.png
    :width: 400px
    :align: center
    :height: 450px
    :alt: alternate text

|
|

Reading Function
#################

.. automodule:: AnalysisActor.utils
    :members: