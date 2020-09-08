# MD Feature Extraction and Evaluation

**MDSimsEval** is a package we created as part of my undergraduate thesis that in a flexible way calculates useful
metrics from a collection of Molecular Dynamics (MD) simulations, stores them and provides a number of methods for
analysis and classification.  
  
More specifically the use case we developped this package for was to define and evaluate models used 
for discriminating agonist vs. antagonist ligands of the 5-HT2A receptor.


[More can be found on the docs](https://mikexydas.github.io/MDSimsEval/).

## Thesis Abstract

Molecular dynamics (MD) is a computer simulation method for analyzing the physical
movements of atoms and molecules. The atoms and molecules are allowed to interact
for a fixed period of time, giving a view of the dynamic &quot;evolution&quot; of the system. Then
the output of these simulations is analyzed in order to arrive to conclusions depending
on the use case.

In our use case we were provided with several simulations between ligands and the 5-HT2A 
receptor which is the main excitatory receptor subtype among the G protein-
coupled receptor (GPCRs) for serotonin and a target for many antipsychotic drugs. Our
simulations were of two classes. Some of the ligands were agonists meaning that they
activated the receptor, while the other were antagonists meaning that they blocked the
activation of the receptor.

Our goal was to find a set of features that was able to discriminate agonists from
antagonists with a degree of certainty. The small discriminative power of the currently
well-known descriptors of the simulations motivated us to dig deeper and extract a
custom-made feature set. We accomplished that by defining a method which is able to
find in a robust way, the residues of the receptor that had the most statistically
significant separability between the two classes.
