.. _tcr-model:

T-cell receptor (TCR) model
===========================

Clonotype definition
^^^^^^^^^^^^^^^^^^^^
Scirpy implements a network-based clonotype definition that enables clustering cells
into clonotypes based on the following options:

- identical CDR3 nucleotide sequences;
- identical CDR3 amino acid sequences;
- similar CDR3 amino acid sequences based on pairwise sequence alignment.

The latter approach is inspired by studies showing that similar TCR sequences also 
share epitope targets (:cite:`Fischer2019`, :cite:`Glanville2017-ay`, :cite:`TCRdist`).
Based on these approaches, Scirpy constructs a global, epitope-focused cell similarity
network. While convergence of the nucleotide-based clonotype definition to the amino 
acid based one hints at selection pressure, sequence alignment based networks offer
the opportunity to identify cells that might recognize the same epitopes.





