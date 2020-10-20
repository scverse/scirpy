doc_working_model = """\

.. note::
    Reading data into *Scirpy* has the following constraints:
     * each cell can have up to four chains (:term:`Dual IR`):
       two :term:`VJ<V(D)J>` and two :term:`VDJ<V(D)J>` chains.
     * Excess chains are removed (those with lowest read count/:term:`UMI` count)
       and cells flagged as :term:`Multichain-cell`.
     * non-productive chains are removed
     * chain loci must be :term:`IGMT locus names<Chain locus>`.

    For more information, see :ref:`receptor-model`.
"""
