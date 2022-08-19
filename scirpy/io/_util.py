from collections import Counter
from scanpy import logging
import pandas as pd
import csv
import os
from airr import RearrangementReader

doc_working_model = """\

.. note::
    Reading data into *Scirpy* has the following constraints:
     * Each cell can have up to four productive chains chains (:term:`Dual IR`):
       two :term:`VJ<V(D)J>` and two :term:`VDJ<V(D)J>` chains.
     * Excess chains are ignored (those with lowest read count/:term:`UMI` count)
       and cells flagged as :term:`Multichain-cell`.
     * Non-productive chains are ignored.
     * Chain loci must be valid :term:`IMGT locus names<Chain locus>`.
     * Excess chains, non-productive chains, chains without a CDR3 sequence,
       or chains with invalid loci are serialized to JSON and stored in the
       `extra_chains` column. They are not used by scirpy except when exporting
       the `AnnData` object to AIRR format.

    For more information, see :ref:`receptor-model`.
"""


class _IOLogger:
    """Logger wrapper that prints identical messages only once"""

    def __init__(self):
        self._warnings = Counter()

    def warning(self, message):
        if not self._warnings[message]:
            logging.warning(message)  # type: ignore

        self._warnings[message] += 1


def _read_airr_rearrangement_df(df: pd.DataFrame, validate=False, debug=False):
    """Like airr.read_rearrangement, but from a data frame instead of a tsv file.

    Provides RearrangementReader with an alternative iterator to its csv.DictReader
    """

    class PdDictReader(csv.DictReader):
        def __init__(self, df, *args, **kwargs):
            super().__init__(os.devnull)
            self.df = df
            self.reader = iter(df.to_dict(orient="records"))

        @property
        def fieldnames(self):
            return self.df.columns.tolist()

        def __next__(self):
            return next(self.reader)

    class PdRearrangementReader(RearrangementReader):
        def __init__(self, df, *args, **kwargs):
            super().__init__(os.devnull, *args, **kwargs)
            self.dict_reader = PdDictReader(df)

    return PdRearrangementReader(df, validate=validate, debug=debug)
