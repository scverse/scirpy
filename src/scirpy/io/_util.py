import csv
import os
from collections import Counter

import pandas as pd
from scanpy import logging

doc_working_model = """\

.. note::
    Since scirpy v0.13, there are no restrictions on the AIRR data that can be stored in the scirpy data structure,
    except that each receptor chain needs to be associated with a cell.

    The scirpy :ref:`receptor-model` is now applied in later step using the :func:`~scirpy.pp.index_chains` function.

    For more information, see :ref:`data-structure`.
"""


def get_rearrangement_reader():
    """Defer importing from airr package until it is used, since this is very slow"""
    from airr import RearrangementReader

    return RearrangementReader


def get_rearrangement_schema():
    """Defer importing from airr package until it is used, since this is very slow"""
    from airr import RearrangementSchema

    return RearrangementSchema


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

    class PdRearrangementReader(get_rearrangement_reader()):
        def __init__(self, df, *args, **kwargs):
            super().__init__(os.devnull, *args, **kwargs)
            self.dict_reader = PdDictReader(df)

    return PdRearrangementReader(df, validate=validate, debug=debug)
