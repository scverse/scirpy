import csv
import os
from collections import Counter
from typing import Any

import pandas as pd
from scanpy import logging

from scirpy.util import _is_na2

doc_working_model = """\

.. note::
    Since scirpy v0.13, there are no restrictions on the AIRR data that can be stored in the scirpy data structure,
    except that each receptor chain needs to be associated with a cell.

    The scirpy :ref:`receptor-model` is now applied in later step using the :func:`~scirpy.pp.index_chains` function.

    For more information, see :ref:`data-structure`.
"""

doc_airr_fields = """\
Even though data without these fields can be imported, the following columns are required by scirpy
for a meaningful analysis:

    * `cell_id`
    * `productive`
    * `locus` containing a valid IMGT locus name
    * at least one of `consensus_count`, `duplicate_count`, or `umi_count`
    * at least one of `junction_aa` or `junction`.
"""


def get_rearrangement_reader():
    """Defer importing from airr package until it is used, since this is very slow"""
    from airr import RearrangementReader

    return RearrangementReader


def get_rearrangement_schema():
    """Defer importing from airr package until it is used, since this is very slow"""
    from airr import RearrangementSchema

    return RearrangementSchema


def _sanitize_airr_value(field: str, value: Any) -> Any:
    """Sanitize a single value of an AIRR rearrangement record.

    Text representations of missing values (e.g. `"nan"`, `"None"`, `""`) are converted to `None` and
    values of typed AIRR fields (e.g. `productive`, `umi_count`) that are still strings are cast to
    their native Python type (e.g. `"True"` -> `True`, `"3"` -> `3`).

    This is the only place where scirpy deals with such text representations. Everything downstream
    can rely on `adata.obsm["airr"]` having consistent data types.

    Parameters
    ----------
    field
        Name of the AIRR rearrangement field. Fields that are not part of the AIRR rearrangement
        schema are only checked for missing values, since their type is unknown.
    value
        The value to sanitize

    Returns
    -------
    The sanitized value.
    """
    if _is_na2(value):
        return None
    if not isinstance(value, str):
        # only strings need to be cast -- everything else is already a native Python (or numpy) type.
        return value
    schema = get_rearrangement_schema()
    converter = {"boolean": schema.to_bool, "integer": schema.to_int, "number": schema.to_float}.get(schema.type(field))
    if converter is None:
        # the field is not part of the schema, or is of a type that doesn't need casting (e.g. `string`)
        return value
    # `validate=True` raises a `ValidationError` for values that cannot be cast. This is consistent
    # with `validate_row`, which is called on the full record in `AirrCell.add_chain`.
    return converter(value, validate=True)


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
