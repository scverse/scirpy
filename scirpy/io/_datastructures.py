"""Datastructures for Adaptive immune receptor (IR) data.

Currently only used as intermediate storage.
See also discussion at https://github.com/theislab/anndata/issues/115
"""

from functools import partial
import itertools
from ..util import _is_na2, _is_true2, _doc_params
from typing import (
    Collection,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Iterator,
    Tuple,
    Any,
)
from airr import RearrangementSchema
import scanpy
import json
import numpy as np
from ._util import doc_working_model
from collections.abc import MutableMapping


class AirrCell(MutableMapping):
    """Data structure for a Cell with immune receptors. Represents one row of `adata.obs`.

    This data structure is compliant with the AIRR rearrangement schema v1.0.
    An AirrCell can hold multiple chains (i.e. rows from the rearrangement TSV)
    which belong to the same cell. A chain is represented as a dictionary, where
    the keys are AIRR-rearrangement fields.

    The AirrCell can, additionally, hold cell-level attributes which can be set
    in a dict-like fashion. Keys marked as "cell-level" via `cell_attribute_fields`
    will be automatically transferred to the cell-level when added through a chain.
    They are required to have the same value for all chains.

    Parameters
    ----------
    cell_id
        cell id or barcode.  Needs to match the cell id used for transcriptomics
        data (i.e. the `adata.obs_names`)
    cell_attribute_fields
        List of field-names which are supposed to be stored at the cell-level
        rather than the chain level. If a chain with these fields is added
        to the cell, they are set on the cell-level instead. If the values already
        exist on the cell-level, a `ValueError` is raised, if they differ from
        the values that are already present.
    logger
        A logger to write messages to. If not specified, use scanpy's default logger.
    """

    #: Identifiers of loci with a :term:`V-J<V(D)J>` junction
    VJ_LOCI = ("TRA", "TRG", "IGK", "IGL")

    #: Identifiers of loci with a :term:`V-D-J<V(D)J>` junction
    VDJ_LOCI = ("TRB", "TRD", "IGH")

    #: Valid chains are IMGT locus names
    #: see https://docs.airr-community.org/en/latest/datarep/rearrangements.html#locus-names
    VALID_LOCI = VJ_LOCI + VDJ_LOCI

    def __init__(
        self,
        cell_id: str,
        cell_attribute_fields: Collection[str] = (),
        *,
        logger: Any = scanpy.logging,
    ):
        self._logger = logger
        self._chain_fields = None
        # A list of fields that are supposed to be stored at the cell level
        # rather than the chain level
        self._cell_attribute_fields = cell_attribute_fields
        # storage for these values, accessible through MutableMapping interface
        self._cell_attrs = dict()
        # A list of AIRR compliant dictionaries
        self._chains = list()
        self["cell_id"] = cell_id

    def __repr__(self):
        return f"AirrCell {self.cell_id} with {len(self.chains)} chains"

    @property
    def cell_id(self) -> str:
        """Unique identifier (barcode) of the cell."""
        return self["cell_id"]

    @property
    def chains(self) -> List[dict]:
        """List of chain-dictionaries added to the cell."""
        return self._chains

    @property
    def fields(self) -> List[str]:
        """Return a list of all fields (chain-level and cell-level)"""
        if self._chain_fields is None:
            return list(self)
        return list(self) + self._chain_fields

    def __delitem__(self, key) -> None:
        del self._cell_attrs[key]

    def __getitem__(self, key):
        return self._cell_attrs[key]

    def __iter__(self) -> Iterator:
        return iter(self._cell_attrs)

    def __len__(self) -> int:
        return len(self._cell_attrs)

    def __setitem__(self, k, v) -> None:
        try:
            existing_value = self._cell_attrs[k]
            if existing_value != v and not _is_na2(existing_value):
                raise ValueError(
                    "Cell-level attributes differ between different chains. "
                    f"Already present: `{existing_value}`. Tried to add `{v}`."
                )
        except KeyError:
            self._cell_attrs[k] = v

    def add_chain(self, chain: Mapping) -> None:
        """Add a chain to the cell.

        A chain is a dictionary following
        the `AIRR Rearrangement Schema <https://docs.airr-community.org/en/latest/datarep/rearrangements.html#productive>`__.
        """
        # ensure consistent ordering
        chain = dict(sorted(chain.items()))
        # sanitize NA values
        chain = {k: None if _is_na2(v) else v for k, v in chain.items()}

        # TODO this should be `.validate_obj` but currently does not work
        # because of https://github.com/airr-community/airr-standards/issues/508
        RearrangementSchema.validate_header(chain.keys())
        RearrangementSchema.validate_row(chain)

        for tmp_field in self._cell_attribute_fields:
            # It is ok if a field specified as cell attribute is not present in the chain
            try:
                self[tmp_field] = chain.pop(tmp_field)
            except KeyError:
                pass

        if self._chain_fields is None:
            self._chain_fields = list(chain.keys())
        elif self._chain_fields != list(chain.keys()):
            raise ValueError("All chains must have the same fields!")

        if "locus" not in chain:
            self._logger.warning(
                "`locus` field not specified, but required for most scirpy functionality. "
            )  # type: ignore
        elif chain["locus"] not in self.VALID_LOCI:
            self._logger.warning(f"Non-standard locus name: {chain['locus']} ")  # type: ignore

        self.chains.append(chain)

    def add_serialized_chains(self, serialized_chains: str) -> None:
        """Add chains serialized as JSON.

        The JSON object needs to be a list of dicts. If `serialized_chains` is
        a value interpreted as NA, the function passes silently and does nothing."""
        if not _is_na2(serialized_chains):
            tmp_chains = json.loads(serialized_chains)
            for chain in tmp_chains:
                tmp_chain = AirrCell.empty_chain_dict()
                tmp_chain.update(chain)
                self.add_chain(tmp_chain)

    def chain_indices(self) -> Dict:
        """
        Returns the indices of the primary and secondary VJ and VDJ chains.

         * only productive chains are considered
         * chains are ranked by their expression
        """
        chain_indices = {"VJ": list(), "VDJ": list()}
        for i, tmp_chain in enumerate(self.chains):
            if "locus" not in tmp_chain:
                continue
            if (
                tmp_chain["locus"] in self.VJ_LOCI
                and tmp_chain["productive"]
                and not _is_na2(tmp_chain["junction_aa"])
            ):
                chain_indices["VJ"].append(i)
            elif (
                tmp_chain["locus"] in self.VDJ_LOCI
                and tmp_chain["productive"]
                and not _is_na2(tmp_chain["junction_aa"])
            ):
                chain_indices["VDJ"].append(i)

        if (
            "duplicate_count" not in self.fields
            and "consensus_count" not in self.fields
            and len(self.chains)  # don't warn for empty cells
        ):
            self._logger.warning(
                "No expression information available. Cannot rank chains by expression. "
            )  # type: ignore

        for junction_type in ["VJ", "VDJ"]:
            chain_indices[junction_type] = sorted(
                chain_indices[junction_type],
                key=partial(self._key_sort_chains, self.chains),
                reverse=True,
            )
            # only keep the (up to) two most highly expressed chains
            chain_indices[junction_type] = chain_indices[junction_type][:2]

        res_dict = {}
        for i, junction_type in itertools.product([0, 1], ["VJ", "VDJ"]):
            res_key = f"{junction_type}_{i+1}"
            try:
                res_dict[res_key] = chain_indices[junction_type][i]
            except IndexError:
                res_dict[res_key] = None

        return res_dict

    @staticmethod
    def _key_sort_chains(chains, idx) -> Tuple:
        """Get key to sort chains by expression. Idx is the index of a chain in `chains`"""
        chain = chains[idx]
        sort_tuple = (
            chain.get("duplicate_count", 0),
            chain.get("consensus_count", 0),
            chain.get("junction", ""),
            chain.get("junction_aa", ""),
        )
        # replace None by -1 to make sure it comes in last
        return tuple(-1 if x is None else x for x in sort_tuple)

    def to_airr_records(self) -> Iterable[dict]:
        """Iterate over chains as AIRR-Rearrangent compliant dictonaries.
        Each dictionary will also include the cell-level information.

        Yields
        ------
        Dictionary representing one row of a AIRR rearrangement table
        """
        for tmp_chain in self.chains:
            chain = AirrCell.empty_chain_dict()
            # add the actual data
            chain.update(tmp_chain)
            # add cell-level attributes
            chain.update(self)
            yield chain

    @staticmethod
    def empty_chain_dict() -> dict:
        """Generate an empty chain dictionary, containing all required AIRR
        columns, but set to `None`"""
        return {field: None for field in RearrangementSchema.required}
