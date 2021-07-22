"""Datastructures for Adaptive immune receptor (IR) data.

Currently only used as intermediate storage.
See also discussion at https://github.com/theislab/anndata/issues/115
"""

from ..util import _is_na2, _is_true2, _doc_params
from typing import Collection, Dict, Iterable, List, Mapping, Optional, Iterator, Tuple
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
        logger=scanpy.logging,
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
        # legacy argument for the old AnnData scheme (when there was no `extra_chains` field)
        self["multi_chain"] = False

    def __repr__(self):
        return "AirrCell {} with {} chains".format(self.cell_id, len(self.chains))

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
        if k == "multi_chain":
            v = _is_true2(v)
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
            # TODO seems this isn't actually ignored. Chain will just be moved to `extra chains`.
            self._logger.warning(f"Non-standard locus name ignored: {chain['locus']} ")  # type: ignore

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

    def _split_chains(self) -> Tuple[bool, dict]:
        """
        Splits the chains into productive VJ, productive VDJ, and extra chains.

        Returns
        -------
        is_multichain
            Boolean that indicates if the current cell is a multichain
        split_chains
            dictionary with the following entries:
              * `vj_chains`: The (up to) two most highly expressed, productive VJ-chains
              * `vdj_chains`: The (up to) two most highly expressed, productive VDJ chains
              * `extra_chains`: All remaining chains
        """
        is_multichain = self["multi_chain"]
        split_chains = {"VJ": list(), "VDJ": list(), "extra": list()}
        for tmp_chain in self.chains:
            if "locus" not in tmp_chain:
                split_chains["extra"].append(tmp_chain)
            elif (
                tmp_chain["locus"] in self.VJ_LOCI
                and tmp_chain["productive"]
                and not _is_na2(tmp_chain["junction_aa"])
            ):
                split_chains["VJ"].append(tmp_chain)
            elif (
                tmp_chain["locus"] in self.VDJ_LOCI
                and tmp_chain["productive"]
                and not _is_na2(tmp_chain["junction_aa"])
            ):
                split_chains["VDJ"].append(tmp_chain)
            else:
                split_chains["extra"].append(tmp_chain)

        if (
            "duplicate_count" not in self.fields
            and "consensus_count" not in self.fields
            and len(self.chains)  # don't warn for empty cells
        ):
            self._logger.warning(
                "No expression information available. Cannot rank chains by expression. "
            )  # type: ignore

        for junction_type in ["VJ", "VDJ"]:
            split_chains[junction_type] = sorted(
                split_chains[junction_type], key=self._key_sort_chains, reverse=True
            )
            # only keep the (up to) two most highly expressed chains
            tmp_extra_chains = split_chains[junction_type][2:]
            # if productive chains are appended to extra chains, it's a multichain cell!
            is_multichain = is_multichain or len(tmp_extra_chains)
            split_chains["extra"].extend(tmp_extra_chains)
            split_chains[junction_type] = split_chains[junction_type][:2]

        return bool(is_multichain), split_chains

    @staticmethod
    def _key_sort_chains(chain) -> Tuple:
        """Get key to sort chains by expression"""
        sort_tuple = (
            chain.get("duplicate_count", 0),
            chain.get("consensus_count", 0),
            chain.get("junction", ""),
            chain.get("junction_aa", ""),
        )
        # replace None by -1 to make sure it comes in last
        return tuple(-1 if x is None else x for x in sort_tuple)

    @staticmethod
    def _serialize_chains(
        chains: List[MutableMapping], include_fields: Optional[Collection[str]] = None
    ) -> str:
        """Serialize chains into a JSON object. This is useful for storing
        an arbitrary number of extra chains in a single column of a dataframe."""
        # convert numpy dtypes to python types
        # https://stackoverflow.com/questions/9452775/converting-numpy-dtypes-to-native-python-types
        for chain in chains:
            for k, v in chain.items():
                try:
                    chain[k] = chain[k].item()
                except AttributeError:
                    pass

        # Filter chains for `include_fields`
        chains_filtered = [
            {k: v for k, v in chain.items() if k in include_fields} for chain in chains
        ]

        return json.dumps(chains_filtered)

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

    @_doc_params(doc_working_model=doc_working_model)
    def to_scirpy_record(
        self, include_fields: Optional[Collection[str]] = None
    ) -> dict:
        """\
        Convert the cell to a scirpy record (i.e. one row of `adata.obs`) according
        to our working model of adaptive immune receptors.

        {doc_working_model}

        Parameters
        ----------
        include_fields
            AIRR fields to include into `adata.obs` (to save space and to not clutter
            `obs`). Set to `None` to include all fields.

        Returns
        -------
        Dictionary representing one row of scirpy's `adata.obs`.
        """
        res_dict = dict()

        if include_fields is None:
            include_fields = self.fields
        # ensure cell_id is always added
        include_fields = set(include_fields)
        include_fields.add("cell_id")

        res_dict["multi_chain"], chain_dict = self._split_chains()
        res_dict["extra_chains"] = self._serialize_chains(
            chain_dict.pop("extra"), include_fields=include_fields
        )

        # add cell-level attributes
        for key in self:
            if key in include_fields:
                res_dict[key] = self[key]

        # add chain-level attributes, do nothing when cell with no chains
        if self._chain_fields is not None:
            for key in self._chain_fields:
                if key in include_fields:
                    for junction_type, tmp_chains in chain_dict.items():
                        for i in range(2):
                            try:
                                tmp_chain = tmp_chains[i]
                            except IndexError:
                                tmp_chain = dict()
                            res_dict[
                                "IR_{}_{}_{}".format(junction_type, i + 1, key)
                            ] = tmp_chain.get(key, None)

        # use this check instead of `is None`, as the fields are missing in an empty cell.
        def _is_nan_or_missing(col):
            return col not in res_dict or res_dict[col] is None

        # in some weird reasons, it can happen that a cell has been called from
        # TCR-seq but no TCR seqs have been found. `has_ir` should be equal
        # to "at least one productive chain"
        res_dict["has_ir"] = not (
            _is_nan_or_missing("IR_VJ_1_junction_aa")
            and _is_nan_or_missing("IR_VDJ_1_junction_aa")
        )

        # if there are no chains at all, we want multi-chain to be nan
        # This is to be consistent with what happens when turning an anndata object into
        # airr_cells and converting it back to anndata.
        if not len(self.chains):
            res_dict["multi_chain"] = np.nan

        if _is_nan_or_missing("IR_VJ_1_junction_aa"):
            assert _is_nan_or_missing(
                "IR_VJ_2_junction_aa"
            ), f"There can't be a secondary chain if there is no primary one: {res_dict}"
        if _is_nan_or_missing("IR_VDJ_1_junction_aa"):
            assert _is_nan_or_missing(
                "IR_VDJ_2_junction_aa"
            ), f"There can't be a secondary chain if there is no primary one: {res_dict}"

        return res_dict

    @staticmethod
    def empty_chain_dict() -> dict:
        """Generate an empty chain dictionary, containing all required AIRR
        columns, but set to `None`"""
        return {field: None for field in RearrangementSchema.required}
