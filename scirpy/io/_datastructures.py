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
    """Data structure for a Cell with immune receptors.

    This data structure is compliant with the AIRR rearrangement schema v1.0.
    An AirrCell holds multiple AirrChains (i.e. rows from the rearrangement TSV)
    which belong to the same cell.

    The AirrCell can, additionally hold cell-level attributes which can be set
    in a dict-like fashion. Keys marked as "cell-level" with in `cell_attribute_fields`
    will be automatically transferred to the cell-level when added through a chain.

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
        A logger to write messages to. If not specified, use the default logger.
    """

    #: Chains with the :term:`V-J<V(D)J>` junction
    VJ_LOCI = ("TRA", "TRG", "IGK", "IGL")

    #: Chains with the :term:`V-D-J<V(D)J>` junction
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
        **kwargs,
    ):
        self._logger = logger
        if "multi_chain" in kwargs:
            # legacy argument for compatibility with old anndata schema.
            self._multi_chain = _is_true2(kwargs["multi_chain"])
        else:
            self._multi_chain = False
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
        return "AirrCell {} with {} chains".format(self.cell_id, len(self.chains))

    @property
    def cell_id(self) -> str:
        return self["cell_id"]

    @property
    def chains(self) -> List[Dict]:
        return self._chains

    @property
    def fields(self) -> List[str]:
        """Return a list of all fields (chain-level and cell-level)"""
        if self._chain_fields is None:
            raise ValueError("No chains have been added yet.")
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
        self._cell_attrs[k] = v

    def add_chain(self, chain: Mapping) -> None:
        """Add a chain ot the cell.

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
            try:
                new_value = chain.pop(tmp_field)
                try:
                    existing_value = self[tmp_field]
                    if existing_value != new_value:
                        raise ValueError(
                            "Cell-level attributes differ between different chains. "
                            f"Already present: `{existing_value}`. Tried to add `{new_value}`."
                        )
                except KeyError:
                    self[tmp_field] = new_value
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
            self._logger.warning(f"Non-standard locus name ignored: {chain['locus']} ")  # type: ignore

        self.chains.append(chain)

    def add_serialized_chains(self, serialized_chains) -> None:
        """Add chains serialized as JSON.

        The JSON object needs to be a list of dicts"""
        tmp_chains = json.loads(serialized_chains)
        for chain in tmp_chains:
            self.add_chain(chain)

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
        is_multichain = self._multi_chain
        split_chains = {"VJ": list(), "VDJ": list(), "extra": list()}
        for tmp_chain in self.chains:
            if "locus" not in tmp_chain:
                split_chains["extra"].append(tmp_chain)
            elif tmp_chain["locus"] in self.VJ_LOCI and tmp_chain["productive"]:
                split_chains["VJ"].append(tmp_chain)
            elif tmp_chain["locus"] in self.VDJ_LOCI and tmp_chain["productive"]:
                split_chains["VDJ"].append(tmp_chain)
            else:
                split_chains["extra"].append(tmp_chain)

        if (
            "duplicate_count" not in self._chain_fields
            and "consensus_count" not in self._chain_fields
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
        return tuple(-1 if x is None else x for x in sort_tuple)

    @staticmethod
    def _serialize_chains(chains: List[Dict]) -> str:
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
        return json.dumps(chains)

    @_doc_params(doc_working_model=doc_working_model)
    def to_scirpy_record(
        self, include_fields: Optional[Collection[str]] = None
    ) -> Dict:
        """Convert the cell to a scirpy record (i.e. one row of `adata.obs`) according
        to our working model of adaptive immune receptors.

        {doc_working_model}

        Parameters
        ----------
        include_fields
            AIRR fields to include into `adata.obs` (to save space and to not clutter
            obs). Set to `None` to include all fields.
        """
        res_dict = dict()
        if include_fields is None:
            include_fields = self.fields
        # ensure cell_id is always added
        include_fields = set(include_fields)
        include_fields.add("cell_id")
        res_dict["multi_chain"], chain_dict = self._split_chains()
        res_dict["extra_chains"] = self._serialize_chains(chain_dict.pop("extra"))
        # add cell-level attributes
        for key in self:
            if key in include_fields:
                res_dict[key] = self[key]

        # add chain-level attributes
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

        # in some weird reasons, it can happen that a cell has been called from
        # TCR-seq but no TCR seqs have been found. `has_ir` should be equal
        # to "at least one productive chain"
        res_dict["has_ir"] = not (
            res_dict["IR_VJ_1_junction_aa"] is None
            and res_dict["IR_VDJ_1_junction_aa"] is None
        )

        # if there are not chains at all, we want multi-chain to be nan
        # This is to be consistent with when turning an anndata object into ir_objs
        # and converting it back to anndata.
        if not len(self.chains):
            res_dict["multi_chain"] = np.nan

        if res_dict["IR_VJ_1_junction_aa"] is None:
            assert (
                res_dict["IR_VJ_2_junction_aa"] is None
            ), "There can't be a secondary chain if there is no primary one: {}".format(
                res_dict
            )
        if res_dict["IR_VDJ_1_junction_aa"] is None:
            assert (
                res_dict["IR_VDJ_2_junction_aa"] is None
            ), "There can't be a secondary chain if there is no primary one: {}".format(
                res_dict
            )

        return res_dict

    @staticmethod
    def empty_chain_dict() -> dict:
        """Generate an empty chain dictionary, containing all required AIRR
        columns, but set to `None`"""
        return {field: None for field in RearrangementSchema.required}
