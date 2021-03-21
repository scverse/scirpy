"""Datastructures for Adaptive immune receptor (IR) data.

Currently only used as intermediate storage.
See also discussion at https://github.com/theislab/anndata/issues/115
"""

from .._compat import Literal
from ..util import _is_na, _is_true, _doc_params
from typing import Collection, Mapping, Union
from airr import RearrangementSchema
from scanpy import logging
import json
import numpy as np
from ._common_doc import doc_working_model


class AirrCell:
    """Data structure for a Cell with immune receptors.

    This data structure is compliant with the AIRR rearrangement schema v1.0.
    An AirrCell holds multiple AirrChains (i.e. rows from the rearrangement TSV)
    which belong to the same cell.

    Parameters
    ----------
    cell_id
        cell id or barcode.  Needs to match the cell id used for transcriptomics
        data (i.e. the `adata.obs_names`)
    multi_chain
        explicitly mark this cell as :term:`Multichain-cell`. Even if this is set to
        `False`, :func:`scirpy.io.from_ir_objs` will consider the cell as multi chain,
        if it has more than two :term:`VJ<V(D)J>` or :term:`VDJ<V(D)J>` chains. However,
        if this is set to `True`, the function will consider it as multi-chain
        regardless of the number of chains.
    """

    #: Chains with the :term:`V-J<V(D)J>` junction
    VJ_LOCI = ("TRA", "TRG", "IGK", "IGL")

    #: Chains with the :term:`V-D-J<V(D)J>` junction
    VDJ_LOCI = ("TRB", "TRD", "IGH")

    # TODO remove?
    #: Valid chains are IMGT locus names or "other"
    #: see https://docs.airr-community.org/en/latest/datarep/rearrangements.html#locus-names
    VALID_LOCI = VJ_LOCI + VDJ_LOCI + ("other",)

    # attributes that are specific for the cell, not the chain, and should
    # be the same for all chains of the same cell.
    _CELL_ATTRIBUTES = ("is_cell",)  # non-standard field by 10x

    def __init__(self, cell_id: str, *, multi_chain: bool = False):

        self._cell_id = cell_id
        self._multi_chain = _is_true(multi_chain)
        self._fields = None
        self.chains = list()

    def __repr__(self):
        return "AirrCell {} with {} chains".format(self._cell_id, len(self.chains))

    @property
    def cell_id(self):
        return self._cell_id

    def add_chain(self, chain: Mapping) -> None:
        """Add a chain ot the cell.

        A chain is a dictionary following
        the `AIRR Rearrangement Schema <https://docs.airr-community.org/en/latest/datarep/rearrangements.html#productive>`__.
        """
        # TODO this should be `.validate_obj` but currently does not work
        # because of https://github.com/airr-community/airr-standards/issues/508
        RearrangementSchema.validate_header(chain.keys())
        RearrangementSchema.validate_row(chain.keys())

        if self._fields is None:
            self._fields = list(chain.keys())
        elif self._fields != list(chain.keys()):
            raise ValueError("All chains must have the same fields!")

        # TODO scirpy warning
        if "locus" not in chain:
            logging.warning(
                "`locus` field not specified, but required for most scirpy functionality. "
            )  # type: ignore
        elif chain["locus"] not in self.VALID_LOCI:
            logging.warning("scirpy only considers valid IGMT locus names. ")  # type: ignore

        self.chains.append(chain)

    def add_serialized_chains(self, serialized_chains):
        tmp_chains = json.loads(serialized_chains)
        for chain in tmp_chains:
            self.add_chain(chain)

    def _split_chains(self):
        """
        Splits the chains into productive VJ, productive VDJ, and extra chains.

        Returns
        -------
            dictionary with the following entries:
              * `vj_chains`: The (up to) two most highly expressed, productive VJ-chains
              * vdj_chains: The (up to) two most highly expressed, productive VDJ chains
              * extra_chains: All remaining chains
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

        # TODO warning
        # if (
        #     "duplicate_count" not in self._fields
        #     and "umi_count" not in self._fields
        #     and "consensus_count" not in self._fields
        # ):
        #     logging.warning(
        #         "No expression information available. Cannot rank chains by expression. "
        #     )  # type: ignore

        for junction_type in ["VJ", "VDJ"]:
            split_chains[junction_type] = sorted(
                split_chains[junction_type], key=self._key_sort_chains, reverse=True
            )
            # only keep the (up to) two most highly expressed chains
            tmp_extra_chains = split_chains[junction_type][2:]
            # if productive chains are appended to extra chains, it's a multichain cell!
            is_multichain |= len(tmp_extra_chains)
            split_chains["extra"] = tmp_extra_chains
            split_chains[junction_type] = split_chains[junction_type][:2]

        return is_multichain, split_chains

    @staticmethod
    def _key_sort_chains(chain):
        """Get key to sort chains by expression"""
        return (
            chain.get("duplicate_count", 0),
            chain.get("umi_count", 0),
            chain.get("consensus_count", 0),
            chain.get("junction", ""),
            chain.get("junction_aa", ""),
        )

    @staticmethod
    def _serialize_chains(chains):
        """Serialize chains into a JSON object. This is useful for storing
        an arbitrary number of extra chains in a single column of a dataframe."""
        return json.dumps(chains)

    # TODO should it be `include_fields` instead?
    @_doc_params(doc_working_model=doc_working_model)
    def to_scirpy_record(
        self, drop_fields: Collection[str] = ("sequence", "sequence_aa")
    ):
        """Convert the cell to a scirpy record (i.e. one row of `adata.obs`) according
        to our working model of adaptive immune receptors.

        {doc_working_model}

        Parameters
        ----------
        drop_fields
            AIRR fields not to include into `adata.obs` (to save space and to not clutter
            obs)
        """
        res_dict = dict()
        res_dict["cell_id"] = self.cell_id
        res_dict["multi_chain"], chain_dict = self._split_chains()
        res_dict["extra_chains"] = self._serialize_chains(chain_dict.pop("extra"))

        for key in self._fields:
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
            _is_na(res_dict["IR_VJ_1_cdr3"]) and _is_na(res_dict["IR_VDJ_1_cdr3"])
        )

        # if there are not chains at all, we want multi-chain to be nan
        # This is to be consistent with when turning an anndata object into ir_objs
        # and converting it back to anndata.
        if not len(self.chains):
            res_dict["multi_chain"] = np.nan

        if _is_na(res_dict["IR_VJ_1_cdr3"]):
            assert _is_na(
                res_dict["IR_VJ_2_cdr3"]
            ), "There can't be a secondary chain if there is no primary one: {}".format(
                res_dict
            )
        if _is_na(res_dict["IR_VDJ_1_cdr3"]):
            assert _is_na(
                res_dict["IR_VDJ_2_cdr3"]
            ), "There can't be a secondary chain if there is no primary one: {}".format(
                res_dict
            )

        return res_dict

        # TODO move .upper() to the functions that consume cdr3 sequences
