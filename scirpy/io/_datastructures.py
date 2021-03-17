"""Datastructures for Adaptive immune receptor (IR) data.

Currently only used as intermediate storage.
See also discussion at https://github.com/theislab/anndata/issues/115
"""

from .._compat import Literal
from ..util import _is_na, _is_true
from typing import Union


class AirrChain:
    """Data structure for an Adaptive Immune Receptor Repertoire Chain.

    The datastructure is compliant with the AIRR rearrangement schema v1.0.
    An instance of this class represents a single row of the AIRR rearrangement schema.

    Parameters
    ----------
    locus
        IGMT locus name or "other".
        See https://docs.airr-community.org/en/latest/datarep/rearrangements.html#locus-names.
    cdr3
        Amino acid sequence of the CDR3 region
    cdr3_nt
        Nucleotide sequence fo the CDR3 region
    expr
        Normalized read count for the CDR3 region.
        Will be UMIs for 10x and TPM for SmartSeq2.
    expr_raw
        Raw read count for the CDR3 regions.
    is_productive
        Is the chain productive?
    v_gene
        gene symbol of v gene
    d_gene
        gene symbol of d gene
    j_gene
        gene symbol of j gene
    c_gene
        gene symbol of c gene
    junction_ins
        nucleotides inserted in the junctions.
        For :term:`VJ<V(D)J>` chains: nucleotides inserted in the VJ junction
        For :term:`VDJ<V(D)J>` chains: sum of nucleotides inserted in the VD + DJ junction
    """

    #: Chains with the :term:`V-J<V(D)J>` junction
    VJ_LOCI = ("TRA", "TRG", "IGK", "IGL")

    #: Chains with the :term:`V-D-J<V(D)J>` junction
    VDJ_LOCI = ("TRB", "TRD", "IGH")

    #: Valid chains are IMGT locus names or "other"
    #: see https://docs.airr-community.org/en/latest/datarep/rearrangements.html#locus-names
    VALID_LOCI = VJ_LOCI + VDJ_LOCI + ("other",)

    # Attributes that are required to be equal for an object to be equal.
    # Used for __eq__ and __hash__
    _EQUALITY_ATTIBUTES = [
        "locus",
        "cdr3",
        "cdr3_nt",
        "expr",
        "expr_raw",
        "is_productive",
        "v_gene",
        "d_gene",
        "j_gene",
        "c_gene",
        "junction_ins",
    ]

    def __init__(
        self,
        locus: Literal["TRA", "TRG", "IGK", "IGL", "TRB", "TRD", "IGH", "other"],
        *,
        cdr3: str = None,
        cdr3_nt: str = None,
        expr: float = None,
        expr_raw: float = None,
        is_productive: bool = None,
        v_gene: str = None,
        d_gene: str = None,
        j_gene: str = None,
        c_gene: str = None,
        junction_ins: int = None,
    ):
        if locus not in self.VALID_LOCI:
            raise ValueError("Invalid chain type: {}".format(locus))

        self.locus = locus
        self.junction_type = (
            "VJ"
            if locus in self.VJ_LOCI
            else ("VDJ" if locus in self.VDJ_LOCI else None)
        )
        self.cdr3 = cdr3.upper() if not _is_na(cdr3) else None
        self.cdr3_nt = cdr3_nt.upper() if not _is_na(cdr3_nt) else None
        self.expr = float(expr)
        self.expr_raw = float(expr_raw) if not _is_na(expr_raw) else None
        self.is_productive = _is_true(is_productive)
        self.v_gene = v_gene
        self.d_gene = d_gene
        self.j_gene = j_gene
        self.c_gene = c_gene
        self.junction_ins = int(junction_ins) if not _is_na(junction_ins) else None

    def __repr__(self):
        return "AirrChain object: " + str(self.__dict__)

    def __eq__(self, other):
        return all(
            getattr(self, attr) == getattr(other, attr)
            for attr in self._EQUALITY_ATTIBUTES
        )

    def __hash__(self):
        return hash(tuple((getattr(self, attr) for attr in self._EQUALITY_ATTIBUTES)))


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

    def __init__(self, cell_id: str, *, multi_chain: bool = False):

        self._cell_id = cell_id
        self.multi_chain = _is_true(multi_chain)
        self.chains = list()

    def __repr__(self):
        return "AirrCell {} with {} chains".format(self._cell_id, len(self.chains))

    @property
    def cell_id(self):
        return self._cell_id

    def add_chain(self, chain: AirrChain) -> None:
        """Add a :class:`AirrChain`"""
        self.chains.append(chain)
