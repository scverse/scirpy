"""Datastructures for TCR data.

Currently only used as intermediate storage.
See also discussion at https://github.com/theislab/anndata/issues/115
"""

from .._compat import Literal
from ..util import _is_na, _is_true


class IrChain:
    """Data structure for an immune cell receptor chain.

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
        For type == TRA: nucleotides inserted in the VJ junction
        For type == TRB: sum of nucleotides inserted in the VD + DJ junction
    """

    #: Chain 1 is the chain with the V-J junction
    VJ_LOCI = ("TRA", "TRG", "IGK", "IGL")

    #: Chain 2 is the chain with the V-D-J junction
    VDJ_LOCI = ("TRB", "TRD", "IGH")

    #: Valid chains are IMGT locus names or "other"
    #: see https://docs.airr-community.org/en/latest/datarep/rearrangements.html#locus-names
    VALID_LOCI = VJ_LOCI + VDJ_LOCI + ("other",)

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
        self.expr_raw = float(expr_raw) if expr_raw is not None else None
        self.is_productive = _is_true(is_productive)
        self.v_gene = v_gene
        self.d_gene = d_gene
        self.j_gene = j_gene
        self.c_gene = c_gene
        self.junction_ins = junction_ins

    def __repr__(self):
        return "IrChain object: " + str(self.__dict__)


class IrCell:
    """Data structure for a Cell with immune receptors.

    An IrCell can hold multiple IrChains.

    Parameters
    ----------
    cell_id
        cell id or barcode.  Needs to match the cell id used for transcriptomics
        data (i.e. the `adata.obs_names`)
    """

    def __init__(self, cell_id: str):

        self._cell_id = cell_id
        self.chains = list()

    def __repr__(self):
        return "IrCell {} with {} chains".format(self._cell_id, len(self.chains))

    @property
    def cell_id(self):
        return self._cell_id

    def add_chain(self, chain: IrChain) -> None:
        """Add a :class:`IrChain`"""
        self.chains.append(chain)
