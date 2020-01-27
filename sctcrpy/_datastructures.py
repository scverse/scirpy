from ._compat import Literal


class TcrChain:
    def __init__(
        self,
        chain_type: Literal["TRA", "TRB", "other"],
        *,
        cdr3: str = None,
        cdr3_nt: str = None,
        expr: float = None,
        expr_raw: float = None,
        is_productive: bool = None,
    ):
        """Data structure for a T cell receptor chain. jj
        
        Parameters
        ----------
        chain_type 
            Currently supported: ["TRA", "TRB", "other"]        
        cdr3 
            Amino acid sequence of the CDR3 region 
        cd3_nt 
            Nucleotide sequence fo the CDR3 region 
        expr 
            Normalized read count for the CDR3 region. 
            Will be UMIs for 10x and TPM for SmartSeq2. 
        expr_raw
            Raw read count for the CDR3 regions.
        is_productive 
            Is the chain productive?
        """
        if chain_type not in ["TRA", "TRB", "other"]:
            raise ValueError("Invalid chain type: {}".format(chain_type))

        self.chain_type = chain_type
        self.cdr3 = cdr3.upper() if cdr3 is not None else None
        self.cdr3_nt = cdr3_nt.upper() if cdr3_nt is not None else None
        self.expr = expr
        self.expr_raw = expr_raw
        self.is_productive = is_productive


class TcrCell:
    def __init__(self, cell_id: str):
        self._cell_id = cell_id
        self._chains = list()

    def __repr__(self):
        return "TcrCell {} with {} chains".format(self._cell_id, len(self._chains))

    @property
    def cell_id(self):
        return self._cell_id

    def add_chain(self, chain: TcrChain):
        self._chains.append(chain)
