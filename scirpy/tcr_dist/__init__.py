import parasail
from ..util._multiprocessing import EnhancedPool as Pool
import itertools
from anndata import AnnData
from typing import Union, Collection, List, Tuple, Dict
from .._compat import Literal
import numpy as np
from scanpy import logging
from ..util import _is_na
import abc
from Levenshtein import distance as levenshtein_dist
import scipy.spatial
import scipy.sparse
from scipy.sparse import coo_matrix, csr_matrix
from ..util import _doc_params

_doc_metrics = """\
metric
    You can choose one of the following metrics: 
      * `identity` -- 1 for identical sequences, 0 otherwise. 
        See :class:`~scirpy.tcr_dist.IdentityDistanceCalculator`. 
      * `levenshtein` -- Levenshtein edit distance.
        See :class:`~scirpy.tcr_dist.LevenshteinDistanceCalculator`. 
      * `alignment` -- Distance based on pairwise sequence alignments using the 
        BLOSUM62 matrix. This option is incompatible with nucleotide sequences. 
        See :class:`~scirpy.tcr_dist.AlignmentDistanceCalculator`. 
      * any instance of :class:`~scirpy.tcr_dist.DistanceCalculator`. 
"""

_doc_cutoff = """\
cutoff
    All distances `> cutoff` will be replaced by `0` and eliminated from the sparse
    matrix. A sensible cutoff depends on the distance metric, you can find 
    information in the corresponding docs. 
"""

_doc_params_distance_calculator = """\
cutoff
    Will eleminate distances > cutoff to make efficient 
    use of sparse matrices. 
n_jobs
    Number of jobs to use for the pairwise distance calculation. 
    If None, use all jobs. 
"""

_doc_dist_mat = """\
Calculates the upper triangle, including the diagonal. 

.. important::
    * Distances are offset by 1 to allow efficient use of sparse matrices 
      (:math:`d' = d+1`). 
    * That means, a `distance > cutoff` is represented as `0`, a `distance == 0` 
      is represented as `1`, a `distance == 1` is represented as `2` and so on. 
    * Only returns distances `<= cutoff`. Larger distances are eliminated 
      from the sparse matrix. 
    * Distances are non-negative. 
"""


@_doc_params(params=_doc_params_distance_calculator)
class DistanceCalculator(abc.ABC):
    """\
    Abstract base class for a :term:`CDR3`-sequence distance calculator.
    
    Parameters
    ----------
    {params}
    
    """

    #: The sparse matrix dtype. Defaults to uint8, constraining the max distance to 255.
    DTYPE = "uint8"

    def __init__(self, cutoff: float, n_jobs: Union[int, None] = None):
        if cutoff > 255:
            raise ValueError(
                "Using a cutoff > 255 is not possible due to the `uint8` dtype used"
            )
        self.cutoff = cutoff
        self.n_jobs = n_jobs

    @_doc_params(dist_mat=_doc_dist_mat)
    @abc.abstractmethod
    def calc_dist_mat(self, seqs: np.ndarray) -> coo_matrix:
        """\
        Calculate pairwise distance matrix of all sequences in `seqs`.
        
        {dist_mat}
       
        Parameters
        ----------
        seqs
            array containing CDR3 sequences. Should not contain duplicates. 

        Returns
        -------
        Sparse, upper triangular distance matrix. 
        """
        pass


@_doc_params(params=_doc_params_distance_calculator)
class IdentityDistanceCalculator(DistanceCalculator):
    """\
    Calculates the Identity-distance between :term:`CDR3` sequences. 

    The identity distance is defined as 
        * `0`, if sequences are identical
        * `1`, if sequences are not identical.

    Choosing a cutoff:
        For this DistanceCalculator, per definition, the cutoff = 0. 
        The `cutoff` argument is ignored. 

    Parameters
    ----------
    {params}
    """

    def __init__(self, cutoff: float = 0, n_jobs: Union[int, None] = None):
        super().__init__(cutoff, n_jobs)

    def calc_dist_mat(self, seqs: np.ndarray) -> coo_matrix:
        """In this case, the offseted distance matrix is the identity matrix. 
        
        More details: :meth:`DistanceCalculator.calc_dist_mat`"""
        return scipy.sparse.identity(len(seqs), dtype=self.DTYPE, format="coo")


@_doc_params(params=_doc_params_distance_calculator)
class LevenshteinDistanceCalculator(DistanceCalculator):
    """\
    Calculates the Levenshtein edit-distance between sequences. 
    
    The edit distance is the total number of deletion, addition and modification 
    events. 

    This class relies on `Python-levenshtein <https://github.com/ztane/python-Levenshtein>`_
    to calculate the distances. 

    Choosing a cutoff:
        Each modification stands for a deletion, addition or modification event. 
        While lacking empirical data, it seems unlikely that CDR3 sequences with more
        than two modifications still recognize the same antigen. 

    Parameters
    ----------
    {params}
    """

    def _compute_row(self, seqs: np.ndarray, i_row: int) -> coo_matrix:
        """Compute a row of the upper diagnomal distance matrix"""
        target = seqs[i_row]

        def coord_generator():
            for j, s2 in enumerate(seqs[i_row:], start=i_row):
                d = levenshtein_dist(target, s2)
                if d <= self.cutoff:
                    yield d + 1, j

        d, col = zip(*coord_generator())
        row = np.zeros(len(col), dtype="int")
        return coo_matrix((d, (row, col)), dtype=self.DTYPE, shape=(1, seqs.size))

    def calc_dist_mat(self, seqs: np.ndarray) -> csr_matrix:
        """Calculate the distance matrix. 

        See :meth:`DistanceCalculator.calc_dist_mat`. """
        p = Pool(self.n_jobs)
        rows = p.starmap_progress(
            self._compute_row,
            zip(itertools.repeat(seqs), range(len(seqs))),
            chunksize=200,
            total=len(seqs),
        )
        p.close()

        score_mat = scipy.sparse.vstack(rows)
        score_mat.eliminate_zeros()
        assert score_mat.shape[0] == score_mat.shape[1]

        return score_mat


@_doc_params(params=_doc_params_distance_calculator)
class AlignmentDistanceCalculator(DistanceCalculator):
    """\
    Calculates distance between sequences based on pairwise sequence alignment. 

    The distance between two sequences is defined as :math:`S_{{1,2}}^{{max}} - S_{{1,2}}`, 
    where :math:`S_{{1,2}}` is the alignment score of sequences 1 and 2 and 
    :math:`S_{{1,2}}^{{max}}` is the max. achievable alignment score of sequences 1 and 2. 
    :math:`S_{{1,2}}^{{max}}` is defined as :math:`\\min(S_{{1,1}}, S_{{2,2}})`. 

    The use of alignment-based distances is heavily inspired by :cite:`TCRdist`. 

    High-performance sequence alignments are calculated leveraging 
    the `parasail library <https://github.com/jeffdaily/parasail-python>`_ (:cite:`Daily2016`).  

    Choosing a cutoff:
        Alignment distances need to be viewed in the light of the substitution matrix. 
        The alignment distance is the difference between the actual alignment 
        score and the max. achievable alignment score. For instance, a mutation 
        from *Leucine* (`L`) to *Isoleucine* (`I`) results in a BLOSUM62 score of `2`. 
        An `L` aligned with `L` achieves a score of `4`. The distance is, therefore, `2`.

        On the other hand, a single *Tryptophane* (`W`) mutating into, e.g.
        *Proline* (`P`) already results in a distance of `15`. 

        We are still lacking empirical data up to which distance a CDR3 sequence still 
        is likely to recognize the same antigen, but reasonable cutoffs are `<15`. 

    Parameters
    ----------
    {params}
    subst_mat
        Name of parasail substitution matrix
    gap_open
        Gap open penalty
    gap_extend
        Gap extend penatly
    """

    def __init__(
        self,
        cutoff: float,
        n_jobs: Union[int, None] = None,
        *,
        subst_mat: str = "blosum62",
        gap_open: int = 11,
        gap_extend: int = 11,
    ):
        super().__init__(cutoff, n_jobs)
        self.subst_mat = subst_mat
        self.gap_open = gap_open
        self.gap_extend = gap_extend

    def _align_row(
        self, seqs: np.ndarray, self_alignment_scores: np.array, i_row: int
    ) -> np.ndarray:
        """Generates a row of the triangular distance matrix. 
        
        Aligns `seqs[i_row]` with all other sequences in `seqs[i_row:]`. 

        Parameters
        ----------
        seqs
            Array of amino acid sequences
        self_alignment_scores
            Array containing the scores of aligning each sequence in `seqs` 
            with itself. This is used as a reference value to turn 
            alignment scores into distances. 
        i_row
            Index of the row in the final distance matrix. Determines the target sequence. 

        Returns
        -------
        The i_th row of the final score matrix. 
        """
        subst_mat = parasail.Matrix(self.subst_mat)
        target = seqs[i_row]
        profile = parasail.profile_create_16(target, subst_mat)

        def coord_generator():
            for j, s2 in enumerate(seqs[i_row:], start=i_row):
                r = parasail.nw_scan_profile_16(
                    profile, s2, self.gap_open, self.gap_extend
                )
                max_score = np.min(self_alignment_scores[[i_row, j]])
                d = max_score - r.score
                if d <= self.cutoff:
                    yield d + 1, j

        d, col = zip(*coord_generator())
        row = np.zeros(len(col), dtype="int")
        return coo_matrix((d, (row, col)), dtype=self.DTYPE, shape=(1, len(seqs)))

    def calc_dist_mat(self, seqs: Collection) -> coo_matrix:
        """Calculate the distances between amino acid sequences based on
        of all-against-all pairwise sequence alignments.

        See :meth:`DistanceCalculator.calc_dist_mat` for more details. 
        """
        # first, calculate self-alignments. We need them as refererence values
        # to turn scores into dists
        self_alignment_scores = np.array(
            [
                parasail.nw_scan_16(
                    s,
                    s,
                    self.gap_open,
                    self.gap_extend,
                    parasail.Matrix(self.subst_mat),
                ).score
                for s in seqs
            ]
        )

        p = Pool(self.n_jobs)
        rows = p.starmap_progress(
            self._align_row,
            zip(
                itertools.repeat(seqs),
                itertools.repeat(self_alignment_scores),
                range(len(seqs)),
            ),
            chunksize=200,
            total=len(seqs),
        )
        p.close()

        score_mat = scipy.sparse.vstack(rows)
        score_mat.eliminate_zeros()
        assert score_mat.shape[0] == score_mat.shape[1]

        return score_mat


@_doc_params(metric=_doc_metrics, cutoff=_doc_cutoff, dist_mat=_doc_dist_mat)
def tcr_dist(
    unique_seqs: np.ndarray,
    *,
    metric: Union[
        Literal["alignment", "identity", "levenshtein"], DistanceCalculator
    ] = "identity",
    cutoff: float = 10,
    n_jobs: Union[int, None] = None,
):
    """\
    Calculate a sequence x sequence distance matrix.

    {dist_mat}
    
    Parameters
    ----------
    unique_seqs
        Numpy array of nucleotide or amino acid sequences. 
        Must not contain duplicates. 
        Note that not all distance metrics support nucleotide sequences. 
    {metric}
    {cutoff}

    Returns
    -------
    Upper triangular distance matrix. 
    """
    if isinstance(metric, DistanceCalculator):
        dist_calc = metric
    elif metric == "alignment":
        dist_calc = AlignmentDistanceCalculator(cutoff=cutoff, n_jobs=n_jobs)
    elif metric == "identity":
        dist_calc = IdentityDistanceCalculator(cutoff=cutoff)
    elif metric == "levenshtein":
        dist_calc = LevenshteinDistanceCalculator(cutoff=cutoff, n_jobs=n_jobs)
    else:
        raise ValueError("Invalid distance metric.")

    dist_mat = dist_calc.calc_dist_mat(unique_seqs)
    return dist_mat


class TcrNeighbors:
    def __init__(
        self,
        adata: AnnData,
        *,
        metric: Union[
            Literal["alignment", "identity", "levenshtein"], DistanceCalculator
        ] = "identity",
        cutoff: float = 0,
        receptor_arms: Literal["TRA", "TRB", "all", "any"] = "all",
        dual_tcr: Literal["primary_only", "all", "any"] = "primary_only",
        sequence: Literal["aa", "nt"] = "aa",
    ):
        """Class to compute Neighborhood graphs of CDR3 sequences. 

        For documentation of the parameters, see :func:`tcr_neighbors`. 
        """
        if metric == "identity" and cutoff != 0:
            raise ValueError("Identity metric only works with cutoff = 0")
        if sequence == "nt" and metric == "alignment":
            raise ValueError(
                "Using nucleotide sequences with alignment metric is not supported. "
            )
        self.adata = adata
        self.metric = metric
        self.cutoff = cutoff
        self.receptor_arms = receptor_arms
        self.dual_tcr = dual_tcr
        self.sequence = sequence
        self._build_index_dict()
        self._dist_mat = None
        logging.debug("Finished initalizing TcrNeighbors object. ")

    @staticmethod
    def _seq_to_cell_idx(
        unique_seqs: np.ndarray, cdr_seqs: np.ndarray
    ) -> Dict[int, List[int]]:
        """
        Compute sequence to cell index for a single chain (e.g. `TRA_1`). 

        Maps cell_idx -> [list, of, seq_idx]. 
        Useful to build a cell x cell matrix from a seq x seq matrix. 

        Computes magic lookup indexes in linear time. 

        Parameters
        ----------
        unique_seqs
            Pool of all unique cdr3 sequences (length = #unique cdr3 sequences)
        cdr_seqs
            CDR3 sequences for the current chain (length = #cells)

        Returns
        -------
        Sequence2Cell mapping    
        """
        # 1) reverse mapping of amino acid sequence to index in sequence-distance matrix
        seq_to_index = {seq: i for i, seq in enumerate(unique_seqs)}

        # 2) indices of cells in adata that have a CDR3 sequence.
        cells_with_chain = np.where(~_is_na(cdr_seqs))[0]

        # 3) indices of the corresponding sequences in the distance matrix.
        seq_inds = {
            chain_id: seq_to_index[cdr_seqs[chain_id]] for chain_id in cells_with_chain
        }

        # 4) list of cell-indices in the cell distance matrix for each sequence
        seq_to_cell = {seq_id: list() for seq_id in seq_to_index.values()}
        for cell_id in cells_with_chain:
            seq_id = seq_inds[cell_id]
            seq_to_cell[seq_id].append(cell_id)

        return seq_to_cell

    def _build_index_dict(self):
        """Build nested dictionary for each receptor arm (TRA, TRB) containing all 
        combinations of receptor_arms x primary/secondary_chain
        
        If the merge mode for either `receptor_arm` or `dual_tcr` is `all`, 
        includes a lookup table that contains the number of CDR3 sequences for
        each cell. 
        """
        receptor_arms = (
            ["TRA", "TRB"]
            if self.receptor_arms not in ["TRA", "TRB"]
            else [self.receptor_arms]
        )
        chain_inds = [1] if self.dual_tcr == "primary_only" else [1, 2]
        sequence = "" if self.sequence == "aa" else "_nt"

        arm_dict = {}
        for arm in receptor_arms:
            cdr_seqs = {
                k: self.adata.obs[f"{arm}_{k}_cdr3{sequence}"].values
                for k in chain_inds
            }
            unique_seqs = np.hstack(list(cdr_seqs.values()))
            unique_seqs = np.unique(unique_seqs[~_is_na(unique_seqs)]).astype(str)
            seq_to_cell = {
                k: self._seq_to_cell_idx(unique_seqs, cdr_seqs[k]) for k in chain_inds
            }
            arm_dict[arm] = {
                "chain_inds": chain_inds,
                "unique_seqs": unique_seqs,
                "seq_to_cell": seq_to_cell,
            }

            # need the count of chains per cell for the `all` strategies.
            if self.receptor_arms == "all" or self.dual_tcr == "all":
                arm_dict[arm]["chains_per_cell"] = np.sum(
                    ~_is_na(
                        self.adata.obs.loc[
                            :, [f"{arm}_{k}_cdr3{sequence}" for k in chain_inds]
                        ]
                    ),
                    axis=1,
                )

        self.index_dict = arm_dict

    def _reduce_dual_all(self, d, chain, cell_row, cell_col):
        """Reduce dual TCRs into a single value when 'all' sequences
        need to match. This requires additional checking effort for the number
        of chains in the given cell, since we can't make the distinction between
        no chain and dist > cutoff based on the distances (both would contain a
        0 in the distance matrix)."""
        chain_count = (
            self.index_dict[chain]["chains_per_cell"][cell_row],
            self.index_dict[chain]["chains_per_cell"][cell_col],
        )
        if len(d) == 1 and chain_count == (1, 1):
            # exactely one chain for both cells -> return that value
            return next(iter(d.values()))
        elif chain_count == (2, 2):
            # two options: either (1 matches 2 and 2 matches 1)
            # or (1 matches 1 and 2 matches 2).
            try:
                # minus 1, because both dists are offseted by 1.
                d1 = d[(1, 2)] + d[(2, 1)] - 1
            except KeyError:
                d1 = None
            try:
                d2 = d[(1, 1)] + d[(2, 2)] - 1
            except KeyError:
                d2 = None

            if d1 is not None and d2 is not None:
                return min(d1, d2)
            elif d1 is not None:
                return d1
            elif d2 is not None:
                return d2
            else:
                return 0

        else:
            return 0

    def _reduce_arms_all(self, values, cell_row, cell_col):
        """Reduce multiple receptor arms into a single value when 'all' sequences
        need to match. This requires additional checking effort for teh number 
        of chains in the given cell, since we can't make the distinction between
        no chain and dist > cutoff based on the distances (both would contain a
        0 in the distance matrix)."""
        values = (x for x in values if x != 0)

        try:
            arm1 = next(values)
        except StopIteration:
            # no value > 0
            return 0

        try:
            arm2 = next(values)
            # two receptor arms -> easy
            # -1 because both distances are offseted by 1
            return arm1 + arm2 - 1
        except StopIteration:
            # only one arm
            tra_chains = (
                self.index_dict["TRA"]["chains_per_cell"][cell_row],
                self.index_dict["TRA"]["chains_per_cell"][cell_col],
            )
            trb_chains = (
                self.index_dict["TRB"]["chains_per_cell"][cell_row],
                self.index_dict["TRB"]["chains_per_cell"][cell_col],
            )
            # Either exactely one chain for TRA or
            # exactely on e chain for TRB for both cells.
            if tra_chains == (0, 0) or trb_chains == (0, 0):
                return arm1
            else:
                return 0

    @staticmethod
    def _reduce_arms_any(lst, *args):
        """Reduce arms when *any* of the sequences needs to match. 
        This is the simpler case. This also works with only one entry
        (e.g. arms = "TRA") """
        # need to exclude 0 values, since the dist mat is offseted by 1.
        try:
            return min(x for x in lst if x != 0)
        except ValueError:
            # no values in generator
            return 0

    @staticmethod
    def _reduce_dual_any(d, *args):
        """Reduce dual tcrs to a single value when *any* of the sequences needs 
        to match (by minimum). This also works with only one entry (i.e. 'primary only')
        """
        # need to exclude 0 values, since the dist mat is offseted by 1.
        try:
            return min(x for x in d.values() if x != 0)
        except ValueError:
            # no values in generator
            return 0

    def _reduce_coord_dict(self, coord_dict):
        """Applies reduction functions to the coord dict.
        Yield (coords, value) pairs. """
        reduce_dual = (
            self._reduce_dual_all if self.dual_tcr == "all" else self._reduce_dual_any
        )
        reduce_arms = (
            self._reduce_arms_all
            if self.receptor_arms == "all"
            else self._reduce_arms_any
        )
        for (cell_row, cell_col), entry in coord_dict.items():
            reduced_dual = (
                reduce_dual(value_dict, chain, cell_row, cell_col)
                for chain, value_dict in entry.items()
            )
            reduced = reduce_arms(reduced_dual, cell_row, cell_col,)
            yield (cell_row, cell_col), reduced

    def _cell_dist_mat_reduce(self):
        """Compute the distance matrix by using custom reduction functions. 
        More flexible than `_build_cell_dist_mat_min`, but requires more memory.
        Reduce dual is called before reduce arms. 
        """
        coord_dict = dict()

        def _add_to_dict(d, c1, c2, cell_row, cell_col, value):
            """Add a value to the nested coord dict"""
            try:
                tmp_dict = d[(cell_row, cell_col)]
                try:
                    tmp_dict2 = tmp_dict[arm]
                    try:
                        if (c1, c2) in tmp_dict2:
                            # can be in arbitrary order apprarently
                            assert (c2, c1) not in tmp_dict2
                            tmp_dict2[(c2, c1)] = value
                        tmp_dict2[(c1, c2)] = value
                    except KeyError:
                        tmp_dict2 = {(c1, c2): value}
                except KeyError:
                    tmp_dict[arm] = {(c1, c2): value}
            except KeyError:
                d[(cell_row, cell_col)] = {arm: {(c1, c2): value}}

        for arm, arm_info in self.index_dict.items():
            dist_mat, seq_to_cell, chain_inds = (
                arm_info["dist_mat"],
                arm_info["seq_to_cell"],
                arm_info["chain_inds"],
            )
            for row, col, value in zip(dist_mat.row, dist_mat.col, dist_mat.data):
                for c1, c2 in itertools.product(chain_inds, repeat=2):
                    for cell_row, cell_col in itertools.product(
                        seq_to_cell[c1][row], seq_to_cell[c2][col]
                    ):
                        # fill upper diagonal. Important: these are dist-mat row,cols
                        # not cell-mat row cols. This is required, because the
                        # itertools.product returns all combinations for the diagonal
                        # but not for the other values.
                        _add_to_dict(coord_dict, c1, c2, cell_row, cell_col, value)
                        if row != col:
                            _add_to_dict(coord_dict, c1, c2, cell_col, cell_row, value)

        logging.debug("Finished constructing coord-dictionary")

        yield from self._reduce_coord_dict(coord_dict)

    def compute_distances(
        self, n_jobs: Union[int, None] = None,
    ):
        """Computes the distances between CDR3 sequences 

        Parameters
        ----------
        n_jobs
            Number of CPUs to use for alignment and levenshtein distance. 
            Default: use all CPUS. 
        """
        for arm, arm_dict in self.index_dict.items():
            arm_dict["dist_mat"] = tcr_dist(
                arm_dict["unique_seqs"],
                metric=self.metric,
                cutoff=self.cutoff,
                n_jobs=n_jobs,
            )
            logging.info("Finished computing {} pairwise distances.".format(arm))

        coords, values = zip(*self._cell_dist_mat_reduce())
        rows, cols = zip(*coords)
        dist_mat = coo_matrix(
            (values, (rows, cols)), shape=(self.adata.n_obs, self.adata.n_obs)
        )
        logging.info("Finished constructing cell x cell distance matrix. ")
        dist_mat.eliminate_zeros()
        self._dist_mat = dist_mat.tocsr()

    @property
    def dist(self):
        """The computed distance matrix. 
        Requires to invoke `compute_distances() first. """
        return self._dist_mat

    @property
    def connectivities(self):
        """Get the weighted adjacecency matrix derived from the distance matrix. 

        The cutoff will be used to normalize the distances. 
        """
        if self.cutoff == 0:
            return self._dist_mat

        connectivities = self._dist_mat.copy()

        # actual distances
        d = connectivities.data - 1

        # structure of the matrix stayes the same, we can safely change the data only
        connectivities.data = (self.cutoff - d) / self.cutoff
        connectivities.eliminate_zeros()
        return connectivities


@_doc_params(metric=_doc_metrics, cutoff=_doc_cutoff, dist_mat=_doc_dist_mat)
def tcr_neighbors(
    adata: AnnData,
    *,
    metric: Union[
        Literal["identity", "alignment", "levenshtein"], DistanceCalculator
    ] = "alignment",
    cutoff: int = 10,
    receptor_arms: Literal["TRA", "TRB", "all", "any"] = "all",
    dual_tcr: Literal["primary_only", "any", "all"] = "primary_only",
    key_added: str = "tcr_neighbors",
    sequence: Literal["aa", "nt"] = "aa",
    inplace: bool = True,
    n_jobs: Union[int, None] = None,
) -> Union[Tuple[csr_matrix, csr_matrix], None]:
    """\
    Construct a neighborhood graph based on :term:`CDR3` sequence similarity. 

    All cells with a CDR3 distance `< cutoff` receive an edge in the graph. 
    Edges are weighted by the distance. 

    Parameters
    ----------
    adata
        annotated data matrix
    {metric}
    {cutoff}

        Two cells with a distance <= the cutoff will be connected. 
        If cutoff = 0, the CDR3 sequences need to be identical. In this 
        case, no alignment is performed. 

    receptor_arms:
         * `"TRA"` - only consider TRA sequences
         * `"TRB"` - only consider TRB sequences
         * `"all"` - both TRA and TRB need to match
         * `"any"` - either TRA or TRB need to match

    dual_tcr:
         * `"primary_only"` - only consider most abundant pair of TRA/TRB chains
         * `"any"` - consider both pairs of TRA/TRB sequences. Distance must be below
           cutoff for any of the chains. 
         * `"all"` - consider both pairs of TRA/TRB sequences. Distance must be below
           cutoff for all of the chains. 

        See also :term:`Dual TCR`
        
    key_added:
        dict key under which the result will be stored in `adata.uns`
        when `inplace` is True.
    sequence:
        Use amino acid (`aa`) or nulceotide (`nt`) sequences?
    inplace:
        If `True`, store the results in `adata.uns`. Otherwise return
        the results. 
    n_jobs:
        Number of cores to use for alignment and levenshtein distance. 
    
    Returns
    -------
    connectivities
        weighted adjacency matrix
    dist
        cell x cell distance matrix with the distances as computed according to `metric`
        offsetted by 1 to make use of sparse matrices. 
    """
    if cutoff == 0:
        metric = "identity"
    ad = TcrNeighbors(
        adata,
        metric=metric,
        cutoff=cutoff,
        receptor_arms=receptor_arms,
        dual_tcr=dual_tcr,
        sequence=sequence,
    )
    ad.compute_distances(n_jobs)
    logging.debug("Finished converting distances to connectivities. ")

    if not inplace:
        return ad.connectivities, ad.dist
    else:
        adata.uns[key_added] = dict()
        adata.uns[key_added]["params"] = {
            "metric": metric,
            "cutoff": cutoff,
            "dual_tcr": dual_tcr,
            "receptor_arms": receptor_arms,
        }
        adata.uns[key_added]["connectivities"] = ad.connectivities
        adata.uns[key_added]["distances"] = ad.dist
