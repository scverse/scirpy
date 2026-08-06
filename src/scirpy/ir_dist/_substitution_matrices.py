from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SubstitutionMatrix:
    alphabet: str
    matrix: np.ndarray


CANONICAL_AA_ALPHABET = "ARNDCQEGHILKMFPSTWYV"
AA_ALPHABET_WITH_AMBIGUOUS = f"{CANONICAL_AA_ALPHABET}BZX"
AA_ALPHABET_WITH_UNKNOWN = f"{AA_ALPHABET_WITH_AMBIGUOUS}*"


def _map_matrix_to_alphabet(
    matrix: np.ndarray,
    source_alphabet: str,
    target_alphabet: str,
) -> np.ndarray:
    """Map a matrix from the source alphabet to the target alphabet.

    Rows and columns for characters that occur only in `target_alphabet` are
    initialized to zero.

    Parameters
    ----------
    matrix:
        Square matrix in the order specified by `source_alphabet`.
    source_alphabet:
        Alphabet describing the rows and columns of `matrix`.
    target_alphabet:
        Alphabet describing the rows and columns of the returned matrix. It must
        contain every character from `source_alphabet`.

    Returns
    -------
    mapped_matrix:
        Matrix in the order specified by `target_alphabet`. Entries for target
        characters absent from `source_alphabet` are initialized to zero.
    """
    expected_shape = (len(source_alphabet), len(source_alphabet))
    if matrix.shape != expected_shape:
        raise ValueError(f"`matrix` must have shape {expected_shape} to match `source_alphabet`.")

    if len(set(source_alphabet)) != len(source_alphabet):
        raise ValueError("`source_alphabet` must not contain duplicate characters.")
    if len(set(target_alphabet)) != len(target_alphabet):
        raise ValueError("`target_alphabet` must not contain duplicate characters.")

    target_indices = {character: index for index, character in enumerate(target_alphabet)}
    missing_characters = set(source_alphabet) - set(target_alphabet)
    if missing_characters:
        raise ValueError(
            f"`target_alphabet` is missing characters from `source_alphabet`: {sorted(missing_characters)}"
        )

    mapped_indices = [target_indices[character] for character in source_alphabet]
    mapped_matrix = np.zeros((len(target_alphabet), len(target_alphabet)), dtype=matrix.dtype)
    mapped_matrix[np.ix_(mapped_indices, mapped_indices)] = matrix
    return mapped_matrix


def _substitution_to_distance_matrix(
    substitution_matrix: np.ndarray,
    alphabet: str = AA_ALPHABET_WITH_UNKNOWN,
    matrix_alphabet: str = CANONICAL_AA_ALPHABET,
    distance_cap: int | None = 4,
    distance_offset: int = 4,
) -> np.ndarray:
    """Create a distance lookup matrix from an amino-acid substitution matrix.

    Parameters
    ----------
    substitution_matrix:
        Amino-acid substitution matrix in the order specified by `matrix_alphabet`.
    alphabet:
        Alphabet describing the rows and columns of the returned distance matrix.
    matrix_alphabet:
        Alphabet describing the rows and columns of `substitution_matrix`.
    distance_cap:
        Maximum distance assigned to a mismatch. If `None`, mismatch distances are uncapped.
    distance_offset:
        Offset from which the substitution score is subtracted.

    Returns
    -------
    distance_matrix:
        Distance lookup matrix in the order specified by `alphabet`.
    """
    expected_shape = (len(matrix_alphabet), len(matrix_alphabet))
    if substitution_matrix.shape != expected_shape:
        raise ValueError(f"`substitution_matrix` must have shape {expected_shape} to match `matrix_alphabet`.")

    distance_matrix = np.zeros(expected_shape, dtype=np.int32)
    for i, aa1 in enumerate(matrix_alphabet):
        for j, aa2 in enumerate(matrix_alphabet):
            distance = 0 if aa1 == aa2 else distance_offset - substitution_matrix[i, j]
            if distance_cap is not None:
                distance = min(distance_cap, distance)
            distance_matrix[i, j] = distance
    return _map_matrix_to_alphabet(distance_matrix, matrix_alphabet, alphabet)


# fmt: off
_BLOSUM62_MATRIX = np.array(
    [
        # A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
        [ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0],  # A
        [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3],  # R
        [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3],  # N
        [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3],  # D
        [ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],  # C
        [-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2],  # Q
        [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2],  # E
        [ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3],  # G
        [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3],  # H
        [-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3],  # I
        [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1],  # L
        [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2],  # K
        [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1],  # M
        [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1],  # F
        [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2],  # P
        [ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2],  # S
        [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0],  # T
        [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3],  # W
        [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1],  # Y
        [ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4],  # V
    ],
    dtype=np.int32,
)
_TCRBLOSUM_ALPHA_MATRIX = np.array(
    [
        # A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
        [ 2, -1, -1, -1,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1,  0,  0, -1,  0, -1,  0],  # A
        [-1,  1,  0,  0,  1,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0, -1],  # R
        [-1,  0,  1,  0,  0,  0,  0,  0,  0, -1, -2,  1,  0,  0,  0,  0,  0,  0,  0, -2],  # N
        [-1,  0,  0,  1, -5,  0,  0,  0,  0, -1, -2,  0,  0,  0,  0,  0,  0,  0,  0, -1],  # D
        [ 0,  1,  0, -5,  2, -4, -4,  0, -2, -5,  0, -5, -4, -4, -4,  0, -6, -2, -5,  0],  # C
        [ 0,  0,  0,  0, -4,  2,  0,  0,  0, -1, -2,  1,  0,  0,  0,  0,  0,  0,  0, -2],  # Q
        [ 0,  0,  0,  0, -4,  0,  1,  0,  1, -1,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0],  # E
        [ 0,  0,  0,  0,  0,  0,  0,  1,  0, -2, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0],  # G
        [ 0,  0,  0,  0, -2,  0,  1,  0,  2,  0,  0, -1,  0,  0,  1,  0,  0,  1,  0,  0],  # H
        [-1,  0, -1, -1, -5, -1, -1, -2,  0,  3,  0, -1,  0,  0,  0,  0,  1, -1,  0,  0],  # I
        [-1, -1, -2, -2,  0, -2,  0, -1,  0,  0,  2, -4,  0,  1,  0, -1, -1, -1,  0,  0],  # L
        [-1,  0,  1,  0, -5,  1, -1, -1, -1, -1, -4,  3,  0, -3,  0, -2, -1, -2, -4, -3],  # K
        [-1,  0,  0,  0, -4,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0, -1,  0],  # M
        [-1,  0,  0,  0, -4,  0,  0,  0,  0,  0,  1, -3,  0,  1,  0,  0,  0,  0,  0,  0],  # F
        [ 0,  0,  0,  0, -4,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],  # P
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -2,  0,  0,  0,  1,  0,  0,  0, -1],  # S
        [-1,  0,  0,  0, -6,  0,  0,  0,  0,  1, -1, -1,  0,  0,  0,  0,  1,  0,  0,  0],  # T
        [ 0,  0,  0,  0, -2,  0,  0,  0,  1, -1, -1, -2,  0,  0,  0,  0,  0,  2,  0, -1],  # W
        [-1,  0,  0,  0, -5,  0,  0,  0,  0,  0,  0, -4, -1,  0,  0,  0,  0,  0,  1, -1],  # Y
        [ 0, -1, -2, -1,  0, -2,  0,  0,  0,  0,  0, -3,  0,  0,  0, -1,  0, -1, -1,  1],  # V
    ],
    dtype=np.int32,
)
_TCRBLOSUM_BETA_MATRIX = np.array(
    [
        # A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
        [ 0,  0,  0,  0, -5,  0, -1,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0, -1,  0],  # A
        [ 0,  2,  0,  0, -4, -1, -1,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0, -1,  0],  # R
        [ 0,  0,  1,  1, -4,  0,  0,  0,  0,  0, -1,  0,  0, -1,  0, -1,  0,  0,  0,  0],  # N
        [ 0,  0,  1,  1, -4,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0, -1,  0,  0,  0,  0],  # D
        [-5, -4, -4, -4,  2, -6, -5,  0, -3, -3, -5, -2, -1, -5, -4,  0, -5, -2, -5, -4],  # C
        [ 0, -1,  0,  0, -6,  2, -1, -1, -1,  0,  1, -1,  0, -2, -1, -2, -1,  0,  0, -1],  # Q
        [-1, -1,  0,  0, -5, -1,  2,  0, -1,  0, -1,  1,  0, -2,  0, -2,  1,  0, -1,  0],  # E
        [ 0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0, -1,  0],  # G
        [ 0,  0,  0,  0, -3, -1, -1,  0,  2,  0,  0, -1,  0,  2,  0, -1,  0,  0,  1,  0],  # H
        [ 0,  0,  0,  0, -3,  0,  0,  0,  0,  2,  0,  0,  2,  0,  0,  0,  0,  0,  0,  0],  # I
        [ 0,  0, -1,  0, -5,  1, -1,  0,  0,  0,  1,  0,  0,  0,  0, -1,  0,  0,  0,  0],  # L
        [ 0,  0,  0,  0, -2, -1,  1,  0, -1,  0,  0,  1,  0, -1,  0,  0,  0,  0, -1,  0],  # K
        [ 0,  0,  0,  0, -1,  0,  0,  0,  0,  2,  0,  0,  2,  0,  0,  0,  0,  0, -1,  0],  # M
        [-1, -1, -1, -1, -5, -2, -2, -1,  2,  0,  0, -1,  0,  2,  0, -2,  0,  0,  2, -1],  # F
        [ 0,  0,  0,  0, -4, -1,  0,  0,  0,  0,  0,  0,  0,  0,  1, -1,  0,  0, -1,  0],  # P
        [ 0,  0, -1, -1,  0, -2, -2,  0, -1,  0, -1,  0,  0, -2, -1,  1,  0,  0, -2,  0],  # S
        [ 0,  0,  0,  0, -5, -1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # T
        [ 0,  0,  0,  0, -2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0],  # W
        [-1, -1,  0,  0, -5,  0, -1, -1,  1,  0,  0, -1, -1,  2, -1, -2,  0,  0,  2, -1],  # Y
        [ 0,  0,  0,  0, -4, -1,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0, -1,  0],  # V
    ],
    dtype=np.int32,
)
# fmt: on


# fmt: off
_BLOSUM62_WITH_AMBIGUOUS_MATRIX = np.array(
    [
        # A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V   B   Z   X
        [ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0],  # A
        [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1],  # R
        [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1],  # N
        [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1],  # D
        [ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2],  # C
        [-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1],  # Q
        [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1],  # E
        [ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1],  # G
        [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1],  # H
        [-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1],  # I
        [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1],  # L
        [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1],  # K
        [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1],  # M
        [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1],  # F
        [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2],  # P
        [ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0],  # S
        [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0],  # T
        [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2],  # W
        [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1],  # Y
        [ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1],  # V
        [-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1],  # B
        [-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1],  # Z
        [ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1],  # X
    ],
    dtype=np.int32,
)
# fmt: on

BLOSUM62 = SubstitutionMatrix(CANONICAL_AA_ALPHABET, _BLOSUM62_MATRIX)
BLOSUM62_WITH_AMBIGUOUS = SubstitutionMatrix(AA_ALPHABET_WITH_AMBIGUOUS, _BLOSUM62_WITH_AMBIGUOUS_MATRIX)
TCRBLOSUM_ALPHA = SubstitutionMatrix(CANONICAL_AA_ALPHABET, _TCRBLOSUM_ALPHA_MATRIX)
TCRBLOSUM_BETA = SubstitutionMatrix(CANONICAL_AA_ALPHABET, _TCRBLOSUM_BETA_MATRIX)
