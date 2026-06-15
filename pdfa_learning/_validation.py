"""
Internal validation utilities for the pdfa-learning package.

This module centralises validation of observed sequences, alphabets,
transition-count matrices, ALERGIA significance parameters, and state
identifiers used during state merging. Its functions are private and are
intended for use by other modules within the package rather than as part
of the public interface.
"""

import numpy as np


def _validate_sequences(sequences):
    """
    Validates the input sequences for a PPTA.

    Parameters
    ----------
    sequences : iterable of str
        Sequences from which the alphabet is obtained.

    Returns
    -------
    list of str
        The validated list of sequences.

    Raises
    ------
    TypeError
        If sequences is None, is a single string, or contains
        non-string elements.
    ValueError
        If sequences is empty.
    """
    if sequences is None:
        raise TypeError("sequences must be an iterable of strings.")

    if isinstance(sequences, str):
        raise TypeError(
            "sequences must be an iterable of strings, not a single string."
        )

    sequences = list(sequences)

    if len(sequences) == 0:
        raise ValueError("sequences must contain at least one sequence.")

    if not all(isinstance(sequence, str) for sequence in sequences):
        raise TypeError("every sequence must be a string.")

    return sequences


def _validate_alphabet(alphabet):
    """
    Validates the input alphabet for a PPTA.

    Parameters
    ----------
    alphabet : iterable of str
        Alphabet to validate.

    Returns
    -------
    list of str
        The validated list of alphabet symbols.

    Raises
    ------
    TypeError
        If alphabet is None, is a single string, or contains
        non-string elements.
    ValueError
        If alphabet is empty, contains non-single-character strings,
        or contains duplicate symbols.
    """
    if alphabet is None:
        raise TypeError("alphabet must be an iterable of strings.")

    if isinstance(alphabet, str):
        raise TypeError(
            "alphabet must be an iterable of strings, not a single string."
        )

    alphabet = list(alphabet)

    if len(alphabet) == 0:
        raise ValueError("alphabet must contain at least one symbol.")

    if not all(isinstance(symbol, str) for symbol in alphabet):
        raise TypeError("every alphabet symbol must be a string.")

    if not all(len(symbol) == 1 for symbol in alphabet):
        raise ValueError("every alphabet symbol must contain exactly one character.")

    if len(alphabet) != len(set(alphabet)):
        raise ValueError("alphabet must not contain duplicate symbols.")

    return alphabet


def _validate_transition_matrix(transition_matrix, alphabet, states):
    """
    Validate a transition-count matrix and its associated dimensions.

    Parameters
    ----------
    transition_matrix : np.ndarray
        Three-dimensional transition-count matrix with shape
        ``(n_symbols, n_states, n_states)``.
    alphabet : collection
        Alphabet associated with the first dimension of the transition
        matrix.
    states : collection
        State identifiers associated with the final two dimensions of
        the transition matrix.

    Raises
    ------
    TypeError
        If transition_matrix is not a NumPy array.
    ValueError
        If transition_matrix is not three-dimensional, its final two
        dimensions are not equal, its dimensions do not agree with
        alphabet or states, or it contains non-finite or negative
        values.
    """
    if not isinstance(transition_matrix, np.ndarray):
        raise TypeError("transition_matrix must be a NumPy array.")

    if transition_matrix.ndim != 3:
        raise ValueError("transition_matrix must be three-dimensional.")

    if transition_matrix.shape[1] != transition_matrix.shape[2]:
        raise ValueError("The final two transition_matrix dimensions must be equal.")

    if transition_matrix.shape[0] != len(alphabet):
        raise ValueError(
            "The first transition_matrix dimension must equal len(alphabet)."
        )

    if transition_matrix.shape[1] != len(states):
        raise ValueError(
            "The transition_matrix state dimensions must equal len(states)."
        )

    if not np.all(np.isfinite(transition_matrix)):
        raise ValueError("transition_matrix must contain only finite values.")

    if np.any(transition_matrix < 0):
        raise ValueError("transition_matrix must not contain negative values.")


def _validate_alpha(alpha):
    """
    Validate the alpha parameter for Hoeffding bound calculations.

    Parameters
    ----------
    alpha : float
        Significance level used by the Hoeffding compatibility test.
        Smaller values generally permit more merges, while larger values 
        generally permit fewer merges. Must lie in the interval ``(0, 2]``.

    Raises
    ------
    TypeError
        If alpha is not a numeric type.
    ValueError
        If alpha is not in the range (0,2].
    """
    if not isinstance(alpha, (int, float, np.number)):
        raise TypeError("alpha must be numeric.")

    if alpha <= 0 or alpha > 2:
        raise ValueError("alpha must be in the range (0, 2].")


def _validate_states_for_merging(q1, q2, states):
    """
    Validate two states before performing a state merge.

    Parameters
    ----------
    q1 : int or str
        First state proposed for merging.
    q2 : int or str
        Second state proposed for merging.
    states : collection
        Valid state identifiers in the automaton.

    Raises
    ------
    ValueError
        If q1 or q2 is not present in states, if q1 and q2 refer
        to the same state, or if either state is the artificial
        initial state "*".
    """
    if q1 not in states:
        raise ValueError(f"q1 must be a valid state. Unknown state: {q1!r}.")

    if q2 not in states:
        raise ValueError(f"q2 must be a valid state. Unknown state: {q2!r}.")

    if q1 == q2:
        raise ValueError("q1 and q2 must refer to different states.")

    if q1 == "*" or q2 == "*":
        raise ValueError("The artificial initial state '*' cannot be merged.")