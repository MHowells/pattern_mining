import itertools as it

import numpy as np
from scipy.stats import norm

from ._validation import (
    _validate_alphabet,
    _validate_transition_matrix,
)
from .state_statistics import get_n


def probability_transition_matrix(transition_matrix, states, alphabet):
    """
    Convert a transition-count matrix into a probability transition matrix.

    Outgoing transition counts for each non-artificial state are divided by
    the number of sequences entering that state. The artificial initial
    state is normalised separately using its initial transition count.

    Parameters
    ----------
    transition_matrix : np.ndarray
        Transition-count matrix with shape
        ``(n_symbols, n_states, n_states)``.
    states : list
        State identifiers corresponding to the final two dimensions of
        transition_matrix.
    alphabet : iterable of str
        Alphabet corresponding to the first dimension of transition_matrix.

    Returns
    -------
    np.ndarray
        Probability transition matrix with the same shape as
        transition_matrix.

    Raises
    ------
    TypeError
        If transition_matrix is not a NumPy array or alphabet has an invalid
        type.
    ValueError
        If alphabet or transition_matrix has invalid contents or dimensions.
    """
    alphabet = _validate_alphabet(alphabet)

    _validate_transition_matrix(
        transition_matrix,
        alphabet,
        states,
    )

    p_mat = transition_matrix.copy().astype(float)
    for j in range(len(alphabet)):
        p_mat[j, 0, :] = transition_matrix[j, 0, :] / transition_matrix[0, 0, :].sum()
        for i in range(1, len(states)):
            p_mat[j, i, :] = transition_matrix[j, i, :] / get_n(
                states[i], transition_matrix, states
            )
    return p_mat


def probability_estimate_of_symbol(p_mat, symbol, alphabet):
    """
    Estimate the probability that a generated sequence contains a symbol.

    The probability is calculated separately for every possible starting
    state and represents encountering the symbol at least once before the
    sequence terminates.

    Parameters
    ----------
    p_mat : np.ndarray
        Probability transition matrix with shape
        ``(n_symbols, n_states, n_states)``.
    symbol : str
        Symbol whose occurrence probability is estimated.
    alphabet : list of str
        Alphabet corresponding to the first dimension of p_mat.

    Returns
    -------
    np.ndarray
        Estimated occurrence probability for each starting state.
    """
    matrix_index = alphabet.index(symbol)
    rho = np.sum(np.delete(p_mat, matrix_index, 0), axis=0)
    inverse = np.linalg.inv(np.identity(p_mat.shape[1]) - rho)
    p_symbol = np.sum(p_mat[matrix_index, :, :], axis=1)
    return np.matmul(inverse, p_symbol)


def probability_estimate_of_pattern(p_mat, pattern, alphabet):
    """
    Estimate the probability that a generated sequence contains a pattern.

    The symbols in pattern must be encountered in the specified order, but
    they do not need to occur consecutively. A probability is calculated
    for every possible starting state.

    Parameters
    ----------
    p_mat : np.ndarray
        Probability transition matrix with shape
        ``(n_symbols, n_states, n_states)``.
    pattern : str
        Non-empty ordered sequence of symbols to encounter.
    alphabet : list of str
        Alphabet corresponding to the first dimension of p_mat.

    Returns
    -------
    np.ndarray
        Estimated pattern probability for each starting state.
    """
    p_pattern = np.identity(p_mat.shape[1])
    for i in range(len(pattern)):
        if i != len(pattern) - 1:
            symbol = pattern[i]
            matrix_index = alphabet.index(symbol)
            rho = np.sum(np.delete(p_mat, matrix_index, 0), axis=0)
            inverse = np.linalg.inv(np.identity(p_mat.shape[1]) - rho)
            gamma = p_mat[matrix_index, :, :]
            p_pattern = np.matmul(p_pattern, inverse)
            p_pattern = np.matmul(p_pattern, gamma)
        else:
            symbol = pattern[i]
            p_pattern = np.matmul(
                p_pattern, probability_estimate_of_symbol(p_mat, symbol, alphabet)
            )
    return p_pattern


def probability_estimate_of_exact_sequence(p_mat, sequence, alphabet):
    """
    Estimate the probability of generating an exact sequence.

    The sequence must be emitted consecutively from the initial
    non-artificial state and the automaton must terminate immediately after
    the final symbol. The probability transition matrix is assumed to
    represent a deterministic automaton.

    Parameters
    ----------
    p_mat : np.ndarray
        Probability transition matrix with shape
        ``(n_symbols, n_states, n_states)``. The first state is assumed to
        be the artificial initial state.
    sequence : str
        Non-empty exact sequence whose probability is estimated.
    alphabet : list of str
        Alphabet corresponding to the first dimension of p_mat.

    Returns
    -------
    float
        Estimated probability of generating the exact sequence.
    """
    indices = [
        alphabet.index(symbol)
        for symbol in sequence
    ]

    p_mat = np.delete(p_mat, 0, axis=1)
    p_mat = np.delete(p_mat, 0, axis=2)

    first_transition = p_mat[indices[0], 0, :]
    p_est = np.sum(first_transition)

    if p_est == 0:
        return 0

    next_state = np.where(first_transition > 0)[0][0]

    if len(sequence) == 1:
        p_est *= 1 - np.sum(
            p_mat[:, next_state, :]
        )
        return p_est

    for i in range(1, len(sequence)):
        transitions = p_mat[
            indices[i],
            next_state,
            :,
        ]

        p_est *= np.sum(transitions)

        destinations = np.where(
            transitions > 0
        )[0]

        if destinations.size == 0:
            return 0

        next_state = destinations[0]

    p_est *= 1 - min(
        np.sum(p_mat[:, next_state, :]),
        1,
    )

    return p_est


def probability_sequence_contains_letter_at_distance_theta(
    p_mat, letter, theta, alphabet
):
    """
    Estimate the probability of encountering a letter after a fixed distance.

    For each starting state, the function calculates the probability that
    letter is emitted after exactly theta preceding transitions.

    Parameters
    ----------
    p_mat : np.ndarray
        Probability transition matrix with shape
        ``(n_symbols, n_states, n_states)``.
    letter : str
        Letter whose occurrence probability is estimated.
    theta : int
        Number of transitions between the starting state and the emission
        of letter. Expected to be non-negative.
    alphabet : list of str
        Alphabet corresponding to the first dimension of p_mat.

    Returns
    -------
    np.ndarray
        Estimated probability for each starting state.
    """
    if not isinstance(theta, (int, np.integer)):
        raise TypeError("theta must be an integer.")

    if theta < 0:
        raise ValueError("theta must be non-negative.")

    matrix_index = alphabet.index(letter)
    tau_theta = np.linalg.matrix_power(np.sum(p_mat, axis=0), theta)
    pi_symbol = np.sum(p_mat[matrix_index, :, :], axis=1)
    return np.matmul(tau_theta, pi_symbol)


def probability_to_encounter_a_pattern_at_a_distance_theta(
    p_mat,
    pattern,
    theta,
    alphabet,
):
    """
    Estimate the probability of encountering a pattern after a fixed distance.

    The first symbol of pattern is emitted after exactly theta preceding
    transitions. The remaining symbols must subsequently be encountered in
    the specified order, but do not need to occur consecutively.

    Parameters
    ----------
    p_mat : np.ndarray
        Probability transition matrix with shape
        ``(n_symbols, n_states, n_states)``.
    pattern : str
        Ordered pattern containing at least two symbols.
    theta : int
        Non-negative number of transitions before the first pattern symbol
        is emitted.
    alphabet : list of str
        Alphabet corresponding to the first dimension of p_mat.

    Returns
    -------
    np.ndarray
        Estimated probability for each starting state.

    Raises
    ------
    TypeError
        If theta is not an integer.
    ValueError
        If pattern contains fewer than two symbols, theta is negative, or a
        pattern symbol is not present in alphabet.
    """
    if len(pattern) < 2:
        raise ValueError(
            "pattern must contain at least two symbols. "
            "Use probability_sequence_contains_letter_at_distance_theta() "
            "for a single symbol."
        )

    if not isinstance(theta, (int, np.integer)):
        raise TypeError("theta must be an integer.")

    if theta < 0:
        raise ValueError("theta must be non-negative.")

    symbol = pattern[0]
    matrix_index = alphabet.index(symbol)

    gamma = p_mat[matrix_index, :, :]
    transition_matrix = np.sum(p_mat, axis=0)
    tau_theta = np.linalg.matrix_power(transition_matrix, theta)
    f_x_theta = np.matmul(tau_theta, gamma)

    remaining_pattern = pattern[1:]

    if len(remaining_pattern) > 1:
        est_of_pattern = probability_estimate_of_pattern(
            p_mat,
            remaining_pattern,
            alphabet,
        )

    else:
        est_of_pattern = probability_estimate_of_symbol(
            p_mat,
            remaining_pattern,
            alphabet,
        )

    return np.matmul(f_x_theta, est_of_pattern)


def proportion_constraint(
    p_mat,
    pattern,
    alphabet,
    sequences,
    alpha,
    p_value="pattern",
):
    """
    Determine whether an estimated probability satisfies a sampling threshold.

    The probability of an ordered pattern or exact sequence is compared
    with a normal-approximation threshold calculated from the probability,
    the number of observed sequences, and alpha.

    Parameters
    ----------
    p_mat : np.ndarray
        Probability transition matrix.
    pattern : str
        Pattern or exact sequence being evaluated.
    alphabet : list of str
        Alphabet associated with the probability transition matrix.
    sequences : collection
        Observed sequences. The number of sequences is used in the sampling
        threshold calculation.
    alpha : float
        Significance level used by the Hoeffding compatibility test.
        Smaller values generally permit more merges, while larger values 
        generally permit fewer merges. Must lie in the interval ``(0, 1)``
        for this test.
    p_value : {"pattern", "sequence"}, default="pattern"
        Whether to estimate the probability of an ordered pattern or an
        exact sequence.

    Returns
    -------
    bool
        True if the estimated probability is at least as large as the
        calculated sampling threshold; otherwise False.

    Raises
    ------
    TypeError
        If alpha is not numeric.
    ValueError
        If p_value is invalid, alpha is outside ``(0, 1)``, sequences is
        empty, or the estimated probability is outside ``[0, 1]``.
    """
    if p_value not in {"pattern", "sequence"}:
        raise ValueError("p_value must be either 'pattern' or 'sequence'.")

    if not isinstance(alpha, (int, float, np.number)):
        raise TypeError("alpha must be numeric.")

    if alpha <= 0 or alpha >= 1:
        raise ValueError(
            "alpha must be in the range (0, 1) for proportion_constraint()"
        )

    if len(sequences) == 0:
        raise ValueError("sequences must contain at least one observed sequence.")

    if p_value == "pattern":
        prob_value = probability_estimate_of_pattern(
            p_mat,
            pattern,
            alphabet,
        )[0]

    else:
        prob_value = probability_estimate_of_exact_sequence(
            p_mat,
            pattern,
            alphabet,
        )

    if not 0 <= prob_value <= 1:
        raise ValueError("The estimated probability must be between 0 and 1.")

    k = abs(norm.ppf(1 - alpha)) * (
        (prob_value * (1 - prob_value) / len(sequences)) ** 0.5
    )

    return bool(prob_value >= k)


def probability_sequence_contains_digram(
    p_mat,
    digram,
    alphabet,
):
    """
    Return the probability that a sequence contains a digram for each
    possible starting state.

    Parameters
    ----------
    p_mat : np.ndarray
        Probability transition matrix.
    digram : str
        Sequence containing exactly two symbols.
    alphabet : list of str
        Alphabet associated with the probability transition matrix.

    Returns
    -------
    np.ndarray
        Probability of encountering the digram from each starting state.

    Raises
    ------
    ValueError
        If digram does not contain exactly two symbols.
    """
    if len(digram) != 2:
        raise ValueError("digram must contain exactly two symbols.")

    symbol_1 = digram[0]
    symbol_2 = digram[1]

    matrix_index_1 = alphabet.index(symbol_1)
    matrix_index_2 = alphabet.index(symbol_2)

    rho = np.sum(
        np.delete(p_mat, matrix_index_1, 0),
        axis=0,
    )

    inverse = np.linalg.inv(np.identity(p_mat.shape[1]) - rho)

    p_symbol_1 = np.sum(
        p_mat[matrix_index_1, :, :],
        axis=1,
    )

    nonzero = []

    for state_index in range(p_mat.shape[1]):
        destinations = np.where(p_mat[matrix_index_1, state_index, :] > 0)[0]

        if np.any(p_mat[matrix_index_1, state_index, :] > 0):
            nonzero.append(destinations[0])
        else:
            nonzero.append(0)

    p_symbol_2 = np.zeros((1, p_mat.shape[1]))

    for i, emitted in enumerate(nonzero):
        p_symbol_2[0, i] = np.sum(
            p_mat[matrix_index_2, emitted, :],
            axis=0,
        )

    tau = np.multiply(
        p_symbol_1,
        p_symbol_2,
    )

    return np.matmul(tau, inverse)


def string_enumerator(alphabet, n):
    """
    Enumerate all strings over an alphabet up to length n.

    Parameters
    ----------
    alphabet : iterable of str
        Symbols used to construct the strings.
    n : int
        Maximum string length. Must be greater than zero.

    Returns
    -------
    list of str
        All possible strings with lengths from 1 to n.

    Raises
    ------
    TypeError
        If n is not an integer.
    ValueError
        If n is less than 1.
    """
    if not isinstance(n, int):
        raise TypeError("n must be an integer.")

    if n <= 0:
        raise ValueError("n must be greater than 0.")

    strings = []

    for i in range(1, n + 1):
        strings += ["".join(x) for x in it.product(alphabet, repeat=i)]

    return strings


def string_probabilities(p_mat, alphabet, strings):
    """
    Estimate the probability of each string in a collection.

    Parameters
    ----------
    p_mat : np.ndarray
        Probability transition matrix with shape
        ``(n_symbols, n_states, n_states)``.
    alphabet : list of str
        Alphabet corresponding to the first dimension of p_mat.
    strings : iterable of str
        Non-empty exact strings whose probabilities are estimated.

    Returns
    -------
    list of tuple
        Pairs containing each input string and its estimated probability.
    """
    return [
        (
            string,
            probability_estimate_of_exact_sequence(
                p_mat,
                string,
                alphabet,
            ),
        )
        for string in strings
    ]
