"""
State compatibility and merging operations for ALERGIA.

This module implements Hoeffding-bound compatibility testing, direct and
recursive state merging, and checks for nondeterministic transitions
created by a merge. It also provides helper functions for selecting state
pairs under the Carrasco and Oncina all-pairs procedure and identifying
red and blue states under the de la Higuera procedure.
"""

import numpy as np
from scipy.stats import norm

from ._validation import (
    _validate_alpha,
    _validate_alphabet,
    _validate_states_for_merging,
    _validate_transition_matrix,
)
from .state_statistics import (
    get_n,
    get_pi,
    get_pi_endpoint,
)


def hoeffding_bound(q1, q2, alpha, transition_matrix, alphabet, states):
    """
    Determine whether two states satisfy the Hoeffding compatibility bound.

    The transition probabilities associated with every symbol, together
    with the terminating probabilities, are compared for the two states.

    Parameters
    ----------
    q1 : int or str
        First state to compare.
    q2 : int or str
        Second state to compare.
    alpha : float
        Significance level used by the Hoeffding compatibility test.
        Smaller values generally permit more merges, while larger values 
        generally permit fewer merges. Must lie in the interval ``(0, 2]``.
    transition_matrix : np.ndarray
        Transition-count matrix with shape
        ``(n_symbols, n_states, n_states)``.
    alphabet : collection of str
        Alphabet associated with the first dimension of transition_matrix.
    states : list
        State identifiers corresponding to the final two dimensions of
        transition_matrix.

    Returns
    -------
    bool
        True if all symbol and terminating probabilities satisfy the
        Hoeffding bound; otherwise False.
    """
    alpha_constant = (np.log(2 / alpha) / 2) ** 0.5

    rhs = alpha_constant * (
        (1 / np.sqrt(get_n(q1, transition_matrix, states)))
        + (1 / np.sqrt(get_n(q2, transition_matrix, states)))
    )

    for z in range(len(alphabet)):
        lhs = abs(
            get_pi(q1, z, transition_matrix, states)
            - get_pi(q2, z, transition_matrix, states)
        )

        if lhs > rhs:
            return False

    lhs = abs(
        get_pi_endpoint(q1, transition_matrix, alphabet, states)
        - get_pi_endpoint(q2, transition_matrix, alphabet, states)
    )

    if lhs > rhs:
        return False

    return True


def merge_two_states(
    q1,
    q2,
    transition_matrix,
    states,
    alphabet,
    red_states=None,
):
    """
    Merge two states in a transition-count matrix.

    The merged state retains the identifier of whichever state appears
    first in the states list. The other state is removed.

    Parameters
    ----------
    q1 : int or str
        First state to merge.
    q2 : int or str
        Second state to merge.
    transition_matrix : np.ndarray
        Transition-count matrix with shape
        ``(n_symbols, n_states, n_states)``.
    states : list
        State identifiers corresponding to the final two dimensions of
        transition_matrix.
    alphabet : iterable of str
        Alphabet corresponding to the first dimension of transition_matrix.
    red_states : list, optional
        Red states to update after the merge.

    Returns
    -------
    transition_matrix_copy : np.ndarray
        Transition-count matrix after merging the states.
    states_copy : list
        Updated state identifiers.
    red_states_copy : list, optional
        Updated red states. Returned only when red_states is provided.

    Raises
    ------
    TypeError
        If transition_matrix is not a NumPy array or alphabet has an invalid
        type.
    ValueError
        If alphabet or transition_matrix has invalid contents or dimensions,
        q1 or q2 is not present in states, q1 and q2 are identical, or the
        artificial initial state is selected for merging.
    """
    alphabet = _validate_alphabet(alphabet)

    _validate_transition_matrix(
        transition_matrix,
        alphabet,
        states,
    )

    _validate_states_for_merging(
        q1,
        q2,
        states,
    )

    return _merge_two_states(
        q1,
        q2,
        transition_matrix,
        states,
        red_states=red_states,
    )


def _merge_two_states(q1, q2, transition_matrix, states, red_states=None):
    """
    Internal function that merges two states in a transition-count matrix.

    The merged state retains the identifier of whichever state appears
    first in the states list. The other state is removed. Assumes that
    the input parameters have already been validated.

    Parameters
    ----------
    q1 : int or str
        First state to merge.
    q2 : int or str
        Second state to merge.
    transition_matrix : np.ndarray
        Transition-count matrix with shape
        ``(n_symbols, n_states, n_states)``.
    states : list
        State identifiers corresponding to the final two dimensions of
        transition_matrix.
    red_states : list, optional
        Red states to update after the merge.

    Returns
    -------
    transition_matrix_copy : np.ndarray
        Transition-count matrix after merging the states.
    states_copy : list
        Updated state identifiers.
    red_states_copy : list, optional
        Updated red states. Returned only when red_states is provided.
    """
    i1 = states.index(q1)
    i2 = states.index(q2)

    which_min = min(i1, i2)
    which_max = max(i1, i2)

    surviving_state = states[which_min]
    removed_state = states[which_max]

    transition_matrix_copy = np.copy(transition_matrix)
    states_copy = states.copy()

    transition_matrix_copy[:, :, which_min] = (
        transition_matrix_copy[:, :, i1] + transition_matrix_copy[:, :, i2]
    )

    transition_matrix_copy = np.delete(
        transition_matrix_copy,
        which_max,
        axis=2,
    )

    transition_matrix_copy[:, which_min, :] = (
        transition_matrix_copy[:, i1, :] + transition_matrix_copy[:, i2, :]
    )

    transition_matrix_copy = np.delete(
        transition_matrix_copy,
        which_max,
        axis=1,
    )

    states_copy.remove(removed_state)

    if red_states is not None:
        updated_red_states = [
            surviving_state if state == removed_state else state for state in red_states
        ]

        red_states_copy = list(dict.fromkeys(updated_red_states))

        return (
            transition_matrix_copy,
            states_copy,
            red_states_copy,
        )

    return transition_matrix_copy, states_copy


def check_is_deterministic(transition_matrix, states, alphabet):
    """
    Identify nondeterministic state pairs in a transition matrix.

    A state is considered nondeterministic when it has positive transitions
    to more than one destination through the same alphabet symbol. For each
    such state-symbol combination, the first pair of destination states is
    returned.

    Parameters
    ----------
    transition_matrix : np.ndarray
        Transition-count matrix with shape
        ``(n_symbols, n_states, n_states)``.
    states : list
        State identifiers corresponding to the final two dimensions of
        transition_matrix.
    alphabet : collection of str
        Alphabet associated with the first dimension of transition_matrix.

    Returns
    -------
    list of tuple
        Pairs of destination state identifiers involved in
        nondeterministic transitions. An empty list indicates that the
        transition matrix is deterministic.
    """
    nondeterministic_pairs = []

    for a in range(len(alphabet)):
        rows = np.where((transition_matrix[a, :, :] > 0).sum(axis=1) > 1)[0]

        for row in rows:
            where_non_det = np.where(transition_matrix[a, row, :] > 0)[0]

            if len(where_non_det) > 2:
                nond_pairs = np.reshape(where_non_det[:2], (1, 2))

            else:
                nond_pairs = np.reshape(where_non_det, (1, 2))

            nondeterministic_pairs += [tuple(states[i] for i in r) for r in nond_pairs]

    return nondeterministic_pairs


def recursive_merge_two_states(
    q1,
    q2,
    transition_matrix,
    states,
    alpha,
    alphabet,
    red_states=None,
    output_level="Suppressed",
    method="carrasco",
):
    """
    Recursively merge states until the resulting automaton is deterministic.

    The initial pair is merged and any nondeterministic state pairs created
    by that merge are considered recursively. A recursive pair is merged
    only when it satisfies the Hoeffding compatibility bound.

    Parameters
    ----------
    q1 : int or str
        First state to merge.
    q2 : int or str
        Second state to merge.
    transition_matrix : np.ndarray
        Transition-count matrix with shape
        ``(n_symbols, n_states, n_states)``.
    states : list
        State identifiers corresponding to the final two dimensions of
        transition_matrix.
    alpha : float
        Significance level used by the Hoeffding compatibility test.
        Smaller values generally permit more merges, while larger values 
        generally permit fewer merges. Must lie in the interval ``(0, 2]``.
    alphabet : iterable of str
        Alphabet corresponding to the first dimension of transition_matrix.
    red_states : list, optional
        Red states to update during a de_la_higuera merge. Required when
        method is ``"de_la_higuera"``.
    output_level : {"Suppressed", "Truncated", "Full"}, default="Suppressed"
        Controls the amount of progress information printed during the
        state-merging process:

        - ``"Suppressed"``: prints no progress information.
        - ``"Truncated"``: prints the main state comparisons and merge
        outcomes, but omits details of the recursive merges.
        - ``"Full"``: prints all available progress information, including
        iteration numbers and the internal recursive merge process.
    method : {"carrasco", "de_la_higuera"}, default="carrasco" 
        Method used to test whether two states are compatible during the 
        state-merging process. ``"carrasco"`` follows the compatibility 
        test presented by Carrasco and Oncina [1]_, while ``"de_la_higuera"`` 
        follows the red-blue merge formulation presented by de la Higuera [2]_. 
        See the Notes and References sections for further information.

    Returns
    -------
    new_matrix : np.ndarray
        Transition-count matrix after the recursive merge attempt.
    new_states : list
        State identifiers corresponding to new_matrix.
    recursive_merge : bool
        True if the complete recursive merge succeeds; otherwise False.
    red_states_result : list, optional
        Red states produced by the merge. Returned only when method is
        ``"de_la_higuera"``.

    Raises
    ------
    TypeError
        If alpha is not numeric, transition_matrix is not a NumPy array, or
        alphabet has an invalid type.
    ValueError
        If output_level or method is invalid, red_states is not provided for the
        de_la_higuera method, alpha is outside ``(0, 2]``, alphabet or
        transition_matrix is invalid, either state is unknown, the states are
        identical, or the artificial initial state is selected.
    
    Notes 
    ----- 
    The available methods implement alternative formulations of the 
    statistical compatibility test used when determining whether two 
    states may be merged.

    References 
    ---------- 
    .. [1] Carrasco, R. C. and Oncina, J. (1994). "Learning Stochastic 
    Regular Grammars by Means of a State Merging Method." In *Grammatical 
    Inference and Applications*, pp. 139–152. 
    .. [2] de la Higuera, C. (2010). *Grammatical Inference: Learning 
    Automata and Grammars*. Cambridge University Press.
    """
    valid_outputs = {"Suppressed", "Truncated", "Full"}

    if output_level not in valid_outputs:
        raise ValueError("output_level must be 'Suppressed', 'Truncated', or 'Full'.")

    valid_methods = {"carrasco", "de_la_higuera"}

    if method not in valid_methods:
        raise ValueError("method must be either 'carrasco' or 'de_la_higuera'.")

    if method == "de_la_higuera" and red_states is None:
        raise ValueError("red_states must be provided when method='de_la_higuera'.")

    _validate_alpha(alpha)

    alphabet = _validate_alphabet(alphabet)

    _validate_transition_matrix(
        transition_matrix,
        alphabet,
        states,
    )

    _validate_states_for_merging(
        q1,
        q2,
        states,
    )

    return _recursive_merge_two_states(
        q1,
        q2,
        transition_matrix,
        states,
        alpha,
        alphabet,
        red_states=red_states,
        output_level=output_level,
        method=method,
    )


def _recursive_merge_two_states(
    q1,
    q2,
    transition_matrix,
    states,
    alpha,
    alphabet,
    red_states=None,
    output_level="Suppressed",
    method="carrasco",
):
    """
    Recursively merge previously validated states until determinism is restored.

    The function assumes that q1, q2, transition_matrix, states, alpha, and
    alphabet have already been validated. It attempts to resolve each
    nondeterministic pair created by the initial merge using the Hoeffding
    compatibility bound.

    Parameters
    ----------
    q1 : int or str
        First state to merge.
    q2 : int or str
        Second state to merge.
    transition_matrix : np.ndarray
        Transition-count matrix with shape
        ``(n_symbols, n_states, n_states)``.
    states : list
        State identifiers corresponding to the final two dimensions of
        transition_matrix.
    alpha : float
        Significance level used by the Hoeffding compatibility test.
        Smaller values generally permit more merges, while larger values 
        generally permit fewer merges. Must lie in the interval ``(0, 2]``.
    alphabet : collection of str
        Alphabet corresponding to the first dimension of transition_matrix.
    red_states : list, optional
        Red states to update during a de_la_higuera merge. Required when
        method is ``"de_la_higuera"``.
    output_level : {"Suppressed", "Truncated", "Full"}, default="Suppressed"
        Amount of progress information printed.
    method : {"carrasco", "de_la_higuera"}, default="carrasco"
        State-merging methodology to use.

    Returns
    -------
    new_matrix : np.ndarray
        Transition-count matrix after the recursive merge attempt.
    new_states : list
        State identifiers corresponding to new_matrix.
    recursive_merge : bool
        True if the complete recursive merge succeeds; otherwise False.
    red_states_result : list, optional
        Red states produced by the merge. Returned only when method is
        ``"de_la_higuera"``.
    """
    if method == "carrasco":
        initial_transition_matrix = np.copy(transition_matrix)
        initial_states = states.copy()

        new_matrix, new_states = _merge_two_states(
            q1,
            q2,
            transition_matrix,
            states,
        )

        non_det_pairs = check_is_deterministic(
            new_matrix,
            new_states,
            alphabet,
        )

        if len(non_det_pairs) > 0 and output_level == "Full":
            print(
                "Merging of states",
                (q1, q2),
                "results in non-deterministic pairs:",
                non_det_pairs,
            )

        recursive_merge = True

        while non_det_pairs:
            if hoeffding_bound(
                non_det_pairs[0][0],
                non_det_pairs[0][1],
                alpha,
                new_matrix,
                alphabet,
                new_states,
            ):
                if output_level == "Full":
                    print(
                        "Successfully merged states",
                        non_det_pairs[0],
                        "into a deterministic state.",
                    )

                new_matrix, new_states = _merge_two_states(
                    non_det_pairs[0][0],
                    non_det_pairs[0][1],
                    new_matrix,
                    new_states,
                )

                non_det_pairs = check_is_deterministic(
                    new_matrix,
                    new_states,
                    alphabet,
                )

                if len(non_det_pairs) > 0 and output_level == "Full":
                    print(
                        "Merging of previous non-deterministic pair "
                        "results in non-deterministic pairs:",
                        non_det_pairs,
                    )

            else:
                recursive_merge = False
                return (
                    initial_transition_matrix,
                    initial_states,
                    recursive_merge,
                )

        return new_matrix, new_states, recursive_merge

    initial_transition_matrix = np.copy(transition_matrix)
    initial_states = states.copy()
    initial_red_states = red_states.copy()

    new_matrix, new_states, red_states = _merge_two_states(
        q1,
        q2,
        transition_matrix,
        states,
        red_states=red_states,
    )

    non_det_pairs = check_is_deterministic(
        new_matrix,
        new_states,
        alphabet,
    )

    if len(non_det_pairs) > 0 and output_level == "Full":
        print(
            "Merging of states",
            (q1, q2),
            "results in non-deterministic pairs:",
            non_det_pairs,
        )

    recursive_merge = True

    while non_det_pairs:
        pair = non_det_pairs[0]

        if hoeffding_bound(
            pair[0],
            pair[1],
            alpha,
            new_matrix,
            alphabet,
            new_states,
        ):
            if output_level == "Full":
                print(
                    "Successfully merged states",
                    pair,
                    "into a deterministic state.",
                )

            new_matrix, new_states, red_states = _merge_two_states(
                pair[0],
                pair[1],
                new_matrix,
                new_states,
                red_states=red_states,
            )

            non_det_pairs = check_is_deterministic(
                new_matrix,
                new_states,
                alphabet,
            )

            if len(non_det_pairs) > 0 and output_level == "Full":
                print(
                    "Merging of previous non-deterministic pair "
                    "results in non-deterministic pairs:",
                    non_det_pairs,
                )

        else:
            recursive_merge = False
            return (
                initial_transition_matrix,
                initial_states,
                recursive_merge,
                initial_red_states,
            )

    return new_matrix, new_states, recursive_merge, red_states


def get_blue_states(transition_matrix, red_states, states):
    """
    Return the blue states associated with a set of red states.

    Blue states are non-red states that can be reached directly through a
    positive outgoing transition from at least one red state.

    Parameters
    ----------
    transition_matrix : np.ndarray
        Transition-count matrix with shape
        ``(n_symbols, n_states, n_states)``.
    red_states : list
        State identifiers currently classified as red.
    states : list
        State identifiers corresponding to the final two dimensions of
        transition_matrix.

    Returns
    -------
    list
        Sorted blue-state identifiers.
    """
    blue_states = []

    for q in red_states:
        blue_states += [
            states[x]
            for x in list(np.where(transition_matrix[:, states.index(q), :] > 0)[1])
        ]

    blue_states = [x for x in blue_states if x not in red_states]

    return sorted(blue_states)


def get_pairs_to_check(states):
    """
    Return all state pairs to consider for merging.

    The artificial initial state ``"*"`` is excluded. Each unordered pair
    of the remaining states is included once.

    Parameters
    ----------
    states : list
        State identifiers, including the artificial initial state ``"*"``.

    Returns
    -------
    list of tuple
        State pairs to consider for merging.
    """
    state_numbers = states.copy()
    state_numbers.remove("*")

    to_check = [
        (state_numbers[j], state_numbers[i])
        for j in range(len(state_numbers))
        for i in range(0, j)
    ]

    return to_check