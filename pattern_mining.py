import numpy as np
import graphviz
from scipy.stats import norm
import itertools as it


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
        raise TypeError(
            "sequences must be an iterable of strings."
        )

    if isinstance(sequences, str):
        raise TypeError(
            "sequences must be an iterable of strings, "
            "not a single string."
        )

    sequences = list(sequences)

    if len(sequences) == 0:
        raise ValueError(
            "sequences must contain at least one sequence."
        )

    if not all(
        isinstance(sequence, str)
        for sequence in sequences
    ):
        raise TypeError(
            "every sequence must be a string."
        )

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
        raise TypeError(
            "alphabet must be an iterable of strings."
        )

    if isinstance(alphabet, str):
        raise TypeError(
            "alphabet must be an iterable of strings, "
            "not a single string."
        )

    alphabet = list(alphabet)

    if len(alphabet) == 0:
        raise ValueError(
            "alphabet must contain at least one symbol."
        )

    if not all(
        isinstance(symbol, str)
        for symbol in alphabet
    ):
        raise TypeError(
            "every alphabet symbol must be a string."
        )

    if not all(
        len(symbol) == 1
        for symbol in alphabet
    ):
        raise ValueError(
            "every alphabet symbol must contain exactly one character."
        )

    if len(alphabet) != len(set(alphabet)):
        raise ValueError(
            "alphabet must not contain duplicate symbols."
        )

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
        raise TypeError(
            "transition_matrix must be a NumPy array."
        )

    if transition_matrix.ndim != 3:
        raise ValueError(
            "transition_matrix must be three-dimensional."
        )

    if transition_matrix.shape[1] != transition_matrix.shape[2]:
        raise ValueError(
            "The final two transition_matrix dimensions must be equal."
        )

    if transition_matrix.shape[0] != len(alphabet):
        raise ValueError(
            "The first transition_matrix dimension must equal len(alphabet)."
        )

    if transition_matrix.shape[1] != len(states):
        raise ValueError(
            "The transition_matrix state dimensions must equal len(states)."
        )

    if not np.all(np.isfinite(transition_matrix)):
        raise ValueError(
            "transition_matrix must contain only finite values."
        )

    if np.any(transition_matrix < 0):
        raise ValueError(
            "transition_matrix must not contain negative values."
        )


def _validate_alpha(alpha):
    """
    Validate the alpha parameter for Hoeffding bound calculations.

    Parameters
    ----------
    alpha : float
        Significance level for the Hoeffding bound.

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
        raise ValueError(
            "alpha must be in the range (0, 2]."
        )


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
        raise ValueError(
            f"q1 must be a valid state. Unknown state: {q1!r}."
        )

    if q2 not in states:
        raise ValueError(
            f"q2 must be a valid state. Unknown state: {q2!r}."
        )

    if q1 == q2:
        raise ValueError(
            "q1 and q2 must refer to different states."
        )

    if q1 == "*" or q2 == "*":
        raise ValueError(
            "The artificial initial state '*' cannot be merged."
        )
    

def get_alphabet(sequences):
    """
    Returns the sorted alphabet across all sequences of a PPTA.

    Parameters
    ----------
    sequences : iterable of str
        Sequences from which the alphabet is obtained.

    Returns
    -------
    list of str
        Sorted unique symbols appearing in the sequences.

    Raises
    ------
    TypeError
        If sequences is None, is a single string, or contains
        non-string elements.
    ValueError
        If sequences is empty.
    """
    sequences = _validate_sequences(sequences)
    return sorted(set("".join(sequences)))


def get_state_paths(sequences, build="breadth"):
    """
    Return the state paths within a PPTA.

    Parameters
    ----------
    sequences : iterable of str
        Sequences used to construct the PPTA.
    build : {"breadth", "depth"}, default="breadth"
        Order in which the state paths are constructed.

    Returns
    -------
    list of str
        State paths in the requested construction order.

    Raises
    ------
    TypeError
        If sequences is None, is a single string, or contains
        non-string elements.
    ValueError
        If sequences is empty or build is not "breadth" or "depth".
    """
    sequences = _validate_sequences(sequences)

    if build not in {"breadth", "depth"}:
        raise ValueError(
            "build must be either 'breadth' or 'depth'."
        )

    if build == "breadth":
        all_paths = [""]
        all_ordered = [""]
        current_node = all_paths[0]
        tracker = 0

        while tracker < len(all_paths):
            this_iter = sorted(
                list(
                    set(
                        [
                            x[: len(current_node) + 1]
                            for x in sequences
                            if len(x) > len(current_node) 
                            and x.startswith(current_node)
                        ]
                    )
                )
            )

            for j in range(len(this_iter)):
                all_paths.append(this_iter[j])
                all_ordered.insert(
                    all_ordered.index(current_node) + 1 + j, 
                    this_iter[j],
                )

            tracker += 1

            if tracker == len(all_paths):
                break

            current_node = all_ordered[
                all_ordered.index(current_node) + 1
            ]

        return all_paths
    
    else:
        all_paths = [""]
        current_node = all_paths[0]
        tracker = 0

        while tracker < len(all_paths):
            this_iter = sorted(
                list(
                    set(
                        [
                            x[: len(current_node) + 1]
                            for x in sequences
                            if (
                                len(x) > len(current_node) 
                                and x.startswith(current_node)
                            )
                        ]
                    )
                )
            )

            for j in range(len(this_iter)):
                all_paths.append(this_iter[j])

            tracker += 1

            if tracker == len(all_paths):
                break

            current_node = all_paths[
                all_paths.index(current_node) + 1
            ]

        return all_paths


def get_transition_matrix(sequences, alphabet, build="breadth"):
    """
    Return the transition-count matrix of a PPTA.

    Parameters
    ----------
    sequences : iterable of str
        Sequences used to construct the PPTA.
    alphabet : iterable of str
        Symbols represented in the transition matrix.
    build : {"breadth", "depth"}, default="breadth"
        Order in which the PPTA states are constructed.

    Returns
    -------
    np.ndarray
        Three-dimensional transition-count matrix with shape
        (number of symbols, number of states, number of states).

    Raises
    ------
    TypeError
        If sequences or alphabet has an invalid type.
    ValueError
        If sequences or alphabet is empty, build is invalid, alphabet
        contains duplicate symbols, or alphabet omits observed symbols.
    """
    alphabet = _validate_alphabet(alphabet)

    observed_symbols = set("".join(sequences))
    missing_symbols = observed_symbols - set(alphabet)

    if missing_symbols:
        raise ValueError(
            "alphabet is missing symbols found in sequences: "
            f"{sorted(missing_symbols)}."
        )

    all_nodes = get_state_paths(sequences, build=build)
    all_nodes.insert(0, "*")

    n = len(all_nodes)

    pathway_matrix = np.zeros((len(alphabet), n, n), dtype=int)
    pathway_matrix[0, 0, 1] = len(sequences)

    for i in range(1, n):
        for k in range(len(alphabet)):
            next_node = all_nodes[i] + alphabet[k]

            if next_node in all_nodes:
                pathway_matrix[k, i, all_nodes.index(next_node)] = len(
                    [x for x in sequences if x.startswith(next_node)]
                )

    return pathway_matrix


def get_initial_states(sequences):
    """
    Return the initial state identifiers of a PPTA.

    The artificial initial state ``"*"`` is followed by an integer
    identifier for each prefix state in the PPTA.

    Parameters
    ----------
    sequences : iterable of str
        Sequences used to construct the PPTA.

    Returns
    -------
    list
        State identifiers, beginning with the artificial initial state
        ``"*"``.

    Raises
    ------
    TypeError
        If sequences is None, is a single string, or contains non-string
        elements.
    ValueError
        If sequences is empty.
    """
    states = list(range(len(get_state_paths(sequences))))
    states.insert(0, "*")
    
    return states


def get_n(q, pathway_matrix, states):
    """
    Return the number of sequences entering a state.

    This calculates ``n(q)`` by summing all transitions entering the
    specified state.

    Parameters
    ----------
    q : int or str
        State identifier.
    pathway_matrix : np.ndarray
        Transition-count matrix with shape
        ``(n_symbols, n_states, n_states)``.
    states : list
        State identifiers corresponding to the final two dimensions of
        pathway_matrix.

    Returns
    -------
    int or float
        Number of sequences entering the state.
    """
    i = states.index(q)
    return pathway_matrix[:, :, i].sum()


def get_endpoint(q, pathway_matrix, states):
    """
    Return the number of sequences terminating at a state.

    The terminating count is calculated as the number of sequences entering
    the state minus the number leaving it.

    Parameters
    ----------
    q : int or str
        State identifier.
    pathway_matrix : np.ndarray
        Transition-count matrix with shape
        ``(n_symbols, n_states, n_states)``.
    states : list
        State identifiers corresponding to the final two dimensions of
        pathway_matrix.

    Returns
    -------
    int or float
        Number of sequences terminating at the state.
    """
    i = states.index(q)
    return pathway_matrix[:, :, i].sum() - pathway_matrix[:, i, :].sum()


def get_pi(q, z, pathway_matrix, states):
    """
    Return the probability of leaving a state via a symbol.

    This calculates ``pi(q, z)`` for the symbol represented by index z in
    the first dimension of the transition-count matrix.

    Parameters
    ----------
    q : int or str
        State identifier.
    z : int
        Index of the symbol in the first dimension of pathway_matrix.
    pathway_matrix : np.ndarray
        Transition-count matrix with shape
        ``(n_symbols, n_states, n_states)``.
    states : list
        State identifiers corresponding to the final two dimensions of
        pathway_matrix.

    Returns
    -------
    float
        Probability of leaving state q via the indexed symbol.
    """
    i = states.index(q)
    return pathway_matrix[z, i, :].sum() / get_n(q, pathway_matrix, states)


def get_pi_endpoint(q, pathway_matrix, alphabet, states):
    """
    Return the probability of terminating at a state.

    The terminating probability is one minus the sum of the probabilities
    of leaving the state through each symbol in the alphabet.

    Parameters
    ----------
    q : int or str
        State identifier.
    pathway_matrix : np.ndarray
        Transition-count matrix with shape
        ``(n_symbols, n_states, n_states)``.
    alphabet : collection of str
        Alphabet associated with the first dimension of pathway_matrix.
    states : list
        State identifiers corresponding to the final two dimensions of
        pathway_matrix.

    Returns
    -------
    float
        Probability of terminating at state q.
    """
    return 1 - sum(
        get_pi(
            q, 
            z, 
            pathway_matrix, 
            states,
        ) for z in range(len(alphabet))
    )


def hoeffding_bound(q1, q2, alpha, pathway_matrix, alphabet, states):
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
        Significance level used to calculate the Hoeffding bound. Expected
        to be in the range ``(0, 2]``.
    pathway_matrix : np.ndarray
        Transition-count matrix with shape
        ``(n_symbols, n_states, n_states)``.
    alphabet : collection of str
        Alphabet associated with the first dimension of pathway_matrix.
    states : list
        State identifiers corresponding to the final two dimensions of
        pathway_matrix.

    Returns
    -------
    bool
        True if all symbol and terminating probabilities satisfy the
        Hoeffding bound; otherwise False.
    """
    alpha_constant = (np.log(2 / alpha) / 2) ** 0.5

    rhs = alpha_constant * (
        (1 / np.sqrt(get_n(q1, pathway_matrix, states)))
        + (1 / np.sqrt(get_n(q2, pathway_matrix, states)))
    )

    for z in range(len(alphabet)):
        lhs = abs(
            get_pi(q1, z, pathway_matrix, states)
            - get_pi(q2, z, pathway_matrix, states)
        )

        if lhs > rhs:
            return False
        
    lhs = abs(
        get_pi_endpoint(q1, pathway_matrix, alphabet, states)
        - get_pi_endpoint(q2, pathway_matrix, alphabet, states)
    )

    if lhs > rhs:
        return False
    
    return True


def merge_two_states(
    q1, 
    q2, 
    pathway_matrix, 
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
    pathway_matrix : np.ndarray
        Transition-count matrix with shape
        ``(n_symbols, n_states, n_states)``.
    states : list
        State identifiers corresponding to the final two dimensions of
        pathway_matrix.
    alphabet : iterable of str
        Alphabet corresponding to the first dimension of pathway_matrix.
    red_states : list, optional
        Red states to update after the merge.

    Returns
    -------
    pathway_matrix_copy : np.ndarray
        Transition-count matrix after merging the states.
    states_copy : list
        Updated state identifiers.
    red_states_copy : list, optional
        Updated red states. Returned only when red_states is provided.

    Raises
    ------
    TypeError
        If pathway_matrix is not a NumPy array or alphabet has an invalid
        type.
    ValueError
        If alphabet or pathway_matrix has invalid contents or dimensions,
        q1 or q2 is not present in states, q1 and q2 are identical, or the
        artificial initial state is selected for merging.
    """
    alphabet = _validate_alphabet(alphabet)

    _validate_transition_matrix(
        pathway_matrix,
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
        pathway_matrix,
        states,
        red_states=red_states,
    )


def _merge_two_states(q1, q2, pathway_matrix, states, red_states=None):
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
    pathway_matrix : np.ndarray
        Transition-count matrix with shape
        ``(n_symbols, n_states, n_states)``.
    states : list
        State identifiers corresponding to the final two dimensions of
        pathway_matrix.
    red_states : list, optional
        Red states to update after the merge.

    Returns
    -------
    pathway_matrix_copy : np.ndarray
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

    pathway_matrix_copy = np.copy(pathway_matrix)
    states_copy = states.copy()

    pathway_matrix_copy[:, :, which_min] = (
        pathway_matrix_copy[:, :, i1]
        + pathway_matrix_copy[:, :, i2]
    )

    pathway_matrix_copy = np.delete(
        pathway_matrix_copy,
        which_max,
        axis=2,
    )

    pathway_matrix_copy[:, which_min, :] = (
        pathway_matrix_copy[:, i1, :]
        + pathway_matrix_copy[:, i2, :]
    )

    pathway_matrix_copy = np.delete(
        pathway_matrix_copy,
        which_max,
        axis=1,
    )

    states_copy.remove(removed_state)

    if red_states is not None:
        updated_red_states = [
            surviving_state if state == removed_state else state
            for state in red_states
        ]

        red_states_copy = list(
            dict.fromkeys(updated_red_states)
        )

        return (
            pathway_matrix_copy,
            states_copy,
            red_states_copy,
        )

    return pathway_matrix_copy, states_copy


def check_is_deterministic(pathway_matrix, states, alphabet):
    """
    Identify nondeterministic state pairs in a transition matrix.

    A state is considered nondeterministic when it has positive transitions
    to more than one destination through the same alphabet symbol. For each
    such state-symbol combination, the first pair of destination states is
    returned.

    Parameters
    ----------
    pathway_matrix : np.ndarray
        Transition-count matrix with shape
        ``(n_symbols, n_states, n_states)``.
    states : list
        State identifiers corresponding to the final two dimensions of
        pathway_matrix.
    alphabet : collection of str
        Alphabet associated with the first dimension of pathway_matrix.

    Returns
    -------
    list of tuple
        Pairs of destination state identifiers involved in
        nondeterministic transitions. An empty list indicates that the
        transition matrix is deterministic.
    """
    nondeterministic_pairs = []

    for a in range(len(alphabet)):
        rows = np.where(
            (pathway_matrix[a, :, :] > 0).sum(axis=1) > 1
        )[0]

        for row in rows:
            where_non_det = np.where(pathway_matrix[a, row, :] > 0)[0]

            if len(where_non_det) > 2:
                nond_pairs = np.reshape(where_non_det[:2], (1, 2))

            else:
                nond_pairs = np.reshape(where_non_det, (1, 2))

            nondeterministic_pairs += [
                tuple(states[i] for i in r) for r in nond_pairs
            ]

    return nondeterministic_pairs


def recursive_merge_two_states(
    q1, 
    q2, 
    pathway_matrix, 
    states, 
    alpha, 
    alphabet, 
    red_states=None, 
    output="Suppressed", 
    method="Carrasco",
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
    pathway_matrix : np.ndarray
        Transition-count matrix with shape
        ``(n_symbols, n_states, n_states)``.
    states : list
        State identifiers corresponding to the final two dimensions of
        pathway_matrix.
    alpha : float
        Significance level used by the Hoeffding compatibility test.
    alphabet : iterable of str
        Alphabet corresponding to the first dimension of pathway_matrix.
    red_states : list, optional
        Red states to update during a Higuera merge. Required when
        method is ``"Higuera"``.
    output : {"Suppressed", "Truncated", "Full"}, default="Suppressed"
        Amount of progress information printed.
    method : {"Carrasco", "Higuera"}, default="Carrasco"
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
        ``"Higuera"``.

    Raises
    ------
    TypeError
        If alpha is not numeric, pathway_matrix is not a NumPy array, or
        alphabet has an invalid type.
    ValueError
        If output or method is invalid, red_states is not provided for the
        Higuera method, alpha is outside ``(0, 2]``, alphabet or
        pathway_matrix is invalid, either state is unknown, the states are
        identical, or the artificial initial state is selected.
    """
    valid_outputs = {"Suppressed", "Truncated", "Full"}

    if output not in valid_outputs:
        raise ValueError(
            "output must be 'Suppressed', 'Truncated', or 'Full'."
        )

    valid_methods = {"Carrasco", "Higuera"}

    if method not in valid_methods:
        raise ValueError(
            "method must be either 'Carrasco' or 'Higuera'."
        )

    if method == "Higuera" and red_states is None:
        raise ValueError(
            "red_states must be provided when method='Higuera'."
        )

    _validate_alpha(alpha)
    
    alphabet = _validate_alphabet(alphabet)

    _validate_transition_matrix(
        pathway_matrix,
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
        pathway_matrix,
        states,
        alpha,
        alphabet,
        red_states=red_states,
        output=output,
        method=method,
    )


def _recursive_merge_two_states(
    q1, 
    q2, 
    pathway_matrix, 
    states, 
    alpha, 
    alphabet, 
    red_states=None, 
    output="Suppressed", 
    method="Carrasco",
):
    """
    Recursively merge previously validated states until determinism is restored.

    The function assumes that q1, q2, pathway_matrix, states, alpha, and
    alphabet have already been validated. It attempts to resolve each
    nondeterministic pair created by the initial merge using the Hoeffding
    compatibility bound.

    Parameters
    ----------
    q1 : int or str
        First state to merge.
    q2 : int or str
        Second state to merge.
    pathway_matrix : np.ndarray
        Transition-count matrix with shape
        ``(n_symbols, n_states, n_states)``.
    states : list
        State identifiers corresponding to the final two dimensions of
        pathway_matrix.
    alpha : float
        Significance level used by the Hoeffding compatibility test.
    alphabet : collection of str
        Alphabet corresponding to the first dimension of pathway_matrix.
    red_states : list, optional
        Red states to update during a Higuera merge. Required when
        method is ``"Higuera"``.
    output : {"Suppressed", "Truncated", "Full"}, default="Suppressed"
        Amount of progress information printed.
    method : {"Carrasco", "Higuera"}, default="Carrasco"
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
        ``"Higuera"``.
    """
    if method == "Carrasco":
        initial_pathway_matrix = np.copy(pathway_matrix)
        initial_states = states.copy()

        new_matrix, new_states = _merge_two_states(
            q1, 
            q2, 
            pathway_matrix, 
            states,
        )

        non_det_pairs = check_is_deterministic(
            new_matrix, 
            new_states, 
            alphabet,
        )

        if len(non_det_pairs) > 0 and output == "Full":
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
                if output == "Full":
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

                if len(non_det_pairs) > 0 and output == "Full":
                    print(
                        "Merging of previous non-deterministic pair "
                        "results in non-deterministic pairs:",
                        non_det_pairs,
                    )

            else:
                recursive_merge = False
                return (
                    initial_pathway_matrix, 
                    initial_states, 
                    recursive_merge,
                )
            
        return new_matrix, new_states, recursive_merge
    
    initial_pathway_matrix = np.copy(pathway_matrix)
    initial_states = states.copy()
    initial_red_states = red_states.copy()

    new_matrix, new_states, red_states = _merge_two_states(
        q1, 
        q2, 
        pathway_matrix, 
        states, 
        red_states=red_states,
    )

    non_det_pairs = check_is_deterministic(
        new_matrix, 
        new_states, 
        alphabet,
    )

    if len(non_det_pairs) > 0 and output == "Full":
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
            if output == "Full":
                print(
                    "Successfully merged states",
                    pair,
                    "into a deterministic state.",
                )

            if any(x in red_states for x in pair):
                if set(pair).issubset(set(red_states)):
                    red_states = [
                        min(pair) if x == max(pair) else x 
                        for x in red_states
                    ]
                else:
                    red_states = [
                        min(pair) if x in pair else x 
                        for x in red_states
                    ]

            new_matrix, new_states = _merge_two_states(
                pair[0], 
                pair[1], 
                new_matrix, 
                new_states,
            )

            non_det_pairs = check_is_deterministic(
                new_matrix, 
                new_states, 
                alphabet,
            )

            if len(non_det_pairs) > 0 and output == "Full":
                print(
                    "Merging of previous non-deterministic pair "
                    "results in non-deterministic pairs:",
                    non_det_pairs,
                )

        else:
            recursive_merge = False
            return (
                initial_pathway_matrix, 
                initial_states, 
                recursive_merge, 
                red_states,
            )
        
    return new_matrix, new_states, recursive_merge, red_states


def get_blue_states(pathway_matrix, red_states, states):
    """
    Return the blue states associated with a set of red states.

    Blue states are non-red states that can be reached directly through a
    positive outgoing transition from at least one red state.

    Parameters
    ----------
    pathway_matrix : np.ndarray
        Transition-count matrix with shape
        ``(n_symbols, n_states, n_states)``.
    red_states : list
        State identifiers currently classified as red.
    states : list
        State identifiers corresponding to the final two dimensions of
        pathway_matrix.

    Returns
    -------
    list
        Sorted blue-state identifiers.
    """
    blue_states = []

    for q in red_states:
        blue_states += [
            states[x]
            for x in list(
                np.where(pathway_matrix[:, states.index(q), :] > 0)[1]
            )
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


def alergia(
    transition_matrix, 
    states, 
    alphabet, 
    alpha, 
    output="Suppressed", 
    method="Carrasco",
):
    """
    Learn a probabilistic deterministic finite automaton using ALERGIA.

    The function repeatedly identifies statistically compatible states and
    attempts to merge them using either the Carrasco all-pairs procedure or
    the Higuera red-blue procedure.

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
    alpha : float
        Significance level used by the Hoeffding compatibility test.
    output : {"Suppressed", "Truncated", "Full"}, default="Suppressed"
        Amount of progress information printed.
    method : {"Carrasco", "Higuera"}, default="Carrasco"
        State-merging methodology.

    Returns
    -------
    current_matrix : np.ndarray
        Final transition-count matrix.
    current_states : list
        Final state identifiers after ALERGIA terminates.
    tracking : dict
        Summary statistics containing the initial and final state counts,
        attempted and successful merges, recursive merge attempts, and
        recursive merge failures.

    Raises
    ------
    TypeError
        If alpha is not numeric, transition_matrix is not a NumPy array, or
        alphabet has an invalid type.
    ValueError
        If output or method is invalid, alpha is outside ``(0, 2]``, or
        alphabet or transition_matrix has invalid contents or dimensions.
    """
    valid_outputs = {"Suppressed", "Truncated", "Full"}

    if output not in valid_outputs:
        raise ValueError(
            "output must be 'Suppressed', 'Truncated', or 'Full'."
        )

    valid_methods = {"Carrasco", "Higuera"}

    if method not in valid_methods:
        raise ValueError(
            "method must be either 'Carrasco' or 'Higuera'."
        )

    _validate_alpha(alpha)

    alphabet = _validate_alphabet(alphabet)

    _validate_transition_matrix(transition_matrix, alphabet, states)
    
    if method == "Carrasco":
        current_matrix = transition_matrix
        current_states = states

        initial_state_count = len(current_states)

        to_check = get_pairs_to_check(states)

        merge_counter = 0
        iter_counter = 0
        attempted_merge_counter = 0
        recursive_attempt_counter = 0
        recursive_failure_counter = 0

        while to_check:
            pair = to_check[0]

            if output in ("Full", "Truncated"):
                print(
                    "The next pair of states to check is:", 
                    to_check[0],
                )

            if output == "Full":
                iter_counter += 1
                print("Iteration", iter_counter)

            attempted_merge_counter += 1

            if hoeffding_bound(
                pair[0],
                pair[1],
                alpha,
                current_matrix,
                alphabet,
                current_states,
            ):
                if output in ("Full", "Truncated"):
                    print(
                        "Hoeffding Bound satisfied for", 
                        to_check[0],
                    )

                recursive_attempt_counter += 1

                (
                    current_matrix,
                    current_states,
                    recursive_merge,
                ) = _recursive_merge_two_states(
                    pair[0],
                    pair[1],
                    current_matrix,
                    current_states,
                    alpha,
                    alphabet,
                    output=output,
                    method="Carrasco",
                )

                if recursive_merge:
                    merge_counter += 1

                    if output in ("Full", "Truncated"):
                        print(
                            "Recursively merged states. "
                            "Successfully merged", 
                            to_check[0],
                        )

                    to_check = get_pairs_to_check(current_states)

                else:
                    recursive_failure_counter += 1

                    if output in ("Full", "Truncated"):
                        print(
                            "Recursive merge process failed. "
                            "Cannot merge", 
                            pair,
                        )

                    to_check.pop(0)
                    
            else:
                if output in ("Full", "Truncated"):
                    print(
                        "Hoeffding Bound not satisfied for", 
                        pair,
                    )

                to_check.pop(0)

        tracking = {
            "initial_states": initial_state_count,
            "final_states": len(current_states),
            "attempted_merges": attempted_merge_counter,
            "successful_merges": merge_counter,
            "recursive_merge_attempts": recursive_attempt_counter,
            "recursive_merge_failures": recursive_failure_counter,
        }

        return current_matrix, current_states, tracking
    
    # Higuera method using red and blue states
    current_matrix = transition_matrix
    current_states = states

    initial_state_count = len(current_states)

    red_states = [0]
    blue_states = get_blue_states(
        current_matrix, 
        red_states, 
        current_states,
    )

    merge_counter = 0
    iter_counter = 0
    attempted_merge_counter = 0
    recursive_attempt_counter = 0
    recursive_failure_counter = 0

    while len(blue_states) > 0:
        if output == "Full":
            iter_counter += 1
            print("Iteration", iter_counter)

        q2 = blue_states[0]
        merged = False

        for q1 in red_states:
            attempted_merge_counter += 1

            if hoeffding_bound(
                q1,
                q2,
                alpha,
                current_matrix,
                alphabet,
                current_states,
            ):
                if output in ("Full", "Truncated"):
                    print(
                        "Hoeffding Bound satisfied for", 
                        (q1, q2),
                    )

                recursive_attempt_counter += 1

                (
                    current_matrix,
                    current_states,
                    recursive_merge,
                    red_states,
                ) = _recursive_merge_two_states(
                    q1,
                    q2,
                    current_matrix,
                    current_states,
                    alpha,
                    alphabet,
                    red_states,
                    output=output,
                    method="Higuera",
                )

                if recursive_merge:
                    merge_counter += 1

                    if output in ("Full", "Truncated"):
                        print("Recursively merged states. "
                              "Successfully merged", 
                              (q1, q2),
                            )
                        
                    merged = True
                    break

                recursive_failure_counter += 1

        if merged == False:
            red_states.append(q2)
            red_states = sorted(red_states)

            if output in ("Full", "Truncated"):
                print(
                    "Hoeffding Bound not satisfied for", 
                    (q1, q2),
                )

        blue_states = get_blue_states(
            current_matrix, 
            red_states, 
            current_states,
        )

    tracking = {
        "initial_states": initial_state_count,
        "final_states": len(current_states),
        "attempted_merges": attempted_merge_counter,
        "successful_merges": merge_counter,
        "recursive_merge_attempts": recursive_attempt_counter,
        "recursive_merge_failures": recursive_failure_counter,
    }

    return current_matrix, current_states, tracking


def probability_transition_matrix(pathway_matrix, states, alphabet):
    """
    Convert a transition-count matrix into a probability transition matrix.

    Outgoing transition counts for each non-artificial state are divided by
    the number of sequences entering that state. The artificial initial
    state is normalised separately using its initial transition count.

    Parameters
    ----------
    pathway_matrix : np.ndarray
        Transition-count matrix with shape
        ``(n_symbols, n_states, n_states)``.
    states : list
        State identifiers corresponding to the final two dimensions of
        pathway_matrix.
    alphabet : iterable of str
        Alphabet corresponding to the first dimension of pathway_matrix.

    Returns
    -------
    np.ndarray
        Probability transition matrix with the same shape as
        pathway_matrix.

    Raises
    ------
    TypeError
        If pathway_matrix is not a NumPy array or alphabet has an invalid
        type.
    ValueError
        If alphabet or pathway_matrix has invalid contents or dimensions.
    """
    alphabet = _validate_alphabet(alphabet)

    _validate_transition_matrix(
        pathway_matrix,
        alphabet,
        states,
    )

    p_mat = pathway_matrix.copy().astype(float)
    for j in range(len(alphabet)):
        p_mat[j, 0, :] = pathway_matrix[j, 0, :] / pathway_matrix[0, 0, :].sum()
        for i in range(1, len(states)):
            p_mat[j, i, :] = pathway_matrix[j, i, :] / get_n(
                states[i], pathway_matrix, states
            )
    return p_mat


def network_visualisation(
    pathway_matrix,
    states,
    alphabet,
    name=None,
    view=True,
    probabilities=False,
    graph_format="pdf",
):
    """
    Render an automaton as a directed Graphviz network.

    States are represented as nodes and positive transitions as directed
    edges. Edge and terminating-state labels display either counts or
    probabilities, depending on the value of probabilities.

    Parameters
    ----------
    pathway_matrix : np.ndarray
        Transition-count matrix with shape
        ``(n_symbols, n_states, n_states)``.
    states : list
        State identifiers corresponding to the final two dimensions of
        pathway_matrix.
    alphabet : iterable of str
        Alphabet corresponding to the first dimension of pathway_matrix.
    name : str, optional
        Graph name and output filename stem. The default output filename is
        ``"my_graph"``.
    view : bool, default=True
        Whether to open the rendered graph using the system's default
        viewer.
    probabilities : bool, default=False
        Whether to label nodes and edges with probabilities rather than
        transition counts.
    graph_format : str, default="pdf"
        Graphviz output format.

    Returns
    -------
    None
        The rendered graph is written to disk.

    Raises
    ------
    TypeError
        If pathway_matrix is not a NumPy array or alphabet has an invalid
        type.
    ValueError
        If alphabet or pathway_matrix has invalid contents or dimensions.
    """
    alphabet = _validate_alphabet(alphabet)

    _validate_transition_matrix(
        pathway_matrix,
        alphabet,
        states,
    )

    if name == None:
        filename = "my_graph"
    else:
        identifier = name
        filename = name

    if probabilities:
        p_mat = probability_transition_matrix(pathway_matrix, states, alphabet)

    dot = graphviz.Digraph(identifier, filename=filename)

    for node in states:
        if node == "*":
            dot.attr("node", shape="circle")
        elif get_pi_endpoint(node, pathway_matrix, alphabet, states) > 0:
            dot.attr("node", shape="doublecircle")
        else:
            dot.attr("node", shape="circle")
        dot.node(str(node), str(node))

    for n, i in enumerate(states):
        if i == "*":
            dot.attr("node", shape="circle")
            dot.node(str(i), str(i), fontsize="14")
        elif get_pi_endpoint(i, pathway_matrix, alphabet, states) > 0:
            dot.attr("node", shape="doublecircle")
            if probabilities:
                dot.node(
                    str(i),
                    "{}: {}".format(
                        i,
                        round(get_pi_endpoint(i, pathway_matrix, alphabet, states), 2),
                    ),
                    fontsize="11",
                    fixedsize="true",
                )
            else:
                dot.node(
                    str(i),
                    "{}: {}".format(i, get_endpoint(i, pathway_matrix, states)),
                    fontsize="12",
                    fixedsize="true",
                )
        else:
            dot.attr("node", shape="circle")
            dot.node(str(i), "{}: 0".format(i), fontsize="12", fixedsize="true")
        for m, j in enumerate(states):
            if any(pathway_matrix[:, n, m] != 0):
                for k in range(len(alphabet)):
                    if pathway_matrix[k, n, m] != 0:
                        if probabilities:
                            if i == "*":
                                dot.edge(
                                    str(i),
                                    str(j),
                                    label="{}".format(round(p_mat[k, n, m], 2)),
                                    arrowsize="0.35",
                                    fontsize="11",
                                )
                            else:
                                dot.edge(
                                    str(i),
                                    str(j),
                                    label="{}: {}".format(
                                        alphabet[k], round(p_mat[k, n, m], 2)
                                    ),
                                    arrowsize="0.35",
                                    fontsize="11",
                                )
                        else:
                            if i == "*":
                                dot.edge(
                                    str(i),
                                    str(j),
                                    label="{}".format(pathway_matrix[k, n, m]),
                                    arrowsize="0.35",
                                    fontsize="11",
                                )
                            else:
                                dot.edge(
                                    str(i),
                                    str(j),
                                    label="{}: {}".format(
                                        alphabet[k], pathway_matrix[k, n, m]
                                    ),
                                    arrowsize="0.35",
                                    fontsize="11",
                                )

    dot.graph_attr["rankdir"] = "LR"
    dot.render(filename, format=graph_format, cleanup=True)

    if view:
        dot.view()


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
    symbols = [sequence[i] for i in range(len(sequence))]
    indices = [alphabet.index(symbol) for symbol in symbols]

    p_mat = np.delete(p_mat, 0, axis=1)
    p_mat = np.delete(p_mat, 0, axis=2)

    p_est = np.sum(p_mat[indices[0], 0, :])

    if p_est == 0:
        return 0

    next_state = np.where(p_mat[indices[0], 0, :] > 0)[0][0]

    if len(sequence) == 1:
        p_est *= 1 - np.sum(p_mat[:, next_state, :])
        return p_est

    for i in range(1, len(sequence)):
        if i != len(sequence) - 1:
            p_est *= np.sum(p_mat[indices[i], next_state, :])
            if np.where(p_mat[indices[i], next_state, :] > 0)[0].size > 0:
                next_state = np.where(p_mat[indices[i], next_state, :] > 0)[0][0]
            else:
                return 0
        else:
            p_est *= np.sum(p_mat[indices[i], next_state, :])
            if np.where(p_mat[indices[i], next_state, :] > 0)[0].size > 0:
                next_state = np.where(p_mat[indices[i], next_state, :] > 0)[0][0]
            else:
                return 0
            p_est *= 1 - min(np.sum(p_mat[:, next_state, :]), 1)

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
        Tail probability used to obtain the standard-normal critical value.
        Must be in the range ``(0, 1)``.
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
        raise ValueError(
            "p_value must be either 'pattern' or 'sequence'."
        )
    
    if not isinstance(alpha, (int, float, np.number)):
        raise TypeError("alpha must be numeric.")

    if alpha <= 0 or alpha >= 1:
        raise ValueError(
            "alpha must be in the range (0, 1) for proportion_constraint()"
        )
    
    if len(sequences) == 0:
        raise ValueError(
            "sequences must contain at least one observed sequence."
        )
    
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
        raise ValueError(
            "The estimated probability must be between 0 and 1."
        )
    
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
        raise ValueError(
            "digram must contain exactly two symbols."
        )
    
    symbol_1 = digram[0]
    symbol_2 = digram[1]

    matrix_index_1 = alphabet.index(symbol_1)
    matrix_index_2 = alphabet.index(symbol_2)

    rho = np.sum(
        np.delete(p_mat, matrix_index_1, 0), 
        axis=0,
    )

    inverse = np.linalg.inv(
        np.identity(p_mat.shape[1]) - rho
    )

    p_symbol_1 = np.sum(
        p_mat[matrix_index_1, :, :], 
        axis=1,
    )

    nonzero = []

    for state_index in range(p_mat.shape[1]):
        destinations = np.where(
            p_mat[matrix_index_1, state_index, :] > 0
        )[0]

        if np.any(
            p_mat[matrix_index_1, state_index, :] > 0
        ):
            nonzero.append(destinations[0])
        else:
            nonzero.append(0)

    p_symbol_2 = np.zeros((1, p_mat.shape[1]))

    for i, emitted in enumerate(nonzero):
        p_symbol_2[0, i] = np.sum(
            p_mat[matrix_index_2, emitted, :], axis=0,
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
        strings += [
            "".join(x) 
            for x in it.product(alphabet, repeat=i)
        ]

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
    probs = [
        (x, probability_estimate_of_exact_sequence(p_mat, x, alphabet)) for x in strings
    ]
    return probs
