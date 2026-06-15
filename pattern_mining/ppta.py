"""
Construction utilities for probabilistic prefix tree acceptors.

This module derives alphabets and prefix-state paths from observed
sequences, constructs transition-count matrices for probabilistic prefix
tree acceptors, and creates their initial state identifiers.
"""

import numpy as np

from ._validation import (
    _validate_alphabet,
    _validate_sequences,
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
        raise ValueError("build must be either 'breadth' or 'depth'.")

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
                            if len(x) > len(current_node) and x.startswith(current_node)
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

            current_node = all_ordered[all_ordered.index(current_node) + 1]

        return all_paths

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

        current_node = all_paths[all_paths.index(current_node) + 1]

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
    sequences = _validate_sequences(sequences)
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

    transition_matrix = np.zeros((len(alphabet), n, n), dtype=int)
    transition_matrix[0, 0, 1] = len(sequences)

    for i in range(1, n):
        for k in range(len(alphabet)):
            next_node = all_nodes[i] + alphabet[k]

            if next_node in all_nodes:
                transition_matrix[k, i, all_nodes.index(next_node)] = len(
                    [x for x in sequences if x.startswith(next_node)]
                )

    return transition_matrix


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