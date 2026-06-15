"""
ALERGIA learning algorithms for probabilistic deterministic finite automata.

This module provides the main routine for learning a probabilistic
deterministic finite automaton from a prefix tree acceptor. It supports 
both the Carrasco and Oncina all pairs methodology and the de la Higuera 
red-blue framework methodology.
"""

from ._validation import (
    _validate_alpha,
    _validate_alphabet,
    _validate_transition_matrix,
)
from .state_merging import (
    _recursive_merge_two_states,
    get_blue_states,
    get_pairs_to_check,
    hoeffding_bound,
)


def alergia(
    transition_matrix,
    states,
    alphabet,
    alpha,
    output_level="Suppressed",
    method="carrasco",
):
    """
    Learn a probabilistic deterministic finite automaton using ALERGIA.

    The function repeatedly identifies statistically compatible states and
    attempts to merge them using either the carrasco all-pairs procedure or
    the de_la_higuera red-blue procedure.

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
        Smaller values generally permit more merges, while larger values 
        generally permit fewer merges. Must lie in the interval ``(0, 2]``.
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
        If output_level or method is invalid, alpha is outside ``(0, 2]``, or
        alphabet or transition_matrix has invalid contents or dimensions.

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

    _validate_alpha(alpha)

    alphabet = _validate_alphabet(alphabet)

    _validate_transition_matrix(transition_matrix, alphabet, states)

    if method == "carrasco":
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

            if output_level in ("Full", "Truncated"):
                print(
                    "The next pair of states to check is:",
                    pair,
                )

            if output_level == "Full":
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
                if output_level in ("Full", "Truncated"):
                    print(
                        "Hoeffding Bound satisfied for",
                        pair,
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
                    output_level=output_level,
                    method="carrasco",
                )

                if recursive_merge:
                    merge_counter += 1

                    if output_level in ("Full", "Truncated"):
                        print(
                            "Recursively merged states. Successfully merged",
                            pair,
                        )

                    to_check = get_pairs_to_check(current_states)

                else:
                    recursive_failure_counter += 1

                    if output_level in ("Full", "Truncated"):
                        print(
                            "Recursive merge process failed. Cannot merge",
                            pair,
                        )

                    to_check.pop(0)

            else:
                if output_level in ("Full", "Truncated"):
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

    # de_la_higuera method using red and blue states
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
        if output_level == "Full":
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
                if output_level in ("Full", "Truncated"):
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
                    output_level=output_level,
                    method="de_la_higuera",
                )

                if recursive_merge:
                    merge_counter += 1

                    if output_level in ("Full", "Truncated"):
                        print(
                            "Recursively merged states. Successfully merged",
                            (q1, q2),
                        )

                    merged = True
                    break

                recursive_failure_counter += 1

        if merged == False:
            red_states.append(q2)
            red_states = sorted(red_states)

            if output_level in ("Full", "Truncated"):
                print(
                    "Could not merge blue state",
                    q2,
                    "with any red state.",
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