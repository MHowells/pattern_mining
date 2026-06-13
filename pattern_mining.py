import numpy as np
import graphviz
from scipy.stats import norm
import itertools as it


def get_alphabet(sequences):
    """
    A function that returns the alphabet of a PPTA.
    """
    return sorted(list(set("".join(sequences))))


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
    ValueError
        If build is not "breadth" or "depth".
    """
    if build not in {"breadth", "depth"}:
        raise ValueError(
            "Invalid build type, build must be 'breadth' or 'depth'."
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
                            if len(x) > len(current_node) and x.startswith(current_node)
                        ]
                    )
                )
            )

            for j in range(len(this_iter)):
                all_paths.append(this_iter[j])
                all_ordered.insert(
                    all_ordered.index(current_node) + 1 + j, this_iter[j]
                )

            tracker += 1

            if tracker == len(all_paths):
                break

            current_node = all_ordered[all_ordered.index(current_node) + 1]

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
                            if len(x) > len(current_node) and x.startswith(current_node)
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
    A function that returns the transition matrix of a PPTA, given a list of sequences.
    """
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
    A function that returns the states of a PPTA.
    """
    states = list(range(len(get_state_paths(sequences))))
    states.insert(0, "*")
    return states


def get_n(q, pathway_matrix, states):
    """
    Gets n(q), the number of pathways entering state q.
    """
    i = states.index(q)
    return pathway_matrix[:, :, i].sum()


def get_endpoint(q, pathway_matrix, states):
    """
    Gets the number of pathways that terminate at state q.
    """
    i = states.index(q)
    return pathway_matrix[:, :, i].sum() - pathway_matrix[:, i, :].sum()


def get_pi(q, z, pathway_matrix, states):
    """
    Gets pi(q, z), the probability of leaving state q via letter z.
    """
    i = states.index(q)
    return pathway_matrix[z, i, :].sum() / get_n(q, pathway_matrix, states)


def get_pi_endpoint(q, pathway_matrix, alphabet, states):
    """
    Gets pi(q), the probability of terminating at state q.
    """
    return 1 - sum(get_pi(q, z, pathway_matrix, states) for z in range(len(alphabet)))


def hoeffding_bound(q1, q2, alpha, pathway_matrix, alphabet, states):
    """
    Returns a Boolean indicating whether the Hoeffding bound is satisfied.
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


def merge_two_states(q1, q2, pathway_matrix, states, red_states=None):
    """
    Merges states q1 and q2 into a new state that replaces the lowest numbered state.
    Returns the new pathway_matrix and state list as copies of the originals.
    """
    i1 = states.index(q1)
    i2 = states.index(q2)
    which_min = min(i1, i2)
    which_max = max(i1, i2)
    pathway_matrix_copy = np.copy(pathway_matrix)
    states_copy = states.copy()
    pathway_matrix_copy[:, :, which_min] = (
        pathway_matrix_copy[:, :, i1] + pathway_matrix_copy[:, :, i2]
    )
    pathway_matrix_copy = np.delete(pathway_matrix_copy, which_max, 2)
    pathway_matrix_copy[:, which_min, :] = (
        pathway_matrix_copy[:, i1, :] + pathway_matrix_copy[:, i2, :]
    )
    pathway_matrix_copy = np.delete(pathway_matrix_copy, which_max, 1)
    states_copy.remove(states[which_max])
    if red_states:
        if max([q1, q2]) in red_states:
            red_states = [
                min([q1, q2]) if x == max([q1, q2]) else x for x in red_states
            ]
    if red_states:
        return pathway_matrix_copy, states_copy, red_states
    else:
        return pathway_matrix_copy, states_copy


def check_is_deterministic(pathway_matrix, states, alphabet):
    """
    Checks whether the newly created state is deterministic.
    Returns a list of non-deterministic state pairs.
    """
    nondeterministic_pairs = []
    for a in range(len(alphabet)):
        rows = np.where((pathway_matrix[a, :, :] > 0).sum(axis=1) > 1)[0]
        pathway_matrix[a, rows, :] > 0
        for row in rows:
            where_non_det = np.where(pathway_matrix[a, row, :] > 0)[0]
            if len(where_non_det) > 2:
                nond_pairs = np.reshape(where_non_det[:2], (1, 2))
            else:
                nond_pairs = np.reshape(where_non_det, (1, 2))
            nondeterministic_pairs += [tuple(states[i] for i in r) for r in nond_pairs]
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
    Recursively merge two states until the PPTA is deterministic.

    Parameters
    ----------
    output : {"Suppressed", "Truncated", "Full"}
        Controls the amount of information printed.
    method : {"Carrasco", "Higuera"}
        State-merging methodology to use.

    Returns
    -------
    If method is "Carrasco":
        new_matrix : np.ndarray
            The new transition matrix after merging.
        new_states : list
            The new list of states after merging.
        recursive_merge : bool
            True if the merge was successful, False if the merge was unsuccessful.
    If method is "Higuera":
        new_matrix : np.ndarray
            The new transition matrix after merging.
        new_states : list
            The new list of states after merging.
        recursive_merge : bool
            True if the merge was successful, False if the merge was unsuccessful.
        red_states : list
            The updated list of red states after merging.

    Raises
    ------
    ValueError
        If output or method is invalid, or if red_states is not supplied
        for the Higuera method.
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
    
    if method == "Carrasco":
        initial_pathway_matrix = np.copy(pathway_matrix)
        initial_states = states.copy()

        new_matrix, new_states = merge_two_states(
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

                new_matrix, new_states = merge_two_states(
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

    new_matrix, new_states, red_states = merge_two_states(
        q1, 
        q2, 
        pathway_matrix, 
        states, 
        red_states,
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

            new_matrix, new_states = merge_two_states(
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
    Returns the blue states of a PPTA given a list of red states.
    """
    blue_states = []
    for q in red_states:
        blue_states += [
            states[x]
            for x in list(np.where(pathway_matrix[:, states.index(q), :] > 0)[1])
        ]
    blue_states = [x for x in blue_states if x not in red_states]
    return sorted(blue_states)


def get_pairs_to_check(states):
    """
    A function to get all pairs of states to check for merging.
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
    Implement the ALERGIA algorithm.

    Parameters
    ----------
    transition_matrix : np.ndarray
        Transition count matrix.
    states : list
        State identifiers corresponding to the transition matrix.
    alphabet : list of str
        Alphabet used by the automaton.
    alpha : float
        Significance level used by the Hoeffding compatibility test.
    output : {"Suppressed", "Truncated", "Full"}, default="Suppressed"
        Amount of progress information printed.
    method : {"Carrasco", "Higuera"}, default="Carrasco"
        State-merging methodology.

    Returns
    -------
    current_matrix : np.ndarray
        Final transition count matrix.
    current_states : list
        Final states after ALERGIA terminates.
    tracking : dict
        Summary statistics for the ALERGIA run.

    Raises
    ------
    TypeError
        If alpha is not numeric.
    ValueError
        If output or method is invalid, or alpha is not in the range (0, 2].
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

    if not isinstance(alpha, (int, float, np.number)):
        raise TypeError("alpha must be numeric.")

    if alpha <= 0 or alpha > 2:
        raise ValueError(
            "alpha must be in the range (0, 2]."
        )
    
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
                ) = recursive_merge_two_states(
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
                ) = recursive_merge_two_states(
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
    A function to return the probability transition matrix.
    """
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
    A function to visualise the PPTA as a network graph using graphviz.
    """
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
    A function to estimate the probability squence starting from each state contains a symbol from the alphabet.
    """
    matrix_index = alphabet.index(symbol)
    rho = np.sum(np.delete(p_mat, matrix_index, 0), axis=0)
    inverse = np.linalg.inv(np.identity(p_mat.shape[1]) - rho)
    p_symbol = np.sum(p_mat[matrix_index, :, :], axis=1)
    return np.matmul(inverse, p_symbol)


def probability_estimate_of_pattern(p_mat, pattern, alphabet):
    """
    A function to estimate the probability of a sequence starting from each state contains a pattern.
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
    A function to estimate the probability of an exact sequence.
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
    A function to estimate the probability that a sequence starting from each state contains a letter at a distance theta from the state.
    """
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
    Estimate the probability that a sequence starting from each state
    encounters a pattern at distance theta from that state.

    Parameters
    ----------
    p_mat : np.ndarray
        Probability transition matrix.
    pattern : str
        Pattern containing at least two symbols.
    theta : int
        Non-negative distance from the starting state.
    alphabet : list of str
        Alphabet associated with the probability transition matrix.

    Returns
    -------
    np.ndarray
        Estimated probability for each starting state.

    Raises
    ------
    TypeError
        If theta is not an integer.
    ValueError
        If pattern contains fewer than two symbols or theta is negative.
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
    Determine whether a pattern or exact sequence covers a significant
    proportion of the model's probability density.

    Parameters
    ----------
    p_mat : np.ndarray
        Probability transition matrix.
    pattern : str
        Pattern or exact sequence being evaluated.
    alphabet : list of str
        Alphabet associated with the probability transition matrix.
    sequences : collection
        Observed sequences used to calculate the sampling term.
    alpha : float
        Significance level used by the Hoeffding compatibility test.
    p_value : {"pattern", "sequence"}, default="pattern"
        Whether to calculate the probability of encountering a pattern
        or the probability of an exact sequence.

    Returns
    -------
    bool
        True if the estimated probability is at least as large as the
        calculated threshold; otherwise False.

    Raises
    ------
    TypeError
        If alpha is not numeric.
    ValueError
        If p_value is invalid, alpha is outside (0, 2), sequences is
        empty, or the estimated probability is invalid.
    """
    if p_value not in {"pattern", "sequence"}:
        raise ValueError(
            "p_value must be either 'pattern' or 'sequence'."
        )

    if not isinstance(alpha, (int, float, np.number)):
        raise TypeError("alpha must be numeric.")

    if alpha <= 0 or alpha > 2:
        raise ValueError(
            "alpha must be in the range (0, 2]."
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
    A function to estimate the probability of all strings in a given list.
    """
    probs = [
        (x, probability_estimate_of_exact_sequence(p_mat, x, alphabet)) for x in strings
    ]
    return probs
