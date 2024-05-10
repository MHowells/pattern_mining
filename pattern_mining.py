import numpy as np
import graphviz
from scipy.stats import norm


def get_alphabet(sequences):
    """
    A function that returns the alphabet of a PPTA.
    """
    return sorted(list(set("".join(sequences))))


def get_state_paths(sequences, build="breadth"):
    """
    A function that returns the paths to a state within a PPTA.

    Parameters:
    build -- the type of build to use, either "breadth" or "depth"
    """
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
    elif build == "depth":
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
    else:
        return "Invalid build type. Please use either 'breadth' or 'depth'."


def transition_matrix(sequences, alphabet, build = "breadth"):
    """
    A function that returns the transition matrix of a PPTA, given a list of sequences.
    """
    all_nodes = get_state_paths(sequences, build=build)
    all_nodes.insert(0, "S")
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
    states.insert(0, "S")
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


def merge_two_states(q1, q2, pathway_matrix, states):
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
    q1, q2, pathway_matrix, states, alpha, alphabet, output="Suppressed"
):
    """
    A function to recursively merge two states until the PPTA is deterministic.
    Set output to "Suppressed" to suppress output to just the final solution, "Truncated" to suppress non-deterministic merge information, or "Full" to show all output.
    """
    if output not in ["Suppressed", "Truncated", "Full"]:
        return "Invalid output type. Please use either 'Suppressed', 'Truncated', or 'Full'."
    initial_pathway_matrix = np.copy(pathway_matrix)
    initial_states = states.copy()
    new_matrix, new_states = merge_two_states(q1, q2, pathway_matrix, states)
    non_det_pairs = check_is_deterministic(new_matrix, new_states, alphabet)
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
                non_det_pairs[0][0], non_det_pairs[0][1], new_matrix, new_states
            )
            non_det_pairs = check_is_deterministic(new_matrix, new_states, alphabet)
            if len(non_det_pairs) > 0 and output == "Full":
                print(
                    "Merging of previous non-deterministic pair results in non-deterministic pairs:",
                    non_det_pairs,
                )
        else:
            recursive_merge = False
            return initial_pathway_matrix, initial_states, recursive_merge
    return new_matrix, new_states, recursive_merge


def get_pairs_to_check(states):
    """
    A function to get all pairs of states to check for merging.
    """
    state_numbers = states.copy()
    state_numbers.remove("S")
    to_check = [
        (state_numbers[j], state_numbers[i])
        for j in range(len(state_numbers))
        for i in range(0, j)
    ]
    return to_check


def alergia(transition_matrix, states, alphabet, alpha, output="Suppressed"):
    """
    A function to implement the Alergia algorithm.
    Set output to "Suppressed" to suppress output to just the final solution, "Truncated" to suppress non-deterministic merge information, or "Full" to show all output.
    """
    if output not in ["Suppressed", "Truncated", "Full"]:
        return "Invalid output type. Please use either 'Suppressed', 'Truncated', or 'Full'."
    current_matrix = transition_matrix
    current_states = states
    to_check = get_pairs_to_check(states)
    checked_states = []
    merge_counter = 0
    while to_check:
        if output in ("Full", "Truncated"):
            print("Current order of state merges to check:", to_check)
        checked_states.append(to_check[0])
        if hoeffding_bound(
            to_check[0][0],
            to_check[0][1],
            alpha,
            current_matrix,
            alphabet,
            current_states,
        ):
            if output in ("Full", "Truncated"):
                print("Hoeffding Bound satisfied for", to_check[0])
            (
                current_matrix,
                current_states,
                recursive_merge,
            ) = recursive_merge_two_states(
                to_check[0][0],
                to_check[0][1],
                current_matrix,
                current_states,
                alpha,
                alphabet,
                output=output,
            )
            if recursive_merge:
                merge_counter += 1
                if output in ("Full", "Truncated"):
                    print("Recursively merged states. Successfully merged", to_check[0])
                to_check = get_pairs_to_check(current_states)
            else:
                if output in ("Full", "Truncated"):
                    print("Recursive merge process failed. Cannot merge", to_check[0])
                to_check.pop(0)
        else:
            if output in ("Full", "Truncated"):
                print("Hoeffding Bound not satisfied for", to_check[0])
            to_check.pop(0)
    return current_matrix, current_states, merge_counter


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
    pathway_matrix, states, alphabet, name=None, view=True, probabilities=False
):
    """
    A function to visualise the PPTA as a network graph using graphviz.
    """
    if name == None:
        filename = "my_graph.gv"
    else:
        identifier = name
        filename = name + ".gv"

    if probabilities:
        p_mat = probability_transition_matrix(pathway_matrix, states, alphabet)

    dot = graphviz.Digraph(identifier, filename=filename)

    for node in states:
        if node == "S":
            dot.attr("node", shape="circle")
        elif get_pi_endpoint(node, pathway_matrix, alphabet, states) > 0:
            dot.attr("node", shape="doublecircle")
        else:
            dot.attr("node", shape="circle")
        dot.node(str(node), str(node))

    for n, i in enumerate(states):
        if i == "S":
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
                            if i == "S":
                                dot.edge(
                                    str(i),
                                    str(j),
                                    label="{}".format(round(p_mat[k, n, m], 2)
                                    ),
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
                            if i == "S":
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
    p_mat, pattern, theta, alphabet
):
    """
    A function to estimate the probability that a sequence starting from each state contains a pattern at a distance theta from the state.
    """
    if len(pattern) < 2:
        return "Please use a pattern of length greater than 1. For a single symbol, use the probability_sequence_contains_letter_at_distance_theta() function."
    symbol = pattern[0]
    matrix_index = alphabet.index(symbol)
    gamma = p_mat[matrix_index, :, :]
    tau_theta = np.linalg.matrix_power(np.sum(p_mat, axis=0), theta)
    f_x_theta = np.matmul(tau_theta, gamma)
    x2_to_xl = pattern[1:]
    if len(x2_to_xl) > 1:
        est_of_pattern = probability_estimate_of_pattern(p_mat, x2_to_xl, alphabet)
    else:
        est_of_pattern = probability_estimate_of_symbol(p_mat, x2_to_xl, alphabet)
    return np.matmul(f_x_theta, est_of_pattern)


def proportion_constraint(p_mat, pattern, alphabet, sequences, alpha):
    """
    Returns a Boolean indicating whether a pattern covers a significant part of the probability density of all sequences.
    """
    prob_value = probability_estimate_of_pattern(p_mat, pattern, alphabet)[0]
    k = abs(norm.ppf(1 - alpha)) * (
        (prob_value * (1 - prob_value) / len(sequences)) ** 0.5
    )
    if prob_value < k:
        return False
    return True


def probability_sequence_contains_digram(p_mat, digram, alphabet):
    """
    A function to return the probability that a sequence contains a digram xy, for each starting state.
    """
    if len(digram) != 2:
        return "Please use a digram of length 2."
    symbol_1 = digram[0]
    symbol_2 = digram[1]
    matrix_index_1 = alphabet.index(symbol_1)
    matrix_index_2 = alphabet.index(symbol_2)

    rho = np.sum(np.delete(p_mat, matrix_index_1, 0), axis=0)
    inverse = np.linalg.inv(np.identity(p_mat.shape[1]) - rho)

    p_symbol_1 = np.sum(p_mat[matrix_index_1, :, :], axis=1)
    nonzero = []
    for i in range(p_mat.shape[1]):
        if any(p_mat[matrix_index_1, i, :] > 0):
            nonzero.append(np.where(p_mat[matrix_index_1, i, :] > 0)[0][0])
        else:
            nonzero.append(0)
    p_symbol_2 = np.zeros((1, p_mat.shape[1]))
    for i, emitted in enumerate(nonzero):
        p_symbol_2[0, i] = np.sum(p_mat[matrix_index_2, emitted, :], axis=0)
    tau = np.multiply(p_symbol_1, p_symbol_2)

    return np.matmul(tau, inverse)
