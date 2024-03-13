import numpy as np

def get_alphabet(sequences):
    """
    A function that returns the alphabet of a PPTA.
    """
    return sorted(list(set(''.join(sequences))))

def get_state_paths(sequences):
    """
    A function that returns the state paths of a PPTA.
    """
    all_paths = []
    for i in range(1, len(max(sequences, key=len)) + 1):
        this_iter = list(set([x[:i] for x in sequences if len(x) > i-1]))
        for j in this_iter:
            all_paths.append(j)
    return all_paths

def transition_matrix(sequences, alphabet):
    """
    A function that returns the transition matrix of a PPTA, given a list of sequences.
    """
    all_nodes = get_state_paths(sequences)
    all_nodes.sort()
    all_nodes.insert(0, "")
    all_nodes.insert(0, "S")
    n = len(all_nodes)
    pathway_matrix = np.zeros((len(alphabet), n, n), dtype=int)
    pathway_matrix[0, 0, 1] = len(sequences)
    for i in range(1, n):
        for k in range(len(alphabet)):
            next_node = all_nodes[i] + alphabet[k]
            if next_node in all_nodes:
                pathway_matrix[k, i, all_nodes.index(next_node)] = len([x for x in sequences if x.startswith(next_node)])
    return pathway_matrix

def get_n(q, pathway_matrix, states):
    """
    Gets n(q), the number of pathways entering state q.
    """
    i = states.index(q)
    return pathway_matrix[:, :, i].sum()


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
    rhs = alpha_constant * ((1/np.sqrt(get_n(q1, pathway_matrix, states))) + (1/np.sqrt(get_n(q2, pathway_matrix, states))))
    for z in range(len(alphabet)):
        lhs = abs(get_pi(q1, z, pathway_matrix, states) - get_pi(q2, z, pathway_matrix, states))
        if lhs > rhs:
            return False
    lhs = abs(get_pi_endpoint(q1, pathway_matrix, alphabet, states) - get_pi_endpoint(q2, pathway_matrix, alphabet, states))
    if lhs > rhs:
        return False
    return True


def merge_two_states(q1, q2, pathway_matrix, states):
    """ 
    Merges states q1 and q2 into a third state that is appeded to the end.
    Returns the new pathway_matrix and state list as copies of the originals.
    """
    i1 = states.index(q1)
    i2 = states.index(q2)
    pathway_matrix_copy = np.copy(pathway_matrix)
    states_copy = states.copy()
    new_column = pathway_matrix_copy[:, :, i1] + pathway_matrix_copy[:, :, i2]
    pathway_matrix_copy = np.dstack((np.delete(pathway_matrix_copy, [i1, i2], 2), new_column))
    new_row = pathway_matrix_copy[:, i1, :] + pathway_matrix_copy[:, i2, :]
    pathway_matrix_copy = np.dstack((np.delete(pathway_matrix_copy, [i1, i2], 1).transpose(0, 2, 1), new_row)).transpose(0, 2, 1)
    next_state = max(states_copy[1:]) + 1
    states_copy.append(next_state)
    states_copy.remove(q1)
    states_copy.remove(q2)
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
        n_rows = rows.shape[0]
        where_non_det = np.where(pathway_matrix[a, rows, :] > 0)[1]
        if len(where_non_det) > 2:
            nond_pairs = np.reshape(where_non_det[:2], (n_rows, 2))
        else: 
            nond_pairs = np.reshape(where_non_det, (n_rows, 2))
        nondeterministic_pairs += [tuple(states[i] for i in r) for r in nond_pairs]
    return nondeterministic_pairs


def recursive_merge_two_states(q1, q2, pathway_matrix, states, alpha, alphabet):
    """
    A function to recursively merge two states until the PPTA is deterministic.
    """
    initial_pathway_matrix = np.copy(pathway_matrix)
    initial_states = states.copy()
    new_matrix, new_states = merge_two_states(q1, q2, pathway_matrix, states)
    non_det_pairs = check_is_deterministic(new_matrix, new_states, alphabet)
    while non_det_pairs:
        if hoeffding_bound(non_det_pairs[0][0], non_det_pairs[0][1], alpha, new_matrix, alphabet, new_states):
            new_matrix, new_states = merge_two_states(non_det_pairs[0][0], non_det_pairs[0][1], new_matrix, new_states)
            non_det_pairs = check_is_deterministic(new_matrix, new_states, alphabet)
        else:
            return initial_pathway_matrix, initial_states
    return new_matrix, new_states


def get_pairs_to_check(states):
    """
    A function to get all pairs of states to check for merging.
    """
    state_numbers = states.copy()
    state_numbers.remove("S")
    to_check = [(state_numbers[j], state_numbers[i]) for j in range(len(state_numbers)) for i in range(0, j)]
    return to_check


def alergia(transition_matrix, states, alphabet, alpha):
    """
    A function to implement the Alergia algorithm.
    """
    current_matrix = transition_matrix
    current_states = states
    to_check = get_pairs_to_check(states)
    checked_states = []
    while to_check:
        checked_states.append(to_check[0])
        if hoeffding_bound(to_check[0][0], to_check[0][1], alpha, current_matrix, alphabet, current_states):
            current_matrix, current_states = recursive_merge_two_states(to_check[0][0], to_check[0][1], current_matrix, current_states, alpha, alphabet)
            to_check = [x for x in get_pairs_to_check(current_states) if x not in checked_states]
        else:
            to_check.pop(0)
    return current_matrix, current_states