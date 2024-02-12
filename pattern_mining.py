import numpy as np

def get_alphabet(sequences):
    """
    A function that returns the alphabet of a PPTA.
    """
    return sorted(list(set(''.join(sequences))))

def state_paths(sequences):
    """
    A function that returns the state paths of a PPTA.
    """
    all_nodes = []
    for i in range(1, len(max(sequences, key=len)) + 1):
        this_iter = list(set([x[:i] for x in sequences if len(x) > i-1]))
        for j in this_iter:
            all_nodes.append(j)
    return all_nodes

def transition_matrix(sequences):
    """
    A function that returns the transition matrix of a PPTA, given a list of sequences.
    """
    alphabet = get_alphabet(sequences)
    all_nodes = state_paths(sequences)
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
    return 1 - sum(get_pi(q, z, pathway_matrix, states) for z in alphabet)


def hoeffding_bound(q1, q2, alpha, pathway_matrix, alphabet, states):
    """ 
    Returns a Boolean indicating whether the Hoeffding bound is satisfied.
    """
    alpha_constant = (np.log(2 / alpha) / 2) ** 0.5
    rhs = np.sqrt(1/2 * np.log(2/0.05)) * ((1/np.sqrt(get_n(q1, pathway_matrix, states))) + (1/np.sqrt(get_n(q2, pathway_matrix, states))))
    for z in alphabet:
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
    for a in alphabet:
        rows = np.where((pathway_matrix[a, :, :] > 0).sum(axis=1) > 1)[0]
        pathway_matrix[a, rows, :] > 0
        n_rows = rows.shape[0]
        nond_pairs = np.reshape(np.where(pathway_matrix[a, rows, :] > 0)[1], (n_rows, 2))
        nondeterministic_pairs += [tuple(states[i] for i in r) for r in nond_pairs]
    return nondeterministic_pairs
