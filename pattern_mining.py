import numpy as np

def get_n(q, pathway_matrix, states):
    """
    Gets n(q), the number of pathways entering state q.
    """
    i = states.index(q)
    if i == 0:
        # Starting state, number entering is number leaving
        return pathway_matrix[:, 0, :].sum()
    # All other states, it's the number entering
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
    next_state = max(states_copy) + 1
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
    expected_number_of_zeroes = len(states) - 1
    for a in alphabet:
        final_row = pathway_matrix[a, -1, :]
        nonzero_entries = np.nonzero(final_row)[0]
        if nonzero_entries.size > 1: 
            nondeterministic_pairs.append(tuple(states[i] for i in nonzero_entries))
    return nondeterministic_pairs