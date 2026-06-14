def get_n(q, transition_matrix, states):
    """
    Return the number of sequences entering a state.

    This calculates ``n(q)`` by summing all transitions entering the
    specified state.

    Parameters
    ----------
    q : int or str
        State identifier.
    transition_matrix : np.ndarray
        Transition-count matrix with shape
        ``(n_symbols, n_states, n_states)``.
    states : list
        State identifiers corresponding to the final two dimensions of
        transition_matrix.

    Returns
    -------
    int or float
        Number of sequences entering the state.
    """
    i = states.index(q)
    return transition_matrix[:, :, i].sum()


def get_endpoint(q, transition_matrix, states):
    """
    Return the number of sequences terminating at a state.

    The terminating count is calculated as the number of sequences entering
    the state minus the number leaving it.

    Parameters
    ----------
    q : int or str
        State identifier.
    transition_matrix : np.ndarray
        Transition-count matrix with shape
        ``(n_symbols, n_states, n_states)``.
    states : list
        State identifiers corresponding to the final two dimensions of
        transition_matrix.

    Returns
    -------
    int or float
        Number of sequences terminating at the state.
    """
    i = states.index(q)
    return transition_matrix[:, :, i].sum() - transition_matrix[:, i, :].sum()


def get_pi(q, z, transition_matrix, states):
    """
    Return the probability of leaving a state via a symbol.

    This calculates ``pi(q, z)`` for the symbol represented by index z in
    the first dimension of the transition-count matrix.

    Parameters
    ----------
    q : int or str
        State identifier.
    z : int
        Index of the symbol in the first dimension of transition_matrix.
    transition_matrix : np.ndarray
        Transition-count matrix with shape
        ``(n_symbols, n_states, n_states)``.
    states : list
        State identifiers corresponding to the final two dimensions of
        transition_matrix.

    Returns
    -------
    float
        Probability of leaving state q via the indexed symbol.
    """
    i = states.index(q)
    return transition_matrix[z, i, :].sum() / get_n(q, transition_matrix, states)


def get_pi_endpoint(q, transition_matrix, alphabet, states):
    """
    Return the probability of terminating at a state.

    The terminating probability is one minus the sum of the probabilities
    of leaving the state through each symbol in the alphabet.

    Parameters
    ----------
    q : int or str
        State identifier.
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
    float
        Probability of terminating at state q.
    """
    return 1 - sum(
        get_pi(
            q,
            z,
            transition_matrix,
            states,
        )
        for z in range(len(alphabet))
    )