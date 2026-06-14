import graphviz

from ._validation import (
    _validate_alphabet,
    _validate_transition_matrix,
)
from .probabilities import probability_transition_matrix
from .state_statistics import (
    get_endpoint,
    get_pi_endpoint,
)


def network_visualisation(
    transition_matrix,
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
    transition_matrix : np.ndarray
        Transition-count matrix with shape
        ``(n_symbols, n_states, n_states)``.
    states : list
        State identifiers corresponding to the final two dimensions of
        transition_matrix.
    alphabet : iterable of str
        Alphabet corresponding to the first dimension of transition_matrix.
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
        If transition_matrix is not a NumPy array or alphabet has an invalid
        type.
    ValueError
        If alphabet or transition_matrix has invalid contents or dimensions.
    """
    alphabet = _validate_alphabet(alphabet)

    _validate_transition_matrix(
        transition_matrix,
        alphabet,
        states,
    )

    if name == None:
        filename = "my_graph"
    else:
        identifier = name
        filename = name

    if probabilities:
        p_mat = probability_transition_matrix(transition_matrix, states, alphabet)

    dot = graphviz.Digraph(identifier, filename=filename)

    for node in states:
        if node == "*":
            dot.attr("node", shape="circle")
        elif get_pi_endpoint(node, transition_matrix, alphabet, states) > 0:
            dot.attr("node", shape="doublecircle")
        else:
            dot.attr("node", shape="circle")
        dot.node(str(node), str(node))

    for n, i in enumerate(states):
        if i == "*":
            dot.attr("node", shape="circle")
            dot.node(str(i), str(i), fontsize="14")
        elif get_pi_endpoint(i, transition_matrix, alphabet, states) > 0:
            dot.attr("node", shape="doublecircle")
            if probabilities:
                dot.node(
                    str(i),
                    "{}: {}".format(
                        i,
                        round(get_pi_endpoint(i, transition_matrix, alphabet, states), 2),
                    ),
                    fontsize="11",
                    fixedsize="true",
                )
            else:
                dot.node(
                    str(i),
                    "{}: {}".format(i, get_endpoint(i, transition_matrix, states)),
                    fontsize="12",
                    fixedsize="true",
                )
        else:
            dot.attr("node", shape="circle")
            dot.node(str(i), "{}: 0".format(i), fontsize="12", fixedsize="true")
        for m, j in enumerate(states):
            if any(transition_matrix[:, n, m] != 0):
                for k in range(len(alphabet)):
                    if transition_matrix[k, n, m] != 0:
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
                                    label="{}".format(transition_matrix[k, n, m]),
                                    arrowsize="0.35",
                                    fontsize="11",
                                )
                            else:
                                dot.edge(
                                    str(i),
                                    str(j),
                                    label="{}: {}".format(
                                        alphabet[k], transition_matrix[k, n, m]
                                    ),
                                    arrowsize="0.35",
                                    fontsize="11",
                                )

    dot.graph_attr["rankdir"] = "LR"
    dot.render(filename, format=graph_format, cleanup=True)

    if view:
        dot.view()