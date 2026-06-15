"""
Graphviz visualisation utilities for probabilistic automata.

This module renders an automaton as a directed Graphviz network using its
transition matrix, state identifiers, and alphabet. Nodes and transitions
may be labelled using either observed transition counts or estimated
probabilities, and the resulting graph can be written to a range of
Graphviz-supported output formats.
"""

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
    filename="my_graph",
    save=False,
    probabilities=False,
    graph_format="pdf",
    node_size=0.5,
    node_fontsize=None,
    edge_fontsize=11,
    line_width=1.1,
):
    """
    Visualise a probabilistic automaton as a directed Graphviz network.

    States are represented as nodes and positive transitions as directed
    edges. Terminating states are displayed using double circles. Node and
    edge labels show either observed counts or estimated probabilities,
    depending on the value of ``probabilities``.

    Parameters
    ----------
    transition_matrix : np.ndarray
        Transition-count matrix with shape
        ``(n_symbols, n_states, n_states)``.
    states : list
        State identifiers corresponding to the final two dimensions of
        ``transition_matrix``.
    alphabet : iterable of str
        Alphabet corresponding to the first dimension of
        ``transition_matrix``.
    filename : str, default="my_graph"
        Name of the Graphviz graph and output filename stem.
    save : bool, default=False
        Whether to render and save the graph to disk.
    probabilities : bool, default=False
        Whether to label nodes and edges with estimated probabilities rather
        than observed transition counts.
    graph_format : str, default="pdf"
        Graphviz output format used when ``save`` is True.
    node_size : float, default=0.5
        Size of the nodes in the graph.
    node_fontsize : float, default=None
        Font size of the node labels. If None, uses the default font size.
    edge_fontsize : float, default=11
        Font size of the edge labels.
    line_width : float, default=1.1
        Width of the lines representing transitions.

    Returns
    -------
    graphviz.Digraph
        The constructed Graphviz directed graph.

    Raises
    ------
    TypeError
        If ``transition_matrix`` is not a NumPy array or ``alphabet`` has an
        invalid type.
    ValueError
        If ``alphabet`` or ``transition_matrix`` has invalid contents or
        dimensions.
    """
    alphabet = _validate_alphabet(alphabet)

    _validate_transition_matrix(
        transition_matrix,
        alphabet,
        states,
    )

    if probabilities:
        label_matrix = probability_transition_matrix(
            transition_matrix,
            states,
            alphabet,
        )
    else:
        label_matrix = transition_matrix

    dot = graphviz.Digraph(
        name=filename,
        filename=filename,
        format=graph_format,
    )

    dot.graph_attr["rankdir"] = "LR"

    # Add states.
    for state in states:
        if state == "*":
            node_label = str(state)
            node_shape = "circle"
            default_font_size = 14
        else:
            endpoint_probability = get_pi_endpoint(
                state,
                transition_matrix,
                alphabet,
                states,
            )

            if endpoint_probability > 0:
                node_shape = "doublecircle"
                default_font_size = 11

                if probabilities:
                    endpoint_label = f"{endpoint_probability:.2f}"
                else:
                    endpoint_count = get_endpoint(
                        state,
                        transition_matrix,
                        states,
                    )
                    endpoint_label = str(endpoint_count)

                node_label = f"{state}: {endpoint_label}"
            else:
                node_label = str(state)
                node_shape = "circle"
                default_font_size = 12

        font_size = (
            default_font_size
            if node_fontsize is None
            else node_fontsize
        )

        dot.node(
            str(state),
            label=node_label,
            shape=node_shape,
            fontsize=str(font_size),
            fixedsize="true",
            width=str(node_size),
            height=str(node_size),
            penwidth=str(line_width),
        )

    # Add positive transitions.
    for source_index, source_state in enumerate(states):
        for destination_index, destination_state in enumerate(states):

            # Combine multiple self-loop labels into a single loop.
            if source_state == destination_state:
                self_loop_labels = []

                for symbol_index, symbol in enumerate(alphabet):
                    transition_count = transition_matrix[
                        symbol_index,
                        source_index,
                        destination_index,
                    ]

                    if transition_count <= 0:
                        continue

                    label_value = label_matrix[
                        symbol_index,
                        source_index,
                        destination_index,
                    ]

                    if probabilities:
                        self_loop_labels.append(
                            f"{symbol}: {label_value:.2f}"
                        )
                    else:
                        self_loop_labels.append(
                            f"{symbol}: {label_value}"
                        )

                if self_loop_labels:
                    dot.edge(
                        str(source_state),
                        str(destination_state),
                        label="\n".join(self_loop_labels),
                        arrowsize="0.35",
                        fontsize=str(edge_fontsize),
                        penwidth=str(line_width),
                    )

            # Keep transitions between different states as separate edges.
            else:
                for symbol_index, symbol in enumerate(alphabet):
                    transition_count = transition_matrix[
                        symbol_index,
                        source_index,
                        destination_index,
                    ]

                    if transition_count <= 0:
                        continue

                    label_value = label_matrix[
                        symbol_index,
                        source_index,
                        destination_index,
                    ]

                    if probabilities:
                        edge_label = f"{symbol}: {label_value:.2f}"
                    else:
                        edge_label = f"{symbol}: {label_value}"

                    dot.edge(
                        str(source_state),
                        str(destination_state),
                        label=edge_label,
                        arrowsize="0.35",
                        fontsize=str(edge_fontsize),
                        penwidth=str(line_width),
                    )

    if save:
        dot.render(cleanup=True)

    return dot