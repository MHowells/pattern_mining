import numpy as np
from scipy.stats import norm
import pattern_mining

list_of_sequences = [
    "0",
    "0",
    "0",
    "0",
    "0",
    "0",
    "0",
    "0",
    "01",
    "01",
    "01",
    "01",
    "01",
    "01",
    "01",
    "10",
    "10",
    "10",
    "10",
    "10",
    "11",
    "11",
    "11",
    "11",
    "11",
    "12",
    "12",
    "1",
    "1",
    "1",
]

alphabet = ["0", "1", "2"]

arnolds_sequences = ["AB", "ABA", "ABB", "ABCA", "AC", "ACC", "BA", "BAA", "BC", "BCA"]

arnolds_alphabet = ["A", "B", "C"]


def test_get_alphabet():
    obtained_alphabet = pattern_mining.get_alphabet(list_of_sequences)
    expected_alphabet = ["0", "1", "2"]
    assert obtained_alphabet == expected_alphabet

    obtained_alphabet = pattern_mining.get_alphabet(arnolds_sequences)
    expected_alphabet = ["A", "B", "C"]
    assert obtained_alphabet == expected_alphabet


def test_get_state_paths():
    obtained_state_paths_breadth = pattern_mining.get_state_paths(
        list_of_sequences, "breadth"
    )
    expected_state_paths_breadth = ["", "0", "1", "01", "10", "11", "12"]
    obtained_state_paths_depth = pattern_mining.get_state_paths(
        list_of_sequences, "depth"
    )
    expected_state_paths_depth = ["", "0", "1", "01", "10", "11", "12"]
    assert obtained_state_paths_breadth == expected_state_paths_breadth
    assert obtained_state_paths_depth == expected_state_paths_depth

    obtained_state_paths_breadth = pattern_mining.get_state_paths(
        arnolds_sequences, "breadth"
    )
    expected_state_paths_breadth = [
        "",
        "A",
        "B",
        "AB",
        "AC",
        "ABA",
        "ABB",
        "ABC",
        "ABCA",
        "ACC",
        "BA",
        "BC",
        "BAA",
        "BCA",
    ]
    obtained_state_paths_depth = pattern_mining.get_state_paths(
        arnolds_sequences, "depth"
    )
    expected_state_paths_depth = [
        "",
        "A",
        "B",
        "AB",
        "AC",
        "BA",
        "BC",
        "ABA",
        "ABB",
        "ABC",
        "ACC",
        "BAA",
        "BCA",
        "ABCA",
    ]
    assert obtained_state_paths_breadth == expected_state_paths_breadth
    assert obtained_state_paths_depth == expected_state_paths_depth


def test_get_transition_matrix():
    obtained_transition_matrix = pattern_mining.get_transition_matrix(
        list_of_sequences, alphabet
    )
    expected_pathway_matrix = np.array(
        [
            [
                [0, 30, 0, 0, 0, 0, 0, 0],
                [0, 0, 15, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 5, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 15, 0, 0, 0, 0],
                [0, 0, 0, 0, 7, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 5, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 2],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ]
    )
    assert np.allclose(obtained_transition_matrix, expected_pathway_matrix)

    obtained_transition_matrix = pattern_mining.get_transition_matrix(
        arnolds_sequences, arnolds_alphabet
    )
    expected_pathway_matrix = np.array(
        [
            [
                [0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ]
    )
    assert np.allclose(obtained_transition_matrix, expected_pathway_matrix)


def test_get_initial_states():
    obtained_states = pattern_mining.get_initial_states(list_of_sequences)
    expected_states = ["*", 0, 1, 2, 3, 4, 5, 6]
    assert obtained_states == expected_states

    obtained_states = pattern_mining.get_initial_states(arnolds_sequences)
    expected_states = ["*", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    assert obtained_states == expected_states


states = ["*", 0, 1, 2, 3, 4, 5, 6]
example_red_states = [0]
states_alternate_names = ["*", "A", "B", "C", "D", "E", "F", "G"]
pathway_matrix = np.array(
    [
        [
            [0, 30, 0, 0, 0, 0, 0, 0],
            [0, 0, 15, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 5, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 15, 0, 0, 0, 0],
            [0, 0, 0, 0, 7, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 5, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
    ]
)

arnolds_states = ["*", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
arnolds_pathway_matrix = np.array(
    [
        [
            [0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
    ]
)


def test_get_n():
    assert pattern_mining.get_n(0, pathway_matrix, states) == 30
    assert pattern_mining.get_n(1, pathway_matrix, states) == 15
    assert pattern_mining.get_n(2, pathway_matrix, states) == 15
    assert pattern_mining.get_n(3, pathway_matrix, states) == 7
    assert pattern_mining.get_n(4, pathway_matrix, states) == 5
    assert pattern_mining.get_n(5, pathway_matrix, states) == 5
    assert pattern_mining.get_n(6, pathway_matrix, states) == 2

    assert pattern_mining.get_n("A", pathway_matrix, states_alternate_names) == 30
    assert pattern_mining.get_n("B", pathway_matrix, states_alternate_names) == 15
    assert pattern_mining.get_n("C", pathway_matrix, states_alternate_names) == 15
    assert pattern_mining.get_n("D", pathway_matrix, states_alternate_names) == 7
    assert pattern_mining.get_n("E", pathway_matrix, states_alternate_names) == 5
    assert pattern_mining.get_n("F", pathway_matrix, states_alternate_names) == 5
    assert pattern_mining.get_n("G", pathway_matrix, states_alternate_names) == 2

    assert pattern_mining.get_n(0, arnolds_pathway_matrix, arnolds_states) == 10
    assert pattern_mining.get_n(1, arnolds_pathway_matrix, arnolds_states) == 6
    assert pattern_mining.get_n(2, arnolds_pathway_matrix, arnolds_states) == 4
    assert pattern_mining.get_n(3, arnolds_pathway_matrix, arnolds_states) == 4
    assert pattern_mining.get_n(4, arnolds_pathway_matrix, arnolds_states) == 2
    assert pattern_mining.get_n(5, arnolds_pathway_matrix, arnolds_states) == 1
    assert pattern_mining.get_n(6, arnolds_pathway_matrix, arnolds_states) == 1
    assert pattern_mining.get_n(7, arnolds_pathway_matrix, arnolds_states) == 1
    assert pattern_mining.get_n(8, arnolds_pathway_matrix, arnolds_states) == 1
    assert pattern_mining.get_n(9, arnolds_pathway_matrix, arnolds_states) == 1
    assert pattern_mining.get_n(10, arnolds_pathway_matrix, arnolds_states) == 2
    assert pattern_mining.get_n(11, arnolds_pathway_matrix, arnolds_states) == 2
    assert pattern_mining.get_n(12, arnolds_pathway_matrix, arnolds_states) == 1
    assert pattern_mining.get_n(13, arnolds_pathway_matrix, arnolds_states) == 1


def test_get_endpoint():
    assert pattern_mining.get_endpoint(0, pathway_matrix, states) == 0
    assert pattern_mining.get_endpoint(1, pathway_matrix, states) == 8
    assert pattern_mining.get_endpoint(2, pathway_matrix, states) == 3
    assert pattern_mining.get_endpoint(3, pathway_matrix, states) == 7
    assert pattern_mining.get_endpoint(4, pathway_matrix, states) == 5
    assert pattern_mining.get_endpoint(5, pathway_matrix, states) == 5
    assert pattern_mining.get_endpoint(6, pathway_matrix, states) == 2

    assert pattern_mining.get_endpoint("A", pathway_matrix, states_alternate_names) == 0
    assert pattern_mining.get_endpoint("B", pathway_matrix, states_alternate_names) == 8
    assert pattern_mining.get_endpoint("C", pathway_matrix, states_alternate_names) == 3
    assert pattern_mining.get_endpoint("D", pathway_matrix, states_alternate_names) == 7
    assert pattern_mining.get_endpoint("E", pathway_matrix, states_alternate_names) == 5
    assert pattern_mining.get_endpoint("F", pathway_matrix, states_alternate_names) == 5
    assert pattern_mining.get_endpoint("G", pathway_matrix, states_alternate_names) == 2

    assert pattern_mining.get_endpoint(0, arnolds_pathway_matrix, arnolds_states) == 0
    assert pattern_mining.get_endpoint(1, arnolds_pathway_matrix, arnolds_states) == 0
    assert pattern_mining.get_endpoint(2, arnolds_pathway_matrix, arnolds_states) == 0
    assert pattern_mining.get_endpoint(3, arnolds_pathway_matrix, arnolds_states) == 1
    assert pattern_mining.get_endpoint(4, arnolds_pathway_matrix, arnolds_states) == 1
    assert pattern_mining.get_endpoint(5, arnolds_pathway_matrix, arnolds_states) == 1
    assert pattern_mining.get_endpoint(6, arnolds_pathway_matrix, arnolds_states) == 1
    assert pattern_mining.get_endpoint(7, arnolds_pathway_matrix, arnolds_states) == 0
    assert pattern_mining.get_endpoint(8, arnolds_pathway_matrix, arnolds_states) == 1
    assert pattern_mining.get_endpoint(9, arnolds_pathway_matrix, arnolds_states) == 1
    assert pattern_mining.get_endpoint(10, arnolds_pathway_matrix, arnolds_states) == 1
    assert pattern_mining.get_endpoint(11, arnolds_pathway_matrix, arnolds_states) == 1
    assert pattern_mining.get_endpoint(12, arnolds_pathway_matrix, arnolds_states) == 1
    assert pattern_mining.get_endpoint(13, arnolds_pathway_matrix, arnolds_states) == 1


def test_get_pi():
    obtained_pi = [
        pattern_mining.get_pi(i, 0, pathway_matrix, states) for i in range(7)
    ]
    expected_pi = [1 / 2, 0, 1 / 3, 0, 0, 0, 0]
    assert np.allclose(obtained_pi, expected_pi)
    obtained_pi = [
        pattern_mining.get_pi(i, 1, pathway_matrix, states) for i in range(7)
    ]
    expected_pi = [1 / 2, 7 / 15, 1 / 3, 0, 0, 0, 0]
    assert np.allclose(obtained_pi, expected_pi)
    obtained_pi = [
        pattern_mining.get_pi(i, 2, pathway_matrix, states) for i in range(7)
    ]
    expected_pi = [0, 0, 2 / 15, 0, 0, 0, 0]
    assert np.allclose(obtained_pi, expected_pi)

    assert (
        pattern_mining.get_pi("C", 2, pathway_matrix, states_alternate_names) == 2 / 15
    )

    obtained_pi = [
        pattern_mining.get_pi(i, 0, arnolds_pathway_matrix, arnolds_states)
        for i in range(14)
    ]
    expected_pi = [3 / 5, 0, 1 / 2, 1 / 4, 0, 0, 0, 1, 0, 0, 1 / 2, 1 / 2, 0, 0]
    assert np.allclose(obtained_pi, expected_pi)
    obtained_pi = [
        pattern_mining.get_pi(i, 1, arnolds_pathway_matrix, arnolds_states)
        for i in range(14)
    ]
    expected_pi = [2 / 5, 2 / 3, 0, 1 / 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert np.allclose(obtained_pi, expected_pi)
    obtained_pi = [
        pattern_mining.get_pi(i, 2, arnolds_pathway_matrix, arnolds_states)
        for i in range(14)
    ]
    expected_pi = [0, 1 / 3, 1 / 2, 1 / 4, 1 / 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert np.allclose(obtained_pi, expected_pi)


def test_get_pi_endpoint():
    obtained_endpoints = [
        pattern_mining.get_pi_endpoint(i, pathway_matrix, alphabet, states)
        for i in range(7)
    ]
    expected_endpoints = [0, 8 / 15, 1 / 5, 1, 1, 1, 1]
    assert np.allclose(obtained_endpoints, expected_endpoints)

    obtained_endpoints = [
        pattern_mining.get_pi_endpoint(
            i, arnolds_pathway_matrix, arnolds_alphabet, arnolds_states
        )
        for i in range(14)
    ]
    expected_endpoints = [0, 0, 0, 1 / 4, 1 / 2, 1, 1, 0, 1, 1, 1 / 2, 1 / 2, 1, 1]
    assert np.allclose(obtained_endpoints, expected_endpoints)


def test_hoeffding_bound():
    assert (
        pattern_mining.hoeffding_bound(0, 3, 0.2, pathway_matrix, alphabet, states)
        == False
    )
    assert (
        pattern_mining.hoeffding_bound(1, 5, 0.2, pathway_matrix, alphabet, states)
        == True
    )
    assert (
        pattern_mining.hoeffding_bound(2, 0, 0.2, pathway_matrix, alphabet, states)
        == True
    )
    assert (
        pattern_mining.hoeffding_bound(5, 0, 0.2, pathway_matrix, alphabet, states)
        == False
    )

    assert (
        pattern_mining.hoeffding_bound(
            "B", "F", 0.2, pathway_matrix, alphabet, states_alternate_names
        )
        == True
    )
    assert (
        pattern_mining.hoeffding_bound(
            "D", "A", 0.2, pathway_matrix, alphabet, states_alternate_names
        )
        == False
    )

    assert (
        pattern_mining.hoeffding_bound(
            0, 1, 0.2, arnolds_pathway_matrix, arnolds_alphabet, arnolds_states
        )
        == True
    )
    assert (
        pattern_mining.hoeffding_bound(
            1, 2, 0.2, arnolds_pathway_matrix, arnolds_alphabet, arnolds_states
        )
        == True
    )
    assert (
        pattern_mining.hoeffding_bound(
            0, 1, 0.9, arnolds_pathway_matrix, arnolds_alphabet, arnolds_states
        )
        == False
    )
    assert (
        pattern_mining.hoeffding_bound(
            1, 3, 0.9, arnolds_pathway_matrix, arnolds_alphabet, arnolds_states
        )
        == True
    )


def test_merge_two_states():
    obtained_pathway_matrix, obtained_states = pattern_mining.merge_two_states(
        0, 1, pathway_matrix, states
    )
    expected_pathway_matrix = np.array(
        [
            [
                [0, 30, 0, 0, 0, 0, 0],
                [0, 15, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 5, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 15, 7, 0, 0, 0],
                [0, 0, 0, 0, 0, 5, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 2],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
        ]
    )
    expected_states = ["*", 0, 2, 3, 4, 5, 6]
    assert np.allclose(obtained_pathway_matrix, expected_pathway_matrix)
    assert obtained_states == expected_states

    obtained_pathway_matrix, obtained_states = pattern_mining.merge_two_states(
        2, 6, pathway_matrix, states
    )
    expected_pathway_matrix = np.array(
        [
            [
                [0, 30, 0, 0, 0, 0, 0],
                [0, 0, 15, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 5, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 15, 0, 0, 0],
                [0, 0, 0, 0, 7, 0, 0],
                [0, 0, 0, 0, 0, 0, 5],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 2, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
        ]
    )
    expected_states = ["*", 0, 1, 2, 3, 4, 5]
    assert np.allclose(obtained_pathway_matrix, expected_pathway_matrix)
    assert obtained_states == expected_states

    obtained_pathway_matrix, obtained_states = pattern_mining.merge_two_states(
        0, 1, arnolds_pathway_matrix, arnolds_states
    )
    expected_pathway_matrix = np.array(
        [
            [
                [0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ]
    )
    expected_states = ["*", 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    assert np.allclose(obtained_pathway_matrix, expected_pathway_matrix)
    assert obtained_states == expected_states


def test_check_is_deterministic():
    obtained_nondeterministic_pairs = pattern_mining.check_is_deterministic(
        pathway_matrix, states, alphabet
    )
    expected_nondeterministic_pairs = []
    assert obtained_nondeterministic_pairs == expected_nondeterministic_pairs

    obtained_pathway_matrix, obtained_states = pattern_mining.merge_two_states(
        0, 1, pathway_matrix, states
    )
    obtained_nondeterministic_pairs = pattern_mining.check_is_deterministic(
        obtained_pathway_matrix, obtained_states, alphabet
    )
    expected_nondeterministic_pairs = [(2, 3)]
    assert obtained_nondeterministic_pairs == expected_nondeterministic_pairs

    obtained_pathway_matrix, obtained_states = pattern_mining.merge_two_states(
        2, 6, pathway_matrix, states
    )
    obtained_nondeterministic_pairs = pattern_mining.check_is_deterministic(
        obtained_pathway_matrix, obtained_states, alphabet
    )
    expected_nondeterministic_pairs = []
    assert obtained_nondeterministic_pairs == expected_nondeterministic_pairs

    obtained_pathway_matrix, obtained_states = pattern_mining.merge_two_states(
        0, 2, pathway_matrix, states
    )
    obtained_nondeterministic_pairs = pattern_mining.check_is_deterministic(
        obtained_pathway_matrix, obtained_states, alphabet
    )
    expected_nondeterministic_pairs = [(1, 4), (0, 5)]
    assert obtained_nondeterministic_pairs == expected_nondeterministic_pairs

    obtained_nondeterministic_pairs = pattern_mining.check_is_deterministic(
        arnolds_pathway_matrix, arnolds_states, arnolds_alphabet
    )
    expected_nondeterministic_pairs = []
    assert obtained_nondeterministic_pairs == expected_nondeterministic_pairs

    obtained_pathway_matrix, obtained_states = pattern_mining.merge_two_states(
        0, 1, arnolds_pathway_matrix, arnolds_states
    )
    obtained_nondeterministic_pairs = pattern_mining.check_is_deterministic(
        obtained_pathway_matrix, obtained_states, arnolds_alphabet
    )
    expected_nondeterministic_pairs = [(2, 3)]
    assert obtained_nondeterministic_pairs == expected_nondeterministic_pairs

    obtained_pathway_matrix, obtained_states = pattern_mining.merge_two_states(
        0, 3, arnolds_pathway_matrix, arnolds_states
    )
    obtained_nondeterministic_pairs = pattern_mining.check_is_deterministic(
        obtained_pathway_matrix, obtained_states, arnolds_alphabet
    )
    expected_nondeterministic_pairs = [(1, 5), (2, 6)]
    assert obtained_nondeterministic_pairs == expected_nondeterministic_pairs


def test_check_is_deterministic_with_multiple_pairs():
    wrong_pathway_matrix = np.array(
        [
            [
                [0, 32, 0, 0, 0, 0, 0, 0],
                [0, 0, 15, 0, 0, 0, 0, 2],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 5, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 15, 0, 0, 0],
                [0, 0, 0, 7, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 5, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 2],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ]
    )
    obtained_nondeterministic_pairs = pattern_mining.check_is_deterministic(
        wrong_pathway_matrix, states, alphabet
    )
    expected_nondeterministic_pairs = [(1, 6), (1, 4), (2, 4)]
    assert obtained_nondeterministic_pairs == expected_nondeterministic_pairs


def test_recursive_merge_two_states():
    # Test where states are merged recursively
    (
        obtained_matrix,
        obtained_states,
        obtained_recursive_merge,
    ) = pattern_mining.recursive_merge_two_states(
        1, 2, pathway_matrix, states, 0.2, alphabet
    )
    expected_matrix = np.array(
        [
            [
                [0, 30, 0, 0, 0, 0],
                [0, 0, 15, 0, 0, 0],
                [0, 0, 0, 0, 5, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 15, 0, 0, 0],
                [0, 0, 0, 12, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 2],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
        ]
    )
    expected_states = ["*", 0, 1, 3, 4, 6]
    expected_recursive_merge = True
    assert np.allclose(obtained_matrix, expected_matrix)
    assert obtained_states == expected_states
    assert obtained_recursive_merge == expected_recursive_merge

    (
        obtained_matrix,
        obtained_states,
        obtained_recursive_merge,
    ) = pattern_mining.recursive_merge_two_states(
        0, 1, arnolds_pathway_matrix, arnolds_states, 0.2, alphabet
    )
    expected_matrix = np.array(
        [
            [
                [0, 10, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 6, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ]
    )
    expected_states = ["*", 0, 2, 4, 5, 6, 7, 8, 9, 12]
    expected_recursive_merge = True
    assert np.allclose(obtained_matrix, expected_matrix)
    assert obtained_states == expected_states
    assert obtained_recursive_merge == expected_recursive_merge

    # Test where Hoeffding's Bound fails during recursive merge
    (
        obtained_matrix,
        obtained_states,
        obtained_recursive_merge,
    ) = pattern_mining.recursive_merge_two_states(
        0, 3, arnolds_pathway_matrix, arnolds_states, 0.9, arnolds_alphabet
    )
    assert np.allclose(obtained_matrix, arnolds_pathway_matrix)
    assert obtained_states == arnolds_states
    assert obtained_recursive_merge == False


def test_get_blue_states():
    obtained_blue_states = pattern_mining.get_blue_states(pathway_matrix, example_red_states, states)
    expected_blue_states = [1, 2]
    assert obtained_blue_states == expected_blue_states

    obtained_blue_states = pattern_mining.get_blue_states(arnolds_pathway_matrix, example_red_states, arnolds_states)
    expected_blue_states = [1, 2]
    assert obtained_blue_states == expected_blue_states


arnolds_pathway_matrix_after_merges = np.array(
    [
        [
            [ 0, 10,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  8,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2],
            [ 0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
        ],
       [
           [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  4,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
        ],
        [
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  2,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  2,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
        ]
    ]
)
arnolds_states_after_merges = ['*', 0, 1, 3, 4, 5, 6, 7, 8, 9, 12]
arnolds_red_states_after_merges = [0, 1, 12]

def test_recursive_merge_two_states_higuera():
    # Test where states are merged recursively
    (
        obtained_matrix,
        obtained_states,
        obtained_recursive_merge,
        red_states,
    ) = pattern_mining.recursive_merge_two_states_higuera(
        1, 2, pathway_matrix, states, 0.2, alphabet, example_red_states
    )
    expected_matrix = np.array(
        [
            [
                [0, 30, 0, 0, 0, 0],
                [0, 0, 15, 0, 0, 0],
                [0, 0, 0, 0, 5, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 15, 0, 0, 0],
                [0, 0, 0, 12, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 2],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
        ]
    )
    expected_states = ["*", 0, 1, 3, 4, 6]
    expected_recursive_merge = True
    assert np.allclose(obtained_matrix, expected_matrix)
    assert obtained_states == expected_states
    assert obtained_recursive_merge == expected_recursive_merge
    assert red_states == example_red_states

    (
        obtained_matrix,
        obtained_states,
        obtained_recursive_merge,
        red_states,
    ) = pattern_mining.recursive_merge_two_states_higuera(
        0, 1, arnolds_pathway_matrix, arnolds_states, 0.2, alphabet, example_red_states
    )
    expected_matrix = np.array(
        [
            [
                [0, 10, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 6, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ]
    )
    expected_states = ["*", 0, 2, 4, 5, 6, 7, 8, 9, 12]
    expected_recursive_merge = True
    assert np.allclose(obtained_matrix, expected_matrix)
    assert obtained_states == expected_states
    assert obtained_recursive_merge == expected_recursive_merge
    assert red_states == example_red_states

    # Test where Hoeffding's Bound fails during recursive merge
    (
        obtained_matrix,
        obtained_states,
        obtained_recursive_merge,
        red_states,
    ) = pattern_mining.recursive_merge_two_states_higuera(
        0, 3, arnolds_pathway_matrix, arnolds_states, 0.9, arnolds_alphabet, example_red_states
    )
    assert np.allclose(obtained_matrix, arnolds_pathway_matrix)
    assert obtained_states == arnolds_states
    assert obtained_recursive_merge == False
    assert red_states == example_red_states

    # Test where a red state gets merged during the recursive merge
    expected_matrix = np.array(
        [
            [
                [ 0, 10,  0,  0,  0,  0,  0],
                [ 0,  0,  8,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  3,  0,  0],
                [ 0,  0,  0,  0,  0,  1,  0],
                [ 0,  0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  0,  0]
            ],
            [
                [ 0,  0,  0,  0,  0,  0,  0],
                [ 0,  4,  0,  0,  0,  0,  0],
                [ 0,  0,  5,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  0,  0]
            ],
            [
                [ 0,  0,  0,  0,  0,  0,  0],
                [ 0,  0,  2,  0,  0,  0,  0],
                [ 0,  0,  0,  3,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  0,  1],
                [ 0,  0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  0,  0]
            ]
        ]
    )
    expected_states = ['*', 0, 1, 4, 5, 8, 9]
    expected_recursive_merge = True
    expected_red_states = [0, 1, 5]
    (
        obtained_matrix,
        obtained_states,
        obtained_recursive_merge,
        red_states,
    ) = pattern_mining.recursive_merge_two_states_higuera(
        1, 3, arnolds_pathway_matrix_after_merges, arnolds_states_after_merges, 0.9, arnolds_alphabet, arnolds_red_states_after_merges
    )
    assert np.allclose(obtained_matrix, expected_matrix)
    assert obtained_states == expected_states
    assert obtained_recursive_merge == expected_recursive_merge
    assert red_states == expected_red_states


def test_get_pairs_to_check():
    obtained_pairs = pattern_mining.get_pairs_to_check(states)
    expected_pairs = [
        (1, 0),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
        (3, 2),
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 3),
        (5, 0),
        (5, 1),
        (5, 2),
        (5, 3),
        (5, 4),
        (6, 0),
        (6, 1),
        (6, 2),
        (6, 3),
        (6, 4),
        (6, 5),
    ]
    assert obtained_pairs == expected_pairs

    obtained_pairs = pattern_mining.get_pairs_to_check(arnolds_states)
    expected_pairs = [
        (1, 0),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
        (3, 2),
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 3),
        (5, 0),
        (5, 1),
        (5, 2),
        (5, 3),
        (5, 4),
        (6, 0),
        (6, 1),
        (6, 2),
        (6, 3),
        (6, 4),
        (6, 5),
        (7, 0),
        (7, 1),
        (7, 2),
        (7, 3),
        (7, 4),
        (7, 5),
        (7, 6),
        (8, 0),
        (8, 1),
        (8, 2),
        (8, 3),
        (8, 4),
        (8, 5),
        (8, 6),
        (8, 7),
        (9, 0),
        (9, 1),
        (9, 2),
        (9, 3),
        (9, 4),
        (9, 5),
        (9, 6),
        (9, 7),
        (9, 8),
        (10, 0),
        (10, 1),
        (10, 2),
        (10, 3),
        (10, 4),
        (10, 5),
        (10, 6),
        (10, 7),
        (10, 8),
        (10, 9),
        (11, 0),
        (11, 1),
        (11, 2),
        (11, 3),
        (11, 4),
        (11, 5),
        (11, 6),
        (11, 7),
        (11, 8),
        (11, 9),
        (11, 10),
        (12, 0),
        (12, 1),
        (12, 2),
        (12, 3),
        (12, 4),
        (12, 5),
        (12, 6),
        (12, 7),
        (12, 8),
        (12, 9),
        (12, 10),
        (12, 11),
        (13, 0),
        (13, 1),
        (13, 2),
        (13, 3),
        (13, 4),
        (13, 5),
        (13, 6),
        (13, 7),
        (13, 8),
        (13, 9),
        (13, 10),
        (13, 11),
        (13, 12),
    ]
    assert obtained_pairs == expected_pairs


def test_alergia():
    obtained_matrix, obtained_states, obtained_merges = pattern_mining.alergia(
        pathway_matrix, states, alphabet, 0.2
    )
    expected_matrix = np.array(
        [
            [
                [0, 30, 0, 0],
                [0, 0, 15, 0],
                [0, 0, 5, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 15, 0],
                [0, 0, 0, 12],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 2, 0],
                [0, 0, 0, 0],
            ],
        ]
    )
    expected_states = ["*", 0, 1, 3]
    expected_merges = 3
    assert np.allclose(obtained_matrix, expected_matrix)
    assert obtained_states == expected_states
    assert obtained_merges == expected_merges

    obtained_matrix, obtained_states, obtained_merges = pattern_mining.alergia(
        arnolds_pathway_matrix, arnolds_states, arnolds_alphabet, 0.2
    )
    expected_matrix = np.array(
        [
            [
                [0, 10],
                [0, 12],
            ],
            [
                [0, 0],
                [0, 9],
            ],
            [
                [0, 0],
                [0, 6],
            ],
        ]
    )
    expected_states = ["*", 0]
    expected_merges = 3
    assert np.allclose(obtained_matrix, expected_matrix)
    assert obtained_states == expected_states
    assert obtained_merges == expected_merges

    obtained_matrix, obtained_states, obtained_merges = pattern_mining.alergia(
        arnolds_pathway_matrix, arnolds_states, arnolds_alphabet, 0.9
    )
    expected_matrix = np.array(
        [
            [
                [0, 10, 0],
                [0, 0, 9],
                [0, 0, 3],
            ],
            [
                [0, 0, 0],
                [0, 4, 0],
                [0, 0, 5],
            ],
            [
                [0, 0, 0],
                [0, 0, 3],
                [0, 3, 0],
            ],
        ]
    )
    expected_states = ["*", 0, 1]
    expected_merges = 5
    assert np.allclose(obtained_matrix, expected_matrix)
    assert obtained_states == expected_states
    assert obtained_merges == expected_merges

    obtained_matrix, obtained_states, obtained_merges = pattern_mining.alergia(
        arnolds_pathway_matrix, arnolds_states, arnolds_alphabet, 0.9, method="Higuera"
    )
    expected_matrix = np.array(
        [
            [
                [ 0, 10,  0,  0],
                [ 0,  0,  9,  0],
                [ 0,  0,  0,  3],
                [ 0,  0,  0,  0]
            ],
            [
                [ 0,  0,  0,  0],
                [ 0,  4,  0,  0],
                [ 0,  0,  5,  0],
                [ 0,  0,  0,  0]
            ],
            [
                [ 0,  0,  0,  0],
                [ 0,  0,  3,  0],
                [ 0,  3,  0,  0],
                [ 0,  0,  0,  0]
            ]
        ]
    )
    expected_states = ["*", 0, 1, 5]
    expected_merges = 4
    assert np.allclose(obtained_matrix, expected_matrix)
    assert obtained_states == expected_states
    assert obtained_merges == expected_merges


final_pathway_matrix = np.array(
    [
        [
            [0, 30, 0, 0],
            [0, 0, 15, 0],
            [0, 0, 5, 0],
            [0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0],
            [0, 0, 15, 0],
            [0, 0, 0, 12],
            [0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 0],
        ],
    ]
)
final_states = ["*", 0, 1, 3]

final_arnolds_pathway_matrix = np.array(
    [
        [
            [0, 10, 0],
            [0, 0, 9],
            [0, 0, 3],
        ],
        [
            [0, 0, 0],
            [0, 4, 0],
            [0, 0, 5],
        ],
        [
            [0, 0, 0],
            [0, 0, 3],
            [0, 3, 0],
        ],
    ]
)
final_arnolds_states = ["*", 0, 1]

final_arnolds_p_matrix_point9 = np.array(
    [
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 9 / 17, 0.0],
            [0.0, 0.0, 0.0, 3 / 17],
            [0.0, 0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 4 / 17, 0.0, 0.0],
            [0.0, 0.0, 5 / 17, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3 / 17, 0.0],
            [0.0, 3 / 17, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
    ]
)


def test_probability_transition_matrix():
    obtained_matrix = pattern_mining.probability_transition_matrix(
        final_pathway_matrix, final_states, alphabet
    )
    expected_matrix = np.array(
        [
            [
                [0, 1, 0, 0],
                [0, 0, 1 / 2, 0],
                [0, 0, 5 / 37, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 1 / 2, 0],
                [0, 0, 0, 12 / 37],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 2 / 37, 0],
                [0, 0, 0, 0],
            ],
        ]
    )
    assert np.allclose(obtained_matrix, expected_matrix)

    obtained_matrix = pattern_mining.probability_transition_matrix(
        final_arnolds_pathway_matrix, final_arnolds_states, arnolds_alphabet
    )
    expected_matrix = np.array(
        [
            [
                [0, 1, 0],
                [0, 0, 9 / 17],
                [0, 0, 3 / 20],
            ],
            [
                [0, 0, 0],
                [0, 4 / 17, 0],
                [0, 0, 5 / 20],
            ],
            [
                [0, 0, 0],
                [0, 0, 3 / 17],
                [0, 3 / 20, 0],
            ],
        ]
    )
    assert np.allclose(obtained_matrix, expected_matrix)


jacquemont_p_matrix = np.array(
    [
        [
            [0, 0.23, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1.0],
            [0, 0.21, 0, 0],
        ],
        [
            [0, 0, 0.31, 0],
            [1.0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0.16, 0],
        ],
        [
            [0.23, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0.26],
        ],
    ]
)
jacquemont_sequences = [
    "ab",
    "bac",
    "baba",
    "abbac",
    "abbaab",
    "ba",
    "abcc",
    "bacc",
    "abccc",
    "baabba",
    "abc",
    "baab",
    "ababc",
    "babac",
    "babaabc",
]
jacquemont_alphabet = ["a", "b", "c"]
jacquemont_states = [
    "*",
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
]


def test_probability_estimate_of_symbol():
    obtained_vector = pattern_mining.probability_estimate_of_symbol(
        jacquemont_p_matrix, "c", jacquemont_alphabet
    )
    expected_vector = np.array([0.47068936, 0.47068936, 0.42719615, 0.42719615])
    assert np.allclose(obtained_vector, expected_vector)


def test_probability_estimate_of_pattern():
    obtained_vector = pattern_mining.probability_estimate_of_pattern(
        jacquemont_p_matrix, "cc", jacquemont_alphabet
    )
    expected_vector = np.array([0.21552208, 0.21552208, 0.1861079, 0.1861079])
    assert np.allclose(obtained_vector, expected_vector)


def test_probability_estimate_of_exact_sequence():
    obtained_probability = pattern_mining.probability_estimate_of_exact_sequence(
        final_arnolds_p_matrix_point9, "ABC", arnolds_alphabet
    )
    expected_probability = 0.0016163599573759896
    assert np.allclose(obtained_probability, expected_probability)


def test_probability_sequence_contains_letter_at_distance_theta():
    obtained_vector = (
        pattern_mining.probability_sequence_contains_letter_at_distance_theta(
            jacquemont_p_matrix, "a", 2, jacquemont_alphabet
        )
    )
    expected_vector = np.array([0.201467, 0.3629, 0.2146, 0.137696])
    assert np.allclose(obtained_vector, expected_vector)


def test_probability_to_encounter_a_pattern_at_a_distance_theta():
    obtained_vector = (
        pattern_mining.probability_to_encounter_a_pattern_at_a_distance_theta(
            jacquemont_p_matrix, "ab", 2, jacquemont_alphabet
        )
    )
    expected_vector = np.array([0.165817, 0.2079, 0.1346, 0.116896])
    assert np.allclose(obtained_vector, expected_vector)

    obtained_vector = (
        pattern_mining.probability_to_encounter_a_pattern_at_a_distance_theta(
            jacquemont_p_matrix, "abc", 2, jacquemont_alphabet
        )
    )
    expected_vector = np.array([0.0773778, 0.0949411, 0.06185016, 0.0546305])
    assert np.allclose(obtained_vector, expected_vector)


def test_proportion_constraint():
    assert (
        pattern_mining.proportion_constraint(
            jacquemont_p_matrix, "cc", jacquemont_alphabet, jacquemont_sequences, 0.05
        )
        == True
    )

    assert (
        pattern_mining.proportion_constraint(
            jacquemont_p_matrix, "bcc", jacquemont_alphabet, jacquemont_sequences, 0.05
        )
        == False
    )

    assert (
        pattern_mining.proportion_constraint(
            final_arnolds_p_matrix_point9,
            "AAC",
            arnolds_alphabet,
            arnolds_sequences,
            0.33,
            p_value="sequence",
        )
        == True
    )

    assert (
        pattern_mining.proportion_constraint(
            final_arnolds_p_matrix_point9,
            "ABC",
            arnolds_alphabet,
            arnolds_sequences,
            0.33,
            p_value="sequence",
        )
        == False
    )


def test_probability_sequence_contains_digram():
    obtained_vector = pattern_mining.probability_sequence_contains_digram(
        jacquemont_p_matrix, "ab", jacquemont_alphabet
    )
    expected_vector = np.array([0.2987013, 0, 0.29800281, 0.28378378])
    assert np.allclose(obtained_vector, expected_vector)


def test_string_enumerator():
    obtained_strings = pattern_mining.string_enumerator(jacquemont_alphabet, 2)
    expected_strings = [
        "a",
        "b",
        "c",
        "aa",
        "ab",
        "ac",
        "ba",
        "bb",
        "bc",
        "ca",
        "cb",
        "cc",
    ]
    assert obtained_strings == expected_strings


def test_string_probabilities():
    obtained_probabilities = pattern_mining.string_probabilities(
        final_arnolds_p_matrix_point9, arnolds_alphabet, ["A", "B", "C"]
    )
    expected_probabilities = [
        ("A", 0.18685121107266434),
        ("B", 0.013840830449826992),
        ("C", 0.06228373702422145),
    ]
    assert obtained_probabilities[0][1] == expected_probabilities[0][1]
