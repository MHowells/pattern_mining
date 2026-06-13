import numpy as np
import pattern_mining as pm

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
    obtained_alphabet = pm.get_alphabet(list_of_sequences)
    expected_alphabet = ["0", "1", "2"]
    assert obtained_alphabet == expected_alphabet

    obtained_alphabet = pm.get_alphabet(arnolds_sequences)
    expected_alphabet = ["A", "B", "C"]
    assert obtained_alphabet == expected_alphabet


def test_get_state_paths():
    obtained_state_paths_breadth = pm.get_state_paths(
        list_of_sequences, "breadth"
    )
    expected_state_paths_breadth = ["", "0", "1", "01", "10", "11", "12"]
    obtained_state_paths_depth = pm.get_state_paths(
        list_of_sequences, "depth"
    )
    expected_state_paths_depth = ["", "0", "1", "01", "10", "11", "12"]
    assert obtained_state_paths_breadth == expected_state_paths_breadth
    assert obtained_state_paths_depth == expected_state_paths_depth

    obtained_state_paths_breadth = pm.get_state_paths(
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
    obtained_state_paths_depth = pm.get_state_paths(
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
    obtained_transition_matrix = pm.get_transition_matrix(
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

    obtained_transition_matrix = pm.get_transition_matrix(
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
    obtained_states = pm.get_initial_states(list_of_sequences)
    expected_states = ["*", 0, 1, 2, 3, 4, 5, 6]
    assert obtained_states == expected_states

    obtained_states = pm.get_initial_states(arnolds_sequences)
    expected_states = ["*", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    assert obtained_states == expected_states