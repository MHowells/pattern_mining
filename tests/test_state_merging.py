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

arnolds_pathway_matrix_after_merges = np.array(
    [
        [
            [0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
    ]
)
arnolds_states_after_merges = ["*", 0, 1, 3, 4, 5, 6, 7, 8, 9, 12]
arnolds_red_states_after_merges = [0, 1, 12]


def test_hoeffding_bound():
    assert (
        pm.hoeffding_bound(0, 3, 0.2, pathway_matrix, alphabet, states)
        == False
    )
    assert (
        pm.hoeffding_bound(1, 5, 0.2, pathway_matrix, alphabet, states)
        == True
    )
    assert (
        pm.hoeffding_bound(2, 0, 0.2, pathway_matrix, alphabet, states)
        == True
    )
    assert (
        pm.hoeffding_bound(5, 0, 0.2, pathway_matrix, alphabet, states)
        == False
    )

    assert (
        pm.hoeffding_bound(
            "B", "F", 0.2, pathway_matrix, alphabet, states_alternate_names
        )
        == True
    )
    assert (
        pm.hoeffding_bound(
            "D", "A", 0.2, pathway_matrix, alphabet, states_alternate_names
        )
        == False
    )

    assert (
        pm.hoeffding_bound(
            0, 1, 0.2, arnolds_pathway_matrix, arnolds_alphabet, arnolds_states
        )
        == True
    )
    assert (
        pm.hoeffding_bound(
            1, 2, 0.2, arnolds_pathway_matrix, arnolds_alphabet, arnolds_states
        )
        == True
    )
    assert (
        pm.hoeffding_bound(
            0, 1, 0.9, arnolds_pathway_matrix, arnolds_alphabet, arnolds_states
        )
        == False
    )
    assert (
        pm.hoeffding_bound(
            1, 3, 0.9, arnolds_pathway_matrix, arnolds_alphabet, arnolds_states
        )
        == True
    )


def test_merge_two_states():
    obtained_pathway_matrix, obtained_states = pm.merge_two_states(
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

    obtained_pathway_matrix, obtained_states = pm.merge_two_states(
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

    obtained_pathway_matrix, obtained_states = pm.merge_two_states(
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
    obtained_nondeterministic_pairs = pm.check_is_deterministic(
        pathway_matrix, states, alphabet
    )
    expected_nondeterministic_pairs = []
    assert obtained_nondeterministic_pairs == expected_nondeterministic_pairs

    obtained_pathway_matrix, obtained_states = pm.merge_two_states(
        0, 1, pathway_matrix, states
    )
    obtained_nondeterministic_pairs = pm.check_is_deterministic(
        obtained_pathway_matrix, obtained_states, alphabet
    )
    expected_nondeterministic_pairs = [(2, 3)]
    assert obtained_nondeterministic_pairs == expected_nondeterministic_pairs

    obtained_pathway_matrix, obtained_states = pm.merge_two_states(
        2, 6, pathway_matrix, states
    )
    obtained_nondeterministic_pairs = pm.check_is_deterministic(
        obtained_pathway_matrix, obtained_states, alphabet
    )
    expected_nondeterministic_pairs = []
    assert obtained_nondeterministic_pairs == expected_nondeterministic_pairs

    obtained_pathway_matrix, obtained_states = pm.merge_two_states(
        0, 2, pathway_matrix, states
    )
    obtained_nondeterministic_pairs = pm.check_is_deterministic(
        obtained_pathway_matrix, obtained_states, alphabet
    )
    expected_nondeterministic_pairs = [(1, 4), (0, 5)]
    assert obtained_nondeterministic_pairs == expected_nondeterministic_pairs

    obtained_nondeterministic_pairs = pm.check_is_deterministic(
        arnolds_pathway_matrix, arnolds_states, arnolds_alphabet
    )
    expected_nondeterministic_pairs = []
    assert obtained_nondeterministic_pairs == expected_nondeterministic_pairs

    obtained_pathway_matrix, obtained_states = pm.merge_two_states(
        0, 1, arnolds_pathway_matrix, arnolds_states
    )
    obtained_nondeterministic_pairs = pm.check_is_deterministic(
        obtained_pathway_matrix, obtained_states, arnolds_alphabet
    )
    expected_nondeterministic_pairs = [(2, 3)]
    assert obtained_nondeterministic_pairs == expected_nondeterministic_pairs

    obtained_pathway_matrix, obtained_states = pm.merge_two_states(
        0, 3, arnolds_pathway_matrix, arnolds_states
    )
    obtained_nondeterministic_pairs = pm.check_is_deterministic(
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
    obtained_nondeterministic_pairs = pm.check_is_deterministic(
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
    ) = pm.recursive_merge_two_states(
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
    ) = pm.recursive_merge_two_states(
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
    ) = pm.recursive_merge_two_states(
        0, 3, arnolds_pathway_matrix, arnolds_states, 0.9, arnolds_alphabet
    )
    assert np.allclose(obtained_matrix, arnolds_pathway_matrix)
    assert obtained_states == arnolds_states
    assert obtained_recursive_merge == False

    # Test where states are merged recursively (Higuera)
    (
        obtained_matrix,
        obtained_states,
        obtained_recursive_merge,
        red_states,
    ) = pm.recursive_merge_two_states(
        1,
        2,
        pathway_matrix,
        states,
        0.2,
        alphabet,
        example_red_states,
        method="Higuera",
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
    ) = pm.recursive_merge_two_states(
        0,
        1,
        arnolds_pathway_matrix,
        arnolds_states,
        0.2,
        alphabet,
        example_red_states,
        method="Higuera",
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

    # Test where Hoeffding's Bound fails during recursive merge (Higuera)
    (
        obtained_matrix,
        obtained_states,
        obtained_recursive_merge,
        red_states,
    ) = pm.recursive_merge_two_states(
        0,
        3,
        arnolds_pathway_matrix,
        arnolds_states,
        0.9,
        arnolds_alphabet,
        example_red_states,
        method="Higuera",
    )
    assert np.allclose(obtained_matrix, arnolds_pathway_matrix)
    assert obtained_states == arnolds_states
    assert obtained_recursive_merge == False
    assert red_states == example_red_states

    # Test where a red state gets merged during the recursive merge (Higuera)
    expected_matrix = np.array(
        [
            [
                [0, 10, 0, 0, 0, 0, 0],
                [0, 0, 8, 0, 0, 0, 0],
                [0, 0, 0, 0, 3, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 4, 0, 0, 0, 0, 0],
                [0, 0, 5, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 2, 0, 0, 0, 0],
                [0, 0, 0, 3, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
        ]
    )
    expected_states = ["*", 0, 1, 4, 5, 8, 9]
    expected_recursive_merge = True
    expected_red_states = [0, 1, 5]
    (
        obtained_matrix,
        obtained_states,
        obtained_recursive_merge,
        red_states,
    ) = pm.recursive_merge_two_states(
        1,
        3,
        arnolds_pathway_matrix_after_merges,
        arnolds_states_after_merges,
        0.9,
        arnolds_alphabet,
        arnolds_red_states_after_merges,
        method="Higuera",
    )
    assert np.allclose(obtained_matrix, expected_matrix)
    assert obtained_states == expected_states
    assert obtained_recursive_merge == expected_recursive_merge
    assert red_states == expected_red_states