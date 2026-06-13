import numpy as np
import pattern_mining as pm

alphabet = ["0", "1", "2"]

arnolds_alphabet = ["A", "B", "C"]

states = ["*", 0, 1, 2, 3, 4, 5, 6]
example_red_states = [0]

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


def test_get_blue_states():
    obtained_blue_states = pm.get_blue_states(
        pathway_matrix, example_red_states, states
    )
    expected_blue_states = [1, 2]
    assert obtained_blue_states == expected_blue_states

    obtained_blue_states = pm.get_blue_states(
        arnolds_pathway_matrix, example_red_states, arnolds_states
    )
    expected_blue_states = [1, 2]
    assert obtained_blue_states == expected_blue_states


def test_get_pairs_to_check():
    obtained_pairs = pm.get_pairs_to_check(states)
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

    obtained_pairs = pm.get_pairs_to_check(arnolds_states)
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
    obtained_matrix, obtained_states, obtained_merges = pm.alergia(
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

    obtained_matrix, obtained_states, obtained_merges = pm.alergia(
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

    obtained_matrix, obtained_states, obtained_merges = pm.alergia(
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

    obtained_matrix, obtained_states, obtained_merges = pm.alergia(
        arnolds_pathway_matrix, arnolds_states, arnolds_alphabet, 0.9, method="Higuera"
    )
    expected_matrix = np.array(
        [
            [[0, 10, 0, 0], [0, 0, 8, 0], [0, 0, 0, 2], [0, 0, 0, 2]],
            [[0, 0, 0, 0], [0, 4, 0, 0], [0, 0, 0, 4], [0, 0, 0, 1]],
            [[0, 0, 0, 0], [0, 0, 3, 0], [0, 2, 0, 0], [0, 0, 0, 1]],
        ]
    )
    expected_states = ["*", 0, 1, 3]
    expected_merges = 7
    assert np.allclose(obtained_matrix, expected_matrix)
    assert obtained_states == expected_states
    assert obtained_merges == expected_merges
