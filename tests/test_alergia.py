import numpy as np
import pattern_mining as pm


def test_get_blue_states_simple_example(simple_pta):
    obtained_blue_states = pm.get_blue_states(
        simple_pta.pathway_matrix, simple_pta.red_states, simple_pta.states
    )
    expected_blue_states = [1, 2]
    assert obtained_blue_states == expected_blue_states


def test_get_blue_states_arnolds_example(arnolds_example):
    obtained_blue_states = pm.get_blue_states(
        arnolds_example.pathway_matrix, arnolds_example.red_states, arnolds_example.states
    )
    expected_blue_states = [1, 2]
    assert obtained_blue_states == expected_blue_states


def test_get_pairs_to_check_simple_example(simple_pta):
    obtained_pairs = pm.get_pairs_to_check(simple_pta.states)
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


def test_get_pairs_to_check_arnolds_example(arnolds_example):
    obtained_pairs = pm.get_pairs_to_check(arnolds_example.states)
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


def test_alergia_simple_example(simple_pta):
    obtained_matrix, obtained_states, obtained_merges = pm.alergia(
        simple_pta.pathway_matrix, simple_pta.states, simple_pta.alphabet, 0.2
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


def test_alergia_arnolds_example_alpha_point_two(arnolds_example):
    obtained_matrix, obtained_states, obtained_merges = pm.alergia(
        arnolds_example.pathway_matrix, arnolds_example.states, arnolds_example.alphabet, 0.2
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


def test_alergia_arnolds_example_alpha_point_nine(arnolds_example):
    obtained_matrix, obtained_states, obtained_merges = pm.alergia(
        arnolds_example.pathway_matrix, arnolds_example.states, arnolds_example.alphabet, 0.9
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
        arnolds_example.pathway_matrix, arnolds_example.states, arnolds_example.alphabet, 0.9, method="Higuera"
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
