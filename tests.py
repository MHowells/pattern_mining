import numpy as np
import pattern_mining

states = [0, 1, 2, 3, 4, 5, 6]
alphabet = [0, 1, 2]
pathway_matrix = np.array([
    [
        [0, 15, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 5, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ],
    [
        [0, 0, 0, 15, 0, 0, 0],
        [0, 0, 7, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 5, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ],
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
])

def test_get_n():
    assert pattern_mining.get_n(0, pathway_matrix) == 30
    assert pattern_mining.get_n(1, pathway_matrix) == 15
    assert pattern_mining.get_n(2, pathway_matrix) == 7
    assert pattern_mining.get_n(3, pathway_matrix) == 15
    assert pattern_mining.get_n(4, pathway_matrix) == 5
    assert pattern_mining.get_n(5, pathway_matrix) == 5
    assert pattern_mining.get_n(6, pathway_matrix) == 2


def test_get_pi():
    assert pattern_mining.get_pi(0, 0, pathway_matrix) == 1/2
    assert pattern_mining.get_pi(0, 1, pathway_matrix) == 1/2
    assert pattern_mining.get_pi(0, 2, pathway_matrix) == 0
    assert pattern_mining.get_pi(1, 0, pathway_matrix) == 0
    assert pattern_mining.get_pi(1, 1, pathway_matrix) == 7/15
    assert pattern_mining.get_pi(1, 2, pathway_matrix) == 0
    assert pattern_mining.get_pi(2, 0, pathway_matrix) == 0
    assert pattern_mining.get_pi(2, 1, pathway_matrix) == 0
    assert pattern_mining.get_pi(2, 2, pathway_matrix) == 0
    assert pattern_mining.get_pi(3, 0, pathway_matrix) == 1/3
    assert pattern_mining.get_pi(3, 1, pathway_matrix) == 1/3
    assert pattern_mining.get_pi(3, 2, pathway_matrix) == 2/15
    assert pattern_mining.get_pi(4, 0, pathway_matrix) == 0
    assert pattern_mining.get_pi(4, 1, pathway_matrix) == 0
    assert pattern_mining.get_pi(4, 2, pathway_matrix) == 0
    assert pattern_mining.get_pi(5, 0, pathway_matrix) == 0
    assert pattern_mining.get_pi(5, 1, pathway_matrix) == 0
    assert pattern_mining.get_pi(5, 2, pathway_matrix) == 0
    assert pattern_mining.get_pi(6, 0, pathway_matrix) == 0
    assert pattern_mining.get_pi(6, 1, pathway_matrix) == 0
    assert pattern_mining.get_pi(6, 2, pathway_matrix) == 0


def test_get_pi_endpoint():
    obtained_endpoints = [pattern_mining.get_pi_endpoint(i, pathway_matrix, alphabet) for i in range(7)]
    expected_endpoints = [0, 8/15, 1, 1/5, 1, 1, 1]
    assert np.allclose(obtained_endpoints, expected_endpoints)


def test_hoeffding_bound():
    assert pattern_mining.hoeffding_bound(0, 1, 0.2, pathway_matrix, alphabet) == True
    assert pattern_mining.hoeffding_bound(1, 5, 0.2, pathway_matrix, alphabet) == True
    assert pattern_mining.hoeffding_bound(2, 0, 0.2, pathway_matrix, alphabet) == False
    assert pattern_mining.hoeffding_bound(5, 0, 0.2, pathway_matrix, alphabet) == False


def test_merge_two_states():
    obtained_pathway_matrix, obtained_states = pattern_mining.merge_two_states(0, 1, pathway_matrix, states)
    expected_pathway_matrix = np.array([
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 5, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 15],
        ],
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 5, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [7, 15, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    ])
    expected_states = [2, 3, 4, 5, 6, 7]
    assert np.allclose(obtained_pathway_matrix, expected_pathway_matrix)
    assert obtained_states == expected_states

    obtained_pathway_matrix, obtained_states = pattern_mining.merge_two_states(2, 6, pathway_matrix, states)
    expected_pathway_matrix = np.array([
        [
            [0, 15, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 5, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 15, 0, 0, 0],
            [0, 0, 0, 0, 0, 7],
            [0, 0, 0, 0, 5, 0],
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
        ]
    ])
    expected_states = [0, 1, 3, 4, 5, 7]
    assert np.allclose(obtained_pathway_matrix, expected_pathway_matrix)
    assert obtained_states == expected_states


def test_check_is_deterministic():
    obtained_nondeterministic_pairs = pattern_mining.check_is_deterministic(pathway_matrix, states, alphabet)
    expected_nondeterministic_pairs = []
    assert obtained_nondeterministic_pairs == expected_nondeterministic_pairs

    obtained_pathway_matrix, obtained_states = pattern_mining.merge_two_states(0, 1, pathway_matrix, states)
    obtained_nondeterministic_pairs = pattern_mining.check_is_deterministic(obtained_pathway_matrix, obtained_states, alphabet)
    expected_nondeterministic_pairs = [(2, 3)]
    assert obtained_nondeterministic_pairs == expected_nondeterministic_pairs

    obtained_pathway_matrix, obtained_states = pattern_mining.merge_two_states(2, 6, pathway_matrix, states)
    obtained_nondeterministic_pairs = pattern_mining.check_is_deterministic(obtained_pathway_matrix, obtained_states, alphabet)
    expected_nondeterministic_pairs = []
    assert obtained_nondeterministic_pairs == expected_nondeterministic_pairs

    obtained_pathway_matrix, obtained_states = pattern_mining.merge_two_states(0, 3, pathway_matrix, states)
    obtained_nondeterministic_pairs = pattern_mining.check_is_deterministic(obtained_pathway_matrix, obtained_states, alphabet)
    expected_nondeterministic_pairs = [(1, 4), (5, 7)]
    assert obtained_nondeterministic_pairs == expected_nondeterministic_pairs