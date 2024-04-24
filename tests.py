import numpy as np
import pattern_mining

list_of_sequences = ["0", "0", "0", "0", "0", "0", "0", "0", "01", "01", 
                     "01", "01", "01", "01", "01", "10", "10", "10", "10", "10",
                     "11", "11", "11", "11", "11", "12", "12", "1", "1", "1"]

alphabet = ['0', '1', '2']

def test_get_alphabet():
    obtained_alphabet = pattern_mining.get_alphabet(list_of_sequences)
    expected_alphabet = ['0', '1', '2']
    assert obtained_alphabet == expected_alphabet

def test_get_state_paths():
    obtained_state_paths_breadth = pattern_mining.get_state_paths(list_of_sequences, "breadth")
    expected_state_paths_breadth = ['', '0', '1', '01', '10', '11', '12']
    obtained_state_paths_depth = pattern_mining.get_state_paths(list_of_sequences, "depth")
    expected_state_paths_depth = ['', '0', '1', '01', '10', '11', '12']
    assert obtained_state_paths_breadth == expected_state_paths_breadth
    assert obtained_state_paths_depth == expected_state_paths_depth

def test_transition_matrix():
    obtained_transition_matrix = pattern_mining.transition_matrix(list_of_sequences, alphabet)
    expected_pathway_matrix = np.array([
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
    ]
])
    assert np.allclose(obtained_transition_matrix, expected_pathway_matrix)

states = ["S", 0, 1, 2, 3, 4, 5, 6]
states_alternate_names = ["S", 'A', 'B', 'C', 'D', 'E', 'F', 'G']
pathway_matrix = np.array([
    [
        [0, 30, 0, 0, 0, 0, 0, 0],
        [0, 0, 15, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 5, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 15, 0, 0, 0],
        [0, 0, 0, 7, 0, 0, 0, 0],
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
    ]
])

def test_get_n():
    assert pattern_mining.get_n(0, pathway_matrix, states) == 30
    assert pattern_mining.get_n(1, pathway_matrix, states) == 15
    assert pattern_mining.get_n(2, pathway_matrix, states) == 7
    assert pattern_mining.get_n(3, pathway_matrix, states) == 15
    assert pattern_mining.get_n(4, pathway_matrix, states) == 5
    assert pattern_mining.get_n(5, pathway_matrix, states) == 5
    assert pattern_mining.get_n(6, pathway_matrix, states) == 2

    assert pattern_mining.get_n("A", pathway_matrix, states_alternate_names) == 30
    assert pattern_mining.get_n("B", pathway_matrix, states_alternate_names) == 15
    assert pattern_mining.get_n("C", pathway_matrix, states_alternate_names) == 7
    assert pattern_mining.get_n("D", pathway_matrix, states_alternate_names) == 15
    assert pattern_mining.get_n("E", pathway_matrix, states_alternate_names) == 5
    assert pattern_mining.get_n("F", pathway_matrix, states_alternate_names) == 5
    assert pattern_mining.get_n("G", pathway_matrix, states_alternate_names) == 2


def test_get_pi():
    assert pattern_mining.get_pi(0, 0, pathway_matrix, states) == 1/2
    assert pattern_mining.get_pi(0, 1, pathway_matrix, states) == 1/2
    assert pattern_mining.get_pi(0, 2, pathway_matrix, states) == 0
    assert pattern_mining.get_pi(1, 0, pathway_matrix, states) == 0
    assert pattern_mining.get_pi(1, 1, pathway_matrix, states) == 7/15
    assert pattern_mining.get_pi(1, 2, pathway_matrix, states) == 0
    assert pattern_mining.get_pi(2, 0, pathway_matrix, states) == 0
    assert pattern_mining.get_pi(2, 1, pathway_matrix, states) == 0
    assert pattern_mining.get_pi(2, 2, pathway_matrix, states) == 0
    assert pattern_mining.get_pi(3, 0, pathway_matrix, states) == 1/3
    assert pattern_mining.get_pi(3, 1, pathway_matrix, states) == 1/3
    assert pattern_mining.get_pi(3, 2, pathway_matrix, states) == 2/15
    assert pattern_mining.get_pi(4, 0, pathway_matrix, states) == 0
    assert pattern_mining.get_pi(4, 1, pathway_matrix, states) == 0
    assert pattern_mining.get_pi(4, 2, pathway_matrix, states) == 0
    assert pattern_mining.get_pi(5, 0, pathway_matrix, states) == 0
    assert pattern_mining.get_pi(5, 1, pathway_matrix, states) == 0
    assert pattern_mining.get_pi(5, 2, pathway_matrix, states) == 0
    assert pattern_mining.get_pi(6, 0, pathway_matrix, states) == 0
    assert pattern_mining.get_pi(6, 1, pathway_matrix, states) == 0
    assert pattern_mining.get_pi(6, 2, pathway_matrix, states) == 0

    assert pattern_mining.get_pi("D", 2, pathway_matrix, states_alternate_names) == 2/15


def test_get_pi_endpoint():
    obtained_endpoints = [pattern_mining.get_pi_endpoint(i, pathway_matrix, alphabet, states) for i in range(7)]
    expected_endpoints = [0, 8/15, 1, 1/5, 1, 1, 1]
    assert np.allclose(obtained_endpoints, expected_endpoints)


def test_hoeffding_bound():
    assert pattern_mining.hoeffding_bound(0, 3, 0.2, pathway_matrix, alphabet, states) == True
    assert pattern_mining.hoeffding_bound(1, 5, 0.2, pathway_matrix, alphabet, states) == True
    assert pattern_mining.hoeffding_bound(2, 0, 0.2, pathway_matrix, alphabet, states) == False
    assert pattern_mining.hoeffding_bound(5, 0, 0.2, pathway_matrix, alphabet, states) == False

    assert pattern_mining.hoeffding_bound("B", "F", 0.2, pathway_matrix, alphabet, states_alternate_names) == True
    assert pattern_mining.hoeffding_bound("C", "A", 0.2, pathway_matrix, alphabet, states_alternate_names) == False


def test_merge_two_states():
    obtained_pathway_matrix, obtained_states = pattern_mining.merge_two_states(0, 1, pathway_matrix, states)
    expected_pathway_matrix = np.array([
        [
            [0, 0, 0, 0, 0, 0, 30],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 5, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 15],
        ],
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 7, 15, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    ])
    expected_states = ["S", 2, 3, 4, 5, 6, 7]
    assert np.allclose(obtained_pathway_matrix, expected_pathway_matrix)
    assert obtained_states == expected_states

    obtained_pathway_matrix, obtained_states = pattern_mining.merge_two_states(2, 6, pathway_matrix, states)
    expected_pathway_matrix = np.array([
        [
            [0, 30, 0, 0, 0, 0, 0],
            [0, 0, 15, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 15, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 7],
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
    expected_states = ["S", 0, 1, 3, 4, 5, 7]
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


def test_check_is_deterministic_with_multiple_pairs():
    wrong_pathway_matrix = np.array([
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
        ]
    ])
    obtained_nondeterministic_pairs = pattern_mining.check_is_deterministic(wrong_pathway_matrix, states, alphabet)
    expected_nondeterministic_pairs = [(1, 6), (1, 4), (2, 4)]
    assert obtained_nondeterministic_pairs == expected_nondeterministic_pairs


def test_recursive_merge_two_states():
    # Test where states are merged recursively
    obtained_matrix, obtained_states = pattern_mining.recursive_merge_two_states(1, 3, pathway_matrix, states, 0.2, alphabet)
    expected_matrix = np.array([
        [
            [ 0, 30,  0,  0,  0,  0],
            [ 0,  0,  0,  0, 15,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  5,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
        ],
        [
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0, 15,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0, 12],
            [ 0,  0,  0,  0,  0,  0],
        ],
        [
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  2,  0,  0],
            [ 0,  0,  0,  0,  0,  0],
        ]
    ])
    expected_states = ["S", 0, 4, 6, 7, 8]
    assert np.allclose(obtained_matrix, expected_matrix)
    assert obtained_states == expected_states

    # Test where Hoeffding's Bound fails during recursive merge
    obtained_matrix, obtained_states = pattern_mining.recursive_merge_two_states(0, 3, pathway_matrix, states, 0.2, alphabet)
    assert np.allclose(obtained_matrix, pathway_matrix)
    assert obtained_states == states


def test_get_pairs_to_check():
    obtained_pairs = pattern_mining.get_pairs_to_check(states)
    expected_pairs = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (4, 3), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5)]
    assert obtained_pairs == expected_pairs


def test_alergia():
    obtained_matrix, obtained_states = pattern_mining.alergia(pathway_matrix, states, alphabet, 0.2)
    expected_matrix = np.array([
        [
            [ 0, 30,  0,  0],
            [ 0,  0,  0, 15],
            [ 0,  0,  0,  5],
            [ 0,  0,  0,  0],
        ],
        [
            [ 0,  0,  0,  0],
            [ 0,  0, 15,  0],
            [ 0,  0,  0,  5],
            [ 0,  0,  0,  7],
        ],
        [
            [ 0,  0,  0,  0],
            [ 0,  0,  0,  0],
            [ 0,  0,  2,  0],
            [ 0,  0,  0,  0],
        ]
    ])
    expected_states = ["S", 0, 9, 10]