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


def test_get_n():
    assert pm.get_n(0, pathway_matrix, states) == 30
    assert pm.get_n(1, pathway_matrix, states) == 15
    assert pm.get_n(2, pathway_matrix, states) == 15
    assert pm.get_n(3, pathway_matrix, states) == 7
    assert pm.get_n(4, pathway_matrix, states) == 5
    assert pm.get_n(5, pathway_matrix, states) == 5
    assert pm.get_n(6, pathway_matrix, states) == 2

    assert pm.get_n("A", pathway_matrix, states_alternate_names) == 30
    assert pm.get_n("B", pathway_matrix, states_alternate_names) == 15
    assert pm.get_n("C", pathway_matrix, states_alternate_names) == 15
    assert pm.get_n("D", pathway_matrix, states_alternate_names) == 7
    assert pm.get_n("E", pathway_matrix, states_alternate_names) == 5
    assert pm.get_n("F", pathway_matrix, states_alternate_names) == 5
    assert pm.get_n("G", pathway_matrix, states_alternate_names) == 2

    assert pm.get_n(0, arnolds_pathway_matrix, arnolds_states) == 10
    assert pm.get_n(1, arnolds_pathway_matrix, arnolds_states) == 6
    assert pm.get_n(2, arnolds_pathway_matrix, arnolds_states) == 4
    assert pm.get_n(3, arnolds_pathway_matrix, arnolds_states) == 4
    assert pm.get_n(4, arnolds_pathway_matrix, arnolds_states) == 2
    assert pm.get_n(5, arnolds_pathway_matrix, arnolds_states) == 1
    assert pm.get_n(6, arnolds_pathway_matrix, arnolds_states) == 1
    assert pm.get_n(7, arnolds_pathway_matrix, arnolds_states) == 1
    assert pm.get_n(8, arnolds_pathway_matrix, arnolds_states) == 1
    assert pm.get_n(9, arnolds_pathway_matrix, arnolds_states) == 1
    assert pm.get_n(10, arnolds_pathway_matrix, arnolds_states) == 2
    assert pm.get_n(11, arnolds_pathway_matrix, arnolds_states) == 2
    assert pm.get_n(12, arnolds_pathway_matrix, arnolds_states) == 1
    assert pm.get_n(13, arnolds_pathway_matrix, arnolds_states) == 1


def test_get_endpoint():
    assert pm.get_endpoint(0, pathway_matrix, states) == 0
    assert pm.get_endpoint(1, pathway_matrix, states) == 8
    assert pm.get_endpoint(2, pathway_matrix, states) == 3
    assert pm.get_endpoint(3, pathway_matrix, states) == 7
    assert pm.get_endpoint(4, pathway_matrix, states) == 5
    assert pm.get_endpoint(5, pathway_matrix, states) == 5
    assert pm.get_endpoint(6, pathway_matrix, states) == 2

    assert pm.get_endpoint("A", pathway_matrix, states_alternate_names) == 0
    assert pm.get_endpoint("B", pathway_matrix, states_alternate_names) == 8
    assert pm.get_endpoint("C", pathway_matrix, states_alternate_names) == 3
    assert pm.get_endpoint("D", pathway_matrix, states_alternate_names) == 7
    assert pm.get_endpoint("E", pathway_matrix, states_alternate_names) == 5
    assert pm.get_endpoint("F", pathway_matrix, states_alternate_names) == 5
    assert pm.get_endpoint("G", pathway_matrix, states_alternate_names) == 2

    assert pm.get_endpoint(0, arnolds_pathway_matrix, arnolds_states) == 0
    assert pm.get_endpoint(1, arnolds_pathway_matrix, arnolds_states) == 0
    assert pm.get_endpoint(2, arnolds_pathway_matrix, arnolds_states) == 0
    assert pm.get_endpoint(3, arnolds_pathway_matrix, arnolds_states) == 1
    assert pm.get_endpoint(4, arnolds_pathway_matrix, arnolds_states) == 1
    assert pm.get_endpoint(5, arnolds_pathway_matrix, arnolds_states) == 1
    assert pm.get_endpoint(6, arnolds_pathway_matrix, arnolds_states) == 1
    assert pm.get_endpoint(7, arnolds_pathway_matrix, arnolds_states) == 0
    assert pm.get_endpoint(8, arnolds_pathway_matrix, arnolds_states) == 1
    assert pm.get_endpoint(9, arnolds_pathway_matrix, arnolds_states) == 1
    assert pm.get_endpoint(10, arnolds_pathway_matrix, arnolds_states) == 1
    assert pm.get_endpoint(11, arnolds_pathway_matrix, arnolds_states) == 1
    assert pm.get_endpoint(12, arnolds_pathway_matrix, arnolds_states) == 1
    assert pm.get_endpoint(13, arnolds_pathway_matrix, arnolds_states) == 1


def test_get_pi():
    obtained_pi = [
        pm.get_pi(i, 0, pathway_matrix, states) for i in range(7)
    ]
    expected_pi = [1 / 2, 0, 1 / 3, 0, 0, 0, 0]
    assert np.allclose(obtained_pi, expected_pi)
    obtained_pi = [
        pm.get_pi(i, 1, pathway_matrix, states) for i in range(7)
    ]
    expected_pi = [1 / 2, 7 / 15, 1 / 3, 0, 0, 0, 0]
    assert np.allclose(obtained_pi, expected_pi)
    obtained_pi = [
        pm.get_pi(i, 2, pathway_matrix, states) for i in range(7)
    ]
    expected_pi = [0, 0, 2 / 15, 0, 0, 0, 0]
    assert np.allclose(obtained_pi, expected_pi)

    assert (
        pm.get_pi("C", 2, pathway_matrix, states_alternate_names) == 2 / 15
    )

    obtained_pi = [
        pm.get_pi(i, 0, arnolds_pathway_matrix, arnolds_states)
        for i in range(14)
    ]
    expected_pi = [3 / 5, 0, 1 / 2, 1 / 4, 0, 0, 0, 1, 0, 0, 1 / 2, 1 / 2, 0, 0]
    assert np.allclose(obtained_pi, expected_pi)
    obtained_pi = [
        pm.get_pi(i, 1, arnolds_pathway_matrix, arnolds_states)
        for i in range(14)
    ]
    expected_pi = [2 / 5, 2 / 3, 0, 1 / 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert np.allclose(obtained_pi, expected_pi)
    obtained_pi = [
        pm.get_pi(i, 2, arnolds_pathway_matrix, arnolds_states)
        for i in range(14)
    ]
    expected_pi = [0, 1 / 3, 1 / 2, 1 / 4, 1 / 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert np.allclose(obtained_pi, expected_pi)


def test_get_pi_endpoint():
    obtained_endpoints = [
        pm.get_pi_endpoint(i, pathway_matrix, alphabet, states)
        for i in range(7)
    ]
    expected_endpoints = [0, 8 / 15, 1 / 5, 1, 1, 1, 1]
    assert np.allclose(obtained_endpoints, expected_endpoints)

    obtained_endpoints = [
        pm.get_pi_endpoint(
            i, arnolds_pathway_matrix, arnolds_alphabet, arnolds_states
        )
        for i in range(14)
    ]
    expected_endpoints = [0, 0, 0, 1 / 4, 1 / 2, 1, 1, 0, 1, 1, 1 / 2, 1 / 2, 1, 1]
    assert np.allclose(obtained_endpoints, expected_endpoints)