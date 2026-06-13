import numpy as np
import pattern_mining as pm
import pytest


def test_validate_transition_matrix_accepts_valid_matrix(
    simple_pta,
):
    pm._validate_transition_matrix(
        simple_pta.pathway_matrix,
        simple_pta.alphabet,
        simple_pta.states,
    )


def test_validate_transition_matrix_raises_for_non_numpy_array(
    simple_pta,
):
    with pytest.raises(
        TypeError,
        match="transition_matrix must be a NumPy array",
    ):
        pm._validate_transition_matrix(
            simple_pta.pathway_matrix.tolist(),
            simple_pta.alphabet,
            simple_pta.states,
        )


@pytest.mark.parametrize(
    "invalid_matrix",
    [
        np.zeros(3),
        np.zeros((3, 3)),
        np.zeros((3, 3, 3, 3)),
    ],
)
def test_validate_transition_matrix_raises_for_invalid_dimensions(
    simple_pta,
    invalid_matrix,
):
    with pytest.raises(
        ValueError,
        match="transition_matrix must be three-dimensional",
    ):
        pm._validate_transition_matrix(
            invalid_matrix,
            simple_pta.alphabet,
            simple_pta.states,
        )


def test_validate_transition_matrix_raises_for_non_square_state_dimensions():
    transition_matrix = np.zeros((3, 4, 5))

    alphabet = ["A", "B", "C"]
    states = ["*", 0, 1, 2]

    with pytest.raises(
        ValueError,
        match="final two transition_matrix dimensions must be equal",
    ):
        pm._validate_transition_matrix(
            transition_matrix,
            alphabet,
            states,
        )


def test_validate_transition_matrix_raises_for_alphabet_dimension_mismatch():
    transition_matrix = np.zeros((2, 3, 3))

    alphabet = ["A", "B", "C"]
    states = ["*", 0, 1]

    with pytest.raises(
        ValueError,
        match="first transition_matrix dimension must equal len",
    ):
        pm._validate_transition_matrix(
            transition_matrix,
            alphabet,
            states,
        )


def test_validate_transition_matrix_raises_for_state_dimension_mismatch():
    transition_matrix = np.zeros((2, 3, 3))

    alphabet = ["A", "B"]
    states = ["*", 0, 1, 2]

    with pytest.raises(
        ValueError,
        match="transition_matrix state dimensions must equal len",
    ):
        pm._validate_transition_matrix(
            transition_matrix,
            alphabet,
            states,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_validate_transition_matrix_raises_for_non_finite_values(
    simple_pta,
    invalid_value,
):
    transition_matrix = simple_pta.pathway_matrix.astype(float).copy()
    transition_matrix[0, 0, 0] = invalid_value

    with pytest.raises(
        ValueError,
        match="transition_matrix must contain only finite values",
    ):
        pm._validate_transition_matrix(
            transition_matrix,
            simple_pta.alphabet,
            simple_pta.states,
        )


def test_get_n_simple_example(simple_pta):
    assert pm.get_n(0, simple_pta.pathway_matrix, simple_pta.states) == 30
    assert pm.get_n(1, simple_pta.pathway_matrix, simple_pta.states) == 15
    assert pm.get_n(2, simple_pta.pathway_matrix, simple_pta.states) == 15
    assert pm.get_n(3, simple_pta.pathway_matrix, simple_pta.states) == 7
    assert pm.get_n(4, simple_pta.pathway_matrix, simple_pta.states) == 5
    assert pm.get_n(5, simple_pta.pathway_matrix, simple_pta.states) == 5
    assert pm.get_n(6, simple_pta.pathway_matrix, simple_pta.states) == 2

    assert pm.get_n("A", simple_pta.pathway_matrix, simple_pta.alternate_state_names) == 30
    assert pm.get_n("B", simple_pta.pathway_matrix, simple_pta.alternate_state_names) == 15
    assert pm.get_n("C", simple_pta.pathway_matrix, simple_pta.alternate_state_names) == 15
    assert pm.get_n("D", simple_pta.pathway_matrix, simple_pta.alternate_state_names) == 7
    assert pm.get_n("E", simple_pta.pathway_matrix, simple_pta.alternate_state_names) == 5
    assert pm.get_n("F", simple_pta.pathway_matrix, simple_pta.alternate_state_names) == 5
    assert pm.get_n("G", simple_pta.pathway_matrix, simple_pta.alternate_state_names) == 2


def test_get_n_arnolds_example(arnolds_example):
    assert pm.get_n(0, arnolds_example.pathway_matrix, arnolds_example.states) == 10
    assert pm.get_n(1, arnolds_example.pathway_matrix, arnolds_example.states) == 6
    assert pm.get_n(2, arnolds_example.pathway_matrix, arnolds_example.states) == 4
    assert pm.get_n(3, arnolds_example.pathway_matrix, arnolds_example.states) == 4
    assert pm.get_n(4, arnolds_example.pathway_matrix, arnolds_example.states) == 2
    assert pm.get_n(5, arnolds_example.pathway_matrix, arnolds_example.states) == 1
    assert pm.get_n(6, arnolds_example.pathway_matrix, arnolds_example.states) == 1
    assert pm.get_n(7, arnolds_example.pathway_matrix, arnolds_example.states) == 1
    assert pm.get_n(8, arnolds_example.pathway_matrix, arnolds_example.states) == 1
    assert pm.get_n(9, arnolds_example.pathway_matrix, arnolds_example.states) == 1
    assert pm.get_n(10, arnolds_example.pathway_matrix, arnolds_example.states) == 2
    assert pm.get_n(11, arnolds_example.pathway_matrix, arnolds_example.states) == 2
    assert pm.get_n(12, arnolds_example.pathway_matrix, arnolds_example.states) == 1
    assert pm.get_n(13, arnolds_example.pathway_matrix, arnolds_example.states) == 1


def test_get_endpoint_simple_example(simple_pta):
    assert pm.get_endpoint(0, simple_pta.pathway_matrix, simple_pta.states) == 0
    assert pm.get_endpoint(1, simple_pta.pathway_matrix, simple_pta.states) == 8
    assert pm.get_endpoint(2, simple_pta.pathway_matrix, simple_pta.states) == 3
    assert pm.get_endpoint(3, simple_pta.pathway_matrix, simple_pta.states) == 7
    assert pm.get_endpoint(4, simple_pta.pathway_matrix, simple_pta.states) == 5
    assert pm.get_endpoint(5, simple_pta.pathway_matrix, simple_pta.states) == 5
    assert pm.get_endpoint(6, simple_pta.pathway_matrix, simple_pta.states) == 2

    assert pm.get_endpoint("A", simple_pta.pathway_matrix, simple_pta.alternate_state_names) == 0
    assert pm.get_endpoint("B", simple_pta.pathway_matrix, simple_pta.alternate_state_names) == 8
    assert pm.get_endpoint("C", simple_pta.pathway_matrix, simple_pta.alternate_state_names) == 3
    assert pm.get_endpoint("D", simple_pta.pathway_matrix, simple_pta.alternate_state_names) == 7
    assert pm.get_endpoint("E", simple_pta.pathway_matrix, simple_pta.alternate_state_names) == 5
    assert pm.get_endpoint("F", simple_pta.pathway_matrix, simple_pta.alternate_state_names) == 5
    assert pm.get_endpoint("G", simple_pta.pathway_matrix, simple_pta.alternate_state_names) == 2


def test_get_endpoint_arnolds_example(arnolds_example):
    assert pm.get_endpoint(0, arnolds_example.pathway_matrix, arnolds_example.states) == 0
    assert pm.get_endpoint(1, arnolds_example.pathway_matrix, arnolds_example.states) == 0
    assert pm.get_endpoint(2, arnolds_example.pathway_matrix, arnolds_example.states) == 0
    assert pm.get_endpoint(3, arnolds_example.pathway_matrix, arnolds_example.states) == 1
    assert pm.get_endpoint(4, arnolds_example.pathway_matrix, arnolds_example.states) == 1
    assert pm.get_endpoint(5, arnolds_example.pathway_matrix, arnolds_example.states) == 1
    assert pm.get_endpoint(6, arnolds_example.pathway_matrix, arnolds_example.states) == 1
    assert pm.get_endpoint(7, arnolds_example.pathway_matrix, arnolds_example.states) == 0
    assert pm.get_endpoint(8, arnolds_example.pathway_matrix, arnolds_example.states) == 1
    assert pm.get_endpoint(9, arnolds_example.pathway_matrix, arnolds_example.states) == 1
    assert pm.get_endpoint(10, arnolds_example.pathway_matrix, arnolds_example.states) == 1
    assert pm.get_endpoint(11, arnolds_example.pathway_matrix, arnolds_example.states) == 1
    assert pm.get_endpoint(12, arnolds_example.pathway_matrix, arnolds_example.states) == 1
    assert pm.get_endpoint(13, arnolds_example.pathway_matrix, arnolds_example.states) == 1


def test_get_pi_simple_example(simple_pta):
    obtained_pi = [
        pm.get_pi(i, 0, simple_pta.pathway_matrix, simple_pta.states) for i in range(7)
    ]
    expected_pi = [1 / 2, 0, 1 / 3, 0, 0, 0, 0]
    assert np.allclose(obtained_pi, expected_pi)
    obtained_pi = [
        pm.get_pi(i, 1, simple_pta.pathway_matrix, simple_pta.states) for i in range(7)
    ]
    expected_pi = [1 / 2, 7 / 15, 1 / 3, 0, 0, 0, 0]
    assert np.allclose(obtained_pi, expected_pi)
    obtained_pi = [
        pm.get_pi(i, 2, simple_pta.pathway_matrix, simple_pta.states) for i in range(7)
    ]
    expected_pi = [0, 0, 2 / 15, 0, 0, 0, 0]
    assert np.allclose(obtained_pi, expected_pi)

    assert (
        pm.get_pi("C", 2, simple_pta.pathway_matrix, simple_pta.alternate_state_names) == 2 / 15
    )


def test_get_pi_arnolds_example(arnolds_example):
    obtained_pi = [
        pm.get_pi(i, 0, arnolds_example.pathway_matrix, arnolds_example.states)
        for i in range(14)
    ]
    expected_pi = [3 / 5, 0, 1 / 2, 1 / 4, 0, 0, 0, 1, 0, 0, 1 / 2, 1 / 2, 0, 0]
    assert np.allclose(obtained_pi, expected_pi)
    obtained_pi = [
        pm.get_pi(i, 1, arnolds_example.pathway_matrix, arnolds_example.states)
        for i in range(14)
    ]
    expected_pi = [2 / 5, 2 / 3, 0, 1 / 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert np.allclose(obtained_pi, expected_pi)
    obtained_pi = [
        pm.get_pi(i, 2, arnolds_example.pathway_matrix, arnolds_example.states)
        for i in range(14)
    ]
    expected_pi = [0, 1 / 3, 1 / 2, 1 / 4, 1 / 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert np.allclose(obtained_pi, expected_pi)


def test_get_pi_endpoint_simple_example(simple_pta):
    obtained_endpoints = [
        pm.get_pi_endpoint(i, simple_pta.pathway_matrix, simple_pta.alphabet, simple_pta.states)
        for i in range(7)
    ]
    expected_endpoints = [0, 8 / 15, 1 / 5, 1, 1, 1, 1]
    assert np.allclose(obtained_endpoints, expected_endpoints)


def test_get_pi_endpoint_arnolds_example(arnolds_example):
    obtained_endpoints = [
        pm.get_pi_endpoint(
            i, arnolds_example.pathway_matrix, arnolds_example.alphabet, arnolds_example.states
        )
        for i in range(14)
    ]
    expected_endpoints = [0, 0, 0, 1 / 4, 1 / 2, 1, 1, 0, 1, 1, 1 / 2, 1 / 2, 1, 1]
    assert np.allclose(obtained_endpoints, expected_endpoints)