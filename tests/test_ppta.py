import numpy as np
import pattern_mining as pm
import pytest


def test_validate_sequences_raises_for_none():
    with pytest.raises(
        TypeError,
        match="sequences must be an iterable of strings",
    ):
        pm._validate_sequences(None)


def test_validate_sequences_raises_for_single_string():
    with pytest.raises(
        TypeError,
        match="sequences must be an iterable of strings",
    ):
        pm._validate_sequences("ABC")


def test_validate_sequences_raises_for_empty_sequences():
    with pytest.raises(
        ValueError,
        match="sequences must contain at least one sequence",
    ):
        pm._validate_sequences([])


@pytest.mark.parametrize(
    "invalid_sequences",
    [
        ["AB", 1],
        ["AB", None],
        [1, 2, 3],
    ],
)
def test_validate_sequences_raises_for_non_string_sequences(
    invalid_sequences,
):
    with pytest.raises(
        TypeError,
        match="every sequence must be a string",
    ):
        pm._validate_sequences(invalid_sequences)


def test_validate_alphabet_raises_for_none():
    with pytest.raises(
        TypeError,
        match="alphabet must be an iterable of strings",
    ):
        pm._validate_alphabet(None)


def test_validate_alphabet_raises_for_single_string():
    with pytest.raises(
        TypeError,
        match="alphabet must be an iterable of strings",
    ):
        pm._validate_alphabet("ABC")


def test_validate_alphabet_raises_for_empty_alphabet():
    with pytest.raises(
        ValueError,
        match="alphabet must contain at least one symbol",
    ):
        pm._validate_alphabet([])


@pytest.mark.parametrize(
    "invalid_alphabet",
    [
        ["A", 1],
        ["A", None],
        [1, 2, 3],
    ],
)
def test_validate_alphabet_raises_for_non_string_symbols(
    invalid_alphabet,
):
    with pytest.raises(
        TypeError,
        match="every alphabet symbol must be a string",
    ):
        pm._validate_alphabet(invalid_alphabet)


@pytest.mark.parametrize(
    "invalid_alphabet_sizes",
    [
        ["A", "AB"],
        ["", "BC"],
    ],
)
def test_validate_alphabet_raises_for_non_single_character_symbols(
    invalid_alphabet_sizes,
):
    with pytest.raises(
        ValueError,
        match="every alphabet symbol must contain exactly one character.",
    ):
        pm._validate_alphabet(invalid_alphabet_sizes)


def test_validate_alphabet_raises_for_duplicate_symbols():
    with pytest.raises(
        ValueError,
        match="alphabet must not contain duplicate symbols",
    ):
        pm._validate_alphabet(["A", "B", "A"])


def test_validate_alphabet_returns_validated_alphabet():
    obtained = pm._validate_alphabet(("A", "B", "C"))

    expected = ["A", "B", "C"]

    assert obtained == expected


def test_get_alphabet_simple_example(simple_pta):
    obtained_alphabet = pm.get_alphabet(simple_pta.sequences)
    expected_alphabet = simple_pta.alphabet
    assert obtained_alphabet == expected_alphabet


def test_get_alphabet_arnolds_example(arnolds_example):
    obtained_alphabet = pm.get_alphabet(arnolds_example.sequences)
    expected_alphabet = arnolds_example.alphabet
    assert obtained_alphabet == expected_alphabet


def test_get_state_paths_raises_for_invalid_build(simple_pta):
    with pytest.raises(
        ValueError,
        match="build must be either 'breadth' or 'depth'.",
    ):
        pm.get_state_paths(
            simple_pta.sequences,
            build="invalid",
        )


def test_get_state_paths_simple_example(simple_pta):
    obtained_state_paths_breadth = pm.get_state_paths(simple_pta.sequences, "breadth")
    expected_state_paths_breadth = ["", "0", "1", "01", "10", "11", "12"]
    obtained_state_paths_depth = pm.get_state_paths(simple_pta.sequences, "depth")
    expected_state_paths_depth = ["", "0", "1", "01", "10", "11", "12"]
    assert obtained_state_paths_breadth == expected_state_paths_breadth
    assert obtained_state_paths_depth == expected_state_paths_depth


def test_get_state_paths_arnolds_example(arnolds_example):
    obtained_state_paths_breadth = pm.get_state_paths(
        arnolds_example.sequences, "breadth"
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
    obtained_state_paths_depth = pm.get_state_paths(arnolds_example.sequences, "depth")
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


def test_get_transition_matrix_raises_when_alphabet_is_missing_symbols():
    sequences = ["AB", "AC"]
    alphabet = ["A", "B"]

    with pytest.raises(
        ValueError,
        match="alphabet is missing symbols found in sequences",
    ):
        pm.get_transition_matrix(
            sequences,
            alphabet,
        )


def test_get_transition_matrix_simple_example(simple_pta):
    obtained_transition_matrix = pm.get_transition_matrix(
        simple_pta.sequences, simple_pta.alphabet
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


def test_get_transition_matrix_arnolds_example(arnolds_example):
    obtained_transition_matrix = pm.get_transition_matrix(
        arnolds_example.sequences, arnolds_example.alphabet
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


def test_get_initial_states_simple_example(simple_pta):
    obtained_states = pm.get_initial_states(simple_pta.sequences)
    expected_states = ["*", 0, 1, 2, 3, 4, 5, 6]
    assert obtained_states == expected_states


def test_get_initial_states_arnolds_example(arnolds_example):
    obtained_states = pm.get_initial_states(arnolds_example.sequences)
    expected_states = ["*", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    assert obtained_states == expected_states
