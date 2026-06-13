import numpy as np
import pattern_mining as pm
import pytest


def test_probability_transition_matrix_simple_example(simple_pta):
    obtained_matrix = pm.probability_transition_matrix(
        simple_pta.final_pathway_matrix, simple_pta.final_states, simple_pta.alphabet
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


def test_probability_transition_matrix_arnolds_example(arnolds_example):
    obtained_matrix = pm.probability_transition_matrix(
        arnolds_example.final_pathway_matrix, arnolds_example.final_states, arnolds_example.alphabet
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


def test_probability_estimate_of_symbol(jacquemont_example):
    obtained_vector = pm.probability_estimate_of_symbol(
        jacquemont_example.probability_matrix, "c", jacquemont_example.alphabet
    )
    expected_vector = np.array([0.47068936, 0.47068936, 0.42719615, 0.42719615])
    assert np.allclose(obtained_vector, expected_vector)


def test_probability_estimate_of_pattern(jacquemont_example):
    obtained_vector = pm.probability_estimate_of_pattern(
        jacquemont_example.probability_matrix, "cc", jacquemont_example.alphabet
    )
    expected_vector = np.array([0.21552208, 0.21552208, 0.1861079, 0.1861079])
    assert np.allclose(obtained_vector, expected_vector)


def test_probability_estimate_of_exact_sequence(arnolds_example):
    obtained_probability = pm.probability_estimate_of_exact_sequence(
        arnolds_example.final_p_matrix_point9, "ABC", arnolds_example.alphabet
    )
    expected_probability = 0.0016163599573759896
    assert np.allclose(obtained_probability, expected_probability)


def test_probability_sequence_contains_letter_at_distance_theta(jacquemont_example):
    obtained_vector = (
        pm.probability_sequence_contains_letter_at_distance_theta(
            jacquemont_example.probability_matrix, "a", 2, jacquemont_example.alphabet
        )
    )
    expected_vector = np.array([0.201467, 0.3629, 0.2146, 0.137696])
    assert np.allclose(obtained_vector, expected_vector)


def test_pattern_at_distance_raises_for_single_symbol(jacquemont_example):
    with pytest.raises(
        ValueError,
        match="pattern must contain at least two symbols",
    ):
        pm.probability_to_encounter_a_pattern_at_a_distance_theta(
            jacquemont_example.probability_matrix,
            "A",
            1,
            jacquemont_example.alphabet,
        )


def test_pattern_at_distance_raises_for_non_integer_theta(jacquemont_example):
    with pytest.raises(TypeError, match="theta must be an integer"):
        pm.probability_to_encounter_a_pattern_at_a_distance_theta(
            jacquemont_example.probability_matrix,
            "AB",
            1.5,
            jacquemont_example.alphabet,
        )


def test_pattern_at_distance_raises_for_negative_theta(jacquemont_example):
    with pytest.raises(ValueError, match="theta must be non-negative"):
        pm.probability_to_encounter_a_pattern_at_a_distance_theta(
            jacquemont_example.probability_matrix,
            "AB",
            -1,
            jacquemont_example.alphabet,
        )


def test_probability_to_encounter_a_pattern_at_a_distance_theta(jacquemont_example):
    obtained_vector = (
        pm.probability_to_encounter_a_pattern_at_a_distance_theta(
            jacquemont_example.probability_matrix, "ab", 2, jacquemont_example.alphabet
        )
    )
    expected_vector = np.array([0.165817, 0.2079, 0.1346, 0.116896])
    assert np.allclose(obtained_vector, expected_vector)

    obtained_vector = (
        pm.probability_to_encounter_a_pattern_at_a_distance_theta(
            jacquemont_example.probability_matrix, "abc", 2, jacquemont_example.alphabet
        )
    )
    expected_vector = np.array([0.0773778, 0.0949411, 0.06185016, 0.0546305])
    assert np.allclose(obtained_vector, expected_vector)


def test_proportion_constraint_jacquemont_example(jacquemont_example):
    assert (
        pm.proportion_constraint(
            jacquemont_example.probability_matrix, "cc", jacquemont_example.alphabet, jacquemont_example.sequences, 0.05
        )
        == True
    )

    assert (
        pm.proportion_constraint(
            jacquemont_example.probability_matrix, "bcc", jacquemont_example.alphabet, jacquemont_example.sequences, 0.05
        )
        == False
    )


def test_proportion_constraint_arnolds_example(arnolds_example):
    assert (
        pm.proportion_constraint(
            arnolds_example.final_p_matrix_point9,
            "AAC",
            arnolds_example.alphabet,
            arnolds_example.sequences,
            0.33,
            p_value="sequence",
        )
        == True
    )

    assert (
        pm.proportion_constraint(
            arnolds_example.final_p_matrix_point9,
            "ABC",
            arnolds_example.alphabet,
            arnolds_example.sequences,
            0.33,
            p_value="sequence",
        )
        == False
    )


def test_probability_sequence_contains_digram(jacquemont_example):
    obtained_vector = pm.probability_sequence_contains_digram(
        jacquemont_example.probability_matrix, "ab", jacquemont_example.alphabet
    )
    expected_vector = np.array([0.2987013, 0, 0.29800281, 0.28378378])
    assert np.allclose(obtained_vector, expected_vector)


def test_string_enumerator(jacquemont_example):
    obtained_strings = pm.string_enumerator(jacquemont_example.alphabet, 2)
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


def test_string_probabilities(arnolds_example):
    obtained_probabilities = pm.string_probabilities(
        arnolds_example.final_p_matrix_point9, arnolds_example.alphabet, ["A", "B", "C"]
    )
    expected_probabilities = [
        ("A", 0.18685121107266434),
        ("B", 0.013840830449826992),
        ("C", 0.06228373702422145),
    ]
    assert obtained_probabilities[0][1] == expected_probabilities[0][1]
