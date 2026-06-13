import numpy as np
import pattern_mining as pm

alphabet = ["0", "1", "2"]

arnolds_sequences = ["AB", "ABA", "ABB", "ABCA", "AC", "ACC", "BA", "BAA", "BC", "BCA"]

arnolds_alphabet = ["A", "B", "C"]

states = ["*", 0, 1, 2, 3, 4, 5, 6]

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

final_pathway_matrix = np.array(
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
final_states = ["*", 0, 1, 3]

final_arnolds_pathway_matrix = np.array(
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
final_arnolds_states = ["*", 0, 1]

final_arnolds_p_matrix_point9 = np.array(
    [
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 9 / 17, 0.0],
            [0.0, 0.0, 0.0, 3 / 17],
            [0.0, 0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 4 / 17, 0.0, 0.0],
            [0.0, 0.0, 5 / 17, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3 / 17, 0.0],
            [0.0, 3 / 17, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
    ]
)

jacquemont_p_matrix = np.array(
    [
        [
            [0, 0.23, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1.0],
            [0, 0.21, 0, 0],
        ],
        [
            [0, 0, 0.31, 0],
            [1.0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0.16, 0],
        ],
        [
            [0.23, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0.26],
        ],
    ]
)
jacquemont_sequences = [
    "ab",
    "bac",
    "baba",
    "abbac",
    "abbaab",
    "ba",
    "abcc",
    "bacc",
    "abccc",
    "baabba",
    "abc",
    "baab",
    "ababc",
    "babac",
    "babaabc",
]
jacquemont_alphabet = ["a", "b", "c"]
jacquemont_states = [
    "*",
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
]


def test_probability_transition_matrix():
    obtained_matrix = pm.probability_transition_matrix(
        final_pathway_matrix, final_states, alphabet
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

    obtained_matrix = pm.probability_transition_matrix(
        final_arnolds_pathway_matrix, final_arnolds_states, arnolds_alphabet
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


def test_probability_estimate_of_symbol():
    obtained_vector = pm.probability_estimate_of_symbol(
        jacquemont_p_matrix, "c", jacquemont_alphabet
    )
    expected_vector = np.array([0.47068936, 0.47068936, 0.42719615, 0.42719615])
    assert np.allclose(obtained_vector, expected_vector)


def test_probability_estimate_of_pattern():
    obtained_vector = pm.probability_estimate_of_pattern(
        jacquemont_p_matrix, "cc", jacquemont_alphabet
    )
    expected_vector = np.array([0.21552208, 0.21552208, 0.1861079, 0.1861079])
    assert np.allclose(obtained_vector, expected_vector)


def test_probability_estimate_of_exact_sequence():
    obtained_probability = pm.probability_estimate_of_exact_sequence(
        final_arnolds_p_matrix_point9, "ABC", arnolds_alphabet
    )
    expected_probability = 0.0016163599573759896
    assert np.allclose(obtained_probability, expected_probability)


def test_probability_sequence_contains_letter_at_distance_theta():
    obtained_vector = (
        pm.probability_sequence_contains_letter_at_distance_theta(
            jacquemont_p_matrix, "a", 2, jacquemont_alphabet
        )
    )
    expected_vector = np.array([0.201467, 0.3629, 0.2146, 0.137696])
    assert np.allclose(obtained_vector, expected_vector)


def test_probability_to_encounter_a_pattern_at_a_distance_theta():
    obtained_vector = (
        pm.probability_to_encounter_a_pattern_at_a_distance_theta(
            jacquemont_p_matrix, "ab", 2, jacquemont_alphabet
        )
    )
    expected_vector = np.array([0.165817, 0.2079, 0.1346, 0.116896])
    assert np.allclose(obtained_vector, expected_vector)

    obtained_vector = (
        pm.probability_to_encounter_a_pattern_at_a_distance_theta(
            jacquemont_p_matrix, "abc", 2, jacquemont_alphabet
        )
    )
    expected_vector = np.array([0.0773778, 0.0949411, 0.06185016, 0.0546305])
    assert np.allclose(obtained_vector, expected_vector)


def test_proportion_constraint():
    assert (
        pm.proportion_constraint(
            jacquemont_p_matrix, "cc", jacquemont_alphabet, jacquemont_sequences, 0.05
        )
        == True
    )

    assert (
        pm.proportion_constraint(
            jacquemont_p_matrix, "bcc", jacquemont_alphabet, jacquemont_sequences, 0.05
        )
        == False
    )

    assert (
        pm.proportion_constraint(
            final_arnolds_p_matrix_point9,
            "AAC",
            arnolds_alphabet,
            arnolds_sequences,
            0.33,
            p_value="sequence",
        )
        == True
    )

    assert (
        pm.proportion_constraint(
            final_arnolds_p_matrix_point9,
            "ABC",
            arnolds_alphabet,
            arnolds_sequences,
            0.33,
            p_value="sequence",
        )
        == False
    )


def test_probability_sequence_contains_digram():
    obtained_vector = pm.probability_sequence_contains_digram(
        jacquemont_p_matrix, "ab", jacquemont_alphabet
    )
    expected_vector = np.array([0.2987013, 0, 0.29800281, 0.28378378])
    assert np.allclose(obtained_vector, expected_vector)


def test_string_enumerator():
    obtained_strings = pm.string_enumerator(jacquemont_alphabet, 2)
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


def test_string_probabilities():
    obtained_probabilities = pm.string_probabilities(
        final_arnolds_p_matrix_point9, arnolds_alphabet, ["A", "B", "C"]
    )
    expected_probabilities = [
        ("A", 0.18685121107266434),
        ("B", 0.013840830449826992),
        ("C", 0.06228373702422145),
    ]
    assert obtained_probabilities[0][1] == expected_probabilities[0][1]
