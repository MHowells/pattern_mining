import numpy as np
import pattern_mining as pm
import pytest


def test_validate_states_for_merging_accepts_valid_states():
    states = ["*", 0, 1, 2]

    pm._validate_states_for_merging(
        0,
        1,
        states,
    )


def test_validate_states_for_merging_raises_for_invalid_q1():
    states = ["*", 0, 1, 2]

    with pytest.raises(
        ValueError,
        match="q1 must be a valid state",
    ):
        pm._validate_states_for_merging(
            3,
            1,
            states,
        )


def test_validate_states_for_merging_raises_for_invalid_q2():
    states = ["*", 0, 1, 2]

    with pytest.raises(
        ValueError,
        match="q2 must be a valid state",
    ):
        pm._validate_states_for_merging(
            0,
            3,
            states,
        )


def test_validate_states_for_merging_raises_for_same_state():
    states = ["*", 0, 1, 2]

    with pytest.raises(
        ValueError,
        match="q1 and q2 must refer to different states",
    ):
        pm._validate_states_for_merging(
            1,
            1,
            states,
        )


@pytest.mark.parametrize(
    ("q1", "q2"),
    [
        ("*", 1),
        (1, "*"),
    ],
)
def test_validate_states_for_merging_raises_for_initial_state(
    q1,
    q2,
):
    states = ["*", 0, 1, 2]

    with pytest.raises(
        ValueError,
        match="artificial initial state",
    ):
        pm._validate_states_for_merging(
            q1,
            q2,
            states,
        )


@pytest.mark.parametrize(
    "invalid_alpha",
    [
        "0.2",
        None,
        [0.2],
        1 + 2j,
    ],
)
def test_validate_alpha_raises_for_non_numeric_values(
    invalid_alpha,
):
    with pytest.raises(
        TypeError,
        match="alpha must be numeric",
    ):
        pm._validate_alpha(invalid_alpha)


@pytest.mark.parametrize(
    "invalid_alpha",
    [
        -1.0,
        -0.1,
        0.0,
        2.1,
        3.0,
    ],
)
def test_validate_alpha_raises_for_values_outside_range(
    invalid_alpha,
):
    with pytest.raises(
        ValueError,
        match=r"alpha must be in the range \(0, 2\]",
    ):
        pm._validate_alpha(invalid_alpha)


def test_hoeffding_bound_simple_example(simple_pta):
    assert (
        pm.hoeffding_bound(
            0, 3, 0.2, simple_pta.transition_matrix, simple_pta.alphabet, simple_pta.states
        )
        == False
    )
    assert (
        pm.hoeffding_bound(
            1, 5, 0.2, simple_pta.transition_matrix, simple_pta.alphabet, simple_pta.states
        )
        == True
    )
    assert (
        pm.hoeffding_bound(
            2, 0, 0.2, simple_pta.transition_matrix, simple_pta.alphabet, simple_pta.states
        )
        == True
    )
    assert (
        pm.hoeffding_bound(
            5, 0, 0.2, simple_pta.transition_matrix, simple_pta.alphabet, simple_pta.states
        )
        == False
    )

    assert (
        pm.hoeffding_bound(
            "B",
            "F",
            0.2,
            simple_pta.transition_matrix,
            simple_pta.alphabet,
            simple_pta.alternate_state_names,
        )
        == True
    )
    assert (
        pm.hoeffding_bound(
            "D",
            "A",
            0.2,
            simple_pta.transition_matrix,
            simple_pta.alphabet,
            simple_pta.alternate_state_names,
        )
        == False
    )


def test_hoeffding_bound_arnolds_example(arnolds_example):
    assert (
        pm.hoeffding_bound(
            0,
            1,
            0.2,
            arnolds_example.transition_matrix,
            arnolds_example.alphabet,
            arnolds_example.states,
        )
        == True
    )
    assert (
        pm.hoeffding_bound(
            1,
            2,
            0.2,
            arnolds_example.transition_matrix,
            arnolds_example.alphabet,
            arnolds_example.states,
        )
        == True
    )
    assert (
        pm.hoeffding_bound(
            0,
            1,
            0.9,
            arnolds_example.transition_matrix,
            arnolds_example.alphabet,
            arnolds_example.states,
        )
        == False
    )
    assert (
        pm.hoeffding_bound(
            1,
            3,
            0.9,
            arnolds_example.transition_matrix,
            arnolds_example.alphabet,
            arnolds_example.states,
        )
        == True
    )


def test_merge_two_states_simple_example(simple_pta):
    obtained_transition_matrix, obtained_states = pm.merge_two_states(
        0, 1, simple_pta.transition_matrix, simple_pta.states, simple_pta.alphabet
    )
    expected_transition_matrix = np.array(
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
    assert np.allclose(obtained_transition_matrix, expected_transition_matrix)
    assert obtained_states == expected_states

    obtained_transition_matrix, obtained_states = pm.merge_two_states(
        2, 6, simple_pta.transition_matrix, simple_pta.states, simple_pta.alphabet
    )
    expected_transition_matrix = np.array(
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
    assert np.allclose(obtained_transition_matrix, expected_transition_matrix)
    assert obtained_states == expected_states


def test_merge_two_states_arnolds_example(arnolds_example):
    obtained_transition_matrix, obtained_states = pm.merge_two_states(
        0,
        1,
        arnolds_example.transition_matrix,
        arnolds_example.states,
        arnolds_example.alphabet,
    )
    expected_transition_matrix = np.array(
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
    assert np.allclose(obtained_transition_matrix, expected_transition_matrix)
    assert obtained_states == expected_states


def test_check_is_deterministic_simple_example(simple_pta):
    obtained_nondeterministic_pairs = pm.check_is_deterministic(
        simple_pta.transition_matrix, simple_pta.states, simple_pta.alphabet
    )
    expected_nondeterministic_pairs = []
    assert obtained_nondeterministic_pairs == expected_nondeterministic_pairs

    obtained_transition_matrix, obtained_states = pm.merge_two_states(
        0, 1, simple_pta.transition_matrix, simple_pta.states, simple_pta.alphabet
    )
    obtained_nondeterministic_pairs = pm.check_is_deterministic(
        obtained_transition_matrix, obtained_states, simple_pta.alphabet
    )
    expected_nondeterministic_pairs = [(2, 3)]
    assert obtained_nondeterministic_pairs == expected_nondeterministic_pairs

    obtained_transition_matrix, obtained_states = pm.merge_two_states(
        2, 6, simple_pta.transition_matrix, simple_pta.states, simple_pta.alphabet
    )
    obtained_nondeterministic_pairs = pm.check_is_deterministic(
        obtained_transition_matrix, obtained_states, simple_pta.alphabet
    )
    expected_nondeterministic_pairs = []
    assert obtained_nondeterministic_pairs == expected_nondeterministic_pairs

    obtained_transition_matrix, obtained_states = pm.merge_two_states(
        0, 2, simple_pta.transition_matrix, simple_pta.states, simple_pta.alphabet
    )
    obtained_nondeterministic_pairs = pm.check_is_deterministic(
        obtained_transition_matrix, obtained_states, simple_pta.alphabet
    )
    expected_nondeterministic_pairs = [(1, 4), (0, 5)]
    assert obtained_nondeterministic_pairs == expected_nondeterministic_pairs


def test_check_is_deterministic_arnolds_example(arnolds_example):
    obtained_nondeterministic_pairs = pm.check_is_deterministic(
        arnolds_example.transition_matrix, arnolds_example.states, arnolds_example.alphabet
    )
    expected_nondeterministic_pairs = []
    assert obtained_nondeterministic_pairs == expected_nondeterministic_pairs

    obtained_transition_matrix, obtained_states = pm.merge_two_states(
        0,
        1,
        arnolds_example.transition_matrix,
        arnolds_example.states,
        arnolds_example.alphabet,
    )
    obtained_nondeterministic_pairs = pm.check_is_deterministic(
        obtained_transition_matrix, obtained_states, arnolds_example.alphabet
    )
    expected_nondeterministic_pairs = [(2, 3)]
    assert obtained_nondeterministic_pairs == expected_nondeterministic_pairs

    obtained_transition_matrix, obtained_states = pm.merge_two_states(
        0,
        3,
        arnolds_example.transition_matrix,
        arnolds_example.states,
        arnolds_example.alphabet,
    )
    obtained_nondeterministic_pairs = pm.check_is_deterministic(
        obtained_transition_matrix, obtained_states, arnolds_example.alphabet
    )
    expected_nondeterministic_pairs = [(1, 5), (2, 6)]
    assert obtained_nondeterministic_pairs == expected_nondeterministic_pairs


def test_check_is_deterministic_with_multiple_pairs_simple_example(simple_pta):
    wrong_transition_matrix = np.array(
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
        wrong_transition_matrix, simple_pta.states, simple_pta.alphabet
    )
    expected_nondeterministic_pairs = [(1, 6), (1, 4), (2, 4)]
    assert obtained_nondeterministic_pairs == expected_nondeterministic_pairs


def test_recursive_merge_raises_for_invalid_output(simple_pta):
    with pytest.raises(ValueError, match="output_level must be"):
        pm.recursive_merge_two_states(
            1,
            2,
            simple_pta.transition_matrix,
            simple_pta.states,
            0.2,
            simple_pta.alphabet,
            output_level="Invalid",
        )


def test_recursive_merge_raises_for_invalid_method(simple_pta):
    with pytest.raises(ValueError, match="method must be"):
        pm.recursive_merge_two_states(
            1,
            2,
            simple_pta.transition_matrix,
            simple_pta.states,
            0.2,
            simple_pta.alphabet,
            method="Invalid",
        )


def test_recursive_merge_higuera_requires_red_states(simple_pta):
    with pytest.raises(
        ValueError,
        match="red_states must be provided",
    ):
        pm.recursive_merge_two_states(
            1,
            2,
            simple_pta.transition_matrix,
            simple_pta.states,
            0.2,
            simple_pta.alphabet,
            method="Higuera",
        )


def test_recursive_merge_two_states_prints_nondeterministic_pairs(
    capsys,
    monkeypatch,
):
    transition_matrix = np.zeros((1, 5, 5), dtype=int)
    states = ["*", 0, 1, 2, 3]
    alphabet = ["A"]

    transition_matrix[0, 2, 1] = 1
    transition_matrix[0, 3, 4] = 1

    monkeypatch.setattr(
        pm,
        "hoeffding_bound",
        lambda *args, **kwargs: False,
    )

    pm._recursive_merge_two_states(
        q1=1,
        q2=2,
        transition_matrix=transition_matrix,
        states=states,
        alpha=0.2,
        alphabet=alphabet,
        output_level="Full",
        method="Carrasco",
    )

    captured = capsys.readouterr()

    assert (
        "Merging of states (1, 2) results in "
        "non-deterministic pairs: [(0, 3)]" in captured.out
    )


def test_recursive_merge_two_states_prints_successful_merge(capsys):
    states = ["*", 0, 1, 2, 3]
    alphabet = ["A"]

    transition_matrix = np.zeros((1, 5, 5), dtype=int)

    transition_matrix[0, 0, 1] = 10
    transition_matrix[0, 1, 3] = 5
    transition_matrix[0, 2, 4] = 5

    new_matrix, new_states, recursive_merge = pm._recursive_merge_two_states(
        0,
        1,
        transition_matrix,
        states,
        alpha=0.05,
        alphabet=alphabet,
        output_level="Full",
        method="Carrasco",
    )

    captured = capsys.readouterr()

    assert (
        "Successfully merged states "
        "(2, 3) "
        "into a deterministic state." in captured.out
    )
    assert recursive_merge is True
    assert new_states == ["*", 0, 2]
    assert new_matrix.shape == (1, 3, 3)


def test_recursive_merge_two_states_prints_new_nondeterministic_pairs(
    capsys,
):
    states = ["*", 0, 1, 2, 3, 4, 5]
    alphabet = ["A"]

    transition_matrix = np.zeros((1, 7, 7), dtype=int)

    transition_matrix[0, 0, 1] = 10
    transition_matrix[0, 1, 3] = 5
    transition_matrix[0, 2, 4] = 5
    transition_matrix[0, 3, 5] = 5
    transition_matrix[0, 4, 6] = 5

    new_matrix, new_states, recursive_merge = pm._recursive_merge_two_states(
        0,
        1,
        transition_matrix,
        states,
        alpha=0.05,
        alphabet=alphabet,
        output_level="Full",
        method="Carrasco",
    )

    captured = capsys.readouterr()

    assert (
        "Merging of previous non-deterministic pair "
        "results in non-deterministic pairs: [(4, 5)]" in captured.out
    )
    assert recursive_merge is True
    assert new_states == ["*", 0, 2, 4]
    assert new_matrix.shape == (1, 4, 4)


def test_recursive_merge_two_states_higuera_prints_nondeterministic_pairs(
    capsys,
):
    states = ["*", 0, 1, 2, 3]
    alphabet = ["A"]
    red_states = [0]

    transition_matrix = np.zeros((1, 5, 5), dtype=int)

    transition_matrix[0, 0, 1] = 10
    transition_matrix[0, 1, 3] = 5
    transition_matrix[0, 2, 4] = 5

    (
        new_matrix,
        new_states,
        recursive_merge,
        new_red_states,
    ) = pm._recursive_merge_two_states(
        0,
        1,
        transition_matrix,
        states,
        alpha=0.05,
        alphabet=alphabet,
        red_states=red_states,
        output_level="Full",
        method="Higuera",
    )

    captured = capsys.readouterr()

    assert (
        "Merging of states "
        "(0, 1) "
        "results in non-deterministic pairs: [(2, 3)]" in captured.out
    )
    assert (
        "Successfully merged states "
        "(2, 3) "
        "into a deterministic state." in captured.out
    )
    assert recursive_merge is True
    assert new_states == ["*", 0, 2]
    assert new_red_states == [0]
    assert new_matrix.shape == (1, 3, 3)


def test_recursive_merge_two_states_keeps_red_states_unique():
    states = ["*", 0, 1, 2, 3]
    alphabet = ["A"]
    red_states = [0, 2, 3]

    transition_matrix = np.zeros((1, 5, 5), dtype=int)

    transition_matrix[0, 0, 1] = 10
    transition_matrix[0, 1, 3] = 5
    transition_matrix[0, 2, 4] = 5

    (
        new_matrix,
        new_states,
        recursive_merge,
        new_red_states,
    ) = pm._recursive_merge_two_states(
        0,
        1,
        transition_matrix,
        states,
        alpha=0.05,
        alphabet=alphabet,
        red_states=red_states,
        method="Higuera",
    )

    assert recursive_merge is True
    assert new_states == ["*", 0, 2]
    assert new_matrix.shape == (1, 3, 3)
    assert new_red_states == [0, 2]
    assert len(new_red_states) == len(set(new_red_states))


def test_recursive_merge_two_states_higuera_prints_new_nondeterministic_pairs(
    capsys,
):
    states = ["*", 0, 1, 2, 3, 4, 5]
    alphabet = ["A"]
    red_states = [0]

    transition_matrix = np.zeros((1, 7, 7), dtype=int)

    transition_matrix[0, 0, 1] = 10
    transition_matrix[0, 1, 3] = 5
    transition_matrix[0, 2, 4] = 5
    transition_matrix[0, 3, 5] = 5
    transition_matrix[0, 4, 6] = 5

    (
        new_matrix,
        new_states,
        recursive_merge,
        new_red_states,
    ) = pm._recursive_merge_two_states(
        0,
        1,
        transition_matrix,
        states,
        alpha=0.05,
        alphabet=alphabet,
        red_states=red_states,
        output_level="Full",
        method="Higuera",
    )

    captured = capsys.readouterr()

    assert (
        "Merging of previous non-deterministic pair "
        "results in non-deterministic pairs: [(4, 5)]" in captured.out
    )
    assert recursive_merge is True
    assert new_states == ["*", 0, 2, 4]
    assert new_red_states == [0]
    assert new_matrix.shape == (1, 4, 4)


def test_recursive_merge_two_states_restores_initial_input(
    monkeypatch,
):
    states = ["*", 0, 1, 2, 3, 4, 5]
    alphabet = ["A"]
    red_states = [0, 2, 3]

    transition_matrix = np.zeros((1, 7, 7), dtype=int)

    nondeterministic_results = iter(
        [
            [(2, 3)],
            [(4, 5)],
        ]
    )

    hoeffding_results = iter(
        [
            True,
            False,
        ]
    )

    monkeypatch.setattr(
        pm,
        "check_is_deterministic",
        lambda *args, **kwargs: next(nondeterministic_results),
    )

    monkeypatch.setattr(
        pm,
        "hoeffding_bound",
        lambda *args, **kwargs: next(hoeffding_results),
    )

    (
        obtained_matrix,
        obtained_states,
        obtained_recursive_merge,
        obtained_red_states,
    ) = pm.recursive_merge_two_states(
        0,
        1,
        transition_matrix,
        states,
        alpha=0.05,
        alphabet=alphabet,
        red_states=red_states,
        method="Higuera",
    )

    assert np.array_equal(
        obtained_matrix,
        transition_matrix,
    )
    assert obtained_states == states
    assert obtained_recursive_merge is False
    assert obtained_red_states == red_states


def test_recursive_merge_two_states_simple_example(simple_pta):
    (
        obtained_matrix,
        obtained_states,
        obtained_recursive_merge,
    ) = pm.recursive_merge_two_states(
        1, 2, simple_pta.transition_matrix, simple_pta.states, 0.2, simple_pta.alphabet
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


def test_recursive_merge_two_states_arnolds_example(arnolds_example):
    (
        obtained_matrix,
        obtained_states,
        obtained_recursive_merge,
    ) = pm.recursive_merge_two_states(
        0,
        1,
        arnolds_example.transition_matrix,
        arnolds_example.states,
        0.2,
        arnolds_example.alphabet,
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

    (
        obtained_matrix,
        obtained_states,
        obtained_recursive_merge,
    ) = pm.recursive_merge_two_states(
        0,
        3,
        arnolds_example.transition_matrix,
        arnolds_example.states,
        0.9,
        arnolds_example.alphabet,
    )
    assert np.allclose(obtained_matrix, arnolds_example.transition_matrix)
    assert obtained_states == arnolds_example.states
    assert obtained_recursive_merge == False


def test_recursive_merge_two_states_with_red_states_simple_example(simple_pta):
    (
        obtained_matrix,
        obtained_states,
        obtained_recursive_merge,
        red_states,
    ) = pm.recursive_merge_two_states(
        1,
        2,
        simple_pta.transition_matrix,
        simple_pta.states,
        0.2,
        simple_pta.alphabet,
        simple_pta.red_states,
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
    assert red_states == simple_pta.red_states


def test_recursive_merge_two_states_with_red_states_arnolds_example(arnolds_example):
    (
        obtained_matrix,
        obtained_states,
        obtained_recursive_merge,
        red_states,
    ) = pm.recursive_merge_two_states(
        0,
        1,
        arnolds_example.transition_matrix,
        arnolds_example.states,
        0.2,
        arnolds_example.alphabet,
        arnolds_example.red_states,
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
    assert red_states == arnolds_example.red_states


def test_recursive_merge_two_states_with_red_states_arnolds_example_failure(
    arnolds_example,
):
    (
        obtained_matrix,
        obtained_states,
        obtained_recursive_merge,
        red_states,
    ) = pm.recursive_merge_two_states(
        0,
        3,
        arnolds_example.transition_matrix,
        arnolds_example.states,
        0.9,
        arnolds_example.alphabet,
        arnolds_example.red_states,
        method="Higuera",
    )
    assert np.allclose(obtained_matrix, arnolds_example.transition_matrix)
    assert obtained_states == arnolds_example.states
    assert obtained_recursive_merge == False
    assert red_states == arnolds_example.red_states


def test_recursive_merge_two_states_with_red_states_arnolds_example_merge_red_state(
    arnolds_example,
):
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
        arnolds_example.transition_matrix_after_merges,
        arnolds_example.states_after_merges,
        0.9,
        arnolds_example.alphabet,
        arnolds_example.red_states_after_merges,
        method="Higuera",
    )
    assert np.allclose(obtained_matrix, expected_matrix)
    assert obtained_states == expected_states
    assert obtained_recursive_merge == expected_recursive_merge
    assert red_states == expected_red_states
