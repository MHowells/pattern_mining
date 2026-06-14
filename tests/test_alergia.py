import numpy as np
import pattern_mining as pm
import pytest


def test_get_blue_states_simple_example(simple_pta):
    obtained_blue_states = pm.get_blue_states(
        simple_pta.pathway_matrix, simple_pta.red_states, simple_pta.states
    )
    expected_blue_states = [1, 2]
    assert obtained_blue_states == expected_blue_states


def test_get_blue_states_arnolds_example(arnolds_example):
    obtained_blue_states = pm.get_blue_states(
        arnolds_example.pathway_matrix,
        arnolds_example.red_states,
        arnolds_example.states,
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


def test_alergia_raises_for_invalid_output(simple_pta):
    with pytest.raises(
        ValueError,
        match="output must be",
    ):
        pm.alergia(
            simple_pta.pathway_matrix,
            simple_pta.states,
            simple_pta.alphabet,
            alpha=0.2,
            output="Invalid",
        )


def test_alergia_raises_for_invalid_method(simple_pta):
    with pytest.raises(
        ValueError,
        match="method must be",
    ):
        pm.alergia(
            simple_pta.pathway_matrix,
            simple_pta.states,
            simple_pta.alphabet,
            alpha=0.2,
            method="Invalid",
        )


def test_alergia_raises_for_invalid_alpha_less_than_or_equal_to_zero(simple_pta):
    with pytest.raises(
        ValueError,
        match="alpha must be in",
    ):
        pm.alergia(
            simple_pta.pathway_matrix,
            simple_pta.states,
            simple_pta.alphabet,
            alpha=0,
        )


def test_alergia_raises_for_invalid_alpha_greater_than_two(simple_pta):
    with pytest.raises(
        ValueError,
        match="alpha must be in",
    ):
        pm.alergia(
            simple_pta.pathway_matrix,
            simple_pta.states,
            simple_pta.alphabet,
            alpha=2.1,
        )


def test_alergia_prints_next_pair_of_states(
    monkeypatch,
    capsys,
):
    states = ["*", 0, 1]
    alphabet = ["A"]

    transition_matrix = np.zeros((1, 3, 3), dtype=int)

    monkeypatch.setattr(
        pm,
        "hoeffding_bound",
        lambda *args, **kwargs: False,
    )

    final_matrix, final_states, tracking = pm.alergia(
        transition_matrix,
        states,
        alphabet,
        alpha=0.05,
        output="Truncated",
        method="Carrasco",
    )

    captured = capsys.readouterr()

    assert "The next pair of states to check is: (1, 0)" in captured.out
    assert final_states == states
    assert np.array_equal(final_matrix, transition_matrix)
    assert tracking["attempted_merges"] == 1
    assert tracking["successful_merges"] == 0


def test_alergia_prints_iteration_number(
    monkeypatch,
    capsys,
):
    states = ["*", 0, 1]
    alphabet = ["A"]

    transition_matrix = np.zeros((1, 3, 3), dtype=int)

    monkeypatch.setattr(
        pm,
        "hoeffding_bound",
        lambda *args, **kwargs: False,
    )

    final_matrix, final_states, tracking = pm.alergia(
        transition_matrix,
        states,
        alphabet,
        alpha=0.05,
        output="Full",
        method="Carrasco",
    )

    captured = capsys.readouterr()

    assert "Iteration 1" in captured.out
    assert final_states == states
    assert np.array_equal(final_matrix, transition_matrix)
    assert tracking["attempted_merges"] == 1
    assert tracking["successful_merges"] == 0


def test_alergia_prints_when_hoeffding_bound_is_satisfied(
    monkeypatch,
    capsys,
):
    states = ["*", 0, 1]
    alphabet = ["A"]

    transition_matrix = np.zeros((1, 3, 3), dtype=int)

    monkeypatch.setattr(
        pm,
        "hoeffding_bound",
        lambda *args, **kwargs: True,
    )

    monkeypatch.setattr(
        pm,
        "_recursive_merge_two_states",
        lambda *args, **kwargs: (
            transition_matrix,
            states,
            False,
        ),
    )

    final_matrix, final_states, tracking = pm.alergia(
        transition_matrix,
        states,
        alphabet,
        alpha=0.05,
        output="Truncated",
        method="Carrasco",
    )

    captured = capsys.readouterr()

    assert "Hoeffding Bound satisfied for (1, 0)" in captured.out
    assert np.array_equal(final_matrix, transition_matrix)
    assert final_states == states
    assert tracking["attempted_merges"] == 1
    assert tracking["recursive_merge_attempts"] == 1
    assert tracking["recursive_merge_failures"] == 1
    assert tracking["successful_merges"] == 0


def test_alergia_prints_successful_recursive_merge(
    monkeypatch,
    capsys,
):
    states = ["*", 0, 1]
    alphabet = ["A"]

    transition_matrix = np.zeros((1, 3, 3), dtype=int)
    merged_matrix = np.zeros((1, 2, 2), dtype=int)
    merged_states = ["*", 0]

    monkeypatch.setattr(
        pm,
        "hoeffding_bound",
        lambda *args, **kwargs: True,
    )

    monkeypatch.setattr(
        pm,
        "_recursive_merge_two_states",
        lambda *args, **kwargs: (
            merged_matrix,
            merged_states,
            True,
        ),
    )

    final_matrix, final_states, tracking = pm.alergia(
        transition_matrix,
        states,
        alphabet,
        alpha=0.05,
        output="Truncated",
        method="Carrasco",
    )

    captured = capsys.readouterr()

    assert "Recursively merged states. " "Successfully merged (1, 0)" in captured.out
    assert np.array_equal(final_matrix, merged_matrix)
    assert final_states == merged_states
    assert tracking["attempted_merges"] == 1
    assert tracking["successful_merges"] == 1
    assert tracking["recursive_merge_attempts"] == 1
    assert tracking["recursive_merge_failures"] == 0


def test_alergia_prints_failed_recursive_merge(
    monkeypatch,
    capsys,
):
    states = ["*", 0, 1]
    alphabet = ["A"]

    transition_matrix = np.zeros((1, 3, 3), dtype=int)

    monkeypatch.setattr(
        pm,
        "hoeffding_bound",
        lambda *args, **kwargs: True,
    )

    monkeypatch.setattr(
        pm,
        "_recursive_merge_two_states",
        lambda *args, **kwargs: (
            transition_matrix,
            states,
            False,
        ),
    )

    final_matrix, final_states, tracking = pm.alergia(
        transition_matrix,
        states,
        alphabet,
        alpha=0.05,
        output="Truncated",
        method="Carrasco",
    )

    captured = capsys.readouterr()

    assert "Recursive merge process failed. " "Cannot merge (1, 0)" in captured.out
    assert np.array_equal(final_matrix, transition_matrix)
    assert final_states == states
    assert tracking["attempted_merges"] == 1
    assert tracking["successful_merges"] == 0
    assert tracking["recursive_merge_attempts"] == 1
    assert tracking["recursive_merge_failures"] == 1


def test_alergia_prints_when_hoeffding_bound_is_not_satisfied(
    monkeypatch,
    capsys,
):
    states = ["*", 0, 1]
    alphabet = ["A"]

    transition_matrix = np.zeros((1, 3, 3), dtype=int)

    monkeypatch.setattr(
        pm,
        "hoeffding_bound",
        lambda *args, **kwargs: False,
    )

    final_matrix, final_states, tracking = pm.alergia(
        transition_matrix,
        states,
        alphabet,
        alpha=0.05,
        output="Truncated",
        method="Carrasco",
    )

    captured = capsys.readouterr()

    assert "Hoeffding Bound not satisfied for (1, 0)" in captured.out
    assert np.array_equal(final_matrix, transition_matrix)
    assert final_states == states
    assert tracking["attempted_merges"] == 1
    assert tracking["successful_merges"] == 0
    assert tracking["recursive_merge_attempts"] == 0
    assert tracking["recursive_merge_failures"] == 0


def test_alergia_higuera_prints_iteration_number(
    monkeypatch,
    capsys,
):
    states = ["*", 0, 1]
    alphabet = ["A"]

    transition_matrix = np.zeros((1, 3, 3), dtype=int)

    blue_state_results = iter(
        [
            [1],
            [],
        ]
    )

    monkeypatch.setattr(
        pm,
        "get_blue_states",
        lambda *args, **kwargs: next(blue_state_results),
    )

    monkeypatch.setattr(
        pm,
        "hoeffding_bound",
        lambda *args, **kwargs: False,
    )

    final_matrix, final_states, tracking = pm.alergia(
        transition_matrix,
        states,
        alphabet,
        alpha=0.05,
        output="Full",
        method="Higuera",
    )

    captured = capsys.readouterr()

    assert "Iteration 1" in captured.out
    assert np.array_equal(final_matrix, transition_matrix)
    assert final_states == states
    assert tracking["attempted_merges"] == 1
    assert tracking["successful_merges"] == 0


def test_alergia_higuera_prints_when_hoeffding_bound_is_satisfied(
    monkeypatch,
    capsys,
):
    states = ["*", 0, 1]
    alphabet = ["A"]

    transition_matrix = np.zeros((1, 3, 3), dtype=int)

    blue_state_results = iter(
        [
            [1],
            [],
        ]
    )

    monkeypatch.setattr(
        pm,
        "get_blue_states",
        lambda *args, **kwargs: next(blue_state_results),
    )

    monkeypatch.setattr(
        pm,
        "hoeffding_bound",
        lambda *args, **kwargs: True,
    )

    monkeypatch.setattr(
        pm,
        "_recursive_merge_two_states",
        lambda *args, **kwargs: (
            transition_matrix,
            states,
            False,
            [0],
        ),
    )

    final_matrix, final_states, tracking = pm.alergia(
        transition_matrix,
        states,
        alphabet,
        alpha=0.05,
        output="Truncated",
        method="Higuera",
    )

    captured = capsys.readouterr()

    assert "Hoeffding Bound satisfied for (0, 1)" in captured.out
    assert np.array_equal(final_matrix, transition_matrix)
    assert final_states == states
    assert tracking["attempted_merges"] == 1
    assert tracking["recursive_merge_attempts"] == 1
    assert tracking["successful_merges"] == 0
    assert tracking["recursive_merge_failures"] == 1


def test_alergia_higuera_prints_successful_recursive_merge(
    monkeypatch,
    capsys,
):
    states = ["*", 0, 1]
    alphabet = ["A"]

    transition_matrix = np.zeros((1, 3, 3), dtype=int)
    merged_matrix = np.zeros((1, 2, 2), dtype=int)
    merged_states = ["*", 0]

    blue_state_results = iter(
        [
            [1],
            [],
        ]
    )

    monkeypatch.setattr(
        pm,
        "get_blue_states",
        lambda *args, **kwargs: next(blue_state_results),
    )

    monkeypatch.setattr(
        pm,
        "hoeffding_bound",
        lambda *args, **kwargs: True,
    )

    monkeypatch.setattr(
        pm,
        "_recursive_merge_two_states",
        lambda *args, **kwargs: (
            merged_matrix,
            merged_states,
            True,
            [0],
        ),
    )

    final_matrix, final_states, tracking = pm.alergia(
        transition_matrix,
        states,
        alphabet,
        alpha=0.05,
        output="Truncated",
        method="Higuera",
    )

    captured = capsys.readouterr()

    assert "Recursively merged states. " "Successfully merged (0, 1)" in captured.out
    assert np.array_equal(final_matrix, merged_matrix)
    assert final_states == merged_states
    assert tracking["attempted_merges"] == 1
    assert tracking["successful_merges"] == 1
    assert tracking["recursive_merge_attempts"] == 1
    assert tracking["recursive_merge_failures"] == 0


def test_alergia_higuera_prints_when_hoeffding_bound_is_not_satisfied(
    monkeypatch,
    capsys,
):
    states = ["*", 0, 1]
    alphabet = ["A"]

    transition_matrix = np.zeros((1, 3, 3), dtype=int)

    blue_state_results = iter(
        [
            [1],
            [],
        ]
    )

    monkeypatch.setattr(
        pm,
        "get_blue_states",
        lambda *args, **kwargs: next(blue_state_results),
    )

    monkeypatch.setattr(
        pm,
        "hoeffding_bound",
        lambda *args, **kwargs: False,
    )

    final_matrix, final_states, tracking = pm.alergia(
        transition_matrix,
        states,
        alphabet,
        alpha=0.05,
        output="Truncated",
        method="Higuera",
    )

    captured = capsys.readouterr()

    assert "Could not merge blue state 1 with any red state." in captured.out
    assert np.array_equal(final_matrix, transition_matrix)
    assert final_states == states
    assert tracking["attempted_merges"] == 1
    assert tracking["successful_merges"] == 0
    assert tracking["recursive_merge_attempts"] == 0
    assert tracking["recursive_merge_failures"] == 0


def test_alergia_simple_example(simple_pta):
    obtained_matrix, obtained_states, obtained_tracking = pm.alergia(
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

    expected_initial_states = len(simple_pta.states)
    expected_final_states = len(expected_states)
    expected_attempted_merges = 16
    expected_recursive_merge_attempts = 5
    expected_recursive_merge_failures = 2

    expected_tracking = {
        "initial_states": expected_initial_states,
        "final_states": expected_final_states,
        "attempted_merges": expected_attempted_merges,
        "successful_merges": expected_merges,
        "recursive_merge_attempts": expected_recursive_merge_attempts,
        "recursive_merge_failures": expected_recursive_merge_failures,
    }

    assert np.allclose(obtained_matrix, expected_matrix)
    assert obtained_states == expected_states
    assert obtained_tracking == expected_tracking


def test_alergia_arnolds_example_alpha_point_two(arnolds_example):
    obtained_matrix, obtained_states, obtained_tracking = pm.alergia(
        arnolds_example.pathway_matrix,
        arnolds_example.states,
        arnolds_example.alphabet,
        0.2,
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

    expected_initial_states = len(arnolds_example.states)
    expected_final_states = len(expected_states)
    expected_attempted_merges = 3
    expected_recursive_merge_attempts = 3
    expected_recursive_merge_failures = 0

    expected_tracking = {
        "initial_states": expected_initial_states,
        "final_states": expected_final_states,
        "attempted_merges": expected_attempted_merges,
        "successful_merges": expected_merges,
        "recursive_merge_attempts": expected_recursive_merge_attempts,
        "recursive_merge_failures": expected_recursive_merge_failures,
    }

    assert np.allclose(obtained_matrix, expected_matrix)
    assert obtained_states == expected_states
    assert obtained_tracking == expected_tracking


def test_alergia_arnolds_example_alpha_point_nine_carrasco(arnolds_example):
    obtained_matrix, obtained_states, obtained_tracking = pm.alergia(
        arnolds_example.pathway_matrix,
        arnolds_example.states,
        arnolds_example.alphabet,
        0.9,
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

    expected_initial_states = len(arnolds_example.states)
    expected_final_states = len(expected_states)
    expected_attempted_merges = 19
    expected_recursive_merge_attempts = 12
    expected_recursive_merge_failures = 7

    expected_tracking = {
        "initial_states": expected_initial_states,
        "final_states": expected_final_states,
        "attempted_merges": expected_attempted_merges,
        "successful_merges": expected_merges,
        "recursive_merge_attempts": expected_recursive_merge_attempts,
        "recursive_merge_failures": expected_recursive_merge_failures,
    }

    assert np.allclose(obtained_matrix, expected_matrix)
    assert obtained_states == expected_states
    assert obtained_tracking == expected_tracking


def test_alergia_arnolds_example_alpha_point_nine_higuera(arnolds_example):
    obtained_matrix, obtained_states, obtained_tracking = pm.alergia(
        arnolds_example.pathway_matrix,
        arnolds_example.states,
        arnolds_example.alphabet,
        0.9,
        method="Higuera",
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

    expected_initial_states = len(arnolds_example.states)
    expected_final_states = len(expected_states)
    expected_attempted_merges = 19
    expected_recursive_merge_attempts = 10
    expected_recursive_merge_failures = 3

    expected_tracking = {
        "initial_states": expected_initial_states,
        "final_states": expected_final_states,
        "attempted_merges": expected_attempted_merges,
        "successful_merges": expected_merges,
        "recursive_merge_attempts": expected_recursive_merge_attempts,
        "recursive_merge_failures": expected_recursive_merge_failures,
    }

    assert np.allclose(obtained_matrix, expected_matrix)
    assert obtained_states == expected_states
    print(obtained_tracking)
    print(expected_tracking)
    assert obtained_tracking == expected_tracking
