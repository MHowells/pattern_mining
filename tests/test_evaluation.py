import numpy as np
import pandas as pd
import pdfa_learning as pm
import pytest


def _make_probability_matrix():
    alphabet = ["A", "B"]

    probability_matrix = np.zeros((2, 4, 4))

    probability_matrix[0, 1, 2] = 0.6
    probability_matrix[1, 1, 3] = 0.4

    return probability_matrix, alphabet


def _make_probability_tables():
    """
    Create example test and training probability tables.
    """
    test_probability_table = pd.DataFrame(
        {
            "sequence": ["A", "B", "C"],
            "count": [2, 1, 1],
            "empirical_probability": [0.50, 0.25, 0.25],
            "pdfa_probability": [0.40, 0.30, 0.00],
        }
    )

    train_probability_table = pd.DataFrame(
        {
            "sequence": ["A", "B"],
            "count": [3, 1],
            "empirical_probability": [0.75, 0.25],
            "pdfa_probability": [0.40, 0.30],
        }
    )

    return test_probability_table, train_probability_table


def test_get_sequence_probability_table():
    probability_matrix, alphabet = _make_probability_matrix()
    sequences = [
        "A",
        "B",
        "A",
        "AA",
    ]

    obtained_table = pm.get_sequence_probability_table(
        sequences,
        probability_matrix,
        alphabet,
    )

    expected_table = pd.DataFrame(
        {
            "sequence": ["A", "B", "AA"],
            "count": [2, 1, 1],
            "empirical_probability": [0.50, 0.25, 0.25],
            "pdfa_probability": [0.60, 0.40, 0.00],
        }
    )

    pd.testing.assert_frame_equal(
        obtained_table,
        expected_table,
    )


def test_get_sequence_probability_table_accepts_generator():
    probability_matrix, alphabet = _make_probability_matrix()

    sequences = (
        sequence
        for sequence in ["A", "A", "B"]
    )

    obtained_table = pm.get_sequence_probability_table(
        sequences,
        probability_matrix,
        alphabet,
    )

    expected_table = pd.DataFrame(
        {
            "sequence": ["A", "B"],
            "count": [2, 1],
            "empirical_probability": [2 / 3, 1 / 3],
            "pdfa_probability": [0.60, 0.40],
        }
    )

    pd.testing.assert_frame_equal(
        obtained_table,
        expected_table,
    )


def test_get_sequence_probability_table_rejects_single_string():
    probability_matrix, alphabet = _make_probability_matrix()

    with pytest.raises(
        TypeError,
        match="sequences must be an iterable of strings, "
        "not a single string",
    ):
        pm.get_sequence_probability_table(
            "A",
            probability_matrix,
            alphabet,
        )


def test_get_sequence_probability_table_rejects_non_iterable():
    probability_matrix, alphabet = _make_probability_matrix()

    with pytest.raises(
        TypeError,
        match="sequences must be an iterable of strings",
    ):
        pm.get_sequence_probability_table(
            123,
            probability_matrix,
            alphabet,
        )


def test_get_sequence_probability_table_rejects_empty_sequences():
    probability_matrix, alphabet = _make_probability_matrix()

    with pytest.raises(
        ValueError,
        match="sequences must contain at least one sequence",
    ):
        pm.get_sequence_probability_table(
            [],
            probability_matrix,
            alphabet,
        )


def test_get_sequence_probability_table_rejects_non_string_value():
    probability_matrix, alphabet = _make_probability_matrix()

    with pytest.raises(
        TypeError,
        match="All values in sequences must be strings",
    ):
        pm.get_sequence_probability_table(
            ["A", 1],
            probability_matrix,
            alphabet,
        )


def test_get_sequence_probability_table_rejects_empty_string():
    probability_matrix, alphabet = _make_probability_matrix()

    with pytest.raises(
        ValueError,
        match="sequences must not contain empty sequences",
    ):
        pm.get_sequence_probability_table(
            ["A", ""],
            probability_matrix,
            alphabet,
        )


def test_calculate_pdfa_metrics():
    (
        test_probability_table,
        train_probability_table,
    ) = _make_probability_tables()

    obtained_metrics = pm.calculate_pdfa_metrics(
        test_probability_table,
        train_probability_table,
    )

    expected_metrics = {
        "mean_absolute_distance": 0.13333333333333333,
        "jensen_shannon_divergence": 0.09972237622541466,
        "sequence_coverage": 2 / 3,
        "probability_mass_coverage": 0.75,
        "training_sequence_pdfa_mass": 0.70,
        "outside_training_sequence_pdfa_mass": 0.30,
        "unseen_test_sequence_proportion": 1 / 3,
        "unseen_test_empirical_mass": 0.25,
        "unseen_test_pdfa_mass": 0.00,
    }

    assert obtained_metrics.keys() == expected_metrics.keys()

    for metric_name, expected_value in expected_metrics.items():
        assert np.isclose(
            obtained_metrics[metric_name],
            expected_value,
        )


def test_calculate_pdfa_metrics_returns_nan_for_zero_pdfa_mass():
    test_probability_table = pd.DataFrame(
        {
            "sequence": ["A", "B"],
            "count": [1, 1],
            "empirical_probability": [0.50, 0.50],
            "pdfa_probability": [0.00, 0.00],
        }
    )

    train_probability_table = pd.DataFrame(
        {
            "sequence": ["A"],
            "count": [1],
            "empirical_probability": [1.00],
            "pdfa_probability": [0.00],
        }
    )

    obtained_metrics = pm.calculate_pdfa_metrics(
        test_probability_table,
        train_probability_table,
    )

    assert np.isnan(
        obtained_metrics["jensen_shannon_divergence"]
    )
    assert obtained_metrics["sequence_coverage"] == 0
    assert obtained_metrics["probability_mass_coverage"] == 0
    assert obtained_metrics["training_sequence_pdfa_mass"] == 0
    assert obtained_metrics["outside_training_sequence_pdfa_mass"] == 1


def test_calculate_pdfa_metrics_rejects_non_dataframe_test_table():
    _, train_probability_table = _make_probability_tables()

    with pytest.raises(
        TypeError,
        match="test_probability_table must be a pandas DataFrame",
    ):
        pm.calculate_pdfa_metrics(
            [],
            train_probability_table,
        )


def test_calculate_pdfa_metrics_rejects_non_dataframe_train_table():
    test_probability_table, _ = _make_probability_tables()

    with pytest.raises(
        TypeError,
        match="train_probability_table must be a pandas DataFrame",
    ):
        pm.calculate_pdfa_metrics(
            test_probability_table,
            [],
        )


def test_calculate_pdfa_metrics_rejects_missing_column():
    (
        test_probability_table,
        train_probability_table,
    ) = _make_probability_tables()

    test_probability_table = test_probability_table.drop(
        columns="count"
    )

    with pytest.raises(
        ValueError,
        match="missing the following required columns: count",
    ):
        pm.calculate_pdfa_metrics(
            test_probability_table,
            train_probability_table,
        )


def test_calculate_pdfa_metrics_rejects_empty_test_table():
    _, train_probability_table = _make_probability_tables()

    test_probability_table = pd.DataFrame(
        columns=[
            "sequence",
            "count",
            "empirical_probability",
            "pdfa_probability",
        ]
    )

    with pytest.raises(
        ValueError,
        match="test_probability_table must contain at least one row",
    ):
        pm.calculate_pdfa_metrics(
            test_probability_table,
            train_probability_table,
        )


def test_calculate_pdfa_metrics_rejects_empty_train_table():
    test_probability_table, _ = _make_probability_tables()

    train_probability_table = pd.DataFrame(
        columns=[
            "sequence",
            "count",
            "empirical_probability",
            "pdfa_probability",
        ]
    )

    with pytest.raises(
        ValueError,
        match="train_probability_table must contain at least one row",
    ):
        pm.calculate_pdfa_metrics(
            test_probability_table,
            train_probability_table,
        )


def test_calculate_pdfa_metrics_rejects_non_finite_empirical_probability():
    (
        test_probability_table,
        train_probability_table,
    ) = _make_probability_tables()

    test_probability_table.loc[
        0,
        "empirical_probability",
    ] = np.nan

    with pytest.raises(
        ValueError,
        match="contains non-finite empirical probabilities",
    ):
        pm.calculate_pdfa_metrics(
            test_probability_table,
            train_probability_table,
        )


def test_calculate_pdfa_metrics_rejects_non_finite_test_pdfa_probability():
    (
        test_probability_table,
        train_probability_table,
    ) = _make_probability_tables()

    test_probability_table.loc[
        0,
        "pdfa_probability",
    ] = np.inf

    with pytest.raises(
        ValueError,
        match="contains non-finite PDFA probabilities",
    ):
        pm.calculate_pdfa_metrics(
            test_probability_table,
            train_probability_table,
        )


def test_calculate_pdfa_metrics_rejects_non_finite_train_pdfa_probability():
    (
        test_probability_table,
        train_probability_table,
    ) = _make_probability_tables()

    train_probability_table.loc[
        0,
        "pdfa_probability",
    ] = np.nan

    with pytest.raises(
        ValueError,
        match="contains non-finite PDFA probabilities",
    ):
        pm.calculate_pdfa_metrics(
            test_probability_table,
            train_probability_table,
        )


def test_calculate_pdfa_metrics_rejects_negative_empirical_probability():
    (
        test_probability_table,
        train_probability_table,
    ) = _make_probability_tables()

    test_probability_table.loc[
        0,
        "empirical_probability",
    ] = -0.10

    with pytest.raises(
        ValueError,
        match="Empirical probabilities must not be negative",
    ):
        pm.calculate_pdfa_metrics(
            test_probability_table,
            train_probability_table,
        )


def test_calculate_pdfa_metrics_rejects_negative_test_pdfa_probability():
    (
        test_probability_table,
        train_probability_table,
    ) = _make_probability_tables()

    test_probability_table.loc[
        0,
        "pdfa_probability",
    ] = -0.10

    with pytest.raises(
        ValueError,
        match="PDFA probabilities must not be negative",
    ):
        pm.calculate_pdfa_metrics(
            test_probability_table,
            train_probability_table,
        )


def test_calculate_pdfa_metrics_rejects_negative_train_pdfa_probability():
    (
        test_probability_table,
        train_probability_table,
    ) = _make_probability_tables()

    train_probability_table.loc[
        0,
        "pdfa_probability",
    ] = -0.10

    with pytest.raises(
        ValueError,
        match="Training-set PDFA probabilities must not be negative",
    ):
        pm.calculate_pdfa_metrics(
            test_probability_table,
            train_probability_table,
        )


def test_calculate_pdfa_metrics_rejects_zero_empirical_probability_sum():
    (
        test_probability_table,
        train_probability_table,
    ) = _make_probability_tables()

    test_probability_table[
        "empirical_probability"
    ] = 0.0

    with pytest.raises(
        ValueError,
        match="empirical test probabilities must have a positive sum",
    ):
        pm.calculate_pdfa_metrics(
            test_probability_table,
            train_probability_table,
        )


def test_evaluate_pdfa():
    probability_matrix, alphabet = _make_probability_matrix()

    train_sequences = [
        "A",
        "A",
        "B",
    ]

    test_sequences = [
        "A",
        "B",
        "AA",
    ]

    obtained_metrics = pm.evaluate_pdfa(
        test_sequences,
        train_sequences,
        probability_matrix,
        alphabet,
    )

    test_probability_table = pm.get_sequence_probability_table(
        test_sequences,
        probability_matrix,
        alphabet,
    )

    train_probability_table = pm.get_sequence_probability_table(
        train_sequences,
        probability_matrix,
        alphabet,
    )

    expected_metrics = pm.calculate_pdfa_metrics(
        test_probability_table,
        train_probability_table,
    )

    assert obtained_metrics == expected_metrics