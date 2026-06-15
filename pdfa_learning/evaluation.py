"""
Functions for evaluating learned probabilistic deterministic finite automata.

This module provides tools for comparing the sequence probabilities assigned
by a learned PDFA with empirical probabilities observed in held-out data.
It includes functions for constructing sequence-level probability tables and
calculating measures of predictive fit, coverage, and probability allocation.
"""

from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from .probabilities import probability_estimate_of_exact_sequence


_REQUIRED_TABLE_COLUMNS = {
    "sequence",
    "count",
    "empirical_probability",
    "pdfa_probability",
}


def _prepare_sequences(
    sequences: Iterable[str],
    *,
    parameter_name: str,
):
    """
    Convert an iterable of sequences to a validated list.

    Parameters
    ----------
    sequences : iterable of str
        Sequences to validate.
    parameter_name : str
        Name of the parameter being validated. This is used in error
        messages.

    Returns
    -------
    list of str
        Validated sequences.

    Raises
    ------
    TypeError
        If sequences is a single string, is not iterable, or contains
        non-string values.
    ValueError
        If no sequences are supplied or an empty sequence is present.
    """
    if isinstance(sequences, str):
        raise TypeError(
            f"{parameter_name} must be an iterable of strings, "
            "not a single string."
        )

    try:
        sequence_list = list(sequences)
    except TypeError as exc:
        raise TypeError(
            f"{parameter_name} must be an iterable of strings."
        ) from exc

    if not sequence_list:
        raise ValueError(
            f"{parameter_name} must contain at least one sequence."
        )

    if not all(isinstance(sequence, str) for sequence in sequence_list):
        raise TypeError(
            f"All values in {parameter_name} must be strings."
        )

    if any(sequence == "" for sequence in sequence_list):
        raise ValueError(
            f"{parameter_name} must not contain empty sequences."
        )

    return sequence_list


def _validate_probability_table(
    table: pd.DataFrame,
    *,
    parameter_name: str,
):
    """
    Validate a sequence probability table.

    Parameters
    ----------
    table : pandas.DataFrame
        Table to validate.
    parameter_name : str
        Name of the parameter being validated.

    Raises
    ------
    TypeError
        If table is not a pandas DataFrame.
    ValueError
        If required columns are missing or the table is empty.
    """
    if not isinstance(table, pd.DataFrame):
        raise TypeError(
            f"{parameter_name} must be a pandas DataFrame."
        )

    missing_columns = _REQUIRED_TABLE_COLUMNS.difference(table.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))

        raise ValueError(
            f"{parameter_name} is missing the following required "
            f"columns: {missing}."
        )

    if table.empty:
        raise ValueError(
            f"{parameter_name} must contain at least one row."
        )


def get_sequence_probability_table(
    sequences: Iterable[str],
    probability_matrix: np.ndarray,
    alphabet: Sequence[str],
):
    """
    Compare empirical and PDFA probabilities for observed sequences.

    A single row is returned for each unique sequence. The empirical
    probability is calculated from the relative frequency of the sequence
    in the supplied data. The PDFA probability is calculated using the
    learned probability transition matrix.

    Parameters
    ----------
    sequences : iterable of str
        Observed sequences. Repeated sequences are used to calculate their
        empirical frequencies.
    probability_matrix : numpy.ndarray
        Probability transition matrix representing the learned PDFA.
    alphabet : sequence of str
        Ordered alphabet corresponding to the first dimension of
        `probability_matrix`.

    Returns
    -------
    pandas.DataFrame
        Sequence-level results with the columns:
        - `sequence`
        - `count`
        - `empirical_probability`
        - `pdfa_probability`

    Raises
    ------
    TypeError
        If sequences is invalid.
    ValueError
        If no sequences are supplied or an empty sequence is present.

    Notes
    -----
    The returned table contains one row per unique sequence rather than one
    row per observation. Sequence order follows the order in which each
    unique sequence first appears in `sequences`.
    """
    sequence_list = _prepare_sequences(
        sequences,
        parameter_name="sequences",
    )

    counts = Counter(sequence_list)
    total_count = len(sequence_list)

    rows = []

    for sequence, count in counts.items():
        empirical_probability = count / total_count

        pdfa_probability = probability_estimate_of_exact_sequence(
            probability_matrix,
            sequence,
            alphabet,
        )

        rows.append(
            {
                "sequence": sequence,
                "count": count,
                "empirical_probability": float(
                    empirical_probability
                ),
                "pdfa_probability": float(pdfa_probability),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "sequence",
            "count",
            "empirical_probability",
            "pdfa_probability",
        ],
    )


def calculate_pdfa_metrics(
    test_probability_table: pd.DataFrame,
    train_probability_table: pd.DataFrame,
):
    """
    Calculate evaluation measures for a learned PDFA.

    Parameters
    ----------
    test_probability_table : pandas.DataFrame
        Sequence probability table for the held-out test sequences, as
        returned by `get_sequence_probability_table`.
    train_probability_table : pandas.DataFrame
        Sequence probability table for the training sequences, as returned
        by `get_sequence_probability_table`.

    Returns
    -------
    dict
        Dictionary containing the following measures:

        `mean_absolute_distance`
            Mean absolute difference between the empirical test
            probabilities and the unnormalised probabilities assigned by
            the PDFA.

        `jensen_shannon_divergence`
            Jensen-Shannon divergence between the empirical and PDFA
            distributions over the observed test support. The PDFA
            probabilities are normalised over the observed test sequences
            before calculating this measure.

        `sequence_coverage`
            Proportion of unique test sequences assigned a non-zero
            probability by the PDFA.

        `probability_mass_coverage`
            Proportion of the empirical test probability mass belonging to
            sequences assigned a non-zero probability by the PDFA.

        `training_sequence_pdfa_mass`
            Total PDFA probability assigned to unique sequences observed in
            the training data.

        `outside_training_sequence_pdfa_mass`
            Remaining PDFA probability mass after subtracting the mass
            assigned to unique training sequences from one.

        `unseen_test_sequence_proportion`
            Proportion of unique test sequences that were not observed in
            the training data.

        `unseen_test_empirical_mass`
            Empirical test probability mass belonging to sequences that
            were not observed in the training data.

        `unseen_test_pdfa_mass`
            Total PDFA probability assigned to observed test sequences that
            were not present in the training data.

    Raises
    ------
    TypeError
        If either input is not a pandas DataFrame.
    ValueError
        If either table is empty, required columns are missing, or
        probability values are invalid.

    Notes
    -----
    Jensen-Shannon divergence cannot be calculated when the PDFA assigns
    zero probability to every sequence in the test table. In this case,
    `jensen_shannon_divergence` is returned as `numpy.nan`.
    """
    _validate_probability_table(
        test_probability_table,
        parameter_name="test_probability_table",
    )

    _validate_probability_table(
        train_probability_table,
        parameter_name="train_probability_table",
    )

    test_df = test_probability_table.copy()
    train_df = train_probability_table.copy()

    empirical_probabilities = test_df[
        "empirical_probability"
    ].to_numpy(dtype=float)

    pdfa_probabilities = test_df[
        "pdfa_probability"
    ].to_numpy(dtype=float)

    train_pdfa_probabilities = train_df[
        "pdfa_probability"
    ].to_numpy(dtype=float)

    if not np.all(np.isfinite(empirical_probabilities)):
        raise ValueError(
            "test_probability_table contains non-finite empirical "
            "probabilities."
        )

    if not np.all(np.isfinite(pdfa_probabilities)):
        raise ValueError(
            "test_probability_table contains non-finite PDFA "
            "probabilities."
        )

    if not np.all(np.isfinite(train_pdfa_probabilities)):
        raise ValueError(
            "train_probability_table contains non-finite PDFA "
            "probabilities."
        )

    if np.any(empirical_probabilities < 0):
        raise ValueError(
            "Empirical probabilities must not be negative."
        )

    if np.any(pdfa_probabilities < 0):
        raise ValueError(
            "PDFA probabilities must not be negative."
        )

    if np.any(train_pdfa_probabilities < 0):
        raise ValueError(
            "Training-set PDFA probabilities must not be negative."
        )

    empirical_probability_sum = empirical_probabilities.sum()

    if empirical_probability_sum <= 0:
        raise ValueError(
            "The empirical test probabilities must have a positive sum."
        )

    empirical_distribution = (
        empirical_probabilities / empirical_probability_sum
    )

    mean_absolute_distance = float(
        np.mean(
            np.abs(
                empirical_distribution - pdfa_probabilities
            )
        )
    )

    pdfa_probability_sum = pdfa_probabilities.sum()

    if pdfa_probability_sum > 0:
        pdfa_distribution = (
            pdfa_probabilities / pdfa_probability_sum
        )

        jensen_shannon_divergence = float(
            jensenshannon(
                empirical_distribution,
                pdfa_distribution,
            )
            ** 2
        )
    else:
        jensen_shannon_divergence = float("nan")

    covered_sequences = pdfa_probabilities > 0

    sequence_coverage = float(
        np.mean(covered_sequences)
    )

    probability_mass_coverage = float(
        empirical_distribution[covered_sequences].sum()
    )

    training_sequence_pdfa_mass = float(
        train_pdfa_probabilities.sum()
    )

    outside_training_sequence_pdfa_mass = float(
        1.0 - training_sequence_pdfa_mass
    )

    training_sequences = set(train_df["sequence"])

    unseen_test_mask = (
        ~test_df["sequence"].isin(training_sequences)
    ).to_numpy()

    unseen_test_sequence_proportion = float(
        np.mean(unseen_test_mask)
    )

    unseen_test_empirical_mass = float(
        empirical_distribution[unseen_test_mask].sum()
    )

    unseen_test_pdfa_mass = float(
        pdfa_probabilities[unseen_test_mask].sum()
    )

    return {
        "mean_absolute_distance": mean_absolute_distance,
        "jensen_shannon_divergence": (
            jensen_shannon_divergence
        ),
        "sequence_coverage": sequence_coverage,
        "probability_mass_coverage": (
            probability_mass_coverage
        ),
        "training_sequence_pdfa_mass": (
            training_sequence_pdfa_mass
        ),
        "outside_training_sequence_pdfa_mass": (
            outside_training_sequence_pdfa_mass
        ),
        "unseen_test_sequence_proportion": (
            unseen_test_sequence_proportion
        ),
        "unseen_test_empirical_mass": (
            unseen_test_empirical_mass
        ),
        "unseen_test_pdfa_mass": (
            unseen_test_pdfa_mass
        ),
    }


def evaluate_pdfa(
    test_sequences: Iterable[str],
    train_sequences: Iterable[str],
    probability_matrix: np.ndarray,
    alphabet: Sequence[str],
):
    """
    Evaluate a learned PDFA against training and test sequences.

    This function constructs sequence probability tables for the training
    and test data and then calculates the PDFA evaluation measures returned
    by `calculate_pdfa_metrics`.

    Parameters
    ----------
    test_sequences : iterable of str
        Held-out sequences used to evaluate the learned PDFA.
    train_sequences : iterable of str
        Sequences used to learn the PDFA.
    probability_matrix : numpy.ndarray
        Probability transition matrix representing the learned PDFA.
    alphabet : sequence of str
        Ordered alphabet corresponding to the first dimension of
        `probability_matrix`.

    Returns
    -------
    dict
        Named PDFA evaluation measures.

    Raises
    ------
    TypeError
        If either sequence collection is invalid.
    ValueError
        If either sequence collection is empty or contains an empty
        sequence.
    """
    test_probability_table = get_sequence_probability_table(
        test_sequences,
        probability_matrix,
        alphabet,
    )

    train_probability_table = get_sequence_probability_table(
        train_sequences,
        probability_matrix,
        alphabet,
    )

    return calculate_pdfa_metrics(
        test_probability_table,
        train_probability_table,
    )