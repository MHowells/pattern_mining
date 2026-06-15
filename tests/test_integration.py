import pdfa_learning as pl


def test_pdfa_learning_pipeline_returns_valid_pdfa_carrasco():
    sequences = [
        "A",
        "A",
        "AB",
        "AB",
        "B",
    ]

    alphabet = pl.get_alphabet(sequences)

    transition_matrix = pl.get_transition_matrix(
        sequences,
        alphabet,
    )

    states = pl.get_initial_states(sequences)

    learned_matrix, learned_states, tracking = pl.alergia(
        transition_matrix,
        states,
        alphabet,
        alpha=0.2,
        method="carrasco",
    )

    probability_matrix = pl.probability_transition_matrix(
        learned_matrix,
        learned_states,
        alphabet,
    )

    assert learned_matrix.shape == (
        len(alphabet),
        len(learned_states),
        len(learned_states),
    )

    assert probability_matrix.shape == learned_matrix.shape

    assert (
        pl.check_is_deterministic(
            learned_matrix,
            learned_states,
            alphabet,
        )
        == []
    )

    assert tracking["initial_states"] == len(states)
    assert tracking["final_states"] == len(learned_states)


def test_pdfa_learning_pipeline_returns_valid_pdfa_higuera():
    sequences = [
        "A",
        "A",
        "AB",
        "AB",
        "B",
    ]

    alphabet = pl.get_alphabet(sequences)

    transition_matrix = pl.get_transition_matrix(
        sequences,
        alphabet,
    )

    states = pl.get_initial_states(sequences)

    learned_matrix, learned_states, tracking = pl.alergia(
        transition_matrix,
        states,
        alphabet,
        alpha=0.2,
        method="de_la_higuera",
    )

    probability_matrix = pl.probability_transition_matrix(
        learned_matrix,
        learned_states,
        alphabet,
    )

    assert learned_matrix.shape == (
        len(alphabet),
        len(learned_states),
        len(learned_states),
    )

    assert probability_matrix.shape == learned_matrix.shape

    assert (
        pl.check_is_deterministic(
            learned_matrix,
            learned_states,
            alphabet,
        )
        == []
    )

    assert tracking["initial_states"] == len(states)
    assert tracking["final_states"] == len(learned_states)
