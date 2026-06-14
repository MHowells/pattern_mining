import pattern_mining as pm


def test_pattern_mining_pipeline_returns_valid_pdfa_carrasco():
    sequences = [
        "A",
        "A",
        "AB",
        "AB",
        "B",
    ]

    alphabet = pm.get_alphabet(sequences)

    transition_matrix = pm.get_transition_matrix(
        sequences,
        alphabet,
    )

    states = pm.get_initial_states(sequences)

    learned_matrix, learned_states, tracking = pm.alergia(
        transition_matrix,
        states,
        alphabet,
        alpha=0.2,
        method="Carrasco",
    )

    probability_matrix = pm.probability_transition_matrix(
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
        pm.check_is_deterministic(
            learned_matrix,
            learned_states,
            alphabet,
        )
        == []
    )

    assert tracking["initial_states"] == len(states)
    assert tracking["final_states"] == len(learned_states)


def test_pattern_mining_pipeline_returns_valid_pdfa_higuera():
    sequences = [
        "A",
        "A",
        "AB",
        "AB",
        "B",
    ]

    alphabet = pm.get_alphabet(sequences)

    transition_matrix = pm.get_transition_matrix(
        sequences,
        alphabet,
    )

    states = pm.get_initial_states(sequences)

    learned_matrix, learned_states, tracking = pm.alergia(
        transition_matrix,
        states,
        alphabet,
        alpha=0.2,
        method="Higuera",
    )

    probability_matrix = pm.probability_transition_matrix(
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
        pm.check_is_deterministic(
            learned_matrix,
            learned_states,
            alphabet,
        )
        == []
    )

    assert tracking["initial_states"] == len(states)
    assert tracking["final_states"] == len(learned_states)
