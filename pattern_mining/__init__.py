"""
Functions for constructing probabilistic prefix-tree acceptors and
learning probabilistic deterministic finite automata using ALERGIA.
"""

from ._validation import (
    _validate_alpha,
    _validate_alphabet,
    _validate_sequences,
    _validate_states_for_merging,
    _validate_transition_matrix,
)

from .ppta import (
    get_alphabet,
    get_initial_states,
    get_state_paths,
    get_transition_matrix,
)

from .state_statistics import (
    get_endpoint,
    get_n,
    get_pi,
    get_pi_endpoint,
)

from .state_merging import (
    _merge_two_states,
    _recursive_merge_two_states,
    check_is_deterministic,
    get_blue_states,
    get_pairs_to_check,
    hoeffding_bound,
    merge_two_states,
    recursive_merge_two_states,
)

from .alergia import alergia

from .probabilities import (
    probability_estimate_of_exact_sequence,
    probability_estimate_of_pattern,
    probability_estimate_of_symbol,
    probability_sequence_contains_digram,
    probability_sequence_contains_letter_at_distance_theta,
    probability_to_encounter_a_pattern_at_a_distance_theta,
    probability_transition_matrix,
    proportion_constraint,
    string_enumerator,
    string_probabilities,
)

from .visualisation import network_visualisation