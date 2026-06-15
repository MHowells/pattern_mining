"""
Basic usage example for the pattern-mining package.

This script demonstrates how to construct a prefix tree acceptor from 
observed sequences, learn a deterministic finite automaton 
using ALERGIA, and convert the learned transition-count matrix into a 
probability transition matrix.
"""

import pattern_mining as pm


sequences = [
    "0", "0", "0", "0", "0", "0", "0", "0", 
    "01", "01", "01", "01", "01", "01", "01", 
    "10", "10", "10", "10", "10", 
    "11", "11", "11", "11", "11", 
    "12", "12", 
    "1", "1", "1"
]

alphabet = pm.get_alphabet(sequences)
states = pm.get_initial_states(sequences) 

pathway_matrix = pm.get_transition_matrix(
    sequences, 
    alphabet, 
)

learned_matrix, learned_states, tracking = pm.alergia(
    pathway_matrix,
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

print("Alphabet:", alphabet)
print("Initial states:", states)
print("Learned states:", learned_states)
print("Tracking:", tracking)
print("Probability matrix shape:", probability_matrix.shape)
print("PDFA:", probability_matrix)