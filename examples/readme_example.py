"""
Basic usage example for the pdfa-learning package.

This script demonstrates how to construct a prefix tree acceptor from 
observed sequences, learn a deterministic finite automaton 
using ALERGIA, and convert the learned transition-count matrix into a 
probability transition matrix.
"""

import pdfa_learning as pl


sequences = [
    "0", "0", "0", "0", "0", "0", "0", "0", 
    "01", "01", "01", "01", "01", "01", "01", 
    "10", "10", "10", "10", "10", 
    "11", "11", "11", "11", "11", 
    "12", "12", 
    "1", "1", "1"
]

alphabet = pl.get_alphabet(sequences)
states = pl.get_initial_states(sequences) 

pathway_matrix = pl.get_transition_matrix(
    sequences, 
    alphabet, 
)

learned_matrix, learned_states, tracking = pl.alergia(
    pathway_matrix,
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

print("Alphabet:", alphabet)
print("Initial states:", states)
print("Learned states:", learned_states)
print("Tracking:", tracking)
print("Probability matrix shape:", probability_matrix.shape)
print("PDFA:", probability_matrix)