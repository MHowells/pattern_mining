"""
Recreating the examples from Arnolds and Gartner (2018) [1] using the 
pdfa-learning package.

This script demonstrates how to recreate the results from the examples in
Arnolds and Gartner (2018) [1] using the pdfa-learning package. It constructs 
a prefix tree acceptor from observed sequences, learns a deterministic
finite automaton using ALERGIA, and converts the learned transition-count
matrix into a probability transition matrix.

References 
---------- 
.. [1] Arnolds, I. V., & Gartner, D. (2018). Improving hospital layout 
planning through clinical pathway mining. Annals of operations research, 
263(1), 453-477.
"""

import pdfa_learning as pl


sequences = [
    "AB", 
    "ABA", 
    "ABB", 
    "ABCA", 
    "AC", 
    "ACC", 
    "BA", 
    "BAA", 
    "BC", 
    "BCA"
]

alphabet = pl.get_alphabet(sequences)
states = pl.get_initial_states(sequences) 

pathway_matrix = pl.get_transition_matrix(
    sequences, 
    alphabet, 
)

learned_matrix_one, learned_states_one, tracking_one = pl.alergia(
    pathway_matrix,
    states,
    alphabet,
    alpha=0.2,
    method="de_la_higuera",
)

probability_matrix_one = pl.probability_transition_matrix(
    learned_matrix_one,
    learned_states_one,
    alphabet,
)

print("Alphabet:", alphabet)
print("Initial states:", states)
print("Learned states for alpha=0.2:", learned_states_one)
print("Tracking for alpha=0.2:", tracking_one)
print("Probability matrix for alpha=0.2 shape:", probability_matrix_one.shape)
print("PDFA for alpha=0.2:", probability_matrix_one)

learned_matrix_two, learned_states_two, tracking_two = pl.alergia(
    pathway_matrix,
    states,
    alphabet,
    alpha=0.8,
    method="de_la_higuera",
)

probability_matrix_two = pl.probability_transition_matrix(
    learned_matrix_two,
    learned_states_two,
    alphabet,
)

print("Alphabet:", alphabet)
print("Initial states:", states)
print("Learned states for alpha=0.8:", learned_states_two)
print("Tracking for alpha=0.8:", tracking_two)
print("Probability matrix for alpha=0.8 shape:", probability_matrix_two.shape)
print("PDFA for alpha=0.8:", probability_matrix_two)