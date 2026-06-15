"""
Recreating the example from Carrasco and Oncina (1994) [1] using the 
pdfa-learning package.

This script demonstrates how to recreate the results from the example in
Carrasco and Oncina (1994) [1] using the pdfa-learning package. It constructs 
a prefix tree acceptor from observed sequences, learns a deterministic
finite automaton using ALERGIA, and converts the learned transition-count
matrix into a probability transition matrix.

References 
---------- 
.. [1] Carrasco, R. C. and Oncina, J. (1994). "Learning Stochastic 
Regular Grammars by Means of a State Merging Method." In *Grammatical 
Inference and Applications*, pp. 139–152. 
"""

import pdfa_learning as pl


sequences = [
    "110",
    "",
    "",
    "",
    "0",
    "",
    "00",
    "00",
    "",
    "",
    "",
    "10110",
    "",
    "",
    "100",
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
    alpha=0.8,
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