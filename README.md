# pdfa-learning

This repository contains code for running the grammatical inference algorithm 
ALERGIA for sequential pattern mining. It includes functions for constructing 
prefix-tree acceptors (PTAs) and probabilistic prefix-tree acceptors (PPTAs), 
learning deterministic finite automata (DFAs) and probabilistic deterministic 
finite automata (PDFAs), estimating pattern and sequence probabilities, and 
visualising the resulting automata. 

There are two approaches to the ALERGIA algorithm implemented in this codebase. 
The first is the `carrasco` approach, found in the original paper for the 
ALERGIA algorithm by Carrasco and Oncina (1994) [[1]](#1). The second is the 
`de_la_higuera` approach, that uses a red-blue framework to solve the algorithm, as 
outlined in de la Higuera (2010) [[2]](#2). The default method is `carrasco`, although 
the `de_la_higuera` approach is cheaper to compute.

## Installing Dependencies

The codebase has been tested with Python 3.9.22, with the requirements 
specified in `requirements.txt`.

To create a virtual environment:

    $ python -m venv env

To start using the new virtual environment:

    $ source env/bin/activate

To install the dependencies:

    $ python -m pip install -r requirements.txt

Alternatively, you can use conda to create a new environment with the required
dependencies by running the following command:

    $ conda env create --file environment.yml

Note that the visualisation functions require Graphviz to be installed on your
system. 

## Input data

The initial data is provided as a list of sequences. Each character in a 
sequence represents a symbol from the alphabet.

For example, given the following list of sequences:

```python
sequences = [
    "0", "0", "0", "0", "0", "0", "0", "0", 
    "01", "01", "01", "01", "01", "01", "01", 
    "10", "10", "10", "10", "10", 
    "11", "11", "11", "11", "11", 
    "12", "12", 
    "1", "1", "1"
]
```

We can derive the following PTA from this data:

![example_ppta](https://github.com/MHowells/pdfa_learning/blob/main/figs/example_pta.svg)

The alphabet, states, and transition-count matrix for this PTA can be
constructed directly from the sequences:

```python
import pdfa_learning as pl

alphabet = pl.get_alphabet(sequences) 
states = pl.get_initial_states(sequences) 

transition_matrix = pl.get_transition_matrix(
    sequences, 
    alphabet, 
)
```

For this example, the alphabet (or actions) is:

```python 
["0", "1", "2"] 
```

The state list contains seven prefix states along with the artificial starting
state `"*"`:

```python
["*", 0, 1, 2, 3, 4, 5, 6]
```

The `transition_matrix` is a three-dimensional NumPy array. Its first dimension 
represents the alphabet symbol, its second dimension represents the current 
state, and its third dimension represents the destination state.

For example:

```python
transition_matrix[0, 1, 2]
```

contains the number of transitions from state `0` to state `1` using 
symbol `"0"`.

Note that the transition from the starting state `"*"` is stored in the first
alphabet layer of the matrix, here, `'0'`. This is not true and does not 
represent a genuine emitted symbol, but has no effect on the results of the 
algorithm.

The full transition-count matrix for this example is:

```python
np.array([
    [
        [0, 30, 0, 0, 0, 0, 0, 0],
        [0, 0, 15, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 5, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 15, 0, 0, 0],
        [0, 0, 0, 7, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 5, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
])
```

## Running ALERGIA

The transition-count matrix can be passed to `alergia()` to learn a smaller 
DFA:

```python
learned_matrix, learned_states, tracking = pl.alergia(
    transition_matrix,
    states,
    alphabet,
    alpha=0.2,
    method="carrasco",
)
```

The `method` parameter can be either:

```python
method="carrasco"
```

or:

```python
method="de_la_higuera"
```

The function returns:

- `learned_matrix`, containing the transition counts of the learned automaton;
- `learned_states`, containing the remaining state identifiers;
- `tracking`, containing information about the attempted, successful, and 
failed merges.

### Choosing the alpha Parameter

The `alpha` parameter controls the tolerance used when deciding whether 
two states are statistically compatible and may therefore be merged. 

In this implementation: 
- smaller `alpha` values produce a wider compatibility bound and generally 
allow more state merges; 
- larger `alpha` values produce a narrower compatibility bound and generally 
result in fewer state merges. 

Consequently, smaller values tend to produce more compact, generalised 
automata, whereas larger values tend to preserve more of the structure 
present in the original prefix tree.

The appropriate value of alpha is application dependent and may be selected 
by comparing the resulting automata using validation data or other 
model selection criteria. Users may wish to use some of the evaluation 
functions contained in `evaluation.py` for this purpose.

### Controlling Print Output

The amount of information printed while the algorithm runs can be controlled 
using the `output_level` parameter:

```python
output_level="Suppressed"
output_level="Truncated"
output_level="Full"
```

- `"Suppressed"` prints no progress information and is the default. 
- `"Truncated"` prints the main state comparisons and merge outcomes. 
- `"Full"` additionally prints iteration numbers and details of recursive merges.

## Probability Deterministic Finite Automata (PDFA)

The learned transition-count matrix (DFA) can be converted into a probability 
transition matrix, or PDFA:

```python
probability_matrix = pl.probability_transition_matrix(
    learned_matrix,
    learned_states,
    alphabet,
)
```

The probability matrix can then be used by the pattern and sequence probability 
functions contained in `pdfa_learning.py`.

For example:

```python
pattern_probability = pl.probability_estimate_of_pattern(
    probability_matrix,
    pattern="01",
    alphabet=alphabet,
)

sequence_probability = pl.probability_estimate_of_exact_sequence(
    probability_matrix,
    sequence="01",
    alphabet=alphabet,
)
```

## Running tests

You can run the complete test suite using the following command:

```bash
$ python -m pytest
```

Run the tests with statement and branch coverage using:

```bash
python -m pytest \
    --cov=pdfa_learning \
    --cov-branch \
    --cov-report=term-missing
```

## Author ORCID

- Matthew Howells: [0000-0002-3931-7027](https://orcid.org/0000-0002-3931-7027)
- Paul Harper: [0000-0001-7894-4907](https://orcid.org/0000-0001-7894-4907)
- Daniel Gartner: [0000-0003-4361-8559](https://orcid.org/0000-0003-4361-8559)
- Geraint Palmer: [0000-0001-7865-6964](https://orcid.org/0000-0001-7865-6964)

## Funding 

This code is funded by an Engineering and Physical Sciences Research Council 
(EPSRC) Enhanced CASE PhD Studentship with Cardiff and Vale University Health 
Board as the project partner (Project reference: 2601327, in relation to 
EP/T517951/1).

## References
<a id="1">[1]</a> 
Carrasco, R.C. and Oncina, J., (1994).
Learning stochastic regular grammars by means of a state merging method.
International Colloquium on Grammatical Inference, (pp. 139-152).

<a id="2">[2]</a> 
De la Higuera, C., (2010). 
Grammatical inference: learning automata and grammars. 
Cambridge University Press.