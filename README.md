# pattern-mining

This is code for running the grammatical inference algorithm ALERGIA for sequential pattern mining. It includes evaluation metrics and methods, as well as code to visualise prefix-tree acceptors (PTAs), deterministic finitie automata (PDAs), and their probabilistic counterparts (PPTAs and PDFAs respectively).

## Input data

Given an initial list of sequences  [ "0", "0", "0", "0", "0", "0", "0", "0", "01", "01", "01", "01", "01", "01", "01", "10", "10", "10", "10", "10", "11", "11", "11", "11", "11", "12", "12", "1", "1", "1"], we can derive the following PPTA from this data:

![example_ppta](https://github.com/MHowells/pattern_mining/blob/main/figs/example_pta.svg)

For this initial PTA, with three actions, or an alphabet, ['0', '1', '2'] and 7 states, [0, 1, 2, 3, 4, 5, 6], the example input would be:

```python
states = ["*", 0, 1, 2, 3, 4, 5, 6]
alphabet = ['0', '1', '2']
pathway_matrix = np.array([
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

The `states` includes an extra state `"*"` representing the starting state. The `pathway_matrix` is a 3D matrix with the first dimension representing the action, the second dimension representing the state, and the third dimension representing the next state. For example, `pathway_matrix[0, 1, 4]` is the number of times that action '0' was taken from state 0 to state 3.

Note that we have assumed that you go from the starting state `"*"` to state `0` by using action `'0'`. This is not true, but has no effect on the results of the algorithm.

There are two approaches to the ALERGIA algorithm implemented in this codebase, dictated by the `method` parameter in the `alergia` function. The first is the `Carrasco` approach, found in the original paper for the ALERGIA algorithmby Carrasco and Oncina (1994)[[1]](#1). The second is the `Higuera` approach, that uses a red-blue framework to solve the algorithm, as outlined in Higuera (2010)[[2]](#2). By default, it is set to `Carrasco`.

## Running tests

```bash
$ python -m pytest tests.py
```

## References
<a id="1">[1]</a> 
Carrasco, R.C. and Oncina, J., (1994).
Learning stochastic regular grammars by means of a state merging method.
International Colloquium on Grammatical Inference, (pp. 139-152).

<a id="2">[2]</a> 
De la Higuera, C., (2010). 
Grammatical inference: learning automata and grammars. 
Cambridge University Press.