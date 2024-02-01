# pattern-mining

Code for running sequential pattern mining algorithm ALERGIA.

## Input data

For an initial PPTA that looks like (instert picture) with three actions or an alphabet [0, 1, 2] and 7 states, [0, 1, 2, 3, 4, 5, 6], the example input would be:

```python
states = ["S", 0, 1, 2, 3, 4, 5, 6]
alphabet = [0, 1, 2]
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

The `states` includes an extra state `"S"` representing the starting state. The `pathway_matrix` is a 3D matrix with the first dimension representing the action, the second dimension representing the state, and the third dimension representing the next state. For example, `pathway_matrix[0, 1, 4]` is the number of times that action 0 was taken from state 0 to state 3.

Note that we have assumed that you go from the starting state `"S"` to state `0` by using action `0`. This is not true, but has no effect on the maths. 

## Running tests

```bash
$ python -m pytest tests.py
```
