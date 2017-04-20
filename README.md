# Simultaneous Linear Solver Interface

For solving systems like

```
  A1 x + B1 y = a1 
  A2 x + B2 y = a2
```

where x, y, a1, a2 are vectors and A1, A2, B1, B2 matrices.
Takes any number of vectors, matrices, as long as the system of equations are consistent.
Use solve_sparse_system for sparse systems.

See __name__ == "__main__" part of slsi.py for examples.
