# adjoint-3D-linear-wave-resistance-python
a 3D BEM solver, optimization, and panel modification scheme for wave resistance using the generalized, linearized free surface conditions and Rankine sources


* starting from a python version of my old 3d linear wave resistance code


# Converting Forward Physics to a full on Adjoint Solver

## Switching to LU and Reusing It for the Adjoint (Discrete Adjoint Setup)

This section documents the first “adjoint-ready” refactor: switching the forward solve to an LU factorization that we can reuse to solve both the **state** system and the **adjoint** system efficiently.

---

### Problem Statement

Our forward (state) solve is a dense linear system coming from the Rankine-source BEM discretization:


\[
A\,\sigma = b
\]




- \(A \in \mathbb{R}^{N\times N}\) is the influence matrix assembled from the hull and free-surface boundary conditions.
- \(\sigma \in \mathbb{R}^{N}\) are the source strengths (unknowns).
- \(b \in \mathbb{R}^{N}\) is the RHS enforcing hull no-penetration and the free-surface condition.

The matrix is **non-symmetric**, so \(A^T\) is not interchangeable with \(A\).

---

### Why LU Factorization?

A dense solve (e.g. `scipy.linalg.solve`) works, but for adjoints and optimization we will solve *multiple* systems with the same matrix $\begin{bmatrix} A \end{bmatrix}$. \[A\]


LU factorization gives us:

- One-time factorization cost: `lu_factor(A)`
- Fast repeated solves:
  - forward: solve \(A\,\sigma = b\)
  - adjoint: solve \(A^T\,\lambda = \frac{\partial J}{\partial \sigma}\)

---

### LU Forward Solve (State Solve)

Use SciPy’s LU utilities:

```python
from scipy.linalg import lu_factor, lu_solve

lu, piv = lu_factor(A)          # factorize A once
sigma   = lu_solve((lu, piv), b)  # solves A * sigma = b  (trans=0 default)

```


### LU Forward Solve (State Solve)
## requirements
* numpy, scipy, matplotlib
* numba

Mathpix Markdown to render the math in this readme in Visual Studio Code