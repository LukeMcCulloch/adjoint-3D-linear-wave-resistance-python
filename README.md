# adjoint-3D-linear-wave-resistance-python
a 3D BEM solver, optimization, and panel modification scheme for wave resistance using the generalized, linearized free surface conditions and Rankine sources


* starting from a python version of my old 3d linear wave resistance code


## Converting Forward Physics to a full on Adjoint Solver

#### 1.) Switching to LU and Reusing It for the Adjoint (Discrete Adjoint Setup)

This section documents the first “adjoint-ready” refactor: switching the forward solve to an LU factorization that we can reuse to solve both the **state** system and the **adjoint** system efficiently.

---

#### 2.) Problem Statement

Our forward (state) solve is a dense linear system coming from the Rankine-source BEM discretization:


$$
A\,\sigma = b
$$



- $A \in \mathbb{R}^{N\times N}$ is the influence matrix assembled from the hull and free-surface boundary conditions.
- $\sigma \in \mathbb{R}^{N}$ are the source strengths (unknowns).
- $b \in \mathbb{R}^{N}$ is the RHS enforcing hull no-penetration and the free-surface condition.

The matrix is **non-symmetric**, so $A^T$ is not interchangeable with $A$.

---

#### 3.) Why LU Factorization?

A dense solve (e.g. `scipy.linalg.solve`) works, but for adjoints and optimization we will solve *multiple* systems with the same matrix $\begin{bmatrix} A \end{bmatrix}$. $$A$$


LU factorization gives us:

- One-time factorization cost: `lu_factor(A)`
- Fast repeated solves:
  - forward: solve $A\,\sigma = b$
  - adjoint: solve $A^T\,\lambda = \frac{\partial J}{\partial \sigma}$

---

#### 4.) LU Forward Solve (State Solve)

For this pass, we use SciPy’s LU utilities:

```python
from scipy.linalg import lu_factor, lu_solve

lu, piv = lu_factor(A)          # factorize A once
sigma   = lu_solve((lu, piv), b)  # solves A * sigma = b  (trans=0 default)

```


## Objective and Adjoint Sensitivities for Wave Resistance (Discrete Formulation)

This section formulates the **objective** we are optimizing (wave resistance coefficient `cw`) and derives the first adjoint ingredient: the gradient of the objective with respect to the state vector of source strengths, $dJ/d\sigma$.

From above we have the forward solve being the dense, non-symmetric linear system

$$
A\,\sigma = b,
$$

where:
- $A \in \mathbb{R}^{N\times N}$ is the influence matrix assembled from the hull and free-surface boundary conditions,
- $\sigma \in \mathbb{R}^{N}$ is the vector of source strengths (unknowns),
- $b \in \mathbb{R}^{N}$ is the right-hand side vector enforcing the hull and free-surface boundary conditions.

---




## Objective definition in our discretization



We store the induced velocity tensor:

- `vel[i, j, m]` = the $m$-th component of the velocity at fieldpoint/panel $i$ induced by a **unit** source strength on panel $j$.

We compute:

$$
U = v_\infty = \texttt{vinf}[0],
\qquad
U^2 = U2.
$$

### Total velocity at each panel
* Given $\sigma$, the total velocity on each panel is
$$
\mathbf{v}_i
= -\mathbf{v}_\infty + \sum_{j=1}^N \sigma_j\,\mathbf{vel}_{ij}.
$$

<!-- (your NumPy: `vtotal = -vinf + einsum("ijm,j->im", vel, sigma)`) -->

```python
vtotal = -vinf[None, :] + np.einsum("ijm,j->im", vel, sigma)

```


### Pressure Coefficient on the Hull Panels

Let $U = v_{\infty,x}$ and $U^2 = U2$.  
On hull panels only ($i = 1,\dots,n_{\text{panels}}$), we compute

$$
cp_i = 1 - \frac{\|\mathbf{v}_i\|^2}{U^2}.
$$


### Force and Objective: Wave Resistance Coefficient

Define the wetted hull set (matching the code):

On hull panels $i = 1..n_{\mathrm{panels}}$:

$$
c_{p,i} = 1 - \frac{\lVert \mathbf{v}_i \rVert^2}{U^2}.
$$


### Pressure-like term (from our code):

$$
p_i = \frac{1}{2}\,\rho\,U^2\,c_{p,i} \;-\; \rho\,g\,z_i.
$$


### And force:

$$
\mathbf{F} = \sum_{i \in \mathrm{wetted\ hull}} A_i\, p_i\, \mathbf{n}_i.
$$

Define the wetted set (of panels)
$$
W = \left\{\, i \le n_{\mathrm{panels}} \;\middle|\; z_i < 0 \,\right\},
$$
and wetted area
$$
S = \sum_{i \in W} A_i.
$$

### Objective
<!-- Finally: -->

$$
J = c_w = -\frac{F_x}{\tfrac{1}{2}\,\rho_{\mathrm{ref}}\,U^2\,S}.
$$


That’s the discrete objective we differentiate.



## First adjoint target: compute $\partial J / \partial \sigma$

We’ll compute

$$
\mathbf{g} \equiv \frac{\partial J}{\partial \boldsymbol{\sigma}} \in \mathbb{R}^N
$$

so we can solve the adjoint system

$$
A^T \boldsymbol{\lambda} = \mathbf{g}.
$$

### Key simplification (important)

Look at the pressure term:

$$
p_i
= \frac{1}{2}\rho U^2\left(1 - \frac{\lVert \mathbf{v}_i \rVert^2}{U^2}\right) - \rho g z_i
= \frac{1}{2}\rho U^2 - \frac{1}{2}\rho \lVert \mathbf{v}_i \rVert^2 - \rho g z_i.
$$

Partial of pressure with respect to $\mathbf{v}_i$:

$$
\frac{\partial p_i}{\partial \mathbf{v}_i} = -\rho\,\mathbf{v}_i.
$$

Then get the partial with respect to $\mathbf{v}_i$ for the $x$-component of force:

Substitue $p$ into to the force eqation, and differentiate as above:

$$
F_x = \sum_{i \in W} A_i\, p_i\, n_{i,x}
\;\;\Rightarrow\;\;
\frac{\partial F_x}{\partial \mathbf{v}_i}
= A_i\,n_{i,x}\,\frac{\partial p_i}{\partial \mathbf{v}_i}
= -\rho\,A_i\,n_{i,x}\,\mathbf{v}_i.
$$

And remember $J$ is $c_w$ and we define shorthand $C$ here as:

$$
J = -\frac{F_x}{C},
\qquad
C = \frac{1}{2}\rho_{\mathrm{ref}} U^2 S,
$$

Upon differentiation with respec to $v_i$ we get

$$
\frac{\partial J}{\partial \mathbf{v}_i}
= -\frac{1}{C}\frac{\partial F_x}{\partial \mathbf{v}_i}
= \frac{\rho\,A_i\,n_{i,x}}{C}\,\mathbf{v}_i,
\qquad i \in W,
$$

and zero otherwise.

Finally, because

$$
\mathbf{v}_i
= -\mathbf{v}_\infty + \sum_j \sigma_j\,\mathbf{vel}_{ij}
\;\;\Rightarrow\;\;
\frac{\partial \mathbf{v}_i}{\partial \sigma_j} = \mathbf{vel}_{ij},
$$

we have

$$
\frac{\partial J}{\partial \sigma_j}
= \sum_{i=1}^N \left(\frac{\partial J}{\partial \mathbf{v}_i}\cdot \mathbf{vel}_{ij}\right).
$$

That’s the vector $\mathbf{g}$ we need.


























## python package requirements
* numpy, scipy, matplotlib
* numba
## documentation requirements
* Optional: Mathpix Markdown ? to render the more latex math 
    * Here we adopt the github subset for maximal compatibility
    * so you do not need it

