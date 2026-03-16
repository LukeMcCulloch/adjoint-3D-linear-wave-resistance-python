# adjoint-3D-linear-wave-resistance-python
a 3D BEM solver, optimization, and panel modification scheme for wave resistance using the generalized, linearized free surface conditions and Rankine sources

## warning: this repo is under periodic heavy development


# Adjoint derivation

Free variables of the optimization:
* $sigma = $ source strengths (the 'u' of our physics)
* $m = $ geometry: perhaps parameterized in some way as shape variables
* $\lambda = $ the Lagrange multipliers, aka the adjoint variables 
* $J = $ our objective
* $R = $ the resiudal of our physics

$$
\mathcal{L} = J\left( m, \sigma\left(m\right) \right) + \lambda\left(R\left( m, \sigma \left(m\right)  \right) \right)
$$

* we note that geometry $m$ is not a function of $\sigma$  but $\sigma$ is a function of m. $ \sigma = \sigma\left(m\right)$ 

With nothing but the above in hand, let's derive everything we need to do physics driven optimization of our geometry.

## First Order Necessary Condition (F.O.N.C.) 
The system gradient must be equal to 0 at the minimum.  (at an extrema or critical point)

<!-- $ \frac{\partial J}{\partial m}  \frac{\partial m}{\partial \sigma} $ -->
    

$\frac{\partial \mathcal{L}}{\partial \sigma} = \mathcal{L}_\sigma = \frac{\partial J}{\partial \sigma}   + \lambda \frac{\partial R}{\partial \sigma} \frac{\partial \sigma}{\partial m} = 0$


$\frac{\partial \mathcal{L}}{\partial m} = \mathcal{L}_m = \frac{\partial J}{\partial m}   + \lambda \frac{\partial R}{\partial m} = 0$


$\frac{\partial \mathcal{L}}{\partial \lambda} = \mathcal{L}_\lambda = R = 0$

## Solution Process

* Solve for the $\sigma$ that drivec the residual $R$ to 0.

    $R = A\sigma -b = 0$
    * note that for the problem we solve in this repo, $A = A\left( m \right)$

* Now we note the adjoint trick:




## Lambda gradient trick, (adoint trick, part 1)

$\mathcal{L}_\sigma = \frac{\partial J}{\partial \sigma}   + \lambda \frac{\partial R}{\partial \sigma}  = 0$

so


$\frac{\partial J}{\partial \sigma}   = - \lambda \frac{\partial R}{\partial \sigma}  $

or, in linear system form:


$$  \frac{\partial R}{\partial \sigma} \lambda  = - \frac{\partial J}{\partial \sigma} $$

$$  \lambda  = - \left[\frac{\partial R}{\partial \sigma}\right] ^{-1} \frac{\partial J}{\partial \sigma} $$

we will come back to this.  Call this substitution 2.




## The shape gradients $\mathcal{L}_m$, (adjoint trick, part 2)

expanding the implicit gradient with respect to m, we have:

$$ \mathcal{L}_m = \frac{\partial J}{\partial m}   +    \frac{\partial J}{\partial \sigma}  \frac{\partial \sigma}{\partial m} +  \lambda \left( \frac{\partial R}{\partial m} +\frac{\partial R}{\partial \sigma} \frac{\partial \sigma}{\partial m} \right) = 0$$

* the $\frac{\partial \sigma}{\partial m}$ are very hard to obtain by traditional methods.

* note first that we have the fact that the residual of our Physics is going to be 0, therefore:

$$ \left( \frac{\partial R}{\partial m} +\frac{\partial R}{\partial \sigma} \frac{\partial \sigma}{\partial m} \right) = 0 $$

And thus we note:

$$  \frac{\partial R}{\partial m} = -\frac{\partial R}{\partial \sigma} \frac{\partial \sigma}{\partial m}  $$

or

$$  \frac{\partial R}{\partial \sigma} \frac{\partial \sigma}{\partial m} = - \frac{\partial R}{\partial m}  $$

$$  \frac{\partial \sigma}{\partial m} = - \left[ \frac{\partial R}{\partial \sigma}  \right]^{-1} \frac{\partial R}{\partial m}  $$


Call the above, sustitution 1.




## we use the adjoint trick substitutions (adjoint trick, part 3):


$$ \mathcal{L}_m = \frac{\partial J}{\partial m}   +    \frac{\partial J}{\partial \sigma}  \frac{\partial \sigma}{\partial m} +  \lambda \left( \frac{\partial R}{\partial m} +\frac{\partial R}{\partial \sigma} \frac{\partial \sigma}{\partial m} \right) = 0$$


sub in the adjoint definition $\frac{\partial R}{\partial \sigma} \lambda  = - \frac{\partial J}{\partial \sigma}$

 $ \lambda  = - \left[ \frac{\partial R}{\partial \sigma} \right]^{-1} \frac{\partial J}{\partial \sigma}$ is easily obtainable with a single linear solve.  Since in our problem $R_\sigma = A$ we just solve the linear system for $\lambda$:

 $$\frac{\partial R}{\partial \sigma} \lambda = - \frac{\partial J}{\partial \sigma} $$


And we also saw above we have the residual relation $  \frac{\partial \sigma}{\partial m} = - \left[ \frac{\partial R}{\partial \sigma}  \right]^{-1} \frac{\partial R}{\partial m}  $

so with all that, this:

$$\frac{\partial J}{\partial m}   +    \frac{\partial J}{\partial \sigma}  \frac{\partial \sigma}{\partial m} +  \lambda \left( \frac{\partial R}{\partial m} +\frac{\partial R}{\partial \sigma} \frac{\partial \sigma}{\partial m} \right) = 0$$

#### Using substitution 1, the above becomes this:

$$\frac{\partial J}{\partial m}   -    \frac{\partial J}{\partial \sigma}  \left[ \frac{\partial R}{\partial \sigma}  \right]^{-1} \frac{\partial R}{\partial m}  +                                                          \lambda \left( \frac{\partial R}{\partial m}   -   \frac{\partial R}{\partial \sigma} \left[ \frac{\partial R}{\partial \sigma}  \right]^{-1} \frac{\partial R}{\partial m} \right) = 0$$

* Asserting the subsitution is valid for the residual part is tatamount to saying the residual is zero?  In any case we are left with this (which does keep the dependence on the residual ;):

$$\frac{\partial J}{\partial m}   -    \frac{\partial J}{\partial \sigma}  \left[ \frac{\partial R}{\partial \sigma}  \right]^{-1} \frac{\partial R}{\partial m}   = 0$$


#### Using substitution 2, we get:


$$\frac{\partial J}{\partial m}   -    \lambda^T\frac{\partial R}{\partial m}   = 0$$





* note that I played fast and loose with the transposes.  I am pretty sure I left some out.  In my code, I keep $\lambda$'s as a column vector on the right $\lambda => \lambda^{T}$, eliminating much of these shinanigans.


* writing this out for our problem, we see:

   $$
   \frac{dJ}{dm} = J_m + \lambda^T(b_m - A_m\sigma) = 0
   $$



# Forward Solver for $R$: the initial physics solver development effort

<!-- -------------------------------------------------------------- 
-->

Steady Linear Wave-Resistance Panel Solver Notes (Rankine Sources + Discretized Free Surface)

These notes document the **discrete** solver we are implementing (and matching to the original Fortran), including:
- unknowns (source strengths)
- influence kernels (panel-induced velocity and $\phi_{xx}$)
- boundary conditions (hull no-penetration + generalized linear free-surface condition)
- assembly of the linear system
- postprocessing: velocities, pressure coefficient, wave elevation, forces, and $c_w$

Everything is written in the same conventions used by our code.

---

## 0) Governing PDE and Decomposition

We solve **Laplace's equation** for a velocity potential $\phi$ in an inviscid, incompressible flow:

$$
\nabla^2 \phi = 0
$$

We use a **source panel method**. The potential is represented as a superposition of constant-strength source distributions on:
- the **hull** (wetted body surface)
- a discretized patch of the **free surface** (planar patch, but sources placed above $z=0$ in our setup)

A uniform free-stream velocity is
$$
\mathbf{v}_\infty = (U,0,0),\qquad U = Fr\,\sqrt{gL}
$$
with gravity $g$ and reference length $L$ (here $L=1$).

The **total velocity** is
$$
\mathbf{v} = \nabla \phi \;-\; \mathbf{v}_\infty
$$
(our code uses the convention `vtotal = -vinf + induced`).

---

## 1) Geometry and Panels

We have:
- points: `points` of shape $(3,n_{\text{points}})$ with rows $(x,y,z)$
- panels: `panels` of shape $(4,N)$, each panel is a quadrilateral given by 4 vertex indices
- $N = N_h + N_{fs}$, where:
  - $N_h$ = number of hull panels
  - $N_{fs}$ = number of free-surface panels

For each panel $j$ we compute:
- `center[j]` : collocation point (panel centroid) in $\mathbb{R}^3$
- `area[j]` : panel area $A_j$
- `coordsys[j]` : a $(3\times 3)$ orthonormal basis whose columns are:
  - $\mathbf{t}_{1,j}$ : tangent 1
  - $\mathbf{t}_{2,j}$ : tangent 2
  - $\mathbf{n}_j$ : panel normal
- `cornerslocal[j]` : $(2\times 4)$ corner coordinates in the local $(\xi,\eta)$ plane

> Normal convention (important): The corners are ordered so that the computed normal points **inside the hull** (as in the original Fortran).

---

## 2) Influence of a Source Panel

Let panel $j$ have unit source strength. The induced **velocity** at a field point $\mathbf{p}$ is:

$$
\mathbf{v}_{j}(\mathbf{p}) = \nabla \phi_j(\mathbf{p})
$$

In code we compute this via `hs_influence(fieldpoint, center_j, coordsys_j, cornerslocal_j)`.

We also compute the global Hessian (second derivatives) at $\mathbf{p}$:
$$
H_j(\mathbf{p}) = \nabla\nabla \phi_j(\mathbf{p})
$$
and in particular we need $\phi_{xx}$ for the free-surface condition.

In code this is `phixx_influence(...)`, and we extract `H[0,0]`.

### Stored influence tensor

We store:
- `vel[i,j,:]` = induced velocity at collocation/field index $i$ due to unit source on panel $j$
- shape: `vel` is $(N,N,3)$

So once $\sigma$ is known:
$$
\mathbf{v}_i = -\mathbf{v}_\infty + \sum_{j=1}^{N}\sigma_j\,\mathbf{vel}_{ij}
$$

In NumPy:
```python
vtotal = -vinf[None,:] + np.einsum("ijm,j->im", vel, sigma)
```



<!-- -----------------------------------------------------------------
 -->

## 3) Symmetry About the Centerplane (Half-Ship)

We enforce symmetry about the $x$-$z$ plane (centerplane) by using an image point:

$$
p' = (x,\,-y,\,z)
$$

For a velocity vector $\mathbf{v}=(v_x,v_y,v_z)$ at the mirrored point, the symmetric half-ship combination is:

- $v_x$ is even in $y$
- $v_y$ is odd in $y$
- $v_z$ is even in $y$

So the combined induced velocity used in the system is:

$$
v_x \leftarrow v_x + v_x',\qquad
v_y \leftarrow v_y - v_y',\qquad
v_z \leftarrow v_z + v_z'
$$

For the Hessian, we only use $\phi_{xx}$; it is even in $y$, so:

$$
\phi_{xx} \leftarrow \phi_{xx} + \phi_{xx}'.
$$

## 4) Boundary Conditions (Discrete Residual)

We assemble a linear system:

$$
A\,\sigma = b
$$

### 4.1 Hull no-penetration (Body BC)

At each hull collocation point $i\in\{1,\dots,N_h\}$:

$$
\mathbf{n}_i \cdot \mathbf{v}_i = 0
$$

Using $ \mathbf{v}_i = -\mathbf{v}_\infty + \sum_j \sigma_j\,\mathbf{vel}_{ij} $:

$$
\mathbf{n}_i \cdot \left(\sum_{j=1}^{N}\sigma_j\,\mathbf{vel}_{ij}\right) 
    = \mathbf{n}_i \cdot \mathbf{v}_\infty
$$

So:

- matrix row entries:
  $$
  A_{ij} = \mathbf{n}_i \cdot \mathbf{vel}_{ij}
  $$
- RHS:
  $$
  b_i = \mathbf{n}_i \cdot \mathbf{v}_\infty
  $$

#### Self-term (singular diagonal on the hull)

For $i=j$ on the hull, we use the known limiting value for a constant source panel:

$$
\mathbf{vel}_{ii} \approx -\frac{1}{2}\mathbf{n}_i
$$

so

$$
A_{ii} \approx \mathbf{n}_i \cdot \left(-\frac{1}{2}\mathbf{n}_i\right) = -\frac{1}{2}.
$$

(In practice we set `vel[:, i, i] = -0.5*n_i` before dotting.)

### 4.2 Free-surface condition (Generalized linearized FS BC)

We enforce the combined (generalized) linear free-surface boundary condition in steady forward motion:

$$
U^2\,\phi_{xx} + g\,\phi_{z} = 0
\quad\text{at}\quad z=0.
$$

We discretize this at each free-surface collocation point $i\in\{N_h+1,\dots,N\}$.

In our code:

- the free-surface collocation point is adjusted by a small $x$ offset $\Delta x$:
  $$
  x \leftarrow x + \Delta x
  $$
- and set to $z=0$:
  $$
  z \leftarrow 0
  $$

This acts like a stabilizing / radiation bias (Sommerfeld-style behavior).

For each $j$:

- we compute $\phi_{xx}^{(j)}(p_i)$ from the Hessian $H_j(p_i)$:
  $$
  \phi_{xx}^{(j)}(p_i) = [H_j(p_i)]_{xx}
  $$
- we compute $\phi_{z}^{(j)}(p_i)$ from the induced velocity:
  $$
  \phi_{z}^{(j)}(p_i) = [\mathbf{vel}_{ij}]_{z}
  $$

Therefore the free-surface row is:

$$
A_{ij}
=
U^2\,\phi_{xx}^{(j)}(p_i) + g\,\phi_{z}^{(j)}(p_i)
$$

and we use:

$$
b_i = 0.
$$

Free-surface sources are placed above $z=0$ in our geometry setup, so the evaluation at $z=0$ is nonsingular.

## 5) Solving for Source Strengths

Once assembled:

$$
A\,\sigma = b
$$

We solve using a dense linear solve (LU factorization reused for adjoints):

```python
lu, piv = lu_factor(A)
sigma = lu_solve((lu, piv), b)          # solves A*sigma=b
```


## 6) Postprocessing: Velocity Components on Each Panel

### Total velocity at each panel collocation

$$
\mathbf{v}_i
=
-\mathbf{v}_\infty
+
\sum_{j=1}^{N}\sigma_j\,\mathbf{vel}_{ij}
$$

### Normal and tangential components

$$
v_{n,i} = \mathbf{n}_i\cdot \mathbf{v}_i,\qquad
v_{t1,i} = \mathbf{t}_{1,i}\cdot \mathbf{v}_i,\qquad
v_{t2,i} = \mathbf{t}_{2,i}\cdot \mathbf{v}_i.
$$

In code (vectorized):

```python
vn  = np.einsum("ij,ij->i", normals, vtotal)
vt1 = np.einsum("ij,ij->i", t1, vtotal)
vt2 = np.einsum("ij,ij->i", t2, vtotal)
```


For a correct solution: on the hull, $v_{n,i}$ should be near zero (no penetration).

## 7) Pressure Coefficient on the Hull

On hull panels only ($i\le N_h$), we compute a steady Bernoulli pressure coefficient:

$$
c_{p,i}
=
1 - \frac{\lVert \mathbf{v}_i\rVert^2}{\lVert \mathbf{v}_\infty\rVert^2}
=
1 - \frac{\lVert \mathbf{v}_i\rVert^2}{U^2}.
$$

In code:

```python
cp[:npanels] = 1.0 - (np.sum(vtotal[:npanels]**2, axis=1) / U2)
```

## 8) Source-Sink Check (Conservation Diagnostic)

For a closed body in source methods, the total source strength integrated over the body should be near zero. We monitor:

$$
\mathrm{sourcesink} = \sum_{j=1}^{N}\sigma_j\,A_j
$$

In code:

```python
sourcesink = float(np.sum(sigma * area))
```
This is a useful diagnostic (it should be small).

## 9) Wave Elevation on the Free Surface

From linearized theory in steady forward motion, our code uses:

$$
\zeta_i = \frac{U}{g}\left(v_{x,i} + U\right)
\quad\text{for}\quad i\in\{N_h+1,\dots,N\}.
$$

In code:

```python
zeta[row] = (U / gravity) * (vtotal[row, 0] + U)
```
(Here vtotal[row,0] is $v_x$ at the free-surface collocation.)



## 10) Force on the Hull

We compute a pressure-like term:

$$
p_i = \frac{1}{2}\rho U^2\,c_{p,i} - \rho g z_i
$$

and integrate over wetted hull panels.

Wetted set:

$$
W = \{\, i \le N_h \mid z_i < 0 \,\}
$$

Force vector:

$$
\mathbf{F} = \sum_{i\in W} A_i\,p_i\,\mathbf{n}_i.
$$

In code:

```python
for i in range(npanels):
    if center[i,2] < 0.0:
        pressure_term = 0.5*rho*U2*cp[i] - rho*gravity*center[i,2]
        force += area[i] * pressure_term * normals[i]
```
The $x$-component $F_x$ is the wave-resistance force component in our sign convention.


## 11) Wave Resistance Coefficient

We form the wetted reference area:

$$
S = \sum_{i\in W} A_i
$$

Then:

$$
c_w = \frac{-F_x}{\tfrac{1}{2}\rho_{\mathrm{ref}} U^2 S}.
$$

In code:

```python
S = sum(area[:npanels][center[:npanels,2] < 0.0])
cw = -force[0] / (0.5 * rho_ref * U**2 * S)
```

## 12) Summary of Core Computations (Discrete Pipeline)

Given geometry (points + quads):

- Compute per-panel geometry: `center`, `coordsys`, `cornerslocal`, `area`.

- Assemble $A$ and $b$ using:
  - hull BC: $A_{ij} = \mathbf{n}_i\cdot \mathbf{vel}_{ij}$, $b_i=\mathbf{n}_i\cdot \mathbf{v}_\infty$
  - free surface BC: $A_{ij}=U^2\phi_{xx}+g\phi_z$, $b_i=0$
  - symmetry by mirrored fieldpoint contribution
  - special hull diagonal: $\mathbf{vel}_{ii}=-\tfrac12 \mathbf{n}_i$

- Solve $A\,\sigma=b$.

- Compute velocities:
  $$
  \mathbf{v}_i
  =
  -\mathbf{v}_\infty
  +
  \sum_{j=1}^{N}\sigma_j\,\mathbf{vel}_{ij}.
  $$

- Compute:
  - $c_p$ on hull
  - $\zeta$ on free surface
  - force $\mathbf{F}$ on wetted hull
  - $c_w$


## 13) Notes on "Deriving the Entire Solver"

A full continuous derivation of the Rankine-source method typically proceeds by:

- representing $\phi$ via Green's identities (boundary integrals)
- choosing a Rankine (free-space) source kernel for Laplace’s equation
- discretizing the boundary into panels with constant source strength
- collocating boundary conditions to produce a linear system

In our implementation we treat the influence formulas (`hs_influence`, `phixx_influence`) as the analytic evaluation of:

- panel-induced $\nabla\phi$ (velocity)
- panel-induced $\phi_{xx}$

Thus, our “derivation” at the code level is essentially:

- define residual rows from boundary conditions
- evaluate analytic influence coefficients
- assemble and solve

If we later want a continuous derivation section, we can add it, but the above captures the solver as actually implemented.


<!-- -----------------------------------------------------------------
 -->
 
<!--  ----------------------------------------------------------------
-->

 
<!-- -----------------------------------------------------------------
 -->
# Discrete Adjoint for Shape Optimization (Full Derivation)



Starting from a python version of my old 3d linear wave resistance code, what do we do to built it into an adjoint solver and physics driven design optimization tool?

This note re-connects the math to the code. We start from the **discrete** physics
$$
R(m,\sigma)=0
$$
and a scalar objective
$$
J(m,\sigma)
$$
and derive the adjoint equation and the final shape gradient.

Throughout:

- $m\in\mathbb{R}^k$ are design variables (shape parameters)
- $\sigma\in\mathbb{R}^N$ are the state unknowns (source strengths)
- $R(m,\sigma)\in\mathbb{R}^N$ is the discrete residual (the assembled boundary conditions)
- $J(m,\sigma)\in\mathbb{R}$ is the objective (e.g. $J=-F_x$ or $J=c_w$)

> **Important sign/notation point:** There are two common conventions for the Lagrangian.  
> They lead to the same final gradient if used consistently, but they differ by a sign in the adjoint equation.

---

## 1. Discrete Physics as a Residual

Our forward solver is a residual equation
$$
R(m,\sigma)=0.
$$

For the Rankine-source panel method, the residual is typically linear in $\sigma$:
$$
R(m,\sigma) = A(m)\,\sigma - b(m) = 0.
$$

- $A(m)\in\mathbb{R}^{N\times N}$ is the influence matrix (depends on geometry $m$)
- $b(m)\in\mathbb{R}^N$ is the RHS (also depends on geometry $m$)
- $\sigma\in\mathbb{R}^N$ is the state

So:
$$
R_\sigma \equiv \frac{\partial R}{\partial \sigma} = A(m),\qquad
R_m \equiv \frac{\partial R}{\partial m} = A_m(m)\,\sigma - b_m(m),
$$
where $A_m$ is a third-order tensor; in practice we apply it as a product $A_m\,\sigma$.

---

## 2. Objective

We choose an objective such as:

### 2.1 Start objective (recommended first): $J=-F_x$
$$
J(m,\sigma) = -F_x(m,\sigma).
$$

### 2.2 Normalized wave resistance coefficient: $J=c_w$
$$
c_w = -\frac{F_x}{\tfrac12\rho_{\text{ref}}U^2\, S(m)}.
$$

When differentiating w.r.t. $\sigma$, the denominator is constant (it depends on geometry and flow parameters, not $\sigma$).  
When differentiating w.r.t. $m$, the denominator contributes extra terms via $S(m)$.

---

## 3. Total Derivative We Want

We want the gradient
$$
\frac{dJ}{dm}
$$
accounting for the fact that $\sigma=\sigma(m)$ is implicitly defined by $R(m,\sigma(m))=0$.

By the chain rule:
$$
\frac{dJ}{dm} = J_m + J_\sigma \,\sigma_m,
$$
where:

- $J_m = \frac{\partial J}{\partial m}$ (explicit dependence on geometry)
- $J_\sigma = \frac{\partial J}{\partial \sigma}$ (sensitivity to the state)
- $\sigma_m = \frac{d\sigma}{dm}$ (implicit response of the state to geometry)

---

## 4. Differentiate the Constraint to Get $\sigma_m$

Differentiate $R(m,\sigma(m))=0$ w.r.t. $m$:
$$
R_m + R_\sigma \,\sigma_m = 0.
$$

So:
$$
\sigma_m = - R_\sigma^{-1} R_m.
$$

Plug into the total derivative:
$$
\frac{dJ}{dm} = J_m - J_\sigma \,R_\sigma^{-1} R_m.
$$

This is correct but expensive because it involves $R_\sigma^{-1}$ applied to something **for every design variable**.

---

## 5. Adjoint Trick

We introduce an adjoint vector $\lambda\in\mathbb{R}^N$ to avoid computing $\sigma_m$.

Define $\lambda$ by:
$$
R_\sigma^T \lambda = J_\sigma^T.
$$

Then
$$
\begin{aligned}
J_\sigma \,R_\sigma^{-1} R_m
&= (J_\sigma^T)^T R_\sigma^{-1} R_m \\
&= (R_\sigma^T\lambda)^T R_\sigma^{-1} R_m \\
&= \lambda^T R_\sigma R_\sigma^{-1} R_m \\
&= \lambda^T R_m.
\end{aligned}
$$

Therefore the full shape gradient becomes:
$$
\boxed{
\frac{dJ}{dm} = J_m - \lambda^T R_m
}
$$

This is the key result.

### For linear residual $R=A\sigma-b$
$$
R_\sigma = A,\qquad R_m = A_m\sigma - b_m
$$
so:
$$
\boxed{
\frac{dJ}{dm} = J_m - \lambda^T (A_m\sigma - b_m)
= J_m + \lambda^T (b_m - A_m\sigma)
}
$$

---

## 6. Two Common Lagrangian Conventions (Sign Confusion Resolved)

### Convention A (most common in PDE-constrained optimization)
$$
\mathcal{L}(m,\sigma,\lambda) = J(m,\sigma) + \lambda^T R(m,\sigma).
$$

Stationarity w.r.t. $\sigma$:
$$
\frac{\partial \mathcal{L}}{\partial \sigma}
= J_\sigma + R_\sigma^T \lambda = 0
\quad\Rightarrow\quad
\boxed{
R_\sigma^T \lambda = -J_\sigma^T
}
$$

Then the gradient becomes:
$$
\boxed{
\frac{dJ}{dm} = J_m + \lambda^T R_m
}
$$
(because this $\lambda$ has the opposite sign relative to the earlier definition).

### Convention B (matches the "adjoint trick" definition used above)
$$
\mathcal{L}(m,\sigma,\lambda) = J(m,\sigma) - \lambda^T R(m,\sigma).
$$

Stationarity w.r.t. $\sigma$:
$$
\frac{\partial \mathcal{L}}{\partial \sigma}
= J_\sigma - R_\sigma^T \lambda = 0
\quad\Rightarrow\quad
\boxed{
R_\sigma^T \lambda = J_\sigma^T
}
$$

Then the gradient becomes:
$$
\boxed{
\frac{dJ}{dm} = J_m - \lambda^T R_m
}
$$

> **Bottom line:** both are correct. They differ only by the sign convention for $\lambda$.  
> Pick one convention and stick to it.

---

## 7. Why You Sometimes See $A^T\lambda = -R_m$

If you are deriving sensitivities from the physics **to get $\sigma_m$** (i.e., tangent/forward sensitivity),
you may write:

From:
$$
R_\sigma \sigma_m = -R_m
$$
you can solve for $\sigma_m$ directly:
$$
A\,\sigma_m = -(A_m\sigma - b_m).
$$

If you introduce an adjoint-like variable to avoid forming $\sigma_m$, you *might* write a system like
$$
A^T \lambda = -R_m
$$
but note:

- This $\lambda$ is **not** the objective adjoint. It is an auxiliary variable related to the constraint derivative.
- The standard discrete adjoint for optimizing $J$ is:
  $$
  A^T \lambda = J_\sigma^T
  $$
  (or with a minus depending on Lagrangian sign).

So:

- $A^T\lambda = J_\sigma^T$ is the **adjoint equation for optimizing**.
- $A\,\sigma_m = -R_m$ is the **tangent equation** for state sensitivity.
- Writing $A^T\lambda = -R_m$ is mixing roles unless you are defining a different multiplier.

---

## 8. What Gradients Do We Need in the Code?

For a gradient-based shape update, at each iteration:

### Step 1: Forward solve (state)
Solve the residual:
$$
R(m,\sigma)=0 \quad\Rightarrow\quad A(m)\sigma=b(m).
$$

### Step 2: Evaluate objective
Compute:
$$
J(m,\sigma).
$$

### Step 3: Compute state sensitivity of objective
Compute:
$$
g_\sigma = J_\sigma \in\mathbb{R}^N.
$$

Example (for $J=-F_x$), this is what we validated numerically:
$$
g_{\sigma,j} = \sum_i \left(\frac{\partial J}{\partial v_i}\cdot \mathrm{vel}_{ij}\right).
$$

### Step 4: Adjoint solve
Solve:
$$
A(m)^T \lambda = g_\sigma^T
$$
(or $= -g_\sigma^T$ depending on your Lagrangian convention).

### Step 5: Compute shape gradient
Compute:
$$
\frac{dJ}{dm} = J_m + \lambda^T (b_m - A_m\sigma)
$$
(or the sign-flipped equivalent).

In practice:

- Compute $b_m$ (vector)
- Compute $w = A_m\sigma$ (vector), without forming $A_m$
- Combine:
  $$
  \lambda^T (b_m - w)
  $$

### Step 6: Apply update
Update design variables:
$$
m \leftarrow m - \alpha \, \nabla_m J
$$
(with line search / trust region / constraints).

---

## 9. Boundary Conditions for the Adjoint

In the **discrete adjoint**, the “boundary conditions” are automatically encoded by the discrete operator $A(m)$:

- The forward BCs are enforced by assembling $A$ and $b$ from:
  - hull no-penetration
  - free-surface linearized BC (generalized kinematic+dynamic)
- The adjoint system uses $A^T$, meaning the coupling of equations is transposed, but **no new continuous BC derivation is required**.

So in code:

- forward: solve `A sigma = b`
- adjoint: solve `A.T lam = dJ_dsigma`

---

## 10. Summary (Implementation Checklist)

At each shape iteration:

1. **Assemble** $A(m), b(m)$
2. **Solve forward** $A\sigma=b$
3. **Compute objective** $J(m,\sigma)$
4. **Compute RHS of the ajdoint equation** $g_\sigma = \partial J/\partial \sigma$
5. **Solve adjoint** $A^T\lambda=g_\sigma^T$
6. **Compute shape gradient**
   $$
   \frac{dJ}{dm} = J_m + \lambda^T(b_m - A_m\sigma)
   $$
7. **Update shape** $m \leftarrow m - \alpha \nabla_m J$

keep chugging until J is minimized

---

## 11. Practical Notes for Our Current Code

- We validated $g_\sigma=\partial(-F_x)/\partial\sigma$ using FD directional checks.
- We validated the adjoint solve residual $\|A^T\lambda-g_\sigma\|/\|g_\sigma\|$ is near machine precision.
- For the beam-scaling test, we validated:
  $$
  \frac{dJ}{dm} \approx 
  \underbrace{\frac{\partial J}{\partial m}\Big|_{\sigma}}_{\text{explicit}} +
  \underbrace{\lambda^T(b_m - A_m\sigma)}_{\text{implicit}}
  $$
  and matched full FD to high accuracy.

Next development step:

- Implement the implicit term $b_m - A_m\sigma$ without finite differences, e.g. via complex-step or analytic/AD inside the residual evaluation.

<!--  ----------------------------------------------------------------
-->

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


* code review:  

We store the induced velocity tensor:

 `vel[i, j, m]` = the $m$-th component of the velocity at fieldpoint/panel $i$ induced by a **unit** source strength on panel $j$.

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

<!-- (our NumPy: `vtotal = -vinf + einsum("ijm,j->im", vel, sigma)`) -->

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

Oops, let's just make it the force


$$
J = c_w * \tfrac{1}{2}\,\rho_{\mathrm{ref}}\,U^2\,S = -F_x.
$$

in the x direction

$$
F_x = \sum_{i \in W} A_i\, p_i\, n_{i,x}
$$



That’s the discrete objective we differentiate.

In python, we had:

```python
    cw = -force[0] / (0.5 * self.params.rho_ref * (U**2) * S) #this will become our objective function!
```

## First adjoint target: compute $\mathbf{g} = \partial J / \partial \sigma$

We’ll compute

$$
\mathbf{g} \equiv \frac{\partial J}{\partial \boldsymbol{\sigma}} \in \mathbb{R}^N
$$

* so we can solve the adjoint system

$$
A^T \boldsymbol{\lambda} = \mathbf{g}.
$$

### Key simplification (important)

* Look at the pressure term:

$$
p_i
= \frac{1}{2}\rho U^2\left(1 - \frac{\lVert \mathbf{v}_i \rVert^2}{U^2}\right) - \rho g z_i
= \frac{1}{2}\rho U^2 - \frac{1}{2}\rho \lVert \mathbf{v}_i \rVert^2 - \rho g z_i.
$$

* So we take the partial of pressure with respect to $\mathbf{v}_i$:

$$
\frac{\partial p_i}{\partial \mathbf{v}_i} = -\rho\,\mathbf{v}_i.
$$

* Then get the partial with respect to $\mathbf{v}_i$ for the $x$-component of force:

* Substitue $p$ into to the force eqation, and differentiate as above:

$$
F_x = \sum_{i \in W} A_i\, p_i\, n_{i,x}
\;\;\Rightarrow\;\;
\frac{\partial F_x}{\partial \mathbf{v}_i}
= A_i\,n_{i,x}\,\frac{\partial p_i}{\partial \mathbf{v}_i}
= -\rho\,A_i\,n_{i,x}\,\mathbf{v}_i.
$$

* And remember $J$ is $c_w$ and we define shorthand $C$ here as:

$$
J = -\frac{F_x}{C},
\qquad
C = \frac{1}{2}\rho_{\mathrm{ref}} U^2 S,
$$

* Upon differentiation with respec to $v_i$ we get

$$
\frac{\partial J}{\partial \mathbf{v}_i}
= -\frac{1}{C}\frac{\partial F_x}{\partial \mathbf{v}_i}
= \frac{\rho\,A_i\,n_{i,x}}{C}\,\mathbf{v}_i,
\qquad i \in W,
$$

and zero otherwise.

* Finally, because

$$
\mathbf{v}_i
= -\mathbf{v}_\infty + \sum_j \sigma_j\,\mathbf{vel}_{ij}
\;\;\Rightarrow\;\;
\frac{\partial \mathbf{v}_i}{\partial \sigma_j} = \mathbf{vel}_{ij},
$$

* we have

$$
\frac{\partial J}{\partial \sigma_j}
= \sum_{i=1}^N \left(\frac{\partial J}{\partial \mathbf{v}_i}\cdot \mathbf{vel}_{ij}\right).
$$

That’s the vector $\mathbf{g}$ we need.

$$ \mathbf{g} = \frac{\partial J}{\partial \sigma_j}$$


* So its components are

$$
g_j = \frac{\partial J}{\partial \sigma_j}, \qquad j = 1,\dots,N.
$$

* And it’s exactly the right-hand side for the discrete adjoint solve that we used LU decomposition to setup:

$$
A^T \boldsymbol{\lambda} = \mathbf{g}.
$$



## 3) Code: compute $\frac{\partial J}{\partial \sigma_j}$ code: ``` dJ_dsigma ``` from our existing arrays

Assuming:

- `vel` is shape `(N, N, 3)`
- `vtotal` is shape `(N, 3)`
- `normals` is shape `(N, 3)`
- `area` is shape `(N,)`
- `center` is shape `(N, 3)`
- `vinf[0] = U`
- `npanels` is the hull panel count




## 4) Then the adjoint solve (reuse LU)


At this stage, you now have the adjoint variable $\boldsymbol{\lambda}$, ready to be used for shape derivatives:

$$
\frac{dJ}{dm}
=
\frac{\partial J}{\partial m}
+
\boldsymbol{\lambda}^T
\left(
\frac{\partial b}{\partial m}
-
\frac{\partial A}{\partial m}\,\boldsymbol{\sigma}
\right).
$$


## 5) A fast sanity check (before any shape derivatives)

Do a finite-difference check w.r.t. $\boldsymbol{\sigma}$ (geometry fixed):

Pick a random perturbation $d\boldsymbol{\sigma}$ and verify:

$$
J(\boldsymbol{\sigma} + \epsilon\, d\boldsymbol{\sigma}) - J(\boldsymbol{\sigma})
\approx
\epsilon \,\frac{dJ}{d\boldsymbol{\sigma}} \cdot d\boldsymbol{\sigma}.
$$




# Checking this Gradient


## 1) `compute_dJ_dsigma_JnegFx` (for $J = -F_x$)

From our force definition, only wetted hull panels contribute. With

$$
p_i
= \frac{1}{2}\rho U^2 c_{p,i} - \rho g z_i
= \frac{1}{2}\rho U^2 - \frac{1}{2}\rho \lVert \mathbf{v}_i \rVert^2 - \rho g z_i,
$$

we get

$$
\frac{\partial (-F_x)}{\partial \mathbf{v}_i}
= -\rho\,A_i\,n_{i,x}\,\mathbf{v}_i,
$$

and therefore

$$
\frac{\partial J}{\partial \sigma_j}
= \sum_i \left(\frac{\partial J}{\partial \mathbf{v}_i}\cdot \mathbf{vel}_{ij}\right).
$$


## 2) `check_dJ_dsigma()` — finite-difference check vs `postprocess()`

This checks the directional derivative along a random direction $d\boldsymbol{\sigma}$:

$$
\frac{J(\boldsymbol{\sigma} + \epsilon\, d\boldsymbol{\sigma}) - J(\boldsymbol{\sigma} - \epsilon\, d\boldsymbol{\sigma})}{2\epsilon}
\approx
\left(\frac{\partial J}{\partial \boldsymbol{\sigma}}\right)^T d\boldsymbol{\sigma}.
$$



## Development Notes

### Sign of the adjoint term: $\frac{\partial J}{\partial \boldsymbol{\sigma}}$

checking the objective:

$$
J = -F_x
$$

Your current derivation in code uses:

$$
\frac{\partial J}{\partial v_i} = -\rho\,A_i\,n_{ix}\,v_i
$$

But that expression is actually $\partial F_x/\partial v_i$, not $\partial(-F_x)/\partial v_i$.

Let’s do it carefully:

$$
F_x = \sum_i A_i\,p_i\,n_{ix}
$$

$$
p_i = \frac{1}{2}\rho U^2 \;-\; \frac{1}{2}\rho \lVert v_i\rVert^2 \;-\; \rho g z_i
$$

so

$$
\frac{\partial p_i}{\partial v_i} = -\rho\,v_i
$$

Therefore:

$$
\frac{\partial F_x}{\partial v_i}
= A_i\,n_{ix}\,(-\rho\,v_i)
= -\rho\,A_i\,n_{ix}\,v_i
$$

And then:

$$
\frac{\partial(-F_x)}{\partial v_i}
= -\frac{\partial F_x}{\partial v_i}
= +\rho\,A_i\,n_{ix}\,v_i
$$

So our scale needs to be **positive**, not negative.


### code

we provide a `postprocess(sigma)` callable that returns $J$ (and can compute from the cached `vel`, `coordsys`, etc.). It can return more too; we only use the objective scalar.



## The Shape Derivatives

shape derivative term:
$$
\lambda^T\left(-\frac{\partial A}{\partial m}\,\sigma + \frac{\partial b}{\partial m}\right)
$$




# Miscellaneous Dev Notes

## Computing $\frac{\partial J }{\partial \sigma}$

$$
J = -F_x
$$


$$
F_x = \sum_i A_i\,p_i\,n_{ix}
$$


$$
p_i = \frac{1}{2}\rho U^2 \;-\; \frac{1}{2}\rho \lVert v_i\rVert^2 \;-\; \rho g z_i
$$






## The Shape Gradient $\frac{\partial J}{\partial m}$


Now we move from “adjoint w.r.t. $\sigma$” to “adjoint w.r.t. shape parameter $m$”.

### The discrete shape gradient dependencies and initial formulation

For one scalar design parameter $m$ (beam scale), with

$$
A(m)\,\sigma(m) = b(m),
\qquad
J(m) = J(\sigma(m), m),
$$

the total derivative is:

$$
\frac{dJ}{dm}
=
\frac{\partial J}{\partial m}
+
\lambda^T\left(
\frac{\partial b}{\partial m}
-
\frac{\partial A}{\partial m}\,\sigma
\right).
$$

For $J = -F_x$, and for a first pass, you can treat $J$ as depending on geometry only through $\sigma$ (i.e., ignore explicit dependence of area/normals/centers inside the force). That gives the “adjoint-only” term:

$$
\frac{dJ}{dm}
\approx
\lambda^T\left(
\frac{\partial b}{\partial m}
-
\frac{\partial A}{\partial m}\,\sigma
\right).
$$

Later we’ll add the explicit $\partial J/\partial m$ from the force integration.



### Beam Scaling - our first shape "function" - a shape "mode"

Define:

- For hull vertices with $z < 0$:
  $$
  y \leftarrow (1+m)\,y
  $$

- Leave free-surface points unchanged.
- Leave the symmetry plane intact (so $y=0$ stays $y=0$).

**Implementation detail:** clearly we should identify hull-used vertex indices from `panels[:, :npanels]` and only modify those vertices.









## Adjoint $\frac{\partial J}{\partial \sigma}$ Right Hand Sides

There are two different “RHS vectors” in play, and it’s easy to conflate them.

### 1) “RHS” in the adjoint solve means $\partial J/\partial \sigma$

The adjoint equation is:

$$
A^T \lambda = \frac{\partial J}{\partial \sigma}
$$

So the RHS is the vector

$$
g_\sigma := \frac{\partial J}{\partial \sigma}
\quad\text{(length $N$).}
$$

If your objective is $J = c_w$, then the adjoint RHS is:

$$
g_\sigma(c_w) = \frac{\partial c_w}{\partial \sigma}
$$

That’s what your current `compute_dJ_dsigma_cw(...)` returns. That’s what I meant by the “cw RHS”.

If your objective is $J = -F_x$, then the adjoint RHS is:

$$
g_\sigma(-F_x) = \frac{\partial(-F_x)}{\partial \sigma}
$$

That’s what `compute_dJ_dsigma_JnegFx(...)` returns.

So yes: `compute_dJ_dsigma_JnegFx` is absolutely a “real adjoint RHS” — not “validation-only”.
The validation simply checks that it’s correct.

### 2) Relationship between the two RHS vectors

In your code (with geometry fixed), you compute:

$$
c_w = \frac{-F_x}{C},
\qquad
C = \frac{1}{2}\rho_{\mathrm{ref}} U^2 S
$$

When differentiating w.r.t. $\sigma$, $C$ is constant (because $S$, $U$, $\rho_{\mathrm{ref}}$ don’t depend on $\sigma$). So:

$$
\frac{\partial c_w}{\partial \sigma}
=
\frac{1}{C}\,\frac{\partial(-F_x)}{\partial \sigma}
$$

So the $c_w$ RHS is just a scaled version of the $-F_x$ RHS.

That means:

- If you can compute $d(-F_x)/d\sigma$, you already have $dc_w/d\sigma$ for free by dividing by $C$.
- Conversely, if you compute $dc_w/d\sigma$, it’s the same direction, just scaled.

**Important:** This is only “free scaling” for derivatives w.r.t. $\sigma$.
For shape derivatives $d/dm$, $S$ *does* depend on geometry, so $c_w$’s shape gradient has extra terms.




## coding notes
* use of dependency injection allows for multiple objectives
    * You’ll have multiple objectives.
    * Today you have J=-Fx and J=cw. Tomorrow you might add:
    * drag + regularization
    * volume constraint penalty
    * fairness penalties
    * multi-speed objectives
    * Injection lets you swap objectives without editing solver internals.
* Testing becomes much easier.
    * You can inject a toy objective with a known gradient and run checks without touching production code.
* Avoid circular imports and module coupling.
    * As soon as your solver imports “objectives” and objectives import solver types (or vice versa), Python packages can get annoying. Injection keeps modules cleaner.

* CUDA/C++ transition.
    *Later you may replace the objective gradient computation with a GPU version. With injection you can select the implementation at runtime (CPU vs GPU) without rewriting everything.
* In summary, Injection is a scalability convenience, not a requirement.



## Geometry Gradients in the Physics (Shape Gradients)



the total derivative is:

$$
\frac{dJ}{dm}
=
\frac{\partial J}{\partial m}
+
\lambda^T\left(
\frac{\partial b}{\partial m}
-
\frac{\partial A}{\partial m}\,\sigma
\right).
$$

The **total** derivative:

$$
\frac{dJ}{dm}
=
\underbrace{\left.\frac{\partial J}{\partial m}\right|_{\sigma}}_{\text{explicit geometry dependence}}
\;+\;
\underbrace{\left(\frac{\partial J}{\partial \sigma}\right)^T \frac{d\sigma}{dm}}_{\text{implicit through }\sigma}
$$

the **implicit** sensitivity through the state (physics) equation:

$$
\lambda^T\left(\frac{\partial b}{\partial m} - \frac{\partial A}{\partial m}\,\sigma\right)
\quad\underbrace{\text{accounts for } d\sigma/dm}_{\text{implicit through }\sigma}
$$


For $J = -F_x$, the explicit term $\left.\partial J/\partial m\right|_{\sigma}$ is **not** small, because even with $\sigma$ held fixed, changing the hull geometry changes:

- panel normals $n_i$
- panel areas $A_i$
- panel centers $z_i$ (hydrostatic term)
- and (critically) the induced velocities `vel` used to form `vtotal` and $c_p$

So your current adjoint term is expected to be smaller than the full FD derivative.
The difference you see


the total derivative (with respect to shape, m) is:

$$
\frac{dJ}{dm}
=
\frac{\partial J}{\partial m}
+
\lambda^T\left(
\frac{\partial b}{\partial m}
-
\frac{\partial A}{\partial m}\,\sigma
\right).
$$
### What goes into ShapeGradients when you start doing it “for real” (no FD)

For a given parameter $m$ (beam scale), “analytic shape gradients” means implementing:



#### Implicit term without FD

the implicit adjoint term is:

$$
\left.\frac{dJ}{dm}\right|_{\mathrm{implicit}}
=
\lambda^T\left(
b_m - (A_m\,\sigma)
\right)
$$


$$
\left.\frac{dJ}{dm}\right|_{\mathrm{implicit}}
=
\lambda^T\left(b_m - w\right)
$$

- compute $b_m$ analytically  
- $w = \frac{\partial A}{\partial m} \sigma$  and denote $A_m = \frac{\partial A}{\partial m}$
- compute $A_m\,\sigma$ analytically (you don’t need the full $A_m$ matrix; you only need the product $A_m\,\sigma$)

#### Explicit term without FD


$$
\frac{\partial J}{\partial m}
$$

$$
\left.\frac{dJ}{dm}\right|_{\mathrm{\sigma}}
=
\lambda^T\left(b_m - w\right)
$$

- compute $\left.\frac{\partial J}{\partial m}\right|_{\sigma}$ analytically  

This includes how the force integral changes due to:

- normals, areas, centers  
- and also due to $\mathbf{v}$ changing because influence kernels depend on geometry  

You can add these incrementally.

### The best “no-FD production” next step (and maps to CUDA)



#### Implement the implicit term analytically as a matrix-free product


$$
\left.\frac{dJ}{dm}\right|_{\mathrm{implicit}}
=
\lambda^T\left(b_m - w\right)
$$

Instead of FD’ing $A_m$, compute the vector

$$
w = A_m\,\sigma
$$

directly by differentiating the row assembly formulas w.r.t. the design variable.

That’s doable for beam scaling because each panel’s geometry changes in a structured way.

Then you compute: 

$$
\left.\frac{dJ}{dm}\right|_{\mathrm{implicit}}
=
\lambda^T\left(b_m - w\right)
$$

This avoids:

- assembling a full dense derivative matrix
- FD cost scaling with $N^2$ per parameter

and it’s exactly the form you’ll want on GPU.


#### Implement the Explicit Term 


$$
\frac{\partial J}{\partial m}
$$

$$
\left.\frac{dJ}{dm}\right|_{\mathrm{\sigma}}
=
\lambda^T\left(b_m - w\right)
$$

- compute $\left.\frac{\partial J}{\partial m}\right|_{\sigma}$ analytically  





Right now your “adjoint dJ/dm” is computing only the **implicit** sensitivity through the state:

$$
\lambda^T\left(\frac{\partial b}{\partial m} - \frac{\partial A}{\partial m}\,\sigma\right)
\quad\underbrace{\text{accounts for } d\sigma/dm}_{\text{implicit through }\sigma}
$$

But your FD “truth” is the **total** derivative:

$$
\frac{dJ}{dm}
=
\underbrace{\left.\frac{\partial J}{\partial m}\right|_{\sigma}}_{\text{explicit geometry dependence}}
\;+\;
\underbrace{\left(\frac{\partial J}{\partial \sigma}\right)^T \frac{d\sigma}{dm}}_{\text{implicit through }\sigma}
$$

For $J = -F_x$, the explicit term $\left.\partial J/\partial m\right|_{\sigma}$ is **not** small, because even with $\sigma$ held fixed, changing the hull geometry changes:

- panel normals $n_i$
- panel areas $A_i$
- panel centers $z_i$ (hydrostatic term)
- and (critically) the induced velocities `vel` used to form `vtotal` and $c_p$

So your current adjoint term is expected to be smaller than the full FD derivative.
The difference you see



**Important:** You don’t have to do this for the entire operator on day 1. Start with a simplified path, validate against your FD harness (which you now trust), then expand.



## Dev Notes: Automatic Differentiation and Adjoint derivatives

## Gradients: where AD helps vs where adjoint helps

You’ve already got AD for B-splines (great). The key is choosing where to apply it.

## Best default in shape optimization

If your solver can be written as

$$
A(m)\,x(m) = b(m),
\qquad
J = J(x,m),
$$

*Note: for our problem here, we have $x = \sigma$, computationally, this is our state in the physics solver*

then the adjoint gives you gradients without ever forming $dx/dm$:

- Solve state:  
  $$
  A x = b
  $$

- Solve adjoint:  
  $$
  A^T \lambda = \frac{\partial J}{\partial x}
  $$

- Gradient:  
  $$
  \frac{dJ}{dm}
  =
  \frac{\partial J}{\partial m}
  +
  \lambda^T\left(
    \frac{\partial b}{\partial m}
    -
    \frac{\partial A}{\partial m}\,x
  \right)
  $$

## Where AD is usually a win

- Geometry parameterization: B-splines \(\rightarrow\) surface points, normals, Jacobians, constraints.
- Derivatives of local geometric quantities (areas, normals, curvature proxies) if coded carefully.

## Where adjoint is usually necessary

- Anything that goes “through the PDE/linear system solve” (because naïve AD through a large linear solve is expensive unless you implement implicit differentiation anyway—which becomes the adjoint).

So the common hybrid is:

- Hand/implicit adjoint for the physics solve,
- AD to provide $\partial A/\partial m$, $\partial b/\partial m$, and $\partial J/\partial m$ (via geometry derivatives), at least initially.

## python package requirements
* numpy, scipy, matplotlib
* numba
## documentation requirements
* Optional: Mathpix Markdown ? to render the more latex math 
    * Here we adopt the github subset for maximal compatibility
    * so you do not need it

