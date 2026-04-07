# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 00:38:34 2026

@author: Luke McCulloch
    tlukemcculloch@gmail.com
"""

# rankine_panel/shape_gradients.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.linalg import lu_factor, lu_solve




# -----------------------------
# we shall use the already existing
# -----------------------------
# PanelFile: pf.npanels, pf.nfspanels, pf.deltax, pf.panels, pf.points, pf.N
# FlowParams: gravity, length, rho_water, rho_ref
# panel_geometry_all(points, panels) -> PanelGeometry (center, coordsys, cornerslocal, area)
# assemble_A_b_vel_nb(center, coordsys, cornerslocal, area, npanels, nfspanels, deltax, vinf_x, gravity)
#   -> A (N,N), b (N,), vel (N,N,3)
#
# SourceStrengthGradients.compute_dJ_dsigma_JnegFx(...) or compute_dJ_dsigma_cw(...)
# Validate_dj_dsigma.make_postprocess_JnegFx(...) etc (optional)


Array = np.ndarray # type shorthand


@dataclass(frozen=True)
class StateAdjoint:
    # baseline (at x0)
    A: Array              # (N,N)
    b: Array              # (N,)
    vel: Array            # (N,N,3)
    vinf: Array           # (3,)
    center: Array         # (N,3)
    coordsys: Array       # (N,3,3)
    area: Array           # (N,)

    sigma: Array          # (N,)
    lam: Array            # (N,)

    # handy
    vtotal: Array         # (N,3)


# =============================================================================
# 1) Geometry parameterization: beam scaling
# =============================================================================

def hull_vertex_indices(panels4xN: Array, npanels: int) -> Array:
    """Unique vertex ids used by hull panels only."""
    return np.unique(np.asarray(panels4xN[:, :npanels]).ravel())


def apply_beam_scale(points0: Array, hull_verts: Array, m: float, z_cut: float = 0.0) -> Array:
    """
    Beam scaling shape modification on hull vertices:

      y <- y * (1 + m)  for hull vertices with z < z_cut

    points0 shape: (3, npoints)
    """
    pts = np.asarray(points0, dtype=np.float64).copy()
    idx = np.asarray(hull_verts, dtype=np.int64)

    mask = pts[2, idx] < z_cut
    idx2 = idx[mask]
    pts[1, idx2] *= (1.0 + m)
    return pts


def beam_scale_tangent(points0: Array, hull_verts: Array, z_cut: float = 0.0) -> Array:
    """
    Tangent dx/dm for the beam scale mapping at m=0:

      y <- y*(1+m)  =>  dy/dm = y

    Returns tangent array dpoints_dm with same shape as points0 (3,npoints).
    """
    pts = np.asarray(points0, dtype=np.float64)
    idx = np.asarray(hull_verts, dtype=np.int64)

    dpts = np.zeros_like(pts)
    mask = pts[2, idx] < z_cut
    idx2 = idx[mask]
    dpts[1, idx2] = pts[1, idx2]  # dy/dm = y
    return dpts


# =============================================================================
# 2) Assembly helpers: from points -> (A,b,vel, caches)
# =============================================================================

def vinf_from_Fr(Fr: float, params) -> Array:
    """our convention: vinf = [Fr*sqrt(L*g), 0, 0]."""
    U = float(Fr * np.sqrt(float(params.length) * float(params.gravity)))
    return np.array([U, 0.0, 0.0], dtype=np.float64)


def assemble_from_points(
    points: Array,
    pf,
    Fr: float,
    params,
    panel_geometry_all: Callable[[Array, Array], object],
    assemble_A_b_vel_nb: Callable[..., Tuple[Array, Array, Array]],
) -> Tuple[Array, Array, Array, Array, Array, Array, Array]:
    """
    Assemble A,b,vel and return geometry caches, using our Numba assembly.
    """
    geom = panel_geometry_all(points, pf.panels)

    vinf = vinf_from_Fr(Fr, params)
    gravity = float(params.gravity)

    center = np.asarray(geom.center.T, dtype=np.float64)                      # (N,3)
    coordsys = np.asarray(np.moveaxis(geom.coordsys, 2, 0), dtype=np.float64) # (N,3,3)
    cornerslocal = np.asarray(np.moveaxis(geom.cornerslocal, 2, 0), dtype=np.float64)  # (N,2,4)
    area = np.asarray(geom.area, dtype=np.float64)

    A, b, vel = assemble_A_b_vel_nb(
        center, coordsys, cornerslocal, area,
        pf.npanels, pf.nfspanels, pf.deltax,
        vinf[0], gravity
    )
    return A, b, vel, vinf, center, coordsys, area


def residual_apply_linear(A: Array, b: Array, sigma: Array) -> Array:
    """R = A sigma - b."""
    return A @ sigma - b


# =============================================================================
# 3) State + adjoint solve at baseline geometry
# =============================================================================

def solve_state_and_adjoint(
    pf,
    points0: Array,
    Fr: float,
    params,
    panel_geometry_all,
    assemble_A_b_vel_nb,
    SourceStrengthGradients,
    objective: str = "JnegFx",
) -> StateAdjoint:
    """
    Solve the forward (A sigma = b) and adjoint (A^T lam = dJ/dsigma).

    objective:
      - "JnegFx" uses compute_dJ_dsigma_JnegFx
      - "cw" uses compute_dJ_dsigma_cw (requires rho_ref and vinf)
    """
    A, b, vel, vinf, center, coordsys, area = assemble_from_points(
        points0, pf, Fr, params, panel_geometry_all, assemble_A_b_vel_nb
    )

    lu, piv = lu_factor(A)
    sigma = lu_solve((lu, piv), b)

    # vtotal = -vinf + vel*sigma (our cached velocity convention)
    vtotal = -vinf[None, :] + np.einsum("ijm,j->im", vel, sigma)

    normals = coordsys[:, :, 2]
    rho_water = float(params.rho_water)

    if objective == "JnegFx":
        dJ_dsigma = SourceStrengthGradients.compute_dJ_dsigma_JnegFx(
            vel=vel, vtotal=vtotal,
            normals=normals,
            area=area,
            center=center,
            npanels=pf.npanels,
            rho_water=rho_water,
        )
    elif objective == "cw":
        dJ_dsigma = SourceStrengthGradients.compute_dJ_dsigma_cw(
            vel=vel, vtotal=vtotal,
            normals=normals,
            area=area,
            center=center,
            npanels=pf.npanels,
            vinf=vinf,
            rho_water=rho_water,
            rho_ref=float(params.rho_ref),
        )
    else:
        raise ValueError(f"Unknown objective='{objective}'")

    lam = lu_solve((lu, piv), dJ_dsigma, trans=1)

    # sanity
    adj_relres = np.linalg.norm(A.T @ lam - dJ_dsigma) / (np.linalg.norm(dJ_dsigma) + 1e-30)
    if adj_relres > 1e-8:
        print("WARNING: adjoint relres is large:", adj_relres)

    return StateAdjoint(
        A=A, b=b, vel=vel, vinf=vinf, center=center, coordsys=coordsys, area=area,
        sigma=sigma, lam=lam, vtotal=vtotal
    )


# =============================================================================
# 4) Shape gradient ingredients (matrix-free)
# =============================================================================

def explicit_term_fd_sigma_fixed(
    pf,
    points0: Array,
    Fr: float,
    params,
    panel_geometry_all,
    assemble_A_b_vel_nb,
    postprocess_factory: Callable[..., Callable[[Array], float]],
    sigma0: Array,
    points_plus: Array,
    points_minus: Array,
    eps_m: float,
) -> float:
    """
    Compute explicit term J_m (sigma fixed) by FD:

      J_m ≈ ( J(x+eps, sigma0) - J(x-eps, sigma0) ) / (2 eps)

    Where postprocess_factory must accept (vel, vinf, coordsys, area, center, npanels, rho, gravity)
    and return a postprocess(sigma)->J callable.

    Note: this evaluates J on perturbed geometry with baseline sigma0.
    """
    # plus
    Ap, bp, velp, vinfp, centerp, coordsyp, areap = assemble_from_points(
        points_plus, pf, Fr, params, panel_geometry_all, assemble_A_b_vel_nb
    )
    post_p = postprocess_factory(
        velp, vinfp, coordsyp, areap, centerp,
        pf.npanels, float(params.rho_water), float(params.gravity)
    )
    Jp_fix = float(post_p(sigma0))

    # minus
    Am, bm, velm, vinfm, centerm, coordsym, aream = assemble_from_points(
        points_minus, pf, Fr, params, panel_geometry_all, assemble_A_b_vel_nb
    )
    post_m = postprocess_factory(
        velm, vinfm, coordsym, aream, centerm,
        pf.npanels, float(params.rho_water), float(params.gravity)
    )
    Jm_fix = float(post_m(sigma0))

    return (Jp_fix - Jm_fix) / (2.0 * eps_m)


def implicit_term_matrix_free_fd(
    pf,
    points0: Array,
    Fr: float,
    params,
    panel_geometry_all,
    assemble_A_b_vel_nb,
    sigma0: Array,
    lam0: Array,
    dpoints_dm: Array,
    eps_m: float,
) -> float:
    """
    Compute implicit term via matrix-free directional derivative:

      lam^T R_m = lam^T R_x[ dx/dm ]

    using FD on residual with sigma held fixed:

      R_x[dx] ≈ ( R(x + eps*dx, sigma0) - R(x - eps*dx, sigma0) ) / (2 eps)

    Returns scalar: lam^T R_x[dx/dm].
    """
    x_plus = points0 + eps_m * dpoints_dm
    x_minus = points0 - eps_m * dpoints_dm

    Ap, bp, _, _, _, _, _ = assemble_from_points(
        x_plus, pf, Fr, params, panel_geometry_all, assemble_A_b_vel_nb
    )
    Rp = residual_apply_linear(Ap, bp, sigma0)

    Am, bm, _, _, _, _, _ = assemble_from_points(
        x_minus, pf, Fr, params, panel_geometry_all, assemble_A_b_vel_nb
    )
    Rm = residual_apply_linear(Am, bm, sigma0)

    Rdir = (Rp - Rm) / (2.0 * eps_m)
    return float(np.dot(lam0, Rdir))


# =============================================================================
# 5) One-stop beam-scale gradient check (prints split + matches full FD)
# =============================================================================

def check_beam_scale_shape_gradient(
    pf,
    points0: Array,
    Fr: float,
    params,
    panel_geometry_all,
    assemble_A_b_vel_nb,
    SourceStrengthGradients,
    postprocess_factory,
    eps_m: float = 1e-5,
    m0: float = 0.0,
    z_cut: float = 0.0,
    objective: str = "JnegFx",
    verbose: bool = True,
) -> None:
    """
    Beam scaling shape-gradient check for one parameter m.

    Compares:
      - FD total (solve at m±eps):     dJ/dm
      - FD explicit (sigma fixed):    J_m
      - implicit adjoint term:        lam^T R_m
      - predicted = explicit + implicit

    This is the exact “gold standard” check we already got to ~1e-10 relative.
    """
    hull_verts = hull_vertex_indices(pf.panels, pf.npanels)

    # baseline state+adjoint
    sa = solve_state_and_adjoint(
        pf, points0, Fr, params,
        panel_geometry_all, assemble_A_b_vel_nb,
        SourceStrengthGradients,
        objective=objective
    )
    sigma0 = sa.sigma
    lam0 = sa.lam

    # baseline objective value (at baseline geometry + sigma0)
    post0 = postprocess_factory(
        sa.vel, sa.vinf, sa.coordsys, sa.area, sa.center,
        pf.npanels, float(params.rho_water), float(params.gravity)
    )
    J0 = float(post0(sigma0))

    # perturbed geometries
    pts_p = apply_beam_scale(points0, hull_verts, m0 + eps_m, z_cut=z_cut)
    pts_m = apply_beam_scale(points0, hull_verts, m0 - eps_m, z_cut=z_cut)

    # FD TOTAL: solve at +/- and evaluate J
    Ap, bp, velp, vinfp, centerp, coordsyp, areap = assemble_from_points(
        pts_p, pf, Fr, params, panel_geometry_all, assemble_A_b_vel_nb
    )
    lup, pivp = lu_factor(Ap)
    sigma_p = lu_solve((lup, pivp), bp)
    Jp = float(postprocess_factory(
        velp, vinfp, coordsyp, areap, centerp,
        pf.npanels, float(params.rho_water), float(params.gravity)
    )(sigma_p))

    Am, bm, velm, vinfm, centerm, coordsym, aream = assemble_from_points(
        pts_m, pf, Fr, params, panel_geometry_all, assemble_A_b_vel_nb
    )
    lum, pivm = lu_factor(Am)
    sigma_m = lu_solve((lum, pivm), bm)
    Jm = float(postprocess_factory(
        velm, vinfm, coordsym, aream, centerm,
        pf.npanels, float(params.rho_water), float(params.gravity)
    )(sigma_m))

    dJ_dm_fd_total = (Jp - Jm) / (2.0 * eps_m)

    # explicit (sigma fixed): J_m
    dJ_dm_explicit = explicit_term_fd_sigma_fixed(
        pf=pf,
        points0=points0,
        Fr=Fr,
        params=params,
        panel_geometry_all=panel_geometry_all,
        assemble_A_b_vel_nb=assemble_A_b_vel_nb,
        postprocess_factory=postprocess_factory,
        sigma0=sigma0,
        points_plus=pts_p,
        points_minus=pts_m,
        eps_m=eps_m,
    )

    # implicit: lam^T R_m = lam^T R_x[dx/dm]
    dpts_dm = beam_scale_tangent(points0, hull_verts, z_cut=z_cut)
    dJ_dm_implicit = implicit_term_matrix_free_fd(
        pf=pf,
        points0=points0,
        Fr=Fr,
        params=params,
        panel_geometry_all=panel_geometry_all,
        assemble_A_b_vel_nb=assemble_A_b_vel_nb,
        sigma0=sigma0,
        lam0=lam0,
        dpoints_dm=dpts_dm,
        eps_m=eps_m,
    )

    dJ_dm_pred = dJ_dm_explicit + dJ_dm_implicit

    abs_err = abs(dJ_dm_fd_total - dJ_dm_pred)
    rel_err = abs_err / (abs(dJ_dm_fd_total) + 1e-30)

    if verbose:
        print("\nBeam scaling shape-gradient check")
        print("  objective =", objective)
        print("  m0 =", m0, " eps_m =", eps_m, " z_cut =", z_cut)
        print("  J0 =", J0)
        print("  FD total dJ/dm       =", dJ_dm_fd_total)
        print("  FD explicit (σ fixed)=", dJ_dm_explicit)
        print("  implicit (λᵀ R_m)    =", dJ_dm_implicit)
        print("  pred (explicit+impl) =", dJ_dm_pred)
        print("  abs err              =", abs_err)
        print("  rel err              =", rel_err)


# =============================================================================
# 6) Hooks for “real” production gradients (placeholders)
# =============================================================================
#
# When we move beyond 1–5 scalar params (beam scale / bow fullness),
# we'll want per-vertex gradients:
#
#   grad_x_total = grad_x_J_explicit + grad_x_phi_implicit
#
# This file currently provides *scalar-parameter* machinery (matrix-free),
# which already scales nicely to "a few parameters" and validates our adjoint logic.
#
# Next step is implementing:
#   residual_shape_vjp(x, sigma, lam) -> grad_x_phi
# without FD and without forming A_m.
#
# That will require differentiating our influence integrals wrt panel geometry.
#
# This file is intentionally structured so we can drop that in later.
#