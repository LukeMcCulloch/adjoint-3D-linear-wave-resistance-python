# -*- coding: utf-8 -*-
"""
Full-scale (N=3360, real fifi.dat mesh) validation of the numba adjoint
pipeline (rankine_panel/numba_adjoint_kernels.py), using REAL lam/sigma
from an actual forward+adjoint solve of J=-Fx -- not the arbitrary weights
used in handderiv_assembly.py's small-sub-mesh validation. This is the
first time the numba pipeline runs at production scale.

WHAT THIS VALIDATES, AND WHY THIS WAY (not exhaustive FD, not a full
revad trace):
  - A full revad Node-trace of all ~3555 vertices x 3 coords is not
    practical (a graph that size in pure Python is far too slow/memory-
    heavy) -- the smaller sub-mesh check in handderiv_assembly.py already
    proved the MATH is right against that oracle; this file's job is
    proving the numba port behaves the same way at REAL SCALE with REAL
    (non-arbitrary) lam/sigma, not re-deriving correctness from scratch.
  - Exhaustively FD-checking every vertex would mean re-running the full
    O(N^2) forward assembly (~19s on this mesh, per prior profiling) once
    per +/-eps per DOF -- for ~10,665 DOF that's the exact cost this whole
    project exists to eliminate. Instead: a small SAMPLE of vertices (a
    few hull, a few free-surface) gets FD-checked directly against the
    production assemble_A_b_vel_nb, which is the meaningful spot-check --
    if the analytic gradient is right at a representative sample, the
    same code path produced every other entry too.

PIPELINE:
  1. Real forward solve: A@sigma=b (production assemble_A_b_vel_nb + LU).
  2. Real objective gradient: dJ/dsigma for J=-Fx (SourceStrengthGradients).
  3. Real adjoint solve: A^T @ lam = dJ/dsigma.
  4. phi(vertices) := lam^T(A(vertices)@sigma - b(vertices)), lam/sigma
     FIXED from steps 1-3 -- run the numba O(N^2) hot loop
     (assemble_dphi_dgeom_nb) over the FULL N=3360 mesh, then the O(N)
     panel_geometry_one_backward + scatter-add conversion to d(vertices).
  5. FD-check a small sample of vertices directly against production
     assemble_A_b_vel_nb (lam, sigma held fixed, NOT re-solved -- matching
     the actual adjoint methodology, where re-solving per perturbation
     would defeat the entire point of the adjoint).
"""
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from rankine_panel.io import read_panel_file
from rankine_panel.geometry import panel_geometry_all
from rankine_panel.numba_kernels import assemble_A_b_vel_nb
from rankine_panel.numba_adjoint_kernels import assemble_dphi_dgeom_nb
from rankine_panel.objectives import SourceStrengthGradients
from rankine_panel.solver import FlowParams
from rankine_panel.gradient_validators import Validate_shape_gradient

from handderiv_panel_geometry import panel_geometry_one_forward, panel_geometry_one_backward


if __name__ == "__main__":
    pf = read_panel_file(os.path.join(_PROJECT_ROOT, "examples", "fifi.dat"))
    N = pf.npanels + pf.nfspanels
    print(f"real mesh: npanels={pf.npanels}, nfspanels={pf.nfspanels}, N={N}, "
          f"N^2={N*N:,} cells, Npoints={pf.points.shape[1]}")

    Fr = 0.3
    params = FlowParams(gravity=9.80665, length=1.0, rho_water=1025.0, rho_ref=1000.0)

    # ---------------- 1) real forward solve ----------------
    t0 = time.perf_counter()
    A, b, vel, vinf, center, coordsys, area = Validate_shape_gradient.assemble_from_points(
        pf.points, pf, Fr, params, panel_geometry_all, assemble_A_b_vel_nb)
    t1 = time.perf_counter()
    print(f"\nforward assembly (production numba): {t1-t0:.2f}s")

    lu, piv = lu_factor(A)
    sigma = lu_solve((lu, piv), b)

    resid = np.linalg.norm(A @ sigma - b) / (np.linalg.norm(b) + 1e-30)
    print(f"forward solve relative residual: {resid:.3e}")

    # ---------------- 2) real objective gradient (J = -Fx) ----------------
    vtotal = -vinf[None, :] + np.einsum("ijm,j->im", vel, sigma)
    normals = coordsys[:, :, 2]
    dJ_dsigma = SourceStrengthGradients.compute_dJ_dsigma_JnegFx(
        vel=vel, vtotal=vtotal, normals=normals, area=area, center=center,
        npanels=pf.npanels, rho_water=params.rho_water)

    # ---------------- 3) real adjoint solve ----------------
    lam = lu_solve((lu, piv), dJ_dsigma, trans=1)
    adj_resid = np.linalg.norm(A.T @ lam - dJ_dsigma) / (np.linalg.norm(dJ_dsigma) + 1e-30)
    print(f"adjoint solve relative residual: {adj_resid:.3e}")

    phi0 = float(lam @ (A @ sigma - b))
    print(f"phi0 = lam^T(A@sigma-b) at the solution: {phi0:.3e}  (expect ~0, sigma solves A@sigma=b)")

    # ---------------- 4) the numba adjoint pipeline, FULL mesh ----------------
    geom = panel_geometry_all(pf.points, pf.panels)
    cornerslocal = np.moveaxis(geom.cornerslocal, 2, 0).astype(np.float64)  # (N,2,4)

    t0 = time.perf_counter()
    d_center_nb, d_coordsys_nb, d_cornerslocal_nb = assemble_dphi_dgeom_nb(
        center, coordsys, cornerslocal, pf.npanels, pf.nfspanels,
        pf.deltax, vinf[0], params.gravity, lam, sigma)
    t1 = time.perf_counter()
    print(f"\nnumba O(N^2) hot loop (assemble_dphi_dgeom_nb), N={N}: {t1-t0:.2f}s")

    # O(N) conversion: per-panel geometry-grad -> vertex-grad, scatter-add
    t0 = time.perf_counter()
    panel_caches = [None] * N
    panel_vert_idx = [None] * N
    for p in range(N):
        vidx = pf.panels[:, p]
        xyz = pf.points[:, vidx].T.copy()
        corners = [xyz[k] for k in range(4)]
        _, _, _, _, cache = panel_geometry_one_forward(corners)
        panel_caches[p] = cache
        panel_vert_idx[p] = vidx

    Npoints = pf.points.shape[1]
    d_vertices = np.zeros((3, Npoints))
    for p in range(N):
        d_c0, d_c1, d_c2, d_c3 = panel_geometry_one_backward(
            panel_caches[p], d_center_nb[p], d_coordsys_nb[p], d_cornerslocal_nb[p], 0.0)
        vidx = panel_vert_idx[p]
        d_vertices[:, vidx[0]] += d_c0
        d_vertices[:, vidx[1]] += d_c1
        d_vertices[:, vidx[2]] += d_c2
        d_vertices[:, vidx[3]] += d_c3
    t1 = time.perf_counter()
    print(f"O(N) geometry->vertex conversion + scatter-add, N={N}: {t1-t0:.2f}s")
    print(f"total gradient magnitude: max|d_vertices|={np.max(np.abs(d_vertices)):.3e}, "
          f"mean|d_vertices|={np.mean(np.abs(d_vertices)):.3e}")

    # ---------------- 5) FD spot-check vs. production assemble_A_b_vel_nb ----------------
    def phi_of_points(points_pert):
        geom_p = panel_geometry_all(points_pert, pf.panels)
        center_p = geom_p.center.T.astype(np.float64)
        coordsys_p = np.moveaxis(geom_p.coordsys, 2, 0).astype(np.float64)
        cornerslocal_p = np.moveaxis(geom_p.cornerslocal, 2, 0).astype(np.float64)
        area_p = geom_p.area.astype(np.float64)
        A_p, b_p, _ = assemble_A_b_vel_nb(center_p, coordsys_p, cornerslocal_p, area_p,
                                           pf.npanels, pf.nfspanels, pf.deltax, vinf[0], params.gravity)
        return float(lam @ (A_p @ sigma - b_p))

    hull_vert = pf.panels[0, 0]           # a hull vertex (panel 0's first corner)
    fs_vert = pf.panels[0, pf.npanels]    # a free-surface vertex (first FS panel's first corner)
    sample_verts = sorted(set([hull_vert, fs_vert, pf.panels[2, 5], pf.panels[1, pf.npanels + 10]]))
    print(f"\nFD spot-check on sample vertices: {sample_verts}")

    eps = 1e-6
    max_rel_err = 0.0
    for v in sample_verts:
        for ax in range(3):
            pts_p = pf.points.copy(); pts_p[ax, v] += eps
            pts_m = pf.points.copy(); pts_m[ax, v] -= eps
            fd = (phi_of_points(pts_p) - phi_of_points(pts_m)) / (2*eps)
            analytic = d_vertices[ax, v]
            denom = max(abs(fd), abs(analytic), 1e-8)
            rel_err = abs(fd - analytic) / denom
            max_rel_err = max(max_rel_err, rel_err)
            print(f"  vertex {v:5d} axis={('x','y','z')[ax]}: analytic={analytic:.6e}  fd={fd:.6e}  rel_err={rel_err:.3e}")

    print(f"\nmax relative error over the sampled DOF: {max_rel_err:.3e}")
