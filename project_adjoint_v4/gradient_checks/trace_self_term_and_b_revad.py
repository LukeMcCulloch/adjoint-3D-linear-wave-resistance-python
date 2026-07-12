# -*- coding: utf-8 -*-
"""
Computes:
    d( A[row,row] self-term ) / d(vertices)   -- turns out to be exactly zero, see below
    d( b[row] )               / d(vertices)   -- for a hull row

This is an Oracle:
    pure Python, builds a full object graph, does dynamic dispatch through operator overloading

Neither of these needs hs_influence/phixx_influence at all -- both depend on
vertices ONLY through the row panel's own normal (panel_geometry_revad's
coordsys output, 3rd column), which we already have a validated Jacobian
for. That's why these are the "free" gaps: no new kernel-level AD, just
using a piece we'd already built and had been discarding elsewhere.

SELF-TERM (hull diagonal, row==col), from assemble_A_b_vel_nb:
    v = -0.5 * normal_row
    A[row,row] = dot(normal_row, v) = -0.5 * dot(normal_row, normal_row)

normal_row is always UNIT LENGTH by construction (panel_geometry_all
explicitly normalizes it: nv /= sqrt(sum(nv*nv))). So
dot(normal_row, normal_row) = 1 identically, for ANY non-degenerate panel,
regardless of vertex positions -- meaning A[row,row] = -0.5 * 1 = -0.5 is a
CONSTANT. Its derivative w.r.t. vertices is exactly zero, not because of a
cancellation we need to trust numerically, but structurally: the normal's
own self-dot is pinned to 1 by the normalization step, so no perturbation
of the panel's vertices can move A[row,row] away from -0.5 (short of
degenerating the panel entirely, i.e. breaking the normalization -- the
parked branch-coverage case). Confirmed numerically below rather than just
asserted, since "should be zero" is exactly the kind of claim worth
checking.

B VECTOR (hull rows), from assemble_A_b_vel_nb:
    b[row] = normal_row[0] * vinf_x       (vinf_x is a CONSTANT, not geometry)

This one is NOT structurally constant -- the x-COMPONENT of a unit vector
varies with geometry even though its magnitude doesn't. So
d(b[row])/d(vertices) = vinf_x * d(normal_row_x)/d(row vertices), genuinely
nonzero in general, and validated against FD below.

FS rows of b are literally the constant 0 (see numba_kernels.py), zero
dependence on anything -- not worth a trace.
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from rankine_panel.revad import jacobian

from rankine_panel.io import read_panel_file
from rankine_panel.geometry import panel_geometry_all, panel_geometry_one

from trace_panel_geometry_revad import _dot


# =============================================================================
# Self-term: A[row,row] = -0.5 * dot(normal_row, normal_row)
# =============================================================================

def self_term_A_revad(row_corners):
    _center_row, coordsys_row, _cornerslocal_row, _area_row = panel_geometry_one(row_corners)
    normal_row = [coordsys_row[r][2] for r in range(3)]
    return _dot(normal_row, normal_row) * (-0.5)


# =============================================================================
# b[row] = normal_row_x * vinf_x  (hull rows)
# =============================================================================

def b_row_revad(row_corners, vinf_x):
    _center_row, coordsys_row, _cornerslocal_row, _area_row = panel_geometry_one(row_corners)
    normal_row_x = coordsys_row[0][2]
    return normal_row_x * vinf_x


# =============================================================================
# 12-scalar packing (one panel's 4 corners) -- reuses trace_panel_geometry_revad's pattern
# =============================================================================

def pack(row_xyz):
    return np.asarray(row_xyz, dtype=np.float64).ravel()


def unpack_nodes(xs):
    return [[xs[3*k + i] for i in range(3)] for k in range(4)]


def run_traced_self_term(xs_nodes):
    return [self_term_A_revad(unpack_nodes(xs_nodes))]


def make_run_traced_b(vinf_x):
    def run_traced_b(xs_nodes):
        return [b_row_revad(unpack_nodes(xs_nodes), vinf_x)]
    return run_traced_b


# =============================================================================
# Plain-Python primals (for FD)
# =============================================================================

def self_term_A_primal(points, panels, row):
    geom = panel_geometry_all(points, panels)
    n = geom.coordsys[:, 2, row]
    return float(-0.5 * np.dot(n, n))


def b_row_primal(points, panels, row, vinf_x):
    geom = panel_geometry_all(points, panels)
    return float(geom.coordsys[0, 2, row] * vinf_x)


if __name__ == "__main__":
    pf = read_panel_file(os.path.join(_PROJECT_ROOT, "examples", "fifi.dat"))

    ROW = 0  # a hull panel
    row_vert_idx = pf.panels[:, ROW]
    row_xyz0 = pf.points[:, row_vert_idx].T.copy()
    x0 = pack(row_xyz0)

    Fr, gravity, length = 0.3, 9.80665, 1.0
    vinf_x = Fr * np.sqrt(length * gravity)

    # ---------------- self-term ----------------
    print(f"=== Self-term A[row={ROW},row={ROW}] ===")
    A0 = self_term_A_primal(pf.points, pf.panels, ROW)
    print("primal value (should be exactly -0.5):", A0)

    J_revad = jacobian(run_traced_self_term, x0)[0]  # (12,)
    eps = 1e-6
    J_fd = np.zeros(12)
    for i in range(12):
        xp = x0.copy(); xp[i] += eps
        xm = x0.copy(); xm[i] -= eps
        points_p = pf.points.copy(); points_p[:, row_vert_idx] = xp.reshape(4, 3).T
        points_m = pf.points.copy(); points_m[:, row_vert_idx] = xm.reshape(4, 3).T
        J_fd[i] = (self_term_A_primal(points_p, pf.panels, ROW) - self_term_A_primal(points_m, pf.panels, ROW)) / (2*eps)

    print("max |revad|:", np.max(np.abs(J_revad)), " (expect ~0, structurally constant)")
    print("max |fd|:   ", np.max(np.abs(J_fd)), " (expect ~0 too, up to FD noise)")

    # ---------------- b[row] ----------------
    print(f"\n=== b[row={ROW}] ===")
    b0 = b_row_primal(pf.points, pf.panels, ROW, vinf_x)
    print("primal value:", b0)

    run_traced_b = make_run_traced_b(vinf_x)
    Jb_revad = jacobian(run_traced_b, x0)[0]  # (12,)
    Jb_fd = np.zeros(12)
    for i in range(12):
        xp = x0.copy(); xp[i] += eps
        xm = x0.copy(); xm[i] -= eps
        points_p = pf.points.copy(); points_p[:, row_vert_idx] = xp.reshape(4, 3).T
        points_m = pf.points.copy(); points_m[:, row_vert_idx] = xm.reshape(4, 3).T
        Jb_fd[i] = (b_row_primal(points_p, pf.panels, ROW, vinf_x) - b_row_primal(points_m, pf.panels, ROW, vinf_x)) / (2*eps)

    diff = np.abs(Jb_revad - Jb_fd)
    denom = np.maximum(np.abs(Jb_revad), np.abs(Jb_fd))
    mask = denom > 1e-8
    max_rel = np.max(diff[mask]/denom[mask]) if mask.any() else 0.0
    print("max |revad - fd| (absolute):", np.max(diff))
    print("max relative error:", max_rel)
