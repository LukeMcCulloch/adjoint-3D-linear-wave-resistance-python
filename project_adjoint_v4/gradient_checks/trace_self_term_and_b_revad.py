# -*- coding: utf-8 -*-
"""
Computes:
    d( A[row,row] self-term ) / d(vertices)   -- for a hull row
    d( b[row] )               / d(vertices)   -- for a hull row

This is an Oracle:
    pure Python, builds a full object graph, does dynamic dispatch through operator overloading

CORRECTED 2026-07-15 -- self_term_A_revad below previously computed
`A[row,row] = -0.5 * dot(normal_row, normal_row)` (a structural constant,
-0.5, with a claimed-and-believed exactly-zero gradient) and was validated
against itself via FD, which of course agreed -- FD on a wrong formula
just confirms the wrong formula self-consistently, it can't catch that the
formula itself doesn't match production. Reading numba_kernels.py::
assemble_A_b_vel_nb directly (not just recalling it) during a later piece
of work revealed the real formula: the `row==col` special case ONLY
replaces the non-mirrored half of the velocity with the closed-form
`-0.5*normal_row` (presumably because hs_influence at a panel's own
center against its own geometry is a genuine singularity) -- the MIRROR
term is still computed and added UNCONDITIONALLY, exactly like every
other hull-row entry. So the true formula is:

SELF-TERM (hull diagonal, row==col), from assemble_A_b_vel_nb:
    v  = -0.5 * normal_row                                  (closed form,
                                                               no hs_influence
                                                               call -- would
                                                               be singular)
    vp = hs_influence(fieldpoint=y-flip(center_row), center_row, coordsys_row,
                       cornerslocal_row)                     (mirror term,
                                                               ALWAYS computed,
                                                               using the row
                                                               panel's OWN
                                                               geometry as
                                                               "col", since
                                                               col==row here)
    v_combined = [v[0]+vp[0], v[1]-vp[1], v[2]+vp[2]]
    A[row,row] = dot(normal_row, v_combined)
               = -0.5 * dot(normal_row,normal_row) + dot(normal_row, vp_combined)
               = -0.5 + dot(normal_row, vp_combined)   (since the normal is
                                                          unit length, but the
                                                          SECOND term is a
                                                          real, generically
                                                          NONZERO "image
                                                          panel across the
                                                          centerline"
                                                          contribution)

Confirmed directly against the real numba-jitted assemble_A_b_vel_nb on
fifi.dat panel 0: production A[0,0] = -0.9345545330215205, NOT -0.5 -- the
old formula was off by the entire mirror-term contribution. A[row,row] is
NOT structurally constant after all; its gradient is genuinely nonzero,
same character as b[row] below, and validated against FD accordingly (not
against a "should be ~0" expectation).

Also notice: normal_row now feeds THIS formula through hs_influence's
fieldpoint AND center AND coordsys AND cornerslocal arguments simultaneously
(since the row panel plays BOTH the row role and the col role at once) --
a genuine instance of trace_full_chain_A_entry_revad.py's "row/col dual
role" note, just realized within a single panel instead of two.

B VECTOR (hull rows), from assemble_A_b_vel_nb:
    b[row] = normal_row[0] * vinf_x       (vinf_x is a CONSTANT, not geometry)

Structurally not constant -- the x-COMPONENT of a unit vector varies with
geometry even though its magnitude doesn't. So d(b[row])/d(vertices) =
vinf_x * d(normal_row_x)/d(row vertices), genuinely nonzero in general,
validated against FD below. This one was never wrong -- b's assembly loop
in numba_kernels.py has no mirror-term logic at all, confirmed by direct
reading, so nothing to correct here.

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
from rankine_panel.influence import hs_influence

from trace_panel_geometry_revad import _dot


# =============================================================================
# Self-term: A[row,row] = -0.5 + dot(normal_row, mirror-combined hs_influence)
# =============================================================================

def self_term_A_revad(row_corners):
    center_row, coordsys_row, cornerslocal_row, _area_row = panel_geometry_one(row_corners)
    normal_row = [coordsys_row[r][2] for r in range(3)]

    v = [normal_row[0] * (-0.5), normal_row[1] * (-0.5), normal_row[2] * (-0.5)]

    fieldpoint_mirror = [center_row[0], center_row[1] * (-1.0), center_row[2]]
    vp = hs_influence(fieldpoint_mirror, center_row, coordsys_row, cornerslocal_row)

    v_combined = [v[0] + vp[0], v[1] - vp[1], v[2] + vp[2]]
    return _dot(normal_row, v_combined)


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
    center_row = geom.center[:, row]
    coordsys_row = geom.coordsys[:, :, row]
    cornerslocal_row = geom.cornerslocal[:, :, row]

    v = -0.5 * n
    pp = np.array([center_row[0], -center_row[1], center_row[2]])
    vp = hs_influence(pp, center_row, coordsys_row, cornerslocal_row)
    v_combined = np.array([v[0] + vp[0], v[1] - vp[1], v[2] + vp[2]])
    return float(np.dot(n, v_combined))


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
    print("primal value (NOT -0.5 -- includes the mirror/image-panel term):", A0)

    J_revad = jacobian(run_traced_self_term, x0)[0]  # (12,)
    eps = 1e-6
    J_fd = np.zeros(12)
    for i in range(12):
        xp = x0.copy(); xp[i] += eps
        xm = x0.copy(); xm[i] -= eps
        points_p = pf.points.copy(); points_p[:, row_vert_idx] = xp.reshape(4, 3).T
        points_m = pf.points.copy(); points_m[:, row_vert_idx] = xm.reshape(4, 3).T
        J_fd[i] = (self_term_A_primal(points_p, pf.panels, ROW) - self_term_A_primal(points_m, pf.panels, ROW)) / (2*eps)

    diff = np.abs(J_revad - J_fd)
    denom = np.maximum(np.abs(J_revad), np.abs(J_fd))
    mask = denom > 1e-8
    max_rel = np.max(diff[mask]/denom[mask]) if mask.any() else 0.0
    print("max |revad - fd| (absolute):", np.max(diff), " (genuinely nonzero gradient now, not ~0)")
    print("max relative error:", max_rel)

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
