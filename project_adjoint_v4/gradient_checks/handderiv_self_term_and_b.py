# -*- coding: utf-8 -*-
"""
HAND-DERIVED (not auto-traced) reverse-mode backward pass for the two
"free" pieces of the hull-row assembly: the diagonal self-term A[row,row]
and the b[row] vector. Same composition idea as handderiv_A_entry.py, but
simpler -- neither piece needs hs_influence or phixx_influence at all, both
depend on the row panel's vertices ONLY through its own normal
(panel_geometry_one's coordsys output, 3rd column). See
trace_self_term_and_b_revad.py's module docstring for the full reasoning
(not repeated in full here); this file is the hand-derived counterpart of
that oracle, same real panel (ROW=0), same formulas.

SELF-TERM (hull diagonal, row==col), from numba_kernels.py::assemble_A_b_vel_nb:
    A[row,row] = -0.5 * dot(normal_row, normal_row)

normal_row is UNIT LENGTH by construction (panel_geometry_all's nv is
explicitly normalized: nv = nv_raw/nv_norm) -- so dot(normal_row,normal_row)
= 1 IDENTICALLY for any non-degenerate panel, and A[row,row] = -0.5*1 = -0.5
is a CONSTANT, structurally, not from a numerical coincidence. Its gradient
w.r.t. vertices should come out essentially exactly zero -- confirmed
below, not just asserted (same discipline as the oracle file).

B VECTOR (hull rows):
    b[row] = normal_row[0] * vinf_x        (vinf_x: a constant, not geometry)

NOT structurally constant -- the x-COMPONENT of a unit vector genuinely
varies with panel orientation even though the vector's magnitude doesn't.
So d(b[row])/d(vertices) = vinf_x * d(normal_row_x)/d(row vertices), real
and nonzero in general.

FS rows of b are the literal constant 0 (see numba_kernels.py) -- zero
dependence on anything, not worth deriving.

Both pieces are pure glue on top of the already-validated
panel_geometry_one_forward/backward -- no new adjoint-rule-table entries,
just: (a) zero out d_cornerslocal and d_area (neither piece touches them),
(b) scatter the one nonzero normal-component adjoint into the right column
of d_coordsys (column 2 = normal; only entry [.,2] can ever be nonzero for
either formula), (c) hand off to panel_geometry_one_backward.
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from rankine_panel.io import read_panel_file
from rankine_panel.revad import jacobian

from handderiv_panel_geometry import panel_geometry_one_forward, panel_geometry_one_backward

from trace_self_term_and_b_revad import (
    self_term_A_revad, b_row_revad, pack, unpack_nodes,
    self_term_A_primal, b_row_primal,
)


# =============================================================================
# Self-term: A[row,row] = -0.5 * dot(normal_row, normal_row)
# =============================================================================

def self_term_A_forward(row_corners):
    center_row, coordsys_row, cornerslocal_row, area_row, cache_row = panel_geometry_one_forward(row_corners)
    normal_row = coordsys_row[:, 2].copy()
    A_val = float(-0.5 * np.dot(normal_row, normal_row))
    cache = dict(cache_row=cache_row, normal_row=normal_row)
    return A_val, cache


def self_term_A_backward(cache, d_A):
    cache_row = cache['cache_row']
    normal_row = cache['normal_row']

    # ---- A = -0.5 * dot(normal_row, normal_row)  ["z=dot(a,a)": d_a = 2a*dz]
    d_normal_row = -normal_row * d_A          # -0.5 * 2 * normal_row * d_A

    d_coordsys_row = np.zeros((3, 3))
    d_coordsys_row[:, 2] = d_normal_row       # only the normal column is ever touched

    d_center_row = np.zeros(3)
    d_cornerslocal_row = np.zeros((2, 4))
    d_area_row = 0.0

    return panel_geometry_one_backward(cache_row, d_center_row, d_coordsys_row, d_cornerslocal_row, d_area_row)


# =============================================================================
# b[row] = normal_row_x * vinf_x  (hull rows)
# =============================================================================

def b_row_forward(row_corners, vinf_x):
    center_row, coordsys_row, cornerslocal_row, area_row, cache_row = panel_geometry_one_forward(row_corners)
    normal_row_x = coordsys_row[0, 2]
    b_val = float(normal_row_x * vinf_x)
    cache = dict(cache_row=cache_row, vinf_x=vinf_x)
    return b_val, cache


def b_row_backward(cache, d_b):
    cache_row = cache['cache_row']
    vinf_x = cache['vinf_x']

    # ---- b = normal_row_x * vinf_x  (vinf_x constant)  ["z=x*c": dx=c*dz]
    d_normal_row_x = vinf_x * d_b

    d_coordsys_row = np.zeros((3, 3))
    d_coordsys_row[0, 2] = d_normal_row_x     # only coordsys[0,2] (normal's x-component) is touched

    d_center_row = np.zeros(3)
    d_cornerslocal_row = np.zeros((2, 4))
    d_area_row = 0.0

    return panel_geometry_one_backward(cache_row, d_center_row, d_coordsys_row, d_cornerslocal_row, d_area_row)


# =============================================================================
# Validation: hand-derived backward passes vs. the trusted revad oracle
# (trace_self_term_and_b_revad.py), same real panel (ROW=0, a hull panel).
# =============================================================================

if __name__ == "__main__":
    pf = read_panel_file(os.path.join(_PROJECT_ROOT, "examples", "fifi.dat"))

    ROW = 0
    row_vert_idx = pf.panels[:, ROW]
    row_xyz0 = pf.points[:, row_vert_idx].T.copy()
    row_corners = [row_xyz0[k] for k in range(4)]

    Fr, gravity, length = 0.3, 9.80665, 1.0
    vinf_x = Fr * np.sqrt(length * gravity)

    labels = ["c%d_%s" % (k, ax) for k in range(4) for ax in "xyz"]

    # ---------------- self-term ----------------
    print(f"=== Self-term A[row={ROW},row={ROW}] ===")
    A_val, cache_A = self_term_A_forward(row_corners)
    print("forward A (hand):  ", A_val, " (expect exactly -0.5)")
    print("forward A (primal):", self_term_A_primal(pf.points, pf.panels, ROW))

    x0 = pack(row_xyz0)
    J_oracle_A = jacobian(lambda xs: [self_term_A_revad(unpack_nodes(xs))], x0)[0]  # (12,)
    d_c0, d_c1, d_c2, d_c3 = self_term_A_backward(cache_A, 1.0)
    J_hand_A = np.concatenate([d_c0, d_c1, d_c2, d_c3])

    err_A = np.max(np.abs(J_hand_A - J_oracle_A))
    print("max |hand|:  ", np.max(np.abs(J_hand_A)), " (expect ~0, structurally constant)")
    print("max |oracle|:", np.max(np.abs(J_oracle_A)), " (expect ~0 too)")
    print(f"max |hand - revad oracle|: {err_A:.3e}")

    # ---------------- b[row] ----------------
    print(f"\n=== b[row={ROW}] ===")
    b_val, cache_b = b_row_forward(row_corners, vinf_x)
    print("forward b (hand):  ", b_val)
    print("forward b (primal):", b_row_primal(pf.points, pf.panels, ROW, vinf_x))

    J_oracle_b = jacobian(lambda xs: [b_row_revad(unpack_nodes(xs), vinf_x)], x0)[0]  # (12,)
    d_c0, d_c1, d_c2, d_c3 = b_row_backward(cache_b, 1.0)
    J_hand_b = np.concatenate([d_c0, d_c1, d_c2, d_c3])

    err_b = np.max(np.abs(J_hand_b - J_oracle_b))
    i_worst = int(np.argmax(np.abs(J_hand_b - J_oracle_b)))
    print(f"max |hand - revad oracle|: {err_b:.3e}")
    print(f"  worst: d(b)/d({labels[i_worst]})  hand={J_hand_b[i_worst]:.8e}  oracle={J_oracle_b[i_worst]:.8e}")
