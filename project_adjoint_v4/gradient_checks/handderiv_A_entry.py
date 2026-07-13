# -*- coding: utf-8 -*-
"""
HAND-DERIVED (not auto-traced) reverse-mode backward pass for a HULL/BODY
row's A[row,col] entry -- the composition step (Track B step 2, item 4 in
the plan). Not a new derivation from scratch: this is glue code that wires
together the three pieces already hand-derived and independently validated
(panel_geometry_one, hs_influence) exactly the way
trace_full_chain_A_entry_revad.py's A_entry_revad composes the oracle
pieces. Read that file's module docstring first (row/col dual role, the
mirror term, why the self-term is excluded) -- not repeated here.

THE FORMULA (from numba_kernels.py::assemble_A_b_vel_nb's hull-row loop,
also trace_full_chain_A_entry_revad.py::A_entry_revad):

  1) center_row, coordsys_row, _, _   = panel_geometry_one(row_corners)
  2) center_col, coordsys_col, cornerslocal_col, _ = panel_geometry_one(col_corners)
  3) normal_row      = coordsys_row[:,2]                    (3rd column)
  4) fieldpoint       = center_row
     fieldpoint_mirror = [center_row[0], -center_row[1], center_row[2]]
  5) v  = hs_influence(fieldpoint,        center_col, coordsys_col, cornerslocal_col)
     vp = hs_influence(fieldpoint_mirror, center_col, coordsys_col, cornerslocal_col)
  6) v_combined = [v[0]+vp[0], v[1]-vp[1], v[2]+vp[2]]        (even,odd,even in y)
  7) A_entry = dot(normal_row, v_combined)

Two things make this composition step nontrivial, beyond just calling the
two already-validated backward passes:

  - The COL panel's geometry (center_col, coordsys_col, cornerslocal_col)
    feeds hs_influence TWICE (once for v, once for vp) -- same "multi-use
    -> sum the paths" rule as everywhere else, just at the level of an
    entire sub-function's output instead of one scalar.
  - The ROW panel's geometry feeds TWO different downstream uses through
    TWO different mechanisms: center_row feeds BOTH fieldpoint (directly)
    AND fieldpoint_mirror (through a y-flip), and coordsys_row feeds
    ONLY its 3rd column (normal_row) -- columns 0,1 (t1,t2) get a ZERO
    adjoint here, since the no-penetration residual never uses the row
    panel's own tangent directions, only its normal. cornerslocal_row and
    area_row are unused entirely (zero adjoint, never touched).

BACKWARD walks 7 -> 1, each step just the adjoint rule for that
composition, then handing off to the two already-validated backward
passes for the parts inside them:
  7) dot(a,b): d_normal_row = v_combined*d_A, d_v_combined = normal_row*d_A
  6) v_combined[0]=v0+vp0 (+,+), [1]=v1-vp1 (+,-), [2]=v2+vp2 (+,+)
  5) hs_influence_backward(cache_v, d_v) and hs_influence_backward(cache_vp, d_vp)
     -- two independent calls into the already-validated hs_influence
     hand-derivation, no new math here
  4) fieldpoint=center_row (copy); fieldpoint_mirror=[+,-,+](center_row)
     -- center_row's total adjoint is the SUM of what comes back through
     BOTH paths
  3) normal_row=coordsys_row[:,2] -- scatter d_normal_row into column 2 of
     a (3,3) zero matrix, columns 0,1 stay exactly zero
  2,1) panel_geometry_one_backward(cache_col, ...) and
       panel_geometry_one_backward(cache_row, ...) -- two independent
       calls into the already-validated geometry hand-derivation
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from rankine_panel.io import read_panel_file
from rankine_panel.geometry import panel_geometry_all
from rankine_panel.revad import jacobian

from handderiv_panel_geometry import panel_geometry_one_forward, panel_geometry_one_backward
from handderiv_hs_influence import hs_influence_forward, hs_influence_backward

from trace_full_chain_A_entry_revad import A_entry_revad, pack, unpack_nodes, A_entry_primal


# =============================================================================
# Forward pass: composes panel_geometry_one_forward (x2) and
# hs_influence_forward (x2), caches everything the backward pass needs
# =============================================================================

def A_entry_forward(row_corners, col_corners):
    center_row, coordsys_row, cornerslocal_row, area_row, cache_row = panel_geometry_one_forward(row_corners)
    center_col, coordsys_col, cornerslocal_col, area_col, cache_col = panel_geometry_one_forward(col_corners)

    normal_row = coordsys_row[:, 2].copy()

    fieldpoint = center_row.copy()
    fieldpoint_mirror = np.array([center_row[0], -center_row[1], center_row[2]])

    v, cache_v = hs_influence_forward(fieldpoint, center_col, coordsys_col, cornerslocal_col)
    vp, cache_vp = hs_influence_forward(fieldpoint_mirror, center_col, coordsys_col, cornerslocal_col)

    v_combined = np.array([v[0] + vp[0], v[1] - vp[1], v[2] + vp[2]])
    A_entry = float(np.dot(normal_row, v_combined))

    cache = dict(cache_row=cache_row, cache_col=cache_col, cache_v=cache_v, cache_vp=cache_vp,
                 normal_row=normal_row, v_combined=v_combined)
    return A_entry, cache


# =============================================================================
# Backward pass: hand-derived, walks the forward steps in exact reverse
# order, delegating to the already-validated panel_geometry_one_backward /
# hs_influence_backward for everything inside those sub-functions.
# =============================================================================

def A_entry_backward(cache, d_A_entry):
    cache_row = cache['cache_row']
    cache_col = cache['cache_col']
    cache_v = cache['cache_v']
    cache_vp = cache['cache_vp']
    normal_row = cache['normal_row']
    v_combined = cache['v_combined']

    # ---- step 7: A_entry = dot(normal_row, v_combined)  ["z=dot(a,b)"]
    d_normal_row = v_combined * d_A_entry
    d_v_combined = normal_row * d_A_entry

    # ---- step 6: v_combined = [v0+vp0, v1-vp1, v2+vp2]
    d_v = np.array([d_v_combined[0], d_v_combined[1], d_v_combined[2]])
    d_vp = np.array([d_v_combined[0], -d_v_combined[1], d_v_combined[2]])

    # ---- step 5: two independent hs_influence backward passes (col panel
    # geometry feeds BOTH -- its adjoint is the SUM of both calls' results)
    d_fieldpoint_v, d_center_col_v, d_coordsys_col_v, d_cornerslocal_col_v = \
        hs_influence_backward(cache_v, d_v[0], d_v[1], d_v[2])
    d_fieldpoint_mirror_vp, d_center_col_vp, d_coordsys_col_vp, d_cornerslocal_col_vp = \
        hs_influence_backward(cache_vp, d_vp[0], d_vp[1], d_vp[2])

    d_center_col = d_center_col_v + d_center_col_vp
    d_coordsys_col = d_coordsys_col_v + d_coordsys_col_vp
    d_cornerslocal_col = d_cornerslocal_col_v + d_cornerslocal_col_vp

    # ---- step 4: fieldpoint=center_row (copy); fieldpoint_mirror=[+,-,+](center_row)
    # center_row's total adjoint sums BOTH paths (direct copy + y-flipped copy)
    d_center_row = d_fieldpoint_v.copy()
    d_center_row[0] += d_fieldpoint_mirror_vp[0]
    d_center_row[1] += -d_fieldpoint_mirror_vp[1]
    d_center_row[2] += d_fieldpoint_mirror_vp[2]

    # ---- step 3: normal_row = coordsys_row[:,2] -- only column 2 is used,
    # columns 0,1 (t1,t2) get EXACTLY zero adjoint here
    d_coordsys_row = np.zeros((3, 3))
    d_coordsys_row[:, 2] = d_normal_row

    # cornerslocal_row and area_row are never used by A_entry -> zero adjoint
    d_cornerslocal_row = np.zeros((2, 4))
    d_area_row = 0.0
    d_area_col = 0.0

    # ---- steps 2,1: hand off to the already-validated geometry backward pass
    d_row_c0, d_row_c1, d_row_c2, d_row_c3 = panel_geometry_one_backward(
        cache_row, d_center_row, d_coordsys_row, d_cornerslocal_row, d_area_row)
    d_col_c0, d_col_c1, d_col_c2, d_col_c3 = panel_geometry_one_backward(
        cache_col, d_center_col, d_coordsys_col, d_cornerslocal_col, d_area_col)

    return (d_row_c0, d_row_c1, d_row_c2, d_row_c3), (d_col_c0, d_col_c1, d_col_c2, d_col_c3)


# =============================================================================
# Validation: hand-derived backward pass vs. the trusted revad oracle
# (trace_full_chain_A_entry_revad.py::A_entry_revad via revad.jacobian),
# same real panel pair (ROW=0, COL=50, both hull, row!=col so the
# self-term substitution doesn't apply) used in that file.
# =============================================================================

if __name__ == "__main__":
    pf = read_panel_file(os.path.join(_PROJECT_ROOT, "examples", "fifi.dat"))

    ROW, COL = 0, 50
    row_vert_idx = pf.panels[:, ROW]
    col_vert_idx = pf.panels[:, COL]
    row_xyz0 = pf.points[:, row_vert_idx].T.copy()
    col_xyz0 = pf.points[:, col_vert_idx].T.copy()
    row_corners = [row_xyz0[k] for k in range(4)]
    col_corners = [col_xyz0[k] for k in range(4)]

    A_entry, cache = A_entry_forward(row_corners, col_corners)
    print("forward A_entry (hand):  ", A_entry)
    print("forward A_entry (primal):", A_entry_primal(pf.points, pf.panels, ROW, COL))

    def run_traced(xs_nodes):
        row_corners_n, col_corners_n = unpack_nodes(xs_nodes)
        return [A_entry_revad(row_corners_n, col_corners_n)]

    x0 = pack(row_xyz0, col_xyz0)
    J_oracle = jacobian(run_traced, x0)  # (1, 24)

    (d_row_c0, d_row_c1, d_row_c2, d_row_c3), (d_col_c0, d_col_c1, d_col_c2, d_col_c3) = \
        A_entry_backward(cache, 1.0)
    row_hand = np.concatenate([d_row_c0, d_row_c1, d_row_c2, d_row_c3])
    col_hand = np.concatenate([d_col_c0, d_col_c1, d_col_c2, d_col_c3])
    J_hand = np.concatenate([row_hand, col_hand])

    labels = (["row_c%d_%s" % (k, ax) for k in range(4) for ax in "xyz"]
              + ["col_c%d_%s" % (k, ax) for k in range(4) for ax in "xyz"])

    diff = np.abs(J_hand - J_oracle[0])
    max_abs_err = float(np.max(diff))
    i_worst = int(np.argmax(diff))

    print(f"\nmax |hand-derived - revad oracle| over all 24 entries: {max_abs_err:.3e}")
    print(f"worst: d(A)/d({labels[i_worst]})  hand={J_hand[i_worst]:.8e}  oracle={J_oracle[0,i_worst]:.8e}")
    for lab, hv, ov in zip(labels, J_hand, J_oracle[0]):
        flag = "  <-- worst" if abs(hv - ov) == max_abs_err else ""
        print(f"  {lab:<12} hand={hv:>16.8e}  oracle={ov:>16.8e}{flag}")
