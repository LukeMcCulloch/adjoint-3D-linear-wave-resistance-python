# -*- coding: utf-8 -*-
"""
HAND-DERIVED (not auto-traced) reverse-mode backward pass for a
FREE-SURFACE row's A[row,col] entry. Same idea as handderiv_A_entry.py
(pure composition of already-validated pieces, no new per-operation math),
but the composition itself is genuinely different from the hull-row case
-- read trace_full_chain_fs_row_revad.py's module docstring first (the
z=0 field-point override, why it's a structural disconnect and not just a
"small" value) if you haven't already; not repeated in full here.

THE FORMULA (from numba_kernels.py::assemble_A_b_vel_nb's FS-row loop,
also trace_full_chain_fs_row_revad.py::A_entry_fs_row_revad):

  1) center_row, _, _, _              = panel_geometry_one(row_corners)
     center_col, coordsys_col, cornerslocal_col, _ = panel_geometry_one(col_corners)
  2) p0 = center_row[0] + deltax   (deltax: constant, not a design var)
     p1 = center_row[1]
     p2 = 0.0                      (LITERAL constant -- NOT center_row[2])
     pp0 = p0                      (reuses the SAME p0, not a recompute)
     pp1 = -center_row[1]
     pp2 = 0.0                     (literal constant)
     fieldpoint        = [p0,  p1,  p2]
     fieldpoint_mirror  = [pp0, pp1, pp2]
  3) v  = hs_influence(fieldpoint,        center_col, coordsys_col, cornerslocal_col)
     vp = hs_influence(fieldpoint_mirror, center_col, coordsys_col, cornerslocal_col)
     v2 = v[2] + vp[2]
  4) H  = phixx_influence(fieldpoint,        center_col, coordsys_col, cornerslocal_col)
     Hp = phixx_influence(fieldpoint_mirror, center_col, coordsys_col, cornerslocal_col)
     phi_xx = H[0][0] + Hp[0][0]
  5) A_entry = phi_xx*U2 + v2*gravity      (U2, gravity: constants)

WHAT'S DIFFERENT FROM THE HULL-ROW COMPOSITION (handderiv_A_entry.py),
worth being deliberate about since it's an easy place to get sloppy:

  - The row panel's `coordsys_row`, `cornerslocal_row`, `area_row` are
    NOT used AT ALL here (no normal-dot-product step for a free-surface
    condition) -- their adjoints are all exactly zero, not "small."
  - `center_row[0]` feeds p0, and pp0 REUSES p0 itself (not a fresh
    `center_row[0]-something` expression) -- so p0's total adjoint (from
    BOTH the real-fieldpoint path through v/H AND the mirror-fieldpoint
    path through vp/Hp, since pp0=p0 literally) flows into
    `center_row[0]` as ONE combined quantity, not two separately-scaled
    contributions the way center_row[1] gets below.
  - `center_row[1]` feeds p1 directly (+1 coefficient) AND pp1=-center_row[1]
    (-1 coefficient) as TWO SEPARATE expressions -- so its adjoint is
    `d(from fieldpoint's p1 path) - d(from fieldpoint_mirror's pp1 path)`,
    the ordinary mirror-term sign flip seen elsewhere in this series.
  - `center_row[2]` gets NOTHING: p2 and pp2 are both literal 0.0 with no
    parents, so no matter what hs_influence_backward/phixx_influence_backward
    compute for "d_fieldpoint[2]" internally, that value has nowhere to
    flow -- it's simply never connected to center_row[2]. This is the
    hand-derivation's version of the oracle's `Node(0.0,[])` trick: the
    VALUE 0.0 is real and used forward, but the ADJOINT path is severed
    by construction, not because the number happens to be small.
  - The col panel's geometry now feeds FOUR sub-calls (v, vp, H, Hp), not
    two -- its adjoint sums all four (same "multi-use" rule, more terms).
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
from handderiv_hs_influence import hs_influence_forward, hs_influence_backward
from handderiv_phixx_influence import phixx_influence_forward, phixx_influence_backward

from trace_full_chain_fs_row_revad import A_entry_fs_row_revad, pack, unpack_nodes, A_entry_fs_row_primal


# =============================================================================
# Forward pass: composes panel_geometry_one_forward (x2), hs_influence_forward
# (x2), phixx_influence_forward (x2)
# =============================================================================

def A_entry_fs_row_forward(row_corners, col_corners, deltax, U2, gravity):
    center_row, coordsys_row, cornerslocal_row, area_row, cache_row = panel_geometry_one_forward(row_corners)
    center_col, coordsys_col, cornerslocal_col, area_col, cache_col = panel_geometry_one_forward(col_corners)

    p0 = center_row[0] + deltax
    p1 = center_row[1]
    p2 = 0.0

    pp0 = p0
    pp1 = -center_row[1]
    pp2 = 0.0

    fieldpoint = np.array([p0, p1, p2])
    fieldpoint_mirror = np.array([pp0, pp1, pp2])

    v, cache_v = hs_influence_forward(fieldpoint, center_col, coordsys_col, cornerslocal_col)
    vp, cache_vp = hs_influence_forward(fieldpoint_mirror, center_col, coordsys_col, cornerslocal_col)
    v2 = v[2] + vp[2]

    H, cache_H = phixx_influence_forward(fieldpoint, center_col, coordsys_col, cornerslocal_col)
    Hp, cache_Hp = phixx_influence_forward(fieldpoint_mirror, center_col, coordsys_col, cornerslocal_col)
    phi_xx = H[0, 0] + Hp[0, 0]

    A_entry = float(U2*phi_xx + gravity*v2)

    cache = dict(cache_row=cache_row, cache_col=cache_col,
                 cache_v=cache_v, cache_vp=cache_vp, cache_H=cache_H, cache_Hp=cache_Hp,
                 U2=U2, gravity=gravity)
    return A_entry, cache


# =============================================================================
# Backward pass: hand-derived, walks the forward steps in exact reverse
# order, delegating to the already-validated backward passes for everything
# inside those sub-functions.
# =============================================================================

def A_entry_fs_row_backward(cache, d_A_entry):
    cache_row = cache['cache_row']
    cache_col = cache['cache_col']
    cache_v = cache['cache_v']
    cache_vp = cache['cache_vp']
    cache_H = cache['cache_H']
    cache_Hp = cache['cache_Hp']
    U2 = cache['U2']
    gravity = cache['gravity']

    # ---- step 5: A_entry = phi_xx*U2 + v2*gravity  (U2, gravity constants)
    d_phi_xx = U2 * d_A_entry
    d_v2 = gravity * d_A_entry

    # ---- step 4: phi_xx = H[0,0] + Hp[0,0]
    d_out_H = np.zeros((3, 3)); d_out_H[0, 0] = d_phi_xx
    d_out_Hp = np.zeros((3, 3)); d_out_Hp[0, 0] = d_phi_xx
    d_fieldpoint_H, d_center_col_H, d_coordsys_col_H, d_cornerslocal_col_H = \
        phixx_influence_backward(cache_H, d_out_H)
    d_fieldpoint_mirror_Hp, d_center_col_Hp, d_coordsys_col_Hp, d_cornerslocal_col_Hp = \
        phixx_influence_backward(cache_Hp, d_out_Hp)

    # ---- step 3: v2 = v[2] + vp[2]  (only the z-component of v,vp is used)
    d_fieldpoint_v, d_center_col_v, d_coordsys_col_v, d_cornerslocal_col_v = \
        hs_influence_backward(cache_v, 0.0, 0.0, d_v2)
    d_fieldpoint_mirror_vp, d_center_col_vp, d_coordsys_col_vp, d_cornerslocal_col_vp = \
        hs_influence_backward(cache_vp, 0.0, 0.0, d_v2)

    # col panel geometry feeds FOUR sub-calls (v, vp, H, Hp) -- sum all four
    d_center_col = d_center_col_v + d_center_col_vp + d_center_col_H + d_center_col_Hp
    d_coordsys_col = d_coordsys_col_v + d_coordsys_col_vp + d_coordsys_col_H + d_coordsys_col_Hp
    d_cornerslocal_col = d_cornerslocal_col_v + d_cornerslocal_col_vp + d_cornerslocal_col_H + d_cornerslocal_col_Hp

    # ---- step 2: fieldpoint=[p0,p1,p2], fieldpoint_mirror=[pp0,pp1,pp2]
    # fieldpoint's total adjoint = sum of what v's and H's backward gave it;
    # fieldpoint_mirror's = sum of what vp's and Hp's backward gave it
    d_field = d_fieldpoint_v + d_fieldpoint_H              # [d_p0, d_p1, d_p2]
    d_field_mirror = d_fieldpoint_mirror_vp + d_fieldpoint_mirror_Hp  # [d_pp0, d_pp1, d_pp2]

    d_center_row = np.zeros(3)
    # p0 = center_row[0]+deltax ; pp0 = p0 (the SAME node, reused) -- both
    # paths' adjoint land on p0 itself, which flows straight through the
    # constant shift into center_row[0]
    d_center_row[0] = d_field[0] + d_field_mirror[0]
    # p1 = center_row[1] (+1) ; pp1 = -center_row[1] (-1) -- two SEPARATE
    # expressions of the same variable, ordinary mirror sign flip
    d_center_row[1] = d_field[1] - d_field_mirror[1]
    # p2 = 0.0, pp2 = 0.0 -- both literal constants, no parents, so
    # d_field[2]/d_field_mirror[2] have nowhere to go; center_row[2] stays
    # exactly 0.0 (structural, not a "should be small" approximation)

    # row panel's coordsys/cornerslocal/area are never used by this formula
    d_coordsys_row = np.zeros((3, 3))
    d_cornerslocal_row = np.zeros((2, 4))
    d_area_row = 0.0
    d_area_col = 0.0

    d_row_c0, d_row_c1, d_row_c2, d_row_c3 = panel_geometry_one_backward(
        cache_row, d_center_row, d_coordsys_row, d_cornerslocal_row, d_area_row)
    d_col_c0, d_col_c1, d_col_c2, d_col_c3 = panel_geometry_one_backward(
        cache_col, d_center_col, d_coordsys_col, d_cornerslocal_col, d_area_col)

    return (d_row_c0, d_row_c1, d_row_c2, d_row_c3), (d_col_c0, d_col_c1, d_col_c2, d_col_c3)


# =============================================================================
# Validation: hand-derived backward pass vs. the trusted revad oracle
# (trace_full_chain_fs_row_revad.py::A_entry_fs_row_revad via revad.jacobian),
# same real panel pair (ROW=300, a free-surface panel; COL=50, a hull panel)
# used in that file.
# =============================================================================

if __name__ == "__main__":
    pf = read_panel_file(os.path.join(_PROJECT_ROOT, "examples", "fifi.dat"))

    Fr = 0.3
    gravity = 9.80665
    length = 1.0
    U = Fr * np.sqrt(length * gravity)
    U2 = U * U
    deltax = pf.deltax

    ROW, COL = 300, 50
    assert ROW >= pf.npanels, "ROW must be a free-surface panel"

    row_vert_idx = pf.panels[:, ROW]
    col_vert_idx = pf.panels[:, COL]
    row_xyz0 = pf.points[:, row_vert_idx].T.copy()
    col_xyz0 = pf.points[:, col_vert_idx].T.copy()
    row_corners = [row_xyz0[k] for k in range(4)]
    col_corners = [col_xyz0[k] for k in range(4)]

    A_entry, cache = A_entry_fs_row_forward(row_corners, col_corners, deltax, U2, gravity)
    print("forward A_entry (hand):  ", A_entry)
    print("forward A_entry (primal):", A_entry_fs_row_primal(pf.points, pf.panels, ROW, COL, deltax, U2, gravity))

    def run_traced(xs_nodes):
        row_corners_n, col_corners_n = unpack_nodes(xs_nodes)
        return [A_entry_fs_row_revad(row_corners_n, col_corners_n, deltax, U2, gravity)]

    x0 = pack(row_xyz0, col_xyz0)
    J_oracle = jacobian(run_traced, x0)  # (1, 24)

    (d_row_c0, d_row_c1, d_row_c2, d_row_c3), (d_col_c0, d_col_c1, d_col_c2, d_col_c3) = \
        A_entry_fs_row_backward(cache, 1.0)
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

    print("\nsanity: d(A)/d(row_c*_z) columns (expect EXACTLY 0.0, structurally):")
    for k in range(4):
        idx = 3*k + 2
        print(f"  row_c{k}_z: hand={J_hand[idx]:+.3e}  oracle={J_oracle[0,idx]:+.3e}")
