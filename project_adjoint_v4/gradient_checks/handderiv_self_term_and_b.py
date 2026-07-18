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

CORRECTED 2026-07-15 -- see trace_self_term_and_b_revad.py's module
docstring for the full story (not repeated in full here): the self-term
is NOT `A[row,row] = -0.5*dot(normal_row,normal_row)` (a structural
constant) as originally believed and validated against itself -- reading
numba_kernels.py::assemble_A_b_vel_nb directly showed the mirror term is
computed and added UNCONDITIONALLY even when row==col, so the true
formula is:

SELF-TERM (hull diagonal, row==col), from numba_kernels.py::assemble_A_b_vel_nb:
    v  = -0.5 * normal_row                          (closed form; no
                                                       hs_influence call --
                                                       would be singular)
    vp = hs_influence(y-flip(center_row), center_row, coordsys_row,
                       cornerslocal_row)              (mirror term, ALWAYS
                                                        computed, using the
                                                        row panel's OWN
                                                        geometry as "col"
                                                        since col==row)
    v_combined = [v[0]+vp[0], v[1]-vp[1], v[2]+vp[2]]
    A[row,row] = dot(normal_row, v_combined)

Confirmed directly against the real numba-jitted assemble_A_b_vel_nb on
fifi.dat panel 0: -0.9345545330215205, not -0.5. Genuinely nonzero
gradient now, same character as b[row] below -- NOT structurally constant.

This composition is genuinely richer than a normal hull-row A_entry
(handderiv_A_entry.py): row_corners feeds hs_influence's fieldpoint AND
center AND coordsys AND cornerslocal arguments SIMULTANEOUSLY, since the
panel plays both the row role and the col role at once -- every one of
those four paths' adjoints has to be summed into row_corners' total
gradient, same "multi-use -> sum the paths" rule as everywhere else in
this series, just with more paths landing on the same panel than usual.

B VECTOR (hull rows):
    b[row] = normal_row[0] * vinf_x        (vinf_x: a constant, not geometry)

Not structurally constant -- the x-COMPONENT of a unit vector genuinely
varies with panel orientation even though the vector's magnitude doesn't.
So d(b[row])/d(vertices) = vinf_x * d(normal_row_x)/d(row vertices), real
and nonzero in general. This one was correct from the start -- b's
assembly loop has no mirror-term logic at all (confirmed by direct
reading of numba_kernels.py), so nothing to fix here.

FS rows of b are the literal constant 0 (see numba_kernels.py) -- zero
dependence on anything, not worth deriving.

self_term_A_forward/_backward compose the already-validated
panel_geometry_one_forward/backward AND hs_influence_forward/backward (no
new adjoint-rule-table entries); b_row_forward/_backward are pure glue on
panel_geometry_one alone, unchanged from before this fix.

WHY THE __main__ FD CHECK BELOW SHOWS ~1e-2 DISAGREEMENT (verified NOT a
bug, 2026-07-15): panel 0's own mirror-self-influence call happens to hit
`edx == 0.0` EXACTLY on one edge (two of panel 0's corners share the same
local xi coordinate -- a genuine, if unremarkable, coincidence of this
specific panel's shape), which trips hs_influence's `m_is_const` branch
(see handderiv_hs_influence.py). FD's central-difference probes, and a
complex-step probe tried during debugging, BOTH perturb that exact-zero
boundary and land on the *other* branch for at least one of their two
evaluations -- the same "FD/any-perturbation-method cannot see an exact
branch boundary" issue already documented in handderiv_hs_influence.py's
BRANCH COVERAGE section, just surfacing here through a real mesh panel
instead of a constructed synthetic case. (Complex-step is doubly affected:
`abs(edx + h*1j) = h` for edx==0 exactly, so even an "infinitesimal"
imaginary step of 1e-20 pushes the guard's `abs(...)` value from 0 up past
the 1e-30 threshold, flipping the branch that a genuine real perturbation
never would.) The trustworthy check is hand vs. the revad oracle (which
branches on `.val`, immune to this): they agree to 3.4e-15, confirmed
during debugging by directly comparing complex-step against BOTH
panel_geometry_one_backward (dense random seed, exact match) and
hs_influence_backward (isolated d_fieldpoint/d_center outputs at this
exact data point, exact match) before finally isolating the mismatch to
this one branch-boundary coincidence in d_cornerslocal. Nothing here needed
fixing; recorded so this doesn't get re-litigated from scratch later.
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

from trace_self_term_and_b_revad import (
    self_term_A_revad, b_row_revad, pack, unpack_nodes,
    self_term_A_primal, b_row_primal,
)


# =============================================================================
# Self-term: A[row,row] = dot(normal_row, [-0.5*normal_row + mirror-combined])
# =============================================================================

def self_term_A_forward(row_corners):
    center_row, coordsys_row, cornerslocal_row, area_row, cache_row = panel_geometry_one_forward(row_corners)
    normal_row = coordsys_row[:, 2].copy()

    v = -0.5 * normal_row

    fieldpoint_mirror = np.array([center_row[0], -center_row[1], center_row[2]])
    vp, cache_vp = hs_influence_forward(fieldpoint_mirror, center_row, coordsys_row, cornerslocal_row)

    v_combined = np.array([v[0] + vp[0], v[1] - vp[1], v[2] + vp[2]])
    A_val = float(np.dot(normal_row, v_combined))

    cache = dict(cache_row=cache_row, cache_vp=cache_vp, normal_row=normal_row, v_combined=v_combined)
    return A_val, cache


def self_term_A_backward(cache, d_A):
    cache_row = cache['cache_row']
    cache_vp = cache['cache_vp']
    normal_row = cache['normal_row']
    v_combined = cache['v_combined']

    # ---- A = dot(normal_row, v_combined)  ["z=dot(a,b)"]
    d_normal_row = v_combined * d_A            # normal_row's FIRST contribution (2 more below)
    d_v_combined = normal_row * d_A

    # ---- v_combined = [v0+vp0, v1-vp1, v2+vp2]
    d_v = np.array([d_v_combined[0], d_v_combined[1], d_v_combined[2]])
    d_vp = np.array([d_v_combined[0], -d_v_combined[1], d_v_combined[2]])

    # ---- v = -0.5 * normal_row
    d_normal_row = d_normal_row + (-0.5) * d_v   # normal_row's SECOND contribution

    # ---- vp = hs_influence(fieldpoint_mirror, center_row, coordsys_row, cornerslocal_row)
    # row_corners feeds THIS call through FOUR separate arguments at once (row
    # plays both row and col here) -- every one of the four returned adjoints
    # is a real, independent path that must be summed in.
    d_fieldpoint_mirror, d_center_row_as_center_arg, d_coordsys_row_as_coordsys_arg, d_cornerslocal_row = \
        hs_influence_backward(cache_vp, d_vp[0], d_vp[1], d_vp[2])

    # ---- fieldpoint_mirror = [center_row[0], -center_row[1], center_row[2]]
    d_center_row_as_fieldpoint = np.array(
        [d_fieldpoint_mirror[0], -d_fieldpoint_mirror[1], d_fieldpoint_mirror[2]])

    # center_row's TOTAL adjoint: sums BOTH paths (as hs_influence's "center"
    # argument directly, AND via the mirrored "fieldpoint" argument)
    d_center_row = d_center_row_as_fieldpoint + d_center_row_as_center_arg

    # coordsys_row's TOTAL adjoint: normal_row's contributions (scattered into
    # column 2) PLUS hs_influence's "coordsys" argument contribution
    d_coordsys_row = np.zeros((3, 3))
    d_coordsys_row[:, 2] = d_normal_row
    d_coordsys_row = d_coordsys_row + d_coordsys_row_as_coordsys_arg

    # cornerslocal_row's adjoint: purely from hs_influence's "cornerslocal" argument
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
    print("forward A (hand):  ", A_val, " (NOT -0.5 -- includes the mirror term)")
    print("forward A (primal):", self_term_A_primal(pf.points, pf.panels, ROW))

    x0 = pack(row_xyz0)
    J_oracle_A = jacobian(lambda xs: [self_term_A_revad(unpack_nodes(xs))], x0)[0]  # (12,)
    d_c0, d_c1, d_c2, d_c3 = self_term_A_backward(cache_A, 1.0)
    J_hand_A = np.concatenate([d_c0, d_c1, d_c2, d_c3])

    err_A = np.max(np.abs(J_hand_A - J_oracle_A))
    print(f"max |hand - revad oracle|: {err_A:.3e}")

    # Third check: FD directly on self_term_A_primal. This one WILL show
    # ~1e-2 disagreement, not machine precision -- that's expected, not a
    # bug, see the module docstring's "WHY THE __main__ FD CHECK BELOW
    # SHOWS ~1e-2 DISAGREEMENT" section: panel 0's own mirror-self-influence
    # call hits an exact edx==0.0 branch boundary on one edge, which FD's
    # perturbations cannot see correctly (same issue as
    # handderiv_hs_influence.py's branch-coverage findings). The
    # hand-vs-oracle check above (agreeing to 3.4e-15) is the trustworthy one.
    eps = 1e-6
    J_fd_A = np.zeros(12)
    for i in range(12):
        xp = x0.copy(); xp[i] += eps
        xm = x0.copy(); xm[i] -= eps
        points_p = pf.points.copy(); points_p[:, row_vert_idx] = xp.reshape(4, 3).T
        points_m = pf.points.copy(); points_m[:, row_vert_idx] = xm.reshape(4, 3).T
        J_fd_A[i] = (self_term_A_primal(points_p, pf.panels, ROW) - self_term_A_primal(points_m, pf.panels, ROW)) / (2*eps)
    err_A_fd = np.max(np.abs(J_hand_A - J_fd_A))
    print(f"max |hand - fd|:           {err_A_fd:.3e}  (expected ~1e-2, see docstring -- not a bug)")

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
