# -*- coding: utf-8 -*-
"""
Compute:
    d(Hessian) / d(vertices)

This is an Oracle:
    pure Python, builds a full object graph, does dynamic dispatch through operator overloading

Combined reverse-mode AD oracle: raw vertex positions -> panel_geometry_all
-> phixx_influence, in ONE Node graph. Same treatment as
trace_full_chain_revad.py (see that file for the fuller explanation of how
composition works), just with phixx_influence in place of hs_influence --
this is the piece needed for the free-surface rows of A (A[row,col] =
U2*phi_xx + gravity*v2 there), where phi_xx comes from the Hessian kernel
rather than the velocity kernel.

phixx_influence is now the single generic function (see influence.py) used
for both this AD trace and the FD reference below -- no separate primal
import needed.

Output is the 3x3 global Hessian (9 scalars), so this needs 9 backward
passes instead of hs_influence's 3, still independent of the 24 inputs.
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
from rankine_panel.influence import phixx_influence



# =============================================================================
# Combined trace: vertex positions (row panel + col panel) -> Hessian
# =============================================================================

def combined_phixx_revad(row_corners, col_corners):
    """
    row_corners, col_corners: each [c0,c1,c2,c3], length-3 Node lists
    returns: 3x3 nested list of Node (global Hessian)
    """
    center_row, _coordsys_row, _cornerslocal_row, _area_row = panel_geometry_one(row_corners)
    center_col, coordsys_col, cornerslocal_col, _area_col = panel_geometry_one(col_corners)

    fieldpoint = center_row
    H = phixx_influence(fieldpoint, center_col, coordsys_col, cornerslocal_col)
    return H


# =============================================================================
# Same 24-scalar packing as trace_full_chain_revad.py
# =============================================================================

def pack(row_xyz, col_xyz):
    return np.concatenate([np.asarray(row_xyz, dtype=np.float64).ravel(),
                            np.asarray(col_xyz, dtype=np.float64).ravel()])


def unpack_nodes(xs):
    row_corners = [[xs[3*k + i] for i in range(3)] for k in range(4)]
    col_corners = [[xs[12 + 3*k + i] for i in range(3)] for k in range(4)]
    return row_corners, col_corners


def run_traced(xs_nodes):
    row_corners, col_corners = unpack_nodes(xs_nodes)
    H = combined_phixx_revad(row_corners, col_corners)
    return [H[a][b] for a in range(3) for b in range(3)]  # flatten row-major, 9 entries


if __name__ == "__main__":
    pf = read_panel_file(os.path.join(_PROJECT_ROOT, "examples", "fifi.dat"))

    ROW, COL = 0, 50
    row_vert_idx = pf.panels[:, ROW]
    col_vert_idx = pf.panels[:, COL]
    row_xyz0 = pf.points[:, row_vert_idx].T.copy()
    col_xyz0 = pf.points[:, col_vert_idx].T.copy()

    x0 = pack(row_xyz0, col_xyz0)

    print(f"Tracing full chain (vertices -> geometry -> Hessian) for row panel {ROW}, col panel {COL}")

    geom0 = panel_geometry_all(pf.points, pf.panels)
    H0 = phixx_influence(geom0.center[:, ROW], geom0.center[:, COL],
                                 geom0.coordsys[:, :, COL], geom0.cornerslocal[:, :, COL])
    print("primal Hessian (3x3):\n", H0)

    J_revad = jacobian(run_traced, x0)  # (9, 24)

    eps = 1e-6
    N = x0.size
    J_fd = np.zeros((9, N), dtype=np.float64)
    for i in range(N):
        xp = x0.copy(); xp[i] += eps
        xm = x0.copy(); xm[i] -= eps

        points_p = pf.points.copy()
        points_m = pf.points.copy()
        points_p[:, row_vert_idx] = xp[0:12].reshape(4, 3).T
        points_p[:, col_vert_idx] = xp[12:24].reshape(4, 3).T
        points_m[:, row_vert_idx] = xm[0:12].reshape(4, 3).T
        points_m[:, col_vert_idx] = xm[12:24].reshape(4, 3).T

        gp = panel_geometry_all(points_p, pf.panels)
        gm = panel_geometry_all(points_m, pf.panels)

        Hp = phixx_influence(gp.center[:, ROW], gp.center[:, COL],
                                     gp.coordsys[:, :, COL], gp.cornerslocal[:, :, COL]).ravel()
        Hm = phixx_influence(gm.center[:, ROW], gm.center[:, COL],
                                     gm.coordsys[:, :, COL], gm.cornerslocal[:, :, COL]).ravel()
        J_fd[:, i] = (Hp - Hm) / (2.0 * eps)

    in_labels = (["row_c%d_%s" % (k, ax) for k in range(4) for ax in "xyz"]
                 + ["col_c%d_%s" % (k, ax) for k in range(4) for ax in "xyz"])
    out_labels = ["H%d%d" % (a, b) for a in range(3) for b in range(3)]

    max_abs_err = 0.0
    worst_abs = None
    max_rel_err = 0.0
    worst_rel = None
    for i, ilab in enumerate(in_labels):
        for o, olab in enumerate(out_labels):
            ad_v, fd_v = J_revad[o, i], J_fd[o, i]
            err = abs(ad_v - fd_v)
            if err > max_abs_err:
                max_abs_err = err
                worst_abs = (ilab, olab, ad_v, fd_v)
            denom = max(abs(ad_v), abs(fd_v))
            if denom > 1e-8:
                rel = err / denom
                if rel > max_rel_err:
                    max_rel_err = rel
                    worst_rel = (ilab, olab, ad_v, fd_v, rel)

    print(f"\nJacobian shape: revad {J_revad.shape}, fd {J_fd.shape}")
    print(f"max |revad - fd| (absolute) over all 9x24 entries: {max_abs_err:.3e}")
    print(f"  worst-abs entry: d{worst_abs[1]}/d{worst_abs[0]}  revad={worst_abs[2]:.8e}  fd={worst_abs[3]:.8e}")
    print(f"max relative error (entries with |value|>1e-8): {max_rel_err:.3e}")
    print(f"  worst-rel entry: d{worst_rel[1]}/d{worst_rel[0]}  revad={worst_rel[2]:.8e}  "
          f"fd={worst_rel[3]:.8e}  rel={worst_rel[4]:.3e}")
