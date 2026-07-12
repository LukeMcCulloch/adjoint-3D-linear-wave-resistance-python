# -*- coding: utf-8 -*-
"""

Computes: 
    d(velocity) / d(vertices)

This is an Oracle:
    pure Python, builds a full object graph, does dynamic dispatch through operator overloading

Combined reverse-mode AD oracle: raw vertex positions -> panel_geometry_all
-> hs_influence, in ONE Node graph. Composes the two already-validated
oracles (trace_panel_geometry_revad.py, trace_hs_influence_revad.py) rather
than re-deriving anything -- this is the artifact that actually matters for
the eventual per-vertex dA/dm, since it mirrors how the real assembly calls
these functions together.

In the production assembly (numba_kernels.py::assemble_A_b_vel_nb), for a
body-BC row/col pair:

    fieldpoint = center[row]                              (row panel's own center)
    v = hs_influence_nb(fieldpoint, center[col], coordsys[col], cornerslocal[col])
    A[row, col] = dot(normal[row], v)   (mirror term omitted here, see note below)

So fieldpoint is NOT an independent point -- it's itself the output of
panel_geometry_all applied to the ROW panel's own 4 vertices. This oracle
traces vertex positions of BOTH the row panel (4 verts) and the col panel
(4 verts) -- 24 scalars total -- through panel_geometry_all for each, then
through hs_influence, giving d(velocity)/d(vertex positions) for both
panels in one pass (3 backward passes, cost independent of the 24 inputs).

Not yet included: the final dot-product-with-normal step to get dA/dm
itself, and the y-mirror term used for half-ship symmetry. Both are cheap,
linear additions once this core chain is trusted -- deliberately left out
here to keep this oracle focused on the nonlinear geometry->kernel chain,
which is the part actually worth an independent correctness check.
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



# =============================================================================
# Combined trace: vertex positions (row panel + col panel) -> velocity
# =============================================================================

def combined_revad(row_corners, col_corners):
    """
    row_corners, col_corners: each [c0,c1,c2,c3], length-3 Node lists (one panel's 4 vertices)
    returns: [vx, vy, vz] as Node

    this is the composition
    """
    center_row, _coordsys_row, _cornerslocal_row, _area_row = panel_geometry_one(row_corners)
    center_col, coordsys_col, cornerslocal_col, _area_col = panel_geometry_one(col_corners)

    fieldpoint = center_row
    vel = hs_influence(fieldpoint, center_col, coordsys_col, cornerslocal_col)
    return list(vel)


# =============================================================================
# 24-scalar packing: row panel's 4 corners + col panel's 4 corners, x(x,y,z)
# =============================================================================

def pack(row_xyz, col_xyz):
    return np.concatenate([np.asarray(row_xyz, dtype=np.float64).ravel(),
                            np.asarray(col_xyz, dtype=np.float64).ravel()])  # (12,)+(12,) -> (24,)


def unpack_nodes(xs):
    row_corners = [[xs[3*k + i] for i in range(3)] for k in range(4)]
    col_corners = [[xs[12 + 3*k + i] for i in range(3)] for k in range(4)]
    return row_corners, col_corners


def run_traced(xs_nodes):
    row_corners, col_corners = unpack_nodes(xs_nodes)
    return combined_revad(row_corners, col_corners)


if __name__ == "__main__":
    pf = read_panel_file(os.path.join(_PROJECT_ROOT, "examples", "fifi.dat"))

    ROW, COL = 0, 50
    row_vert_idx = pf.panels[:, ROW]
    col_vert_idx = pf.panels[:, COL]
    row_xyz0 = pf.points[:, row_vert_idx].T.copy()  # (4,3)
    col_xyz0 = pf.points[:, col_vert_idx].T.copy()  # (4,3)

    x0 = pack(row_xyz0, col_xyz0)

    print(f"Tracing full chain (vertices -> geometry -> velocity) for row panel {ROW}, col panel {COL}")

    geom0 = panel_geometry_all(pf.points, pf.panels)
    v0 = hs_influence(geom0.center[:, ROW], geom0.center[:, COL],
                              geom0.coordsys[:, :, COL], geom0.cornerslocal[:, :, COL])
    print("primal velocity:", v0)

    J_revad = jacobian(run_traced, x0)  # (3, 24)

    eps = 1e-6
    N = x0.size
    J_fd = np.zeros((3, N), dtype=np.float64)
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

        vp = hs_influence(gp.center[:, ROW], gp.center[:, COL],
                                  gp.coordsys[:, :, COL], gp.cornerslocal[:, :, COL])
        vm = hs_influence(gm.center[:, ROW], gm.center[:, COL],
                                  gm.coordsys[:, :, COL], gm.cornerslocal[:, :, COL])
        J_fd[:, i] = (vp - vm) / (2.0 * eps)

    in_labels = (["row_c%d_%s" % (k, ax) for k in range(4) for ax in "xyz"]
                 + ["col_c%d_%s" % (k, ax) for k in range(4) for ax in "xyz"])
    out_labels = ["vx", "vy", "vz"]

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
    print(f"max |revad - fd| (absolute) over all 3x24 entries: {max_abs_err:.3e}")
    print(f"  worst-abs entry: d{worst_abs[1]}/d{worst_abs[0]}  revad={worst_abs[2]:.8e}  fd={worst_abs[3]:.8e}")
    print(f"max relative error (entries with |value|>1e-8): {max_rel_err:.3e}")
    print(f"  worst-rel entry: d{worst_rel[1]}/d{worst_rel[0]}  revad={worst_rel[2]:.8e}  "
          f"fd={worst_rel[3]:.8e}  rel={worst_rel[4]:.3e}")
