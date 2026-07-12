# -*- coding: utf-8 -*-
"""
Computes:
    d( A[row,col] ) / d(vertices)      -- for a HULL/BODY row.
    (the free-surface-row case, with its z=0 field-point override, is a
    separate file: trace_full_chain_fs_row_revad.py)

This is an Oracle:
    pure Python, builds a full object graph, does dynamic dispatch through operator overloading

THE ROW/COL DUAL ROLE (read this before touching this file):
--------------------------------------------------------------
Every panel in the mesh can appear as BOTH a "row" and a "col" in different
entries of A, and it plays a genuinely different geometric role each time:

  - As a ROW, a panel supplies the FIELD POINT: the point at which we ask
    "what is the induced velocity here, and does it satisfy the boundary
    condition (no-penetration for hull rows, the linearized free-surface
    condition for FS rows)?" For a hull/body row, that field point is simply
    the row panel's own center -- see trace_panel_geometry_revad.py's
    center_row. (For an FS row it is NOT simply the center -- see the
    separate fs_row file for why.) A row also supplies its own NORMAL
    vector, used to dot against the induced velocity to form the
    no-penetration residual.

  - As a COL, a panel supplies the SOURCE geometry: its center, coordsys,
    and corners_local feed hs_influence/phixx_influence as the panel whose
    unit source strength is inducing a velocity/Hessian elsewhere. A col
    panel's own normal is never used for its col role.

So A[row,col] = normal_row . velocity(fieldpoint_row, geometry_col), and the
row panel's vertices and the col panel's vertices enter this formula through
two DIFFERENT mechanisms (row: fieldpoint + normal; col: full source
geometry) -- that's why the combined trace below calls panel_geometry_revad
twice, once per role, and uses different pieces of each result.

THE MIRROR TERM:
-----------------
Half-ship symmetry: the induced velocity is evaluated once at the row's
field point and once at its y-reflection, then combined with the even/odd
rule (vx,vz even in y; vy odd in y) before being dotted with the normal.
This models the mirrored (port-side) copy of every source panel without
actually meshing it. It only affects A, not b (see the file docstring notes
in gradient_validators.py / numba_kernels.py for why b doesn't need it: b's
only geometry dependence is through the row's own normal against a constant
free-stream vector, which needs no source-panel mirroring at all).

THE SELF-TERM (row==col, hull diagonal) is NOT covered by this file -- it's
a closed-form substitution (velocity := -0.5*normal_row) that never calls
hs_influence at all, so its derivative is just -0.5 * d(normal_row)/d(row
vertices), directly available from panel_geometry_revad's coordsys output
with no new tracing.
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
# Combined trace: vertex positions (row panel + col panel) -> A[row,col]
# =============================================================================

def A_entry_revad(row_corners, col_corners):
    """
    row_corners, col_corners: each [c0,c1,c2,c3], length-3 Node lists
    returns: A_entry (scalar Node) = normal_row . (mirror-combined velocity)

    Body/hull row only -- field point is the row panel's own (unmodified)
    center. See module docstring for the FS-row variant, and for why row
    and col vertices enter through different mechanisms.
    """
    center_row, coordsys_row, _cornerslocal_row, _area_row = panel_geometry_one(row_corners)
    center_col, coordsys_col, cornerslocal_col, _area_col = panel_geometry_one(col_corners)

    normal_row = [coordsys_row[r][2] for r in range(3)]  # 3rd column = nv

    fieldpoint = center_row
    fieldpoint_mirror = [center_row[0], center_row[1] * (-1.0), center_row[2]]  # y-flip only

    v = hs_influence(fieldpoint, center_col, coordsys_col, cornerslocal_col)
    vp = hs_influence(fieldpoint_mirror, center_col, coordsys_col, cornerslocal_col)

    v_combined = [v[0] + vp[0], v[1] - vp[1], v[2] + vp[2]]  # even,odd,even in y

    return _dot(normal_row, v_combined)


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
    return [A_entry_revad(row_corners, col_corners)]  # jacobian() wants a sequence of outputs


# =============================================================================
# Plain-Python primal (for FD), replicating exactly the production formula
# in numba_kernels.py::assemble_A_b_vel_nb's body-BC row loop, for one
# (row,col) pair, without paying for the full O(N^2) assembly.
# =============================================================================

def A_entry_primal(points, panels, row, col):
    geom = panel_geometry_all(points, panels)
    p = geom.center[:, row].copy()
    pp = p.copy(); pp[1] *= -1.0
    n_row = geom.coordsys[:, 2, row]

    v = hs_influence(p, geom.center[:, col], geom.coordsys[:, :, col], geom.cornerslocal[:, :, col])
    vp = hs_influence(pp, geom.center[:, col], geom.coordsys[:, :, col], geom.cornerslocal[:, :, col])
    v_combined = np.array([v[0] + vp[0], v[1] - vp[1], v[2] + vp[2]])
    return float(np.dot(n_row, v_combined))


if __name__ == "__main__":
    pf = read_panel_file(os.path.join(_PROJECT_ROOT, "examples", "fifi.dat"))

    ROW, COL = 0, 50  # both hull panels; row != col, so the self-term substitution doesn't apply
    row_vert_idx = pf.panels[:, ROW]
    col_vert_idx = pf.panels[:, COL]
    row_xyz0 = pf.points[:, row_vert_idx].T.copy()
    col_xyz0 = pf.points[:, col_vert_idx].T.copy()

    x0 = pack(row_xyz0, col_xyz0)

    print(f"Tracing A[row={ROW},col={COL}] (vertices -> geometry -> velocity -> dot-with-normal, mirrored)")
    A0 = A_entry_primal(pf.points, pf.panels, ROW, COL)
    print("primal A[row,col]:", A0)

    J_revad = jacobian(run_traced, x0)  # (1, 24)

    eps = 1e-6
    N = x0.size
    J_fd = np.zeros(N, dtype=np.float64)
    for i in range(N):
        xp = x0.copy(); xp[i] += eps
        xm = x0.copy(); xm[i] -= eps

        points_p = pf.points.copy()
        points_m = pf.points.copy()
        points_p[:, row_vert_idx] = xp[0:12].reshape(4, 3).T
        points_p[:, col_vert_idx] = xp[12:24].reshape(4, 3).T
        points_m[:, row_vert_idx] = xm[0:12].reshape(4, 3).T
        points_m[:, col_vert_idx] = xm[12:24].reshape(4, 3).T

        Ap = A_entry_primal(points_p, pf.panels, ROW, COL)
        Am = A_entry_primal(points_m, pf.panels, ROW, COL)
        J_fd[i] = (Ap - Am) / (2.0 * eps)

    in_labels = (["row_c%d_%s" % (k, ax) for k in range(4) for ax in "xyz"]
                 + ["col_c%d_%s" % (k, ax) for k in range(4) for ax in "xyz"])

    diff = np.abs(J_revad[0] - J_fd)
    denom = np.maximum(np.abs(J_revad[0]), np.abs(J_fd))
    max_abs_err = float(np.max(diff))
    i_worst_abs = int(np.argmax(diff))
    mask = denom > 1e-8
    rel = np.zeros_like(diff)
    rel[mask] = diff[mask] / denom[mask]
    max_rel_err = float(np.max(rel))
    i_worst_rel = int(np.argmax(rel))

    print(f"\nJacobian shape: revad {J_revad.shape}, fd {J_fd.shape}")
    print(f"max |revad - fd| (absolute) over all 24 entries: {max_abs_err:.3e}")
    print(f"  worst-abs: d(A)/d({in_labels[i_worst_abs]})  revad={J_revad[0,i_worst_abs]:.8e}  fd={J_fd[i_worst_abs]:.8e}")
    print(f"max relative error (entries with |value|>1e-8): {max_rel_err:.3e}")
    print(f"  worst-rel: d(A)/d({in_labels[i_worst_rel]})  revad={J_revad[0,i_worst_rel]:.8e}  fd={J_fd[i_worst_rel]:.8e}")
