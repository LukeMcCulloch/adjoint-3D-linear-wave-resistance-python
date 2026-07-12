# -*- coding: utf-8 -*-
"""
Computes:
    d( A[row,col] ) / d(vertices)      -- for a FREE-SURFACE row.
    (the hull/body-row case, with the dot-with-normal + mirror term, is
    trace_full_chain_A_entry_revad.py -- read that file's docstring first,
    the row/col dual-role and mirror-term explanations there apply here too
    and aren't repeated in full.)

This is an Oracle:
    pure Python, builds a full object graph, does dynamic dispatch through operator overloading

THE FREE-SURFACE FIELD-POINT OVERRIDE (the one genuinely new thing here):
---------------------------------------------------------------------------
For a free-surface row, the field point where the boundary condition gets
evaluated is NOT the row panel's own center. From
numba_kernels.py::assemble_A_b_vel_nb's free-surface-row loop:

    p0 = center[row,0] + deltax   # x: real geometry + a CONSTANT offset
    p1 = center[row,1]            # y: real geometry, unmodified
    p2 = 0.0                      # z: a literal constant -- NOT center[row,2]

p2 is not "the row panel's z, which happens to be near zero" -- it is
disconnected from the row panel's vertices entirely. So in this trace,
center_row's z-component (which DOES have a real, nonzero derivative w.r.t.
the row panel's vertices, same as always) is computed and then simply never
used; the field point's z is built as Node(0.0, []) -- a leaf with no
parents. Get this backwards (use center_row[2] instead of a literal 0) and
you'd get a plausible but wrong nonzero gradient contribution.

THE FORMULA (also from assemble_A_b_vel_nb, free-surface-row loop):

    v   = hs_influence(fieldpoint,        center_col, coordsys_col, cornerslocal_col)
    vp  = hs_influence(fieldpoint_mirror, center_col, coordsys_col, cornerslocal_col)
    v2  = v[2] + vp[2]                          # z-component, even in y

    H   = phixx_influence(fieldpoint,        center_col, coordsys_col, cornerslocal_col)
    Hp  = phixx_influence(fieldpoint_mirror, center_col, coordsys_col, cornerslocal_col)
    phi_xx = H[0,0] + Hp[0,0]                   # even in y (straight sum, not a difference)

    A[row,col] = U2*phi_xx + gravity*v2

No dot-with-normal here (unlike the hull-row case) -- the free-surface
condition doesn't involve the row panel's own orientation at all.
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from rankine_panel.revad import Node, jacobian

from rankine_panel.io import read_panel_file
from rankine_panel.geometry import panel_geometry_all
from rankine_panel.influence import hs_influence, phixx_influence

from trace_panel_geometry_revad import panel_geometry_revad


# =============================================================================
# Combined trace: vertex positions (FS row panel + col panel) -> A[row,col]
# =============================================================================

def A_entry_fs_row_revad(row_corners, col_corners, deltax, U2, gravity):
    """
    row_corners, col_corners: each [c0,c1,c2,c3], length-3 Node lists
    deltax, U2, gravity: plain floats (flow/geometry constants, not design variables)
    returns: A_entry (scalar Node)
    """
    center_row, _coordsys_row, _cornerslocal_row, _area_row = panel_geometry_revad(row_corners)
    center_col, coordsys_col, cornerslocal_col, _area_col = panel_geometry_revad(col_corners)

    # field point: x and y trace through real geometry, z is a hard-set
    # constant (0.0) with NO dependence on any vertex -- see module docstring
    p0 = center_row[0] + deltax
    p1 = center_row[1]
    p2 = Node(0.0, [])

    pp0 = p0
    pp1 = center_row[1] * (-1.0)
    pp2 = Node(0.0, [])

    fieldpoint = [p0, p1, p2]
    fieldpoint_mirror = [pp0, pp1, pp2]

    v = hs_influence(fieldpoint, center_col, coordsys_col, cornerslocal_col)
    vp = hs_influence(fieldpoint_mirror, center_col, coordsys_col, cornerslocal_col)
    v2 = v[2] + vp[2]

    H = phixx_influence(fieldpoint, center_col, coordsys_col, cornerslocal_col)
    Hp = phixx_influence(fieldpoint_mirror, center_col, coordsys_col, cornerslocal_col)
    phi_xx = H[0][0] + Hp[0][0]

    return phi_xx * U2 + v2 * gravity


# =============================================================================
# Same 24-scalar packing as the other combined-chain files
# =============================================================================

def pack(row_xyz, col_xyz):
    return np.concatenate([np.asarray(row_xyz, dtype=np.float64).ravel(),
                            np.asarray(col_xyz, dtype=np.float64).ravel()])


def unpack_nodes(xs):
    row_corners = [[xs[3*k + i] for i in range(3)] for k in range(4)]
    col_corners = [[xs[12 + 3*k + i] for i in range(3)] for k in range(4)]
    return row_corners, col_corners


def make_run_traced(deltax, U2, gravity):
    def run_traced(xs_nodes):
        row_corners, col_corners = unpack_nodes(xs_nodes)
        return [A_entry_fs_row_revad(row_corners, col_corners, deltax, U2, gravity)]
    return run_traced


# =============================================================================
# Plain-Python primal (for FD), replicating the free-surface-row formula
# from numba_kernels.py::assemble_A_b_vel_nb exactly, for one (row,col)
# pair, without paying for the full O(N^2) assembly.
# =============================================================================

def A_entry_fs_row_primal(points, panels, row, col, deltax, U2, gravity):
    geom = panel_geometry_all(points, panels)

    p = np.array([geom.center[0, row] + deltax, geom.center[1, row], 0.0])
    pp = np.array([p[0], -p[1], 0.0])

    v = hs_influence(p, geom.center[:, col], geom.coordsys[:, :, col], geom.cornerslocal[:, :, col])
    vp = hs_influence(pp, geom.center[:, col], geom.coordsys[:, :, col], geom.cornerslocal[:, :, col])
    v2 = v[2] + vp[2]

    H = phixx_influence(p, geom.center[:, col], geom.coordsys[:, :, col], geom.cornerslocal[:, :, col])
    Hp = phixx_influence(pp, geom.center[:, col], geom.coordsys[:, :, col], geom.cornerslocal[:, :, col])
    phi_xx = H[0, 0] + Hp[0, 0]

    return float(U2 * phi_xx + gravity * v2)


if __name__ == "__main__":
    pf = read_panel_file(os.path.join(_PROJECT_ROOT, "examples", "fifi.dat"))

    Fr = 0.3
    gravity = 9.80665
    length = 1.0
    U = Fr * np.sqrt(length * gravity)
    U2 = U * U
    deltax = pf.deltax

    ROW, COL = 300, 50  # ROW is a free-surface panel (>= npanels=288), COL is a hull panel
    assert ROW >= pf.npanels, "ROW must be a free-surface panel for this file"

    row_vert_idx = pf.panels[:, ROW]
    col_vert_idx = pf.panels[:, COL]
    row_xyz0 = pf.points[:, row_vert_idx].T.copy()
    col_xyz0 = pf.points[:, col_vert_idx].T.copy()

    x0 = pack(row_xyz0, col_xyz0)

    print(f"Tracing A[row={ROW} (FS panel), col={COL} (hull panel)]  deltax={deltax}  U2={U2:.6f}  gravity={gravity}")
    A0 = A_entry_fs_row_primal(pf.points, pf.panels, ROW, COL, deltax, U2, gravity)
    print("primal A[row,col]:", A0)

    run_traced = make_run_traced(deltax, U2, gravity)
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

        Ap = A_entry_fs_row_primal(points_p, pf.panels, ROW, COL, deltax, U2, gravity)
        Am = A_entry_fs_row_primal(points_m, pf.panels, ROW, COL, deltax, U2, gravity)
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
    max_rel_err = float(np.max(rel)) if mask.any() else 0.0
    i_worst_rel = int(np.argmax(rel))

    print(f"\nJacobian shape: revad {J_revad.shape}, fd {J_fd.shape}")
    print(f"max |revad - fd| (absolute) over all 24 entries: {max_abs_err:.3e}")
    print(f"  worst-abs: d(A)/d({in_labels[i_worst_abs]})  revad={J_revad[0,i_worst_abs]:.8e}  fd={J_fd[i_worst_abs]:.8e}")
    print(f"max relative error (entries with |value|>1e-8): {max_rel_err:.3e}")
    print(f"  worst-rel: d(A)/d({in_labels[i_worst_rel]})  revad={J_revad[0,i_worst_rel]:.8e}  fd={J_fd[i_worst_rel]:.8e}")

    # sanity check on the override itself: d(A)/d(row_c*_z) should NOT be
    # forced to zero (center_row's z still feeds phixx/hs_influence's
    # "center" argument when THIS panel acts as a col elsewhere -- but here,
    # as a row, z only ever entered through fieldpoint[2], which is the
    # disconnected constant). Since row is not also col here, we expect the
    # row_c*_z columns to show near-zero sensitivity -- print them explicitly
    # rather than asserting, since "near-zero" isn't the same as "exactly
    # zero" once mixed with FD noise.
    print("\nsanity: d(A)/d(row_c*_z) columns (expect ~0, since fieldpoint.z never used row's real z):")
    for k in range(4):
        idx = 3*k + 2
        print(f"  row_c{k}_z: revad={J_revad[0, idx]:+.3e}  fd={J_fd[idx]:+.3e}")
