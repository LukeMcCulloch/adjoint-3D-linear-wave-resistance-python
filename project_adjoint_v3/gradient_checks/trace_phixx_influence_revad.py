# -*- coding: utf-8 -*-
"""
This is an Oracle:
    pure Python, builds a full object graph, does dynamic dispatch through operator overloading

Reverse-mode AD oracle for phixx_influence, traced through revad.py's Node
graph -- same treatment as trace_hs_influence_revad.py, see that file for the
tracing-vs-AST-transformation discussion.

Deliberately built from rankine_panel.numba_kernels.phixx_influence_nb's
formula (the actual production kernel used by assemble_A_b_vel_nb), NOT
rankine_panel.influence.phixx_influence -- the two currently disagree (an
extra "double the first block to match fortran" duplication in the unused
influence.py version that phixx_influence_nb does not have; parked, not
resolved here). Cross-checked via FD directly against phixx_influence_nb
itself (a numba @njit function, but plain-callable with ordinary float
arrays -- no numba-specific machinery needed to FD it).

Output is the 3x3 global Hessian -- 9 scalars -- so this needs 9 backward
passes (one per output entry) instead of hs_influence's 3, still independent
of the 23 scalar inputs.
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_AD_REPO_SRC = r"C:\tlm\projects\automatic-differentiation-schemes-in-python\src"
for p in (_PROJECT_ROOT, _AD_REPO_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from revad import Node, sqrt as _sqrt, jacobian  # from _AD_REPO_SRC, see trace_hs_influence_revad.py TODO

from rankine_panel.io import read_panel_file
from rankine_panel.geometry import panel_geometry_all
from rankine_panel.numba_kernels import phixx_influence_nb as phixx_influence_primal


# =============================================================================
# Node-traced port of phixx_influence_nb -- line-by-line structural match,
# same branches, same loop, same (no-duplication) accumulation.
# =============================================================================

def phixx_influence_revad(fieldpoint, center, coordsys, corners_local, eps=1e-6):
    """
    fieldpoint, center: length-3 sequences of Node
    coordsys: 3x3 nested list of Node, columns are (t1,t2,n)
    corners_local: 2x4 nested list of Node
    returns: 3x3 nested list of Node (global Hessian)
    """
    dx0 = fieldpoint[0] - center[0]
    dy0 = fieldpoint[1] - center[1]
    dz0 = fieldpoint[2] - center[2]

    x = coordsys[0][0]*dx0 + coordsys[1][0]*dy0 + coordsys[2][0]*dz0
    y = coordsys[0][1]*dx0 + coordsys[1][1]*dy0 + coordsys[2][1]*dz0
    z = coordsys[0][2]*dx0 + coordsys[1][2]*dy0 + coordsys[2][2]*dz0

    zero = Node(0.0, [])
    dd = [[zero, zero, zero], [zero, zero, zero], [zero, zero, zero]]

    ip1 = (1, 2, 3, 0)

    for k in range(4):
        k2 = ip1[k]
        xk, yk = corners_local[0][k], corners_local[1][k]
        xk2, yk2 = corners_local[0][k2], corners_local[1][k2]

        edx = xk2 - xk
        edy = yk2 - yk
        d = _sqrt(edx*edx + edy*edy)
        if d.val <= eps:
            continue

        r1 = _sqrt((x-xk)*(x-xk) + (y-yk)*(y-yk) + z*z)
        r2 = _sqrt((x-xk2)*(x-xk2) + (y-yk2)*(y-yk2) + z*z)

        s = r1 + r2
        denom1 = s*s - d*d
        if abs(denom1.val) <= 1e-30:
            continue

        big = (r1*r2) * (r1*r2 + (x-xk)*(x-xk2) + (y-yk)*(y-yk2) + z*z)
        if abs(big.val) <= 1e-30:
            continue

        dd[0][0] = dd[0][0] + (edy*2.0/denom1) * ((x-xk)/r1 + (x-xk2)/r2)
        dd[0][1] = dd[0][1] + (edx*(-2.0)/denom1) * ((x-xk)/r1 + (x-xk2)/r2)
        dd[0][2] = dd[0][2] + (z*edy*(r1+r2)) / big

        dd[1][0] = dd[1][0] + (edy*2.0/denom1) * ((y-yk)/r1 + (y-yk2)/r2)
        dd[1][1] = dd[1][1] + (edx*(-2.0)/denom1) * ((y-yk)/r1 + (y-yk2)/r2)
        dd[1][2] = dd[1][2] + ((z*(-1.0)*edx)*(r1+r2)) / big

        dd[2][0] = dd[2][0] + (edy*2.0 * ((z/r1) + (z/r2))) / denom1
        dd[2][1] = dd[2][1] + (edx*(-2.0) * ((z/r1) + (z/r2))) / denom1
        dd[2][2] = dd[2][2] + (((x-xk)*(y-yk2) - (x-xk2)*(y-yk)) * (r1+r2)) / big

    scale = -1.0 / (4.0 * np.pi)
    dd = [[dd[a][b] * scale for b in range(3)] for a in range(3)]

    # hess_global = coordsys @ dd @ coordsys^T
    tmp = [[None, None, None], [None, None, None], [None, None, None]]
    for a in range(3):
        for b in range(3):
            tmp[a][b] = coordsys[a][0]*dd[0][b] + coordsys[a][1]*dd[1][b] + coordsys[a][2]*dd[2][b]

    out = [[None, None, None], [None, None, None], [None, None, None]]
    for a in range(3):
        for b in range(3):
            out[a][b] = tmp[a][0]*coordsys[b][0] + tmp[a][1]*coordsys[b][1] + tmp[a][2]*coordsys[b][2]

    return out


# =============================================================================
# Same 23-scalar packing as trace_hs_influence_revad.py
# =============================================================================

def pack(fieldpoint, center, coordsys, corners_local):
    flat = list(fieldpoint) + list(center) + list(coordsys.ravel()) + list(corners_local.ravel())
    return np.array(flat, dtype=np.float64)


def unpack_nodes(xs):
    fieldpoint = xs[0:3]
    center = xs[3:6]
    coordsys = [[xs[6 + 3*i + j] for j in range(3)] for i in range(3)]
    corners_local = [[xs[15 + 4*i + j] for j in range(4)] for i in range(2)]
    return fieldpoint, center, coordsys, corners_local


def run_traced(xs_nodes):
    fieldpoint, center, coordsys, corners_local = unpack_nodes(xs_nodes)
    out = phixx_influence_revad(fieldpoint, center, coordsys, corners_local)
    return [out[a][b] for a in range(3) for b in range(3)]  # flatten row-major, 9 entries


if __name__ == "__main__":
    pf = read_panel_file(os.path.join(_PROJECT_ROOT, "examples", "fifi.dat"))
    geom = panel_geometry_all(pf.points, pf.panels)

    ROW, COL = 0, 50
    fieldpoint = geom.center[:, ROW].copy()
    center = geom.center[:, COL].copy()
    coordsys = geom.coordsys[:, :, COL].copy()
    corners_local = geom.cornerslocal[:, :, COL].copy()

    x0 = pack(fieldpoint, center, coordsys, corners_local)

    print(f"Tracing phixx_influence at real panel pair (fieldpoint=panel {ROW}, source panel {COL})")
    H0 = phixx_influence_primal(fieldpoint, center, coordsys, corners_local)
    print("primal value (3x3):\n", H0)

    J_revad = jacobian(run_traced, x0)  # (9, 23)

    eps = 1e-6
    N = x0.size
    J_fd = np.zeros((9, N), dtype=np.float64)
    for i in range(N):
        xp = x0.copy(); xp[i] += eps
        xm = x0.copy(); xm[i] -= eps
        fp = fieldpoint.copy(); cp = center.copy(); csp = coordsys.copy(); clp = corners_local.copy()
        fm = fieldpoint.copy(); cm = center.copy(); csm = coordsys.copy(); clm = corners_local.copy()

        def scatter(v, fld, ctr, csy, cl):
            fld[:] = v[0:3]; ctr[:] = v[3:6]
            csy[:, :] = v[6:15].reshape(3, 3)
            cl[:, :] = v[15:23].reshape(2, 4)

        scatter(xp, fp, cp, csp, clp)
        scatter(xm, fm, cm, csm, clm)

        Hp = phixx_influence_primal(fp, cp, csp, clp).ravel()
        Hm = phixx_influence_primal(fm, cm, csm, clm).ravel()
        J_fd[:, i] = (Hp - Hm) / (2.0 * eps)

    labels = (["fieldpoint%d" % i for i in range(3)]
              + ["center%d" % i for i in range(3)]
              + ["coordsys%d%d" % (i, j) for i in range(3) for j in range(3)]
              + ["cornersloc%d%d" % (i, j) for i in range(2) for j in range(4)])
    out_labels = ["H%d%d" % (a, b) for a in range(3) for b in range(3)]

    max_abs_err = 0.0
    worst = None
    for i, lab in enumerate(labels):
        for o, olab in enumerate(out_labels):
            err = abs(J_revad[o, i] - J_fd[o, i])
            if err > max_abs_err:
                max_abs_err = err
                worst = (lab, olab, J_revad[o, i], J_fd[o, i])

    print(f"\nJacobian shape: revad {J_revad.shape}, fd {J_fd.shape}")
    print(f"max |revad - fd| over all 9x23 Jacobian entries: {max_abs_err:.3e}")
    print(f"worst entry: d{worst[1]}/d{worst[0]}  revad={worst[2]:.8e}  fd={worst[3]:.8e}")
