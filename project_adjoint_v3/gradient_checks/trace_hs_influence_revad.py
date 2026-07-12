# -*- coding: utf-8 -*-
"""
Computes: 
    d(velocity) / d(panel geometry)
    
This is an Oracle:
    pure Python, builds a full object graph, does dynamic dispatch through operator overloading
    
Reverse-mode AD oracle for hs_influence, traced through revad.py's Node graph
(operator-overload tracing -- NOT AST/source transformation).

This is a correctness oracle, not a performance path: it exists to derive and
validate the full sensitivity of hs_influence's output velocity (3,) to every
scalar input (fieldpoint(3), center(3), coordsys(3,3), corners_local(2,4) --
23 scalars total), computed in ONE reverse-mode Jacobian pass (3 backward
passes, one per output component -- cost independent of the 23 inputs, which
is the whole point of reverse mode).

Cross-checked against central-FD directly on the existing trusted plain-
Python rankine_panel.influence.hs_influence, using REAL panel geometry from
fifi.dat (not synthetic data) -- because hs_influence has data-dependent
branches (degenerate-edge guards), a trace only captures whichever branch the
specific inputs given to it took. Real, representative panel pairs exercise
the normal (non-degenerate) path, which is what the production assembly hits
overwhelmingly; that's the branch coverage this oracle validates.
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
from revad import Node, atan, sqrt as _sqrt, log as _log, jacobian # getting from _AD_REPO_SRC nearby but not in this repo! #todo: bring revad over so we don't have weird dependencies

from rankine_panel.io import read_panel_file
from rankine_panel.geometry import panel_geometry_all
from rankine_panel.influence import hs_influence as hs_influence_primal


# =============================================================================
# Node-traced port of hs_influence (rankine_panel/influence.py::hs_influence,
# same formula as rankine_panel/numba_kernels.py::hs_influence_nb).
# Line-by-line structural match to the primal -- same branches, same loop --
# just built from Node objects instead of plain floats.
# =============================================================================

def hs_influence_revad(fieldpoint, center, coordsys, corners_local, eps=1e-6):
    """
    fieldpoint, center: length-3 sequences of Node
    coordsys: 3x3 nested list of Node, columns are (t1,t2,n)
    corners_local: 2x4 nested list of Node
    returns: [vx, vy, vz] as Node
    """
    dx0 = fieldpoint[0] - center[0]
    dy0 = fieldpoint[1] - center[1]
    dz0 = fieldpoint[2] - center[2]

    x = coordsys[0][0]*dx0 + coordsys[1][0]*dy0 + coordsys[2][0]*dz0
    y = coordsys[0][1]*dx0 + coordsys[1][1]*dy0 + coordsys[2][1]*dz0
    z = coordsys[0][2]*dx0 + coordsys[1][2]*dy0 + coordsys[2][2]*dz0

    # node objects:
    dphi0 = Node(0.0, [])
    dphi1 = Node(0.0, [])
    dphi2 = Node(0.0, [])

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
        num = s - d
        den = s + d
        if num.val <= 0.0 or den.val <= 0.0:
            continue

        L = _log(num / den)

        dphi0 = dphi0 + (edy / d) * L
        dphi1 = dphi1 - (edx / d) * L

        if abs(edx.val) < 1e-30:
            m = Node(float(np.sign(edy.val) * 1e30), [])
        else:
            m = edy / edx

        e1 = (x - xk)*(x - xk) + z*z
        h1 = (x - xk)*(y - yk)
        r1 = _sqrt((x-xk)*(x-xk) + (y-yk)*(y-yk) + z*z)

        e2 = (x - xk2)*(x - xk2) + z*z
        h2 = (x - xk2)*(y - yk2)
        r2 = _sqrt((x-xk2)*(x-xk2) + (y-yk2)*(y-yk2) + z*z)

        denom1 = z * r1
        denom2 = z * r2

        if abs(denom1.val) > 1e-30 and abs(denom2.val) > 1e-30:
            dphi2 = dphi2 + atan((m*e1 - h1) / denom1) - atan((m*e2 - h2) / denom2)

    scale = -1.0 / (4.0 * np.pi)
    dphi0 = dphi0 * scale
    dphi1 = dphi1 * scale
    dphi2 = dphi2 * scale

    vx = coordsys[0][0]*dphi0 + coordsys[0][1]*dphi1 + coordsys[0][2]*dphi2
    vy = coordsys[1][0]*dphi0 + coordsys[1][1]*dphi1 + coordsys[1][2]*dphi2
    vz = coordsys[2][0]*dphi0 + coordsys[2][1]*dphi1 + coordsys[2][2]*dphi2

    return [vx, vy, vz]


# =============================================================================
# Flatten/unflatten: 23 scalars = fieldpoint(3) + center(3) + coordsys(9) + corners_local(8)
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
    return hs_influence_revad(fieldpoint, center, coordsys, corners_local)


# =============================================================================
# Main: real panel geometry, revad Jacobian vs. central-FD on the primal
# =============================================================================

if __name__ == "__main__":
    pf = read_panel_file(os.path.join(_PROJECT_ROOT, "examples", "fifi.dat"))
    geom = panel_geometry_all(pf.points, pf.panels)

    ROW, COL = 0, 50  # distinct real panels, non-adjacent, avoids the i==j self-term special case
    fieldpoint = geom.center[:, ROW].copy()
    center = geom.center[:, COL].copy()
    coordsys = geom.coordsys[:, :, COL].copy()
    corners_local = geom.cornerslocal[:, :, COL].copy()

    x0 = pack(fieldpoint, center, coordsys, corners_local)

    print(f"Tracing hs_influence at real panel pair (fieldpoint=panel {ROW}, source panel {COL})")
    print("primal value:", hs_influence_primal(fieldpoint, center, coordsys, corners_local))

    J_revad = jacobian(run_traced, x0)  # (3, 23)

    # central-FD Jacobian on the trusted plain-Python primal
    eps = 1e-6
    N = x0.size
    J_fd = np.zeros((3, N), dtype=np.float64)
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

        vp = hs_influence_primal(fp, cp, csp, clp)
        vm = hs_influence_primal(fm, cm, csm, clm)
        J_fd[:, i] = (vp - vm) / (2.0 * eps)

    labels = (["fieldpoint%d" % i for i in range(3)]
              + ["center%d" % i for i in range(3)]
              + ["coordsys%d%d" % (i, j) for i in range(3) for j in range(3)]
              + ["cornersloc%d%d" % (i, j) for i in range(2) for j in range(4)])

    print(f"\n{'input':<14}{'d vx (revad)':>16}{'d vx (fd)':>16}{'d vy (revad)':>16}"
          f"{'d vy (fd)':>16}{'d vz (revad)':>16}{'d vz (fd)':>16}")
    max_abs_err = 0.0
    for i, lab in enumerate(labels):
        row_ad = J_revad[:, i]
        row_fd = J_fd[:, i]
        err = np.max(np.abs(row_ad - row_fd))
        max_abs_err = max(max_abs_err, err)
        print(f"{lab:<14}{row_ad[0]:>16.8e}{row_fd[0]:>16.8e}{row_ad[1]:>16.8e}"
              f"{row_fd[1]:>16.8e}{row_ad[2]:>16.8e}{row_fd[2]:>16.8e}")

    print(f"\nmax |revad - fd| over all 3x23 Jacobian entries: {max_abs_err:.3e}")
