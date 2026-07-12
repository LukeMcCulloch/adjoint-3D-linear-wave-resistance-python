# -*- coding: utf-8 -*-
"""
Computes:
    d(Hessian) / d(panel geometry)

Validates rankine_panel.influence.phixx_influence's reverse-mode gradient
directly -- same consolidation as trace_hs_influence_revad.py, see that
file's docstring for the "one generic function, no separate oracle copy"
mechanism, not repeated here.

phixx_influence now matches the production numba_kernels.py::
phixx_influence_nb formula (no "double the first block" duplication) --
that discrepancy is now resolved as part of consolidating into one
function (see influence.py's docstring on phixx_influence for the full
reasoning; it was a parked, unresolved discrepancy before this).

Output is the 3x3 global Hessian -- 9 scalars -- so this needs 9 backward
passes (one per output entry) instead of hs_influence's 3, still
independent of the 23 scalar inputs.
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
from rankine_panel.influence import phixx_influence
from rankine_panel.revad import jacobian


# =============================================================================
# Same 23-scalar packing as trace_hs_influence_revad.py
# =============================================================================

def pack(fieldpoint, center, coordsys, corners_local):
    flat = list(fieldpoint) + list(center) + list(coordsys.ravel()) + list(corners_local.ravel())
    return np.array(flat, dtype=np.float64)


def unpack_nodes(xs):
    fieldpoint = xs[0:3]
    center = xs[3:6]
    coordsys = np.array(xs[6:15], dtype=object).reshape(3, 3)
    corners_local = np.array(xs[15:23], dtype=object).reshape(2, 4)
    return fieldpoint, center, coordsys, corners_local


def run_traced(xs_nodes):
    fieldpoint, center, coordsys, corners_local = unpack_nodes(xs_nodes)  # already Node-valued by the time they get here, via jacobian()
    out = phixx_influence(fieldpoint, center, coordsys, corners_local)
    return list(out.ravel())  # flatten row-major, 9 entries


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
    H0 = phixx_influence(fieldpoint, center, coordsys, corners_local)
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

        Hp = phixx_influence(fp, cp, csp, clp).ravel()
        Hm = phixx_influence(fm, cm, csm, clm).ravel()
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
