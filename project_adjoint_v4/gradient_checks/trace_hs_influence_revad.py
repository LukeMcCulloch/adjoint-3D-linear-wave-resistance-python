# -*- coding: utf-8 -*-
"""
Computes:
    d(velocity) / d(panel geometry)

Validates rankine_panel.influence.hs_influence's reverse-mode gradient
directly -- there is no separate "oracle" function anymore. hs_influence
itself is now GENERIC (see its docstring in influence.py): called with
plain float64 arrays it behaves exactly as before (verified: 0.0 diff vs.
the pre-refactor version); called with arrays of revad.Node it builds a
full reverse-mode graph via operator-overload tracing (NOT AST/source
transformation) and gets differentiated by revad.jacobian(). Both the FD
reference below and the AD trace call the SAME function -- one source of
truth, no hand-duplicated kernel copy to drift out of sync.

Validates the full sensitivity of hs_influence's output velocity (3,) to
every scalar input (fieldpoint(3), center(3), coordsys(3,3),
corners_local(2,4) -- 23 scalars total), computed in ONE reverse-mode
Jacobian pass (3 backward passes, one per output component -- cost
independent of the 23 inputs, the whole point of reverse mode).

Branch-coverage caveat (unchanged from before the consolidation): real,
representative panel geometry from fifi.dat exercises the normal
(non-degenerate) path, which is what the production assembly hits
overwhelmingly; a genuinely degenerate panel is not exercised here.
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
from rankine_panel.influence import hs_influence
from rankine_panel.revad import jacobian


# =============================================================================
# Flatten/unflatten: 23 scalars = fieldpoint(3) + center(3) + coordsys(9) + corners_local(8)
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
    return list(hs_influence(fieldpoint, center, coordsys, corners_local))


# =============================================================================
# Main: real panel geometry, revad Jacobian vs. central-FD -- both via hs_influence
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
    print("primal value:", hs_influence(fieldpoint, center, coordsys, corners_local))

    J_revad = jacobian(run_traced, x0)  # (3, 23)

    # central-FD Jacobian, on the SAME hs_influence, called with plain floats
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

        vp = hs_influence(fp, cp, csp, clp)
        vm = hs_influence(fm, cm, csm, clm)
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
