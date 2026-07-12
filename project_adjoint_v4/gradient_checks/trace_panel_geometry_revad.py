# -*- coding: utf-8 -*-
"""
Computes:
    d(panel geometry) / d(vertices)

Validates rankine_panel.geometry.panel_geometry_all's reverse-mode gradient
directly -- same consolidation as trace_hs_influence_revad.py /
trace_phixx_influence_revad.py, see those files for the "one generic
function, no separate oracle copy" mechanism. One difference worth noting:
panel_geometry_all is vectorized over ALL N panels and sits on the live
production forward-solve path, so its dispatch is at the top of the
function (untouched float64 branch vs. a dtype=object/np.vectorize(sqrt)
branch) rather than per-operation -- see its docstring in geometry.py.

panel_geometry_one (also in geometry.py) is a thin convenience wrapper: it
builds the minimal 4-vertex/1-panel arrays panel_geometry_all expects and
squeezes the result back to per-panel shape, so this trace script (which
only ever needs one or two panels at a time) doesn't have to construct a
full N-panel mesh. It is NOT a second implementation of the formula --
panel_geometry_all is still the only place the math is written.

Traces ONE panel at a time -- 12 scalar inputs (4 corners x xyz), 21 scalar
outputs (center 3 + coordsys 9 + cornerslocal 8 + area 1), 21 backward
passes. Cross-checked against central-FD on the real production
panel_geometry_all, perturbing one real panel's actual vertex coordinates
from fifi.dat.

Branch caveat (same as the kernel oracles): the area>1e-10 guard is only
exercised on its "normal" (non-degenerate) side here, since we're tracing
with real, well-formed panel data.
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


# =============================================================================
# small Node-vector helper -- generic (works for float or Node), still used
# by trace_full_chain_A_entry_revad.py for the dot-with-normal step
# =============================================================================

def _dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


# =============================================================================
# 12-scalar packing: 4 corners x (x,y,z)
# =============================================================================

def pack(corners_xyz):
    return np.asarray(corners_xyz, dtype=np.float64).ravel()  # (4,3) -> (12,)


def unpack_nodes(xs):
    corners = [[xs[3*k + i] for i in range(3)] for k in range(4)]
    return corners


def run_traced(xs_nodes):
    corners = unpack_nodes(xs_nodes)
    center, coordsys, cornerslocal, area = panel_geometry_one(corners)
    out = list(center)
    out += [coordsys[r][c] for r in range(3) for c in range(3)]
    out += [cornerslocal[a][b] for a in range(2) for b in range(4)]
    out += [area]
    return out  # 3 + 9 + 8 + 1 = 21


if __name__ == "__main__":
    pf = read_panel_file(os.path.join(_PROJECT_ROOT, "examples", "fifi.dat"))

    PANEL = 0
    vert_idx = pf.panels[:, PANEL]  # 4 point indices for this panel
    corners_xyz0 = pf.points[:, vert_idx].T.copy()  # (4,3)

    x0 = pack(corners_xyz0)

    print(f"Tracing panel_geometry_all at real panel {PANEL} (vertex indices {vert_idx})")

    geom0 = panel_geometry_all(pf.points, pf.panels)
    print("primal center:", geom0.center[:, PANEL])
    print("primal area:  ", geom0.area[PANEL])

    J_revad = jacobian(run_traced, x0)  # (21, 12)

    eps = 1e-6
    N = x0.size
    J_fd = np.zeros((21, N), dtype=np.float64)
    for i in range(N):
        xp = x0.copy(); xp[i] += eps
        xm = x0.copy(); xm[i] -= eps

        points_p = pf.points.copy()
        points_m = pf.points.copy()
        points_p[:, vert_idx] = xp.reshape(4, 3).T
        points_m[:, vert_idx] = xm.reshape(4, 3).T

        gp = panel_geometry_all(points_p, pf.panels)
        gm = panel_geometry_all(points_m, pf.panels)

        out_p = np.concatenate([gp.center[:, PANEL], gp.coordsys[:, :, PANEL].ravel(),
                                 gp.cornerslocal[:, :, PANEL].ravel(), [gp.area[PANEL]]])
        out_m = np.concatenate([gm.center[:, PANEL], gm.coordsys[:, :, PANEL].ravel(),
                                 gm.cornerslocal[:, :, PANEL].ravel(), [gm.area[PANEL]]])
        J_fd[:, i] = (out_p - out_m) / (2.0 * eps)

    in_labels = ["c%d_%s" % (k, ax) for k in range(4) for ax in "xyz"]
    out_labels = (["center%d" % i for i in range(3)]
                  + ["coordsys%d%d" % (i, j) for i in range(3) for j in range(3)]
                  + ["cornersloc%d%d" % (i, j) for i in range(2) for j in range(4)]
                  + ["area"])

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
            if denom > 1e-8:  # skip near-zero entries, relative error ill-defined there
                rel = err / denom
                if rel > max_rel_err:
                    max_rel_err = rel
                    worst_rel = (ilab, olab, ad_v, fd_v, rel)

    print(f"\nJacobian shape: revad {J_revad.shape}, fd {J_fd.shape}")
    print(f"max |revad - fd| (absolute) over all 21x12 entries: {max_abs_err:.3e}")
    print(f"  worst-abs entry: d{worst_abs[1]}/d{worst_abs[0]}  revad={worst_abs[2]:.8e}  fd={worst_abs[3]:.8e}")
    print(f"max relative error (entries with |value|>1e-8): {max_rel_err:.3e}")
    print(f"  worst-rel entry: d{worst_rel[1]}/d{worst_rel[0]}  revad={worst_rel[2]:.8e}  "
          f"fd={worst_rel[3]:.8e}  rel={worst_rel[4]:.3e}")

    # Real FD truncation error should shrink roughly as eps**2 as eps shrinks
    # (down to where cancellation error starts to dominate and reverses the
    # trend) -- confirmed earlier in this project's history: as eps drops by
    # one order of magnitude, the error above drops by two, down to ~1e-6,
    # then grows again from floating-point roundoff. That behavior is the
    # signature of a CORRECT analytic reference being compared to an
    # imperfect FD estimate, not a bug in this trace.
