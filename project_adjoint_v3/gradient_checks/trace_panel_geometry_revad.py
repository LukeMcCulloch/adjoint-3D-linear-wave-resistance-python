# -*- coding: utf-8 -*-
"""
This is an Oracle:
    pure Python, builds a full object graph, does dynamic dispatch through operator overloading

Reverse-mode AD oracle for panel_geometry_all, traced through revad.py's
Node graph -- same treatment as trace_hs_influence_revad.py /
trace_phixx_influence_revad.py.

This is the missing link between the two kernel oracles (which differentiate
w.r.t. panel geometry: center/coordsys/corners_local) and what we actually
want eventually: derivatives w.r.t. raw VERTEX positions. panel_geometry_all
(rankine_panel/geometry.py) is exactly that link -- 4 corner vertices in,
center/coordsys/cornerslocal/area out, via midpoints, a sqrt-normalized
tangent vector, two cross products, and centroid-moment corrections.

Traces ONE panel at a time (the production function is vectorized over N
panels; here we differentiate a single panel's 4 corners w.r.t. its own
geometry outputs -- 12 scalar inputs, 21 scalar outputs, 21 backward passes).
Cross-checked against central-FD on the real production panel_geometry_all,
perturbing one real panel's actual vertex coordinates from fifi.dat.

Branch caveat (same as the kernel oracles): the area>1e-10 guard is only
exercised on its "normal" (non-degenerate) side here, since we're tracing
with real, well-formed panel data.
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


# =============================================================================
# small Node-vector helpers
# =============================================================================

def _cross(a, b):
    return [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]


def _dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def _norm(a):
    return _sqrt(_dot(a, a))


# =============================================================================
# Node-traced port of panel_geometry_all for a SINGLE panel.
# Line-by-line structural match to the vectorized primal in geometry.py.
# =============================================================================

def panel_geometry_revad(corners):
    """
    corners: [c0, c1, c2, c3], each a length-3 list of Node (one panel's 4 vertices)
    returns: center(3 Node), coordsys (3x3 nested list, cols t1,t2,n),
             cornerslocal (2x4 nested list), area (Node)
    """
    c0, c1, c2, c3 = corners

    m1 = [(c0[i] + c1[i]) * 0.5 for i in range(3)]
    m2 = [(c1[i] + c2[i]) * 0.5 for i in range(3)]
    m3 = [(c2[i] + c3[i]) * 0.5 for i in range(3)]
    m4 = [(c3[i] + c0[i]) * 0.5 for i in range(3)]

    iv_raw = [m2[i] - m4[i] for i in range(3)]
    iv_norm = _norm(iv_raw)
    iv = [iv_raw[i] / iv_norm for i in range(3)]

    jvbar = [m3[i] - m1[i] for i in range(3)]

    nv_raw = _cross(iv, jvbar)
    nv_norm = _norm(nv_raw)
    nv = [nv_raw[i] / nv_norm for i in range(3)]

    jv = _cross(nv, iv)

    # coordsys[row][col], columns are (iv, jv, nv)
    coordsys = [[iv[r], jv[r], nv[r]] for r in range(3)]

    center0 = [(m1[i] + m2[i] + m3[i] + m4[i]) * 0.25 for i in range(3)]

    corners_list = [c0, c1, c2, c3]
    d = [[corners_list[k][i] - center0[i] for i in range(3)] for k in range(4)]

    xi = [_dot(d[k], iv) for k in range(4)]
    eta = [_dot(d[k], jv) for k in range(4)]

    ip1 = (1, 2, 3, 0)
    xi_next = [xi[ip1[k]] for k in range(4)]
    eta_next = [eta[ip1[k]] for k in range(4)]
    dxi = [xi_next[k] - xi[k] for k in range(4)]
    deta = [eta_next[k] - eta[k] for k in range(4)]

    area = Node(0.0, [])
    for k in range(4):
        area = area + deta[k] * (xi_next[k] + xi[k])
    area = area * 0.5

    momentxi = Node(0.0, [])
    momenteta = Node(0.0, [])
    for k in range(4):
        momentxi = momentxi + dxi[k] * (eta_next[k]*eta_next[k] + eta[k]*eta_next[k] + eta[k]*eta[k])
        momenteta = momenteta + deta[k] * (xi_next[k]*xi_next[k] + xi[k]*xi_next[k] + xi[k]*xi[k])
    momentxi = momentxi * (-1.0 / 6.0)
    momenteta = momenteta * (1.0 / 6.0)

    # real panels: area > 1e-10 always holds; not tracing the degenerate branch here (parked)
    xic = momenteta / area
    etac = momentxi / area

    cornerslocal = [[xi[k] - xic for k in range(4)], [eta[k] - etac for k in range(4)]]

    center = [center0[i] + iv[i]*xic + jv[i]*etac for i in range(3)]

    return center, coordsys, cornerslocal, area


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
    center, coordsys, cornerslocal, area = panel_geometry_revad(corners)
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
    
    
    
    # real FD truncation error should shrink roughly as eps**2 as eps shrinks (down to where cancellation error start to dominate and reverse the trend)
    # indeed in testing we see as eps drops by 1 order of mag, eps error measures drop by 2 orders of magnitude.
    #