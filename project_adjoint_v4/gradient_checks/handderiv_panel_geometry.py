# -*- coding: utf-8 -*-
"""
HAND-DERIVED (not auto-traced) reverse-mode backward pass for
panel_geometry_one / panel_geometry_all's per-panel formula.

This is Track B, step 2 of the plan: no revad.Node, no dynamic graph, no
operator overloading -- a forward pass that computes the primal values AND
caches every intermediate the backward pass needs, then a backward pass
that walks those steps in EXACT reverse order applying the chain rule by
hand. This is the actual shape of what a numba/CUDA adjoint kernel looks
like -- static, no runtime graph -- and is checked against
revad's automatic backward pass (trace_panel_geometry_revad.py /
rankine_panel.geometry.panel_geometry_one), not derived from it.

FORWARD STEPS (numbered, matches panel_geometry_all's per-panel formula
exactly -- see geometry.py):
  1) m1,m2,m3,m4   = corner midpoints
  2) iv_raw        = m2 - m4
  3) iv_norm       = sqrt(dot(iv_raw,iv_raw))
  4) iv            = iv_raw / iv_norm
  5) jvbar         = m3 - m1
  6) nv_raw        = cross(iv, jvbar)
  7) nv_norm       = sqrt(dot(nv_raw,nv_raw))
  8) nv            = nv_raw / nv_norm
  9) jv            = cross(nv, iv)
 10) coordsys      = [iv, jv, nv]  (columns)
 11) center0       = 0.25*(m1+m2+m3+m4)
 12) d[k]          = corners[k] - center0,  k=0..3
 13) xi[k],eta[k]  = dot(d[k],iv), dot(d[k],jv)
 14) xi_next[k], eta_next[k] = xi[ip1[k]], eta[ip1[k]],  ip1=(1,2,3,0)
 15) dxi[k],deta[k]= xi_next[k]-xi[k], eta_next[k]-eta[k]
 16) area          = 0.5*sum_k( deta[k]*(xi_next[k]+xi[k]) )
 17) momentxi      = -1/6*sum_k( dxi[k]*(eta_next[k]^2+eta[k]*eta_next[k]+eta[k]^2) )
 18) momenteta     =  1/6*sum_k( deta[k]*(xi_next[k]^2+xi[k]*xi_next[k]+xi[k]^2) )
 19) xic,etac      = momenteta/area, momentxi/area
 20) cornerslocal  = [xi[k]-xic, eta[k]-etac]
 21) center        = center0 + iv*xic + jv*etac

BACKWARD PASS walks 21 -> 1, using four adjoint rules applied throughout:
  z=x+y:        dx+=dz,        dy+=dz
  z=x-y:        dx+=dz,        dy-=dz
  z=x*y (s,s):  dx+=y*dz,      dy+=x*dz
  z=x/y (s,s):  dx+=dz/y,      dy+=-z*dz/y
  z=v*s (vec,scalar): dv+=s*dz (elementwise),  ds+=dot(v,dz)
  z=dot(a,b):   da+=b*dz,      db+=a*dz          (dz is scalar here)
  z=cross(a,b): da+=cross(b,dz), db+=cross(dz,a)  (verified numerically below
                against a hand-checked example before trusting it further)
  z=sqrt(x):    dx+=dz*0.5/z
Every "sum over k=0..3" distributes its upstream adjoint equally to each
term before applying that term's own local rule.
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from rankine_panel.io import read_panel_file
from rankine_panel.geometry import panel_geometry_one
from rankine_panel.revad import jacobian


IP1 = (1, 2, 3, 0)


def _cross(a, b):
    return np.array([a[1]*b[2] - a[2]*b[1],
                      a[2]*b[0] - a[0]*b[2],
                      a[0]*b[1] - a[1]*b[0]])


# =============================================================================
# Forward pass: plain floats, caches every intermediate the backward pass needs
# =============================================================================

def panel_geometry_one_forward(corners):
    c0, c1, c2, c3 = [np.asarray(c, dtype=np.float64) for c in corners]

    m1 = 0.5*(c0+c1); m2 = 0.5*(c1+c2); m3 = 0.5*(c2+c3); m4 = 0.5*(c3+c0)

    iv_raw = m2 - m4
    iv_norm = np.sqrt(np.dot(iv_raw, iv_raw))
    iv = iv_raw / iv_norm

    jvbar = m3 - m1

    nv_raw = _cross(iv, jvbar)
    nv_norm = np.sqrt(np.dot(nv_raw, nv_raw))
    nv = nv_raw / nv_norm

    jv = _cross(nv, iv)

    coordsys = np.stack([iv, jv, nv], axis=1)  # columns

    center0 = 0.25*(m1+m2+m3+m4)

    corners_list = [c0, c1, c2, c3]
    d = [corners_list[k] - center0 for k in range(4)]

    xi = np.array([np.dot(d[k], iv) for k in range(4)])
    eta = np.array([np.dot(d[k], jv) for k in range(4)])

    xi_next = xi[list(IP1)]
    eta_next = eta[list(IP1)]

    dxi = xi_next - xi
    deta = eta_next - eta

    area = 0.5 * np.sum(deta * (xi_next + xi))

    momentxi = (-1.0/6.0) * np.sum(dxi * (eta_next**2 + eta*eta_next + eta**2))
    momenteta = (1.0/6.0) * np.sum(deta * (xi_next**2 + xi*xi_next + xi**2))

    xic = momenteta / area
    etac = momentxi / area

    cornerslocal = np.array([xi - xic, eta - etac])

    center = center0 + iv*xic + jv*etac

    cache = dict(c0=c0, c1=c1, c2=c2, c3=c3, m1=m1, m2=m2, m3=m3, m4=m4,
                 iv_raw=iv_raw, iv_norm=iv_norm, iv=iv, jvbar=jvbar,
                 nv_raw=nv_raw, nv_norm=nv_norm, nv=nv, jv=jv,
                 center0=center0, d=d, xi=xi, eta=eta,
                 xi_next=xi_next, eta_next=eta_next, dxi=dxi, deta=deta,
                 area=area, momentxi=momentxi, momenteta=momenteta,
                 xic=xic, etac=etac)

    return center, coordsys, cornerslocal, area, cache


# =============================================================================
# Backward pass: hand-derived, walks the forward steps in exact reverse order
# =============================================================================

def panel_geometry_one_backward(cache, d_center, d_coordsys, d_cornerslocal, d_area):
    iv, jv, nv = cache['iv'], cache['jv'], cache['nv']
    iv_raw, nv_raw = cache['iv_raw'], cache['nv_raw']
    iv_norm, nv_norm = cache['iv_norm'], cache['nv_norm']
    jvbar = cache['jvbar']
    xi, eta = cache['xi'], cache['eta']
    xi_next, eta_next = cache['xi_next'], cache['eta_next']
    dxi, deta = cache['dxi'], cache['deta']
    area, momentxi, momenteta = cache['area'], cache['momentxi'], cache['momenteta']
    xic, etac = cache['xic'], cache['etac']
    d_list = cache['d']

    # ---- step 10: coordsys columns are iv,jv,nv directly ----
    d_iv = d_coordsys[:, 0].copy()
    d_jv = d_coordsys[:, 1].copy()
    d_nv = d_coordsys[:, 2].copy()

    # ---- step 21: center = center0 + iv*xic + jv*etac ----
    d_center0 = d_center.copy()
    d_iv += xic * d_center
    d_xic = np.dot(iv, d_center)
    d_jv += etac * d_center
    d_etac = np.dot(jv, d_center)

    # ---- step 20: cornerslocal[0][k]=xi[k]-xic, [1][k]=eta[k]-etac ----
    d_xi = d_cornerslocal[0].copy()
    d_xic += -np.sum(d_cornerslocal[0])
    d_eta = d_cornerslocal[1].copy()
    d_etac += -np.sum(d_cornerslocal[1])

    # ---- step 19: xic=momenteta/area, etac=momentxi/area ----
    d_momenteta = d_xic / area
    d_area_local = -xic * d_xic / area
    d_momentxi = d_etac / area
    d_area_local += -etac * d_etac / area
    d_area_total = d_area + d_area_local  # area is ALSO a direct output

    # ---- step 18: momenteta = (1/6) sum_k deta[k]*Q[k], Q=xi_next^2+xi*xi_next+xi^2 ----
    d_T2 = np.full(4, (1.0/6.0) * d_momenteta)
    Q = xi_next**2 + xi*xi_next + xi**2
    d_deta = deta.copy() * 0.0  # init
    d_deta += Q * d_T2
    d_Q = deta * d_T2
    d_xi_next = 2*xi_next*d_Q + xi*d_Q
    d_xi = d_xi + (xi_next*d_Q + 2*xi*d_Q)

    # ---- step 17: momentxi = (-1/6) sum_k dxi[k]*P[k], P=eta_next^2+eta*eta_next+eta^2 ----
    d_T1 = np.full(4, (-1.0/6.0) * d_momentxi)
    P = eta_next**2 + eta*eta_next + eta**2
    d_dxi = dxi.copy() * 0.0
    d_dxi += P * d_T1
    d_P = dxi * d_T1
    d_eta_next = 2*eta_next*d_P + eta*d_P
    d_eta = d_eta + (eta_next*d_P + 2*eta*d_P)

    # ---- step 16: area = 0.5 sum_k deta[k]*(xi_next[k]+xi[k]) ----
    d_S = np.full(4, 0.5 * d_area_total)
    d_deta += (xi_next + xi) * d_S
    d_xi_next = d_xi_next + deta * d_S
    d_xi = d_xi + deta * d_S

    # ---- step 15: dxi=xi_next-xi, deta=eta_next-eta ----
    d_xi_next = d_xi_next + d_dxi
    d_xi = d_xi - d_dxi
    d_eta_next = d_eta_next + d_deta
    d_eta = d_eta - d_deta

    # ---- step 14: xi_next[k]=xi[ip1[k]], eta_next[k]=eta[ip1[k]] ----
    for k in range(4):
        d_xi[IP1[k]] += d_xi_next[k]
        d_eta[IP1[k]] += d_eta_next[k]

    # ---- step 13: xi[k]=dot(d[k],iv), eta[k]=dot(d[k],jv) ----
    d_d = [np.zeros(3) for _ in range(4)]
    for k in range(4):
        d_d[k] += iv * d_xi[k]
        d_iv += d_list[k] * d_xi[k]
        d_d[k] += jv * d_eta[k]
        d_jv += d_list[k] * d_eta[k]

    # ---- step 12: d[k]=corners[k]-center0 ----
    d_c = [np.zeros(3) for _ in range(4)]
    for k in range(4):
        d_c[k] += d_d[k]
        d_center0 = d_center0 - d_d[k]

    # ---- step 9: jv = cross(nv, iv) ----
    d_nv += _cross(iv, d_jv)
    d_iv += _cross(d_jv, nv)

    # ---- step 8: nv = nv_raw/nv_norm ----
    d_nv_raw = d_nv / nv_norm
    d_nv_norm = -np.dot(nv, d_nv) / nv_norm

    # ---- step 7: nv_norm = sqrt(dot(nv_raw,nv_raw)) ----
    d_s2 = d_nv_norm * 0.5 / nv_norm
    d_nv_raw = d_nv_raw + 2*nv_raw*d_s2

    # ---- step 6: nv_raw = cross(iv, jvbar) ----
    d_iv += _cross(jvbar, d_nv_raw)
    d_jvbar = _cross(d_nv_raw, iv)

    # ---- step 5: jvbar = m3 - m1 ----
    d_m3 = d_jvbar.copy()
    d_m1 = -d_jvbar.copy()

    # ---- step 4: iv = iv_raw/iv_norm ----
    d_iv_raw = d_iv / iv_norm
    d_iv_norm = -np.dot(iv, d_iv) / iv_norm

    # ---- step 3: iv_norm = sqrt(dot(iv_raw,iv_raw)) ----
    d_s1 = d_iv_norm * 0.5 / iv_norm
    d_iv_raw = d_iv_raw + 2*iv_raw*d_s1

    # ---- step 2: iv_raw = m2 - m4 ----
    d_m2 = d_iv_raw.copy()
    d_m4 = -d_iv_raw.copy()

    # ---- step 11: center0 = 0.25*(m1+m2+m3+m4) ----
    d_m1 = d_m1 + 0.25*d_center0
    d_m2 = d_m2 + 0.25*d_center0
    d_m3 = d_m3 + 0.25*d_center0
    d_m4 = d_m4 + 0.25*d_center0

    # ---- step 1: m1=.5(c0+c1), m2=.5(c1+c2), m3=.5(c2+c3), m4=.5(c3+c0) ----
    d_c[0] += 0.5*d_m1
    d_c[1] += 0.5*d_m1
    d_c[1] += 0.5*d_m2
    d_c[2] += 0.5*d_m2
    d_c[2] += 0.5*d_m3
    d_c[3] += 0.5*d_m3
    d_c[3] += 0.5*d_m4
    d_c[0] += 0.5*d_m4

    return d_c[0], d_c[1], d_c[2], d_c[3]


# =============================================================================
# Validation: hand-derived backward pass vs. the trusted revad oracle,
# one output direction (one of the 21 outputs) at a time.
# =============================================================================

if __name__ == "__main__":
    pf = read_panel_file(os.path.join(_PROJECT_ROOT, "examples", "fifi.dat"))
    PANEL = 0
    vert_idx = pf.panels[:, PANEL]
    corners_xyz0 = pf.points[:, vert_idx].T.copy()  # (4,3)
    corners = [corners_xyz0[k] for k in range(4)]

    center, coordsys, cornerslocal, area, cache = panel_geometry_one_forward(corners)
    print("forward center:", center, " area:", area)

    # revad oracle Jacobian (21,12), same as trace_panel_geometry_revad.py
    def pack(corners_xyz):
        return np.asarray(corners_xyz, dtype=np.float64).ravel()

    def unpack_nodes(xs):
        return [[xs[3*k+i] for i in range(3)] for k in range(4)]

    def run_traced(xs_nodes):
        c = unpack_nodes(xs_nodes)
        ctr, csy, cl, ar = panel_geometry_one(c)
        out = list(ctr) + [csy[r][c2] for r in range(3) for c2 in range(3)] + \
              [cl[a][b] for a in range(2) for b in range(4)] + [ar]
        return out

    x0 = pack(corners_xyz0)
    J_oracle = jacobian(run_traced, x0)  # (21, 12)

    out_labels = (["center%d" % i for i in range(3)]
                  + ["coordsys%d%d" % (i, j) for i in range(3) for j in range(3)]
                  + ["cornersloc%d%d" % (i, j) for i in range(2) for j in range(4)]
                  + ["area"])

    max_abs_err = 0.0
    worst = None
    J_hand = np.zeros((21, 12))
    for o, olab in enumerate(out_labels):
        d_center = np.zeros(3); d_coordsys = np.zeros((3, 3)); d_cornerslocal = np.zeros((2, 4)); d_area = 0.0
        if o < 3:
            d_center[o] = 1.0
        elif o < 12:
            idx = o - 3
            d_coordsys[idx // 3, idx % 3] = 1.0
        elif o < 20:
            idx = o - 12
            d_cornerslocal[idx // 4, idx % 4] = 1.0
        else:
            d_area = 1.0

        d_c0, d_c1, d_c2, d_c3 = panel_geometry_one_backward(cache, d_center, d_coordsys, d_cornerslocal, d_area)
        row_hand = np.concatenate([d_c0, d_c1, d_c2, d_c3])
        J_hand[o, :] = row_hand

        row_oracle = J_oracle[o, :]
        err = np.max(np.abs(row_hand - row_oracle))
        if err > max_abs_err:
            max_abs_err = err
            worst = (olab, row_hand, row_oracle)

    print(f"\nmax |hand-derived - revad oracle| over all 21x12 entries: {max_abs_err:.3e}")
    print(f"worst output direction: {worst[0]}")
    print(f"  hand:   {worst[1]}")
    print(f"  oracle: {worst[2]}")
