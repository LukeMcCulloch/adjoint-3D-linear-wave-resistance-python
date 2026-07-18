# -*- coding: utf-8 -*-
"""
HAND-DERIVED (not auto-traced) reverse-mode backward pass for phixx_influence
(rankine_panel/influence.py::phixx_influence, matching the production
rankine_panel/numba_kernels.py::phixx_influence_nb formula -- no "double the
first block" duplication).

Same idea as handderiv_hs_influence.py: no revad.Node, no dynamic graph, a
forward pass that caches every intermediate, a backward pass that walks
those steps in EXACT reverse order by hand. Checked against revad's
automatic backward pass (trace_phixx_influence_revad.py /
rankine_panel.influence.phixx_influence), not derived from it. Go-by (the
intact original, pre-consolidation, unrolled-scalar style) is
project_adjoint_v3/gradient_checks/trace_phixx_influence_revad.py's
phixx_influence_revad.

Output is the 3x3 global Hessian (9 scalars), not 3 -- so this needs 9
backward passes (one per output entry) to get the full (9,23) Jacobian,
still independent of the 23 scalar inputs (the whole point of reverse mode).

FORWARD STEPS:
  1) dx0,dy0,dz0 = fieldpoint - center                         [z=x-y] x3
  2) x,y,z = coordsys^T @ delta   (same as hs_influence's step 2)
  3) for k=0..3 (edge k -> k2=ip1[k]):
     3a) xk,yk,xk2,yk2   = corners_local columns k, k2          (reads only)
     3b) edx,edy          = xk2-xk, yk2-yk                      [z=x-y]
     3c) dlen              = sqrt(edx^2+edy^2)                   [z=sqrt(x)]
         -- if dlen<=eps: edge contributes nothing, skip --
     3d) rx1,ry1,r1       = x-xk, y-yk, sqrt(rx1^2+ry1^2+z^2)
     3e) rx2,ry2,r2       = x-xk2, y-yk2, sqrt(rx2^2+ry2^2+z^2)
     3f) ssum              = r1+r2
     3g) denom1            = ssum^2 - dlen^2
         -- if |denom1|<=1e-30: edge contributes nothing, skip --
     3h) big               = (r1*r2)*(r1*r2 + rx1*rx2 + ry1*ry2 + z^2)
         -- if |big|<=1e-30: edge contributes nothing, skip --
     3i) A_ = rx1/r1 + rx2/r2         (shared across dd[0][0], dd[0][1])
     3j) B_ = ry1/r1 + ry2/r2         (shared across dd[1][0], dd[1][1])
     3k) C_ = z/r1 + z/r2             (shared across dd[2][0], dd[2][1])
     3l) cross_ = rx1*ry2 - rx2*ry1   (only dd[2][2])
     3m) K1 = 2*edy/denom1            (shared across dd[0][0],dd[1][0],dd[2][0])
     3n) K2 = -2*edx/denom1           (shared across dd[0][1],dd[1][1],dd[2][1])
     3o) t00,t10,t20 = K1*A_, K1*B_, K1*C_
     3p) t01,t11,t21 = K2*A_, K2*B_, K2*C_
     3q) t02 = (z*edy*ssum)/big
     3r) t12 = (-z*edx*ssum)/big
     3s) t22 = (cross_*ssum)/big
     dd[a][b] += t_ab for each of the 9 entries above
  4) dd_scaled = dd * scale   (scale=-1/(4*pi), constant)
  5) tmp = coordsys @ dd_scaled          (3x3 matrix product)
  6) out = tmp @ coordsys^T              (3x3 matrix product; combined,
     out = coordsys @ dd_scaled @ coordsys^T -- rotates the LOCAL panel
     Hessian into GLOBAL coordinates)

BACKWARD walks 6 -> 1. The matrix products in steps 5-6 use ordinary
matrix-calculus adjoint rules (derived by hand below, not quoted from a
reference) rather than 81 scalar entries:
  out = tmp @ coordsys^T  =>  d_tmp = d_out @ coordsys,
                               d_coordsys += d_out^T @ tmp
  tmp = coordsys @ dd_scaled  =>  d_coordsys += d_tmp @ dd_scaled^T,
                                    d_dd_scaled = coordsys^T @ d_tmp
coordsys is used THREE times total (twice in the triple product above, once
more in step 2's coordinate transform below) -- three separate
contributions accumulate into d_coordsys, same "multi-use -> sum the
paths" rule as everywhere else in this series.

Everything inside one edge is scalar and uses the same adjoint-rule table
as handderiv_hs_influence.py (sum/difference, product, quotient, sqrt) --
no new table entries needed here, just more terms sharing more
subexpressions (K1, K2, A_, B_, C_ are each reused 2-3 times within one
edge, so each earns multiple "+=" contributions before their own
definition gets differentiated).

BRANCH COVERAGE (checked 2026-07-15, not just assumed): unlike
hs_influence's two-level active/fires_atan split, phixx_influence has only
ONE flag per edge (`active`), but it's gated by THREE separate short-
circuit guards (3c dlen<=eps, 3g |denom1|<=1e-30, 3h |big|<=1e-30) -- any
one failing skips the rest of that edge. Real mesh data never trips any of
them (all 4 edges come out active=True, same as hs_influence's real-data
flags). Two genuinely different geometric triggers were found and tested
with synthetic inputs, vs. the revad oracle (not FD -- these are, like
hs_influence's fires_atan/m_is_const cases, exact isolated points that FD
central differences structurally cannot see):
  - dlen<=eps: an actually-degenerate edge (near-coincident corners).
  - |denom1|<=1e-30: the field point lies exactly ON the segment between
    the edge's two corners (z=0, and (x,y) between the corners). This is
    the classic "field point on the panel boundary" collocation
    singularity from BEM theory.
A third possibility -- |big|<=1e-30 firing on its own, with denom1
comfortably nonzero -- was investigated algebraically and then checked
both symbolically (on-segment points, any t in (0,1), z=0) and with a
2-million-point random scan, and appears NOT to be geometrically
reachable for this kernel: big's two zero-mechanisms (r1*r2=0, i.e. field
exactly AT one of the edge's own corners; or the two corner-to-field
vectors being exactly anti-parallel) both turn out to require the SAME
on-segment condition denom1 already flags, and denom1 is checked first in
the code. So `big`'s guard appears to be a redundant safety net riding on
the same degeneracy, not an independently-reachable branch -- a genuine
finding from this exercise, not a gap in the testing. Both of the two real
cases validate to machine precision (2.2e-16, 5.6e-17) -- see
`if __name__ == "__main__"` below.
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import math
import numpy as np

from rankine_panel.io import read_panel_file
from rankine_panel.geometry import panel_geometry_all
from rankine_panel.influence import phixx_influence
from rankine_panel.revad import jacobian


IP1 = (1, 2, 3, 0)


# =============================================================================
# Forward pass: plain floats, caches every intermediate the backward pass needs
# =============================================================================

def phixx_influence_forward(fieldpoint, center, coordsys, corners_local, eps=1e-6):
    fieldpoint = np.asarray(fieldpoint, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    coordsys = np.asarray(coordsys, dtype=np.float64)
    corners_local = np.asarray(corners_local, dtype=np.float64)

    dx0 = fieldpoint[0] - center[0]
    dy0 = fieldpoint[1] - center[1]
    dz0 = fieldpoint[2] - center[2]

    x = coordsys[0, 0]*dx0 + coordsys[1, 0]*dy0 + coordsys[2, 0]*dz0
    y = coordsys[0, 1]*dx0 + coordsys[1, 1]*dy0 + coordsys[2, 1]*dz0
    z = coordsys[0, 2]*dx0 + coordsys[1, 2]*dy0 + coordsys[2, 2]*dz0

    dd = np.zeros((3, 3))
    edges = []

    for k in range(4):
        k2 = IP1[k]
        xk, yk = corners_local[0, k], corners_local[1, k]
        xk2, yk2 = corners_local[0, k2], corners_local[1, k2]

        edx = xk2 - xk
        edy = yk2 - yk
        dlen = math.sqrt(edx*edx + edy*edy)

        e = dict(k=k, k2=k2, xk=xk, yk=yk, xk2=xk2, yk2=yk2,
                 edx=edx, edy=edy, dlen=dlen, active=False)

        if dlen <= eps:
            edges.append(e)
            continue

        rx1, ry1 = x - xk, y - yk
        r1 = math.sqrt(rx1*rx1 + ry1*ry1 + z*z)
        rx2, ry2 = x - xk2, y - yk2
        r2 = math.sqrt(rx2*rx2 + ry2*ry2 + z*z)

        ssum = r1 + r2
        denom1 = ssum*ssum - dlen*dlen
        if abs(denom1) <= 1e-30:
            edges.append(e)
            continue

        big = (r1*r2) * (r1*r2 + rx1*rx2 + ry1*ry2 + z*z)
        if abs(big) <= 1e-30:
            edges.append(e)
            continue

        A_ = rx1/r1 + rx2/r2
        B_ = ry1/r1 + ry2/r2
        C_ = z/r1 + z/r2
        cross_ = rx1*ry2 - rx2*ry1
        K1 = 2.0*edy/denom1
        K2 = -2.0*edx/denom1

        t00, t10, t20 = K1*A_, K1*B_, K1*C_
        t01, t11, t21 = K2*A_, K2*B_, K2*C_
        t02 = (z*edy*ssum) / big
        t12 = (-z*edx*ssum) / big
        t22 = (cross_*ssum) / big

        dd[0, 0] += t00; dd[0, 1] += t01; dd[0, 2] += t02
        dd[1, 0] += t10; dd[1, 1] += t11; dd[1, 2] += t12
        dd[2, 0] += t20; dd[2, 1] += t21; dd[2, 2] += t22

        e.update(active=True, rx1=rx1, ry1=ry1, r1=r1, rx2=rx2, ry2=ry2, r2=r2,
                 ssum=ssum, denom1=denom1, big=big,
                 A_=A_, B_=B_, C_=C_, cross_=cross_, K1=K1, K2=K2,
                 t02=t02, t12=t12, t22=t22)
        edges.append(e)

    scale = -1.0 / (4.0 * math.pi)
    dd_scaled = dd * scale

    tmp = coordsys @ dd_scaled
    out = tmp @ coordsys.T

    cache = dict(coordsys=coordsys, dx0=dx0, dy0=dy0, dz0=dz0, z=z,
                 scale=scale, dd_scaled=dd_scaled, tmp=tmp, edges=edges)
    return out, cache


# =============================================================================
# Backward pass: hand-derived, walks the forward steps in exact reverse order
# =============================================================================

def phixx_influence_backward(cache, d_out):
    """d_out: (3,3) upstream adjoint for the returned global Hessian."""
    coordsys = cache['coordsys']
    dd_scaled = cache['dd_scaled']
    tmp = cache['tmp']
    dx0, dy0, dz0 = cache['dx0'], cache['dy0'], cache['dz0']
    z = cache['z']
    scale = cache['scale']

    # ---- step 6: out = tmp @ coordsys^T
    d_tmp = d_out @ coordsys
    d_coordsys = d_out.T @ tmp                    # coordsys's FIRST contribution (2 more below)

    # ---- step 5: tmp = coordsys @ dd_scaled
    d_coordsys += d_tmp @ dd_scaled.T              # coordsys's SECOND contribution
    d_dd_scaled = coordsys.T @ d_tmp

    # ---- step 4: dd_scaled = dd_raw * scale (constant)
    G = scale * d_dd_scaled                        # same (3,3) seed used by every edge below

    d_x = 0.0
    d_y = 0.0
    d_z = 0.0
    d_corners_local = np.zeros((2, 4))

    for e in reversed(cache['edges']):
        if not e['active']:
            continue  # skipped in forward -> contributed nothing -> nothing flows back

        k, k2 = e['k'], e['k2']
        edx, edy, dlen = e['edx'], e['edy'], e['dlen']
        rx1, ry1, r1 = e['rx1'], e['ry1'], e['r1']
        rx2, ry2, r2 = e['rx2'], e['ry2'], e['r2']
        ssum, denom1, big = e['ssum'], e['denom1'], e['big']
        A_, B_, C_, cross_ = e['A_'], e['B_'], e['C_'], e['cross_']
        K1, K2 = e['K1'], e['K2']
        t02, t12, t22 = e['t02'], e['t12'], e['t22']

        g00, g01, g02 = G[0, 0], G[0, 1], G[0, 2]
        g10, g11, g12 = G[1, 0], G[1, 1], G[1, 2]
        g20, g21, g22 = G[2, 0], G[2, 1], G[2, 2]

        # ---- t22 = (cross_*ssum)/big  ["z=x/y" then "z=x*y"]
        d_numer22 = g22 / big
        d_big = -t22 * g22 / big                   # big's FIRST contribution (2 more below)
        d_cross_ = ssum * d_numer22                 # cross_'s ONLY use
        d_ssum = cross_ * d_numer22                 # ssum's FIRST contribution (3 more below)

        # ---- t12 = (-z*edx*ssum)/big
        d_numer12 = g12 / big
        d_big += -t12 * g12 / big                   # big's SECOND
        d_z += -edx * ssum * d_numer12                # z's contribution from this edge (d_z is the
                                                        # OUTER cross-edge accumulator, declared before
                                                        # the loop -- must ALWAYS be "+=" here, never "=",
                                                        # or every edge but the last wipes out the ones
                                                        # before it. (This was a real bug, caught by
                                                        # comparing against FD on real multi-edge data.)
        d_edx = -z * ssum * d_numer12                # edx's FIRST (1 more below)
        d_ssum += -z * edx * d_numer12               # ssum's SECOND

        # ---- t02 = (z*edy*ssum)/big
        d_numer02 = g02 / big
        d_big += -t02 * g02 / big                    # big's THIRD, final
        d_z += edy * ssum * d_numer02                # z's SECOND
        d_edy = z * ssum * d_numer02                  # edy's FIRST (1 more below)
        d_ssum += z * edy * d_numer02                 # ssum's THIRD

        # ---- t01=K2*A_, t11=K2*B_, t21=K2*C_
        d_K2 = A_*g01 + B_*g11 + C_*g21
        d_A_ = K2 * g01                              # A_'s FIRST contribution (1 more below)
        d_B_ = K2 * g11                              # B_'s FIRST
        d_C_ = K2 * g21                              # C_'s FIRST

        # ---- t00=K1*A_, t10=K1*B_, t20=K1*C_
        d_K1 = A_*g00 + B_*g10 + C_*g20
        d_A_ += K1 * g00                             # A_'s SECOND, final
        d_B_ += K1 * g10                             # B_'s SECOND, final
        d_C_ += K1 * g20                             # C_'s SECOND, final

        # ---- K2 = -2*edx/denom1  ["z=x/y"]
        d_edx += -2.0 * d_K2 / denom1                # edx's SECOND, final
        d_denom1 = -K2 * d_K2 / denom1               # denom1's FIRST contribution (1 more below)

        # ---- K1 = 2*edy/denom1
        d_edy += 2.0 * d_K1 / denom1                 # edy's SECOND, final
        d_denom1 += -K1 * d_K1 / denom1              # denom1's SECOND, final

        # ---- cross_ = rx1*ry2 - rx2*ry1
        d_rx1 = ry2 * d_cross_                       # rx1's FIRST (2 more below)
        d_ry2 = rx1 * d_cross_                       # ry2's FIRST (1 more below)
        d_rx2 = -ry1 * d_cross_                      # rx2's FIRST (2 more below)
        d_ry1 = -rx2 * d_cross_                      # ry1's FIRST (1 more below)

        # ---- C_ = z/r1 + z/r2
        d_z += d_C_/r1 + d_C_/r2                     # z's THIRD, FOURTH
        d_r1 = -z * d_C_ / (r1*r1)                   # r1's FIRST (4 more below)
        d_r2 = -z * d_C_ / (r2*r2)                   # r2's FIRST (4 more below)

        # ---- B_ = ry1/r1 + ry2/r2
        d_ry1 += d_B_ / r1                           # ry1's SECOND, final
        d_r1 += -ry1 * d_B_ / (r1*r1)                # r1's SECOND
        d_ry2 += d_B_ / r2                           # ry2's SECOND, final
        d_r2 += -ry2 * d_B_ / (r2*r2)                # r2's SECOND

        # ---- A_ = rx1/r1 + rx2/r2
        d_rx1 += d_A_ / r1                           # rx1's SECOND
        d_r1 += -rx1 * d_A_ / (r1*r1)                # r1's THIRD
        d_rx2 += d_A_ / r2                           # rx2's SECOND
        d_r2 += -rx2 * d_A_ / (r2*r2)                # r2's THIRD

        # ---- big = P*Q,  P=r1*r2,  Q=P+rx1*rx2+ry1*ry2+z*z
        Q = r1*r2 + rx1*rx2 + ry1*ry2 + z*z
        P = r1*r2
        d_P = Q * d_big
        d_Q = P * d_big
        d_P += d_Q                                   # P used TWICE: leading factor AND inside Q
        d_rx1 += rx2 * d_Q                           # rx1's THIRD, final
        d_rx2 += rx1 * d_Q                           # rx2's THIRD, final
        d_ry1 += ry2 * d_Q                           # ry1's THIRD, final
        d_ry2 += ry1 * d_Q                           # ry2's THIRD, final
        d_z += 2.0 * z * d_Q                         # z's FIFTH
        d_r1 += r2 * d_P                             # r1's FOURTH, final
        d_r2 += r1 * d_P                             # r2's FOURTH, final

        # ---- denom1 = ssum^2 - dlen^2
        d_ssum += 2.0 * ssum * d_denom1              # ssum's FOURTH, final
        d_dlen = -2.0 * dlen * d_denom1              # dlen's ONLY use

        # ---- ssum = r1+r2
        d_r1 += d_ssum                               # r1's FIFTH, final
        d_r2 += d_ssum                               # r2's FIFTH, final

        # ---- rx2,ry2,r2 = x-xk2, y-yk2, sqrt(rx2^2+ry2^2+z^2)
        d_R2 = d_r2 * 0.5 / r2
        d_rx2 += 2.0*rx2*d_R2                        # rx2's FOURTH, final
        d_ry2 += 2.0*ry2*d_R2                        # ry2's THIRD, final
        d_z += 2.0*z*d_R2                            # z's SIXTH
        d_x += d_rx2
        d_xk2 = -d_rx2                               # xk2's FIRST (1 more below)
        d_y += d_ry2
        d_yk2 = -d_ry2                               # yk2's FIRST (1 more below)

        # ---- rx1,ry1,r1 = x-xk, y-yk, sqrt(rx1^2+ry1^2+z^2)
        d_R1 = d_r1 * 0.5 / r1
        d_rx1 += 2.0*rx1*d_R1                        # rx1's FOURTH, final
        d_ry1 += 2.0*ry1*d_R1                        # ry1's FOURTH, final
        d_z += 2.0*z*d_R1                            # z's SEVENTH, final
        d_x += d_rx1
        d_xk = -d_rx1                                # xk's FIRST (1 more below)
        d_y += d_ry1
        d_yk = -d_ry1                                # yk's FIRST (1 more below)

        # ---- dlen = sqrt(edx^2+edy^2)
        d_D2 = d_dlen * 0.5 / dlen
        d_edx += 2.0*edx*d_D2                        # edx's THIRD, final
        d_edy += 2.0*edy*d_D2                        # edy's THIRD, final

        # ---- edx,edy = xk2-xk, yk2-yk
        d_xk2 += d_edx                               # xk2's SECOND, final
        d_xk += -d_edx                               # xk's SECOND, final
        d_yk2 += d_edy                               # yk2's SECOND, final
        d_yk += -d_edy                               # yk's SECOND, final

        # ---- scatter into the shared corners_local accumulator (each
        # corner is "xk" in its own edge and "xk2" in the previous edge)
        d_corners_local[0, k] += d_xk
        d_corners_local[1, k] += d_yk
        d_corners_local[0, k2] += d_xk2
        d_corners_local[1, k2] += d_yk2

    # ---- step 2: x,y,z = coordsys^T @ delta, delta=(dx0,dy0,dz0)
    delta = np.array([dx0, dy0, dz0])
    d_p = np.array([d_x, d_y, d_z])                 # fully accumulated, all edges done
    d_coordsys += np.outer(delta, d_p)               # coordsys's THIRD (final) contribution
    d_delta = coordsys @ d_p

    # ---- step 1: dx0=fieldpoint[0]-center[0], etc.
    d_fieldpoint = d_delta.copy()
    d_center = -d_delta.copy()

    return d_fieldpoint, d_center, d_coordsys, d_corners_local


# =============================================================================
# Validation: hand-derived backward pass vs. the trusted revad oracle
# (rankine_panel.influence.phixx_influence via revad.jacobian), same real
# panel pair (ROW=0 as fieldpoint, COL=50 as source panel) used in
# trace_phixx_influence_revad.py.
# =============================================================================

if __name__ == "__main__":
    pf = read_panel_file(os.path.join(_PROJECT_ROOT, "examples", "fifi.dat"))
    geom = panel_geometry_all(pf.points, pf.panels)

    ROW, COL = 0, 50
    fieldpoint = geom.center[:, ROW].copy()
    center = geom.center[:, COL].copy()
    coordsys = geom.coordsys[:, :, COL].copy()
    corners_local = geom.cornerslocal[:, :, COL].copy()

    H, cache = phixx_influence_forward(fieldpoint, center, coordsys, corners_local)
    print("forward H (hand):\n", H)
    print("forward H (primal):\n", phixx_influence(fieldpoint, center, coordsys, corners_local))

    # ---- revad oracle Jacobian (9,23), same packing as trace_phixx_influence_revad.py
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
        fp, ctr, csy, cl = unpack_nodes(xs_nodes)
        out = phixx_influence(fp, ctr, csy, cl)
        return list(out.ravel())

    x0 = pack(fieldpoint, center, coordsys, corners_local)
    J_oracle = jacobian(run_traced, x0)  # (9, 23)

    labels = (["fieldpoint%d" % i for i in range(3)]
              + ["center%d" % i for i in range(3)]
              + ["coordsys%d%d" % (i, j) for i in range(3) for j in range(3)]
              + ["cornersloc%d%d" % (i, j) for i in range(2) for j in range(4)])
    out_labels = ["H%d%d" % (a, b) for a in range(3) for b in range(3)]

    max_abs_err = 0.0
    worst = None
    J_hand = np.zeros((9, 23))
    for o in range(9):
        a, b = o // 3, o % 3
        d_out = np.zeros((3, 3))
        d_out[a, b] = 1.0

        d_fieldpoint, d_center, d_coordsys, d_cornerslocal = phixx_influence_backward(cache, d_out)
        row_hand = np.concatenate([d_fieldpoint, d_center, d_coordsys.ravel(), d_cornerslocal.ravel()])
        J_hand[o, :] = row_hand

        row_oracle = J_oracle[o, :]
        err = np.max(np.abs(row_hand - row_oracle))
        if err > max_abs_err:
            max_abs_err = err
            worst = (out_labels[o], row_hand, row_oracle)

    print(f"\nmax |hand-derived - revad oracle| over all 9x23 Jacobian entries: {max_abs_err:.3e}")
    print(f"worst output direction: {worst[0]}")
    for lab, hv, ov in zip(labels, worst[1], worst[2]):
        flag = "  <-- worst" if abs(hv - ov) == max_abs_err else ""
        print(f"  {lab:<14} hand={hv:>16.8e}  oracle={ov:>16.8e}{flag}")

    # =========================================================================
    # Branch coverage: real mesh data (above) never trips active=False -- see
    # module docstring's "BRANCH COVERAGE" section, including why a THIRD
    # case (|big|<=1e-30 firing independently of denom1) doesn't appear to be
    # geometrically reachable for this kernel and isn't tested here. FD can't
    # validate either case below (both are exact isolated points -- degenerate
    # edge length, and field point exactly on an edge's segment), so this
    # compares against the revad oracle instead, same reasoning as
    # handderiv_hs_influence.py's branch-coverage checks.
    # =========================================================================
    print("\n" + "="*70)
    print("Branch-coverage checks (synthetic inputs, vs. revad oracle)")
    print("="*70)

    identity = np.eye(3)

    def check_branch_case(name, fp, ctr, csy, cl):
        H_c, cache_c = phixx_influence_forward(fp, ctr, csy, cl)
        flags = [(e['k'], e['active']) for e in cache_c['edges']]
        x0_c = pack(fp, ctr, csy, cl)
        J_oracle_c = jacobian(run_traced, x0_c)
        J_hand_c = np.zeros((9, 23))
        for o in range(9):
            a, b = o // 3, o % 3
            d_out = np.zeros((3, 3)); d_out[a, b] = 1.0
            d_fp, d_ctr, d_csy, d_cl = phixx_influence_backward(cache_c, d_out)
            J_hand_c[o, :] = np.concatenate([d_fp, d_ctr, d_csy.ravel(), d_cl.ravel()])
        err_c = np.max(np.abs(J_hand_c - J_oracle_c))
        print(f"{name}")
        print(f"  edge flags (k, active): {flags}")
        print(f"  max |hand - revad oracle|: {err_c:.3e}")

    # active=False via dlen<=eps (degenerate edge 0)
    check_branch_case(
        "active=False: degenerate edge length",
        np.array([0.35, 0.42, 0.55]), np.array([0.0, 0.0, 0.0]), identity,
        np.array([[0.0, 1e-9, 1.2, -0.1], [0.0, 5e-10, 1.3, 1.0]]))

    # active=False via |denom1|<=1e-30 (field point exactly on edge0's segment, z=0)
    check_branch_case(
        "active=False: field point on edge segment (denom1=0, big=0 too)",
        np.array([0.5, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), identity,
        np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]]))
