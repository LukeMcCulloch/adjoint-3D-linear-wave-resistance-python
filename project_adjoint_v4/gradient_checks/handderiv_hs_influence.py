# -*- coding: utf-8 -*-
"""
HAND-DERIVED (not auto-traced) reverse-mode backward pass for hs_influence
(rankine_panel/influence.py::hs_influence, same formula as
rankine_panel/numba_kernels.py::hs_influence_nb).

(see hs_influence_revad from v3, hs_influence_revad in gradient_checks.py for the original function)

Same idea as handderiv_panel_geometry.py: no revad.Node, no dynamic graph,
no operator overloading -- a forward pass that computes the primal AND
caches every intermediate the backward pass needs, then a backward pass
that walks those steps in EXACT reverse order applying the chain rule by
hand. Checked against revad's automatic backward pass
(trace_hs_influence_revad.py / rankine_panel.influence.hs_influence, which
is itself generic over plain float64 and Node), not derived from it.

This forward pass is written in fully unrolled scalar form (dx0/dy0/dz0,
not coordsys.T @ (...)) specifically so every adjoint rule application is
visible -- it computes exactly the same numbers as the compact matrix form
in influence.py, just spelled out. See the note in
trace_hs_influence_revad.py's history: this unrolled style is how the
ORIGINAL hs_influence_revad (project_adjoint_v3) was written, before
influence.py's later consolidation switched to the compact `coordsys.T @
(fieldpoint-center)` form for the production/oracle code. Both forms
compute identical values; this file is a separate, from-scratch hand
derivation, not an extraction from either.

FORWARD STEPS (numbered):
  1) dx0,dy0,dz0 = fieldpoint - center                          [z=x-y] x3
  2) x = c00*dx0 + c10*dy0 + c20*dz0
     y = c01*dx0 + c11*dy0 + c21*dz0
     z = c02*dx0 + c12*dy0 + c22*dz0                              (p = coordsys^T @ delta)
  3) for k=0..3 (edge from corner k to corner k2=ip1[k]):
     3a) xk,yk,xk2,yk2   = corners_local columns k, k2            (reads only)
     3b) edx,edy          = xk2-xk, yk2-yk                        [z=x-y]
     3c) dlen              = sqrt(edx^2+edy^2)                     [z=sqrt(x)]
         -- if dlen<=eps: edge contributes NOTHING, skip to next k --
     3d) rx1,ry1,r1       = x-xk, y-yk, sqrt(rx1^2+ry1^2+z^2)
     3e) rx2,ry2,r2       = x-xk2, y-yk2, sqrt(rx2^2+ry2^2+z^2)
         (production recomputes r1/r2 a second time later for denom1/denom2
          -- mathematically identical, so this derivation uses ONE r1/r2,
          fed into BOTH downstream uses; a variable used twice just means
          two adjoint contributions land on it, same as `iv` or `m` below)
     3f) s,num,den        = r1+r2, s-dlen, s+dlen
         -- if num<=0 or den<=0: edge contributes nothing, skip --
     3g) frac,L           = num/den, log(frac)
     3h) term0,term1      = (edy/dlen)*L, (edx/dlen)*L
         dphi0 += term0 ;  dphi1 -= term1
     3i) m = edy/edx  (or a DATA-CONSTANT literal if |edx|<1e-30 -- that
         branch has NO gradient path back to edx/edy, by construction)
     3j) e1,h1            = rx1^2+z^2, rx1*ry1
     3k) e2,h2            = rx2^2+z^2, rx2*ry2
     3l) denom1,denom2    = z*r1, z*r2
         -- if not(|denom1|>1e-30 and |denom2|>1e-30): no atan term --
     3m) arg1,arg2        = (m*e1-h1)/denom1, (m*e2-h2)/denom2
     3n) atan1,atan2v     = atan(arg1), atan(arg2)
         dphi2 += atan1 - atan2v
  4) scale = -1/(4*pi) (constant);  dphi *= scale
  5) v = coordsys @ dphi                                          (vx,vy,vz)

BACKWARD walks 5 -> 1. Same adjoint-rule table as handderiv_panel_geometry.py,
plus two new entries used here:
  z = atan(x):      dx += dz/(1+x^2)
  z = log(x):        dx += dz/x
Every edge's contribution to dphi0/dphi1/dphi2 is one term in a SUM over
k=0..3 -- same "sum's adjoint broadcasts unchanged to every term" rule as
panel_geometry's edge sums. x, y, z, and corners_local are each touched by
MULTIPLE edges (x/y/z by every non-skipped edge; each corner by exactly two
edges -- once as "xk" in its own edge, once as "xk2" in the previous edge)
so their adjoints are accumulators that only get read once the edge loop
has fully finished writing into them -- exactly the `iv`/`jv` pattern from
panel_geometry, just with edges standing in for the loop over k.
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
from rankine_panel.influence import hs_influence
from rankine_panel.revad import jacobian


IP1 = (1, 2, 3, 0)


# =============================================================================
# Forward pass: plain floats, caches every intermediate the backward pass needs
# =============================================================================

def hs_influence_forward(fieldpoint, center, coordsys, corners_local, eps=1e-6):
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

    dphi0 = 0.0
    dphi1 = 0.0
    dphi2 = 0.0

    edges = []

    for k in range(4):
        k2 = IP1[k]
        xk, yk = corners_local[0, k], corners_local[1, k]
        xk2, yk2 = corners_local[0, k2], corners_local[1, k2]

        edx = xk2 - xk
        edy = yk2 - yk
        dlen = math.sqrt(edx*edx + edy*edy)

        e = dict(k=k, k2=k2, xk=xk, yk=yk, xk2=xk2, yk2=yk2,
                 edx=edx, edy=edy, dlen=dlen, active=False, fires_atan=False)

        if dlen <= eps:
            edges.append(e)
            continue

        rx1, ry1 = x - xk, y - yk
        r1 = math.sqrt(rx1*rx1 + ry1*ry1 + z*z)
        rx2, ry2 = x - xk2, y - yk2
        r2 = math.sqrt(rx2*rx2 + ry2*ry2 + z*z)

        s = r1 + r2
        num = s - dlen
        den = s + dlen
        if num <= 0.0 or den <= 0.0:
            edges.append(e)
            continue

        frac = num / den
        L = math.log(frac)
        term0 = (edy / dlen) * L
        term1 = (edx / dlen) * L
        dphi0 += term0
        dphi1 -= term1

        if abs(edx) < 1e-30:
            m = float(np.sign(edy) * 1e30)
            m_is_const = True
        else:
            m = edy / edx
            m_is_const = False

        e1 = rx1*rx1 + z*z
        h1 = rx1*ry1
        e2 = rx2*rx2 + z*z
        h2 = rx2*ry2
        denom1 = z*r1
        denom2 = z*r2

        e.update(active=True, rx1=rx1, ry1=ry1, r1=r1, rx2=rx2, ry2=ry2, r2=r2,
                 frac=frac, L=L, m=m, m_is_const=m_is_const,
                 e1=e1, h1=h1, e2=e2, h2=h2, denom1=denom1, denom2=denom2)

        if abs(denom1) > 1e-30 and abs(denom2) > 1e-30:
            arg1 = (m*e1 - h1) / denom1
            arg2 = (m*e2 - h2) / denom2
            atan1 = math.atan(arg1)
            atan2v = math.atan(arg2)
            dphi2 += atan1 - atan2v
            e.update(fires_atan=True, arg1=arg1, arg2=arg2)

        edges.append(e)

    scale = -1.0 / (4.0 * math.pi)
    dphi0 *= scale
    dphi1 *= scale
    dphi2 *= scale

    v = coordsys @ np.array([dphi0, dphi1, dphi2])
    
    
    # what is this cache?: (at this point it looks simpler to write the AST source to source machine)
    #
    # forward pass caches every intermediate
    cache = dict(coordsys=coordsys, dx0=dx0, dy0=dy0, dz0=dz0, z=z,
                 scale=scale, dphi0=dphi0, dphi1=dphi1, dphi2=dphi2,
                 edges=edges)
    return v, cache


# =============================================================================
# Backward pass: hand-derived, walks the forward steps in exact reverse order
# =============================================================================

def hs_influence_backward(cache, d_vx, d_vy, d_vz):
    
    #----------------------------------------------
    # set up the basic data at the leaf nodes:
    #
    coordsys = cache['coordsys']
    dx0, dy0, dz0 = cache['dx0'], cache['dy0'], cache['dz0']
    z = cache['z']
    scale = cache['scale']
    dphi_scaled = np.array([cache['dphi0'], cache['dphi1'], cache['dphi2']])

    # ---- step 5: v = coordsys @ dphi_scaled  (matrix-vector product,
    # v[i] = sum_j coordsys[i,j]*dphi[j]). Each coordsys[i,j] entry appears
    # in exactly ONE output v[i] -> one contribution each; each dphi[j]
    # appears in ALL THREE outputs (once per row) -> three contributions,
    # i.e. d_dphi = coordsys^T @ d_v.
    d_v = np.array([d_vx, d_vy, d_vz])
    d_coordsys = np.outer(d_v, dphi_scaled)       # coordsys's FIRST contribution (step 2 adds a second)
    d_dphi_scaled = coordsys.T @ d_v

    # ---- step 4: dphi_raw *= scale (scale is a constant, not a design var)
    d_dphi_raw = scale * d_dphi_scaled
    g0, g1, g2 = d_dphi_raw[0], d_dphi_raw[1], d_dphi_raw[2]

    # ---- step 3: the 4-edge loop, walked in reverse. Order AMONG edges
    # doesn't matter (independent additive terms), but x/y/z and
    # corners_local accumulate ACROSS edges, so those accumulators live
    # outside the loop and only get consumed once every edge has added its
    # share -- read only after step 2 below, once fully accumulated.
    d_x = 0.0
    d_y = 0.0
    d_z = 0.0
    d_corners_local = np.zeros((2, 4))

    for e in reversed(cache['edges']):
        if not e['active']:
            continue  # skipped in forward (degenerate edge, or num/den<=0)
                      # -> contributed nothing to dphi0/1/2 -> nothing flows back

        k, k2 = e['k'], e['k2']
        edx, edy, dlen = e['edx'], e['edy'], e['dlen']
        rx1, ry1, r1 = e['rx1'], e['ry1'], e['r1']
        rx2, ry2, r2 = e['rx2'], e['ry2'], e['r2']
        frac, L = e['frac'], e['L']

        d_edx = 0.0
        d_edy = 0.0
        d_dlen = 0.0
        d_r1 = 0.0
        d_r2 = 0.0
        d_rx1 = 0.0; d_ry1 = 0.0; d_rx2 = 0.0; d_ry2 = 0.0

        # ---- step (n)/(m): dphi2 += atan1 - atan2v (only if it fired)
        # atan1=atan(arg1): d(arg1) += d(atan1)/(1+arg1^2)  ["z=atan(x)"]
        # arg1 = w1/denom1, w1 = m*e1-h1  ["z=x/y" then "z=x*y","z=x-y"]
        if e['fires_atan']:
            m, e1, h1, e2, h2 = e['m'], e['e1'], e['h1'], e['e2'], e['h2']
            denom1, denom2 = e['denom1'], e['denom2']
            arg1, arg2 = e['arg1'], e['arg2']

            d_atan1 = g2
            d_atan2v = -g2
            d_arg1 = d_atan1 / (1.0 + arg1*arg1)
            d_arg2 = d_atan2v / (1.0 + arg2*arg2)

            d_w1 = d_arg1 / denom1
            d_denom1 = -arg1 * d_arg1 / denom1
            d_m = e1 * d_w1
            d_e1 = m * d_w1
            d_h1 = -d_w1

            d_w2 = d_arg2 / denom2
            d_denom2 = -arg2 * d_arg2 / denom2
            d_m += e2 * d_w2                     # m's SECOND contribution (used for both edge-corners)
            d_e2 = m * d_w2
            d_h2 = -d_w2

            # ---- step (l): denom1=z*r1, denom2=z*r2  ["z=x*y"]
            d_z += r1 * d_denom1
            d_r1 += z * d_denom1
            d_z += r2 * d_denom2
            d_r2 += z * d_denom2

            # ---- step (k): e2=rx2^2+z^2, h2=rx2*ry2
            d_rx2 += 2*rx2*d_e2
            d_z += 2*z*d_e2
            d_rx2 += ry2*d_h2
            d_ry2 += rx2*d_h2

            # ---- step (j): e1=rx1^2+z^2, h1=rx1*ry1
            d_rx1 += 2*rx1*d_e1
            d_z += 2*z*d_e1
            d_rx1 += ry1*d_h1
            d_ry1 += rx1*d_h1

            # ---- step (i): m=edy/edx (skip entirely if m was a data-constant
            # literal in forward -- that branch has no dependency on edx/edy)
            if not e['m_is_const']:
                d_edy += d_m/edx
                d_edx += -m*d_m/edx

        # ---- step (h): term0=(edy/dlen)*L, term1=(edx/dlen)*L
        # dphi0 += term0 -> d(term0)=g0 ; dphi1 -= term1 -> d(term1)=-g1
        # (sum's adjoint broadcasts unchanged to every edge's own term)
        d_term0 = g0
        d_term1 = -g1
        p_h = edy / dlen
        d_p_h = L * d_term0
        d_L = p_h * d_term0                      # L's FIRST contribution (term1 below adds a second)
        d_edy += d_p_h / dlen
        d_dlen += -p_h * d_p_h / dlen
        q_h = edx / dlen
        d_q_h = L * d_term1
        d_L += q_h * d_term1                     # L's SECOND (final) contribution
        d_edx += d_q_h / dlen
        d_dlen += -q_h * d_q_h / dlen

        # ---- step (g): frac=num/den, L=log(frac)  ["z=log(x)": dx+=dz/x]
        # den isn't cached directly -- s and dlen are, so reconstruct it
        # (den = s+dlen, exactly how it was built in forward step (f))
        # rather than dividing back out of frac, which is fragile near frac~0.
        d_frac = d_L / frac
        s = r1 + r2
        den = s + dlen
        d_num = d_frac / den
        d_den = -frac * d_frac / den

        # ---- step (f): s=r1+r2, num=s-dlen, den=s+dlen  ["z=x-y","z=x+y"]
        d_s = d_num + d_den
        d_dlen += -d_num + d_den
        d_r1 += d_s
        d_r2 += d_s

        # ---- step (e): rx2=x-xk2, ry2=y-yk2, r2=sqrt(rx2^2+ry2^2+z^2)
        d_R2 = d_r2 * 0.5 / r2
        d_rx2 += 2*rx2*d_R2
        d_ry2 += 2*ry2*d_R2
        d_z += 2*z*d_R2
        d_x += d_rx2
        d_xk2 = -d_rx2
        d_y += d_ry2
        d_yk2 = -d_ry2

        # ---- step (d): rx1=x-xk, ry1=y-yk, r1=sqrt(rx1^2+ry1^2+z^2)
        d_R1 = d_r1 * 0.5 / r1
        d_rx1 += 2*rx1*d_R1
        d_ry1 += 2*ry1*d_R1
        d_z += 2*z*d_R1
        d_x += d_rx1
        d_xk = -d_rx1
        d_y += d_ry1
        d_yk = -d_ry1

        # ---- step (c): dlen=sqrt(edx^2+edy^2)  ["z=sqrt(x)"]
        d_D2 = d_dlen * 0.5 / dlen
        d_edx += 2*edx*d_D2
        d_edy += 2*edy*d_D2

        # ---- step (b): edx=xk2-xk, edy=yk2-yk  ["z=x-y"]
        d_xk2 += d_edx
        d_xk += -d_edx
        d_yk2 += d_edy
        d_yk += -d_edy

        # ---- step (a): scatter into the shared corners_local accumulator.
        # Each corner is "xk" in its OWN edge and "xk2" in the PREVIOUS edge
        # -- two different edges' contributions land on the same global
        # column, hence +=, same pattern as the ip1-scatter in geometry.
        d_corners_local[0, k] += d_xk
        d_corners_local[1, k] += d_yk
        d_corners_local[0, k2] += d_xk2
        d_corners_local[1, k2] += d_yk2

    # ---- step 2: x = c00*dx0+c10*dy0+c20*dz0  (and y, z rows)
    # This is p = coordsys^T @ delta, delta=(dx0,dy0,dz0): coordsys[i,j]
    # appears once, in p[j]'s formula, multiplied by delta[i]
    # -> d_coordsys[i,j] += delta[i]*d_p[j], i.e. d_coordsys += outer(delta, d_p).
    # delta[i] appears in all three of x,y,z (once per column j)
    # -> d_delta = coordsys @ d_p.
    delta = np.array([dx0, dy0, dz0])
    d_p = np.array([d_x, d_y, d_z])              # fully accumulated, all edges done
    d_coordsys += np.outer(delta, d_p)            # coordsys's SECOND (final) contribution
    d_delta = coordsys @ d_p

    # ---- step 1: dx0=fieldpoint[0]-center[0], etc.  ["z=x-y"]
    d_fieldpoint = d_delta.copy()
    d_center = -d_delta.copy()

    return d_fieldpoint, d_center, d_coordsys, d_corners_local


# =============================================================================
# Validation: hand-derived backward pass vs. the trusted revad oracle
# (rankine_panel.influence.hs_influence, differentiated via revad.jacobian),
# same real panel pair (ROW=0 as fieldpoint, COL=50 as source panel) used in
# trace_hs_influence_revad.py.
# =============================================================================

if __name__ == "__main__":
    pf = read_panel_file(os.path.join(_PROJECT_ROOT, "examples", "fifi.dat"))
    geom = panel_geometry_all(pf.points, pf.panels)

    ROW, COL = 0, 50
    fieldpoint = geom.center[:, ROW].copy()
    center = geom.center[:, COL].copy()
    coordsys = geom.coordsys[:, :, COL].copy()
    corners_local = geom.cornerslocal[:, :, COL].copy()

    v, cache = hs_influence_forward(fieldpoint, center, coordsys, corners_local)
    print("forward v (hand):  ", v)
    print("forward v (primal):", hs_influence(fieldpoint, center, coordsys, corners_local))

    # ---- revad oracle Jacobian (3,23), same packing as trace_hs_influence_revad.py
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
        return list(hs_influence(fp, ctr, csy, cl))

    x0 = pack(fieldpoint, center, coordsys, corners_local)
    J_oracle = jacobian(run_traced, x0)  # (3, 23)

    labels = (["fieldpoint%d" % i for i in range(3)]
              + ["center%d" % i for i in range(3)]
              + ["coordsys%d%d" % (i, j) for i in range(3) for j in range(3)]
              + ["cornersloc%d%d" % (i, j) for i in range(2) for j in range(4)])

    max_abs_err = 0.0
    worst = None
    J_hand = np.zeros((3, 23))
    out_names = ["vx", "vy", "vz"]
    for o in range(3):
        d_vx = 1.0 if o == 0 else 0.0
        d_vy = 1.0 if o == 1 else 0.0
        d_vz = 1.0 if o == 2 else 0.0

        d_fieldpoint, d_center, d_coordsys, d_cornerslocal = hs_influence_backward(cache, d_vx, d_vy, d_vz)
        row_hand = np.concatenate([d_fieldpoint, d_center, d_coordsys.ravel(), d_cornerslocal.ravel()])
        J_hand[o, :] = row_hand

        row_oracle = J_oracle[o, :]
        err = np.max(np.abs(row_hand - row_oracle))
        if err > max_abs_err:
            max_abs_err = err
            worst = (out_names[o], row_hand, row_oracle)

    print(f"\nmax |hand-derived - revad oracle| over all 3x23 Jacobian entries: {max_abs_err:.3e}")
    print(f"worst output direction: {worst[0]}")
    for lab, hv, ov in zip(labels, worst[1], worst[2]):
        flag = "  <-- worst" if abs(hv - ov) == max_abs_err else ""
        print(f"  {lab:<14} hand={hv:>16.8e}  oracle={ov:>16.8e}{flag}")
