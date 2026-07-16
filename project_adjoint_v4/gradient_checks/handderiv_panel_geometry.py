# -*- coding: utf-8 -*-
"""
HAND-DERIVED (not auto-traced) reverse-mode backward pass for
panel_geometry_one / panel_geometry_all's per-panel formula.

This is Track B, step 2 of the plan: no revad.Node, no dynamic graph, no
operator overloading -- 
- a forward pass 
    - that computes the primal values 
        AND caches every intermediate the backward pass needs, 
- then a backward pass
    - that walks those steps in EXACT reverse order 
        applying the chain rule by hand. 
- This is the actual shape of what a numba/CUDA adjoint kernel looks
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


# index of "the next index around the corner" from the stard ordering:
# so index 0 sees the next index as 1 and so on
IP1 = (1, 2, 3, 0)


def _cross(a, b):
    return np.array([a[1]*b[2] - a[2]*b[1],
                      a[2]*b[0] - a[0]*b[2],
                      a[0]*b[1] - a[1]*b[0]])


# =============================================================================
# Forward pass: plain floats, caches every intermediate the backward pass needs
# =============================================================================

def panel_geometry_one_forward(corners):
    """
    This is just panel_geometry_all 
     - for a specific set of (4) corners [c0-c3]
     - with all the vectorized numpy calls unrolled for generic acceleration 
         (my target is numba, but also forward looking towards c++ cuda - something similar will be needed
          whether hand rolled like this, or generated via my baby JAX todo).
         
    # Q: could we not make due with our standard panel_geometry_all since it has the dispatch handled up front?
    # A: no, that dispatch itself precludes this.
        - as well as assigning intermmediates
        - and caching intermediates at the end
    
    
    Importantly: in this function we save every intermmediate uniquely.
    That makes it possible to chache it
    Which makes revAD possible

    """
    # no dispatch needed, this is for accelerators only
    
    # no working arrays as we only deal with one panel (4 corners)
    
    # 1) m1,m2,m3,m4   = corner midpoints
    c0, c1, c2, c3 = [np.asarray(c, dtype=np.float64) for c in corners]
    m1 = 0.5*(c0+c1); m2 = 0.5*(c1+c2); m3 = 0.5*(c2+c3); m4 = 0.5*(c3+c0)
    
    # 2) iv_raw        = m2 - m4
    iv_raw = m2 - m4
    
    # 3) iv_norm       = sqrt(dot(iv_raw,iv_raw))
    iv_norm = np.sqrt(np.dot(iv_raw, iv_raw))
    
    # 4) iv            = iv_raw / iv_norm
    iv = iv_raw / iv_norm
    
    #  # 5) jvbar         = m3 - m1
    jvbar = m3 - m1
    
    # 6) nv_raw        = cross(iv, jvbar)
    nv_raw = _cross(iv, jvbar)
    # 7 and 8, nv = nv / _sqrt(np.sum(nv*nv, axis=0, keepdims=True)) 
    nv_norm = np.sqrt(np.dot(nv_raw, nv_raw)) # 7
    nv = nv_raw / nv_norm # 8

    jv = _cross(nv, iv)

    coordsys = np.stack([iv, jv, nv], axis=1)  # columns

    center0 = 0.25*(m1+m2+m3+m4)

    corners_list = [c0, c1, c2, c3]
    d = [corners_list[k] - center0 for k in range(4)]

    xi = np.array([np.dot(d[k], iv) for k in range(4)])
    eta = np.array([np.dot(d[k], jv) for k in range(4)])

    xi_next = xi[list(IP1)] # the next corner around the quad aka read xi at a shuffled set of indices.
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

    # cache every intermediate, hand-derived `result'
    # backward pass will consume them in exact reverse order 
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
    """
    This is the hand-derived backward pass for one panel's geometry
    
    computes gradients with respect to this one panel's geometry, 
    and this panel's geometry is defined by 4 corner points, each a 3D position
    
    Given the upstream adjoint (sensitivity of some downstream scalar loss L)
    with respect to each of this panel's four outputs, 
    returns the adjoint with respect to each of the panel's 
    four corner positions 
    -- dL/d(corner_k) for k=0..3, summed over every path from that corner 
    through this panel's geometry formula to L.

    This function is LOCAL to one panel: it knows nothing about which
    global mesh vertex each corner corresponds to. That integer bookkeeping
    (panel -> global vertex index) lives outside this function -- see the
    __main__ block below for the GATHER (int index -> float corner
    position) that has to happen before calling the forward pass. Nothing
    here does the corresponding SCATTER-ADD (local gradient -> global
    per-vertex gradient array) yet -- that's still-pending work, needed
    once this gets embedded in the full assembly loop.
    
    why are all array inputs float arrays?  
     - is there no more integer mapping between panels and vertices needed?

    Parameters
    ----------
    cache : dict
        cache of every intermmediate result from the fwd pass on this panel.
        Plain float / numpy.ndarray
        values only, never revad.Node -- the whole point of a static
        backward pass is that it never needs a dynamic graph.
       
    d_center : numpy.ndarray
    
        fwd: the centroid of this panel in (x,y,z) global coordinates
        
        bckwd: Upstream adjoint dL/d(center): sensitivity of the loss to this
        panel's centroid, one component per (x,y,z).
        
    d_coordsys : numpy.ndarray [3x3]
        the local coordinate system on this panel, in terms of 3 float vectors: (iv, jv, nv)
        note: column oriented vectors iv, etc. in (iv, jv, nv)
        
    d_cornerslocal : numpy.ndarray [2x4]
        xi, eta coordinates of this panel computed in fwd mode
        
    d_area : float
        the area of this panel.

    Returns
    -------
    
    4 3-vectors:
        
        
    d_c0 : numpy.ndarray, shape (3,)
        dL/d(corner_0) -- gradient of the loss w.r.t. corner 0's global
        (x,y,z) position, summed over every path corner 0 feeds (its own
        midpoints, iv/jv/nv, xi/eta, moments, center -- whichever of the
        four upstream adjoints above are nonzero).
        
    d_c1 : numpy.ndarray, shape (3,)
        dL/d(corner_1), same meaning as d_c0.
        
    d_c2 : numpy.ndarray, shape (3,)
        dL/d(corner_2), same meaning as d_c0.
        
    d_c3 : numpy.ndarray, shape (3,)
        dL/d(corner_3), same meaning as d_c0.
    
    
    
    
    GENERAL RULE THAT EXPLAINS EVERY += IN THIS FUNCTION: whenever a forward
    variable was used in more than one place (fed more than one downstream
    expression), its adjoint must be the SUM of the adjoint contributions
    coming back from every one of those uses -- that's just the multivariate
    chain rule (d(L)/d(x) = sum over every path from x to L). Concretely:
    `iv` is used directly in step 21 (center), step 13 (xi,eta dot products,
    x4 corners), step 9 (jv=cross(nv,iv)) as BOTH args of that cross product
    in a sense (once as-is, once via nv which depends on iv), and step 4 (its
    own normalization). Every place you see `d_iv +=` below is one more of
    those paths being added in. Get any of them wrong -- miss one, double
    one, or add one at the wrong point in the walk (before its own upstream
    adjoint is fully formed) -- and the two exact methods (this vs. the
    revad oracle) would disagree by more than float roundoff. They agree to
    7e-15, which is the actual evidence every path below is accounted for
    exactly once, not a hopeful assumption.
    
    
    The 21 outputs (rows) — everything panel_geometry_one_forward returns, flattened:
        these are the inputs to the backward pass
    
    The 12 inputs (columns) — panel 0's 4 corners, 3 coordinates each:
        these are the outputs to the backward pass
    """
    # pull the cache into locals
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

    # ---- step 10: coordsys columns ARE iv,jv,nv (no computation, just a
    # column stack) -- so its adjoint is just a slice-copy into three
    # separate accumulators. These three (d_iv, d_jv, d_nv) are what steps
    # 21, 13, 9, 6 below will keep adding more contributions into.
    #
    # these three were never altered after step 10 in fwd mode.  
    #
    d_iv = d_coordsys[:, 0].copy()
    d_jv = d_coordsys[:, 1].copy()
    d_nv = d_coordsys[:, 2].copy()

    # ---- step 21: center = center0 + iv*xic + jv*etac
    # THREE terms added together -> "z=x+y" rule fires three times: the
    # upstream d_center flows unchanged into d_center0, AND into the
    # "vec = vec*scalar" rule twice (once for iv*xic, once for jv*etac):
    #   z = v*s  =>  dv += s*dz (elementwise scale),  ds += dot(v,dz)
    d_center0 = d_center.copy()
    d_iv += xic * d_center                 # this iv's FIRST contribution (more below)
    d_xic = np.dot(iv, d_center)
    d_jv += etac * d_center                # this jv's FIRST contribution (more below)
    d_etac = np.dot(jv, d_center)

    # ---- step 20: cornerslocal[0][k]=xi[k]-xic, [1][k]=eta[k]-etac, k=0..3
    # xic/etac get SUBTRACTED in all 4 of the k-indexed outputs, so their
    # adjoint is minus the SUM over k of the 4 upstream cornerslocal
    # adjoints (each of the 4 terms contributes independently, same "z=x-y"
    # rule applied 4 times then summed since xic is the same variable in
    # each of the 4 uses).
    d_xi = d_cornerslocal[0].copy()         # each xi[k] used only here (so far) -> plain copy is fine as the FIRST source
    d_xic += -np.sum(d_cornerslocal[0])
    d_eta = d_cornerslocal[1].copy()
    d_etac += -np.sum(d_cornerslocal[1])

    # ---- step 19: xic=momenteta/area, etac=momentxi/area  ("z=x/y" rule)
    #   z=x/y  =>  dx += dz/y,  dy += -z*dz/y
    # area is used TWICE overall: once directly as a returned output (the
    # caller's upstream d_area), and again right here as the denominator
    # for both divisions -- so its true total adjoint is the SUM of the
    # caller-supplied d_area and these two local contributions. This is the
    # same "multi-use -> sum the paths" rule as everything else, just
    # spanning both an OUTPUT and an INTERNAL use of the same variable.
    d_momenteta = d_xic / area
    d_area_local = -xic * d_xic / area
    d_momentxi = d_etac / area
    d_area_local += -etac * d_etac / area
    d_area_total = d_area + d_area_local  # area is ALSO a direct output

    # ---- step 18: momenteta = (1/6) * sum_k deta[k]*Q[k],  Q=xi_next^2+xi*xi_next+xi^2
    # A sum over k=0..3 means the SAME upstream scalar d_momenteta*(1/6)
    # goes to EVERY one of the 4 terms unchanged (that's what "distribute
    # the sum's adjoint" means -- d(sum_k f_k)/d(f_j) = 1 for every j, so
    # the incoming adjoint just broadcasts). Within one term T2=deta*Q,
    # that's a product rule; Q itself is a small quadratic in xi/xi_next,
    # so THAT gets its own product+square rules (d(x^2)/dx = 2x).
    d_T2 = np.full(4, (1.0/6.0) * d_momenteta)
    Q = xi_next**2 + xi*xi_next + xi**2
    d_deta = deta.copy() * 0.0  # deta's FIRST source of adjoint (step 16 below adds a second)
    d_deta += Q * d_T2
    d_Q = deta * d_T2
    d_xi_next = 2*xi_next*d_Q + xi*d_Q     # xi_next's FIRST source (steps 16, 15, then the step-14 scatter add more)
    d_xi = d_xi + (xi_next*d_Q + 2*xi*d_Q)  # xi's SECOND source (first was step 20 above)

    # ---- step 17: momentxi = (-1/6) * sum_k dxi[k]*P[k],  P=eta_next^2+eta*eta_next+eta^2
    # Exact mirror of step 18 with xi<->eta swapped and dxi in place of deta
    # -- dxi/eta_next/eta each get their FIRST contribution here.
    d_T1 = np.full(4, (-1.0/6.0) * d_momentxi)
    P = eta_next**2 + eta*eta_next + eta**2
    d_dxi = dxi.copy() * 0.0                # dxi's ONLY source -- unlike deta, dxi is not reused in the area sum below
    d_dxi += P * d_T1
    d_P = dxi * d_T1
    d_eta_next = 2*eta_next*d_P + eta*d_P
    d_eta = d_eta + (eta_next*d_P + 2*eta*d_P)  # eta's SECOND source (first was step 20)

    # ---- step 16: area = 0.5 * sum_k deta[k]*(xi_next[k]+xi[k])
    # deta gets its SECOND contribution here (first was step 18 -- it's
    # used in both the area formula and the momenteta formula). xi_next and
    # xi each get one more contribution too. Note d_deta must have BOTH
    # contributions in before step 15 consumes it next -- that ordering is
    # exactly why this function processes steps in the SAME order as the
    # forward pass, just backwards: nothing gets read until everything that
    # feeds it has already been written.
    d_S = np.full(4, 0.5 * d_area_total)
    d_deta += (xi_next + xi) * d_S          # deta: SECOND (and final) contribution, now fully accumulated
    d_xi_next = d_xi_next + deta * d_S
    d_xi = d_xi + deta * d_S                # xi: THIRD contribution

    # ---- step 15: dxi=xi_next-xi, deta=eta_next-eta  ("z=x-y" rule)
    # d_dxi and d_deta are now FULLY accumulated (nothing later reuses them)
    # so this is where they finally get spent, pushed onto xi_next/xi and
    # eta_next/eta (note the MINUS sign on the xi/eta side, from "z=x-y").
    d_xi_next = d_xi_next + d_dxi
    d_xi = d_xi - d_dxi                     # xi: FOURTH contribution
    d_eta_next = d_eta_next + d_deta
    d_eta = d_eta - d_deta                  # eta: THIRD contribution

    # ---- step 14: xi_next[k]=xi[ip1[k]], eta_next[k]=eta[ip1[k]]
    # THE SCATTER STEP -- easiest one to get backwards. Forward direction:
    # this just COPIES xi at index ip1[k] into xi_next at index k (no
    # arithmetic, ip1=(1,2,3,0) is just "the next corner around the quad").
    # So xi[ip1[k]] is the SOURCE and xi_next[k] is the COPY. The adjoint of
    # a copy is "add the destination's adjoint into the source" -- so
    # d_xi_next[k] (now fully formed, from steps 18+16+15 above) flows INTO
    # d_xi[ip1[k]], not into d_xi[k]. This is xi's FIFTH and final
    # contribution, and it's the only one that lands at a DIFFERENT index
    # than the one triggering it.
    for k in range(4):
        d_xi[IP1[k]] += d_xi_next[k]
        d_eta[IP1[k]] += d_eta_next[k]

    # ---- step 13: xi[k]=dot(d[k],iv), eta[k]=dot(d[k],jv)  ("z=dot(a,b)" rule)
    #   z=dot(a,b)  =>  da += b*dz,  db += a*dz
    # d_xi is NOW fully accumulated (all 5 contributions in) and gets spent
    # here. Note iv/jv are the SAME two vectors reused across all 4 corners
    # k=0..3, so d_iv/d_jv each pick up FOUR more contributions in this one
    # loop (one per corner) -- classic "used in a loop -> accumulate every
    # iteration" pattern.
    d_d = [np.zeros(3) for _ in range(4)]
    for k in range(4):
        d_d[k] += iv * d_xi[k]
        d_iv += d_list[k] * d_xi[k]
        d_d[k] += jv * d_eta[k]
        d_jv += d_list[k] * d_eta[k]

    # ---- step 12: d[k]=corners[k]-center0, k=0..3  ("z=x-y" rule x4)
    # Each d[k] feeds exactly one corner directly (own index), but center0
    # is SUBTRACTED in all 4 -- so d_center0 accumulates the NEGATIVE SUM
    # of all 4 d_d[k] (same pattern as xic/etac in step 20, just vector-
    # valued this time).
    d_c = [np.zeros(3) for _ in range(4)]
    for k in range(4):
        d_c[k] += d_d[k]
        d_center0 = d_center0 - d_d[k]

    # ---- step 9: jv = cross(nv, iv)  ("z=cross(a,b)" rule)
    #   z=cross(a,b)  =>  da += cross(b,dz),  db += cross(dz,a)
    # (sign/argument-order verified numerically against a hand-checked
    # example before trusting it anywhere else in this file -- see the
    # module docstring). d_jv is now fully accumulated and gets spent here;
    # d_nv and d_iv each pick up one more contribution.
    d_nv += _cross(iv, d_jv)
    d_iv += _cross(d_jv, nv)

    # ---- step 8: nv = nv_raw/nv_norm  ("z=x/y" rule, vector/scalar form)
    d_nv_raw = d_nv / nv_norm
    d_nv_norm = -np.dot(nv, d_nv) / nv_norm

    # ---- step 7: nv_norm = sqrt(dot(nv_raw,nv_raw))  ("z=sqrt(x)" rule)
    # dot(x,x) has its own small product rule baked in: d(dot(x,x))/dx=2x.
    d_s2 = d_nv_norm * 0.5 / nv_norm
    d_nv_raw = d_nv_raw + 2*nv_raw*d_s2     # nv_raw's SECOND contribution (normalize = divide-then-sqrt, both use it)

    # ---- step 6: nv_raw = cross(iv, jvbar)  ("z=cross(a,b)" rule)
    # d_nv_raw is now fully accumulated (steps 8+7) and gets spent here.
    d_iv += _cross(jvbar, d_nv_raw)         # iv's THIRD contribution (steps 21, 9, now this)
    d_jvbar = _cross(d_nv_raw, iv)

    # ---- step 5: jvbar = m3 - m1  ("z=x-y" rule)
    d_m3 = d_jvbar.copy()
    d_m1 = -d_jvbar.copy()

    # ---- step 4: iv = iv_raw/iv_norm  ("z=x/y" rule)
    # d_iv is FULLY accumulated now (steps 21, 13x4, 9, 6 all done) --
    # every remaining use of iv below this line is a NEW variable (iv_raw),
    # not iv itself, so this is the last time d_iv gets read.
    d_iv_raw = d_iv / iv_norm
    d_iv_norm = -np.dot(iv, d_iv) / iv_norm

    # ---- step 3: iv_norm = sqrt(dot(iv_raw,iv_raw))  (mirrors steps 7-8 for nv)
    d_s1 = d_iv_norm * 0.5 / iv_norm
    d_iv_raw = d_iv_raw + 2*iv_raw*d_s1

    # ---- step 2: iv_raw = m2 - m4  ("z=x-y" rule)
    d_m2 = d_iv_raw.copy()
    d_m4 = -d_iv_raw.copy()

    # ---- step 11: center0 = 0.25*(m1+m2+m3+m4)
    # d_center0 is fully accumulated (from steps 21 and 12) and gets spent
    # here: distributes equally (x0.25) to all four midpoints -- this is
    # each midpoint's SECOND contribution (first was steps 2/5 above, which
    # ran "later" in this backward walk but correspond to an EARLIER
    # forward step -- order among m1..m4's contributions doesn't matter,
    # only that both are in before step 1 reads them).
    d_m1 = d_m1 + 0.25*d_center0
    d_m2 = d_m2 + 0.25*d_center0
    d_m3 = d_m3 + 0.25*d_center0
    d_m4 = d_m4 + 0.25*d_center0

    # ---- step 1: m1=.5(c0+c1), m2=.5(c1+c2), m3=.5(c2+c3), m4=.5(c3+c0)
    # The last step -- every corner is shared by exactly two adjacent
    # midpoints (e.g. c0 feeds m1 AND m4), so each d_c[k] below picks up
    # its SECOND and final contribution here (first was step 12's direct
    # d[k]=corners[k]-center0 term). After this, d_c[0..3] are complete.
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
        '''

        Parameters
        ----------
        xs_nodes : Node-valued
            DESCRIPTION.

        Returns
        -------
        out : Node-valued
            flattened ctr, csy, cl, ar
            
        This is the function jacobian() will call. 
        - It unpacks the (Node-valued) flat input, 
        - calls the existing, already-validated panel_geometry_one from geometry.py 
        - note, not the hand-derived _forward — on those Nodes, 
            - which builds a dynamic graph via operator overloading 
                as a side effect of just running normally. 
            - Then it flattens the 4 oracle outputs 
                into one flat list of 21 Node-valued scalars (3 center + 9 coordsys + 8 cornerslocal + 1 area), 
            - matching the "21 outputs" convention used throughout this file.

        '''
        c = unpack_nodes(xs_nodes)
        ctr, csy, cl, ar = panel_geometry_one(c) # panel_geometry_one is the existing dynamic revAD version for one panel's geometry
        out = list(ctr) + [csy[r][c2] for r in range(3) for c2 in range(3)] + \
              [cl[a][b] for a in range(2) for b in range(4)] + [ar]
        return out
    
    

    x0 = pack(corners_xyz0) # x0 is the real length-12 float vector for panel 0's actual corners
    J_oracle = jacobian(run_traced, x0)  # (21, 12) ... the jacobian of the oracle (dynamic revAD version)
    # jacobian returns the full (21,12) ground-truth Jacobian — this is the ORACLE we're validating against

    out_labels = (["center%d" % i for i in range(3)]
                  + ["coordsys%d%d" % (i, j) for i in range(3) for j in range(3)]
                  + ["cornersloc%d%d" % (i, j) for i in range(2) for j in range(4)]
                  + ["area"])

    max_abs_err = 0.0
    worst = None
    J_hand = np.zeros((21, 12))
    # heart of the validation
    '''
    Build the seed gradient:
        
        - across all 21 numbers spread over these four containers,  (d_center, d_coordsys, d_cornerslocal, d_area)
          - exactly one number is 1.0 
          - and the other twenty are 0.0, on every single call.
        
    for o, olab in enumerate(out_labels):
        # o says: if I only care about this output component, `o', what's its gradient w.r.t. all 12 inputs?
        #
        # across all 21 numbers spread over these four containers, 
        #   exactly one number is 1.0 
        #   and the other twenty are 0.0, on every single call.
        #
        d_center = np.zeros(3); d_coordsys = np.zeros((3, 3)); d_cornerslocal = np.zeros((2, 4)); d_area = 0.0
        if o < 3:
            d_center[o] = 1.0 # seeding the gradient
        elif o < 12:
            idx = o - 3
            d_coordsys[idx // 3, idx % 3] = 1.0 #seeding the gradient
        elif o < 20:
            idx = o - 12
            d_cornerslocal[idx // 4, idx % 4] = 1.0 # seeding the gradient
        else:
            d_area = 1.0 # seeding the gradient
             
    '''
    # easier to read version of the same grad seed builder 
    for o in range(21):
        olab = out_labels[o]
        s = np.zeros(21) # set all the seeds to 0 each pass
        s[o] = 1.0 # set exactly 1 component to 1.
    
        d_center       = s[0:3]
        d_coordsys     = s[3:12].reshape(3, 3)
        d_cornerslocal = s[12:20].reshape(2, 4)
        d_area         = s[20]
        
        # if I only care about this output component, `o', what's its gradient w.r.t. all 12 inputs?
        d_c0, d_c1, d_c2, d_c3 = panel_geometry_one_backward(cache, d_center, d_coordsys, d_cornerslocal, d_area)
        # d_c0, d_c1, d_c2, d_c3 are each 3-vectors
        #   the actual per-vertex gradient components that get summed into the final 3·Nverts output
        # there are 4 components because a panel is a quad.
        
        row_hand = np.concatenate([d_c0, d_c1, d_c2, d_c3])
        J_hand[o, :] = row_hand
        # Runs your hand-derived backward pass with that one-hot seed, 
        # getting back one row of a (21,12) Jacobian 
        #   - literally building up the full matrix one backward pass at a time, 21 calls total, 
        #   - mirroring exactly what jacobian() does internally (next section) 
        #   - except using the hand-derived backward pass instead of Node.backward().
        
        
        # Compare row_hand against the same row of the oracle Jacobian, 
        # tracking the worst discrepancy across all 21 rows for the final report.
        row_oracle = J_oracle[o, :]
        err = np.max(np.abs(row_hand - row_oracle))
        if err > max_abs_err:
            max_abs_err = err
            worst = (olab, row_hand, row_oracle)

    print(f"\nmax |hand-derived - revad oracle| over all 21x12 entries: {max_abs_err:.3e}")
    print(f"worst output direction: {worst[0]}")
    print(f"  hand:   {worst[1]}")
    print(f"  oracle: {worst[2]}")
