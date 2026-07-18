# -*- coding: utf-8 -*-
"""
Numba-jitted reverse-mode backward passes for the adjoint shape-gradient
pipeline -- the numba twin of gradient_checks/handderiv_*.py, same math,
restructured for numba's constraints. This is the piece the TODO comment
in numba_kernels.py::assemble_A_b_vel_nb has been pointing at ("Replace
with analytic/AD ... reverse-mode VJP (seeded by lam) for full per-vertex
DOF").

SCOPE: only the O(N^2) hot-loop pieces need numba -- hs_influence's and
phixx_influence's backward passes, and (eventually) the assembly loop's
backward pass with its scatter-add. panel_geometry's backward is O(N), not
O(N^2) (called once per panel, not once per panel PAIR), and production
never numba-jits panel geometry either (see numba_kernels.py:
assemble_A_b_vel_nb takes center/coordsys/cornerslocal/area as already-
computed arguments, built by the plain vectorized panel_geometry_all, not
inside the numba function) -- so panel_geometry_one_backward stays plain
Python/numpy, no port needed there.

WHAT CHANGES FROM THE PURE-PYTHON handderiv_hs_influence.py, AND WHY:
  - No `cache` dict, no per-edge dict-of-dicts. Numba doesn't support
    heterogeneous dicts or lists-of-dicts with data-dependent keys. Fixed-
    size (4,) numpy arrays replace them -- one array per cached quantity
    (edx_a, r1_a, m_a, ...), one boolean array per flag (active, fires_atan,
    m_is_const) replacing "is this key present in the dict".
  - Forward and backward are ONE function, not two. There's no cache
    object to hand from a separate _forward call to a separate _backward
    call across a numba function boundary (returning a dict of arrays
    would work but adds overhead and complexity for no benefit here) --
    this also happens to be the natural shape a CUDA kernel takes: one
    thread does forward-then-backward for its one (row,col) pair, nothing
    persists between kernel launches.
  - `np.sign(edy) * 1e30` for the degenerate-m fallback matches
    hs_influence_nb's own convention exactly (not math.copysign, which the
    plain-Python hand-derivation used) -- ported to match the ACTUAL numba
    kernel this is meant to sit alongside, not the pure-Python version.

Everything else -- every adjoint rule, every "+=", every ordering
decision -- is identical to handderiv_hs_influence.py; see that file's
docstring for the full derivation and the adjoint-rule table. This file
assumes that derivation is already understood and correct (validated
there to 6.57e-15 against the revad oracle, including branch coverage);
it is purely a restructuring for numba, validated against THAT already-
trusted pure-Python version, not re-derived from scratch.
"""
from __future__ import annotations
import numpy as np
import math

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


def _require_numba():
    if njit is None:
        raise ImportError("Numba not available. Install with `pip install numba` or `conda install numba`.")


if njit is not None:

    @njit(cache=True)
    def hs_influence_backward_nb(fieldpoint, center, coordsys, corners_local,
                                  d_vx, d_vy, d_vz, eps=1e-6):
        """
        Numba-jitted hand-derived reverse-mode backward pass for
        hs_influence_nb. Forward (to build cached intermediates) and
        backward (to consume them) in ONE function -- see module docstring.

        Returns: d_fieldpoint (3,), d_center (3,), d_coordsys (3,3), d_cornerslocal (2,4)
        """
        # ================= FORWARD (same formula as hs_influence_nb) =================
        dx0 = fieldpoint[0] - center[0]
        dy0 = fieldpoint[1] - center[1]
        dz0 = fieldpoint[2] - center[2]

        x = coordsys[0, 0]*dx0 + coordsys[1, 0]*dy0 + coordsys[2, 0]*dz0
        y = coordsys[0, 1]*dx0 + coordsys[1, 1]*dy0 + coordsys[2, 1]*dz0
        z = coordsys[0, 2]*dx0 + coordsys[1, 2]*dy0 + coordsys[2, 2]*dz0

        dphi0 = 0.0
        dphi1 = 0.0
        dphi2 = 0.0

        active = np.zeros(4, dtype=np.bool_)
        fires_atan = np.zeros(4, dtype=np.bool_)
        m_is_const = np.zeros(4, dtype=np.bool_)
        edx_a = np.zeros(4); edy_a = np.zeros(4); dlen_a = np.zeros(4)
        rx1_a = np.zeros(4); ry1_a = np.zeros(4); r1_a = np.zeros(4)
        rx2_a = np.zeros(4); ry2_a = np.zeros(4); r2_a = np.zeros(4)
        frac_a = np.zeros(4); L_a = np.zeros(4)
        m_a = np.zeros(4); e1_a = np.zeros(4); h1_a = np.zeros(4)
        e2_a = np.zeros(4); h2_a = np.zeros(4)
        denom1_a = np.zeros(4); denom2_a = np.zeros(4)
        arg1_a = np.zeros(4); arg2_a = np.zeros(4)

        ip1_0 = 1; ip1_1 = 2; ip1_2 = 3; ip1_3 = 0

        for k in range(4):
            if k == 0: k2 = ip1_0
            elif k == 1: k2 = ip1_1
            elif k == 2: k2 = ip1_2
            else: k2 = ip1_3

            xk = corners_local[0, k]; yk = corners_local[1, k]
            xk2 = corners_local[0, k2]; yk2 = corners_local[1, k2]

            edx = xk2 - xk
            edy = yk2 - yk
            dlen = math.sqrt(edx*edx + edy*edy)
            edx_a[k] = edx; edy_a[k] = edy; dlen_a[k] = dlen

            if dlen <= eps:
                continue

            rx1 = x - xk; ry1 = y - yk
            r1 = math.sqrt(rx1*rx1 + ry1*ry1 + z*z)
            rx2 = x - xk2; ry2 = y - yk2
            r2 = math.sqrt(rx2*rx2 + ry2*ry2 + z*z)

            s = r1 + r2
            num = s - dlen
            den = s + dlen
            if num <= 0.0 or den <= 0.0:
                continue

            frac = num / den
            L = math.log(frac)
            term0 = (edy / dlen) * L
            term1 = (edx / dlen) * L
            dphi0 += term0
            dphi1 -= term1

            if abs(edx) < 1e-30:
                m = np.sign(edy) * 1e30
                m_is_const[k] = True
            else:
                m = edy / edx
                m_is_const[k] = False

            e1 = rx1*rx1 + z*z
            h1 = rx1*ry1
            e2 = rx2*rx2 + z*z
            h2 = rx2*ry2
            denom1 = z*r1
            denom2 = z*r2

            active[k] = True
            rx1_a[k] = rx1; ry1_a[k] = ry1; r1_a[k] = r1
            rx2_a[k] = rx2; ry2_a[k] = ry2; r2_a[k] = r2
            frac_a[k] = frac; L_a[k] = L
            m_a[k] = m; e1_a[k] = e1; h1_a[k] = h1; e2_a[k] = e2; h2_a[k] = h2
            denom1_a[k] = denom1; denom2_a[k] = denom2

            if abs(denom1) > 1e-30 and abs(denom2) > 1e-30:
                arg1 = (m*e1 - h1) / denom1
                arg2 = (m*e2 - h2) / denom2
                dphi2 += math.atan(arg1) - math.atan(arg2)
                fires_atan[k] = True
                arg1_a[k] = arg1; arg2_a[k] = arg2

        scale = -1.0 / (4.0 * math.pi)
        dphi0_s = dphi0 * scale
        dphi1_s = dphi1 * scale
        dphi2_s = dphi2 * scale

        # forward value (needed by callers composing this into A_entry/self-term,
        # which need v's VALUE for the outer dot-product's own adjoint rule --
        # not just its gradient)
        vx = coordsys[0, 0]*dphi0_s + coordsys[0, 1]*dphi1_s + coordsys[0, 2]*dphi2_s
        vy = coordsys[1, 0]*dphi0_s + coordsys[1, 1]*dphi1_s + coordsys[1, 2]*dphi2_s
        vz = coordsys[2, 0]*dphi0_s + coordsys[2, 1]*dphi1_s + coordsys[2, 2]*dphi2_s

        # ================= BACKWARD =================
        # ---- step 5: v = coordsys @ dphi_scaled
        d_coordsys = np.zeros((3, 3))
        d_coordsys[0, 0] = d_vx*dphi0_s; d_coordsys[0, 1] = d_vx*dphi1_s; d_coordsys[0, 2] = d_vx*dphi2_s
        d_coordsys[1, 0] = d_vy*dphi0_s; d_coordsys[1, 1] = d_vy*dphi1_s; d_coordsys[1, 2] = d_vy*dphi2_s
        d_coordsys[2, 0] = d_vz*dphi0_s; d_coordsys[2, 1] = d_vz*dphi1_s; d_coordsys[2, 2] = d_vz*dphi2_s

        d_dphi0_s = coordsys[0, 0]*d_vx + coordsys[1, 0]*d_vy + coordsys[2, 0]*d_vz
        d_dphi1_s = coordsys[0, 1]*d_vx + coordsys[1, 1]*d_vy + coordsys[2, 1]*d_vz
        d_dphi2_s = coordsys[0, 2]*d_vx + coordsys[1, 2]*d_vy + coordsys[2, 2]*d_vz

        # ---- step 4: dphi_raw *= scale (constant)
        g0 = scale * d_dphi0_s
        g1 = scale * d_dphi1_s
        g2 = scale * d_dphi2_s

        d_x = 0.0
        d_y = 0.0
        d_z = 0.0
        d_corners_local = np.zeros((2, 4))

        for k in range(3, -1, -1):
            if not active[k]:
                continue

            if k == 0: k2 = ip1_0
            elif k == 1: k2 = ip1_1
            elif k == 2: k2 = ip1_2
            else: k2 = ip1_3

            edx = edx_a[k]; edy = edy_a[k]; dlen = dlen_a[k]
            rx1 = rx1_a[k]; ry1 = ry1_a[k]; r1 = r1_a[k]
            rx2 = rx2_a[k]; ry2 = ry2_a[k]; r2 = r2_a[k]
            frac = frac_a[k]; L = L_a[k]

            d_edx = 0.0; d_edy = 0.0; d_dlen = 0.0
            d_r1 = 0.0; d_r2 = 0.0
            d_rx1 = 0.0; d_ry1 = 0.0; d_rx2 = 0.0; d_ry2 = 0.0

            if fires_atan[k]:
                m = m_a[k]; e1 = e1_a[k]; h1 = h1_a[k]; e2 = e2_a[k]; h2 = h2_a[k]
                denom1 = denom1_a[k]; denom2 = denom2_a[k]
                arg1 = arg1_a[k]; arg2 = arg2_a[k]

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
                d_m += e2 * d_w2
                d_e2 = m * d_w2
                d_h2 = -d_w2

                d_z += r1 * d_denom1
                d_r1 += z * d_denom1
                d_z += r2 * d_denom2
                d_r2 += z * d_denom2

                d_rx2 += 2*rx2*d_e2
                d_z += 2*z*d_e2
                d_rx2 += ry2*d_h2
                d_ry2 += rx2*d_h2

                d_rx1 += 2*rx1*d_e1
                d_z += 2*z*d_e1
                d_rx1 += ry1*d_h1
                d_ry1 += rx1*d_h1

                if not m_is_const[k]:
                    d_edy += d_m/edx
                    d_edx += -m*d_m/edx

            d_term0 = g0
            d_term1 = -g1
            p_h = edy / dlen
            d_p_h = L * d_term0
            d_L = p_h * d_term0
            d_edy += d_p_h / dlen
            d_dlen += -p_h * d_p_h / dlen
            q_h = edx / dlen
            d_q_h = L * d_term1
            d_L += q_h * d_term1
            d_edx += d_q_h / dlen
            d_dlen += -q_h * d_q_h / dlen

            d_frac = d_L / frac
            s = r1 + r2
            den = s + dlen
            d_num = d_frac / den
            d_den = -frac * d_frac / den

            d_s = d_num + d_den
            d_dlen += -d_num + d_den
            d_r1 += d_s
            d_r2 += d_s

            d_R2 = d_r2 * 0.5 / r2
            d_rx2 += 2.0*rx2*d_R2
            d_ry2 += 2.0*ry2*d_R2
            d_z += 2.0*z*d_R2
            d_x += d_rx2
            d_xk2 = -d_rx2
            d_y += d_ry2
            d_yk2 = -d_ry2

            d_R1 = d_r1 * 0.5 / r1
            d_rx1 += 2.0*rx1*d_R1
            d_ry1 += 2.0*ry1*d_R1
            d_z += 2.0*z*d_R1
            d_x += d_rx1
            d_xk = -d_rx1
            d_y += d_ry1
            d_yk = -d_ry1

            d_D2 = d_dlen * 0.5 / dlen
            d_edx += 2.0*edx*d_D2
            d_edy += 2.0*edy*d_D2

            d_xk2 += d_edx
            d_xk += -d_edx
            d_yk2 += d_edy
            d_yk += -d_edy

            d_corners_local[0, k] += d_xk
            d_corners_local[1, k] += d_yk
            d_corners_local[0, k2] += d_xk2
            d_corners_local[1, k2] += d_yk2

        # ---- step 2: x,y,z = coordsys^T @ delta
        d_coordsys[0, 0] += dx0*d_x; d_coordsys[0, 1] += dx0*d_y; d_coordsys[0, 2] += dx0*d_z
        d_coordsys[1, 0] += dy0*d_x; d_coordsys[1, 1] += dy0*d_y; d_coordsys[1, 2] += dy0*d_z
        d_coordsys[2, 0] += dz0*d_x; d_coordsys[2, 1] += dz0*d_y; d_coordsys[2, 2] += dz0*d_z

        d_dx0 = coordsys[0, 0]*d_x + coordsys[0, 1]*d_y + coordsys[0, 2]*d_z
        d_dy0 = coordsys[1, 0]*d_x + coordsys[1, 1]*d_y + coordsys[1, 2]*d_z
        d_dz0 = coordsys[2, 0]*d_x + coordsys[2, 1]*d_y + coordsys[2, 2]*d_z

        d_fieldpoint = np.empty(3)
        d_fieldpoint[0] = d_dx0; d_fieldpoint[1] = d_dy0; d_fieldpoint[2] = d_dz0
        d_center = np.empty(3)
        d_center[0] = -d_dx0; d_center[1] = -d_dy0; d_center[2] = -d_dz0

        return vx, vy, vz, d_fieldpoint, d_center, d_coordsys, d_corners_local


    @njit(cache=True)
    def phixx_influence_backward_nb(fieldpoint, center, coordsys, corners_local,
                                     d_out, eps=1e-6):
        """
        Numba-jitted hand-derived reverse-mode backward pass for
        phixx_influence_nb. Same structure/rationale as
        hs_influence_backward_nb above -- see that function's and this
        module's docstrings. Ported from handderiv_phixx_influence.py;
        phixx has only ONE per-edge flag (`active`, not the two-plus-a-
        subcase hs_influence needs), so the per-edge cache is simpler.

        d_out: (3,3) upstream adjoint for the returned global Hessian.
        Returns: d_fieldpoint (3,), d_center (3,), d_coordsys (3,3), d_cornerslocal (2,4)
        """
        # ================= FORWARD (same formula as phixx_influence_nb) =================
        dx0 = fieldpoint[0] - center[0]
        dy0 = fieldpoint[1] - center[1]
        dz0 = fieldpoint[2] - center[2]

        x = coordsys[0, 0]*dx0 + coordsys[1, 0]*dy0 + coordsys[2, 0]*dz0
        y = coordsys[0, 1]*dx0 + coordsys[1, 1]*dy0 + coordsys[2, 1]*dz0
        z = coordsys[0, 2]*dx0 + coordsys[1, 2]*dy0 + coordsys[2, 2]*dz0

        dd = np.zeros((3, 3))

        active = np.zeros(4, dtype=np.bool_)
        edx_a = np.zeros(4); edy_a = np.zeros(4); dlen_a = np.zeros(4)
        rx1_a = np.zeros(4); ry1_a = np.zeros(4); r1_a = np.zeros(4)
        rx2_a = np.zeros(4); ry2_a = np.zeros(4); r2_a = np.zeros(4)
        ssum_a = np.zeros(4); denom1_a = np.zeros(4); big_a = np.zeros(4)
        A_a = np.zeros(4); B_a = np.zeros(4); C_a = np.zeros(4); cross_a = np.zeros(4)
        K1_a = np.zeros(4); K2_a = np.zeros(4)
        t02_a = np.zeros(4); t12_a = np.zeros(4); t22_a = np.zeros(4)

        ip1_0 = 1; ip1_1 = 2; ip1_2 = 3; ip1_3 = 0

        for k in range(4):
            if k == 0: k2 = ip1_0
            elif k == 1: k2 = ip1_1
            elif k == 2: k2 = ip1_2
            else: k2 = ip1_3

            xk = corners_local[0, k]; yk = corners_local[1, k]
            xk2 = corners_local[0, k2]; yk2 = corners_local[1, k2]

            edx = xk2 - xk
            edy = yk2 - yk
            dlen = math.sqrt(edx*edx + edy*edy)
            edx_a[k] = edx; edy_a[k] = edy; dlen_a[k] = dlen

            if dlen <= eps:
                continue

            rx1 = x - xk; ry1 = y - yk
            r1 = math.sqrt(rx1*rx1 + ry1*ry1 + z*z)
            rx2 = x - xk2; ry2 = y - yk2
            r2 = math.sqrt(rx2*rx2 + ry2*ry2 + z*z)

            ssum = r1 + r2
            denom1 = ssum*ssum - dlen*dlen
            if abs(denom1) <= 1e-30:
                continue

            big = (r1*r2) * (r1*r2 + rx1*rx2 + ry1*ry2 + z*z)
            if abs(big) <= 1e-30:
                continue

            A_ = rx1/r1 + rx2/r2
            B_ = ry1/r1 + ry2/r2
            C_ = z/r1 + z/r2
            cross_ = rx1*ry2 - rx2*ry1
            K1 = 2.0*edy/denom1
            K2 = -2.0*edx/denom1

            t00 = K1*A_; t10 = K1*B_; t20 = K1*C_
            t01 = K2*A_; t11 = K2*B_; t21 = K2*C_
            t02 = (z*edy*ssum) / big
            t12 = (-z*edx*ssum) / big
            t22 = (cross_*ssum) / big

            dd[0, 0] += t00; dd[0, 1] += t01; dd[0, 2] += t02
            dd[1, 0] += t10; dd[1, 1] += t11; dd[1, 2] += t12
            dd[2, 0] += t20; dd[2, 1] += t21; dd[2, 2] += t22

            active[k] = True
            rx1_a[k] = rx1; ry1_a[k] = ry1; r1_a[k] = r1
            rx2_a[k] = rx2; ry2_a[k] = ry2; r2_a[k] = r2
            ssum_a[k] = ssum; denom1_a[k] = denom1; big_a[k] = big
            A_a[k] = A_; B_a[k] = B_; C_a[k] = C_; cross_a[k] = cross_
            K1_a[k] = K1; K2_a[k] = K2
            t02_a[k] = t02; t12_a[k] = t12; t22_a[k] = t22

        scale = -1.0 / (4.0 * math.pi)
        dd_scaled = dd * scale

        tmp = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                tmp[a, b] = coordsys[a, 0]*dd_scaled[0, b] + coordsys[a, 1]*dd_scaled[1, b] + coordsys[a, 2]*dd_scaled[2, b]

        # forward value (needed by callers composing this into the FS-row
        # A_entry, which needs H[0,0]'s VALUE, not just its gradient)
        out = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                out[a, b] = tmp[a, 0]*coordsys[b, 0] + tmp[a, 1]*coordsys[b, 1] + tmp[a, 2]*coordsys[b, 2]

        # ================= BACKWARD =================
        # ---- step 6: out = tmp @ coordsys^T  =>  d_tmp = d_out @ coordsys,
        # d_coordsys (FIRST contribution) = d_out^T @ tmp
        d_tmp = np.zeros((3, 3))
        d_coordsys = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                s1 = 0.0
                s2 = 0.0
                for a in range(3):
                    s1 += d_out[i, a] * coordsys[a, j]     # d_tmp[i,j]
                    s2 += d_out[a, i] * tmp[a, j]           # d_coordsys[i,j], first contribution
                d_tmp[i, j] = s1
                d_coordsys[i, j] = s2

        # ---- step 5: tmp = coordsys @ dd_scaled  =>  d_coordsys += d_tmp @ dd_scaled^T,
        # d_dd_scaled = coordsys^T @ d_tmp
        d_dd_scaled = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                s1 = 0.0
                s2 = 0.0
                for b in range(3):
                    s1 += d_tmp[i, b] * dd_scaled[j, b]     # d_coordsys[i,j], second contribution
                for a in range(3):
                    s2 += coordsys[a, i] * d_tmp[a, j]      # d_dd_scaled[i,j]
                d_coordsys[i, j] += s1
                d_dd_scaled[i, j] = s2

        # ---- step 4: dd_scaled = dd_raw * scale (constant)
        G = scale * d_dd_scaled  # same (3,3) seed used by every edge below

        d_x = 0.0
        d_y = 0.0
        d_z = 0.0
        d_corners_local = np.zeros((2, 4))

        for k in range(3, -1, -1):
            if not active[k]:
                continue

            if k == 0: k2 = ip1_0
            elif k == 1: k2 = ip1_1
            elif k == 2: k2 = ip1_2
            else: k2 = ip1_3

            edx = edx_a[k]; edy = edy_a[k]; dlen = dlen_a[k]
            rx1 = rx1_a[k]; ry1 = ry1_a[k]; r1 = r1_a[k]
            rx2 = rx2_a[k]; ry2 = ry2_a[k]; r2 = r2_a[k]
            ssum = ssum_a[k]; denom1 = denom1_a[k]; big = big_a[k]
            A_ = A_a[k]; B_ = B_a[k]; C_ = C_a[k]; cross_ = cross_a[k]
            K1 = K1_a[k]; K2 = K2_a[k]
            t02 = t02_a[k]; t12 = t12_a[k]; t22 = t22_a[k]

            g00 = G[0, 0]; g01 = G[0, 1]; g02 = G[0, 2]
            g10 = G[1, 0]; g11 = G[1, 1]; g12 = G[1, 2]
            g20 = G[2, 0]; g21 = G[2, 1]; g22 = G[2, 2]

            # ---- t22 = (cross_*ssum)/big
            d_numer22 = g22 / big
            d_big = -t22 * g22 / big
            d_cross_ = ssum * d_numer22
            d_ssum = cross_ * d_numer22

            # ---- t12 = (-z*edx*ssum)/big
            d_numer12 = g12 / big
            d_big += -t12 * g12 / big
            d_z += -edx * ssum * d_numer12   # d_z is the OUTER cross-edge accumulator --
                                              # MUST be "+=" here, never "=" (this exact
                                              # class of bug was caught and fixed in the
                                              # pure-Python version -- see
                                              # handderiv_phixx_influence.py's history)
            d_edx = -z * ssum * d_numer12
            d_ssum += -z * edx * d_numer12

            # ---- t02 = (z*edy*ssum)/big
            d_numer02 = g02 / big
            d_big += -t02 * g02 / big
            d_z += edy * ssum * d_numer02
            d_edy = z * ssum * d_numer02
            d_ssum += z * edy * d_numer02

            # ---- t01=K2*A_, t11=K2*B_, t21=K2*C_
            d_K2 = A_*g01 + B_*g11 + C_*g21
            d_A_ = K2 * g01
            d_B_ = K2 * g11
            d_C_ = K2 * g21

            # ---- t00=K1*A_, t10=K1*B_, t20=K1*C_
            d_K1 = A_*g00 + B_*g10 + C_*g20
            d_A_ += K1 * g00
            d_B_ += K1 * g10
            d_C_ += K1 * g20

            # ---- K2 = -2*edx/denom1
            d_edx += -2.0 * d_K2 / denom1
            d_denom1 = -K2 * d_K2 / denom1

            # ---- K1 = 2*edy/denom1
            d_edy += 2.0 * d_K1 / denom1
            d_denom1 += -K1 * d_K1 / denom1

            # ---- cross_ = rx1*ry2 - rx2*ry1
            d_rx1 = ry2 * d_cross_
            d_ry2 = rx1 * d_cross_
            d_rx2 = -ry1 * d_cross_
            d_ry1 = -rx2 * d_cross_

            # ---- C_ = z/r1 + z/r2
            d_z += d_C_/r1 + d_C_/r2
            d_r1 = -z * d_C_ / (r1*r1)
            d_r2 = -z * d_C_ / (r2*r2)

            # ---- B_ = ry1/r1 + ry2/r2
            d_ry1 += d_B_ / r1
            d_r1 += -ry1 * d_B_ / (r1*r1)
            d_ry2 += d_B_ / r2
            d_r2 += -ry2 * d_B_ / (r2*r2)

            # ---- A_ = rx1/r1 + rx2/r2
            d_rx1 += d_A_ / r1
            d_r1 += -rx1 * d_A_ / (r1*r1)
            d_rx2 += d_A_ / r2
            d_r2 += -rx2 * d_A_ / (r2*r2)

            # ---- big = P*Q,  P=r1*r2,  Q=P+rx1*rx2+ry1*ry2+z*z
            Q = r1*r2 + rx1*rx2 + ry1*ry2 + z*z
            P = r1*r2
            d_P = Q * d_big
            d_Q = P * d_big
            d_P += d_Q
            d_rx1 += rx2 * d_Q
            d_rx2 += rx1 * d_Q
            d_ry1 += ry2 * d_Q
            d_ry2 += ry1 * d_Q
            d_z += 2.0 * z * d_Q
            d_r1 += r2 * d_P
            d_r2 += r1 * d_P

            # ---- denom1 = ssum^2 - dlen^2
            d_ssum += 2.0 * ssum * d_denom1
            d_dlen = -2.0 * dlen * d_denom1

            # ---- ssum = r1+r2
            d_r1 += d_ssum
            d_r2 += d_ssum

            # ---- rx2,ry2,r2 = x-xk2, y-yk2, sqrt(rx2^2+ry2^2+z^2)
            d_R2 = d_r2 * 0.5 / r2
            d_rx2 += 2.0*rx2*d_R2
            d_ry2 += 2.0*ry2*d_R2
            d_z += 2.0*z*d_R2
            d_x += d_rx2
            d_xk2 = -d_rx2
            d_y += d_ry2
            d_yk2 = -d_ry2

            # ---- rx1,ry1,r1 = x-xk, y-yk, sqrt(rx1^2+ry1^2+z^2)
            d_R1 = d_r1 * 0.5 / r1
            d_rx1 += 2.0*rx1*d_R1
            d_ry1 += 2.0*ry1*d_R1
            d_z += 2.0*z*d_R1
            d_x += d_rx1
            d_xk = -d_rx1
            d_y += d_ry1
            d_yk = -d_ry1

            # ---- dlen = sqrt(edx^2+edy^2)
            d_D2 = d_dlen * 0.5 / dlen
            d_edx += 2.0*edx*d_D2
            d_edy += 2.0*edy*d_D2

            # ---- edx,edy = xk2-xk, yk2-yk
            d_xk2 += d_edx
            d_xk += -d_edx
            d_yk2 += d_edy
            d_yk += -d_edy

            d_corners_local[0, k] += d_xk
            d_corners_local[1, k] += d_yk
            d_corners_local[0, k2] += d_xk2
            d_corners_local[1, k2] += d_yk2

        # ---- step 2: x,y,z = coordsys^T @ delta, delta=(dx0,dy0,dz0)
        d_coordsys[0, 0] += dx0*d_x; d_coordsys[0, 1] += dx0*d_y; d_coordsys[0, 2] += dx0*d_z
        d_coordsys[1, 0] += dy0*d_x; d_coordsys[1, 1] += dy0*d_y; d_coordsys[1, 2] += dy0*d_z
        d_coordsys[2, 0] += dz0*d_x; d_coordsys[2, 1] += dz0*d_y; d_coordsys[2, 2] += dz0*d_z

        d_dx0 = coordsys[0, 0]*d_x + coordsys[0, 1]*d_y + coordsys[0, 2]*d_z
        d_dy0 = coordsys[1, 0]*d_x + coordsys[1, 1]*d_y + coordsys[1, 2]*d_z
        d_dz0 = coordsys[2, 0]*d_x + coordsys[2, 1]*d_y + coordsys[2, 2]*d_z

        d_fieldpoint = np.empty(3)
        d_fieldpoint[0] = d_dx0; d_fieldpoint[1] = d_dy0; d_fieldpoint[2] = d_dz0
        d_center = np.empty(3)
        d_center[0] = -d_dx0; d_center[1] = -d_dy0; d_center[2] = -d_dz0

        return out, d_fieldpoint, d_center, d_coordsys, d_corners_local


    @njit(cache=True)
    def assemble_dphi_dgeom_nb(center, coordsys, cornerslocal, npanels, nfspanels,
                                deltax, vinf_x, gravity, lam, sigma):
        """
        The numba-jitted O(N^2) hot loop: mirrors assemble_A_b_vel_nb's exact
        row/col structure (see that function's docstring's TODO -- this is
        what it was pointing at), but instead of building A and b, it
        directly accumulates d(phi)/d(each panel's own geometry outputs),
        where phi = lam^T(A@sigma - b), lam/sigma held fixed.

        Returns d_center (N,3), d_coordsys (N,3,3), d_cornerslocal (N,2,4)
        -- gradients w.r.t. each panel's OWN center/coordsys/cornerslocal,
        NOT yet vertices. Converting to d(vertices) is a SEPARATE, O(N) step
        (panel_geometry_one_backward per panel + scatter-add via the real
        panel->vertex connectivity) -- deliberately NOT done here, since
        panel_geometry's own cost is O(N) not O(N^2) and production doesn't
        numba-jit it either (see module docstring). This function's whole
        job is the expensive O(N^2) part only.

        Every cell here composes hs_influence_backward_nb /
        phixx_influence_backward_nb exactly the way handderiv_A_entry.py /
        handderiv_A_entry_fs_row.py / handderiv_self_term_and_b.py compose
        the pure-Python versions -- same math, same "sum every path"
        accumulation, just writing into per-panel ARRAY SLOTS instead of
        returning per-call tuples.
        """
        N = npanels + nfspanels
        U2 = vinf_x * vinf_x

        d_center = np.zeros((N, 3))
        d_coordsys = np.zeros((N, 3, 3))
        d_cornerslocal = np.zeros((N, 2, 4))

        # ---- b vector (hull rows only): b[row] = coordsys[row,0,2]*vinf_x,
        # phi contains -lam[row]*b[row] -- pure closed form, no hs_influence call.
        for row in range(npanels):
            d_coordsys[row, 0, 2] += -lam[row] * vinf_x

        # ---- hull rows ----
        for row in range(npanels):
            p0 = center[row, 0]; p1 = center[row, 1]; p2 = center[row, 2]
            fieldpoint = np.array([p0, p1, p2])
            fieldpoint_mirror = np.array([p0, -p1, p2])
            ni0 = coordsys[row, 0, 2]; ni1 = coordsys[row, 1, 2]; ni2 = coordsys[row, 2, 2]
            normal_row = np.array([ni0, ni1, ni2])

            for col in range(N):
                seed = lam[row] * sigma[col]
                coordsys_col = coordsys[col]
                cornerslocal_col = cornerslocal[col]
                center_col = center[col]

                if col == row:
                    # ---- self-term: v = -0.5*normal_row (closed form), vp = mirror
                    v = -0.5 * normal_row
                    d_v_combined = normal_row * seed
                    d_v = d_v_combined.copy()
                    d_vp = np.array([d_v_combined[0], -d_v_combined[1], d_v_combined[2]])

                    vpx, vpy, vpz, d_fieldpoint_mirror, d_center_asArg, d_coordsys_asArg, d_cl_asArg = \
                        hs_influence_backward_nb(fieldpoint_mirror, center_col, coordsys_col, cornerslocal_col,
                                                  d_vp[0], d_vp[1], d_vp[2])

                    v_combined = np.array([v[0]+vpx, v[1]-vpy, v[2]+vpz])
                    d_normal_row = v_combined * seed
                    d_normal_row += -0.5 * d_v

                    d_center_row_mirror = np.array([d_fieldpoint_mirror[0], -d_fieldpoint_mirror[1], d_fieldpoint_mirror[2]])
                    d_center_row = d_center_row_mirror + d_center_asArg

                    d_center[row] += d_center_row
                    d_coordsys[row, :, 2] += d_normal_row
                    d_coordsys[row] += d_coordsys_asArg
                    d_cornerslocal[row] += d_cl_asArg

                else:
                    # ---- ordinary hull-row A_entry (col may be hull or FS -- generic)
                    d_v_combined = normal_row * seed
                    d_v = d_v_combined.copy()
                    d_vp = np.array([d_v_combined[0], -d_v_combined[1], d_v_combined[2]])

                    vx, vy, vz, d_fp_v, d_center_col_v, d_coordsys_col_v, d_cl_col_v = \
                        hs_influence_backward_nb(fieldpoint, center_col, coordsys_col, cornerslocal_col,
                                                  d_v[0], d_v[1], d_v[2])
                    vpx, vpy, vpz, d_fp_vp, d_center_col_vp, d_coordsys_col_vp, d_cl_col_vp = \
                        hs_influence_backward_nb(fieldpoint_mirror, center_col, coordsys_col, cornerslocal_col,
                                                  d_vp[0], d_vp[1], d_vp[2])

                    v_combined = np.array([vx+vpx, vy-vpy, vz+vpz])
                    d_normal_row = v_combined * seed

                    d_fp_vp_mirrored = np.array([d_fp_vp[0], -d_fp_vp[1], d_fp_vp[2]])
                    d_center_row = d_fp_v + d_fp_vp_mirrored

                    d_center[row] += d_center_row
                    d_coordsys[row, :, 2] += d_normal_row
                    d_center[col] += d_center_col_v + d_center_col_vp
                    d_coordsys[col] += d_coordsys_col_v + d_coordsys_col_vp
                    d_cornerslocal[col] += d_cl_col_v + d_cl_col_vp

        # ---- FS rows (col may be hull or FS -- generic, same formula either way) ----
        for i in range(nfspanels):
            row = npanels + i
            p0 = center[row, 0] + deltax
            p1 = center[row, 1]
            fieldpoint = np.array([p0, p1, 0.0])
            fieldpoint_mirror = np.array([p0, -p1, 0.0])

            for col in range(N):
                seed = lam[row] * sigma[col]
                d_phi_xx = U2 * seed
                d_v2 = gravity * seed

                coordsys_col = coordsys[col]
                cornerslocal_col = cornerslocal[col]
                center_col = center[col]

                d_out_H = np.zeros((3, 3)); d_out_H[0, 0] = d_phi_xx
                _, d_fp_H, d_center_col_H, d_coordsys_col_H, d_cl_col_H = \
                    phixx_influence_backward_nb(fieldpoint, center_col, coordsys_col, cornerslocal_col, d_out_H)
                _, d_fp_Hp, d_center_col_Hp, d_coordsys_col_Hp, d_cl_col_Hp = \
                    phixx_influence_backward_nb(fieldpoint_mirror, center_col, coordsys_col, cornerslocal_col, d_out_H)

                _, _, _, d_fp_v, d_center_col_v, d_coordsys_col_v, d_cl_col_v = \
                    hs_influence_backward_nb(fieldpoint, center_col, coordsys_col, cornerslocal_col, 0.0, 0.0, d_v2)
                _, _, _, d_fp_vp, d_center_col_vp, d_coordsys_col_vp, d_cl_col_vp = \
                    hs_influence_backward_nb(fieldpoint_mirror, center_col, coordsys_col, cornerslocal_col, 0.0, 0.0, d_v2)

                d_field = d_fp_v + d_fp_H
                d_field_mirror = d_fp_vp + d_fp_Hp

                d_center_row0 = d_field[0] + d_field_mirror[0]
                d_center_row1 = d_field[1] - d_field_mirror[1]
                # d_center_row2 = 0.0 -- p2/pp2 are literal constants, no path

                d_center[row, 0] += d_center_row0
                d_center[row, 1] += d_center_row1

                d_center[col] += d_center_col_v + d_center_col_vp + d_center_col_H + d_center_col_Hp
                d_coordsys[col] += d_coordsys_col_v + d_coordsys_col_vp + d_coordsys_col_H + d_coordsys_col_Hp
                d_cornerslocal[col] += d_cl_col_v + d_cl_col_vp + d_cl_col_H + d_cl_col_Hp

        return d_center, d_coordsys, d_cornerslocal
