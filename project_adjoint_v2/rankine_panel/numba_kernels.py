# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 23:50:28 2026

@author: tluke
"""

# rankine_panel/numba_kernels.py
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


# -----------------------------
# Numba influence kernels
# -----------------------------
if njit is not None:

    @njit(cache=True)
    def hs_influence_nb(fieldpoint, center, coordsys, corners_local, eps=1e-6):
        """
        fieldpoint: (3,)
        center: (3,)
        coordsys: (3,3) columns are (t1,t2,n)
        corners_local: (2,4)
        returns vel_global: (3,)
        """
        # p = coordsys^T @ (fieldpoint-center)
        dx0 = fieldpoint[0] - center[0]
        dy0 = fieldpoint[1] - center[1]
        dz0 = fieldpoint[2] - center[2]

        # coordsys.T multiply (manual)
        x = coordsys[0,0]*dx0 + coordsys[1,0]*dy0 + coordsys[2,0]*dz0
        y = coordsys[0,1]*dx0 + coordsys[1,1]*dy0 + coordsys[2,1]*dz0
        z = coordsys[0,2]*dx0 + coordsys[1,2]*dy0 + coordsys[2,2]*dz0

        dphi0 = 0.0
        dphi1 = 0.0
        dphi2 = 0.0

        ip1_0 = 1
        ip1_1 = 2
        ip1_2 = 3
        ip1_3 = 0

        for k in range(4):
            if k == 0:
                k2 = ip1_0
            elif k == 1:
                k2 = ip1_1
            elif k == 2:
                k2 = ip1_2
            else:
                k2 = ip1_3

            xk = corners_local[0, k]
            yk = corners_local[1, k]
            xk2 = corners_local[0, k2]
            yk2 = corners_local[1, k2]

            edx = xk2 - xk
            edy = yk2 - yk
            d = math.sqrt(edx*edx + edy*edy)
            if d <= eps:
                continue

            r1 = math.sqrt((x-xk)*(x-xk) + (y-yk)*(y-yk) + z*z)
            r2 = math.sqrt((x-xk2)*(x-xk2) + (y-yk2)*(y-yk2) + z*z)

            s = r1 + r2
            num = s - d
            den = s + d
            if num <= 0.0 or den <= 0.0:
                continue

            L = math.log(num / den)

            dphi0 += (edy / d) * L
            dphi1 -= (edx / d) * L

            # # robust angle term via atan2 (see earlier discussion)
            # e1 = (x - xk)*(x - xk) + z*z
            # h1 = (x - xk)*(y - yk)
            # n1 = edy*e1 - edx*h1
            # d1 = edx*z*r1

            # e2 = (x - xk2)*(x - xk2) + z*z
            # h2 = (x - xk2)*(y - yk2)
            # n2 = edy*e2 - edx*h2
            # d2 = edx*z*r2

            # dphi2 += (math.atan2(n1, d1) - math.atan2(n2, d2))
            
            
            #
            # --- Fortran-matching angle term (NO atan2) ---
            # m = dy/dx for the edge (k -> k2)
            if abs(edx) < 1e-30:
                # emulate the "huge slope" behavior instead of branching differently
                m = np.sign(edy) * 1e30
            else:
                m = edy / edx
    
            e1 = (x - xk)**2 + z*z
            h1 = (x - xk)*(y - yk)
            r1 = np.sqrt((x-xk)**2  + (y-yk)**2  + z*z)
    
            e2 = (x - xk2)**2 + z*z
            h2 = (x - xk2)*(y - yk2)
            r2 = np.sqrt((x-xk2)**2 + (y-yk2)**2 + z*z)
    
            # Note: Fortran uses plocal(3)*r in denominator
            denom1 = z * r1
            denom2 = z * r2
    
            # Guard only to avoid NaNs; Fortran would blow up if denom=0,
            # but with sources above FS you should not hit z==0 here.
            if abs(denom1) > 1e-30 and abs(denom2) > 1e-30:
                dphi2 += np.arctan((m*e1 - h1) / denom1) - np.arctan((m*e2 - h2) / denom2)
                
            #
            # --- Fortran-matching angle term (NO atan2) ---
            # 
                

        scale = -1.0 / (4.0 * math.pi)
        dphi0 *= scale
        dphi1 *= scale
        dphi2 *= scale

        # vel_global = coordsys @ dphi
        vx = coordsys[0,0]*dphi0 + coordsys[0,1]*dphi1 + coordsys[0,2]*dphi2
        vy = coordsys[1,0]*dphi0 + coordsys[1,1]*dphi1 + coordsys[1,2]*dphi2
        vz = coordsys[2,0]*dphi0 + coordsys[2,1]*dphi1 + coordsys[2,2]*dphi2

        out = np.empty(3, dtype=np.float64)
        out[0] = vx
        out[1] = vy
        out[2] = vz
        return out


    @njit(cache=True)
    def phixx_influence_nb(fieldpoint, center, coordsys, corners_local, eps=1e-6):
        """
        Returns global Hessian (3,3). Only (0,0) is used in your FS BC,
        but we return full matrix to match Fortran.
        """
        dx0 = fieldpoint[0] - center[0]
        dy0 = fieldpoint[1] - center[1]
        dz0 = fieldpoint[2] - center[2]

        x = coordsys[0,0]*dx0 + coordsys[1,0]*dy0 + coordsys[2,0]*dz0
        y = coordsys[0,1]*dx0 + coordsys[1,1]*dy0 + coordsys[2,1]*dz0
        z = coordsys[0,2]*dx0 + coordsys[1,2]*dy0 + coordsys[2,2]*dz0

        dd = np.zeros((3,3), dtype=np.float64)

        ip1_0 = 1
        ip1_1 = 2
        ip1_2 = 3
        ip1_3 = 0

        for k in range(4):
            if k == 0:
                k2 = ip1_0
            elif k == 1:
                k2 = ip1_1
            elif k == 2:
                k2 = ip1_2
            else:
                k2 = ip1_3

            xk = corners_local[0, k]
            yk = corners_local[1, k]
            xk2 = corners_local[0, k2]
            yk2 = corners_local[1, k2]

            edx = xk2 - xk
            edy = yk2 - yk
            d = math.sqrt(edx*edx + edy*edy)
            if d <= eps:
                continue

            r1 = math.sqrt((x-xk)*(x-xk) + (y-yk)*(y-yk) + z*z)
            r2 = math.sqrt((x-xk2)*(x-xk2) + (y-yk2)*(y-yk2) + z*z)

            s = r1 + r2
            denom1 = s*s - d*d
            if abs(denom1) <= 1e-30:
                continue

            big = (r1*r2) * (r1*r2
                             + (x-xk)*(x-xk2)
                             + (y-yk)*(y-yk2)
                             + z*z)
            if abs(big) <= 1e-30:
                continue

            dd[0,0] += (2.0*edy/denom1) * ((x-xk)/r1 + (x-xk2)/r2)
            dd[0,1] += (-2.0*edx/denom1) * ((x-xk)/r1 + (x-xk2)/r2)
            dd[0,2] += (z*edy*(r1+r2)) / big

            dd[1,0] += (2.0*edy/denom1) * ((y-yk)/r1 + (y-yk2)/r2)
            dd[1,1] += (-2.0*edx/denom1) * ((y-yk)/r1 + (y-yk2)/r2)
            dd[1,2] += ((-z*edx)*(r1+r2)) / big

            dd[2,0] += (2.0*edy * ((z/r1) + (z/r2))) / denom1
            dd[2,1] += (-2.0*edx * ((z/r1) + (z/r2))) / denom1
            dd[2,2] += ((( (x-xk)*(y-yk2) - (x-xk2)*(y-yk) ) * (r1+r2)) / big)

        dd *= (-1.0 / (4.0 * math.pi))

        # hess_global = coordsys @ dd @ coordsys^T (manual)
        # First tmp = coordsys @ dd
        tmp = np.zeros((3,3), dtype=np.float64)
        for a in range(3):
            for b in range(3):
                tmp[a,b] = coordsys[a,0]*dd[0,b] + coordsys[a,1]*dd[1,b] + coordsys[a,2]*dd[2,b]

        out = np.zeros((3,3), dtype=np.float64)
        for a in range(3):
            for b in range(3):
                out[a,b] = tmp[a,0]*coordsys[b,0] + tmp[a,1]*coordsys[b,1] + tmp[a,2]*coordsys[b,2]
        return out


    # -----------------------------
    # Numba assembly (returns A, b, vel)
    # vel[row, col, :] is stored so postprocessing is fast.
    # -----------------------------
    @njit(cache=True)
    def assemble_A_b_vel_nb(center, coordsys, cornerslocal, area,
                            npanels, nfspanels, deltax,
                            vinf_x, gravity):
        """
        center: (N,3)
        coordsys: (N,3,3)
        cornerslocal: (N,2,4)
        area: (N,)  (unused here but kept for API match)
        Returns:
          A: (N,N)
          b: (N,)
          vel: (N,N,3)
        """
        N = npanels + nfspanels
        A = np.zeros((N, N), dtype=np.float64)
        b = np.zeros((N,), dtype=np.float64)
        vel = np.zeros((N, N, 3), dtype=np.float64)

        U2 = vinf_x * vinf_x

        # RHS: b[i] = n_i · vinf for hull rows, 0 for FS rows
        for i in range(npanels):
            ni0 = coordsys[i,0,2]
            ni1 = coordsys[i,1,2]
            ni2 = coordsys[i,2,2]
            b[i] = ni0*vinf_x  # vinf=(U,0,0)
        for i in range(npanels, N):
            b[i] = 0.0

        # Body BC rows
        for row in range(npanels):
            p0 = center[row,0]
            p1 = center[row,1]
            p2 = center[row,2]
            pp0 = p0
            pp1 = -p1
            pp2 = p2

            ni0 = coordsys[row,0,2]
            ni1 = coordsys[row,1,2]
            ni2 = coordsys[row,2,2]

            # hull cols
            for col in range(npanels):
                if row == col:
                    v0 = -0.5 * ni0
                    v1 = -0.5 * ni1
                    v2 = -0.5 * ni2
                else:
                    v = hs_influence_nb(np.array([p0,p1,p2]), center[col], coordsys[col], cornerslocal[col])
                    v0 = v[0]; v1 = v[1]; v2 = v[2]

                vp = hs_influence_nb(np.array([pp0,pp1,pp2]), center[col], coordsys[col], cornerslocal[col])

                # symmetry: vx even, vy odd, vz even
                v0 = v0 + vp[0]
                v1 = v1 - vp[1]
                v2 = v2 + vp[2]

                vel[row, col, 0] = v0
                vel[row, col, 1] = v1
                vel[row, col, 2] = v2

                A[row, col] = ni0*v0 + ni1*v1 + ni2*v2

            # fs cols
            for j in range(nfspanels):
                col = npanels + j
                v = hs_influence_nb(np.array([p0,p1,p2]), center[col], coordsys[col], cornerslocal[col])
                vp = hs_influence_nb(np.array([pp0,pp1,pp2]), center[col], coordsys[col], cornerslocal[col])
                v0 = v[0] + vp[0]
                v1 = v[1] - vp[1]
                v2 = v[2] + vp[2]

                vel[row, col, 0] = v0
                vel[row, col, 1] = v1
                vel[row, col, 2] = v2
                A[row, col] = ni0*v0 + ni1*v1 + ni2*v2

        # Free-surface BC rows
        for i in range(nfspanels):
            row = npanels + i

            p0 = center[row,0] + deltax
            p1 = center[row,1]
            p2 = 0.0
            pp0 = p0
            pp1 = -p1
            pp2 = 0.0

            # body cols
            for col in range(npanels):
                v = hs_influence_nb(np.array([p0,p1,p2]), center[col], coordsys[col], cornerslocal[col])
                vp = hs_influence_nb(np.array([pp0,pp1,pp2]), center[col], coordsys[col], cornerslocal[col])
                v0 = v[0] + vp[0]
                v1 = v[1] - vp[1]
                v2 = v[2] + vp[2]
                vel[row, col, 0] = v0
                vel[row, col, 1] = v1
                vel[row, col, 2] = v2

                H = phixx_influence_nb(np.array([p0,p1,p2]), center[col], coordsys[col], cornerslocal[col])
                Hp = phixx_influence_nb(np.array([pp0,pp1,pp2]), center[col], coordsys[col], cornerslocal[col])
                phi_xx = H[0,0] + Hp[0,0]

                A[row, col] = U2*phi_xx + gravity*v2

            # fs cols
            for j in range(nfspanels):
                col = npanels + j
                v = hs_influence_nb(np.array([p0,p1,p2]), center[col], coordsys[col], cornerslocal[col])
                vp = hs_influence_nb(np.array([pp0,pp1,pp2]), center[col], coordsys[col], cornerslocal[col])
                v0 = v[0] + vp[0]
                v1 = v[1] - vp[1]
                v2 = v[2] + vp[2]
                vel[row, col, 0] = v0
                vel[row, col, 1] = v1
                vel[row, col, 2] = v2

                H = phixx_influence_nb(np.array([p0,p1,p2]), center[col], coordsys[col], cornerslocal[col])
                Hp = phixx_influence_nb(np.array([pp0,pp1,pp2]), center[col], coordsys[col], cornerslocal[col])
                phi_xx = H[0,0] + Hp[0,0]

                A[row, col] = U2*phi_xx + gravity*v2

        return A, b, vel
