# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 22:33:11 2026

@author: tluke
"""

from __future__ import annotations
import numpy as np

PI = np.pi


def hs_influence(fieldpoint: np.ndarray,
                 center: np.ndarray,
                 coordsys: np.ndarray,
                 corners_local: np.ndarray,
                 eps: float = 1e-6) -> np.ndarray:
    """
    Port of Fortran hsinfluence: returns global velocity (∇φ) induced by a quad source panel.
    """
    p = coordsys.T @ (fieldpoint - center)
    x, y, z = p[0], p[1], p[2]

    c = corners_local
    ip1 = (1, 2, 3, 0)
    dphi = np.zeros(3, dtype=float)

    for k in range(4):
        k2 = ip1[k]
        xk, yk = c[0, k], c[1, k]
        xk2, yk2 = c[0, k2], c[1, k2]

        dx = xk2 - xk
        dy = yk2 - yk
        d = np.sqrt(dx*dx + dy*dy)
        if d <= eps:
            continue

        r1 = np.sqrt((x-xk)**2  + (y-yk)**2  + z*z)
        r2 = np.sqrt((x-xk2)**2 + (y-yk2)**2 + z*z)

        s = r1 + r2
        num = s - d
        den = s + d
        if num <= 0.0 or den <= 0.0:
            continue

        L = np.log(num / den)

        dphi[0] += (dy / d) * L
        dphi[1] -= (dx / d) * L

        # # robust angle term via atan2
        # e1 = (x - xk)**2 + z*z
        # h1 = (x - xk)*(y - yk)
        # n1 = dy*e1 - dx*h1
        # d1 = dx*z*r1

        # e2 = (x - xk2)**2 + z*z
        # h2 = (x - xk2)*(y - yk2)
        # n2 = dy*e2 - dx*h2
        # d2 = dx*z*r2

        # dphi[2] += (np.arctan2(n1, d1) - np.arctan2(n2, d2))
        
        #
        # --- Fortran-matching angle term (NO atan2) ---
        # m = dy/dx for the edge (k -> k2)
        if abs(dx) < 1e-30:
            # emulate the "huge slope" behavior instead of branching differently
            m = np.sign(dy) * 1e30
        else:
            m = dy / dx

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
            dphi[2] += np.arctan((m*e1 - h1) / denom1) - np.arctan((m*e2 - h2) / denom2)

    dphi *= (-1.0 / (4.0 * PI))
    return coordsys @ dphi


def phixx_influence(fieldpoint: np.ndarray,
                    center: np.ndarray,
                    coordsys: np.ndarray,
                    corners_local: np.ndarray,
                    eps: float = 1e-6) -> np.ndarray:
    """
    Port of Fortran phixxinfluence: returns global Hessian of φ.
    (Omitting the duplicated ddphi(1,*) block present in my old Fortran code.)
    """
    p = coordsys.T @ (fieldpoint - center)
    x, y, z = p[0], p[1], p[2]

    c = corners_local
    ip1 = (1, 2, 3, 0)

    ddphi = np.zeros((3, 3), dtype=float)

    for k in range(4):
        k2 = ip1[k]
        xk, yk = c[0, k], c[1, k]
        xk2, yk2 = c[0, k2], c[1, k2]

        dx = xk2 - xk
        dy = yk2 - yk
        d = np.sqrt(dx*dx + dy*dy)
        if d <= eps:
            continue

        r1 = np.sqrt((x-xk)**2  + (y-yk)**2  + z*z)
        r2 = np.sqrt((x-xk2)**2 + (y-yk2)**2 + z*z)

        s = r1 + r2
        denom1 = (s*s - d*d)
        if abs(denom1) <= 1e-30:
            continue

        big = (r1*r2) * (r1*r2
                         + (x-xk)*(x-xk2)
                         + (y-yk)*(y-yk2)
                         + z*z)
        if abs(big) <= 1e-30:
            continue

        ddphi[0,0] += (2.0*dy/denom1) * ((x-xk)/r1 + (x-xk2)/r2)
        ddphi[0,1] += (-2.0*dx/denom1) * ((x-xk)/r1 + (x-xk2)/r2)
        ddphi[0,2] += (z*dy*(r1+r2)) / big

        ddphi[1,0] += (2.0*dy/denom1) * ((y-yk)/r1 + (y-yk2)/r2)
        ddphi[1,1] += (-2.0*dx/denom1) * ((y-yk)/r1 + (y-yk2)/r2)
        ddphi[1,2] += ((-z*dx)*(r1+r2)) / big

        ddphi[2,0] += (2.0*dy * ((z/r1) + (z/r2))) / denom1
        ddphi[2,1] += (-2.0*dx * ((z/r1) + (z/r2))) / denom1
        ddphi[2,2] += ((( (x-xk)*(y-yk2) - (x-xk2)*(y-yk) ) * (r1+r2)) / big)
        
        # double the first block to match fortran
        ddphi[0,0] += (2.0*dy/denom1) * ((x-xk)/r1 + (x-xk2)/r2)
        ddphi[0,1] += (-2.0*dx/denom1) * ((x-xk)/r1 + (x-xk2)/r2)
        ddphi[0,2] += (z*dy*(r1+r2)) / big

    ddphi *= (-1.0 / (4.0 * PI))

    # global Hessian: coordsys * ddphi * coordsys^T
    return coordsys @ ddphi @ coordsys.T
