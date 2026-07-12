# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 22:33:11 2026

@author: tluke
"""

from __future__ import annotations
import numpy as np

from .revad import Node, sqrt, log, atan

PI = np.pi


def hs_influence(fieldpoint: np.ndarray,
                 center: np.ndarray,
                 coordsys: np.ndarray,
                 corners_local: np.ndarray,
                 eps: float = 1e-6) -> np.ndarray:
    """
    Port of Fortran hsinfluence: returns global velocity (∇φ) induced by a quad source panel.

    GENERIC over two calling conventions, same formula, same function:
      - plain float64 arrays in -> plain float64 array out (ordinary numeric use)
      - arrays of revad.Node in -> array of Node out (reverse-mode AD use,
        e.g. gradient_checks/trace_hs_influence_revad.py)
    Dispatch happens through sqrt/log/atan (imported from .revad, which know
    how to handle either a Node or a plain float) and through Node's
    comparison/abs dunders for the branch conditions below -- no code path
    here needs to know or care which kind of input it got. See
    rankine_panel/revad.py and gradient_checks/README_AD_ORACLE.md.
    """
    fieldpoint = np.asarray(fieldpoint, dtype=object)
    center = np.asarray(center, dtype=object)
    coordsys = np.asarray(coordsys, dtype=object)
    corners_local = np.asarray(corners_local, dtype=object)
    is_ad = isinstance(fieldpoint.flat[0], Node) or isinstance(coordsys.flat[0], Node)

    p = coordsys.T @ (fieldpoint - center)
    x, y, z = p[0], p[1], p[2]

    c = corners_local
    ip1 = (1, 2, 3, 0)
    dphi = np.array([0.0, 0.0, 0.0], dtype=object)

    for k in range(4):
        k2 = ip1[k]
        xk, yk = c[0, k], c[1, k]
        xk2, yk2 = c[0, k2], c[1, k2]

        dx = xk2 - xk
        dy = yk2 - yk
        d = sqrt(dx*dx + dy*dy)
        if d <= eps:
            continue

        r1 = sqrt((x-xk)**2  + (y-yk)**2  + z*z)
        r2 = sqrt((x-xk2)**2 + (y-yk2)**2 + z*z)

        s = r1 + r2
        num = s - d
        den = s + d
        if num <= 0.0 or den <= 0.0:
            continue

        L = log(num / den)

        dphi[0] = dphi[0] + (dy / d) * L
        dphi[1] = dphi[1] - (dx / d) * L

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
            # emulate the "huge slope" behavior instead of branching differently.
            # Deliberately a plain float, not a Node: this is a numerical
            # safety substitute, not a real function of the geometry (same
            # "disconnected constant" idea as the free-surface z=0 override
            # elsewhere) -- Node.__rmul__ absorbs a plain float as a zero-
            # derivative constant automatically, no explicit Node(...,[]) needed.
            dy_val = dy.val if isinstance(dy, Node) else dy
            m = np.sign(dy_val) * 1e30
        else:
            m = dy / dx

        e1 = (x - xk)**2 + z*z
        h1 = (x - xk)*(y - yk)
        r1 = sqrt((x-xk)**2  + (y-yk)**2  + z*z)

        e2 = (x - xk2)**2 + z*z
        h2 = (x - xk2)*(y - yk2)
        r2 = sqrt((x-xk2)**2 + (y-yk2)**2 + z*z)

        # Note: Fortran uses plocal(3)*r in denominator
        denom1 = z * r1
        denom2 = z * r2

        # Guard only to avoid NaNs; Fortran would blow up if denom=0,
        # but with sources above FS you should not hit z==0 here.
        if abs(denom1) > 1e-30 and abs(denom2) > 1e-30:
            dphi[2] = dphi[2] + atan((m*e1 - h1) / denom1) - atan((m*e2 - h2) / denom2)

    dphi = dphi * (-1.0 / (4.0 * PI))
    out = coordsys @ dphi

    if not is_ad:
        out = out.astype(np.float64)
    return out


def phixx_influence(fieldpoint: np.ndarray,
                    center: np.ndarray,
                    coordsys: np.ndarray,
                    corners_local: np.ndarray,
                    eps: float = 1e-6) -> np.ndarray:
    """
    Port of Fortran phixxinfluence: returns global Hessian of φ.

    GENERIC over plain float64 arrays and arrays of revad.Node, same as
    hs_influence above -- see that function's docstring for the dispatch
    mechanism, not repeated here.

    Matches the production numba_kernels.py::phixx_influence_nb formula
    (no duplicated first block). A previous version of this file had an
    extra "double the first block to match fortran" duplication that
    phixx_influence_nb never had -- a parked, unresolved discrepancy between
    two previously-independent code paths. Consolidating this into one
    generic function (used for both AD and plain numeric evaluation) forces
    picking one formula; matching production is the correct choice here.
    Reversible if a Fortran cross-check later shows the duplication was
    actually right -- worth noting the docstring on the old version already
    said "(Omitting the duplicated ddphi(1,*) block...)" while the code
    beneath it still had that block, i.e. this was already an internal
    inconsistency, not a deliberate, verified design choice.
    """
    fieldpoint = np.asarray(fieldpoint, dtype=object)
    center = np.asarray(center, dtype=object)
    coordsys = np.asarray(coordsys, dtype=object)
    corners_local = np.asarray(corners_local, dtype=object)
    is_ad = isinstance(fieldpoint.flat[0], Node) or isinstance(coordsys.flat[0], Node)

    p = coordsys.T @ (fieldpoint - center)
    x, y, z = p[0], p[1], p[2]

    c = corners_local
    ip1 = (1, 2, 3, 0)

    ddphi = np.zeros((3, 3), dtype=object)

    for k in range(4):
        k2 = ip1[k]
        xk, yk = c[0, k], c[1, k]
        xk2, yk2 = c[0, k2], c[1, k2]

        dx = xk2 - xk
        dy = yk2 - yk
        d = sqrt(dx*dx + dy*dy)
        if d <= eps:
            continue

        r1 = sqrt((x-xk)**2  + (y-yk)**2  + z*z)
        r2 = sqrt((x-xk2)**2 + (y-yk2)**2 + z*z)

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

        ddphi[0,0] = ddphi[0,0] + (dy*2.0/denom1) * ((x-xk)/r1 + (x-xk2)/r2)
        ddphi[0,1] = ddphi[0,1] + (dx*(-2.0)/denom1) * ((x-xk)/r1 + (x-xk2)/r2)
        ddphi[0,2] = ddphi[0,2] + (z*dy*(r1+r2)) / big

        ddphi[1,0] = ddphi[1,0] + (dy*2.0/denom1) * ((y-yk)/r1 + (y-yk2)/r2)
        ddphi[1,1] = ddphi[1,1] + (dx*(-2.0)/denom1) * ((y-yk)/r1 + (y-yk2)/r2)
        ddphi[1,2] = ddphi[1,2] + ((z*(-1.0)*dx)*(r1+r2)) / big

        ddphi[2,0] = ddphi[2,0] + (dy*2.0 * ((z/r1) + (z/r2))) / denom1
        ddphi[2,1] = ddphi[2,1] + (dx*(-2.0) * ((z/r1) + (z/r2))) / denom1
        ddphi[2,2] = ddphi[2,2] + (((x-xk)*(y-yk2) - (x-xk2)*(y-yk)) * (r1+r2)) / big

    ddphi = ddphi * (-1.0 / (4.0 * PI))

    # global Hessian: coordsys * ddphi * coordsys^T
    out = coordsys @ ddphi @ coordsys.T

    if not is_ad:
        out = out.astype(np.float64)
    return out
