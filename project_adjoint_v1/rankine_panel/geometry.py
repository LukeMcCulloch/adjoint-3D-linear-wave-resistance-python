# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 22:31:52 2026

@author: tluke
"""

from __future__ import annotations
import numpy as np
from .types import PanelGeometry


def panel_geometry_all(points: np.ndarray, panels: np.ndarray) -> PanelGeometry:
    """
    Vectorized translation of my Fortran panelgeometry.

    points: (3, npoints)
    panels: (4, N) 0-based indices into points columns
    returns PanelGeometry with:
      center: (3,N)
      coordsys: (3,3,N)
      cornerslocal: (2,4,N)
      area: (N,)
    """
    points = np.asarray(points, dtype=float)
    panels = np.asarray(panels, dtype=np.int64)
    N = panels.shape[1]
    ip1 = np.array([1, 2, 3, 0], dtype=int)

    # corners: (3,4,N)
    corners = points[:, panels]  # fancy indexing => (3,4,N)

    m1 = 0.5*(corners[:,0,:] + corners[:,1,:])
    m2 = 0.5*(corners[:,1,:] + corners[:,2,:])
    m3 = 0.5*(corners[:,2,:] + corners[:,3,:])
    m4 = 0.5*(corners[:,3,:] + corners[:,0,:])

    iv = m2 - m4
    iv /= np.sqrt(np.sum(iv*iv, axis=0, keepdims=True))

    jvbar = m3 - m1

    # nv = cross(iv, jvbar)
    nv = np.vstack([
        iv[1,:]*jvbar[2,:] - iv[2,:]*jvbar[1,:],
        iv[2,:]*jvbar[0,:] - iv[0,:]*jvbar[2,:],
        iv[0,:]*jvbar[1,:] - iv[1,:]*jvbar[0,:],
    ])
    nv /= np.sqrt(np.sum(nv*nv, axis=0, keepdims=True))

    # jv = cross(nv, iv)
    jv = np.vstack([
        nv[1,:]*iv[2,:] - nv[2,:]*iv[1,:],
        nv[2,:]*iv[0,:] - nv[0,:]*iv[2,:],
        nv[0,:]*iv[1,:] - nv[1,:]*iv[0,:],
    ])

    coordsys = np.zeros((3,3,N), dtype=float)
    coordsys[:,0,:] = iv
    coordsys[:,1,:] = jv
    coordsys[:,2,:] = nv

    center0 = 0.25*(m1+m2+m3+m4)

    d = corners - center0[:, None, :]  # (3,4,N)

    # local coords of corners
    xi  = np.einsum("i k n, i n -> k n", d, iv)  # (4,N)
    eta = np.einsum("i k n, i n -> k n", d, jv)  # (4,N)

    cornerslocal = np.zeros((2,4,N), dtype=float)
    cornerslocal[0,:,:] = xi
    cornerslocal[1,:,:] = eta

    xi_next  = xi[ip1, :]
    eta_next = eta[ip1, :]

    dxi  = xi_next - xi
    deta = eta_next - eta

    area = 0.5 * np.sum(deta * (xi_next + xi), axis=0)

    momentxi  = np.sum(dxi  * (eta_next**2 + eta*eta_next + eta**2), axis=0)
    momenteta = np.sum(deta * (xi_next**2  + xi*xi_next   + xi**2), axis=0)

    momentxi  *= (-1.0/6.0)
    momenteta *= ( 1.0/6.0)

    # Match my Fortran: only correct centroid if area > 1e-10 (not abs(area))
    xic  = np.zeros(N)
    etac = np.zeros(N)
    mask = area > 1e-10
    xic[mask]  = momenteta[mask] / area[mask]
    etac[mask] = momentxi[mask]  / area[mask]

    # cornerslocal[0,:,:] -= xic[None, None, :]
    # cornerslocal[1,:,:] -= etac[None, None, :]
    cornerslocal[0,:,:] -= xic[None, :]
    cornerslocal[1,:,:] -= etac[None, :]

    center = center0 + iv * xic[None, :] + jv * etac[None, :]

    return PanelGeometry(center=center, coordsys=coordsys, cornerslocal=cornerslocal, area=area)
