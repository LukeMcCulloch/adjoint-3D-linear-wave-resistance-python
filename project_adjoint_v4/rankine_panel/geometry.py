# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 22:31:52 2026

@author: tluke
"""

from __future__ import annotations
import numpy as np
from .types import PanelGeometry
from .revad import Node, sqrt


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

    GENERIC over plain float64 points and points containing revad.Node, same
    intent as rankine_panel.influence.hs_influence/phixx_influence -- but
    dispatch here is at the TOP of the function, not per-operation, and
    that's deliberate: unlike those two, this function is vectorized over
    ALL N panels at once AND sits on the live production forward-solve path
    (RankineWaveResistanceSolver.build_geometry calls it directly). The
    ordinary float64 branch below is left byte-for-byte the same formula as
    before -- np.sqrt, dtype=float throughout -- zero risk to production
    performance. The AD branch uses np.vectorize(sqrt) in place of np.sqrt
    (confirmed: np.sqrt itself does not know how to operate on a Node --
    raises trying to call a nonexistent Node.sqrt() method) and
    dtype=object arrays throughout; slower, but this path only runs when
    explicitly given Node-valued points (gradient_checks/, never the
    production solve).
    """
    is_ad = isinstance(np.asarray(points).flat[0], Node)
    dtype = object if is_ad else float
    _sqrt = np.vectorize(sqrt) if is_ad else np.sqrt

    points = np.asarray(points, dtype=dtype)
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
    iv = iv / _sqrt(np.sum(iv*iv, axis=0, keepdims=True))

    jvbar = m3 - m1

    # nv = cross(iv, jvbar)
    nv = np.vstack([
        iv[1,:]*jvbar[2,:] - iv[2,:]*jvbar[1,:],
        iv[2,:]*jvbar[0,:] - iv[0,:]*jvbar[2,:],
        iv[0,:]*jvbar[1,:] - iv[1,:]*jvbar[0,:],
    ])
    nv = nv / _sqrt(np.sum(nv*nv, axis=0, keepdims=True))

    # jv = cross(nv, iv)
    jv = np.vstack([
        nv[1,:]*iv[2,:] - nv[2,:]*iv[1,:],
        nv[2,:]*iv[0,:] - nv[0,:]*iv[2,:],
        nv[0,:]*iv[1,:] - nv[1,:]*iv[0,:],
    ])

    coordsys = np.zeros((3,3,N), dtype=dtype)
    coordsys[:,0,:] = iv
    coordsys[:,1,:] = jv
    coordsys[:,2,:] = nv

    center0 = 0.25*(m1+m2+m3+m4)

    d = corners - center0[:, None, :]  # (3,4,N)

    # local coords of corners
    xi  = np.einsum("i k n, i n -> k n", d, iv)  # (4,N)
    eta = np.einsum("i k n, i n -> k n", d, jv)  # (4,N)

    cornerslocal = np.zeros((2,4,N), dtype=dtype)
    cornerslocal[0,:,:] = xi
    cornerslocal[1,:,:] = eta

    xi_next  = xi[ip1, :]
    eta_next = eta[ip1, :]

    dxi  = xi_next - xi
    deta = eta_next - eta

    area = 0.5 * np.sum(deta * (xi_next + xi), axis=0)

    momentxi  = np.sum(dxi  * (eta_next**2 + eta*eta_next + eta**2), axis=0)
    momenteta = np.sum(deta * (xi_next**2  + xi*xi_next   + xi**2), axis=0)

    momentxi  = momentxi  * (-1.0/6.0)
    momenteta = momenteta * ( 1.0/6.0)

    # Match my Fortran: only correct centroid if area > 1e-10 (not abs(area))
    xic  = np.zeros(N, dtype=dtype)
    etac = np.zeros(N, dtype=dtype)
    mask = area > 1e-10
    xic[mask]  = momenteta[mask] / area[mask]
    etac[mask] = momentxi[mask]  / area[mask]

    # cornerslocal[0,:,:] -= xic[None, None, :]
    # cornerslocal[1,:,:] -= etac[None, None, :]
    cornerslocal[0,:,:] -= xic[None, :]
    cornerslocal[1,:,:] -= etac[None, :]

    center = center0 + iv * xic[None, :] + jv * etac[None, :]

    return PanelGeometry(center=center, coordsys=coordsys, cornerslocal=cornerslocal, area=area)


def panel_geometry_one(corners):
    """
    Convenience wrapper around panel_geometry_all for exactly ONE panel,
    given its 4 corner vertices directly (rather than a points/panels
    index pair) -- used by the gradient_checks/ AD traces, which only ever
    need one or two panels' worth of geometry at a time, not the full mesh.

    corners: [c0,c1,c2,c3], each a length-3 sequence (plain floats OR
             revad.Node) -- one panel's 4 vertices, in order.

    Returns (center, coordsys, cornerslocal, area) with the SAME shapes
    panel_geometry_revad used to return (center: (3,), coordsys: (3,3),
    cornerslocal: (2,4), area: scalar) -- this just builds the minimal
    points/panels arrays panel_geometry_all expects, calls it (the one
    real implementation, no separate formula here), and squeezes the
    trailing N=1 axis back out.
    """
    points = np.empty((3, 4), dtype=object)
    for k in range(4):
        for i in range(3):
            points[i, k] = corners[k][i]
    panels = np.array([[0], [1], [2], [3]], dtype=np.int64)

    geom = panel_geometry_all(points, panels)
    center = geom.center[:, 0]
    coordsys = geom.coordsys[:, :, 0]
    cornerslocal = geom.cornerslocal[:, :, 0]
    area = geom.area[0]
    return center, coordsys, cornerslocal, area
