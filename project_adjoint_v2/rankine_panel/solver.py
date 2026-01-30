# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 22:34:52 2026

@author: tluke
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve as dense_solve
from scipy.linalg import lu_factor, lu_solve

from .types import PanelFile, PanelGeometry
from .geometry import panel_geometry_all
from .influence import hs_influence, phixx_influence
from .numba_kernels import _require_numba, assemble_A_b_vel_nb
#from .numba_kernels import assemble_A_b_vel_nb
from .slae import solve_simqit
'''
not from rankine_panel.numba_kernels ... 
(relative imports are more robust when people run scripts in different ways).
'''
# @dataclass
# class FlowParams:
#     gravity: float   = 9.81
#     length: float    = 1.0       # fortran constants module sets this; pass the right one
#     rho_water: float = 1025.0
#     rho_ref: float   = 1000.0   # used in cw denominator in fortran code

@dataclass
class FlowParams:
    gravity: float   = 9.80665
    length: float    = 1.0
    rho_water: float = 1025.0
    rho_ref: float   = 1000.0

@dataclass
class FlowResult:
    sigma: np.ndarray     # (N,)
    vtotal: np.ndarray    # (N,3)
    vn: np.ndarray        # (N,)
    vt1: np.ndarray       # (N,)
    vt2: np.ndarray       # (N,)
    cp: np.ndarray        # (N,)
    zeta: np.ndarray      # (N,)
    force: np.ndarray     # (3,)
    cw: float
    sourcesink: float
    iter_count: int = 0  # dense solve has no iterations; keep for compatibility


def mirror_y(p: np.ndarray) -> np.ndarray:
    pm = p.copy()
    pm[1] *= -1.0
    return pm


def apply_halfship_symmetry(v: np.ndarray, vp: np.ndarray) -> np.ndarray:
    # vx even, vy odd, vz even
    v = v.copy()
    v[0] += vp[0]
    v[1] -= vp[1]
    v[2] += vp[2]
    return v


class RankineWaveResistanceSolver:
    
    def __init__(self, params: FlowParams | None = None):
        self.params = params or FlowParams()


    def build_geometry(self, pf: PanelFile) -> PanelGeometry:
        return panel_geometry_all(pf.points, pf.panels)


    def vinf_from_Fr(self, Fr: float) -> np.ndarray:
        U = Fr * np.sqrt(self.params.length * self.params.gravity)
        return np.array([U, 0.0, 0.0], dtype=float)


    def assemble_A_b(self, pf: PanelFile, geom: PanelGeometry, Fr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Assemble A and b exactly like my Fortran (but without SIMQIT padding).
        Returns (A, b, vinf)
        """
        npanels = pf.npanels
        nfspanels = pf.nfspanels
        N = pf.N

        vinf = self.vinf_from_Fr(Fr)
        U2 = vinf[0] ** 2
        g = self.params.gravity

        center = geom.center.T        # (N,3) easier row-wise
        coordsys = np.moveaxis(geom.coordsys, 2, 0)      # (N,3,3)
        cornerslocal = np.moveaxis(geom.cornerslocal, 2, 0)  # (N,2,4)

        normals = coordsys[:, :, 2]   # (N,3)

        A = np.zeros((N, N), dtype=float)
        b = np.zeros((N,), dtype=float)

        # RHS
        b[:npanels] = normals[:npanels] @ vinf
        b[npanels:] = 0.0

        # PART 1: Body BC
        for i in range(npanels):
            row = i
            p = center[row].copy()
            pp = mirror_y(p)
            ni = normals[row]

            # hull panels
            for j in range(npanels):
                col = j
                if row == col:
                    v = -0.5 * ni
                else:
                    v = hs_influence(p, center[col], coordsys[col], cornerslocal[col])

                vp = hs_influence(pp, center[col], coordsys[col], cornerslocal[col])
                v = apply_halfship_symmetry(v, vp)
                A[row, col] = np.dot(ni, v)

            # free surface panels
            for j in range(nfspanels):
                col = npanels + j
                v = hs_influence(p, center[col], coordsys[col], cornerslocal[col])
                vp = hs_influence(pp, center[col], coordsys[col], cornerslocal[col])
                v = apply_halfship_symmetry(v, vp)
                A[row, col] = np.dot(ni, v)

        # PART 2: Free Surface BC
        for i in range(nfspanels):
            row = npanels + i

            p = center[row].copy()
            p[0] += pf.deltax
            p[2] = 0.0
            pp = mirror_y(p)

            # body panels
            for j in range(npanels):
                col = j
                v = hs_influence(p, center[col], coordsys[col], cornerslocal[col])
                vp = hs_influence(pp, center[col], coordsys[col], cornerslocal[col])
                v = apply_halfship_symmetry(v, vp)

                H = phixx_influence(p, center[col], coordsys[col], cornerslocal[col])
                Hp = phixx_influence(pp, center[col], coordsys[col], cornerslocal[col])
                phi_xx = H[0, 0] + Hp[0, 0]

                A[row, col] = U2 * phi_xx + g * v[2]

            # free surface panels
            for j in range(nfspanels):
                col = npanels + j
                v = hs_influence(p, center[col], coordsys[col], cornerslocal[col])
                vp = hs_influence(pp, center[col], coordsys[col], cornerslocal[col])
                v = apply_halfship_symmetry(v, vp)

                H = phixx_influence(p, center[col], coordsys[col], cornerslocal[col])
                Hp = phixx_influence(pp, center[col], coordsys[col], cornerslocal[col])
                phi_xx = H[0, 0] + Hp[0, 0]

                A[row, col] = U2 * phi_xx + g * v[2]

        return A, b, vinf
    

    # # solve with numpy and scipy.linalg.solve
    # def solve_Numpy(self, pf: PanelFile, geom: PanelGeometry, Fr: float) -> FlowResult:
    #     """
    #     End-to-end: assemble, solve sigma, recompute vtotal, cp, zeta, force, cw.
    #     """
    #     npanels = pf.npanels
    #     nfspanels = pf.nfspanels
    #     N = pf.N

    #     A, b, vinf = self.assemble_A_b(pf, geom, Fr)
        
    #     #-------------------------------
    #     # SOLVE:
    #     #
    #     #sigma = dense_solve(A, b)  # dense solve for toy version  => scipy.linalg.solve
        
    #     # modification for reuse 
    #     lu, piv = lu_factor(A)
    #     sigma = lu_solve((lu, piv), b)
    #     #
    #     # END SOLVE
    #     #-------------------------------

    #     # Recompute vtotal without storing vel(i,j) (saves huge memory)
    #     center = geom.center.T
    #     coordsys = np.moveaxis(geom.coordsys, 2, 0)
    #     cornerslocal = np.moveaxis(geom.cornerslocal, 2, 0)
    #     normals = coordsys[:, :, 2]
    #     t1 = coordsys[:, :, 0]
    #     t2 = coordsys[:, :, 1]

    #     vtotal = np.zeros((N, 3), dtype=float)
    #     vtotal[:] = -vinf[None, :]

    #     # vtotal(i) += sum_j sigma(j) * vel(i,j)
    #     for i in range(N):
    #         p = center[i].copy()
    #         pp = mirror_y(p)

    #         for j in range(N):
    #             if i < npanels and j < npanels and i == j:
    #                 # same singular handling used in fortran assembly
    #                 v = -0.5 * normals[i]
    #             else:
    #                 v = hs_influence(p, center[j], coordsys[j], cornerslocal[j])

    #             vp = hs_influence(pp, center[j], coordsys[j], cornerslocal[j])
    #             v = apply_halfship_symmetry(v, vp)
    #             vtotal[i] += sigma[j] * v

    #     vn = np.einsum("ij,ij->i", normals, vtotal)
    #     vt1 = np.einsum("ij,ij->i", t1, vtotal)
    #     vt2 = np.einsum("ij,ij->i", t2, vtotal)

    #     # cp on hull panels only
    #     cp = np.zeros(N, dtype=float)
    #     U2 = np.dot(vinf, vinf)
    #     cp[:npanels] = 1.0 - (np.sum(vtotal[:npanels]**2, axis=1) / U2)

    #     # sourcesink check (fortran printed this)
    #     sourcesink = float(np.sum(sigma * geom.area))
    #     # print or log it if desired

    #     # wave elevation on FS
    #     zeta = np.zeros(N, dtype=float)
    #     U = vinf[0]
    #     g = self.params.gravity
    #     for i in range(nfspanels):
    #         row = npanels + i
    #         zeta[row] = (U / g) * (vtotal[row, 0] + U)

    #     # forces on hull (my fortran code uses rho=1025 for force)
    #     rho = self.params.rho_water
    #     force = np.zeros(3, dtype=float)

    #     for i in range(npanels):
    #         if center[i, 2] < 0.0:
    #             pressure_term = 0.5 * rho * U2 * cp[i] - rho * g * center[i, 2]
    #             force += geom.area[i] * pressure_term * normals[i]

    #     # reference area S (wetted area) used in cw denom
    #     S = float(np.sum(geom.area[:npanels][center[:npanels, 2] < 0.0]))
    #     cw = -force[0] / (0.5 * self.params.rho_ref * (U**2) * S)

    #     return FlowResult(
    #         sigma=sigma, vtotal=vtotal, vn=vn, vt1=vt1, vt2=vt2,
    #         cp=cp, zeta=zeta, force=force, cw=float(cw), 
    #         sourcesink=sourcesink, iter_count=0
    #     )
    
    
    # solve with NUMBA
    def solve(self, pf: PanelFile, geom: PanelGeometry, Fr: float) -> FlowResult:
        """
        End-to-end: assemble, solve sigma, recompute vtotal, cp, zeta, force, cw.
        """
        npanels = pf.npanels
        nfspanels = pf.nfspanels
        N = pf.N

        vinf = self.vinf_from_Fr(Fr)
        g = self.params.gravity
        
        # Prepare arrays in shapes Numba expects
        center = geom.center.T.astype(np.float64)                      # (N,3)
        coordsys = np.moveaxis(geom.coordsys, 2, 0).astype(np.float64) # (N,3,3)
        cornerslocal = np.moveaxis(geom.cornerslocal, 2, 0).astype(np.float64)  # (N,2,4)
        area = geom.area.astype(np.float64)
        
        # Assemble in Numba (this will compile on first run, then be fast)
        A, b, vel = assemble_A_b_vel_nb(center, coordsys, cornerslocal, area,
                                        npanels, nfspanels, pf.deltax,
                                        vinf[0], g)
        
        # scipy.linalg.solve
        ##sigma = dense_solve(A, b)  # dense solve for toy version  => scipy.linalg.solve
        #sigma_lu = dense_solve(A, b)
        
        #-------------------------------
        # SOLVE:
        #
        # # using scipy.linalg.solve:
        # #sigma = dense_solve(A, b)  # dense solve for toy version  => scipy.linalg.solve
        
        # modification for reuse (slightly slower, but it will be worth it)
        lu, piv = lu_factor(A) # pre factor for use in adjoints and optimization
        
        # Forward (state): A sigma = b
        sigma = lu_solve((lu, piv), b)          # trans=0 (default)
        
        # Adjoint: A^T lambda = g (coming in version 2)
        #lam = lu_solve((lu, piv), dJ_dsigma, trans=1)   # trans=1 for adjoint ONLY => # solves A.T * sigma = rhs_adj
        

        #
        # END SOLVE
        #-------------------------------
        N = pf.N
        
        
        
        amatmax = float(np.max(np.abs(A)))
        sing = 1e-6 * amatmax
        av = float(np.sqrt(1.0 / N))
        dx = 1e-6
        
        
        # sigma_smq, iters = solve_simqit(A, b, sing, av, dx)

        # print("||sigma_lu - sigma_simqit|| / ||sigma_simqit|| =", 
        #       np.linalg.norm(sigma_lu - sigma_smq) / (np.linalg.norm(sigma_smq) + 1e-30))
        
        
        #sigma = sigma_lu
        #sigma = sigma_smq
        
        # check the residuals:
        r1 = np.linalg.norm(A @ sigma - b) / (np.linalg.norm(b) + 1e-30)
        r2 = np.linalg.norm(A.T @ sigma - b) / (np.linalg.norm(b) + 1e-30)
        print("relres A*s=b :", r1)
        print("symmetry check on relres^T => AT*s=b:", r2)
        
        # Fast vtotal from stored vel
        # vtotal = -vinf + sum_j sigma[j] * vel[:,j,:]
        #vtotal = -vinf[None, :] + np.einsum("ijm,j->im", vel, sigma)
        vtotal = -vinf[None,:] + np.einsum("ijm,j->im", vel, sigma)
        
        # Now vn/vt/cp/zeta/force/cw exactly as before (vectorized)
        normals = coordsys[:, :, 2]
        t1 = coordsys[:, :, 0]
        t2 = coordsys[:, :, 1]
        
        vn  = np.einsum("ij,ij->i", normals, vtotal)
        vt1 = np.einsum("ij,ij->i", t1, vtotal)
        vt2 = np.einsum("ij,ij->i", t2, vtotal)
        
        
        # cp on hull panels only
        U2 = float(vinf[0] * vinf[0])
        cp = np.zeros(pf.N, dtype=np.float64)
        cp[:npanels] = 1.0 - (np.sum(vtotal[:npanels]**2, axis=1) / U2)
        
        
        # sourcesink check (fortran printed this)
        sourcesink = float(np.sum(sigma * geom.area))
        
        # wave elevation on FS
        zeta = np.zeros(pf.N, dtype=np.float64)
        U = vinf[0]
        for i in range(nfspanels):
            row = npanels + i
            zeta[row] = (U / g) * (vtotal[row, 0] + U)

        # forces on hull (my fortran code uses rho=1025 for force)
        rho = self.params.rho_water
        force = np.zeros(3, dtype=float)

        for i in range(npanels):
            if center[i, 2] < 0.0:
                pressure_term = 0.5 * rho * U2 * cp[i] - rho * g * center[i, 2]
                force += geom.area[i] * pressure_term * normals[i]

        # reference area S (wetted area) used in cw denom
        S = float(np.sum(geom.area[:npanels][center[:npanels, 2] < 0.0]))
        cw = -force[0] / (0.5 * self.params.rho_ref * (U**2) * S)

        return FlowResult(
            sigma=sigma, vtotal=vtotal, vn=vn, vt1=vt1, vt2=vt2,
            cp=cp, zeta=zeta, force=force, cw=float(cw), 
            sourcesink=sourcesink, iter_count=0
        )

