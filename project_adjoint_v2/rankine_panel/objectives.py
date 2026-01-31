# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 11:45:52 2026

@author: tluke

NOTE: these are acutally gradients of the objective 

We want to minimize wave resistance by changing the bow geometry

so for this first pass 

J = Fx ~ Cw 

m = a bow shape function

"""


import numpy as np


class SourceStrengthGradients(object):
    '''
    D J / D sigma
    '''
    


    @staticmethod
    def compute_dJ_dsigma_cw(
        vel, vtotal, normals, area, center,
        npanels, vinf, rho_water, rho_ref):
        '''
        J => d Cw / d sigma
        Analytic dJ/dsigma for J = -F_x/(0.5 rho Cd U**2)

    
        Parameters
        ----------
        vel : TYPE
            DESCRIPTION.
        vtotal : TYPE
            DESCRIPTION.
        normals : TYPE
            DESCRIPTION.
        area : TYPE
            DESCRIPTION.
        center : TYPE
            DESCRIPTION.
        npanels : TYPE
            DESCRIPTION.
        vinf : TYPE
            DESCRIPTION.
        rho_water : TYPE
            DESCRIPTION.
        rho_ref : TYPE
            DESCRIPTION.
    
        Returns
        -------
        dJ_dsigma : TYPE
            DESCRIPTION.
            
        This function computes 
        the gradient of sigma (source strengths) 
        with respect to Cw, the coefficient of wave drag
    
        '''
        
        N = vel.shape[0]
        U = float(vinf[0])
        U2 = U * U
    
        wetted = center[:npanels, 2] < 0.0
        S = float(np.sum(area[:npanels][wetted]))
        C = 0.5 * rho_ref * U2 * S  # cw denominator
    
        # dJ/dv is nonzero only on wetted hull panels
        dJ_dv = np.zeros((N, 3), dtype=np.float64)
    
        # scale_i = rho * A_i * n_ix / C
        scale = np.zeros(npanels, dtype=np.float64)
        scale[wetted] = (rho_water * area[:npanels][wetted] * normals[:npanels, 0][wetted]) / C
    
        dJ_dv[:npanels, :] = scale[:, None] * vtotal[:npanels, :]
    
        # dJ/dsigma_j = sum_i dJ_dv[i] · vel[i,j]
        dJ_dsigma = np.einsum("ik,ijk->j", dJ_dv, vel) # goes over all panels, including fs where the obj is 0
        
        # # hull-only version (equivalent)
        # dJ_dsigma = np.einsum("ik,ijk->j", dJ_dv[:npanels], vel[:npanels])
        return dJ_dsigma
    
    
    # @staticmethod
    # def compute_dJ_dsigma_JnegFx(
    #     vel: np.ndarray,        # (N, N, 3)
    #     vtotal: np.ndarray,     # (N, 3)
    #     normals: np.ndarray,    # (N, 3)
    #     area: np.ndarray,       # (N,)
    #     center: np.ndarray,     # (N, 3)
    #     npanels: int,
    #     rho_water: float,) -> np.ndarray:
        
    #     """
    #     J => d Fx / d sigma
    #     Analytic dJ/dsigma for J = -F_x.

    #     for our wetted hull panels:
    #       dJ/dv_i = -rho * area_i * n_ix * v_i
    #     and dJ/dsigma_j = sum_i (dJ/dv_i)·vel[i,j]
    #     """
    #     N = vel.shape[0]
    #     wetted = center[:npanels, 2] < 0.0

    #     dJ_dv = np.zeros((N, 3), dtype=np.float64)

    #     scale = np.zeros(npanels, dtype=np.float64)
    #     scale[wetted] = rho_water * area[:npanels][wetted] * normals[:npanels, 0][wetted]

    #     dJ_dv[:npanels, :] = scale[:, None] * vtotal[:npanels, :]

    #     dJ_dsigma = np.einsum("ik,ijk->j", dJ_dv, vel)
    #     return dJ_dsigma
    
    
    @staticmethod
    def compute_dJ_dsigma_JnegFx(
        vel: np.ndarray,        # (N, N, 3)
        vtotal: np.ndarray,     # (N, 3)
        normals: np.ndarray,    # (N, 3)
        area: np.ndarray,       # (N,)
        center: np.ndarray,     # (N, 3)
        npanels: int,
        rho_water: float,) -> np.ndarray:
        
        """
        Analytic dJ/dsigma for J = -F_x.
        
        dJ/d sigma => d Fx / d sigma

        With your force definition, for wetted hull panels:
          dJ/dv_i = -rho * area_i * n_ix * v_i
        and dJ/dsigma_j = sum_i (dJ/dv_i)·vel[i,j]
        """
        N = vel.shape[0]
        wetted = center[:npanels, 2] < 0.0

        dJ_dv = np.zeros((N, 3), dtype=np.float64)

        scale = np.zeros(npanels, dtype=np.float64)
        scale[wetted] = rho_water * area[:npanels][wetted] * normals[:npanels, 0][wetted]

        dJ_dv[:npanels, :] = scale[:, None] * vtotal[:npanels, :]

        dJ_dsigma = np.einsum("ik,ijk->j", dJ_dv, vel)
        return dJ_dsigma
    
    
class ShapeGradients(object):
    '''
    D J / D m
    '''
    
    # @staticmethod
    # def hull_vertex_indices(panels4xN: np.ndarray, npanels: int) -> np.ndarray:
    #     return np.unique(panels4xN[:, :npanels].ravel())
    
    
    # @staticmethod
    # def apply_beam_scale(points0: np.ndarray, hull_verts: np.ndarray, m: float, z_cut=0.0) -> np.ndarray:
    #     pts = points0.copy()
    #     idx = hull_verts
    #     mask = pts[2, idx] < z_cut
    #     idx2 = idx[mask]
    #     pts[1, idx2] *= (1.0 + m)
    #     return pts
    
    
    # @staticmethod
    # def JnegFx_from_sigma(postprocess_JnegFx, sigma):
    #     return float(postprocess_JnegFx(sigma))
    
    
    # @staticmethod
    # def check_shape_grad_beam_scale(
    #     pf,
    #     points0,
    #     Fr,
    #     params,
    #     assemble_from_points,          # function(points)->(A,b,vel,vinf,center,coordsys,area)
    #     objectives_module,             # Objectives.compute_dJ_dsigma_JnegFx
    #     eps_m=1e-5,
    #     m0=0.0
    # ):
    #     hull_verts = hull_vertex_indices(pf.panels, pf.npanels)
    
    #     # ---- baseline
    #     A, b, vel, vinf, center, coordsys, area = assemble_from_points(points0, pf, Fr, params)
    #     lu, piv = lu_factor(A)
    #     sigma = lu_solve((lu, piv), b)
    #     vtotal = -vinf[None, :] + np.einsum("ijm,j->im", vel, sigma)
    
    #     normals = coordsys[:, :, 2]
    #     rho = params.rho_water
    #     gravity = params.gravity
    
    #     # objective closure J = -Fx from sigma
    #     postprocess = Validate_dj_dsigma.make_postprocess_JnegFx(
    #         vel, vinf, coordsys, area, center, pf.npanels, rho, gravity
    #     )
    #     J0 = JnegFx_from_sigma(postprocess, sigma)
    
    #     # adjoint RHS for J=-Fx and adjoint solve
    #     dJ_dsigma = objectives_module.compute_dJ_dsigma_JnegFx(
    #         vel=vel, vtotal=vtotal, normals=normals, area=area, center=center,
    #         npanels=pf.npanels, rho_water=rho
    #     )
    #     lam = lu_solve((lu, piv), dJ_dsigma, trans=1)
    
    #     # ---- plus
    #     pts_p = apply_beam_scale(points0, hull_verts, m0 + eps_m)
    #     Ap, bp, velp, vinfp, centerp, coordsyp, areap = assemble_from_points(pts_p, pf, Fr, params)
    #     lup, pivp = lu_factor(Ap)
    #     sigmap = lu_solve((lup, pivp), bp)
    #     Jp = JnegFx_from_sigma(Validate_dj_dsigma.make_postprocess_JnegFx(
    #         velp, vinfp, coordsyp, areap, centerp, pf.npanels, rho, gravity
    #     ), sigmap)
    
    #     # ---- minus
    #     pts_m = apply_beam_scale(points0, hull_verts, m0 - eps_m)
    #     Am, bm, velm, vinfm, centerm, coordsym, aream = assemble_from_points(pts_m, pf, Fr, params)
    #     lum, pivm = lu_factor(Am)
    #     sigmam = lu_solve((lum, pivm), bm)
    #     Jm = JnegFx_from_sigma(Validate_dj_dsigma.make_postprocess_JnegFx(
    #         velm, vinfm, coordsym, aream, centerm, pf.npanels, rho, gravity
    #     ), sigmam)
    
    #     dJ_dm_fd = (Jp - Jm) / (2.0 * eps_m)
    
    #     # operator FD for adjoint term
    #     dA_dm = (Ap - Am) / (2.0 * eps_m)
    #     db_dm = (bp - bm) / (2.0 * eps_m)
    
    #     dJ_dm_adj = float(lam @ (db_dm - dA_dm @ sigma))
    
    #     print("\nBeam scale shape-gradient check (J=-Fx)")
    #     print("  m0 =", m0, " eps_m =", eps_m)
    #     print("  J0 =", J0)
    #     print("  FD dJ/dm      =", dJ_dm_fd)
    #     print("  adjoint dJ/dm =", dJ_dm_adj)
    #     print("  abs err       =", abs(dJ_dm_fd - dJ_dm_adj))
    #     print("  rel err       =", abs(dJ_dm_fd - dJ_dm_adj) / (abs(dJ_dm_fd) + 1e-30))