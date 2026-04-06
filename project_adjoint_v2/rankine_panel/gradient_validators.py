# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 14:08:17 2026

@author: Luke McCulloch
"""

import numpy as np

from scipy.linalg import lu_factor, lu_solve

from .objectives import SourceStrengthGradients


class Validate_dj_dsigma(object):
    
    
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
    #     Analytic dJ/dsigma for J = -F_x.

    #     With your force definition, for wetted hull panels:
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
    def make_postprocess_JnegFx(
            vel: np.ndarray,
            vinf: np.ndarray,
            coordsys: np.ndarray,   # (N,3,3)
            area: np.ndarray,
            center: np.ndarray,     # (N,3)
            npanels: int,
            rho_water: float,
            gravity: float,):
        """

        Parameters
        ----------
        vel : np.ndarray
            DESCRIPTION.
        vinf : np.ndarray
            DESCRIPTION.
        coordsys : np.ndarray
            DESCRIPTION.
        area : np.ndarray
            DESCRIPTION.
        center : np.ndarray
            DESCRIPTION.
        npanels : int
            DESCRIPTION.
        rho_water : float
            DESCRIPTION.
        gravity : float
            DESCRIPTION.

        Returns
        -------
        function : postprocess
            This function sums the pressures up over the panels
            giving J <=> -Fx as the result

        """
        
        normals = coordsys[:, :, 2]
        U2 = float(vinf[0] * vinf[0])
        

        def postprocess(sigma: np.ndarray) -> float:
            """
            

            Parameters
            ----------
            sigma : np.ndarray
                of source strengths

            Returns
            -------
            float
                J <=> -Fx 

            """
            vtotal = -vinf[None, :] + np.einsum("ijm,j->im", vel, sigma)

            cp = np.zeros(vtotal.shape[0], dtype=np.float64)
            cp[:npanels] = 1.0 - (np.sum(vtotal[:npanels] ** 2, axis=1) / U2)

            force = np.zeros(3, dtype=np.float64)
            for i in range(npanels):
                if center[i, 2] < 0.0:
                    pressure_term = 0.5 * rho_water * U2 * cp[i] - rho_water * gravity * center[i, 2]
                    force += area[i] * pressure_term * normals[i]

            J = -force[0]  # J = -Fx
            return float(J)

        return postprocess

    @staticmethod
    def check_dJ_dsigma(
            sigma: np.ndarray,
            dJ_dsigma: np.ndarray,
            postprocess,
            n_tests: int = 5,
            eps: float = 1e-6,
            seed: int = 0,) -> None:
        """
        Directional derivative check:
          (J(s+e d) - J(s-e d)) / (2e)  ≈  dJ_dsigma · d
        """
        rng = np.random.default_rng(seed)
        sigma = np.asarray(sigma, dtype=np.float64)
        dJ_dsigma = np.asarray(dJ_dsigma, dtype=np.float64)

        def eval_J(sig):
            out = postprocess(sig)
            return float(out[0] if isinstance(out, (tuple, list)) else out)

        J0 = eval_J(sigma)
        print("check_dJ_dsigma (J = -Fx)")
        print("  eps =", eps)
        print("  J0  =", J0)

        for t in range(n_tests):
            d = rng.standard_normal(sigma.shape[0]).astype(np.float64)
            d /= (np.linalg.norm(d) + 1e-30)

            Jp = eval_J(sigma + eps * d)
            Jm = eval_J(sigma - eps * d)

            fd = (Jp - Jm) / (2.0 * eps)
            an = float(np.dot(dJ_dsigma, d))

            abs_err = abs(fd - an)
            rel_err = abs_err / (abs(fd) + 1e-30)

            print(f"  test {t+1:02d}: fd={fd:+.6e}  an={an:+.6e}  abs={abs_err:.3e}  rel={rel_err:.3e}")
            
            
            
            

class Validate_shape_gradient(object):
    """
    """
    # Geometry utilities for shape gradient checks.
    
    
    @staticmethod
    def hull_vertex_indices(panels4xN: np.ndarray, npanels: int) -> np.ndarray:
        """Unique vertex ids used by hull panels only."""
        return np.unique(panels4xN[:, :npanels].ravel())
    
    
    @staticmethod
    def apply_beam_scale(points0: np.ndarray,
                         hull_verts: np.ndarray,
                         m: float,
                         z_cut: float = 0.0) -> np.ndarray:
        """
        Hull shape modification 
        
        Beam scaling mode on hull vertices:
          y <- y*(1+m) for vertices with z < z_cut
        """
        pts = points0.copy()
        idx = hull_verts
        mask = pts[2, idx] < z_cut
        idx2 = idx[mask]
        pts[1, idx2] *= (1.0 + m) # change the hull's beam (witdth) # works for complex m
        #wihtout introducing any geometry machinery
        return pts
    
    
    def apply_beam_scale_smooth(points0, hull_verts, m, z0=0.0, band=0.02):
        """
        Smoothly transitions from full scaling below z0-band to no scaling above z0+band.
        band is in same units as z (your length scaling).
        """
        pts = points0.copy()
        idx = hull_verts
    
        z = pts[2, idx]
        # smoothstep t in [0,1] where t=0 below z0-band, t=1 above z0+band
        t = (z - (z0 - band)) / (2*band)
        t = np.clip(t, 0.0, 1.0)
        t = t*t*(3 - 2*t)  # smoothstep
    
        scale = 1.0 + m*(1.0 - t)  # full scaling below, none above
        pts[1, idx] *= scale
        return pts
    
    
    @staticmethod
    def assemble_from_points(points: np.ndarray,
                             pf,
                             Fr: float,
                             params,
                             panel_geometry_all,
                             assemble_A_b_vel_nb):
        """
        Assemble A,b,vel (and return geometry caches) from raw points/panels.
        Returns:
          A: (N,N), b: (N,), vel: (N,N,3),
          vinf: (3,),
          center: (N,3), coordsys: (N,3,3), area: (N,)
        """
        geom = panel_geometry_all(points, pf.panels)
    
        vinf = np.array([Fr*np.sqrt(params.length*params.gravity), 0.0, 0.0], dtype=np.float64)
        gravity = float(params.gravity)
    
        center = geom.center.T.astype(np.float64)                      # (N,3)
        coordsys = np.moveaxis(geom.coordsys, 2, 0).astype(np.float64) # (N,3,3)
        cornerslocal = np.moveaxis(geom.cornerslocal, 2, 0).astype(np.float64)  # (N,2,4)
        area = geom.area.astype(np.float64)
    
        A, b, vel = assemble_A_b_vel_nb(center, coordsys, cornerslocal, area,
                                        pf.npanels, pf.nfspanels, pf.deltax,
                                        vinf[0], gravity)
    
        return A, b, vel, vinf, center, coordsys, area
    
    
    @staticmethod
    def check_shape_grad_beam_scale(
        pf,
        points0: np.ndarray,
        Fr: float,
        params,
        panel_geometry_all,
        assemble_A_b_vel_nb,
        Objectives, # dependency injection (why do it this way vs import and call?)
        Validate_dj_dsigma,
        eps_m: float = 1e-5,
        m0: float = 0.0,
        z_cut: float = 0.0):
        """
        Shape-gradient check for one parameter m (beam scaling).
        Objective is J = -Fx.
    
        Compares:
          FD full: (J(m+e)-J(m-e))/(2e)
          Adjoint: lam^T (db/dm - (dA/dm) sigma)
    
        Notes:
          - This uses FD for dA/dm and db/dm ONLY as a validation harness.
          - In production, you'll replace these with analytic derivatives.
        """
        
        hull_verts = Validate_shape_gradient.hull_vertex_indices(pf.panels, pf.npanels)
    
        # ---------------- baseline ----------------
        A, b, vel, vinf, center, coordsys, area = Validate_shape_gradient.assemble_from_points(
            points0, pf, Fr, params, panel_geometry_all, assemble_A_b_vel_nb
        )
    
        # baseline sigma from A0 sigma = b0
        lu, piv = lu_factor(A)
        sigma = lu_solve((lu, piv), b)
        
    
        vtotal = -vinf[None, :] + np.einsum("ijm,j->im", vel, sigma)
        normals = coordsys[:, :, 2]
    
        rho = float(params.rho_water)
        gravity = float(params.gravity)
    
        postprocess = Validate_dj_dsigma.make_postprocess_JnegFx(
            vel, vinf, coordsys, area, center, pf.npanels, rho, gravity)
        
        J0 = float(postprocess(sigma))
        
        # SourceStrengthGradients = Objectives
        dJ_dsigma = Objectives.compute_dJ_dsigma_JnegFx(
            vel=vel, vtotal=vtotal, normals=normals,
            area=area, center=center, npanels=pf.npanels, rho_water=rho)
        
        lam = lu_solve((lu, piv), dJ_dsigma, trans=1)
    
        adj_relres = np.linalg.norm(A.T @ lam - dJ_dsigma) / (np.linalg.norm(dJ_dsigma) + 1e-30)
        print("\nAdjoint solve relres (J=-Fx):", adj_relres)
        
        
        # ---------------- plus ----------------
        pts_p = Validate_shape_gradient.apply_beam_scale(points0, hull_verts, m0 + eps_m, z_cut=z_cut)
        
        Ap, bp, velp, vinfp, centerp, coordsyp, areap = Validate_shape_gradient.assemble_from_points(
            pts_p, pf, Fr, params, panel_geometry_all, assemble_A_b_vel_nb)
        
        lup, pivp = lu_factor(Ap)
        sigma_plus = lu_solve((lup, pivp), bp)
        
        J_plus = float(Validate_dj_dsigma.make_postprocess_JnegFx(
            velp, vinfp, coordsyp, areap, centerp, pf.npanels, rho, gravity
        )(sigma_plus))
        
        
        # ---------------- minus ----------------
        pts_m = Validate_shape_gradient.apply_beam_scale(points0, hull_verts, m0 - eps_m, z_cut=z_cut)
        
        Am, bm, velm, vinfm, centerm, coordsym, aream = Validate_shape_gradient.assemble_from_points(
            pts_m, pf, Fr, params, panel_geometry_all, assemble_A_b_vel_nb)
        
        lum, pivm = lu_factor(Am)
        sigma_minus = lu_solve((lum, pivm), bm)
        
        J_minus = float(Validate_dj_dsigma.make_postprocess_JnegFx(
            velm, vinfm, coordsym, aream, centerm, pf.npanels, rho, gravity
        )(sigma_minus))
        
        
        # --- Explicit term: geometry changes, sigma held fixed (sigma0) ---
        post_p = Validate_dj_dsigma.make_postprocess_JnegFx(
            velp, vinfp, coordsyp, areap, centerp, pf.npanels, rho, gravity)
        
        post_m = Validate_dj_dsigma.make_postprocess_JnegFx(
            velm, vinfm, coordsym, aream, centerm, pf.npanels, rho, gravity)

        J_plus_fix  = float(post_p(sigma))
        J_minus_fix = float(post_m(sigma))
        
        
    
        # FD gradient of full objective dJ / dm
        dJ_dm_fd = (J_plus - J_minus) / (2.0 * eps_m)
        
        
        # FD gradients of implicit (dJ / d sigma)(d sigma / dm)
        dJ_dm_explicit_fd = (J_plus_fix - J_minus_fix) / (2.0 * eps_m)
        
    
        # Operator FD for adjoint formula
        dA_dm = (Ap - Am) / (2.0 * eps_m)
        db_dm = (bp - bm) / (2.0 * eps_m)
        
        
        # --- Implicit (adjoint) term through sigma via the adjoint idenity A,b ---
        # i.e. this is (dJ/dsigma)(dsigma/dm) from 
        dJ_dm_adj = float(lam @ (db_dm - dA_dm @ sigma))
        
        
        # --- Total predicted gradient = explicit + implicit ---
        dJ_dm_pred = dJ_dm_explicit_fd + dJ_dm_adj
    
        # print("\nBeam scaling shape-gradient check (objective J=-Fx)")
        # print("  m0 =", m0, " eps_m =", eps_m, " z_cut =", z_cut)
        # print("  J0 =", J0)
        # print("  FD dJ/dm      =", dJ_dm_fd)
        # print("  adjoint dJ/dm =", dJ_dm_adj)
        # print("  abs err       =", abs(dJ_dm_fd - dJ_dm_adj))
        # print("  rel err       =", abs(dJ_dm_fd - dJ_dm_adj) / (abs(dJ_dm_fd) + 1e-30))
        
        
        print("\nBeam scaling shape-gradient check (objective J=-Fx)")
        print("  m0 =", m0, " eps_m =", eps_m, " z_cut =", z_cut)
        print("  J0 =", J0)
        print("  FD total dJ/dm      =", dJ_dm_fd)
        print("  FD explicit (σ fixed)=", dJ_dm_explicit_fd)
        print("  adjoint implicit     =", dJ_dm_adj)
        print("  pred (explicit+adj)  =", dJ_dm_pred)
        print("  abs err              =", abs(dJ_dm_fd - dJ_dm_pred))
        print("  rel err              =", abs(dJ_dm_fd - dJ_dm_pred) / (abs(dJ_dm_fd) + 1e-30))
    
    
    ##########################
    
    
    
    # @staticmethod
    # def check_shape_grad_beam_scale(pf, points0, Fr, params, numba_assemble,
    #                                 eps_m=1e-5, m0=0.0):
    #     """
    #     Compares:
    #       FD: (J(m+e)-J(m-e))/(2e)
    #       Adjoint: lam^T (b_m - A_m sigma)
    
    #     Uses J = -Fx
    #     """
    #     hull_verts = hull_vertex_indices(pf.panels, pf.npanels)
    
    #     # --- baseline assemble/solve ---
    #     A, b, vel, vinf, center, coordsys, area = assemble_from_points(points0, pf, Fr, params, numba_assemble)
    #     lu, piv = lu_factor(A)
    #     sigma = lu_solve((lu, piv), b)
    
    #     J0, vtotal0 = eval_JnegFx(vel, vinf, coordsys, area, center, pf.npanels, params.rho_water, params.gravity, sigma)
    
    #     dJ_dsigma = Validate_dj_dsigma.compute_dJ_dsigma_JnegFx(
    #         vel=vel, vtotal=vtotal0, normals=coordsys[:, :, 2], area=area, center=center,
    #         npanels=pf.npanels, rho_water=params.rho_water
    #     )
    
    #     lam = lu_solve((lu, piv), dJ_dsigma, trans=1)
    #     adj_relres = np.linalg.norm(A.T @ lam - dJ_dsigma) / (np.linalg.norm(dJ_dsigma) + 1e-30)
    #     print("adjoint solve relres:", adj_relres)
    
    #     # --- assemble at m+eps and m-eps ---
    #     pts_p = apply_beam_scale(points0, hull_verts, m0 + eps_m)
    #     Ap, bp, velp, vinfp, centerp, coordsyp, areap = assemble_from_points(pts_p, pf, Fr, params, numba_assemble)
    #     lup, pivp = lu_factor(Ap)
    #     sigmap = lu_solve((lup, pivp), bp)
    #     Jp, _ = eval_JnegFx(velp, vinfp, coordsyp, areap, centerp, pf.npanels, params.rho_water, params.gravity, sigmap)
    
    #     pts_m = apply_beam_scale(points0, hull_verts, m0 - eps_m)
    #     Am, bm, velm, vinfm, centerm, coordsym, aream = assemble_from_points(pts_m, pf, Fr, params, numba_assemble)
    #     lum, pivm = lu_factor(Am)
    #     sigmam = lu_solve((lum, pivm), bm)
    #     Jm, _ = eval_JnegFx(velm, vinfm, coordsym, aream, centerm, pf.npanels, params.rho_water, params.gravity, sigmam)
    
    #     # FD gradient of full objective
    #     dJ_dm_fd = (Jp - Jm) / (2.0 * eps_m)
    
    #     # operator FD for adjoint gradient (using baseline sigma, lam)
    #     dA_dm = (Ap - Am) / (2.0 * eps_m)
    #     db_dm = (bp - bm) / (2.0 * eps_m)
    #     dJ_dm_adj = float(lam @ (db_dm - dA_dm @ sigma))
    
    #     print("J0 =", J0)
    #     print("dJ/dm FD     =", dJ_dm_fd)
    #     print("dJ/dm adjoint=", dJ_dm_adj)
    #     print("abs err      =", abs(dJ_dm_fd - dJ_dm_adj))
    #     print("rel err      =", abs(dJ_dm_fd - dJ_dm_adj) / (abs(dJ_dm_fd) + 1e-30))