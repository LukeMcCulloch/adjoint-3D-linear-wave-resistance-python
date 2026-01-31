# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 14:08:17 2026

@author: Luke McCulloch
"""

import numpy as np

class Validate_dj_dsigma(object):
    
    
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

        With your force definition, for wetted hull panels:
          dJ/dv_i = -rho * area_i * n_ix * v_i
        and dJ/dsigma_j = sum_i (dJ/dv_i)·vel[i,j]
        """
        N = vel.shape[0]
        wetted = center[:npanels, 2] < 0.0

        dJ_dv = np.zeros((N, 3), dtype=np.float64)

        scale = np.zeros(npanels, dtype=np.float64)
        scale[wetted] = -rho_water * area[:npanels][wetted] * normals[:npanels, 0][wetted]

        dJ_dv[:npanels, :] = scale[:, None] * vtotal[:npanels, :]

        dJ_dsigma = np.einsum("ik,ijk->j", dJ_dv, vel)
        return dJ_dsigma

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
        
        normals = coordsys[:, :, 2]
        U2 = float(vinf[0] * vinf[0])

        def postprocess(sigma: np.ndarray) -> float:
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
            
            
            
            