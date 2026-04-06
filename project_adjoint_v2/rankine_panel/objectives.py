# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 11:45:52 2026

@author: tluke

NOTE: these are acutally gradients of the objective 

And a stubbed out class for objectives

We want to minimize wave resistance by changing the bow geometry

so for this first pass 

J = Fx ~ Cw 

m = a bow shape function

"""


import numpy as np




class Objectives(object):
    """
     J
    """
    

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
        
        (please review the definition of sigma (and Fx)

        With your force definition, for wetted hull panels:
          dJ/dv_i = d(-Fx)/dv = (-)-rho * area_i * n_ix * v_i
                              =     rho * area_i * n_ix * v_i
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
    
    
    
    def residual_apply(m, sigma_fixed):
        """
        Return r(m) = A(m) @ sigma_fixed - b(m)
        Must be complex-safe if using complex-step.
        """
        # 1) build geometry from m  (allow complex)
        # 2) assemble/apply A(m) to sigma_fixed (prefer matrix-free apply if you have it)
        # 3) assemble b(m)
        # 4) return A_sigma - b
        return #r
    
    def bm_minus_Am_sigma(m0, sigma_fixed, eps=1e-30):
        '''
        Parameters
        ----------
        m0 : TYPE
            DESCRIPTION.
        sigma_fixed : TYPE
            DESCRIPTION.
        eps : TYPE, optional
            DESCRIPTION. The default is 1e-30.

        Returns
        -------
        TYPE
            The gradient of the residual physics with respect to the geometry variables m
            
        No need for explicit 
        A_m storing of derivative matrices
        Only requires your geometry + influence evaluation to be complex-safe.
        Gives a very strong correctness oracle for whatever you later do with reverse-mode or hand-coded sensitivities.

        '''
        r = residual_apply(m0 + 1j*eps, sigma_fixed)
        rm = np.imag(r) / eps
        return -rm