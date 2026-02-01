# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 15:06:43 2026

@author: tluke
"""


from rankine_panel.io import read_panel_file
from rankine_panel.geometry import panel_geometry_all
from rankine_panel.numba_kernels import assemble_A_b_vel_nb
from rankine_panel.gradient_validators import Validate_shape_gradient, Validate_dj_dsigma
from rankine_panel.objectives import SourceStrengthGradients          # wherever you put compute_dJ_dsigma_JnegFx
from rankine_panel.solver import FlowParams              # or your params class



#if __name__ == '''__main__''':
if __name__ == "__main__":
    print("\nChecking the Shape Gradients\n")
    pf = read_panel_file("fifi.dat")
    params = FlowParams(gravity=9.80665, length=1.0)
    Fr = 0.3
    
    Validate_shape_gradient.check_shape_grad_beam_scale(
        pf=pf,
        points0=pf.points,
        Fr=Fr,
        params=params,
        panel_geometry_all=panel_geometry_all,
        assemble_A_b_vel_nb=assemble_A_b_vel_nb,
        Objectives=SourceStrengthGradients,
        Validate_dj_dsigma=Validate_dj_dsigma,
        eps_m=1e-5,
        m0=0.0,
        z_cut=0.0  # below z=0 only
    )