# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 11:45:52 2026

@author: tluke
"""


import numpy as np


def compute_dJ_dsigma_cw(
    vel, vtotal, normals, area, center,
    npanels, vinf, rho_water, rho_ref
):
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
    dJ_dsigma = np.einsum("ik,ijk->j", dJ_dv, vel)
    return dJ_dsigma