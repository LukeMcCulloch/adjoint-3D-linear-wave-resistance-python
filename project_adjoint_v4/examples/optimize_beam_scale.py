# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 2026

@author: Luke McCulloch
    tlukemcculloch@gmail.com

Adjoint-driven shape optimization demo.

This is the piece that actually *uses* the adjoint machinery validated in
gradient_checks/check_shape_gradient.py: at every iteration we

  1) assemble A(m), b(m)                              (geometry -> physics)
  2) solve the state:      A sigma = b                 (forward solve)
  3) evaluate J = -Fx and  dJ/dsigma                    (objective)
  4) solve the adjoint:    A^T lam = dJ/dsigma          (adjoint solve)
  5) form the shape gradient:
         dJ/dm = J_m|sigma + lam^T (db/dm - (dA/dm) sigma)
  6) take a gradient-descent step on the scalar shape parameter m
     (beam scaling: y <- y*(1+m) on submerged hull vertices)

and repeat until the gradient is small or we run out of iterations.

Usage:
  python examples/optimize_beam_scale.py fifi.dat optbeam 0.3 --max-iter 6 --step 0.05

Note: each gradient evaluation costs 3 full geometry+influence assemblies
(baseline, m+eps, m-eps), and each line-search trial costs 1 more. On the
fifi.dat Wigley mesh (N=3360) one assembly takes on the order of ten seconds,
so a handful of iterations takes several minutes. That's expected.

TODO(AD): step 5's dA/dm, db/dm (inside
gradient_validators.py::compute_beam_scale_gradient) are finite-differenced
today. That's a deliberate stepping stone, not the destination -- see the
TODO(AD) notes in gradient_validators.py and numba_kernels.py. It's fine for
a single scalar shape parameter, but it must be replaced with analytic/AD
derivatives before this scales to many-DOF (e.g. per-vertex) shape
variables.
"""

import os
import sys
import shlex
import argparse

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from rankine_panel import (
    read_panel_file,
    RankineWaveResistanceSolver,
    FlowParams,
    panel_geometry_all,
    assemble_A_b_vel_nb,
    SourceStrengthGradients,
)
from rankine_panel.vtkio import write_vtp_polydata_cell_scalar
from rankine_panel.gradient_validators import Validate_shape_gradient, Validate_dj_dsigma


def eval_J_only(pf, points, Fr, params):
    """
    Cheap forward-only evaluation of J=-Fx at a given geometry (no adjoint
    solve). Used for line-search trials, where we only need the objective
    value, not its gradient.
    """
    A, b, vel, vinf, center, coordsys, area = Validate_shape_gradient.assemble_from_points(
        points, pf, Fr, params, panel_geometry_all, assemble_A_b_vel_nb
    )
    from scipy.linalg import lu_factor, lu_solve
    lu, piv = lu_factor(A)
    sigma = lu_solve((lu, piv), b)

    postprocess = Validate_dj_dsigma.make_postprocess_JnegFx(
        vel, vinf, coordsys, area, center, pf.npanels,
        float(params.rho_water), float(params.gravity),
    )
    J = float(postprocess(sigma))

    U = float(vinf[0])
    U2 = U * U
    wetted = center[:pf.npanels, 2] < 0.0
    S = float(np.sum(area[:pf.npanels][wetted]))
    cw = J / (0.5 * float(params.rho_ref) * U2 * S)
    return J, cw


def optimize_beam_scale(
    pf, points0, Fr, params,
    max_iter=6, step_init=0.05, tol=1e-3,
    m_bounds=(-0.6, 3.0), z_cut=0.0, eps_m=1e-5,
    backtrack_factor=0.5, max_backtracks=4, armijo_c1=1e-4,
):
    """
    Steepest-descent-with-backtracking optimizer on the scalar beam-scale
    shape parameter m, driven entirely by the adjoint shape gradient.

    Returns (m_final, history) where history is a list of dicts, one per
    accepted iteration.
    """
    m = 0.0
    alpha = step_init
    history = []

    for it in range(max_iter):
        result = Validate_shape_gradient.compute_beam_scale_gradient(
            pf, points0, Fr, params, panel_geometry_all, assemble_A_b_vel_nb,
            SourceStrengthGradients, Validate_dj_dsigma,
            eps_m=eps_m, m0=m, z_cut=z_cut, include_fd_total=False,
        )
        J = result["J0"]
        dJdm = result["dJdm"]

        hull_verts = Validate_shape_gradient.hull_vertex_indices(pf.panels, pf.npanels)
        U = Fr * np.sqrt(params.length * params.gravity)
        wetted = result["center"][:pf.npanels, 2] < 0.0
        S = float(np.sum(result["area"][:pf.npanels][wetted]))
        cw = J / (0.5 * params.rho_ref * U * U * S)

        print(f"\n[iter {it}] m={m:+.6f}  J(-Fx)={J:.6e}  Cw={cw:.6e}  dJ/dm={dJdm:+.6e}")

        if abs(dJdm) < tol:
            print(f"  converged: |dJ/dm|={abs(dJdm):.3e} < tol={tol:.3e}")
            history.append(dict(iter=it, m=m, J=J, cw=cw, dJdm=dJdm, alpha=0.0, backtracks=0))
            break

        direction = -1.0 if dJdm > 0.0 else 1.0  # descent direction on m

        trial_alpha = alpha
        accepted = False
        backtracks = 0
        for bt in range(max_backtracks + 1):
            m_trial = float(np.clip(m + trial_alpha * direction, m_bounds[0], m_bounds[1]))
            pts_trial = Validate_shape_gradient.apply_beam_scale(points0, hull_verts, m_trial, z_cut=z_cut)
            J_trial, cw_trial = eval_J_only(pf, pts_trial, Fr, params)

            # Armijo sufficient-decrease: J_trial <= J - c1*trial_alpha*|dJdm|
            if J_trial <= J - armijo_c1 * trial_alpha * abs(dJdm):
                accepted = True
                break
            trial_alpha *= backtrack_factor
            backtracks += 1

        print(f"  step: m -> {m_trial:+.6f}  (alpha={trial_alpha:.4g}, backtracks={backtracks}, "
              f"{'accepted' if accepted else 'NOT accepted -- taking best trial anyway'})")

        history.append(dict(iter=it, m=m, J=J, cw=cw, dJdm=dJdm, alpha=trial_alpha, backtracks=backtracks))

        m = m_trial
        # grow the step slightly for next iteration if we didn't need to backtrack much
        alpha = min(trial_alpha * (1.5 if backtracks == 0 else 1.0), step_init * 4.0)

        if trial_alpha < 1e-8:
            print("  line search underflow, stopping.")
            break

    return m, history


def main(arg_string: str | None = None):
    if arg_string is not None:
        argv = shlex.split(arg_string)
    else:
        argv = sys.argv[1:]

    ap = argparse.ArgumentParser(description="Adjoint-driven beam-scale wave-resistance optimizer.")
    ap.add_argument("inputfile")
    ap.add_argument("out_handle")
    ap.add_argument("Fr", type=float)
    ap.add_argument("--max-iter", type=int, default=6)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args(argv)

    pf = read_panel_file(args.inputfile)
    params = FlowParams(gravity=9.80665, length=1.0)
    Fr = args.Fr

    print("title:", pf.title)
    print("N panels:", pf.N, " hull:", pf.npanels, " fs:", pf.nfspanels)
    print("Fr:", Fr)

    m_final, history = optimize_beam_scale(
        pf, pf.points, Fr, params,
        max_iter=args.max_iter, step_init=args.step, tol=args.tol,
    )

    if not history:
        print("No optimization steps were taken.")
        return

    # ---- Re-run the full forward solver at the optimized geometry and write VTPs ----
    hull_verts = Validate_shape_gradient.hull_vertex_indices(pf.panels, pf.npanels)
    points_opt = Validate_shape_gradient.apply_beam_scale(pf.points, hull_verts, m_final, z_cut=0.0)

    from rankine_panel.types import PanelFile
    pf_opt = PanelFile(
        title=pf.title + " (beam-scale optimized)",
        i=pf.i, j=pf.j,
        npanels=pf.npanels, nfspanels=pf.nfspanels, npoints=pf.npoints,
        deltax=pf.deltax, panels=pf.panels, points=points_opt,
    )

    solver = RankineWaveResistanceSolver(params)
    geom0 = solver.build_geometry(pf)
    geom_opt = solver.build_geometry(pf_opt)

    res0 = solver.solve(pf, geom0, Fr)
    res_opt = solver.solve(pf_opt, geom_opt, Fr)

    J0, Jf = -res0.force[0], -res_opt.force[0]
    cw0, cwf = res0.cw, res_opt.cw

    print("\n=== Optimization summary (fresh forward solves, ground truth) ===")
    print(f"  m: 0.0 -> {m_final:+.6f}")
    print(f"  J(-Fx): {J0:.6e} -> {Jf:.6e}  ({100.0*(Jf-J0)/abs(J0):+.2f}%)")
    print(f"  Cw:     {cw0:.6e} -> {cwf:.6e}  ({100.0*(cwf-cw0)/abs(cw0):+.2f}%)")

    write_vtp_polydata_cell_scalar(
        f"{args.out_handle}_baseline_wave.vtp",
        pf.points, pf.panels,
        scalar1=res0.zeta, scalar2=res0.sigma,
        namescalar1="zeta", namescalar2="sigma",
    )
    write_vtp_polydata_cell_scalar(
        f"{args.out_handle}_baseline_pressure.vtp",
        pf.points, pf.panels[:, :pf.npanels],
        scalar1=res0.cp[:pf.npanels], namescalar1="cp",
    )
    write_vtp_polydata_cell_scalar(
        f"{args.out_handle}_baseline_VelComponents.vtp",
        pf.points, pf.panels,
        scalar1=res0.vn, scalar2=res0.vt1, scalar3=res0.vt2,
        namescalar1="vn", namescalar2="vt1", namescalar3="vt2",
    )

    write_vtp_polydata_cell_scalar(
        f"{args.out_handle}_wave.vtp",
        points_opt, pf.panels,
        scalar1=res_opt.zeta, scalar2=res_opt.sigma,
        namescalar1="zeta", namescalar2="sigma",
    )
    write_vtp_polydata_cell_scalar(
        f"{args.out_handle}_pressure.vtp",
        points_opt, pf.panels[:, :pf.npanels],
        scalar1=res_opt.cp[:pf.npanels], namescalar1="cp",
    )
    write_vtp_polydata_cell_scalar(
        f"{args.out_handle}_VelComponents.vtp",
        points_opt, pf.panels,
        scalar1=res_opt.vn, scalar2=res_opt.vt1, scalar3=res_opt.vt2,
        namescalar1="vn", namescalar2="vt1", namescalar3="vt2",
    )

    print("\nwrote:", f"{args.out_handle}_baseline_wave.vtp")
    print("wrote:", f"{args.out_handle}_baseline_pressure.vtp")
    print("wrote:", f"{args.out_handle}_baseline_VelComponents.vtp")
    print("wrote:", f"{args.out_handle}_wave.vtp")
    print("wrote:", f"{args.out_handle}_pressure.vtp")
    print("wrote:", f"{args.out_handle}_VelComponents.vtp")


if __name__ == "__main__":
    main()
