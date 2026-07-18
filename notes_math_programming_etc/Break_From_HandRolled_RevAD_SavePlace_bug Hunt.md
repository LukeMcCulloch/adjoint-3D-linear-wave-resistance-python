

# Where we left off: 
* assemble_dphi_dgeom_nb (numba) faithfully reproduces the hand-derived composition (verified exhaustively 
- forward values, individual kernels, per-panel accumulation for panel 5, panel_geometry_one_backward's response to real seeds 
- all confirmed correct against independent checks). 
- But the aggregate gradient at full mesh scale is ~93× too large at every sampled vertex, while FD of the same composed formula (bypassing production entirely) gives the right answer. 
- Isolating further: 
    - panel-as-row contributions look roughly right; 
    - panel-as-col contributions (where one panel is the source for up to 3,360 different field points) are where the error concentrates 
        - something isn't canceling that should. That's a real, unresolved bug in the underlying gradient composition, not the numba port. 

This is sitting in project_adjoint_v4/rankine_panel/numba_adjoint_kernels.py and gradient_checks/validate_full_mesh_numba.py whenever we come back to it.