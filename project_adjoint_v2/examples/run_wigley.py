# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 22:40:34 2026

@author: tluke
"""

# path modification to run the package code from here
import os
import sys
import shlex



# Add the project root (parent of this file's folder) to sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
    
# if not os.path.isabs(inputfile):
#     # First try relative to where the script lives
#     candidate = os.path.join(_THIS_DIR, inputfile)
#     if os.path.exists(candidate):
#         inputfile = candidate
#     # else: it will fall back to whatever the current working directory is
    
import numpy as np
from rankine_panel import read_panel_file, RankineWaveResistanceSolver, FlowParams
from rankine_panel.vtkio import write_vtp_polydata_cell_scalar
from rankine_panel.report import append_output_dat

#print("rankine_panel loaded from:", os.path.abspath(rankine_panel.__file__))


def usage():
    print("Usage:")
    print("  python examples/run_wigley.py <inputfile> <output_handle> <FroudeNumber>")
    print("Example:")
    print("  python examples/run_wigley.py fifi.dat jan2026out 0.3")



# def main():
    
#     if len(sys.argv) != 4:
#         usage()
#         sys.exit(1)
        
#         inputfile = sys.argv[1]
#         out_handle = sys.argv[2]
#         Fr = float(sys.argv[3])
    
#         pf = read_panel_file(inputfile)
        
#     else:
#         inputfile = "fifi.dat"
#         out_handle = "LinearFreeSurfaceCodeV1_output"
#         Fr = 0.3#0.35  # example
#         pf = read_panel_file("fifi.dat")

def main(arg_string: str | None = None):
    # If called with a single string, treat it like a shell command line
    if arg_string is not None:
        argv = ["run_wigley.py"] + shlex.split(arg_string)
    else:
        argv = sys.argv

    if len(argv) != 4:
        print("Usage: run_wigley.py <inputfile> <output_handle> <Fr>")
        return
    
    #inputfile = os.path.join(_THIS_DIR, inputfile)  # if inputfile is just "fifi.dat"
    
    inputfile = argv[1]
    out_handle = argv[2]
    Fr = float(argv[3])
   
    pf = read_panel_file(inputfile)

    # Set your physical constants here.
    # In my Fortran: vinf = Fr*sqrt(length*gravity)
    #params = FlowParams(gravity=9.81, length=1.0)
    # Match your Fortran constants defaults
    params = FlowParams(gravity=9.80665, length=1.0)

    solver = RankineWaveResistanceSolver(params)
    geom = solver.build_geometry(pf)

    res = solver.solve(pf, geom, Fr)
    
    # ---- Write text output (append), like Fortran ----
    out_dat = f"{out_handle}.dat"

    print("title:", pf.title)
    print("N panels:", pf.N, " hull:", pf.npanels, " fs:", pf.nfspanels)
    print("deltax:", pf.deltax)
    print("cw:", res.cw)
    print("force:", res.force)
    print("max|vn| (hull):", np.max(np.abs(res.vn[:pf.npanels])))
    print("sourcesink sum:", float(np.sum(res.sigma * geom.area)))
    print("Writing VTK")
    
    
    # Build profilewave (2,128) similar to the Fortran extraction:
    # It used panels [npanels .. npanels+127] (1-based) => in Python: [npanels-1 .. npanels+126]
    # BUT the Fortran code does:
    #   do i=npanels,npanels+128
    # which is 129 samples (inclusive) but profilewave is (2,128). That's a mismatch in the Fortran snippet.  Oops
    # We do 128 samples, starting at the first free-surface panel.
    n = 128
    start = pf.npanels
    end = min(pf.npanels + n, pf.N)

    center = geom.center  # (3,N)
    profilewave = np.zeros((2, n), dtype=float)
    xs = center[0, start:end] + pf.deltax
    zs = res.zeta[start:end]
    profilewave[0, :end-start] = xs
    profilewave[1, :end-start] = zs

    append_output_dat(out_dat, res.iter_count, res.sourcesink, res.cw, profilewave)

    
    
    # # wave.vtp: all panels, cell scalars zeta and sigma
    # write_vtp_polydata_cell_scalar(
    #     "wave.vtp",
    #     pf.points,
    #     pf.panels,          # (4, N) 0-based already from our reader
    #     scalar1=res.zeta,
    #     scalar2=res.sigma,
    #     namescalar1="zeta",
    #     namescalar2="sigma",
    # )
    
    # # pressure.vtp: hull panels only, cell scalar cp
    # write_vtp_polydata_cell_scalar(
    #     "pressure.vtp",
    #     pf.points,
    #     pf.panels[:, :pf.npanels],
    #     scalar1=res.cp[:pf.npanels],
    #     namescalar1="cp",
    # )
    
    # # VelComponents.vtp: all panels, cell scalars vn, vt1, vt2
    # write_vtp_polydata_cell_scalar(
    #     "VelComponents.vtp",
    #     pf.points,
    #     pf.panels,
    #     scalar1=res.vn,
    #     scalar2=res.vt1,
    #     scalar3=res.vt2,
    #     namescalar1="vn",
    #     namescalar2="vt1",
    #     namescalar3="vt2",
        
    # ---- Write VTP outputs ----
    # wave.vtp: all panels, cell scalars zeta and sigma
    write_vtp_polydata_cell_scalar(
        f"{out_handle}_wave.vtp",
        pf.points,
        pf.panels,
        scalar1=res.zeta,
        scalar2=res.sigma,
        namescalar1="zeta",
        namescalar2="sigma",
    )

    # pressure.vtp: hull panels only, cell scalar cp
    write_vtp_polydata_cell_scalar(
        f"{out_handle}_pressure.vtp",
        pf.points,
        pf.panels[:, :pf.npanels],
        scalar1=res.cp[:pf.npanels],
        namescalar1="cp",
    )

    # VelComponents.vtp: all panels, cell scalars vn, vt1, vt2
    write_vtp_polydata_cell_scalar(
        f"{out_handle}_VelComponents.vtp",
        pf.points,
        pf.panels,
        scalar1=res.vn,
        scalar2=res.vt1,
        scalar3=res.vt2,
        namescalar1="vn",
        namescalar2="vt1",
        namescalar3="vt2",
    )
    
    
    # ---- Console summary ----
    print("title:", pf.title)
    print("input:", inputfile)
    print("out handle:", out_handle)
    print("Fr:", Fr)
    print("deltax:", pf.deltax)
    print("cw:", res.cw)
    print("force:", res.force)
    print("sourcesink:", res.sourcesink)
    print("max|vn| hull:", float(np.max(np.abs(res.vn[:pf.npanels]))))
    print("wrote:", out_dat)
    print("wrote:", f"{out_handle}_wave.vtp")
    print("wrote:", f"{out_handle}_pressure.vtp")
    print("wrote:", f"{out_handle}_VelComponents.vtp")

if __name__ == "__main__":
    #main() # allow whatever input from a shell.
    main("fifi.dat jan2026out 0.3") # spyder command 
