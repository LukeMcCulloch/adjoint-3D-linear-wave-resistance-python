"""
package Rankine Source and generalized linearized free surface solver for wave resistance around a floating body at constant speed (Re)

todo: adding a pyproject.toml
    and have users do pip install -e .
    After this local install into python, rankine_panel is importable everywhere
"""
from .io import read_panel_file
from .geometry import panel_geometry_all
from .solver import RankineWaveResistanceSolver, FlowParams
from .vtkio import write_vtp_polydata_cell_scalar, write_vtp_polydata_cell_vector
from .report import append_output_dat