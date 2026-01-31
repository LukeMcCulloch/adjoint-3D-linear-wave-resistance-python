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



from .numba_kernels import assemble_A_b_vel_nb
from .objectives import SourceStrengthGradients, ShapeGradients
from .gradient_validators import Validate_dj_dsigma, Validate_shape_gradient
#from .gradient_validators import Objectives          # wherever you put compute_dJ_dsigma_JnegFx
#from .validate_dj_dsigma import Validate_dj_dsigma
from .solver import FlowParams              # or your params class
