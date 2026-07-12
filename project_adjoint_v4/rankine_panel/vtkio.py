# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 22:55:26 2026

@author: tluke
"""

# rankine_panel/vtkio.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Optional, Sequence, Union, TextIO

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[int]]


def _write_tag(f: TextIO, tag: str) -> None:
    f.write(tag)
    if not tag.endswith("\n"):
        f.write("\n")


def _data_array_int(f: TextIO,
                    data: ArrayLike,
                    typestr: str,
                    namestr: str,
                    tag: Optional[str] = None) -> None:
    arr = np.asarray(data, dtype=np.int32).ravel()
    if tag is not None and len(tag) > 0:
        _write_tag(f, f'<DataArray {typestr} {namestr} {tag} format="ascii">')
    else:
        _write_tag(f, f'<DataArray {typestr} {namestr} format="ascii">')

    # Match Fortran-ish spacing (not important for VTK, but keeps it readable)
    # Write many ints per line
    per_line = 16
    for i in range(0, arr.size, per_line):
        chunk = arr[i:i+per_line]
        f.write(" ".join(str(int(x)) for x in chunk))
        f.write("\n")

    _write_tag(f, "</DataArray>")


def _data_array_float(f: TextIO,
                      data: ArrayLike,
                      typestr: str,
                      namestr: str,
                      tag: Optional[str] = None) -> None:
    arr = np.asarray(data, dtype=np.float64).ravel()
    if tag is not None and len(tag) > 0:
        _write_tag(f, f'<DataArray {typestr} {namestr} {tag} format="ascii">')
    else:
        _write_tag(f, f'<DataArray {typestr} {namestr} format="ascii">')

    per_line = 8
    for i in range(0, arr.size, per_line):
        chunk = arr[i:i+per_line]
        # Fortran used G16.6; this is close enough and VTK doesn't care
        f.write(" ".join(f"{float(x):.6g}" for x in chunk))
        f.write("\n")

    _write_tag(f, "</DataArray>")


def _vtkxml_open(filename: str, vtk_type: str) -> TextIO:
    f = open(filename, "w", newline="\n")
    _write_tag(f, f'<VTKFile type="{vtk_type}">')
    return f


def _vtkxml_close(f: TextIO) -> None:
    _write_tag(f, "</VTKFile>")
    f.close()


def _flatten_points(points3xn: np.ndarray) -> np.ndarray:
    """
    points3xn: (3, npoints) like your Fortran points array.
    VTK expects x1 y1 z1 x2 y2 z2 ... => points.T.ravel() (C order).
    """
    pts = np.asarray(points3xn, dtype=np.float64)
    assert pts.ndim == 2 and pts.shape[0] == 3
    return pts.T.ravel()


def _flatten_polys_quads(panels4xN: np.ndarray) -> np.ndarray:
    """
    panels4xN: (4, npolys) with 0-based indices.
    Fortran wrote RESHAPE((panels-1), (/4*npolys/)) which yields:
      [p1c1 p1c2 p1c3 p1c4  p2c1 ...]
    That matches panels.T.ravel() in Python.
    """
    polys = np.asarray(panels4xN, dtype=np.int32)
    assert polys.ndim == 2 and polys.shape[0] == 4
    return polys.T.ravel()


def write_vtp_polydata_cell_scalar(filename: str,
                                   points3xn: np.ndarray,
                                   panels4xN: np.ndarray,
                                   scalar1: Optional[np.ndarray] = None,
                                   scalar2: Optional[np.ndarray] = None,
                                   scalar3: Optional[np.ndarray] = None,
                                   namescalar1: str = "scalar1",
                                   namescalar2: str = "scalar2",
                                   namescalar3: str = "scalar3",
                                   nverts: int = 0,
                                   nlines: int = 0,
                                   nstrips: int = 0) -> None:
    """
    Python port of VtkXmlPolyDataCellScalar (PolyData, quads, CellData scalars).
    """
    npoints = int(points3xn.shape[1])
    npolys = int(panels4xN.shape[1])

    pts = _flatten_points(points3xn)
    polys = _flatten_polys_quads(panels4xN)

    f = _vtkxml_open(filename, "PolyData")
    try:
        _write_tag(f, "<PolyData>")
        _write_tag(
            f,
            f'<Piece NumberOfPoints="{npoints}" NumberOfVerts="{nverts}" '
            f'NumberOfLines="{nlines}" NumberOfStrips="{nstrips}" NumberOfPolys="{npolys}">'
        )

        _write_tag(f, "<Points>")
        _data_array_float(f, pts, 'type="Float32"', 'NumberOfComponents="3"')
        _write_tag(f, "</Points>")

        _write_tag(f, "<Polys>")
        _data_array_int(f, polys, 'type="Int32"', 'Name="connectivity"')

        offsets = np.arange(1, npolys + 1, dtype=np.int32) * 4
        _data_array_int(f, offsets, 'type="Int32"', 'Name="offsets"')
        _write_tag(f, "</Polys>")

        _write_tag(f, "<CellData>")

        if scalar1 is not None:
            _data_array_float(f, np.asarray(scalar1, dtype=np.float64).ravel(),
                              'type="Float32"', f'Name="{namescalar1}"')
        if scalar2 is not None:
            _data_array_float(f, np.asarray(scalar2, dtype=np.float64).ravel(),
                              'type="Float32"', f'Name="{namescalar2}"')
        if scalar3 is not None:
            _data_array_float(f, np.asarray(scalar3, dtype=np.float64).ravel(),
                              'type="Float32"', f'Name="{namescalar3}"')

        _write_tag(f, "</CellData>")
        _write_tag(f, "</Piece>")
        _write_tag(f, "</PolyData>")
    finally:
        _vtkxml_close(f)


def write_vtp_polydata_cell_vector(filename: str,
                                   points3xn: np.ndarray,
                                   panels4xN: np.ndarray,
                                   vector1: Optional[np.ndarray] = None,
                                   vector2: Optional[np.ndarray] = None,
                                   vector3: Optional[np.ndarray] = None,
                                   namevector1: str = "vector1",
                                   namevector2: str = "vector2",
                                   namevector3: str = "vector3",
                                   nverts: int = 0,
                                   nlines: int = 0,
                                   nstrips: int = 0) -> None:
    """
    Port of VtkXmlPolyDataCellVector (PolyData, quads, CellData vectors).
    vector* should be shape (3, npolys) or (npolys, 3).
    """
    npoints = int(points3xn.shape[1])
    npolys = int(panels4xN.shape[1])

    pts = _flatten_points(points3xn)
    polys = _flatten_polys_quads(panels4xN)

    def _flatten_vec(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        if v.ndim != 2:
            raise ValueError("vector must be 2D")
        if v.shape == (3, npolys):
            return v.T.ravel()
        if v.shape == (npolys, 3):
            return v.ravel()
        raise ValueError(f"vector shape must be (3,{npolys}) or ({npolys},3), got {v.shape}")

    f = _vtkxml_open(filename, "PolyData")
    try:
        _write_tag(f, "<PolyData>")
        _write_tag(
            f,
            f'<Piece NumberOfPoints="{npoints}" NumberOfVerts="{nverts}" '
            f'NumberOfLines="{nlines}" NumberOfStrips="{nstrips}" NumberOfPolys="{npolys}">'
        )

        _write_tag(f, "<Points>")
        _data_array_float(f, pts, 'type="Float32"', 'NumberOfComponents="3"')
        _write_tag(f, "</Points>")

        _write_tag(f, "<Polys>")
        _data_array_int(f, polys, 'type="Int32"', 'Name="connectivity"')
        offsets = np.arange(1, npolys + 1, dtype=np.int32) * 4
        _data_array_int(f, offsets, 'type="Int32"', 'Name="offsets"')
        _write_tag(f, "</Polys>")

        _write_tag(f, "<CellData>")

        if vector1 is not None:
            _data_array_float(f, _flatten_vec(vector1),
                              'type="Float32"', f'Name="{namevector1}"',
                              'NumberOfComponents="3"')
        if vector2 is not None:
            _data_array_float(f, _flatten_vec(vector2),
                              'type="Float32"', f'Name="{namevector2}"',
                              'NumberOfComponents="3"')
        if vector3 is not None:
            _data_array_float(f, _flatten_vec(vector3),
                              'type="Float32"', f'Name="{namevector3}"',
                              'NumberOfComponents="3"')

        _write_tag(f, "</CellData>")
        _write_tag(f, "</Piece>")
        _write_tag(f, "</PolyData>")
    finally:
        _vtkxml_close(f)
