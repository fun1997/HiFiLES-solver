/*!
 * \file mesh_reader.h
 * \author - Original code: HiFiLES Aerospace Computing Laboratory (ACL)
 *                                Aero/Astro Department. Stanford University.
 *         - Current development: Weiqi Shen
 *                                University of Florida
 *
 * High Fidelity Large Eddy Simulation (HiFiLES) Code.
 *
 * HiFiLES is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * HiFiLES is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with HiFiLES.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once
#include <iostream>
#include <fstream>

#include "global.h"
#include "mesh.h"
#include "solution.h"

class mesh_reader
{
public:
  // #### constructors ####

  // default constructor/destructor
  mesh_reader(string in_fileName, mesh *in_mesh);

  ~mesh_reader();

  //partially read mesh connectivity based on total number of processors
  void partial_read_connectivity(int kstart, int in_num_cells);
  //read vertices
  void read_vertices(void);
  //read boundary
  void read_boundary(void);

private:
  string fname;       //mesh file name
  ifstream mesh_file; //input file stream
  int mesh_format;
  mesh *mesh_ptr; //pointer to mesh object

  /* -------------------methods----------------------------*/
  //read header and store it in mesh object pointed by mesh_ptr
  void read_header(void);
  //mesh format specific header readers
  void read_header_gambit(void);
  void read_header_gmsh(void);
  //mesh format specific partial connectivity readers
  void partial_read_connectivity_gambit(int kstart, int in_num_cells);
  void partial_read_connectivity_gmsh(int kstart, int in_num_cells);
  //mesh format specific vertices readers
  void read_vertices_gambit(void);
  void read_vertices_gmsh(void);
  //mesh format specific boundary condition reader
  void read_boundary_gambit();
  void read_boundary_gmsh();

};