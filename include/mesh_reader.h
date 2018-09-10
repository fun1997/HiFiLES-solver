#pragma once
#include <iostream>
#include <fstream>

#include "hf_array.h"
#include "error.h"
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