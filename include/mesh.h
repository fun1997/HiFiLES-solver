#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include "global.h"
#include "hf_array.h"
#include "error.h"

using namespace std;

class mesh
{
public:
  friend class mesh_reader;

  // default constructor/destructor
  mesh();

  ~mesh();

  // #### APIs ####

  //get functions
  int get_num_cells();
  int get_num_cells(int in_type);

  //get max number of shape point of the specific type of element
  int get_max_n_spts(int in_type);

  //tool functions

#ifdef _MPI
  //repartition mesh based on the number of processors
  void repartition_mesh(int nproc, int rank);
#endif // _MPI

  //method to fill iv2ivg and modify c2v using local vertex indices
  void create_iv2ivg(void);

  //set up vertex connectivity
  void set_vertex_connectivity(void);

  //set up face connectivity
  void set_face_connectivity(void);

  /*------------------data---------------------*/
  // statistics
  int num_cells;        //defined in mesh reading, number of local cells
  int num_verts;        //defined in mesh reading, number of local vertics
  //int num_edges;        //defined in CompConnectivity
  int num_inters;       //defined in CompConnectivity
  int num_cells_global; //total number of cells
  int num_verts_global; //total number of vertics
  int n_bdy;            //number of boundary groups
  int n_dims;           //number of dimensions of the mesh
  int n_unmatched_inters;
  //cell
  hf_array<int> c2v;    //ID of vertices making up each cell.
  hf_array<int> c2n_v;  //Number of vertices in each cell.
  hf_array<int> ctype;  //Cell type.
  hf_array<int> ic2icg; //index of cell to index of cell globally.
  hf_array<int> c2f;    //cell to face
  //vertices
  hf_array<double> xv;       //Array of physical vertex locations (x,y,z).
  hf_array<int> iv2ivg;      //Index of vertex on processor to index of vertex globally.
  hf_array<vector<int>> v2c; //vertex to cell vector array
  hf_array<int> v2n_c;       //vertex to number of cell
  //faces
  hf_array<int> f2c;     //face to cell
  hf_array<int> f2v;     //face to vertex
  hf_array<int> f2nv;    //face to number of vertex
  hf_array<int> f2loc_f; //face to local face id
  hf_array<int> num_f_per_c;//number of face per cell
  hf_array<int> rot_tag;//rotation tag for coupling face
  hf_array<int> unmatched_inters;
  //boundary conditions
  hf_array<int> bc_id; //cell->face with value of index in run_input.bc_list 

private:
//get the corner vertex consecutive order
int get_corner_vert_in_order(const int &in_ic, const int &in_vert);

//get corner vertex list on the face
int get_corner_vlist_face(const int &in_ic, const int &in_face, hf_array<int> &out_vlist);

//compare 2 faces, return 1 if coincide,0 if not
int compare_faces(hf_array<int>& vlist1, hf_array<int>& vlist2, int &rtag);

//compare 2 boundary faces for gmsh
int compare_faces_boundary(hf_array<int> &vlist1, hf_array<int> &vlist2);


  

};