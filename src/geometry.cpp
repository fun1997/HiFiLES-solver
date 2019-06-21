/*!
 * \file geometry.cpp
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

#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <vector>

#include "../include/geometry.h"
#include "../include/mesh_reader.h"
#include "../include/solver.h"

#ifdef _GPU
#include "../include/util.h"
#endif

using namespace std;

void SetInput(struct solution *FlowSol)
{

  /*! Basic allocation using the input file. */
  FlowSol->rank = 0;
  FlowSol->nproc = 1;

#ifdef _MPI

  /*! Get MPI rank and nproc. */
  MPI_Comm_rank(MPI_COMM_WORLD, &FlowSol->rank);
  MPI_Comm_size(MPI_COMM_WORLD, &FlowSol->nproc);

#ifdef _GPU

  /*! Associate a GPU to each rank. */

  // Cluster:
#ifdef _YOSEMITESAM
  if (FlowSol->rank == 0)
  {
    cout << "setting CUDA devices on yosemitesam ..." << endl;
  }
  if ((FlowSol->rank % 2) == 0)
  {
    cudaSetDevice(0);
  }
  if ((FlowSol->rank % 2) == 1)
  {
    cudaSetDevice(1);
  }
#endif

  // Enrico:
#ifdef _ENRICO
  if (FlowSol->rank == 0)
  {
    cout << "setting CUDA devices on enrico ..." << endl;
  }
  if (FlowSol->rank == 0)
  {
    cudaSetDevice(2);
  }
  else if (FlowSol->rank == 1)
  {
    cudaSetDevice(0);
  }
  else if (FlowSol->rank == 2)
  {
    cudaSetDevice(3);
  }
#endif

#ifndef _ENRICO
#ifndef _YOSEMITESAM
  // NOTE: depening on system arcihtecture, this may not be the GPU device you want
  // i.e. one of the devices may be a (non-scientific-computing) graphics card
  cudaSetDevice(FlowSol->rank);
#endif
#endif

#endif

#endif
}

void GeoPreprocess(struct solution *FlowSol, mesh &mesh_data)
{

  /////////////////////////////////////////////////
  /// Read mesh and set up connectivity
  /////////////////////////////////////////////////
  ReadMesh(FlowSol, mesh_data);
  /////////////////////////////////////////////////
  /// Initializing Elements
  /////////////////////////////////////////////////

  // Count the number of elements of each type
  int num_tris = mesh_data.get_num_cells(TRI);
  int num_quads = mesh_data.get_num_cells(QUAD);
  int num_tets = mesh_data.get_num_cells(TET);
  int num_pris = mesh_data.get_num_cells(PRISM);
  int num_hexas = mesh_data.get_num_cells(HEX);

  // Error checking
  if (FlowSol->n_dims == 2 && (num_tets != 0 || num_pris != 0 || num_hexas != 0))
  {
    FatalError("Error in mesh reader, n_dims=2 and 3d elements exists");
  }
  if (FlowSol->n_dims == 3 && (num_tris != 0 || num_quads != 0))
  {
    FatalError("Error in mesh reader, n_dims=3 and 2d elements exists");
  }

  // For each element type, count the maximum number of shape points per element
  FlowSol->n_ele_types = 5;
  hf_array<int> max_n_spts(FlowSol->n_ele_types);

  max_n_spts(TRI) = mesh_data.get_max_n_spts(TRI);
  max_n_spts(QUAD) = mesh_data.get_max_n_spts(QUAD);
  max_n_spts(TET) = mesh_data.get_max_n_spts(TET);
  max_n_spts(PRISM) = mesh_data.get_max_n_spts(PRISM);
  max_n_spts(HEX) = mesh_data.get_max_n_spts(HEX);

  // Initialize the mesh_eles

  FlowSol->mesh_eles.setup(FlowSol->n_ele_types);

  FlowSol->mesh_eles(0) = &FlowSol->mesh_eles_tris;
  FlowSol->mesh_eles(1) = &FlowSol->mesh_eles_quads;
  FlowSol->mesh_eles(2) = &FlowSol->mesh_eles_tets;
  FlowSol->mesh_eles(3) = &FlowSol->mesh_eles_pris;
  FlowSol->mesh_eles(4) = &FlowSol->mesh_eles_hexas;

  for (int i = 0; i < FlowSol->n_ele_types; i++)
  {
    FlowSol->mesh_eles(i)->set_rank(FlowSol->rank);
  }

  if (FlowSol->rank == 0)
    cout << endl
         << "---------------- Flux Reconstruction Preprocessing ----------------" << endl;

  if (FlowSol->rank == 0)
    cout << "initializing elements" << endl;
  if (FlowSol->rank == 0)
    cout << "tris" << endl;
  FlowSol->mesh_eles_tris.setup(num_tris, max_n_spts(TRI));
  if (FlowSol->rank == 0)
    cout << "quads" << endl;
  FlowSol->mesh_eles_quads.setup(num_quads, max_n_spts(QUAD));
  if (FlowSol->rank == 0)
    cout << "tets" << endl;
  FlowSol->mesh_eles_tets.setup(num_tets, max_n_spts(TET));
  if (FlowSol->rank == 0)
    cout << "pris" << endl;
  FlowSol->mesh_eles_pris.setup(num_pris, max_n_spts(PRISM));
  if (FlowSol->rank == 0)
    cout << "hexas" << endl;
  FlowSol->mesh_eles_hexas.setup(num_hexas, max_n_spts(HEX));
  if (FlowSol->rank == 0)
    cout << "done initializing elements" << endl;

  // Set shape for each cell
  hf_array<int> local_c(mesh_data.num_cells); //index of overall element to index in one type of element

  int tris_count = 0;
  int quads_count = 0;
  int tets_count = 0;
  int pris_count = 0;
  int hexas_count = 0;

  hf_array<double> pos(FlowSol->n_dims);

  if (FlowSol->rank == 0)
    cout << "setting elements shape ... ";
  for (int i = 0; i < mesh_data.num_cells; i++)
  {
    if (mesh_data.ctype(i) == TRI) //tri
    {
      local_c(i) = tris_count;
      FlowSol->mesh_eles_tris.set_n_spts(tris_count, mesh_data.c2n_v(i));
      FlowSol->mesh_eles_tris.set_ele2global_ele(tris_count, mesh_data.ic2icg(i));

      for (int j = 0; j < mesh_data.c2n_v(i); j++)
      {
        pos(0) = mesh_data.xv(mesh_data.c2v(i, j), 0);
        pos(1) = mesh_data.xv(mesh_data.c2v(i, j), 1);
        FlowSol->mesh_eles_tris.set_shape_node(j, tris_count, pos);
      }

      for (int j = 0; j < 3; j++)
      {
        FlowSol->mesh_eles_tris.set_bcid(tris_count, j, mesh_data.bc_id(i, j));
      }

      tris_count++;
    }
    else if (mesh_data.ctype(i) == QUAD) // quad
    {
      local_c(i) = quads_count;
      FlowSol->mesh_eles_quads.set_n_spts(quads_count, mesh_data.c2n_v(i));
      FlowSol->mesh_eles_quads.set_ele2global_ele(quads_count, mesh_data.ic2icg(i));
      for (int j = 0; j < mesh_data.c2n_v(i); j++)
      {
        pos(0) = mesh_data.xv(mesh_data.c2v(i, j), 0);
        pos(1) = mesh_data.xv(mesh_data.c2v(i, j), 1);
        FlowSol->mesh_eles_quads.set_shape_node(j, quads_count, pos);
      }

      for (int j = 0; j < 4; j++)
      {
        FlowSol->mesh_eles_quads.set_bcid(quads_count, j, mesh_data.bc_id(i, j));
      }

      quads_count++;
    }
    else if (mesh_data.ctype(i) == TET) //tet
    {
      local_c(i) = tets_count;
      FlowSol->mesh_eles_tets.set_n_spts(tets_count, mesh_data.c2n_v(i));
      FlowSol->mesh_eles_tets.set_ele2global_ele(tets_count, mesh_data.ic2icg(i));
      for (int j = 0; j < mesh_data.c2n_v(i); j++)
      {
        pos(0) = mesh_data.xv(mesh_data.c2v(i, j), 0);
        pos(1) = mesh_data.xv(mesh_data.c2v(i, j), 1);
        pos(2) = mesh_data.xv(mesh_data.c2v(i, j), 2);
        FlowSol->mesh_eles_tets.set_shape_node(j, tets_count, pos);
      }

      for (int j = 0; j < 4; j++)
      {
        FlowSol->mesh_eles_tets.set_bcid(tets_count, j, mesh_data.bc_id(i, j));
      }

      tets_count++;
    }
    else if (mesh_data.ctype(i) == PRISM) //pri
    {
      local_c(i) = pris_count;
      FlowSol->mesh_eles_pris.set_n_spts(pris_count, mesh_data.c2n_v(i));
      FlowSol->mesh_eles_pris.set_ele2global_ele(pris_count, mesh_data.ic2icg(i));
      for (int j = 0; j < mesh_data.c2n_v(i); j++)
      {
        pos(0) = mesh_data.xv(mesh_data.c2v(i, j), 0);
        pos(1) = mesh_data.xv(mesh_data.c2v(i, j), 1);
        pos(2) = mesh_data.xv(mesh_data.c2v(i, j), 2);
        FlowSol->mesh_eles_pris.set_shape_node(j, pris_count, pos);
      }

      for (int j = 0; j < 5; j++)
      {
        FlowSol->mesh_eles_pris.set_bcid(pris_count, j, mesh_data.bc_id(i, j));
      }

      pris_count++;
    }
    else if (mesh_data.ctype(i) == HEX) //hex
    {
      local_c(i) = hexas_count;
      FlowSol->mesh_eles_hexas.set_n_spts(hexas_count, mesh_data.c2n_v(i));
      FlowSol->mesh_eles_hexas.set_ele2global_ele(hexas_count, mesh_data.ic2icg(i));
      for (int j = 0; j < mesh_data.c2n_v(i); j++)
      {
        pos(0) = mesh_data.xv(mesh_data.c2v(i, j), 0);
        pos(1) = mesh_data.xv(mesh_data.c2v(i, j), 1);
        pos(2) = mesh_data.xv(mesh_data.c2v(i, j), 2);
        FlowSol->mesh_eles_hexas.set_shape_node(j, hexas_count, pos);
      }

      for (int j = 0; j < 6; j++)
      {
        FlowSol->mesh_eles_hexas.set_bcid(hexas_count, j, mesh_data.bc_id(i, j));
      }

      hexas_count++;
    }
  }
  if (FlowSol->rank == 0)
    cout << "done." << endl;

  // set transforms
  if (FlowSol->rank == 0)
    cout << "setting element transforms ... " << endl;
  for (int i = 0; i < FlowSol->n_ele_types; i++)
    if (FlowSol->mesh_eles(i)->get_n_eles() != 0)
      FlowSol->mesh_eles(i)->set_transforms(); //static mesh transform
  if (FlowSol->rank == 0)
    cout << "done." << endl;
  // set on gpu (important - need to do this before we set connectivity, so that pointers point to GPU memory)
#ifdef _GPU
  for (int i = 0; i < FlowSol->n_ele_types; i++)
  {
    if (FlowSol->mesh_eles(i)->get_n_eles() != 0)
    {

      if (FlowSol->rank == 0)
        cout << "Moving eles to GPU ... " << endl;
      FlowSol->mesh_eles(i)->mv_all_cpu_gpu();
    }
  }
#endif

  // ------------------------------------
  // Initializing Interfaces
  // ------------------------------------

  int n_int_inters = 0;
  int n_bdy_inters = 0;
  int n_cyc_loc = 0; //number of local paired cyclic interface

  // -------------------------------------------------------
  // Split the cyclic faces as being internal or mpi faces
  // -------------------------------------------------------

  hf_array<double> loc_center_inter_0(FlowSol->n_dims), loc_center_inter_1(FlowSol->n_dims);
  hf_array<double> loc_vert_0(MAX_V_PER_F, FlowSol->n_dims), loc_vert_1(MAX_V_PER_F, FlowSol->n_dims);

  //initialize cyclic displacement
  hf_array<double> delta_cyclic(FlowSol->n_dims);
  delta_cyclic(0) = run_input.dx_cyclic;
  delta_cyclic(1) = run_input.dy_cyclic;
  if (FlowSol->n_dims == 3)
  {
    delta_cyclic(2) = run_input.dz_cyclic;
  }

  double tol = 1.e-6;
  int bcid_f, found, rtag;
  int ic_l, ic_r;

  for (int i = 0; i < mesh_data.n_unmatched_inters; i++)//for each unmatched interface
  {
    int i1 = mesh_data.unmatched_inters(i); //index of unmatched interface
    bcid_f = mesh_data.bc_id(mesh_data.f2c(i1, 0), mesh_data.f2loc_f(i1, 0));
    if (bcid_f == -1 || bcid_f == -3) //mpi internal face or coupled cyclic face, skip
      continue;

    if (run_input.bc_list(bcid_f).get_bc_flag() == CYCLIC)
    { //if is cyclic interface

      //calculate position of center of each cyclic interface
      loc_center_inter_0.initialize_to_zero();

      for (int k = 0; k < mesh_data.f2nv(i1); k++)
        for (int m = 0; m < FlowSol->n_dims; m++)
          loc_center_inter_0(m) += mesh_data.xv(mesh_data.f2v(i1, k), m) / (double)mesh_data.f2nv(i1);

      found = 0;
      for (int j = i+1; j < mesh_data.n_unmatched_inters; j++)//loop over all other uncounted and unmatched interfaces including other bdy and mpi internal face
      {

        int i2 = mesh_data.unmatched_inters(j); //index of unmatched interface

        if (bcid_f != mesh_data.bc_id(mesh_data.f2c(i2, 0), mesh_data.f2loc_f(i2, 0)) || mesh_data.f2nv(i1) != mesh_data.f2nv(i2))
          continue; //mpi face, coupled cyclic face or other boundary face or not the same kind of face

        loc_center_inter_1.initialize_to_zero();

        //calculate center of the unmatched interface
        for (int k = 0; k < mesh_data.f2nv(i2); k++)
          for (int m = 0; m < FlowSol->n_dims; m++)
            loc_center_inter_1(m) += mesh_data.xv(mesh_data.f2v(i2, k), m) / (double)mesh_data.f2nv(i2);

        if (check_cyclic(delta_cyclic, loc_center_inter_0, loc_center_inter_1, tol, FlowSol))//if matched
        {

          found = 1;
          mesh_data.f2c(i1, 1) = mesh_data.f2c(i2, 0); //couple up

          mesh_data.bc_id(mesh_data.f2c(i1, 0), mesh_data.f2loc_f(i1, 0)) = -1;   //become default interior face
          mesh_data.bc_id(mesh_data.f2c(i2, 0), mesh_data.f2loc_f(i2, 0)) = -3; //set the matched interface as coupled cyclic interface(=delete that face)

          mesh_data.f2loc_f(i1, 1) = mesh_data.f2loc_f(i2, 0);

          n_cyc_loc++;
          for (int k = 0; k < mesh_data.f2nv(i1); k++)
          {
            for (int m = 0; m < FlowSol->n_dims; m++)
            {
              loc_vert_0(k, m) = mesh_data.xv(mesh_data.f2v(i1, k), m);
              loc_vert_1(k, m) = mesh_data.xv(mesh_data.f2v(i2, k), m);
            }
          }
          compare_cyclic_faces(loc_vert_0, loc_vert_1, mesh_data.f2nv(i1), rtag, delta_cyclic, tol, FlowSol);
          mesh_data.rot_tag(i1) = rtag;
          break;
        }
      }

      if (found == 0) // Corresponding cyclic edges belongs to another processsor or doesn't exist
      {
        mesh_data.bc_id(mesh_data.f2c(i1, 0), mesh_data.f2loc_f(i1, 0)) = -1; //set to be mpi_interface
      }
    }
  }

#ifdef _MPI


  // ---------------------------------
  //  Initialize MPI faces
  //  --------------------------------

  int max_mpi_inters = mesh_data.n_unmatched_inters - 2 * n_cyc_loc; //place holder for face arrays
  hf_array<int> f_mpi2f(max_mpi_inters);
  FlowSol->n_mpi_inters = 0;
  int n_seg_mpi_inters = 0;
  int n_tri_mpi_inters = 0;
  int n_quad_mpi_inters = 0;

  for (int i = 0; i < mesh_data.n_unmatched_inters; i++)
  {
    int i1 = mesh_data.unmatched_inters(i);//index of interface
    bcid_f = mesh_data.bc_id(mesh_data.f2c(i1, 0), mesh_data.f2loc_f(i1, 0));
    ic_r = mesh_data.f2c(i1, 1);
    if (ic_r == -1 && bcid_f == -1)
    { // if mpi_interface or mpi cyclic setting
      if (FlowSol->nproc == 1)
      {
        cout << "ic=" << mesh_data.f2c(i1, 0) << endl;
        cout << "local_face=" << mesh_data.f2loc_f(i1, 0) << endl;
        FatalError("Can't find coupled cyclic interface");
      }

      mesh_data.bc_id(mesh_data.f2c(i1, 0), mesh_data.f2loc_f(i1, 0)) = -2; // flag as mpi_interface or mpi_cyclic
      f_mpi2f(FlowSol->n_mpi_inters++) = i1;                               //set local mpiface to local face index

      if (mesh_data.f2nv(i1) == 2)
        n_seg_mpi_inters++;
      else if (mesh_data.f2nv(i1) == 3)
        n_tri_mpi_inters++;
      else if (mesh_data.f2nv(i1) == 4)
        n_quad_mpi_inters++;
    }
  }

  FlowSol->n_mpi_inter_types = 3;
  FlowSol->mesh_mpi_inters.setup(FlowSol->n_mpi_inter_types);

  for (int i = 0; i < FlowSol->n_mpi_inter_types; i++)
    FlowSol->mesh_mpi_inters(i).set_nproc(FlowSol->nproc, FlowSol->rank);

  FlowSol->mesh_mpi_inters(0).setup(n_seg_mpi_inters, 0);
  FlowSol->mesh_mpi_inters(1).setup(n_tri_mpi_inters, 1);
  FlowSol->mesh_mpi_inters(2).setup(n_quad_mpi_inters, 2);

  hf_array<int> mpifaces_part(FlowSol->nproc); //number of interface send to each processor

  // Call function that takes in f_mpi2f,f2v and returns a new hf_array f_mpi2f, and an hf_array mpiface_part
  // that contains the number of faces to send to each processor
  // the new hf_array f_mpi2f is in good order i.e. proc1,proc2,....

  match_mpifaces(mesh_data.f2v, mesh_data.f2nv, mesh_data.xv, f_mpi2f, mpifaces_part, delta_cyclic, FlowSol->n_mpi_inters, tol, FlowSol);

  hf_array<int> rot_tag_mpi(FlowSol->n_mpi_inters);
  find_rot_mpifaces(mesh_data.f2v, mesh_data.f2nv, mesh_data.xv, f_mpi2f, rot_tag_mpi, mpifaces_part, delta_cyclic, FlowSol->n_mpi_inters, tol, FlowSol);

  //Initialize the mpi faces
  //with local element type wise index, local element type, rotation tag and interface type
  int i_seg_mpi = 0;
  int i_tri_mpi = 0;
  int i_quad_mpi = 0;

  for (int i_mpi = 0; i_mpi < FlowSol->n_mpi_inters; i_mpi++) //loop over all local mpi_inters
  {
    int i = f_mpi2f(i_mpi);
    ic_l = mesh_data.f2c(i, 0);

    if (mesh_data.f2nv(i) == 2)
    {
      FlowSol->mesh_mpi_inters(0).set_mpi(i_seg_mpi, mesh_data.ctype(ic_l), local_c(ic_l), mesh_data.f2loc_f(i, 0), rot_tag_mpi(i_mpi), FlowSol);
      i_seg_mpi++;
    }
    else if (mesh_data.f2nv(i) == 3)
    {
      FlowSol->mesh_mpi_inters(1).set_mpi(i_tri_mpi, mesh_data.ctype(ic_l), local_c(ic_l), mesh_data.f2loc_f(i, 0), rot_tag_mpi(i_mpi), FlowSol);
      i_tri_mpi++;
    }
    else if (mesh_data.f2nv(i) == 4)
    {
      FlowSol->mesh_mpi_inters(2).set_mpi(i_quad_mpi, mesh_data.ctype(ic_l), local_c(ic_l), mesh_data.f2loc_f(i, 0), rot_tag_mpi(i_mpi), FlowSol);
      i_quad_mpi++;
    }
  }

  // Initialize Nout_proc
  int icount = 0; //starting index of interface send to each processor
  int request_seg = 0;
  int request_tri = 0;
  int request_quad = 0;
  //set number of requests to each processor for each type of mpi_interface
  for (int p = 0; p < FlowSol->nproc; p++)
  {
    // For all faces to send to processor p, split between face types
    int Nout_seg = 0;
    int Nout_tri = 0;
    int Nout_quad = 0;

    for (int j = 0; j < mpifaces_part(p); j++)
    {
      int i_mpi = icount + j;
      int i = f_mpi2f(i_mpi);
      if (mesh_data.f2nv(i) == 2)
        Nout_seg++;
      else if (mesh_data.f2nv(i) == 3)
        Nout_tri++;
      else if (mesh_data.f2nv(i) == 4)
        Nout_quad++;
    }
    icount += mpifaces_part(p);

    if (Nout_seg != 0)
    {
      FlowSol->mesh_mpi_inters(0).set_nout_proc(Nout_seg, p);
      request_seg++;
    }
    if (Nout_tri != 0)
    {
      FlowSol->mesh_mpi_inters(1).set_nout_proc(Nout_tri, p);
      request_tri++;
    }
    if (Nout_quad != 0)
    {
      FlowSol->mesh_mpi_inters(2).set_nout_proc(Nout_quad, p);
      request_quad++;
    }
  }

  FlowSol->mesh_mpi_inters(0).set_mpi_requests(request_seg);
  FlowSol->mesh_mpi_inters(1).set_mpi_requests(request_tri);
  FlowSol->mesh_mpi_inters(2).set_mpi_requests(request_quad);

#ifdef _GPU

  for (int i = 0; i < FlowSol->n_mpi_inter_types; i++)
    FlowSol->mesh_mpi_inters(i).mv_all_cpu_gpu();
#endif

#endif

  // ---------------------------------------
  // Initializing internal and bdy faces
  // ---------------------------------------

  // Count the number of int_inters and bdy_inters
  int n_seg_int_inters = 0;
  int n_tri_int_inters = 0;
  int n_quad_int_inters = 0;

  int n_seg_bdy_inters = 0;
  int n_tri_bdy_inters = 0;
  int n_quad_bdy_inters = 0;

  for (int i = 0; i < mesh_data.num_inters; i++)
  {
    bcid_f = mesh_data.bc_id(mesh_data.f2c(i, 0), mesh_data.f2loc_f(i, 0));
    ic_r = mesh_data.f2c(i, 1);

    if (bcid_f != -2) //not an mpi face or mpi cyclic face
    {
      if (bcid_f == -1) // internal interface or internal cyclic face
      {
        if (ic_r == -1)
        {
          FatalError("Error: Interior interface has i_cell_right=-1. Should not be here, exiting");
        }
        n_int_inters++;
        if (mesh_data.f2nv(i) == 2)
          n_seg_int_inters++;
        if (mesh_data.f2nv(i) == 3)
          n_tri_int_inters++;
        if (mesh_data.f2nv(i) == 4)
          n_quad_int_inters++;
      }
      else // boundary interface
      {
        if (bcid_f != -3)
        { //not a coupled cyclic face(deleted one)
          n_bdy_inters++;
          if (mesh_data.f2nv(i) == 2)
          {
            n_seg_bdy_inters++;
          }
          else if (mesh_data.f2nv(i) == 3)
          {
            n_tri_bdy_inters++;
          }
          else if (mesh_data.f2nv(i) == 4)
          {
            n_quad_bdy_inters++;
          }
        }
      }
    }
  }

  FlowSol->n_int_inter_types = 3;
  FlowSol->mesh_int_inters.setup(FlowSol->n_int_inter_types);
  FlowSol->mesh_int_inters(0).setup(n_seg_int_inters, 0);
  FlowSol->mesh_int_inters(1).setup(n_tri_int_inters, 1);
  FlowSol->mesh_int_inters(2).setup(n_quad_int_inters, 2);

  FlowSol->n_bdy_inter_types = 3;
  FlowSol->mesh_bdy_inters.setup(FlowSol->n_bdy_inter_types);
  FlowSol->mesh_bdy_inters(0).setup(n_seg_bdy_inters, 0);
  FlowSol->mesh_bdy_inters(1).setup(n_tri_bdy_inters, 1);
  FlowSol->mesh_bdy_inters(2).setup(n_quad_bdy_inters, 2);

  int i_seg_int = 0;
  int i_tri_int = 0;
  int i_quad_int = 0;

  int i_seg_bdy = 0;
  int i_tri_bdy = 0;
  int i_quad_bdy = 0;

  for (int i = 0; i < mesh_data.num_inters; i++)
  {
    bcid_f = mesh_data.bc_id(mesh_data.f2c(i, 0), mesh_data.f2loc_f(i, 0));
    ic_l = mesh_data.f2c(i, 0);
    ic_r = mesh_data.f2c(i, 1);

    if (bcid_f != -2) // internal/local cyclic or boundary edge
    {
      if (bcid_f == -1)//internal/local cyclic
      {
        if (mesh_data.f2nv(i) == 2)
        {
          FlowSol->mesh_int_inters(0).set_interior(i_seg_int, mesh_data.ctype(ic_l), mesh_data.ctype(ic_r), local_c(ic_l), local_c(ic_r), mesh_data.f2loc_f(i, 0), mesh_data.f2loc_f(i, 1), mesh_data.rot_tag(i), FlowSol);
          i_seg_int++;
        }
        if (mesh_data.f2nv(i) == 3)
        {
          FlowSol->mesh_int_inters(1).set_interior(i_tri_int, mesh_data.ctype(ic_l), mesh_data.ctype(ic_r), local_c(ic_l), local_c(ic_r), mesh_data.f2loc_f(i, 0), mesh_data.f2loc_f(i, 1), mesh_data.rot_tag(i), FlowSol);
          i_tri_int++;
        }
        if (mesh_data.f2nv(i) == 4)
        {
          FlowSol->mesh_int_inters(2).set_interior(i_quad_int, mesh_data.ctype(ic_l), mesh_data.ctype(ic_r), local_c(ic_l), local_c(ic_r), mesh_data.f2loc_f(i, 0), mesh_data.f2loc_f(i, 1), mesh_data.rot_tag(i), FlowSol);
          i_quad_int++;
        }
      }
      else // boundary face other than cyclic face
      {
        if (bcid_f != -3)//not a coupled local cyclic face(deleted one)
        {
          if (mesh_data.f2nv(i) == 2)
          {
            FlowSol->mesh_bdy_inters(0).set_boundary(i_seg_bdy, bcid_f, mesh_data.ctype(ic_l), local_c(ic_l), mesh_data.f2loc_f(i, 0), FlowSol);
            i_seg_bdy++;
          }
          else if (mesh_data.f2nv(i) == 3)
          {
            FlowSol->mesh_bdy_inters(1).set_boundary(i_tri_bdy, bcid_f, mesh_data.ctype(ic_l), local_c(ic_l), mesh_data.f2loc_f(i, 0), FlowSol);
            i_tri_bdy++;
          }
          else if (mesh_data.f2nv(i) == 4)
          {
            FlowSol->mesh_bdy_inters(2).set_boundary(i_quad_bdy, bcid_f, mesh_data.ctype(ic_l), local_c(ic_l), mesh_data.f2loc_f(i, 0), FlowSol);
            i_quad_bdy++;
          }
        }
      }
    }
  }

  // calculate wall distance for smagorinsky or S-A models
  if ((run_input.LES && run_input.SGS_model == 0) || run_input.RANS)
  {

    if (FlowSol->rank == 0)
      cout << "calculating wall distance... " << endl;

    int n_seg_noslip_inters = 0;
    int n_tri_noslip_inters = 0;
    int n_quad_noslip_inters = 0;
    int order = run_input.order;
    int n_fpts_per_inter_seg = order + 1;
    int n_fpts_per_inter_tri = (order + 2) * (order + 1) / 2;
    int n_fpts_per_inter_quad = (order + 1) * (order + 1);

    //No-slip wall flux point coordinates for wall models.
	  hf_array< hf_array<double> > loc_noslip_bdy;

    for (int i = 0; i < mesh_data.num_inters; i++)
    {

      bcid_f = mesh_data.bc_id(mesh_data.f2c(i, 0), mesh_data.f2loc_f(i, 0));

      int bctype_f = run_input.bc_list(bcid_f).get_bc_flag();
      // All types of no-slip wall
      if (bctype_f == ISOTHERM_WALL || bctype_f == ADIABAT_WALL)
      {
        // segs
        if (mesh_data.f2nv(i) == 2)
          n_seg_noslip_inters++;

        // tris
        if (mesh_data.f2nv(i) == 3)
          n_tri_noslip_inters++;

        // quads
        if (mesh_data.f2nv(i) == 4)
          n_quad_noslip_inters++;
      }
    }

#ifdef _MPI

    //global No-slip wall flux point coordinates for wall models.
	  hf_array< hf_array<double> > loc_noslip_bdy_global;

    hf_array<int> n_seg_inters_array(FlowSol->nproc);
    hf_array<int> n_tri_inters_array(FlowSol->nproc);
    hf_array<int> n_quad_inters_array(FlowSol->nproc);
    hf_array<int> kstart_seg(FlowSol->nproc);
    hf_array<int> kstart_tri(FlowSol->nproc);
    hf_array<int> kstart_quad(FlowSol->nproc);
    int n_global_seg_noslip_inters = 0;
    int n_global_tri_noslip_inters = 0;
    int n_global_quad_noslip_inters = 0;

    /*! Communicate to all processors the total number of no-slip boundary
    inters, the maximum number of no-slip boundary inters on any single
    partition, and the number of no-slip inters on each partition. */

    MPI_Allgather(&n_seg_noslip_inters, 1, MPI_INT, n_seg_inters_array.get_ptr_cpu(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&n_tri_noslip_inters, 1, MPI_INT, n_tri_inters_array.get_ptr_cpu(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&n_quad_noslip_inters, 1, MPI_INT, n_quad_inters_array.get_ptr_cpu(), 1, MPI_INT, MPI_COMM_WORLD);

    for (int i = 0; i < FlowSol->nproc; i++)
    {
      kstart_seg(i) = n_global_seg_noslip_inters;
      kstart_tri(i) = n_global_tri_noslip_inters;
      kstart_quad(i) = n_global_quad_noslip_inters;

      n_global_seg_noslip_inters += n_seg_inters_array(i);
      n_global_tri_noslip_inters += n_tri_inters_array(i);
      n_global_quad_noslip_inters += n_quad_inters_array(i);
    }

#endif

    // Allocate arrays for coordinates of points on no-slip boundaries
    loc_noslip_bdy.setup(FlowSol->n_bdy_inter_types);
    loc_noslip_bdy(0).setup(FlowSol->n_dims, n_fpts_per_inter_seg, n_seg_noslip_inters);
    loc_noslip_bdy(1).setup(FlowSol->n_dims, n_fpts_per_inter_tri, n_tri_noslip_inters);
    loc_noslip_bdy(2).setup(FlowSol->n_dims, n_fpts_per_inter_quad, n_quad_noslip_inters);

    n_seg_noslip_inters = 0;
    n_tri_noslip_inters = 0;
    n_quad_noslip_inters = 0;

    // Get coordinates
    for (int i = 0; i < mesh_data.num_inters; i++)
    {

      ic_l = mesh_data.f2c(i, 0);
      bcid_f = mesh_data.bc_id(mesh_data.f2c(i, 0), mesh_data.f2loc_f(i, 0));
      int bctype_f = run_input.bc_list(bcid_f).get_bc_flag();
      // All types of no-slip wall
      if (bctype_f == ISOTHERM_WALL || bctype_f == ADIABAT_WALL)
      {

        // segs
        if (mesh_data.f2nv(i) == 2)
        {
          for (int j = 0; j < n_fpts_per_inter_seg; j++)
          {
            for (int k = 0; k < FlowSol->n_dims; k++)
            {

              // find coordinates
              loc_noslip_bdy(0)(k, j, n_seg_noslip_inters) = *get_loc_fpts_ptr_cpu(mesh_data.ctype(ic_l), local_c(ic_l), mesh_data.f2loc_f(i, 0), j, k, FlowSol);
            }
          }
          n_seg_noslip_inters++;
        }
        // tris
        if (mesh_data.f2nv(i) == 3)
        {
          for (int j = 0; j < n_fpts_per_inter_tri; j++)
          {
            for (int k = 0; k < FlowSol->n_dims; k++)
            {

              loc_noslip_bdy(1)(k, j, n_tri_noslip_inters) = *get_loc_fpts_ptr_cpu(mesh_data.ctype(ic_l), local_c(ic_l), mesh_data.f2loc_f(i, 0), j, k, FlowSol);
            }
          }
          n_tri_noslip_inters++;
        }
        // quads
        if (mesh_data.f2nv(i) == 4)
        {
          for (int j = 0; j < n_fpts_per_inter_quad; j++)
          {
            for (int k = 0; k < FlowSol->n_dims; k++)
            {

              loc_noslip_bdy(2)( k,j, n_quad_noslip_inters) = *get_loc_fpts_ptr_cpu(mesh_data.ctype(ic_l), local_c(ic_l), mesh_data.f2loc_f(i, 0), j, k, FlowSol);
            }
          }

          n_quad_noslip_inters++;
        }
      }
    }

#ifdef _MPI

    // Allocate global arrays for coordinates of points on no-slip boundaries
    loc_noslip_bdy_global.setup(FlowSol->n_bdy_inter_types);
    loc_noslip_bdy_global(0).setup(FlowSol->n_dims, n_fpts_per_inter_seg, n_global_seg_noslip_inters);
    loc_noslip_bdy_global(1).setup(FlowSol->n_dims, n_fpts_per_inter_tri, n_global_tri_noslip_inters);
    loc_noslip_bdy_global(2).setup(FlowSol->n_dims, n_fpts_per_inter_quad, n_global_quad_noslip_inters);

    //copy loc_noslip_bdy to the corresponding position of the global array
    int temp_ptr, temp_blk_siz;
    //seg
    temp_ptr = kstart_seg(FlowSol->rank) * n_fpts_per_inter_seg * FlowSol->n_dims;
    temp_blk_siz = n_seg_noslip_inters * n_fpts_per_inter_seg * FlowSol->n_dims;
    copy(loc_noslip_bdy(0).get_ptr_cpu(), loc_noslip_bdy(0).get_ptr_cpu(temp_blk_siz),
         loc_noslip_bdy_global(0).get_ptr_cpu(temp_ptr));
    //tri
    temp_ptr = kstart_tri(FlowSol->rank) * n_fpts_per_inter_tri * FlowSol->n_dims;
    temp_blk_siz = n_tri_noslip_inters * n_fpts_per_inter_tri * FlowSol->n_dims;
    copy(loc_noslip_bdy(1).get_ptr_cpu(), loc_noslip_bdy(1).get_ptr_cpu(temp_blk_siz),
         loc_noslip_bdy_global(1).get_ptr_cpu(temp_ptr));
    //quad
    temp_ptr = kstart_quad(FlowSol->rank) * n_fpts_per_inter_quad * FlowSol->n_dims;
    temp_blk_siz = n_quad_noslip_inters * n_fpts_per_inter_quad * FlowSol->n_dims;
    copy(loc_noslip_bdy(2).get_ptr_cpu(), loc_noslip_bdy(2).get_ptr_cpu(temp_blk_siz),
         loc_noslip_bdy_global(2).get_ptr_cpu(temp_ptr));
    // Broadcast coordinates of interface points to all partitions
    for (int np = 0; np < FlowSol->nproc; np++)
    {
      MPI_Bcast(loc_noslip_bdy_global(0).get_ptr_cpu(kstart_seg(np) * n_fpts_per_inter_seg * FlowSol->n_dims), n_seg_inters_array(np) * n_fpts_per_inter_seg * FlowSol->n_dims, MPI_DOUBLE, np, MPI_COMM_WORLD);
      MPI_Bcast(loc_noslip_bdy_global(1).get_ptr_cpu(kstart_tri(np) * n_fpts_per_inter_tri * FlowSol->n_dims), n_tri_inters_array(np) * n_fpts_per_inter_tri * FlowSol->n_dims, MPI_DOUBLE, np, MPI_COMM_WORLD);
      MPI_Bcast(loc_noslip_bdy_global(2).get_ptr_cpu(kstart_quad(np) * n_fpts_per_inter_quad * FlowSol->n_dims), n_quad_inters_array(np) * n_fpts_per_inter_quad * FlowSol->n_dims, MPI_DOUBLE, np, MPI_COMM_WORLD);
    }

    // Calculate distance of every solution point to nearest point on no-slip boundary for every partition
    for (int i = 0; i < FlowSol->n_ele_types; i++)
      FlowSol->mesh_eles(i)->calc_wall_distance(n_global_seg_noslip_inters, n_global_tri_noslip_inters, n_global_quad_noslip_inters, loc_noslip_bdy_global);

#else // serial

    // Calculate distance of every solution point to nearest point on no-slip boundary
    for (int i = 0; i < FlowSol->n_ele_types; i++)
      FlowSol->mesh_eles(i)->calc_wall_distance(n_seg_noslip_inters, n_tri_noslip_inters, n_quad_noslip_inters, loc_noslip_bdy);

#endif
  }

// set on GPU
#ifdef _GPU
  if (FlowSol->rank == 0)
    cout << "Moving interfaces to GPU ... " << endl;
  for (int i = 0; i < FlowSol->n_int_inter_types; i++)
    FlowSol->mesh_int_inters(i).mv_all_cpu_gpu();

  for (int i = 0; i < FlowSol->n_bdy_inter_types; i++)
    FlowSol->mesh_bdy_inters(i).mv_all_cpu_gpu();

  if (FlowSol->rank == 0)
    cout << "Moving wall_distance to GPU ... " << endl;
  for (int i = 0; i < FlowSol->n_ele_types; i++)
    FlowSol->mesh_eles(i)->mv_wall_distance_cpu_gpu();

  for (int i = 0; i < FlowSol->n_ele_types; i++)
    FlowSol->mesh_eles(i)->mv_wall_distance_mag_cpu_gpu();
#endif

}

void ReadMesh(struct solution *FlowSol, mesh &mesh_data)
{

  if (FlowSol->rank == 0)
    cout << endl
         << "----------------------- Mesh Preprocessing ------------------------" << endl;

  /*------------------Parallel read element connectivity--------------------*/
  mesh_reader m_r(run_input.mesh_file, &mesh_data); //initialize mesh reader
  FlowSol->n_dims = mesh_data.n_dims;               //copy dimension to flowsol
  FlowSol->num_cells_global = mesh_data.num_cells_global; //copy total number of cell to solution

  if (FlowSol->rank == 0)
    cout << "reading connectivity ... " << endl;

  //calculate the start and end index to read based on the number of processors
  //waiting to be further partitioned
  int kstart, temp_num_cells;
#ifdef _MPI
  // Assign a number of cells for each processor
  temp_num_cells = mesh_data.num_cells_global / FlowSol->nproc;
  kstart = FlowSol->rank * temp_num_cells;

  // Last processor has more cells
  if (FlowSol->rank == (FlowSol->nproc - 1))
    temp_num_cells += (mesh_data.num_cells_global - FlowSol->nproc * temp_num_cells);
#else
  kstart = 0;
  temp_num_cells = mesh_data.num_cells_global;
#endif

  m_r.partial_read_connectivity(kstart, temp_num_cells);

  if (FlowSol->rank == 0)
    cout << "done reading connectivity" << endl;

/*----------------Repartition mesh--------------------*/
#ifdef _MPI
  // Call method to repartition the mesh
  if (FlowSol->nproc != 1)
    mesh_data.repartition_mesh(FlowSol->nproc, FlowSol->rank);
#endif

  mesh_data.create_iv2ivg(); //fill and reorder iv2ivg in ascending order

  /*----------------Read vertics belongs to this processor--------------------*/
  if (FlowSol->rank == 0)
    cout << "reading vertices" << endl;

  m_r.read_vertices();

  if (FlowSol->rank == 0)
    cout << "done reading vertices" << endl;

  /*----------------Compute face connectivity-------------------- */

  if (FlowSol->rank == 0)
    cout << "Setting up face/vertex connectivity" << endl;

  CompConnectivity(mesh_data);

  if (FlowSol->rank == 0)
    cout << "Done setting up face/vertex connectivity" << endl;

  /*----------------Read boundary condition--------------------*/

  if (FlowSol->rank == 0)
    cout << "reading boundary conditions" << endl;
  //read boundary condition groups in the mesh file
  m_r.read_boundary();
  //read boundary condition parameters in the input file
  run_input.read_boundary_param();
  if (FlowSol->rank == 0)
  {
    for (int i = 0; i < mesh_data.n_bdy; i++)
    {
      cout << run_input.bc_list(i).get_bc_name() << " -> " << run_input.bc_list(i).get_bc_type() << endl;
    }
    cout << "done reading boundary conditions" << endl;
  }
  
}

/*! method to create list of faces & edges from the mesh */
void CompConnectivity(mesh &mesh_data)
{
  // Determine how many cells share each node
  mesh_data.set_vertex_connectivity();

  mesh_data.set_face_connectivity();
}

// method to compare two faces and check if they match

void compare_cyclic_faces(hf_array<double> &xvert1, hf_array<double> &xvert2, int &num_v_per_f, int &rtag, hf_array<double> &delta_cyclic, double tol, struct solution *FlowSol)
{
  int found = 0;
  if (FlowSol->n_dims == 2)
  {
    found = 1;
    rtag = 0;
  }
  else if (FlowSol->n_dims == 3)
  {
    if (num_v_per_f == 4) // quad face
    {
      //printf("cell 1, x1=%2.8f, x2=%2.8f, x3=%2.8f\n cell 2(1), x1=%2.8f, x2=%2.8f, x3=%2.8f\n",xvert1(0,0),xvert1(0,1),xvert1(0,2),xvert2(0,0),xvert2(0,1),xvert2(0,2));
      //printf("cell 2(2), x1=%2.8f, x2=%2.8f, x3=%2.8f\n cell 2(3), x1=%2.8f, x2=%2.8f, x3=%2.8f\n cell 2(4), x1=%2.8f, x2=%2.8f, x3=%2.8f\n",xvert2(1,0),xvert2(1,1),xvert2(1,2),xvert2(2,0),xvert2(2,1),xvert2(2,2),xvert2(3,0),xvert2(3,1),xvert2(3,2));
      // Determine rot_tag based on xvert1(0)
      // vert1(0) matches vert2(1), rot_tag=0
      if (
          (abs(abs(xvert1(0, 0) - xvert2(1, 0)) - delta_cyclic(0)) < tol && abs(abs(xvert1(0, 1) - xvert2(1, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(1, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(1, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(1, 1)) - delta_cyclic(1)) < tol && abs(abs(xvert1(0, 2) - xvert2(1, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(1, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(1, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(1, 2)) - delta_cyclic(2)) < tol))
      {
        rtag = 0;
        found = 1;
      }
      // vert1(0) matches vert2(3), rot_tag=1
      else if (
          (abs(abs(xvert1(0, 0) - xvert2(3, 0)) - delta_cyclic(0)) < tol && abs(abs(xvert1(0, 1) - xvert2(3, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(3, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(3, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(3, 1)) - delta_cyclic(1)) < tol && abs(abs(xvert1(0, 2) - xvert2(3, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(3, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(3, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(3, 2)) - delta_cyclic(2)) < tol))
      {
        rtag = 1;
        found = 1;
      }
      // vert1(0) matches vert2(0), rot_tag=2
      else if (
          (abs(abs(xvert1(0, 0) - xvert2(0, 0)) - delta_cyclic(0)) < tol && abs(abs(xvert1(0, 1) - xvert2(0, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(0, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(0, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(0, 1)) - delta_cyclic(1)) < tol && abs(abs(xvert1(0, 2) - xvert2(0, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(0, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(0, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(0, 2)) - delta_cyclic(2)) < tol))
      {
        rtag = 2;
        found = 1;
      }
      // vert1(0) matches vert2(2), rot_tag=3
      else if (
          (abs(abs(xvert1(0, 0) - xvert2(2, 0)) - delta_cyclic(0)) < tol && abs(abs(xvert1(0, 1) - xvert2(2, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(2, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(2, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(2, 1)) - delta_cyclic(1)) < tol && abs(abs(xvert1(0, 2) - xvert2(2, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(2, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(2, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(2, 2)) - delta_cyclic(2)) < tol))
      {
        rtag = 3;
        found = 1;
      }
    }
    else if (num_v_per_f == 3) // tri face
    {
      //printf("cell 1, x1=%f, x2=%f, x3=%f\n cell 2, x1=%f, x2=%f, x3=%f\n",xvert1(0,0),xvert1(1,0),xvert1(2,0),xvert2(0,0),xvert2(1,0),xvert2(2,0));
      //printf("cell 1, y1=%f, y2=%f, y3=%f\n cell 2, y1=%f, y2=%f, y3=%f\n",xvert1(0,1),xvert1(1,1),xvert1(2,1),xvert2(0,1),xvert2(1,1),xvert2(2,1));
      // Determine rot_tag based on xvert1(0)
      // vert1(0) matches vert2(0), rot_tag=0
      if (
          (abs(abs(xvert1(0, 0) - xvert2(0, 0)) - delta_cyclic(0)) < tol && abs(abs(xvert1(0, 1) - xvert2(0, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(0, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(0, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(0, 1)) - delta_cyclic(1)) < tol && abs(abs(xvert1(0, 2) - xvert2(0, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(0, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(0, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(0, 2)) - delta_cyclic(2)) < tol))
      {
        rtag = 0;
        found = 1;
      }
      // vert1(0) matches vert2(2), rot_tag=1
      else if (
          (abs(abs(xvert1(0, 0) - xvert2(2, 0)) - delta_cyclic(0)) < tol && abs(abs(xvert1(0, 1) - xvert2(2, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(2, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(2, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(2, 1)) - delta_cyclic(1)) < tol && abs(abs(xvert1(0, 2) - xvert2(2, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(2, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(2, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(2, 2)) - delta_cyclic(2)) < tol))
      {
        rtag = 1;
        found = 1;
      }
      // vert1(0) matches vert2(1), rot_tag=2
      else if (
          (abs(abs(xvert1(0, 0) - xvert2(1, 0)) - delta_cyclic(0)) < tol && abs(abs(xvert1(0, 1) - xvert2(1, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(1, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(1, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(1, 1)) - delta_cyclic(1)) < tol && abs(abs(xvert1(0, 2) - xvert2(1, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(1, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(1, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(1, 2)) - delta_cyclic(2)) < tol))
      {
        rtag = 2;
        found = 1;
      }
    }
    else
    {
      FatalError("ERROR: Haven't implemented this face type in compare_cyclic_face yet....");
    }
  }

  if (found == 0)
    FatalError("Could not match vertices in compare faces");
}

bool check_cyclic(hf_array<double> &delta_cyclic, hf_array<double> &loc_center_inter_0, hf_array<double> &loc_center_inter_1, double tol, struct solution *FlowSol)
{

  bool output;
  if (FlowSol->n_dims == 3)
  {
    output =
        (abs(abs(loc_center_inter_0(0) - loc_center_inter_1(0)) - delta_cyclic(0)) < tol && abs(loc_center_inter_0(1) - loc_center_inter_1(1)) < tol && abs(loc_center_inter_0(2) - loc_center_inter_1(2)) < tol) ||

        (abs(loc_center_inter_0(0) - loc_center_inter_1(0)) < tol && abs(abs(loc_center_inter_0(1) - loc_center_inter_1(1)) - delta_cyclic(1)) < tol && abs(loc_center_inter_0(2) - loc_center_inter_1(2)) < tol) ||

        (abs(loc_center_inter_0(0) - loc_center_inter_1(0)) < tol && abs(loc_center_inter_0(1) - loc_center_inter_1(1)) < tol && abs(abs(loc_center_inter_0(2) - loc_center_inter_1(2)) - delta_cyclic(2)) < tol);
  }
  else if (FlowSol->n_dims == 2)
  {
    output =
        (abs(abs(loc_center_inter_0(0) - loc_center_inter_1(0)) - delta_cyclic(0)) < tol && abs(loc_center_inter_0(1) - loc_center_inter_1(1)) < tol) ||

        (abs(loc_center_inter_0(0) - loc_center_inter_1(0)) < tol && abs(abs(loc_center_inter_0(1) - loc_center_inter_1(1)) - delta_cyclic(1)) < tol);
  }

  return output;
}

#ifdef _MPI
//try to find matched mpi interface across processors by comparing centroid of interfaces
void match_mpifaces(hf_array<int> &in_f2v, hf_array<int> &in_f2nv, hf_array<double> &in_xv, hf_array<int> &inout_f_mpi2f, hf_array<int> &out_mpifaces_part, hf_array<double> &delta_cyclic, int n_mpi_faces, double tol, struct solution *FlowSol)
{

  int icount;

  hf_array<int> matched(n_mpi_faces);
  hf_array<int> old_f_mpi2f;

  old_f_mpi2f = inout_f_mpi2f;

  hf_array<double> delta_zero(FlowSol->n_dims);
  delta_zero.initialize_to_zero();

  // Calculate the centroid of each face
  hf_array<double> loc_center_inter(FlowSol->n_dims, n_mpi_faces);
  loc_center_inter.initialize_to_zero();

  for (int i = 0; i < n_mpi_faces; i++)
  {
    for (int m = 0; m < FlowSol->n_dims; m++)           //for each dimension
      for (int k = 0; k < in_f2nv(old_f_mpi2f(i)); k++) //for each vertex on the interface
        loc_center_inter(m, i) += in_xv(in_f2v(old_f_mpi2f(i), k), m) / (double)in_f2nv(old_f_mpi2f(i));
  }

  // Initialize matched with 0
  matched.initialize_to_zero();

  //local number of mpi interface send to each processor
  out_mpifaces_part.initialize_to_zero();

  // Exchange the number of mpi_faces to receive
  // Create hf_array mpfaces_from
  hf_array<int> mpifaces_from(FlowSol->nproc); //number of mpi interface on each processor
  MPI_Allgather(&n_mpi_faces, 1, MPI_INT, mpifaces_from.get_ptr_cpu(), 1, MPI_INT, MPI_COMM_WORLD);

  hf_array<double> in_loc_center_inter;//interface centeriod location buffer
  hf_array<double> loc_center_1(FlowSol->n_dims);
  hf_array<double> loc_center_2(FlowSol->n_dims);

  // Begin the exchange
  icount = 0;
  for (int p = 0; p < FlowSol->nproc; p++)//index of processor to send data
  {
    if (p == FlowSol->rank) // Send data
    {
      in_loc_center_inter = loc_center_inter; //load data to buffer
    }
    else
      in_loc_center_inter.setup(FlowSol->n_dims, mpifaces_from(p));//allocate space to buffer                                                  //prepare receive buffer
    MPI_Bcast(in_loc_center_inter.get_ptr_cpu(), FlowSol->n_dims * mpifaces_from(p), MPI_DOUBLE, p, MPI_COMM_WORLD); //broadcast to other processors or receive broadcast from p

 //make sure f_mpi2f have same order across processors
    if (p < FlowSol->rank)
    {
      // Loop over local mpi_edges
      for (int iloc = 0; iloc < n_mpi_faces; iloc++)
      {
        if (!matched(iloc)) // if local edge hasn't been matched yet
        {
          // Loop over remote edges just received
          for (int irem = 0; irem < mpifaces_from(p); irem++)
          {
            for (int m = 0; m < FlowSol->n_dims; m++) //load centroids
            {
              loc_center_1(m) = in_loc_center_inter(m, irem);
              loc_center_2(m) = loc_center_inter(m, iloc);
            }
            if (check_cyclic(delta_cyclic, loc_center_1, loc_center_2, tol, FlowSol) ||
                check_cyclic(delta_zero, loc_center_1, loc_center_2, tol, FlowSol))
            {
              matched(iloc) = 1;
              out_mpifaces_part(p)++;
              inout_f_mpi2f(icount) = old_f_mpi2f(iloc); //old local prime, processor minor
              icount++;
              break;
            }
          }
        }
      }
    }
    else if (p > FlowSol->rank) 
    {
      // Loop over remote edges
      for (int irem = 0; irem < mpifaces_from(p); irem++)
      {
        for (int iloc = 0; iloc < n_mpi_faces; iloc++)
        {
          if (!matched(iloc)) // if local edge hasn't been matched yet
          {
            for (int m = 0; m < FlowSol->n_dims; m++)
            {
              loc_center_1(m) = in_loc_center_inter(m, irem);
              loc_center_2(m) = loc_center_inter(m, iloc);
            }
            // Check if it matches vertex iloc
            if (check_cyclic(delta_cyclic, loc_center_1, loc_center_2, tol, FlowSol) ||
                check_cyclic(delta_zero, loc_center_1, loc_center_2, tol, FlowSol))
            {
              matched(iloc) = 1;
              out_mpifaces_part(p)++;
              inout_f_mpi2f(icount) = old_f_mpi2f(iloc); //old remote prime, processor minor
              icount++;
              break;
            }
          }
        }
      }
    }
  }

  // Check that every edge has been matched
  for (int i = 0; i < n_mpi_faces; i++)
  {
    if (!matched(i))
    {
      cout << "rank=" << FlowSol->rank << "i=" << i << endl;
      FatalError("Some mpi_faces were not matched!!! could try changing tol, exiting!");
    }
  }
}

void find_rot_mpifaces(hf_array<int> &in_f2v, hf_array<int> &in_f2nv, hf_array<double> &in_xv, hf_array<int> &in_f_mpi2f, hf_array<int> &out_rot_tag_mpi, hf_array<int> &mpifaces_part, hf_array<double> delta_cyclic, int n_mpi_faces, double tol, struct solution *FlowSol)
{

  int Nout;
  int n_vert_out;
  int count1, count2, count3;
  int rtag;

  // Count the number of messages(processor) to send
  int request_count = 0;
  for (int p = 0; p < FlowSol->nproc; p++)
  {
    if (mpifaces_part(p) != 0)
      request_count++;
  }

  hf_array<MPI_Request> mpi_in_requests(request_count);
  hf_array<MPI_Request> mpi_out_requests(request_count);

  // Count total number of vertices to send
  n_vert_out = 0;
  for (int i_mpi = 0; i_mpi < n_mpi_faces; i_mpi++)
  {
    n_vert_out += in_f2nv(in_f_mpi2f(i_mpi));
  }
  //create an hf_array contain all vertex on local mpi interface
  hf_array<double> xyz_vert_out(FlowSol->n_dims, n_vert_out);
  hf_array<double> xyz_vert_in(FlowSol->n_dims, n_vert_out);

  int Nmess = 0; //number of message to wait
  int sk = 0;    //start index for each message

  count2 = 0;        //start index of interface send to this processor
  count3 = 0;        //index of vertex in sending buffer
  request_count = 0; //request counter

  for (int p = 0; p < FlowSol->nproc; p++) //for each processor
  {
    count1 = 0;                                //total number of vertex send to/receive from p
    for (int i = 0; i < mpifaces_part(p); i++) //for each interface send to/receive from p
    {
      int i_mpi = count2 + i;
      for (int k = 0; k < in_f2nv(in_f_mpi2f(i_mpi)); k++) //for each vertex on that interface
      {
        for (int m = 0; m < FlowSol->n_dims; m++)
          xyz_vert_out(m, count3) = in_xv(in_f2v(in_f_mpi2f(i_mpi), k), m); //copy to sending buffer
        count3++;
      }
      count1 += in_f2nv(in_f_mpi2f(i_mpi));
    }

    Nout = count1;
    count2 += mpifaces_part(p);

    if (Nout) //if have common interface with p
    {
      MPI_Isend(xyz_vert_out.get_ptr_cpu(0, sk), Nout * FlowSol->n_dims, MPI_DOUBLE, p, p, MPI_COMM_WORLD, mpi_out_requests.get_ptr_cpu(request_count));           //from me to p tag p
      MPI_Irecv(xyz_vert_in.get_ptr_cpu(0, sk), Nout * FlowSol->n_dims, MPI_DOUBLE, p, FlowSol->rank, MPI_COMM_WORLD, mpi_in_requests.get_ptr_cpu(request_count)); //from p to me tag me
      sk += Nout;
      Nmess++;
      request_count++;
    }
  }

  MPI_Waitall(Nmess, mpi_in_requests.get_ptr_cpu(), MPI_STATUSES_IGNORE);
  MPI_Waitall(Nmess, mpi_out_requests.get_ptr_cpu(), MPI_STATUSES_IGNORE);

  hf_array<double> loc_vert_0(MAX_V_PER_F, FlowSol->n_dims);
  hf_array<double> loc_vert_1(MAX_V_PER_F, FlowSol->n_dims);

  count1 = 0;
  for (int i_mpi = 0; i_mpi < n_mpi_faces; i_mpi++) //for each local mpi interface
  {
    for (int k = 0; k < in_f2nv(in_f_mpi2f(i_mpi)); k++) //for each vertex of that interface
    {
      for (int m = 0; m < FlowSol->n_dims; m++) //compare vertex from received and local
      {
        loc_vert_0(k, m) = xyz_vert_out(m, count1);
        loc_vert_1(k, m) = xyz_vert_in(m, count1);
      }
      count1++;
    }

    compare_mpi_faces(loc_vert_0, loc_vert_1, in_f2nv(in_f_mpi2f(i_mpi)), rtag, delta_cyclic, tol, FlowSol);
    out_rot_tag_mpi(i_mpi) = rtag;
  }
}

// method to compare two faces and check if they match
void compare_mpi_faces(hf_array<double> &xvert1, hf_array<double> &xvert2, int &num_v_per_f, int &rtag, hf_array<double> &delta_cyclic, double tol, struct solution *FlowSol)
{
  int found = 0;
  if (FlowSol->n_dims == 2)
  {
    found = 1;
    rtag = 0;
  }
  else if (FlowSol->n_dims == 3)
  {
    if (num_v_per_f == 4) // quad face
    {
      // Determine rot_tag based on xvert1(0)
      // vert1(0) matches vert2(1), rot_tag=0
      if (
          (abs(abs(xvert1(0, 0) - xvert2(1, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(1, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(1, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(1, 0)) - delta_cyclic(0)) < tol && abs(abs(xvert1(0, 1) - xvert2(1, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(1, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(1, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(1, 1)) - delta_cyclic(1)) < tol && abs(abs(xvert1(0, 2) - xvert2(1, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(1, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(1, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(1, 2)) - delta_cyclic(2)) < tol))
      {
        rtag = 0;
        found = 1;
      }
      // vert1(0) matches vert2(3), rot_tag=1
      else if (
          (abs(abs(xvert1(0, 0) - xvert2(3, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(3, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(3, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(3, 0)) - delta_cyclic(0)) < tol && abs(abs(xvert1(0, 1) - xvert2(3, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(3, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(3, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(3, 1)) - delta_cyclic(1)) < tol && abs(abs(xvert1(0, 2) - xvert2(3, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(3, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(3, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(3, 2)) - delta_cyclic(2)) < tol))
      {
        rtag = 1;
        found = 1;
      }
      // vert1(0) matches vert2(0), rot_tag=2
      else if (
          (abs(abs(xvert1(0, 0) - xvert2(0, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(0, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(0, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(0, 0)) - delta_cyclic(0)) < tol && abs(abs(xvert1(0, 1) - xvert2(0, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(0, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(0, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(0, 1)) - delta_cyclic(1)) < tol && abs(abs(xvert1(0, 2) - xvert2(0, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(0, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(0, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(0, 2)) - delta_cyclic(2)) < tol))
      {
        rtag = 2;
        found = 1;
      }
      // vert1(0) matches vert2(2), rot_tag=3
      else if (
          (abs(abs(xvert1(0, 0) - xvert2(2, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(2, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(2, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(2, 0)) - delta_cyclic(0)) < tol && abs(abs(xvert1(0, 1) - xvert2(2, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(2, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(2, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(2, 1)) - delta_cyclic(1)) < tol && abs(abs(xvert1(0, 2) - xvert2(2, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(2, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(2, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(2, 2)) - delta_cyclic(2)) < tol))
      {
        rtag = 3;
        found = 1;
      }
    }
    else if (num_v_per_f == 3) // tri face
    {
      // vert1(0) matches vert2(0), rot_tag=0
      if (
          (abs(abs(xvert1(0, 0) - xvert2(0, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(0, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(0, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(0, 0)) - delta_cyclic(0)) < tol && abs(abs(xvert1(0, 1) - xvert2(0, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(0, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(0, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(0, 1)) - delta_cyclic(1)) < tol && abs(abs(xvert1(0, 2) - xvert2(0, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(0, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(0, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(0, 2)) - delta_cyclic(2)) < tol))
      {
        rtag = 0;
        found = 1;
      }
      // vert1(0) matches vert2(2), rot_tag=1
      else if (
          (abs(abs(xvert1(0, 0) - xvert2(2, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(2, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(2, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(2, 0)) - delta_cyclic(0)) < tol && abs(abs(xvert1(0, 1) - xvert2(2, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(2, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(2, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(2, 1)) - delta_cyclic(1)) < tol && abs(abs(xvert1(0, 2) - xvert2(2, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(2, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(2, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(2, 2)) - delta_cyclic(2)) < tol))
      {
        rtag = 1;
        found = 1;
      }
      // vert1(0) matches vert2(1), rot_tag=2
      else if (
          (abs(abs(xvert1(0, 0) - xvert2(1, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(1, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(1, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(1, 0)) - delta_cyclic(0)) < tol && abs(abs(xvert1(0, 1) - xvert2(1, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(1, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(1, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(1, 1)) - delta_cyclic(1)) < tol && abs(abs(xvert1(0, 2) - xvert2(1, 2))) < tol) ||
          (abs(abs(xvert1(0, 0) - xvert2(1, 0))) < tol && abs(abs(xvert1(0, 1) - xvert2(1, 1))) < tol && abs(abs(xvert1(0, 2) - xvert2(1, 2)) - delta_cyclic(2)) < tol))
      {
        rtag = 2;
        found = 1;
      }
    }
    else
    {
      FatalError("ERROR: Haven't implemented this face type in compare_cyclic_face yet....");
    }
  }

  if (found == 0)
    FatalError("Could not match vertices in compare faces");
}

#endif
