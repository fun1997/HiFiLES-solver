
#include <algorithm>

#include "../include/mesh.h"
#include "../include/funcs.h"
#ifdef _MPI
#include "mpi.h"
#include "metis.h"
#include "parmetis.h"
#endif // _MPI

mesh::mesh()
{
  /*! Number of edges/faces for different type of cells. */
  num_f_per_c.setup(5);
  num_f_per_c(0) = 3;
  num_f_per_c(1) = 4;
  num_f_per_c(2) = 4;
  num_f_per_c(3) = 5;
  num_f_per_c(4) = 6;
}

mesh::~mesh()
{
}

int mesh::get_num_cells()
{
  return num_cells;
}

int mesh::get_num_cells(int in_type)
{
  int count = 0;
  for (int i = 0; i < num_cells; i++)
    if (ctype(i) == in_type)
      count++;

  return count;
}

int mesh::get_max_n_spts(int in_type)
{
  int max_n_spts = 0;
  for (int i = 0; i < num_cells; i++)
    if (ctype(i) == in_type && c2n_v(i) > max_n_spts) // triangle
      max_n_spts = c2n_v(i);

  return max_n_spts;
}

#ifdef _MPI
void mesh::repartition_mesh(int nproc, int rank)
{

  // Create hf_array that stores the number of cells per proc
  hf_array<int> kprocs(nproc);

  MPI_Allgather(&num_cells, 1, MPI_INT, kprocs.get_ptr_cpu(), 1, MPI_INT, MPI_COMM_WORLD);

  // element distribution
  hf_array<idx_t> elmdist(nproc + 1);
  elmdist[0] = 0;
  for (int p = 0; p < nproc; p++)
    elmdist[p + 1] = elmdist[p] + kprocs(p);

  // ptr to eind
  hf_array<idx_t> eptr(num_cells + 1);
  eptr[0] = 0;
  for (int i = 0; i < num_cells; i++)
  {
    if (ctype(i) == 0)
      eptr[i + 1] = eptr[i] + 3;
    else if (ctype(i) == 1)
      eptr[i + 1] = eptr[i] + 4;
    else if (ctype(i) == 2)
      eptr[i + 1] = eptr[i] + 4;
    else if (ctype(i) == 3)
      eptr[i + 1] = eptr[i] + 6;
    else if (ctype(i) == 4)
      eptr[i + 1] = eptr[i] + 8;
    else
      cout << "unknown element type, in repartitioning" << endl;
  }

  // local element to vertex
  hf_array<idx_t> eind(eptr[num_cells]);
  int sk = 0;
  for (int i = 0; i < num_cells; i++) //for each element
  {
    for (int j = 0; j < eptr[i + 1] - eptr[i]; j++) //for each vertex of the element
    {
      eind[sk++] = get_corner_vert_in_order(i, j);
    }
  }

  //weight per element

  hf_array<idx_t> elmwgt(num_cells);
  for (int i = 0; i < num_cells; i++)
  {
    if (ctype(i) == 0)
      elmwgt[i] = 1;
    else if (ctype(i) == 1)
      elmwgt[i] = 1;
    else if (ctype(i) == 2)
      elmwgt[i] = 1;
    else if (ctype(i) == 3)
      elmwgt[i] = 1;
    else if (ctype(i) == 4)
      elmwgt[i] = 1;
  }

  idx_t wgtflag = 0;
  idx_t numflag = 0;
  idx_t ncon = 1;

  idx_t ncommonnodes;

  if (n_dims == 2)
    ncommonnodes = 2;
  else if (n_dims == 3)
    ncommonnodes = 3;

  idx_t nparts = nproc;

  hf_array<real_t> tpwgts(nparts);
  for (int i = 0; i < nparts; i++)
    tpwgts[i] = 1. / (double)nproc;

  hf_array<real_t> ubvec(ncon);
  for (int i = 0; i < ncon; i++)
    ubvec[i] = 1.05;

  idx_t options[10];

  options[0] = 1;
  options[1] = 7;
  options[2] = 0;

  MPI_Comm comm;
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);

  idx_t edgecut;
  hf_array<idx_t> part(num_cells); //array contain the rank number each local call belongs to

  if (rank == 0)
    cout << "Before parmetis" << endl;

  ParMETIS_V3_PartMeshKway(elmdist.get_ptr_cpu(),
                           eptr.get_ptr_cpu(),
                           eind.get_ptr_cpu(),
                           elmwgt.get_ptr_cpu(),
                           &wgtflag,
                           &numflag,
                           &ncon,
                           &ncommonnodes,
                           &nparts,
                           tpwgts.get_ptr_cpu(),
                           ubvec.get_ptr_cpu(),
                           options,
                           &edgecut,
                           part.get_ptr_cpu(),
                           &comm);

  if (rank == 0)
    cout << "After parmetis " << endl;

  // Now create sending buffer
  hf_array<hf_array<int> > outlist(nproc);        //c2v send to each processor
  hf_array<hf_array<int> > outlist_c2n_v(nproc);  //c2nv send to each processor
  hf_array<hf_array<int> > outlist_ctype(nproc);  //ctype send to each processor
  hf_array<hf_array<int> > outlist_ic2icg(nproc); //ic2icg send to each processor

  hf_array<int> outK(nproc); //on this processor, number of elements need to send to each processor
  outK.initialize_to_zero();
  for (int i = 0; i < num_cells; i++)
    ++outK[part[i]];

  hf_array<int> inK(nproc); //number of elements to receive from each processor

  MPI_Alltoall(outK.get_ptr_cpu(), 1, MPI_INT, inK.get_ptr_cpu(), 1, MPI_INT, MPI_COMM_WORLD);

  hf_array<int> newkprocs(nproc); //total number of element on each processor
  MPI_Allreduce(outK.get_ptr_cpu(), newkprocs.get_ptr_cpu(), nproc, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  int totalinK = newkprocs[rank]; //total number of element belong to this processor

  //number of reveive/send requests
  int num_inrequests = 0;
  int num_outrequests = 0;
  for (int i = 0; i < nproc; i++)
  {
    if (inK[i] != 0)
      num_inrequests++;
    if (outK[i] != 0)
      num_outrequests++;
  }

  // declare new hf_array c2v
  hf_array<int> new_c2v(MAX_V_PER_C, totalinK);

  // declare new c2n_v,ctype
  hf_array<int> new_c2n_v(totalinK);
  hf_array<int> new_ctype(totalinK);
  hf_array<int> new_ic2icg(totalinK);

  //declare mpi requests and status
  hf_array<MPI_Request> inrequests(num_inrequests);
  hf_array<MPI_Request> inrequests_c2n_v(num_inrequests);
  hf_array<MPI_Request> inrequests_ctype(num_inrequests);
  hf_array<MPI_Request> inrequests_ic2icg(num_inrequests);

  hf_array<MPI_Request> outrequests(num_outrequests);
  hf_array<MPI_Request> outrequests_c2n_v(num_outrequests);
  hf_array<MPI_Request> outrequests_ctype(num_outrequests);
  hf_array<MPI_Request> outrequests_ic2icg(num_outrequests);

  hf_array<MPI_Status> instatus(num_inrequests);
  hf_array<MPI_Status> outstatus(num_outrequests);

  // Make exchange for arrays c2v,c2n_v,ctype,ic2icg

  int cnt = 0;
  int inrequest_counter = 0;
  for (int p = 0; p < nproc; p++) //for each processor, ensure ascending order of ic2icg
  {
    if (inK[p] != 0) //if have data to receive
    {
      MPI_Irecv(new_c2v.get_ptr_cpu(0, cnt), MAX_V_PER_C * inK[p], MPI_INT, p, p, MPI_COMM_WORLD, inrequests.get_ptr_cpu()+inrequest_counter);
      MPI_Irecv(new_c2n_v.get_ptr_cpu(cnt), inK[p], MPI_INT, p, MAX_PROCESSOR_AVAILABLE + p, MPI_COMM_WORLD, inrequests_c2n_v.get_ptr_cpu()+inrequest_counter);
      MPI_Irecv(new_ctype.get_ptr_cpu(cnt), inK[p], MPI_INT, p, 2 * MAX_PROCESSOR_AVAILABLE + p, MPI_COMM_WORLD, inrequests_ctype.get_ptr_cpu()+inrequest_counter);
      MPI_Irecv(new_ic2icg.get_ptr_cpu(cnt), inK[p], MPI_INT, p, 3 * MAX_PROCESSOR_AVAILABLE + p, MPI_COMM_WORLD, inrequests_ic2icg.get_ptr_cpu()+inrequest_counter);
      cnt = cnt + inK[p];
      inrequest_counter++;
    }
  }

  int outrequest_counter = 0;
  for (int p = 0; p < nproc; p++)
  {
    if (outK[p] != 0) //if this processor have elements to send to processor p
    {
      cnt = 0;

      outlist[p].setup(MAX_V_PER_C, outK[p]);
      outlist_c2n_v[p].setup(outK[p]);
      outlist_ctype[p].setup(outK[p]);
      outlist_ic2icg[p].setup(outK[p]);

      for (int i = 0; i < num_cells; i++) //loop over all local elements, ensure ascending order of ic2icg
      {
        if (part[i] == p) //if this element send to processor p
        {
          for (int v = 0; v < MAX_V_PER_C; v++)
            outlist[p](v, cnt) = c2v(i, v);

          outlist_c2n_v[p][cnt] = c2n_v(i);
          outlist_ctype[p][cnt] = ctype(i);
          outlist_ic2icg[p][cnt] = ic2icg(i);
          cnt++;
        }
      }
      MPI_Isend(outlist[p].get_ptr_cpu(), MAX_V_PER_C * outK[p], MPI_INT, p, rank, MPI_COMM_WORLD, outrequests.get_ptr_cpu()+outrequest_counter);
      MPI_Isend(outlist_c2n_v[p].get_ptr_cpu(), outK[p], MPI_INT, p, MAX_PROCESSOR_AVAILABLE + rank, MPI_COMM_WORLD, outrequests_c2n_v.get_ptr_cpu()+outrequest_counter);
      MPI_Isend(outlist_ctype[p].get_ptr_cpu(), outK[p], MPI_INT, p, 2 * MAX_PROCESSOR_AVAILABLE + rank, MPI_COMM_WORLD, outrequests_ctype.get_ptr_cpu()+outrequest_counter);
      MPI_Isend(outlist_ic2icg[p].get_ptr_cpu(), outK[p], MPI_INT, p, 3 * MAX_PROCESSOR_AVAILABLE + rank, MPI_COMM_WORLD, outrequests_ic2icg.get_ptr_cpu()+outrequest_counter);
      outrequest_counter++;
    }
  }

  MPI_Waitall(num_inrequests, inrequests.get_ptr_cpu(), instatus.get_ptr_cpu());
  MPI_Waitall(num_inrequests, inrequests_c2n_v.get_ptr_cpu(), instatus.get_ptr_cpu());
  MPI_Waitall(num_inrequests, inrequests_ctype.get_ptr_cpu(), instatus.get_ptr_cpu());
  MPI_Waitall(num_inrequests, inrequests_ic2icg.get_ptr_cpu(), instatus.get_ptr_cpu());

  MPI_Waitall(num_outrequests, outrequests.get_ptr_cpu(), outstatus.get_ptr_cpu());
  MPI_Waitall(num_outrequests, outrequests_c2n_v.get_ptr_cpu(), outstatus.get_ptr_cpu());
  MPI_Waitall(num_outrequests, outrequests_ctype.get_ptr_cpu(), outstatus.get_ptr_cpu());
  MPI_Waitall(num_outrequests, outrequests_ic2icg.get_ptr_cpu(), outstatus.get_ptr_cpu());

  //setup final connectivity arrays
  c2v.setup(totalinK, MAX_V_PER_C);
  c2n_v = new_c2n_v;
  ctype = new_ctype;
  ic2icg = new_ic2icg; //increasing order

  for (int i = 0; i < totalinK; i++)
    for (int j = 0; j < MAX_V_PER_C; j++)
      c2v(i, j) = new_c2v(j, i);

  num_cells = totalinK; //update new local cell number

  MPI_Barrier(MPI_COMM_WORLD);
}
#endif // _MPI

void mesh::create_iv2ivg()
{
  vector<int> vrtlist;
  for (int i = 0; i < num_cells * MAX_V_PER_C; i++)
  {
    if (c2v(i) != -1) //if not -1
      vrtlist.push_back(c2v(i));
  }

  // Sort the vertices
  sort(vrtlist.begin(), vrtlist.end());

  //unique
  std::vector<int>::iterator new_end = unique(vrtlist.begin(), vrtlist.end());
  num_verts = new_end - vrtlist.begin();

  //copy to iv2ivg
  iv2ivg.setup(num_verts);
  copy(vrtlist.begin(), new_end, iv2ivg.get_ptr_cpu());

#ifdef _MPI

  // Now modify c2v using local vertex index
  for (int i = 0; i < num_cells; i++)
  {
    for (int j = 0; j < c2n_v(i); j++)
    {
      int index = index_locate_int(c2v(i, j), iv2ivg.get_ptr_cpu(), num_verts);
      if (index == -1)
      {
        FatalError("Could not find value in index_locate");
      }
      else
      {
        c2v(i, j) = index;
      }
    }
  }

#endif
}

void mesh::set_vertex_connectivity(void)
{
  v2c.setup(num_verts);
  for (int ic = 0; ic < num_cells; ic++) //for each cell
  {
    for (int k = 0; k < c2n_v(ic); k++) //for each vertex
    {
      v2c(c2v(ic, k)).push_back(ic); //ascending order
    }
  }

  v2n_c.setup(num_verts);
  for (int v = 0; v < num_verts; v++)
    v2n_c(v) = (int)v2c(v).size();
}

void mesh::set_face_connectivity(void)
{
  int max_inters = num_cells * MAX_F_PER_C; //place holder for face arrays

  f2c.setup(max_inters, 2);
  f2v.setup(max_inters, MAX_V_PER_F);
  f2nv.setup(max_inters);
  f2loc_f.setup(max_inters, 2);
  c2f.setup(num_cells, MAX_F_PER_C);
  rot_tag.setup(max_inters);
  unmatched_inters.setup(max_inters);

  //initialize to -1 which means unmatched
  f2c.initialize_to_value(-1);
  f2loc_f.initialize_to_value(-1);
  c2f.initialize_to_value(-1);

  hf_array<int> vlist, vlist2;           //list of vertex on a surface
  num_inters = 0;                        //initialize the number of interfaces
  n_unmatched_inters=0;

  for (int ic = 0; ic < num_cells; ic++) // Loop over all the cells
  {
    for (int k = 0; k < num_f_per_c(ctype(ic)); k++) //Loop over all faces of that cell
    {
      if (c2f(ic, k) != -1)
        continue; // we have counted that face already,skip

      // Get vertices of local face k of cell ic,vlist has size of num_v_per_f
      int num_v_per_f = get_corner_vlist_face(ic, k, vlist);

      // loop over the vertex on the face to find out the common cell they share
      vector<int> intersection = v2c(vlist(0));
      std::vector<int>::iterator it_intersect;
      for (int i = 1; i < num_v_per_f; i++)
      {
        it_intersect = set_intersection(v2c(vlist(i)).begin(), v2c(vlist(i)).end(), intersection.begin(), intersection.end(), intersection.begin()); //get the intersection of 2 sorted data
        intersection.resize(it_intersect - intersection.begin());
      }

      std::vector<int>::iterator it;

      if (intersection.size() ==2) //if share 2 cells(self and the other common cell)
      {
        for (it = intersection.begin(); it != intersection.end(); it++) //loop over each cell in the intersection
        {
          if (*it != ic) //skip self
          {
            // Loop over faces of cell *it (which shares face k)
            for (int k2 = 0; k2 < num_f_per_c(ctype(*it)); k2++)
            {
              // Get local vertices of local face k2 of cell ic2
              int num_v_per_f2 = get_corner_vlist_face(*it, k2, vlist2);

              if (num_v_per_f2 == num_v_per_f)
              {
                // Compare the list of vertices
                // For 3D returns the orientation of face2 wrt face1 (rtag)
                int rtag;
                if (compare_faces(vlist, vlist2, rtag)) //if found the face
                {
                  c2f(ic, k) = num_inters;
                  c2f(*it, k2) = num_inters;

                  f2c(num_inters, 0) = ic;
                  f2c(num_inters, 1) = *it;

                  f2loc_f(num_inters, 0) = k;
                  f2loc_f(num_inters, 1) = k2;

                  for (int i = 0; i < num_v_per_f; i++)
                    f2v(num_inters, i) = vlist(i);

                  f2nv(num_inters) = num_v_per_f;
                  rot_tag(num_inters) = rtag;

                  num_inters++;
                  break;
                }
              }
            }
          }
        }
      }
      else if(intersection.size()==1)//cant find other cells, the face is either mpiface or boundary
      {
        f2c(num_inters, 0) = ic;
        f2c(num_inters, 1) = -1; //not coupled

        f2loc_f(num_inters, 0) = k;
        f2loc_f(num_inters, 1) = -1; //not coupled

        c2f(ic, k) = num_inters;

        for (int i = 0; i < num_v_per_f; i++)
          f2v(num_inters, i) = vlist(i);

        f2nv(num_inters) = num_v_per_f;

        unmatched_inters(n_unmatched_inters) = num_inters;
        
        n_unmatched_inters++;
        num_inters++;
      }
      else
      {
        FatalError("More than two cells share one face")
      }
    } // end of loop over k
  }   // end of loop over ic
}

int mesh::get_corner_vert_in_order(const int &in_ic, const int &in_vert)
{
  int in_ctype = ctype(in_ic);
  int in_n_spts = c2n_v(in_ic);
  int out_v;
  if (in_ctype == 0) // Tri
  {
    if (in_n_spts == 3 || in_n_spts == 6)
    {
      out_v = in_vert;
    }
    else
      FatalError("in_nspt not implemented");
  }
  else if (in_ctype == 1) // Quad
  {
    if (is_perfect_square(in_n_spts))
    {
      int n_spts_1d = round(sqrt(in_n_spts));
      if (in_vert == 0)
        out_v = 0;
      else if (in_vert == 1)
        out_v = n_spts_1d - 1;
      else if (in_vert == 2)
        out_v = in_n_spts - 1;
      else if (in_vert == 3)
        out_v = in_n_spts - n_spts_1d;
    }
    else if (in_n_spts == 8)
    {
      out_v = in_vert;
    }
    else
      FatalError("in_nspt not implemented");
  }
  else if (in_ctype == 2) // Tet
  {
    if (in_n_spts == 4 || in_n_spts == 10)
      out_v = in_vert;
    else
      FatalError("in_nspt not implemented");
  }
  else if (in_ctype == 3) // Prism
  {
    if (in_n_spts == 6 || in_n_spts == 15)
      out_v = in_vert;
    else
      FatalError("in_nspt not implemented");
  }
  else if (in_ctype == 4) // Hex
  {
    if (is_perfect_cube(in_n_spts))
    {
      int n_spts_1d = round(pow(in_n_spts, 1. / 3.));
      int shift = n_spts_1d * n_spts_1d * (n_spts_1d - 1);
      if (in_vert == 0)
      {
        out_v = 0;
      }
      else if (in_vert == 1)
      {
        out_v = n_spts_1d - 1;
      }
      else if (in_vert == 2)
      {
        out_v = n_spts_1d * n_spts_1d - 1;
      }
      else if (in_vert == 3)
      {
        out_v = n_spts_1d * (n_spts_1d - 1);
      }
      else if (in_vert == 4)
      {
        out_v = shift;
      }
      else if (in_vert == 5)
      {
        out_v = n_spts_1d - 1 + shift;
      }
      else if (in_vert == 6)
      {
        out_v = in_n_spts - 1;
      }
      else if (in_vert == 7)
      {
        out_v = in_n_spts - n_spts_1d;
      }
    }
    else if (in_n_spts == 20)
    {
      out_v = in_vert;
    }
    else
      FatalError("in_nspt not implemented");
  }
  return c2v(in_ic, out_v);
}

int mesh::get_corner_vlist_face(const int &in_ic, const int &in_face, hf_array<int> &out_vlist) //only corner vertex
{
  int num_v_per_f;
  int in_n_spts = c2n_v(in_ic);
  int in_ctype = ctype(in_ic);

  if (in_ctype == 0) // Triangle
  {
    num_v_per_f = 2;
    out_vlist.setup(num_v_per_f);
    if (in_face == 0)
    {
      out_vlist(0) = 0;
      out_vlist(1) = 1;
    }
    else if (in_face == 1)
    {
      out_vlist(0) = 1;
      out_vlist(1) = 2;
    }
    else if (in_face == 2)
    {
      out_vlist(0) = 2;
      out_vlist(1) = 0;
    }
  }
  else if (in_ctype == 1) // Quad
  {
    num_v_per_f = 2;
    out_vlist.setup(num_v_per_f);
    if (is_perfect_square(in_n_spts))
    {
      int n_spts_1d = round(sqrt(in_n_spts));
      if (in_face == 0)
      {
        out_vlist(0) = 0;
        out_vlist(1) = n_spts_1d - 1;
      }
      else if (in_face == 1)
      {
        out_vlist(0) = n_spts_1d - 1;
        out_vlist(1) = in_n_spts - 1;
      }
      else if (in_face == 2)
      {
        out_vlist(0) = in_n_spts - 1;
        out_vlist(1) = in_n_spts - n_spts_1d;
      }
      else if (in_face == 3)
      {
        out_vlist(0) = in_n_spts - n_spts_1d;
        out_vlist(1) = 0;
      }
    }
    else if (in_n_spts == 8)
    {
      if (in_face == 0)
      {
        out_vlist(0) = 0;
        out_vlist(1) = 1;
      }
      else if (in_face == 1)
      {
        out_vlist(0) = 1;
        out_vlist(1) = 2;
      }
      else if (in_face == 2)
      {
        out_vlist(0) = 2;
        out_vlist(1) = 3;
      }
      else if (in_face == 3)
      {
        out_vlist(0) = 3;
        out_vlist(1) = 0;
      }
    }
    else
    {
      cout << "in_nspts=" << in_n_spts << endl;
      cout << "ctype=" << in_ctype << endl;
      FatalError("in_nspt not implemented");
    }
  }
  else if (in_ctype == 2) // Tet
  {
    num_v_per_f = 3;
    out_vlist.setup(num_v_per_f);
      if (in_face == 0)
      {
        out_vlist(0) = 1;
        out_vlist(1) = 2;
        out_vlist(2) = 3;
      }
      else if (in_face == 1)
      {
        out_vlist(0) = 0;
        out_vlist(1) = 3;
        out_vlist(2) = 2;
      }
      else if (in_face == 2)
      {
        out_vlist(0) = 0;
        out_vlist(1) = 1;
        out_vlist(2) = 3;
      }
      else if (in_face == 3)
      {
        out_vlist(0) = 0;
        out_vlist(1) = 2;
        out_vlist(2) = 1;
      }
  }
  else if (in_ctype == 3) // Prism
  {
    if (in_face == 0)
    {
      num_v_per_f = 3;
      out_vlist.setup(num_v_per_f);
      out_vlist(0) = 0;
      out_vlist(1) = 2;
      out_vlist(2) = 1;
    }
    else if (in_face == 1)
    {
      num_v_per_f = 3;
      out_vlist.setup(num_v_per_f);
      out_vlist(0) = 3;
      out_vlist(1) = 4;
      out_vlist(2) = 5;
    }
    else if (in_face == 2)
    {
      num_v_per_f = 4;
      out_vlist.setup(num_v_per_f);
      out_vlist(0) = 0;
      out_vlist(1) = 1;
      out_vlist(2) = 4;
      out_vlist(3) = 3;
    }
    else if (in_face == 3)
    {
      num_v_per_f = 4;
      out_vlist.setup(num_v_per_f);
      out_vlist(0) = 1;
      out_vlist(1) = 2;
      out_vlist(2) = 5;
      out_vlist(3) = 4;
    }
    else if (in_face == 4)
    {
      num_v_per_f = 4;
      out_vlist.setup(num_v_per_f);
      out_vlist(0) = 2;
      out_vlist(1) = 0;
      out_vlist(2) = 3;
      out_vlist(3) = 5;
    }
  }
  else if (in_ctype == 4) // Hexas
  {
    num_v_per_f = 4;
    out_vlist.setup(num_v_per_f);
    if (is_perfect_cube(in_n_spts))
    {
      int n_spts_1d = round(pow(in_n_spts, 1. / 3.));
      int shift = n_spts_1d * n_spts_1d * (n_spts_1d - 1);
      if (in_face == 0)
      {
        out_vlist(0) = n_spts_1d - 1;
        out_vlist(1) = 0;
        out_vlist(2) = n_spts_1d * (n_spts_1d - 1);
        out_vlist(3) = n_spts_1d * n_spts_1d - 1;
      }
      else if (in_face == 1)
      {
        out_vlist(0) = 0;
        out_vlist(1) = n_spts_1d - 1;
        out_vlist(2) = n_spts_1d - 1 + shift;
        out_vlist(3) = shift;
      }
      else if (in_face == 2)
      {
        out_vlist(0) = n_spts_1d - 1;
        out_vlist(1) = n_spts_1d * n_spts_1d - 1;
        out_vlist(2) = in_n_spts - 1;
        out_vlist(3) = n_spts_1d - 1 + shift;
      }
      else if (in_face == 3)
      {
        out_vlist(0) = n_spts_1d * n_spts_1d - 1;
        out_vlist(1) = n_spts_1d * (n_spts_1d - 1);
        out_vlist(2) = in_n_spts - n_spts_1d;
        out_vlist(3) = in_n_spts - 1;
      }
      else if (in_face == 4)
      {
        out_vlist(0) = n_spts_1d * (n_spts_1d - 1);
        out_vlist(1) = 0;
        out_vlist(2) = shift;
        out_vlist(3) = in_n_spts - n_spts_1d;
      }
      else if (in_face == 5)
      {
        out_vlist(0) = shift;
        out_vlist(1) = n_spts_1d - 1 + shift;
        out_vlist(2) = in_n_spts - 1;
        out_vlist(3) = in_n_spts - n_spts_1d;
      }
    }
    else if (in_n_spts == 20)
    {
      if (in_face == 0)
      {
        out_vlist(0) = 1;
        out_vlist(1) = 0;
        out_vlist(2) = 3;
        out_vlist(3) = 2;
      }
      else if (in_face == 1)
      {
        out_vlist(0) = 0;
        out_vlist(1) = 1;
        out_vlist(2) = 5;
        out_vlist(3) = 4;
      }
      else if (in_face == 2)
      {
        out_vlist(0) = 1;
        out_vlist(1) = 2;
        out_vlist(2) = 6;
        out_vlist(3) = 5;
      }
      else if (in_face == 3)
      {
        out_vlist(0) = 2;
        out_vlist(1) = 3;
        out_vlist(2) = 7;
        out_vlist(3) = 6;
      }
      else if (in_face == 4)
      {
        out_vlist(0) = 3;
        out_vlist(1) = 0;
        out_vlist(2) = 4;
        out_vlist(3) = 7;
      }
      else if (in_face == 5)
      {
        out_vlist(0) = 4;
        out_vlist(1) = 5;
        out_vlist(2) = 6;
        out_vlist(3) = 7;
      }
    }
    else
      FatalError("n_spts not implemented");
  }
  else
  {
    cout << "in_ctype = " << in_ctype << endl;
    FatalError("ERROR: Haven't implemented other 3D Elements yet");
  }
  for (int i = 0; i < num_v_per_f; i++) //get final vertex index
    out_vlist(i) = c2v(in_ic, out_vlist(i));
  return num_v_per_f;
}

int mesh::compare_faces(hf_array<int> &vlist1, hf_array<int> &vlist2, int &rtag)
{
  /* Looking at a face from *inside* the cell, the nodes *must* be numbered in *CW* order
   * (this is in agreement with Gambit; Gmsh does not care about local face numberings)
   *
   * The variable 'rtag' matches up the local node numbers of two overlapping faces from
   * different cells (specifically, it is which node from face 2 matches node 0 from face 1)
   */
  int found;
  int num_v_per_f = vlist1.get_dim(0);
  if (vlist1.get_dim(0) != vlist2.get_dim(0))
    FatalError("Size of vlist1 must equal to vlist2");

  if (num_v_per_f == 2) // edge
  {
    if ((vlist1(0) == vlist2(0) && vlist1(1) == vlist2(1)) ||
        (vlist1(0) == vlist2(1) && vlist1(1) == vlist2(0)))
    {
      found = 1;
      rtag = 0;
    }
    else
      found = 0;
  }
  else if (num_v_per_f == 3) // tri face
  {
    //rot_tag==0
    if (vlist1(0) == vlist2(0) &&
        vlist1(1) == vlist2(2) &&
        vlist1(2) == vlist2(1))
    {
      rtag = 0;
      found = 1;
    }
    //rot_tag==1
    else if (vlist1(0) == vlist2(2) &&
             vlist1(1) == vlist2(1) &&
             vlist1(2) == vlist2(0))
    {
      rtag = 1;
      found = 1;
    }
    //rot_tag==2
    else if (vlist1(0) == vlist2(1) &&
             vlist1(1) == vlist2(0) &&
             vlist1(2) == vlist2(2))
    {
      rtag = 2;
      found = 1;
    }
    else
      found = 0;
  }
  else if (num_v_per_f == 4) // quad face
  {
    //rot_tag==0
    if (vlist1(0) == vlist2(1) &&
        vlist1(1) == vlist2(0) &&
        vlist1(2) == vlist2(3) &&
        vlist1(3) == vlist2(2))
    {
      rtag = 0;
      found = 1;
    }
    //rot_tag==1
    else if (vlist1(0) == vlist2(3) &&
             vlist1(1) == vlist2(2) &&
             vlist1(2) == vlist2(1) &&
             vlist1(3) == vlist2(0))
    {
      rtag = 1;
      found = 1;
    }
    //rot_tag==2
    else if (vlist1(0) == vlist2(0) &&
             vlist1(1) == vlist2(3) &&
             vlist1(2) == vlist2(2) &&
             vlist1(3) == vlist2(1))
    {
      rtag = 2;
      found = 1;
    }
    //rot_tag==3
    else if (vlist1(0) == vlist2(2) &&
             vlist1(1) == vlist2(1) &&
             vlist1(2) == vlist2(0) &&
             vlist1(3) == vlist2(3))
    {
      rtag = 3;
      found = 1;
    }
    else
      found = 0;
  }
  else
  {
    FatalError("ERROR: Haven't implemented this face type in compare_face yet....");
  }
  return found;
}

int mesh::compare_faces_boundary(hf_array<int> &vlist1, hf_array<int> &vlist2)
{

  int found;
  int num_v_per_f = vlist1.get_dim(0);
  if (vlist1.get_dim(0) != vlist2.get_dim(0))
    FatalError("Size of vlist1 must equal to vlist2");
  if (!(num_v_per_f == 2 || num_v_per_f == 3 || num_v_per_f == 4))
    FatalError("Boundary face type not recognized (expecting a linear edge, tri, or quad)");

  int count = 0;
  for (int j = 0; j < num_v_per_f; j++)
  {
    for (int k = 0; k < num_v_per_f; k++)
    {
      if (vlist1(j) == vlist2(k))
      {
        count++;
        break;
      }
    }
  }

  if (count == num_v_per_f)
    found = 1;
  else
    found = 0;
  return found;
}
