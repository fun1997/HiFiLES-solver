/*!
 * \file output.cpp
 * \author - Original code: SD++ developed by Patrice Castonguay, Antony Jameson,
 *                          Peter Vincent, David Williams (alphabetical by surname).
 *         - Current development: Aerospace Computing Laboratory (ACL)
 *                                Aero/Astro Department. Stanford University.
 * \version 0.1.0
 *
 * High Fidelity Large Eddy Simulation (HiFiLES) Code.
 * Copyright (C) 2014 Aerospace Computing Laboratory (ACL).
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

// Used for making sub-directories
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../include/global.h"
#include "../include/output.h"
#include "../include/hf_array.h"
#include "../include/geometry.h"
#include "../include/solver.h"
#include "../include/funcs.h"
#include "../include/error.h"

#ifdef _TECIO
#include "TECIO.h"
#endif

#ifdef _GPU
#include "../include/util.h"
#endif

using namespace std;

// #### constructors ####

// default constructor

output::output(struct solution *in_sol)
{
  FlowSol = in_sol;
  if (run_input.write_type == 2) //initialize cgns parameter
    setup_CGNS();
}

output::~output() { }


void output::setup_CGNS(void)
{
  #ifdef _CGNS
  hf_array<int> npele_list_local(FlowSol->n_ele_types);//local array for number of plot element per type
  hf_array<int>npele_list(FlowSol->n_ele_types,FlowSol->nproc);//global array for number of plot element per type per processor
  int n_ppts_per_ele,n_peles_per_ele;
  int p_res=run_input.p_res;

//initialize typewise local plot element start/end index
  pele_start.setup(FlowSol->n_ele_types);
  pele_end.setup(FlowSol->n_ele_types);
  pele_start.initialize_to_zero();
  pele_end.initialize_to_zero();
  pnode_start = 1;//local start index of node

  //initialize global element number
  glob_npeles = 0;//global number of plot elements
  glob_npnodes = 0;//global number of nodes

  sum_npele.setup(FlowSol->n_ele_types);//global typewise number of plot elements
  sum_npele.initialize_to_zero();

  //store local number of plot elements
  for (int i = 0; i < FlowSol->n_ele_types; i++)
  {
    if (FlowSol->mesh_eles(i)->get_n_eles())
      npele_list_local(i) = FlowSol->mesh_eles(i)->get_n_eles() *  FlowSol->mesh_eles(i)->get_n_peles_per_ele();
    else
      npele_list_local(i) = 0;
  }

//gather infomation from other processors
#ifdef _MPI
  MPI_Allgather(npele_list_local.get_ptr_cpu(), FlowSol->n_ele_types, MPI_INT, npele_list.get_ptr_cpu(), FlowSol->n_ele_types, MPI_INT, MPI_COMM_WORLD);
#else
  npele_list = npele_list_local;
#endif // _MPI

  //calculate global number of plot elements and nodes, typewise number of plot elements and nodes before this rank
  for (int j = 0; j < FlowSol->nproc; j++)//for each processor
  {
    for (int i = 0; i < FlowSol->n_ele_types; i++)//for each type
    {
      if (npele_list(i, j))//if have element
      {
        switch (i) //calc n_ppts and n_peles per element
        {
        case 0:
          n_ppts_per_ele = (p_res + 1) * p_res / 2;
          n_peles_per_ele = (p_res - 1) * (p_res - 1);
          break;
        case 1:
          n_ppts_per_ele = p_res * p_res;
          n_peles_per_ele = (p_res - 1) * (p_res - 1);
          break;
        case 2:
          n_ppts_per_ele = (p_res + 2) * (p_res + 1) * p_res / 6;
          n_peles_per_ele = (p_res - 1) * (p_res) * (p_res + 1) / 6 + 4 * (p_res - 2) * (p_res - 1) * (p_res) / 6 + (p_res - 3) * (p_res - 2) * (p_res - 1) / 6;
          break;
        case 3:
          n_ppts_per_ele = (p_res + 1) * (p_res) * (p_res) / 2;
          n_peles_per_ele = ((p_res - 1) * (p_res - 1) * (p_res - 1));
          break;
        case 4:
          n_ppts_per_ele = p_res * p_res * p_res;
          n_peles_per_ele = (p_res - 1) * (p_res - 1) * (p_res - 1);
          break;
        }

        glob_npeles += npele_list(i, j);
        glob_npnodes += npele_list(i, j) * n_ppts_per_ele / n_peles_per_ele;
        sum_npele(i) += npele_list(i, j); //global number of each type of element
      }

      if (j == FlowSol->rank - 1)
      {
        pele_start(i) = sum_npele(i); //set local typewise plot element start index despite the existence of other type of element
      }
      if (j == FlowSol->rank)
        pele_end(i) = sum_npele(i); //set local typewise plot element end index despite the existence of other type of element
    }
    if (j == FlowSol->rank - 1)
      pnode_start += glob_npnodes; //set start index for local nodes
  }

  //transform to local plot element start index, order by element type
  for (int i = 1; i < FlowSol->n_ele_types; i++) //start from quad to hex
    for (int j = 0; j < i; j++)//loop over former types
    {
      pele_start(i) += sum_npele(j);
      pele_end(i) += sum_npele(j);
    }
  //start index start form 1
  for (int i = 0; i < FlowSol->n_ele_types; i++)
    pele_start(i)++;

#endif // _CGNS
}

// method to write out a tecplot file
void output::write_tec(int in_file_num)
{
  int i,j,k,l,m;

  hf_array<double> pos_ppts_temp;
  hf_array<double> disu_ppts_temp;
  hf_array<double> grad_disu_ppts_temp;
  hf_array<double> disu_average_ppts_temp;
  hf_array<double> diag_ppts_temp;

  /*! Sensor data for shock capturing at plot points */
  hf_array<double> sensor_ppts_temp;

  /*! Plot sub-element connectivity hf_array (node IDs) */
  hf_array<int> con;

  int n_ppts_per_ele;
  int n_dims = FlowSol->n_dims;
  int n_fields;
  int n_diag_fields;
  int n_average_fields;
  int num_pts, num_elements,n_peles_per_ele,n_verts_per_ele;

  char  file_name_s[256];
  char  dumpnum_s[256];
  char *file_name;
  string fields("");

  ofstream write_tec;
  write_tec.precision(15);

  // number of additional diagnostic fields
  n_diag_fields = run_input.n_diagnostic_fields;

  // number of additional time-averaged diagnostic fields
  n_average_fields = run_input.n_average_fields;

#ifdef _MPI
//create a folder if there're more than one processors
  if(FlowSol->nproc!=1)
  {
    sprintf(dumpnum_s,"%s_%.09d",run_input.data_file_name.c_str(),in_file_num);//folder name
    sprintf(file_name_s,"%s_%.09d/%s_%.09d_p%.04d.plt",run_input.data_file_name.c_str(),in_file_num,run_input.data_file_name.c_str(),in_file_num,FlowSol->rank);//file name
      /*! Master node creates a subdirectory to store .plt files */
    if (FlowSol->rank == 0) 
      {
        struct stat st = {0};
        if (stat(dumpnum_s, &st) == -1) {
            mkdir(dumpnum_s, 0755);
          }
      }
  }
  else
      sprintf(file_name_s,"%s_%.09d_p%.04d.plt",run_input.data_file_name.c_str(),in_file_num,FlowSol->rank);
  MPI_Barrier(MPI_COMM_WORLD);
  if (FlowSol->rank==0) cout << "Writing Tecplot file number " << in_file_num << " ...." << flush;
#else
  sprintf(file_name_s,"%s_%.09d_p%.04d.plt",run_input.data_file_name.c_str(),in_file_num,0);
  cout << "Writing Tecplot file number " << in_file_num << " ...." << flush;
#endif

  file_name = &file_name_s[0];
  write_tec.open(file_name);

  // write header
  write_tec << "Title = \"HiFiLES Solution\"" << endl;

  // string of field names
  if (run_input.equation==0)
    {
      if(n_dims==2)
        {
          fields += "Variables = \"x\", \"y\", \"rho\", \"mom_x\", \"mom_y\", \"ene\"";
        }
      else if(n_dims==3)
        {
          fields += "Variables = \"x\", \"y\", \"z\", \"rho\", \"mom_x\", \"mom_y\", \"mom_z\", \"ene\"";
        }
      else
        {
          cout << "ERROR: Invalid number of dimensions ... " << endl;
        }
    }
  else if (run_input.equation==1)
    {
      if(n_dims==2)
        {
          fields += "Variables = \"x\", \"y\", \"rho\"";
        }
      else if(n_dims==3)
        {
          fields += "Variables = \"x\", \"y\", \"z\", \"rho\"";
        }
      else
        {
          cout << "ERROR: Invalid number of dimensions ... " << endl;
        }
    }

  if (run_input.turb_model==1) {
    fields += ", \"mu_tilde\"";
  }

  // append the names of the time-average diagnostic fields
  if(n_average_fields>0)
    {
      fields += ", ";

      for(m=0;m<n_average_fields;m++)
        fields += "\"" + run_input.average_fields(m) + "\", ";

    }

  // append the names of the diagnostic fields
  if(n_diag_fields>0)
    {
      fields += ", ";

      for(m=0;m<n_diag_fields-1;m++)
        fields += "\"" + run_input.diagnostic_fields(m) + "\", ";

      fields += "\"" + run_input.diagnostic_fields(n_diag_fields-1) + "\"";
    }

  // write field names to file
  write_tec << fields << endl;

  int time_iter = 0;

  for(i=0;i<FlowSol->n_ele_types;i++)
    {
      if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {

          n_fields = FlowSol->mesh_eles(i)->get_n_fields();
          n_ppts_per_ele = FlowSol->mesh_eles(i)->get_n_ppts_per_ele();
          num_pts = (FlowSol->mesh_eles(i)->get_n_eles())*n_ppts_per_ele;
          num_elements = (FlowSol->mesh_eles(i)->get_n_eles())*(FlowSol->mesh_eles(i)->get_n_peles_per_ele());
          n_peles_per_ele=FlowSol->mesh_eles(i)->get_n_peles_per_ele();
          n_verts_per_ele = FlowSol->mesh_eles(i)->get_n_verts_per_ele();
          pos_ppts_temp.setup(n_ppts_per_ele,n_dims);
          disu_ppts_temp.setup(n_ppts_per_ele,n_fields);
          disu_ppts_temp.initialize_to_zero();
          grad_disu_ppts_temp.setup(n_ppts_per_ele,n_fields,n_dims);
          grad_disu_ppts_temp.initialize_to_zero();
          diag_ppts_temp.setup(n_ppts_per_ele,n_diag_fields);
          disu_average_ppts_temp.setup(n_ppts_per_ele,n_average_fields);
          disu_average_ppts_temp.initialize_to_zero();
          /*! Temporary field for sensor hf_array at plot points */
          sensor_ppts_temp.setup(n_ppts_per_ele);


          // write element specific header
          if(FlowSol->mesh_eles(i)->get_ele_type()==0) // tri
            {
              write_tec << "ZONE N = " << num_pts << ", E = " << num_elements << ", DATAPACKING = POINT, ZONETYPE = FETRIANGLE" << endl;
            }
          else if(FlowSol->mesh_eles(i)->get_ele_type()==1) // quad
            {
              write_tec << "ZONE N = " << num_pts << ", E = " << num_elements << ", DATAPACKING = POINT, ZONETYPE = FEQUADRILATERAL" << endl;
            }
          else if (FlowSol->mesh_eles(i)->get_ele_type()==2) // tet
            {
              write_tec << "ZONE N = " << num_pts << ", E = " << num_elements << ", DATAPACKING = POINT, ZONETYPE = FETETRAHEDRON" << endl;
            }
          else if (FlowSol->mesh_eles(i)->get_ele_type()==3) // prisms
            {
              write_tec << "ZONE N = " << num_pts << ", E = " << num_elements << ", DATAPACKING = POINT, ZONETYPE = FEBRICK" << endl;
            }
          else if(FlowSol->mesh_eles(i)->get_ele_type()==4) // hexa
            {
              write_tec << "ZONE N = " << num_pts << ", E = " << num_elements << ", DATAPACKING = POINT, ZONETYPE = FEBRICK" << endl;
            }
          else
            {
              FatalError("Invalid element type");
            }

          if(time_iter == 0)
            {
              write_tec <<"SolutionTime=" << FlowSol->time << endl;
              time_iter = 1;
            }

          // write element specific data

          for(j=0;j<FlowSol->mesh_eles(i)->get_n_eles();j++)
            {
              FlowSol->mesh_eles(i)->calc_pos_ppts(j,pos_ppts_temp);
              FlowSol->mesh_eles(i)->calc_disu_ppts(j,disu_ppts_temp);
              FlowSol->mesh_eles(i)->calc_grad_disu_ppts(j,grad_disu_ppts_temp);

              if(run_input.shock_cap)
              {
                /*! Calculate the sensor at the plot points */
                FlowSol->mesh_eles(i)->calc_sensor_ppts(j,sensor_ppts_temp);
              }

              /*! Calculate the time averaged fields at the plot points */
              if(n_average_fields > 0)
                {
                  FlowSol->mesh_eles(i)->calc_time_average_ppts(j,disu_average_ppts_temp);
                }

              /*! Calculate the diagnostic fields at the plot points */
              if(n_diag_fields > 0)
                {
                  FlowSol->mesh_eles(i)->calc_diagnostic_fields_ppts(j, disu_ppts_temp, grad_disu_ppts_temp, sensor_ppts_temp, diag_ppts_temp, FlowSol->time);
                }

              for(k=0;k<n_ppts_per_ele;k++)
                {
                  for(l=0;l<n_dims;l++)
                    {
                      write_tec << pos_ppts_temp(k,l) << " ";
                    }

                  for(l=0;l<n_fields;l++)
                    {
                      if ( isnan(disu_ppts_temp(k,l))) {
                          FatalError("Nan in tecplot file, exiting");
                        }
                      else {
                          write_tec << disu_ppts_temp(k,l) << " ";
                        }
                    }

                  /*! Write out optional time-averaged diagnostic fields */
                  for(l=0;l<n_average_fields;l++)
                    {
                      if ( isnan(disu_average_ppts_temp(k,l))) {
                          FatalError("Nan in tecplot file, exiting");
                        }
                      else {
                          write_tec << disu_average_ppts_temp(k,l) << " ";
                        }
                    }

                  /*! Write out optional diagnostic fields */
                  for(l=0;l<n_diag_fields;l++)
                    {
                      if ( isnan(diag_ppts_temp(k,l))) {
                          FatalError("Nan in tecplot file, exiting");
                        }
                      else {
                          write_tec << diag_ppts_temp(k,l) << " ";
                        }
                    }

                  write_tec << endl;
                }
            }

            // write element specific connectivity
            con = FlowSol->mesh_eles(i)->get_connectivity_plot();
            for (int j = 0; j < FlowSol->mesh_eles(i)->get_n_eles(); j++)
            {
              for (int k = 0; k < n_peles_per_ele; k++)
              {
                for (int l = 0; l < n_verts_per_ele; l++)
                {
                  write_tec << con(l, k) + j * n_ppts_per_ele + 1;
                  if (l != n_verts_per_ele - 1)
                    write_tec << " ";
                }
                write_tec << endl;
              }
            }
        }
    }

  write_tec.close();

#ifdef _MPI
  MPI_Barrier(MPI_COMM_WORLD);
  if (FlowSol->rank==0) cout << "done." << endl;
#else
  cout << "done." << endl;
#endif

}

/*! Method to write out a Paraview .vtu file.
Used in run mode.
input: in_file_num																						current timestep
input: FlowSol																								solution structure
output: Mesh_<in_file_num>.vtu																(serial) data file
output: Mesh_<in_file_num>/Mesh_<in_file_num>_<rank>.vtu			(parallel) data file containing portion of domain owned by current node. Files contained in directory Mesh_<in_file_num>.
output: Mesh_<in_file_num>.pvtu																(parallel) file stitching together all .vtu files (written by master node)
*/

void output::write_vtu(int in_file_num)
{
  int i,j,k,l,m;
  /*! Current rank */
  int my_rank = 0;
  /*! No. of processes */
  int n_proc = 1;
  /*! No. of solution fields */
  int n_fields;
  /*! No. of optional diagnostic fields */
  int n_diag_fields;
  /*! No. of optional time-averaged diagnostic fields */
  int n_average_fields;
  /*! No. of dimensions */
  int n_dims;
  /*! No. of elements */
  int n_eles;
  /*! Number of plot points in element */
  int n_points;
  /*! Number of plot sub-elements in element */
  int n_cells;
  /*! No. of vertices per element */
  int n_verts;
  /*! Element type */
  int ele_type;

  /*! Plot point coordinates */
  hf_array<double> pos_ppts_temp;
  /*! Solution data at plot points */
  hf_array<double> disu_ppts_temp;
  /*! Solution gradient data at plot points */
  hf_array<double> grad_disu_ppts_temp;
  /*! Diagnostic field data at plot points */
  hf_array<double> diag_ppts_temp;
  /*! Time-averaged diagnostic field data at plot points */
  hf_array<double> disu_average_ppts_temp;
  /*! Grid velocity at plot points */
  hf_array<double> grid_vel_ppts_temp;
  /*! Sensor data for shock capturing at plot points */
  hf_array<double> sensor_ppts_temp;

  /*! Plot sub-element connectivity hf_array (node IDs) */
  hf_array<int> con;

  /*! VTK element types (different to HiFiLES element type) */
  /*! tri, quad, tet, prism , hex */
  /*! See vtkCellType.h for full list */
  int vtktypes[5] = {5,9,10,13,12};

  /*! File names */
  char vtu_s[256];
  char dumpnum_s[256];
  char pvtu_s[256];
  /*! File name pointers needed for opening files */
  char *vtu;
  char *pvtu;
  char *dumpnum;

  /*! Output files */
  ofstream write_vtu;
  write_vtu.precision(15);
  ofstream write_pvtu;
  write_pvtu.precision(15);

  /*! no. of optional diagnostic fields */
  n_diag_fields = run_input.n_diagnostic_fields;

  /*! no. of optional time-averaged diagnostic fields */
  n_average_fields = run_input.n_average_fields;

#ifdef _MPI

  /*! Get rank of each process */
  my_rank = FlowSol->rank;
  n_proc   = FlowSol->nproc;
  /*! Dump number */
  sprintf(dumpnum_s,"%s_%.09d",run_input.data_file_name.c_str(),in_file_num);
  /*! Each rank writes a .vtu file in a subdirectory named 'dumpnum_s' created by master process */
  sprintf(vtu_s,"%s_%.09d/%s_%.09d_%d.vtu",run_input.data_file_name.c_str(),in_file_num,run_input.data_file_name.c_str(),in_file_num,my_rank);
  /*! On rank 0, write a .pvtu file to gather data from all .vtu files */
  sprintf(pvtu_s,"%s_%.09d.pvtu",run_input.data_file_name.c_str(),in_file_num);

#else

  /*! Only write a vtu file in serial */
  sprintf(dumpnum_s,"%s_%.09d",run_input.data_file_name.c_str(),in_file_num);
  sprintf(vtu_s,"%s_%.09d.vtu",run_input.data_file_name.c_str(),in_file_num);

#endif

  /*! Point to names */
  vtu = &vtu_s[0];
  pvtu = &pvtu_s[0];
  dumpnum = &dumpnum_s[0];

#ifdef _MPI

  /*! Master node creates a subdirectory to store .vtu files */
  if (my_rank == 0) {
      struct stat st = {0};
      if (stat(dumpnum, &st) == -1) {
          mkdir(dumpnum, 0755);
        }
      /*! Delete old .vtu files from directory */
      //remove(strcat(dumpnum,"/*.vtu"));
    }

  /*! Master node writes the .pvtu file */
  if (my_rank == 0) {
      cout << "Writing Paraview file " << dumpnum << " ...." << flush;

      write_pvtu.open(pvtu);
      write_pvtu << "<?xml version=\"1.0\" ?>" << endl;
      write_pvtu << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\" compressor=\"vtkZLibDataCompressor\">" << endl;
      write_pvtu << "	<PUnstructuredGrid GhostLevel=\"1\">" << endl;

      /*! Write point data */
      write_pvtu << "		<PPointData Scalars=\"Density\" Vectors=\"Velocity\">" << endl;
      write_pvtu << "			<PDataArray type=\"Float32\" Name=\"Density\" />" << endl;
      write_pvtu << "			<PDataArray type=\"Float32\" Name=\"Velocity\" NumberOfComponents=\"3\" />" << endl;
      write_pvtu << "			<PDataArray type=\"Float32\" Name=\"Energy\" />" << endl;

      /*! write out modified turbulent viscosity */
      if (run_input.turb_model==1) {
        write_pvtu << "			<PDataArray type=\"Float32\" Name=\"Mu_Tilde\" />" << endl;
      }

      // Optional time-averaged diagnostic fields
      for(m=0;m<n_average_fields;m++)
        {
          write_pvtu << "			<PDataArray type=\"Float32\" Name=\"" << run_input.average_fields(m) << "\" />" << endl;
        }

      // Optional diagnostic fields
      for(m=0;m<n_diag_fields;m++)
        {
          write_pvtu << "			<PDataArray type=\"Float32\" Name=\"" << run_input.diagnostic_fields(m) << "\" />" << endl;
        }

      write_pvtu << "		</PPointData>" << endl;

      /*! Write points */
      write_pvtu << "		<PPoints>" << endl;
      write_pvtu << "			<PDataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" />" << endl;
      write_pvtu << "		</PPoints>" << endl;

      /*! Write names of source .vtu files to include */
      for (i=0;i<n_proc;++i) {
          write_pvtu << "		<Piece Source=\"" << dumpnum << "/" << dumpnum <<"_" << i << ".vtu" << "\" />" << endl;
        }

      /*! Write footer */
      write_pvtu << "	</PUnstructuredGrid>" << endl;
      write_pvtu << "</VTKFile>" << endl;
      write_pvtu.close();
    }

#else

  /*! In serial, don't write a .pvtu file. */
  cout << "Writing Paraview file " << dumpnum << " ... " << flush;

#endif

#ifdef _MPI

  /*! Wait for all processes to get to this point, otherwise there won't be a directory to put .vtus into */
  MPI_Barrier(MPI_COMM_WORLD);

#endif

  /*! Each process writes its own .vtu file */
  write_vtu.open(vtu);
  /*! File header */
  write_vtu << "<?xml version=\"1.0\" ?>" << endl;
  write_vtu << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\" compressor=\"vtkZLibDataCompressor\">" << endl;
  write_vtu << "	<UnstructuredGrid>" << endl;

  /*! Loop over element types */
  for(i=0;i<FlowSol->n_ele_types;i++)
    {
      /*! no. of elements of type i */
      n_eles = FlowSol->mesh_eles(i)->get_n_eles();
      /*! Only proceed if there any elements of type i */
      if (n_eles!=0) {
          /*! element type */
          ele_type = FlowSol->mesh_eles(i)->get_ele_type();

          /*! no. of plot points per ele */
          n_points = FlowSol->mesh_eles(i)->get_n_ppts_per_ele();

          /*! no. of plot sub-elements per ele */
          n_cells  = FlowSol->mesh_eles(i)->get_n_peles_per_ele();

          /*! no. of vertices per ele */
          n_verts  = FlowSol->mesh_eles(i)->get_n_verts_per_ele();

          /*! no. of solution fields */
          n_fields = FlowSol->mesh_eles(i)->get_n_fields();

          /*! no. of dimensions */
          n_dims = FlowSol->mesh_eles(i)->get_n_dims();

          /*! Temporary hf_array of plot point coordinates */
          pos_ppts_temp.setup(n_points,n_dims);

          /*! Temporary solution hf_array at plot points */
          disu_ppts_temp.setup(n_points,n_fields);
          disu_ppts_temp.initialize_to_zero();
          /*! Temporary hf_array of time averaged fields at the plot points */
          if(n_average_fields > 0) {
            disu_average_ppts_temp.setup(n_points,n_average_fields);
            disu_average_ppts_temp.initialize_to_zero();
          }

          if(n_diag_fields > 0) {
            /*! Temporary solution hf_array at plot points */
            grad_disu_ppts_temp.setup(n_points,n_fields,n_dims);
            grad_disu_ppts_temp.initialize_to_zero();
            /*! Temporary diagnostic field hf_array at plot points */
            diag_ppts_temp.setup(n_points,n_diag_fields);

            /*! Temporary field for sensor hf_array at plot points */
            sensor_ppts_temp.setup(n_points);

          }

          /*! Temporary grid velocity hf_array at plot points */
          if (run_input.motion) {
            FlowSol->mesh_eles(i)->set_grid_vel_ppts();
            grid_vel_ppts_temp = FlowSol->mesh_eles(i)->get_grid_vel_ppts();
          }

          con.setup(n_verts,n_cells);
          con = FlowSol->mesh_eles(i)->get_connectivity_plot();

          /*! Loop over individual elements and write their data as a separate VTK DataArray */
          for(j=0;j<n_eles;j++)
            {
              write_vtu << "		<Piece NumberOfPoints=\"" << n_points << "\" NumberOfCells=\"" << n_cells << "\">" << endl;

              /*! Calculate the prognostic (solution) fields at the plot points */
              FlowSol->mesh_eles(i)->calc_disu_ppts(j,disu_ppts_temp);


              /*! Calculate time averaged diagnostic fields at the plot points */
              if(n_average_fields > 0) {
                FlowSol->mesh_eles(i)->calc_time_average_ppts(j,disu_average_ppts_temp);
              }

              if(n_diag_fields > 0) {
                /*! Calculate the gradient of the prognostic fields at the plot points */
                FlowSol->mesh_eles(i)->calc_grad_disu_ppts(j,grad_disu_ppts_temp);

                if(run_input.shock_cap)
                {
                  /*! Calculate the sensor at the plot points */
                  FlowSol->mesh_eles(i)->calc_sensor_ppts(j,sensor_ppts_temp);
                }

                /*! Calculate the diagnostic fields at the plot points */
                FlowSol->mesh_eles(i)->calc_diagnostic_fields_ppts(j, disu_ppts_temp, grad_disu_ppts_temp, sensor_ppts_temp, diag_ppts_temp, FlowSol->time);
              }

              /*! write out solution to file */
              write_vtu << "			<PointData>" << endl;

              /*! density */
              write_vtu << "				<DataArray type= \"Float32\" Name=\"Density\" format=\"ascii\">" << endl;
              for(k=0;k<n_points;k++)
                {
                  write_vtu << disu_ppts_temp(k,0) << " ";
                }
              write_vtu << endl;
              write_vtu << "				</DataArray>" << endl;

              /*! velocity */
              write_vtu << "				<DataArray type= \"Float32\" NumberOfComponents=\"3\" Name=\"Velocity\" format=\"ascii\">" << endl;
              for(k=0;k<n_points;k++)
                {
                  /*! Divide momentum components by density to obtain velocity components */
                  write_vtu << disu_ppts_temp(k,1)/disu_ppts_temp(k,0) << " " << disu_ppts_temp(k,2)/disu_ppts_temp(k,0) << " ";

                  /*! In 2D the z-component of velocity is not stored, but Paraview needs it so write a 0. */
                  if(n_dims==2)
                    {
                      write_vtu << 0.0 << " ";
                    }
                  /*! In 3D just write the z-component of velocity */
                  else
                    {
                      write_vtu << disu_ppts_temp(k,3)/disu_ppts_temp(k,0) << " ";
                    }
                }
              write_vtu << endl;
              write_vtu << "				</DataArray>" << endl;

              /*! energy */
              write_vtu << "				<DataArray type= \"Float32\" Name=\"Energy\" format=\"ascii\">" << endl;
              for(k=0;k<n_points;k++)
                {
                  /*! In 2D energy is the 4th solution component */
                  if(n_dims==2)
                    {
                      write_vtu << disu_ppts_temp(k,3)/disu_ppts_temp(k,0) << " ";
                    }
                  /*! In 3D energy is the 5th solution component */
                  else
                    {
                      write_vtu << disu_ppts_temp(k,4)/disu_ppts_temp(k,0) << " ";
                    }
                }
              write_vtu << endl;
              write_vtu << "				</DataArray>" << endl;

              /*! modified turbulent viscosity */
              if (run_input.turb_model == 1) {
                write_vtu << "				<DataArray type= \"Float32\" Name=\"Nu_Tilde\" format=\"ascii\">" << endl;
                for(k=0;k<n_points;k++)
                {
                  /*! In 2D nu_tilde is the 5th solution component */
                  if(n_dims==2)
                  {
                    write_vtu << disu_ppts_temp(k,4)/disu_ppts_temp(k,0) << " ";
                  }
                  /*! In 3D nu_tilde is the 6th solution component */
                  else
                  {
                    write_vtu << disu_ppts_temp(k,5)/disu_ppts_temp(k,0) << " ";
                  }
                }
                /*! End the line and finish writing DataArray and PointData objects */
                write_vtu << endl;
                write_vtu << "				</DataArray>" << endl;
              }

              if (run_input.motion) {
                /*! grid velocity */
                write_vtu << "				<DataArray type= \"Float32\" NumberOfComponents=\"3\" Name=\"GridVelocity\" format=\"ascii\">" << endl;
                for(k=0;k<n_points;k++)
                {
                  write_vtu << grid_vel_ppts_temp(0,k,j) << " " << grid_vel_ppts_temp(1,k,j) << " ";

                  /*! In 2D the z-component of velocity is not stored, but Paraview needs it so write a 0. */
                  if(n_fields==4)
                  {
                    write_vtu << 0.0 << " ";
                  }
                  /*! In 3D just write the z-component of velocity */
                  else
                  {
                    write_vtu << grid_vel_ppts_temp(2,k,j) << " ";
                  }
                }
                write_vtu << endl;
                write_vtu << "				</DataArray>" << endl;
              }

              /*! Write out optional time-averaged diagnostic fields */
              for(m=0;m<n_average_fields;m++)
                {
                  write_vtu << "				<DataArray type= \"Float32\" Name=\"" << run_input.average_fields(m) << "\" format=\"ascii\">" << endl;
                  for(k=0;k<n_points;k++)
                    {
                      write_vtu << disu_average_ppts_temp(k,m) << " ";
                    }

                  /*! End the line and finish writing DataArray object */
                  write_vtu << endl;
                  write_vtu << "				</DataArray>" << endl;
                }

              /*! Write out optional diagnostic fields */
              for(m=0;m<n_diag_fields;m++)
                {
                  write_vtu << "				<DataArray type= \"Float32\" Name=\"" << run_input.diagnostic_fields(m) << "\" format=\"ascii\">" << endl;
                  for(k=0;k<n_points;k++)
                    {
                      write_vtu << diag_ppts_temp(k,m) << " ";
                    }

                  /*! End the line and finish writing DataArray object */
                  write_vtu << endl;
                  write_vtu << "				</DataArray>" << endl;
                }

              /*! finish writing PointData object */
              write_vtu << "			</PointData>" << endl;

              /*! Calculate the plot coordinates */
              FlowSol->mesh_eles(i)->calc_pos_ppts(j,pos_ppts_temp);

              /*! write out the plot coordinates */
              write_vtu << "			<Points>" << endl;
              write_vtu << "				<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">" << endl;

              /*! Loop over plot points in element */
              for(k=0;k<n_points;k++)
                {
                  for(l=0;l<n_dims;l++)
                    {
                      write_vtu << pos_ppts_temp(k,l) << " ";
                    }

                  /*! If 2D, write a 0 as the z-component */
                  if(n_dims==2)
                    {
                      write_vtu << "0 ";
                    }
                }

              write_vtu << endl;
              write_vtu << "				</DataArray>" << endl;
              write_vtu << "			</Points>" << endl;

              /*! write out Cell data: connectivity, offsets, element types */
              write_vtu << "			<Cells>" << endl;

              /*! Write connectivity hf_array */
              write_vtu << "				<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" << endl;

              for(k=0;k<n_cells;k++)
                {
                  for(l=0;l<n_verts;l++)
                    {
                      write_vtu << con(l,k) << " ";
                    }
                  write_vtu << endl;
                }
              write_vtu << "				</DataArray>" << endl;

              /*! Write cell numbers */
              write_vtu << "				<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" << endl;
              for(k=0;k<n_cells;k++)
                {
                  write_vtu << (k+1)*n_verts << " ";
                }
              write_vtu << endl;
              write_vtu << "				</DataArray>" << endl;

              /*! Write VTK element type */
              write_vtu << "				<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">" << endl;
              for(k=0;k<n_cells;k++)
                {
                  write_vtu << vtktypes[i] << " ";
                }
              write_vtu << endl;
              write_vtu << "				</DataArray>" << endl;

              /*! Write cell and piece footers */
              write_vtu << "			</Cells>" << endl;
              write_vtu << "		</Piece>" << endl;
            }
        }
    }

  /*! Write footer of file */
  write_vtu << "	</UnstructuredGrid>" << endl;
  write_vtu << "</VTKFile>" << endl;

  /*! Close the .vtu file */
  write_vtu.close();

#ifdef _MPI
if(my_rank==0) cout<<"done."<<endl;
#else
  cout << "done." << endl;
#endif
}



void output::write_CGNS(int in_file_num)
{
#ifdef _CGNS
#ifdef _MPI
/*! write parallel CGNS file*/
  int F, B, Z, S, Cx, Cy, Cz;
  int E[5];                                             //element node
  int Fs_rho, Fs_rhou, Fs_rhov, Fs_rhow, Fs_rhoe, Fs_mu, Fs_s ; //field variable
  hf_array<int> Fs_diag, Fs_avg;
  char fname[256];
  cgsize_t sizes[3];
  int temp_ptr, temp_ptr2;
  hf_array<cgsize_t> conn; //connectivity

  //data arrays
  hf_array<double> pos_ppts_temp;
  hf_array<double> disu_ppts_temp;
  hf_array<double> grad_disu_ppts_temp;
  hf_array<double> disu_average_ppts_temp;
  hf_array<double> diag_ppts_temp;
  hf_array<double> sensor_ppts_temp;

  int n_dims = FlowSol->n_dims;
  int n_eles, n_fields, n_ppts_per_ele;
  int n_diag_fields = run_input.n_diagnostic_fields;
  int n_average_fields = run_input.n_average_fields;

  sprintf(fname, "%s_%.09d.cgns", run_input.data_file_name.c_str(), in_file_num);
  cgp_mpi_comm(MPI_COMM_WORLD);
  cgp_pio_mode(CGP_INDEPENDENT);
  if (FlowSol->rank == 0)
    cout << "Writing CGNS file " << fname << " ...." << flush;

  /* open the file and create base and zone */
  sizes[0] = glob_npnodes;
  sizes[1] = glob_npeles;
  sizes[2] = 0;

  if (cgp_open(fname, CG_MODE_WRITE, &F) ||
      cg_base_write(F, "Base", n_dims, n_dims, &B) ||
      cg_zone_write(F, B, "Zone", sizes, Unstructured, &Z))
    cgp_error_exit();

  /* create data nodes for coordinates */
  if (cgp_coord_write(F, B, Z, RealDouble, "CoordinateX", &Cx) ||
      cgp_coord_write(F, B, Z, RealDouble, "CoordinateY", &Cy))
    cgp_error_exit();
  if (n_dims == 3)
    if (cgp_coord_write(F, B, Z, RealDouble, "CoordinateZ", &Cz))
      cgp_error_exit();

  /*! create solution nodes */
  if (cg_sol_write(F, B, Z, "Solution", Vertex, &S))
    cgp_error_exit();
  //default solutions
  if (cgp_field_write(F, B, Z, S, RealDouble, "Density", &Fs_rho) ||
      cgp_field_write(F, B, Z, S, RealDouble, "MomentumX", &Fs_rhou) ||
      cgp_field_write(F, B, Z, S, RealDouble, "MomentumY", &Fs_rhov) ||
      cgp_field_write(F, B, Z, S, RealDouble, "EnergyStagnationDensity", &Fs_rhoe))
    cgp_error_exit();
  if (n_dims == 3)
    if (cgp_field_write(F, B, Z, S, RealDouble, "MomentumZ", &Fs_rhow))
      cgp_error_exit();
  if (run_input.turb_model)
    if (cgp_field_write(F, B, Z, S, RealDouble, "mu", &Fs_mu))
      cgp_error_exit();
  //diagnostic fields
  if (n_diag_fields)
  {
    Fs_diag.setup(n_diag_fields);
    for (int i = 0; i < n_diag_fields; i++)
      if (cgp_field_write(F, B, Z, S, RealDouble, run_input.diagnostic_fields(i).c_str(), Fs_diag.get_ptr_cpu(i)))
        cgp_error_exit();
  }
  //average fields
  if (n_average_fields)
  {
    Fs_avg.setup(n_diag_fields);
    for (int i = 0; i < n_average_fields; i++)
      if (cgp_field_write(F, B, Z, S, RealDouble, run_input.average_fields(i).c_str(), Fs_avg.get_ptr_cpu(i)))
        cgp_error_exit();
  }
  //write solution data
  temp_ptr = pnode_start;//pointer to the next node index to write the solution

  for (int i = 0; i < FlowSol->n_ele_types; i++) //for each type of element
  {
    n_eles = FlowSol->mesh_eles(i)->get_n_eles();
    if (n_eles) //have element
    {
      n_ppts_per_ele = FlowSol->mesh_eles(i)->get_n_ppts_per_ele();
      n_fields = FlowSol->mesh_eles(i)->get_n_fields();
      pos_ppts_temp.setup(n_ppts_per_ele, n_dims);
      disu_ppts_temp.setup(n_ppts_per_ele, n_fields);
      grad_disu_ppts_temp.setup(n_ppts_per_ele, n_fields, n_dims);
      if (n_diag_fields)
        diag_ppts_temp.setup(n_ppts_per_ele, n_diag_fields);
      if (n_average_fields)
        disu_average_ppts_temp.setup(n_ppts_per_ele, n_average_fields);
      if (run_input.shock_cap)
        sensor_ppts_temp.setup(n_ppts_per_ele);

      for (int j = 0; j < n_eles; j++) //for each element
      {
        //coordinate
        FlowSol->mesh_eles(i)->calc_pos_ppts(j, pos_ppts_temp); //get sub selemt coord
        temp_ptr2 = temp_ptr + n_ppts_per_ele - 1;

        if (cgp_coord_write_data(F, B, Z, Cx, &temp_ptr, &temp_ptr2, pos_ppts_temp.get_ptr_cpu()) ||
            cgp_coord_write_data(F, B, Z, Cy, &temp_ptr, &temp_ptr2, pos_ppts_temp.get_ptr_cpu(n_ppts_per_ele)))
          cgp_error_exit();
        if (n_dims == 3)
          if (cgp_coord_write_data(F, B, Z, Cz, &temp_ptr, &temp_ptr2, pos_ppts_temp.get_ptr_cpu(n_ppts_per_ele * 2)))
            cgp_error_exit();

       //default solution
        FlowSol->mesh_eles(i)->calc_disu_ppts(j, disu_ppts_temp);
        if (cgp_field_write_data(F, B, Z, S, Fs_rho, &temp_ptr, &temp_ptr2, disu_ppts_temp.get_ptr_cpu()) ||
            cgp_field_write_data(F, B, Z, S, Fs_rhou, &temp_ptr, &temp_ptr2, disu_ppts_temp.get_ptr_cpu(n_ppts_per_ele)) ||
            cgp_field_write_data(F, B, Z, S, Fs_rhov, &temp_ptr, &temp_ptr2, disu_ppts_temp.get_ptr_cpu(2 * n_ppts_per_ele)))
          cgp_error_exit();
        if (n_dims == 2)
        {
          if (cgp_field_write_data(F, B, Z, S, Fs_rhoe, &temp_ptr, &temp_ptr2, disu_ppts_temp.get_ptr_cpu(3 * n_ppts_per_ele)))
            cgp_error_exit();
          if (run_input.turb_model)
            if (cgp_field_write_data(F, B, Z, S, Fs_mu, &temp_ptr, &temp_ptr2, disu_ppts_temp.get_ptr_cpu(4 * n_ppts_per_ele)))
              cgp_error_exit();
        }
        else
        {
          if (cgp_field_write_data(F, B, Z, S, Fs_rhow, &temp_ptr, &temp_ptr2, disu_ppts_temp.get_ptr_cpu(3 * n_ppts_per_ele)) ||
              cgp_field_write_data(F, B, Z, S, Fs_rhoe, &temp_ptr, &temp_ptr2, disu_ppts_temp.get_ptr_cpu(4 * n_ppts_per_ele)))
            cgp_error_exit();
          if (run_input.turb_model)
            if (cgp_field_write_data(F, B, Z, S, Fs_mu, &temp_ptr, &temp_ptr2, disu_ppts_temp.get_ptr_cpu(5 * n_ppts_per_ele)))
              cgp_error_exit();
        }

        if (run_input.shock_cap)
          /*! Calculate the sensor at the plot points */
          FlowSol->mesh_eles(i)->calc_sensor_ppts(j, sensor_ppts_temp);

        /*! Calculate the diagnostic fields at the plot points */
        if (n_diag_fields > 0)
        {
          FlowSol->mesh_eles(i)->calc_grad_disu_ppts(j, grad_disu_ppts_temp);
          FlowSol->mesh_eles(i)->calc_diagnostic_fields_ppts(j, disu_ppts_temp, grad_disu_ppts_temp, sensor_ppts_temp, diag_ppts_temp, FlowSol->time);
          for (int k = 0; k < n_diag_fields; k++)
            if (cgp_field_write_data(F, B, Z, S, Fs_diag(k), &temp_ptr, &temp_ptr2, diag_ppts_temp.get_ptr_cpu(k * n_ppts_per_ele)))
              cgp_error_exit();
        }

        /*! Calculate the time averaged fields at the plot points */
        if (n_average_fields > 0)
        {
          FlowSol->mesh_eles(i)->calc_time_average_ppts(j, disu_average_ppts_temp);
          for (int k = 0; k < n_average_fields; k++)
            if (cgp_field_write_data(F, B, Z, S, Fs_avg(k), &temp_ptr, &temp_ptr2, disu_average_ppts_temp.get_ptr_cpu(k * n_ppts_per_ele)))
              cgp_error_exit();
        }
      temp_ptr += n_ppts_per_ele; //move pointer to next element
      }
    }
  }

  /* create data node for elements */
  temp_ptr = pnode_start; //pointer to next node index to write
  temp_ptr2 = 1;          //pointer to next index of each type of element
  for (int i = 0; i < FlowSol->n_ele_types; i++)
  {
    n_eles = FlowSol->mesh_eles(i)->get_n_eles();

    if (sum_npele(i)) //if have such type of element globally
    {
      switch (i) //write element node
      {
      case 0:
        cgp_section_write(F, B, Z, "Tri", TRI_3, temp_ptr2, temp_ptr2 + sum_npele(i) - 1, 0, &E[0]);
        temp_ptr2 += sum_npele(i);
        break;
      case 1:
        cgp_section_write(F, B, Z, "Quad", QUAD_4, temp_ptr2, temp_ptr2 + sum_npele(i) - 1, 0, &E[1]);
        temp_ptr2 += sum_npele(i);
        break;
      case 2:
        cgp_section_write(F, B, Z, "Tetra", TETRA_4, temp_ptr2, temp_ptr2 + sum_npele(i) - 1, 0, &E[2]);
        temp_ptr2 += sum_npele(i);
        break;
      case 3:
        cgp_section_write(F, B, Z, "Pris", PENTA_6, temp_ptr2, temp_ptr2 + sum_npele(i) - 1, 0, &E[3]);
        temp_ptr2 += sum_npele(i);
        break;
      case 4:
        cgp_section_write(F, B, Z, "Hex", HEXA_8, temp_ptr2, temp_ptr2 + sum_npele(i) - 1, 0, &E[4]);
        temp_ptr2 += sum_npele(i);
        break;
      }

      if (n_eles) //if have such type element locally, write connectivity
      {
        //calculate connectivity
        calc_connectivity(conn, i, temp_ptr);
        // write the element connectivity in parallel
        if (cgp_elements_write_data(F, B, Z, E[i], pele_start(i), pele_end(i), conn.get_ptr_cpu()))
          cgp_error_exit();
      }
    }
  }
  conn.setup(0); //free memory

  /* close the file */
  cgp_close(F);

#else
/*! write serial CGNS file*/

  int F, B, Z, S, Cx, Cy, Cz;
  int E[5];                                             //element node
  int Fs_rho, Fs_rhou, Fs_rhov, Fs_rhow, Fs_rhoe, Fs_mu, Fs_s ; //field variable
  hf_array<int> Fs_diag, Fs_avg;
  char fname[256];
  cgsize_t sizes[3];
  int temp_ptr, temp_ptr2;
  hf_array<cgsize_t> conn; //connectivity

  //data arrays
  hf_array<double> pos_ppts_temp;
  hf_array<double> disu_ppts_temp;
  hf_array<double> grad_disu_ppts_temp;
  hf_array<double> disu_average_ppts_temp;
  hf_array<double> diag_ppts_temp;
  hf_array<double> sensor_ppts_temp;

  int n_dims = FlowSol->n_dims;
  int n_eles, n_fields, n_ppts_per_ele;
  int n_diag_fields = run_input.n_diagnostic_fields;
  int n_average_fields = run_input.n_average_fields;

  sprintf(fname, "%s_%.09d.cgns", run_input.data_file_name.c_str(), in_file_num);

  if (FlowSol->rank == 0)
    cout << "Writing CGNS file " << fname << " ...." << flush;

 /* open the file and create base and zone */
  sizes[0] = glob_npnodes;
  sizes[1] = glob_npeles;
  sizes[2] = 0;

  if (cg_open(fname, CG_MODE_WRITE, &F) ||
      cg_base_write(F, "Base", n_dims, n_dims, &B) ||
      cg_zone_write(F, B, "Zone", sizes, Unstructured, &Z)||
      cg_sol_write(F, B, Z, "Solution", Vertex, &S))
    cg_error_exit();

//write solution data
  temp_ptr = 1;//next node index to write

  for (int i = 0; i < FlowSol->n_ele_types; i++) //for each type of element
  {
    n_eles = FlowSol->mesh_eles(i)->get_n_eles();
    if (n_eles) //have element
    {
      n_ppts_per_ele = FlowSol->mesh_eles(i)->get_n_ppts_per_ele();
      n_fields = FlowSol->mesh_eles(i)->get_n_fields();
      pos_ppts_temp.setup(n_ppts_per_ele, n_dims);
      disu_ppts_temp.setup(n_ppts_per_ele, n_fields);
      grad_disu_ppts_temp.setup(n_ppts_per_ele, n_fields, n_dims);
      if (n_diag_fields)
        diag_ppts_temp.setup(n_ppts_per_ele, n_diag_fields);
      if (n_average_fields)
        disu_average_ppts_temp.setup(n_ppts_per_ele, n_average_fields);
      if (run_input.shock_cap)
        sensor_ppts_temp.setup(n_ppts_per_ele);

      for (int j = 0; j < n_eles; j++) //for each element
      {
        //coordinate
        FlowSol->mesh_eles(i)->calc_pos_ppts(j, pos_ppts_temp); //get sub selemt coord
        temp_ptr2 = temp_ptr + n_ppts_per_ele - 1;

        if (cg_coord_partial_write(F, B, Z, RealDouble, "CoordinateX", &temp_ptr, &temp_ptr2, pos_ppts_temp.get_ptr_cpu(), &Cx) ||
            cg_coord_partial_write(F, B, Z, RealDouble, "CoordinateY", &temp_ptr, &temp_ptr2, pos_ppts_temp.get_ptr_cpu(n_ppts_per_ele), &Cy))
          cg_error_exit();
        if (n_dims == 3)
          if (cg_coord_partial_write(F, B, Z, RealDouble, "CoordinateZ", &temp_ptr, &temp_ptr2, pos_ppts_temp.get_ptr_cpu(n_ppts_per_ele * 2), &Cz))
            cg_error_exit();

        //default solution
        FlowSol->mesh_eles(i)->calc_disu_ppts(j, disu_ppts_temp);
        if (cg_field_partial_write(F, B, Z, S, RealDouble, "Density", &temp_ptr, &temp_ptr2, disu_ppts_temp.get_ptr_cpu(), &Fs_rho) ||
            cg_field_partial_write(F, B, Z, S, RealDouble, "MomentumX", &temp_ptr, &temp_ptr2, disu_ppts_temp.get_ptr_cpu(n_ppts_per_ele), &Fs_rhou) ||
            cg_field_partial_write(F, B, Z, S, RealDouble, "MomentumY", &temp_ptr, &temp_ptr2, disu_ppts_temp.get_ptr_cpu(2 * n_ppts_per_ele), &Fs_rhov))
          cg_error_exit();
        if (n_dims == 2)
        {
          if (cg_field_partial_write(F, B, Z, S, RealDouble, "EnergyStagnationDensity", &temp_ptr, &temp_ptr2, disu_ppts_temp.get_ptr_cpu(3 * n_ppts_per_ele), &Fs_rhoe))
            cg_error_exit();
          if (run_input.turb_model)
            if (cg_field_partial_write(F, B, Z, S, RealDouble, "mu", &temp_ptr, &temp_ptr2, disu_ppts_temp.get_ptr_cpu(4 * n_ppts_per_ele), &Fs_mu))
              cg_error_exit();
        }
        else
        {
          if (cg_field_partial_write(F, B, Z, S, RealDouble, "MomentumZ", &temp_ptr, &temp_ptr2, disu_ppts_temp.get_ptr_cpu(3 * n_ppts_per_ele), &Fs_rhow) ||
              cg_field_partial_write(F, B, Z, S, RealDouble, "EnergyStagnationDensity", &temp_ptr, &temp_ptr2, disu_ppts_temp.get_ptr_cpu(4 * n_ppts_per_ele), &Fs_rhoe))
            cg_error_exit();
          if (run_input.turb_model)
            if (cg_field_partial_write(F, B, Z, S, RealDouble, "mu", &temp_ptr, &temp_ptr2, disu_ppts_temp.get_ptr_cpu(5 * n_ppts_per_ele), &Fs_mu))
              cg_error_exit();
        }

        if (run_input.shock_cap)
          /*! Calculate the sensor at the plot points */
          FlowSol->mesh_eles(i)->calc_sensor_ppts(j, sensor_ppts_temp);

        /*! Calculate the diagnostic fields at the plot points */
        if (n_diag_fields > 0)
        {
          FlowSol->mesh_eles(i)->calc_grad_disu_ppts(j, grad_disu_ppts_temp);
          FlowSol->mesh_eles(i)->calc_diagnostic_fields_ppts(j, disu_ppts_temp, grad_disu_ppts_temp, sensor_ppts_temp, diag_ppts_temp, FlowSol->time);
          for (int k = 0; k < n_diag_fields; k++)
            if (cg_field_partial_write(F, B, Z, S, RealDouble, run_input.diagnostic_fields(i).c_str(), &temp_ptr, &temp_ptr2, diag_ppts_temp.get_ptr_cpu(k * n_ppts_per_ele), Fs_diag.get_ptr_cpu(k)))
              cg_error_exit();
        }

        /*! Calculate the time averaged fields at the plot points */
        if (n_average_fields > 0)
        {
          FlowSol->mesh_eles(i)->calc_time_average_ppts(j, disu_average_ppts_temp);
          for (int k = 0; k < n_average_fields; k++)
            if (cg_field_partial_write(F, B, Z, S, RealDouble, run_input.average_fields(i).c_str(), &temp_ptr, &temp_ptr2, disu_average_ppts_temp.get_ptr_cpu(k * n_ppts_per_ele), &Fs_avg(k)))
              cg_error_exit();
        }
        temp_ptr += n_ppts_per_ele; //move pointer to next element
      }
    }
  }

  /* create data node for elements */
  temp_ptr = 1; //next node index to write
  temp_ptr2 = 1;//next start point of element type
  for (int i = 0; i < FlowSol->n_ele_types; i++)
  {
    n_eles = FlowSol->mesh_eles(i)->get_n_eles();

      if (n_eles) //if have such type element locally, write connectivity
      {
        //calculate connectivity
        calc_connectivity(conn, i, temp_ptr);

        switch (i) //write element node
        {
        case 0:
          cg_section_write(F, B, Z, "Tri", TRI_3, temp_ptr2, temp_ptr2 + sum_npele(i) - 1, 0, conn.get_ptr_cpu(), &E[0]);
          temp_ptr2 += sum_npele(i);
          break;
        case 1:
          cg_section_write(F, B, Z, "Quad", QUAD_4, temp_ptr2, temp_ptr2 + sum_npele(i) - 1, 0, conn.get_ptr_cpu(), &E[1]);
          temp_ptr2 += sum_npele(i);
          break;
        case 2:
          cg_section_write(F, B, Z, "Tetra", TETRA_4, temp_ptr2, temp_ptr2 + sum_npele(i) - 1, 0, conn.get_ptr_cpu(), &E[2]);
          temp_ptr2 += sum_npele(i);
          break;
        case 3:
          cg_section_write(F, B, Z, "Pris", PENTA_6, temp_ptr2, temp_ptr2 + sum_npele(i) - 1, 0, conn.get_ptr_cpu(), &E[3]);
          temp_ptr2 += sum_npele(i);
          break;
        case 4:
          cg_section_write(F, B, Z, "Hex", HEXA_8, temp_ptr2, temp_ptr2 + sum_npele(i) - 1, 0, conn.get_ptr_cpu(), &E[4]);
          temp_ptr2 += sum_npele(i);
          break;
        }
      }
  }
  conn.setup(0); //free memory

  /* close the file */
  cg_close(F);
  
#endif
  if (FlowSol->rank == 0)
    cout << "done" << endl;
#endif
}


/*! Method to write out a probe file.
Used in run mode.
input: FlowSol						solution structure
output: probe_<probe_index>.dat		probe data file
*/

void output::write_probe(void)
{
    /*! Current rank*/
    int myrank=FlowSol->rank;
    /*! No. of solution fields */
    int n_fields;
    /*! No. of optional diagnostic fields */
    int n_probe_fields=run_probe.n_probe_fields;
    /*! No. of probes */
    int n_probe=run_probe.n_probe;
    /*! No. of dimensions */
    int n_dims=FlowSol->n_dims;
    /*! reference location of probe point */
    hf_array<double> loc_probe_point_temp(n_dims);
    /*! solution data at probe points */
    hf_array<double> disu_probe_point_temp;
    /*! solution gradient data at probe points */
    hf_array<double> grad_disu_probe_point_temp;
    /*! file name */
    char probe_data[256];
    /*! output file object*/
    ofstream write_probe;
    /*! set file name*/
    char folder[]="Probes";

    /*! master node create a directory to store .dat*/
    if(myrank==0)
    {
        struct stat st = {0};
        if(stat(folder,&st)==-1)
        {
            mkdir(folder,0755);
        }
    }
#ifdef _MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif // _MPI
    /*! every node write .dat*/
    if(myrank ==0) cout<<"writing probe point data...";
    for (int i=0; i<n_probe; i++) //loop over every probe point i
    {
        if(run_probe.p2c(i)!=-1)//if probe point belongs to this processor
        {
            //copy probe point property
            n_fields=FlowSol->mesh_eles(run_probe.p2t(i))->get_n_fields();//number of computing fields
            disu_probe_point_temp.setup(n_fields);
            disu_probe_point_temp.initialize_to_zero();
            for (int j=0;j<n_dims;j++)
                loc_probe_point_temp(j)=run_probe.loc_probe(j,i);

            //check if file exist
            bool exist=true;
            struct stat st= {0};
            sprintf(probe_data,"Probes/probe_%.04d.dat",i);//generate file name
            if (stat(probe_data,&st)==-1) exist = false;

            write_probe.open(probe_data,ios_base::out|ios_base::app);//open file
            if (!write_probe.is_open())
            {
                write_probe.open(probe_data,ios_base::out|ios_base::app);
                if (!write_probe.is_open())
                    FatalError("Cannont open input file for reading.");
            }

            if(exist==false)//if doesn't exist write headers
            {
                write_probe<<"NOTE: ALL OUTPUTS ARE DIMENSIONAL IN SI UNITS"<<endl;
                write_probe<<"Probe position"<<endl;
                    write_probe<<setw(15)<<setprecision(5)<<run_probe.pos_probe(0,i)*run_input.L_free_stream<<setw(15)<<setprecision(5)<<run_probe.pos_probe(1,i)*run_input.L_free_stream;
                    if (n_dims==3)
                        write_probe<<setw(15)<<setprecision(5)<<run_probe.pos_probe(2,i)*run_input.L_free_stream<<endl;
                    else
                        write_probe<<endl;
                /*! write gambit surface information*/
                if (run_probe.probe_layout==2 &&run_probe.output_normal==true)
                {
                    write_probe<<"Surface normal"<<endl;
                    write_probe<<setw(15)<<setprecision(5)<<run_probe.surf_normal(0,i)<<setw(15)<<setprecision(5)<<run_probe.surf_normal(1,i);
                    if(n_dims==3)
                        write_probe<<setw(15)<<setprecision(5)<<run_probe.surf_normal(2,i)<<endl;
                    else
                        write_probe<<endl;

                    write_probe<<"Surface area"<<endl;
                        write_probe<<setw(15)<<setprecision(5)<<run_probe.surf_area(i)*run_input.L_free_stream*run_input.L_free_stream<<endl;
                }
                /*! write field titles*/
                write_probe<<setw(17)<<"time";
                for(int j=0; j<n_probe_fields; j++)
                    write_probe<<setw(17)<<run_probe.probe_fields(j);

                write_probe<<endl;
            }

            //calculate fields data on probe points
            FlowSol->mesh_eles(run_probe.p2t(i))->set_opp_probe(loc_probe_point_temp);//calculate solution on upts to probe points matrix
            FlowSol->mesh_eles(run_probe.p2t(i))->calc_disu_probepoints(run_probe.p2c(i),disu_probe_point_temp);//calculate solution on the reference probe point

            /*! Start writing data*/
            if (run_input.viscous)
                write_probe<<setw(17)<<setprecision(10)<<FlowSol->time*run_input.L_free_stream/run_input.uvw_ref;
            else
                write_probe<<setw(17)<<setprecision(10)<<FlowSol->time;
            for (int j=0; j<n_probe_fields; j++)//write transient fields
            {

                if (run_probe.probe_fields(j)=="rho")
                {
                    if(run_input.viscous)
                        write_probe<<setw(17)<<setprecision(10)<<disu_probe_point_temp(0)*run_input.rho_ref;
                    else
                        write_probe<<setw(17)<<setprecision(10)<<disu_probe_point_temp(0);
                }
                else if (run_probe.probe_fields(j)=="u")
                {
                    if (run_input.viscous)
                        write_probe<<setw(17)<<setprecision(10)<<disu_probe_point_temp(1)/disu_probe_point_temp(0)*run_input.uvw_ref;
                    else
                        write_probe<<setw(17)<<setprecision(10)<<disu_probe_point_temp(1)/disu_probe_point_temp(0);
                }
                else if (run_probe.probe_fields(j)=="v")
                {
                    if(run_input.viscous)
                        write_probe<<setw(17)<<setprecision(10)<<disu_probe_point_temp(2)/disu_probe_point_temp(0)*run_input.uvw_ref;
                    else
                        write_probe<<setw(17)<<setprecision(10)<<disu_probe_point_temp(2)/disu_probe_point_temp(0);
                }
                else if (run_probe.probe_fields(j)=="w")
                {
                    if(n_dims==3)
                    {
                        if(run_input.viscous)
                            write_probe<<setw(17)<<setprecision(10)<<disu_probe_point_temp(3)/disu_probe_point_temp(0)*run_input.uvw_ref;
                        else
                            write_probe<<setw(17)<<setprecision(10)<<disu_probe_point_temp(3)/disu_probe_point_temp(0);
                    }

                    else FatalError("2 dimensional elements don't have z velocity");
                }
                else if (run_probe.probe_fields(j)== "specific_energy")//e
                {
                    if(run_input.viscous)
                        write_probe<<setw(17)<<setprecision(10)<<disu_probe_point_temp(n_dims+1)/disu_probe_point_temp(0)*run_input.uvw_ref*run_input.uvw_ref;
                    else
                        write_probe<<setw(17)<<setprecision(10)<<disu_probe_point_temp(n_dims+1)/disu_probe_point_temp(0);
                }
                else if (run_probe.probe_fields(j)=="pressure")
                {
                    double v_sq = 0.;
                    double pressure;
                    for (int m=0; m<n_dims; m++)
                        v_sq += (disu_probe_point_temp(m+1)*disu_probe_point_temp(m+1));
                    v_sq /= disu_probe_point_temp(0)*disu_probe_point_temp(0);
                    // cout<<disu_probe_point_temp(0);
                    // Compute pressure
                    pressure = (run_input.gamma-1.0)*( disu_probe_point_temp(n_dims+1) - 0.5*disu_probe_point_temp(0)*v_sq);
                    if (run_input.viscous)
                        write_probe<<setw(17)<<setprecision(10)<<pressure*run_input.rho_ref*run_input.uvw_ref*run_input.uvw_ref;
                    else
                        write_probe<<setw(17)<<setprecision(10)<<pressure;
                }
                else FatalError("Probe field not implemented yet!");
            }
            write_probe<<endl;
            write_probe.close();//close file
        }

    }
    if (myrank==0) cout<<"done."<<endl;
}

void output::write_restart(int in_file_num)
{

  char file_name_s[256], file_name_s2[256], folder[50];
  char *file_name;
  ofstream restart_file, restart_mesh;
  restart_file.precision(15);
  restart_mesh.precision(15);

#ifdef _MPI
if (FlowSol->nproc>1)
{
  sprintf(file_name_s,"Rest_%.09d/Rest_%.09d_p%.04d.dat",in_file_num,in_file_num,FlowSol->rank);
  sprintf(folder,"Rest_%.09d",in_file_num);
      if(FlowSol->rank==0)
    {
        struct stat st = {0};
        if(stat(folder,&st)==-1)
        {
            mkdir(folder,0755);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

  else//==1
      sprintf(file_name_s,"Rest_%.09d_p%.04d.dat",in_file_num,FlowSol->rank);
      sprintf(file_name_s2,"Rest_%s_%.09d_p%.04d.dat",run_input.data_file_name.c_str(),in_file_num,FlowSol->rank);
  if (FlowSol->rank==0) cout << "Writing Restart file number " << in_file_num << " ...." << endl;
#else
  sprintf(file_name_s,"Rest_%.09d_p%.04d.dat",in_file_num,0);
  sprintf(file_name_s2,"Rest_%s_%.09d_p%.04d.dat",run_input.data_file_name.c_str(),in_file_num,0);
  cout << "Writing Restart file number " << in_file_num << " ...." << endl;
#endif


  file_name = &file_name_s[0];
  restart_file.open(file_name);

  restart_file << FlowSol->time << endl;

  if (run_input.restart_mesh_out) {
    file_name = &file_name_s2[0];
    restart_mesh.open(file_name);
    restart_mesh << FlowSol->time << endl;
  }

  //header
  for (int i=0;i<FlowSol->n_ele_types;i++) {
      if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {

          FlowSol->mesh_eles(i)->write_restart_info(restart_file);
          FlowSol->mesh_eles(i)->write_restart_data(restart_file);

          // Output handy file of point locations for easy post-processing
          if (run_input.restart_mesh_out) {
            FlowSol->mesh_eles(i)->write_restart_info(restart_mesh);
            FlowSol->mesh_eles(i)->write_restart_mesh(restart_mesh);
          }

        }
    }

  restart_file.close();

}

void output::CalcForces(int in_file_num) {

  char file_name_s[256], *file_name;
  char forcedir_s[256], *forcedir;
  struct stat st = {0};
  ofstream coeff_file;
  bool write_dir, write_forces;
  hf_array<double> temp_inv_force(FlowSol->n_dims);
  hf_array<double> temp_vis_force(FlowSol->n_dims);
  double temp_cl, temp_cd;
  int my_rank;

  // set write flags
  if (run_input.restart_flag==0) {
    write_dir = (in_file_num == 1);
    write_forces = ((in_file_num % (run_input.monitor_cp_freq)) == 0) || (in_file_num == 1);
  }
  else {
    write_dir = (in_file_num == run_input.restart_iter+1);
    write_forces = ((in_file_num % (run_input.monitor_cp_freq)) == 0) || (in_file_num == run_input.restart_iter+1);
  }

  // set name of directory to store output files
  sprintf(forcedir_s,"force_files");
  forcedir = &forcedir_s[0];

  // Create directory and set name of files
#ifdef _MPI

  my_rank = FlowSol->rank;

  // Master node creates a subdirectory to store cp_*.dat files
  if ((my_rank == 0) && (write_dir))
    {
      if (stat(forcedir, &st) == -1)
        {
          mkdir(forcedir, 0755);
        }
    }

  if (write_forces)
    {
      sprintf(file_name_s,"force_files/cp_%.09d_p%.04d.dat",in_file_num,my_rank);
      file_name = &file_name_s[0];

      // open files for writing
      coeff_file.open(file_name);
    }

#else

  if (write_dir)
    {
      if (stat(forcedir, &st) == -1)
        {
          mkdir(forcedir, 0755);
        }
    }

  if (write_forces)
    {
      sprintf(file_name_s,"force_files/cp_%.09d_p%.04d.dat",in_file_num,0);
      file_name = &file_name_s[0];

      // open file for writing
      coeff_file.open(file_name);
    }

#endif

  // zero the forces and coeffs
  for (int m=0;m<FlowSol->n_dims;m++)
    {
      FlowSol->inv_force(m) = 0.;
      FlowSol->vis_force(m) = 0.;
    }

  FlowSol->coeff_lift = 0.0;
  FlowSol->coeff_drag = 0.0;

  // loop over elements and compute forces on solid surfaces
  for(int i=0;i<FlowSol->n_ele_types;i++)
    {
      if (FlowSol->mesh_eles(i)->get_n_eles()!=0)
        {
          // compute surface forces and coefficients
          FlowSol->mesh_eles(i)->compute_wall_forces(temp_inv_force, temp_vis_force, temp_cl, temp_cd, coeff_file, write_forces);

          // set surface forces
          for (int m=0;m<FlowSol->n_dims;m++) {
              FlowSol->inv_force(m) += temp_inv_force(m);
              FlowSol->vis_force(m) += temp_vis_force(m);
            }

          // set lift and drag coefficients
          FlowSol->coeff_lift += temp_cl;
          FlowSol->coeff_drag += temp_cd;
        }
    }

#ifdef _MPI

  hf_array<double> inv_force_global(FlowSol->n_dims);
  hf_array<double> vis_force_global(FlowSol->n_dims);
  double coeff_lift_global=0.0;
  double coeff_drag_global=0.0;

  for (int m=0;m<FlowSol->n_dims;m++) {
      inv_force_global(m) = 0.;
      vis_force_global(m) = 0.;
      MPI_Reduce(&FlowSol->inv_force(m),&inv_force_global(m),1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      MPI_Reduce(&FlowSol->vis_force(m),&vis_force_global(m),1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    }

  MPI_Reduce(&FlowSol->coeff_lift,&coeff_lift_global,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  MPI_Reduce(&FlowSol->coeff_drag,&coeff_drag_global,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

  for (int m=0;m<FlowSol->n_dims;m++)
    {
      FlowSol->inv_force(m) = inv_force_global(m);
      FlowSol->vis_force(m) = vis_force_global(m);
    }

  FlowSol->coeff_lift = coeff_lift_global;
  FlowSol->coeff_drag = coeff_drag_global;

#endif

  if (write_forces) { coeff_file.close(); }
}

// Calculate integral diagnostic quantities
void output::CalcIntegralQuantities(int in_file_num) {

  int nintq = run_input.n_integral_quantities;

     // initialize to zero
    for(int j=0; j<nintq; ++j)
    {
        FlowSol->integral_quantities(j) = 0.0;
    }
  // Loop over element types
  for(int i=0;i<FlowSol->n_ele_types;i++)
    {
      if (FlowSol->mesh_eles(i)->get_n_eles()!=0)
        {
          FlowSol->mesh_eles(i)->CalcIntegralQuantities(nintq, FlowSol->integral_quantities);
        }
    }

#ifdef _MPI

  hf_array<double> integral_quantities_global(nintq);
  for(int j=0;j<nintq;++j)
    {
      integral_quantities_global(j) = 0.0;
      MPI_Reduce(&FlowSol->integral_quantities(j),&integral_quantities_global(j),1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      FlowSol->integral_quantities(j) = integral_quantities_global(j);
    }
#endif

}

// Calculate time averaged diagnostic quantities
void output::CalcTimeAverageQuantities(void) {

  // Loop over element types
  for(int i=0;i<FlowSol->n_ele_types;i++)
    {
      if (FlowSol->mesh_eles(i)->get_n_eles()!=0)
        {
          FlowSol->mesh_eles(i)->CalcTimeAverageQuantities(FlowSol->time);
        }
    }
}

void output::compute_error(int in_file_num)
{
  int n_fields;

  //HACK (assume same number of fields for all elements)
  for(int i=0;i<FlowSol->n_ele_types;i++) {
      if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
          n_fields = FlowSol->mesh_eles(i)->get_n_fields();
        }
    }

  hf_array<double> error(2,n_fields);
  hf_array<double> temp_error(2,n_fields);

  for (int i=0; i<n_fields; i++)
    {
      error(0,i) = 0.;
      error(1,i) = 0.;
    }

  //Compute the error
  for(int i=0;i<FlowSol->n_ele_types;i++) {
      if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {
          temp_error = FlowSol->mesh_eles(i)->compute_error(run_input.error_norm_type,FlowSol->time);

          for(int j=0;j<n_fields; j++) {
              error(0,j) += temp_error(0,j);
              if(FlowSol->viscous) {
                  error(1,j) += temp_error(1,j);
                }
            }
        }
    }

#ifdef _MPI
  int n_err_vals = 2*n_fields;

  hf_array<double> error_global(2,n_fields);
  for (int i=0; i<n_fields; i++)
    {
      error_global(0,i) = 0.;
      error_global(1,i) = 0.;
    }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Reduce(error.get_ptr_cpu(),error_global.get_ptr_cpu(),n_err_vals,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

  error = error_global;
#endif

  if (FlowSol->rank==0)
    {
      if (run_input.error_norm_type==1) // L1 norm
        {
          error = error;
        }
      else if (run_input.error_norm_type==2) // L2 norm
        {
          for(int j=0;j<n_fields; j++) {
              error(0,j) = sqrt(error(0,j));
              if(FlowSol->viscous) {
                  error(1,j) = sqrt(error(1,j));
                }
            }
        }

      for(int j=0;j<n_fields; j++) {
          cout << scientific << " sol error, field " << j << " = " << setprecision(13) << error(0,j) << endl;
        }
      if(FlowSol->viscous)
        {
          for(int j=0;j<n_fields; j++) {
              cout << scientific << " grad error, field " << j << " = " << setprecision(13) << error(1,j) << endl;
            }
        }

    }

  // Writing error to file

  char  file_name_s[256] ;
  char *file_name;
  int r_flag;

  if (FlowSol->rank==0)
    {
      sprintf(file_name_s,"error000.dat");
      file_name = &file_name_s[0];
      ofstream write_error;

      write_error.open(file_name,ios::app);
      write_error << in_file_num << ", ";
      write_error <<  run_input.order << ", ";
      write_error <<  scientific << run_input.c_tet << ", ";
      write_error << run_input.mesh_file << ", ";
      write_error << run_input.upts_type_tri << ", ";
      write_error << run_input.upts_type_quad << ", ";
      write_error << run_input.fpts_type_tri << ", ";
      write_error << run_input.adv_type << ", ";
      write_error << run_input.riemann_solve_type << ", ";
      write_error << scientific << run_input.error_norm_type  << ", " ;

      for(int j=0;j<n_fields; j++) {
          write_error << scientific << error(0,j);
          if((j == (n_fields-1)) && FlowSol->viscous==0)
            {
              write_error << endl;
            }
          else
            {
              write_error <<", ";
            }
        }

      if(FlowSol->viscous) {
          for(int j=0;j<n_fields; j++) {
              write_error << scientific << error(1,j);
              if(j == (n_fields-1))
                {
                  write_error << endl;
                }
              else
                {
                  write_error <<", ";
                }
            }
        }

      write_error.close();

      double etol = 1.0e-5;

      r_flag = 0;

      //HACK
      /*
     if( ((abs(ene_hist - error(0,n_fields-1))/ene_hist) < etol && (abs(grad_ene_hist - error(1,n_fields-1))/grad_ene_hist) < etol) || (abs(error(0,n_fields-1)) > abs(ene_hist)) )
     {
     r_flag = 1;
     }
     */

      FlowSol->ene_hist = error(0,n_fields-1);
      FlowSol->grad_ene_hist = error(1,n_fields-1);
    }

  //communicate exit_state across processors
#ifdef _MPI
  MPI_Bcast(&r_flag,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
#endif

#ifdef _MPI
  if(r_flag)
    {
      MPI_Finalize();
    }
#endif

  if(r_flag)
    {
      cout << "Tolerance achieved " << endl;
      exit(0);
    }

}

void output::CalcNormResidual(void) {

  int n_upts = 0;
  int n_fields;

  if (FlowSol->n_dims==2) n_fields = 4;
  else n_fields = 5;

  if (run_input.turb_model==1) {
    n_fields++;
  }

  double sum[6]={0.0,0.0,0.0,0.0,0.0,0.0};
  
  if (run_input.res_norm_type == 0) {
    // Infinity Norm
    for(int i=0; i<FlowSol->n_ele_types; i++) {
      if (FlowSol->mesh_eles(i)->get_n_eles() != 0) {
#ifdef _GPU
        FlowSol->mesh_eles(i)->cp_div_tconf_upts_gpu_cpu();
        FlowSol->mesh_eles(i)->cp_src_upts_gpu_cpu();
#endif
        for(int j=0; j<n_fields; j++)
          sum[j] = max(sum[j], FlowSol->mesh_eles(i)->compute_res_upts(run_input.res_norm_type, j));
      }
    }
  }
  else {
    // 1- or 2-Norm
    for(int i=0; i<FlowSol->n_ele_types; i++) {
      if (FlowSol->mesh_eles(i)->get_n_eles() != 0) {
#ifdef _GPU
        FlowSol->mesh_eles(i)->cp_div_tconf_upts_gpu_cpu();
        FlowSol->mesh_eles(i)->cp_src_upts_gpu_cpu();
#endif
        n_upts += FlowSol->mesh_eles(i)->get_n_eles()*FlowSol->mesh_eles(i)->get_n_upts_per_ele();

        for(int j=0; j<n_fields; j++)
          sum[j] += FlowSol->mesh_eles(i)->compute_res_upts(run_input.res_norm_type, j);
      }
    }
  }

#ifdef _MPI
  double sum_global[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  if (run_input.res_norm_type == 0) {
    // Get maximum
    MPI_Reduce(sum, sum_global, 6, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  }
  else {
    // Get sum
    int n_upts_global = 0;
    MPI_Reduce(&n_upts, &n_upts_global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(sum, sum_global, 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    n_upts = n_upts_global;
  }

  for(int i=0; i<n_fields; i++) sum[i] = sum_global[i];

#endif

  if (FlowSol->rank == 0) {

    // Compute the norm
    for(int i=0; i<n_fields; i++) {
      if (run_input.res_norm_type==0) { FlowSol->norm_residual(i) = sum[i]; } // Infinity Norm
      else if (run_input.res_norm_type==1) { FlowSol->norm_residual(i) = sum[i] / n_upts; } // L1 norm
      else if (run_input.res_norm_type==2) { FlowSol->norm_residual(i) = sqrt(sum[i]) / n_upts; } // L2 norm
      else FatalError("norm_type not recognized");

      if (isnan(FlowSol->norm_residual(i))) {
        FatalError("NaN residual encountered. Exiting");
      }
    }
  }
}

void output::HistoryOutput(int in_file_num, clock_t init, ofstream *write_hist) {

  int i, n_fields;
  clock_t final;
  // TODO: write heads when starting from a restart file
  bool open_hist, write_heads;
  int n_diags = run_input.n_integral_quantities;
  double in_time = FlowSol->time;

  if (FlowSol->n_dims==2) n_fields = 4;
  else n_fields = 5;

  if (run_input.turb_model==1) {
    n_fields++;
  }

  // set write flag
  if (run_input.restart_flag==0) {
    open_hist = (in_file_num == 1);
    write_heads = (((in_file_num % (run_input.monitor_res_freq*20)) == 0) || (in_file_num == 1));
  }
  else {
    open_hist = (in_file_num == run_input.restart_iter+1);
    write_heads = (((in_file_num % (run_input.monitor_res_freq*20)) == 0) || (in_file_num == run_input.restart_iter+1));
  }

  if (FlowSol->rank == 0) {

    // Open history file
    if (open_hist) {

      write_hist->open("history.plt", ios::out);
      write_hist->precision(15);
      write_hist[0] << "TITLE = \"HiFiLES simulation\"" << endl;

      write_hist[0] << "VARIABLES = \"Iteration\"";

      // Add residual and variables
      if (FlowSol->n_dims==2)
        {
            write_hist[0] << ",\"log<sub>10</sub>(Res[<greek>r</greek>])\",\"log<sub>10</sub>(Res[<greek>r</greek>v<sub>x</sub>])\",\"log<sub>10</sub>(Res[<greek>r</greek>v<sub>y</sub>])\",\"log<sub>10</sub>(Res[<greek>r</greek>E])\"";
      if(run_input.turb_model)
      {
          write_hist[0] <<",\"mu_tilde\"";
          //write_hist[0] << ",\"F<sub>x</sub>(Total)\",\"F<sub>y</sub>(Total)\",\"CL</sub>(Total)\",\"CD</sub>(Total)\"";
      }
      if(run_input.calc_force)
        write_hist[0] << ",\"F<sub>x</sub>(Total)\",\"F<sub>y</sub>(Total)\",\"CL</sub>(Total)\",\"CD</sub>(Total)\"";
      }
      else {
       write_hist[0] <<  ",\"log<sub>10</sub>(Res[<greek>r</greek>])\",\"log<sub>10</sub>(Res[<greek>r</greek>v<sub>x</sub>])\",\"log<sub>10</sub>(Res[<greek>r</greek>v<sub>y</sub>])\",\"log<sub>10</sub>(Res[<greek>r</greek>v<sub>z</sub>])\",\"log<sub>10</sub>(Res[<greek>r</greek>E])\"";

       if(run_input.turb_model)
      {
          write_hist[0] <<",\"<greek>mu</greek><sub>tilde</sub>\"";
          //write_hist[0] << ",\"log<sub>10</sub>(Res[<greek>r</greek>E])\",\"F<sub>x</sub>(Total)\",\"F<sub>y</sub>(Total)\",\"F<sub>z</sub>(Total)\",\"CL</sub>(Total)\",\"CD</sub>(Total)\"";

      }
       if(run_input.calc_force)
        write_hist[0] << ",\"log<sub>10</sub>(Res[<greek>r</greek>E])\",\"F<sub>x</sub>(Total)\",\"F<sub>y</sub>(Total)\",\"F<sub>z</sub>(Total)\",\"CL</sub>(Total)\",\"CD</sub>(Total)\"";
      }


      // Add integral diagnostics
      for(i=0; i<n_diags; i++)
        write_hist[0] << ",\"Diagnostics[" << run_input.integral_quantities(i) << "]\"";

      // Add physical and computational time
      write_hist[0] << ",\"Time<sub>Physical</sub>\",\"Time<sub>Comp</sub>(m)\"" << endl;

      write_hist[0] << "ZONE T= \"Convergence history\"" << endl;
    }

    // Write the header
    if (write_heads) {
      if (FlowSol->n_dims==2) {
        if (n_fields == 4) cout << "\n  Iter       Res[Rho]   Res[RhoVelx]   Res[RhoVely]      Res[RhoE]       Fx_Total       Fy_Total" << endl;
        else cout << "\n  Iter       Res[Rho]   Res[RhoVelx]   Res[RhoVely]      Res[RhoE]   Res[MuTilde]       Fx_Total       Fy_Total" << endl;
      }
      else {
        if (n_fields == 5) cout <<  "\n  Iter       Res[Rho]   Res[RhoVelx]   Res[RhoVely]   Res[RhoVelz]      Res[RhoE]       Fx_Total       Fy_Total       Fz_Total" << endl;
        else cout <<  "\n  Iter       Res[Rho]   Res[RhoVelx]   Res[RhoVely]   Res[RhoVelz]      Res[RhoE]   Res[MuTilde]       Fx_Total       Fy_Total       Fz_Total" << endl;
      }
    }

    // Output residuals
    cout.precision(8);
    cout.setf(ios::fixed, ios::floatfield);
    cout.width(6); cout << in_file_num;
    write_hist[0] << in_file_num;
    for(i=0; i<n_fields; i++) {
      cout.width(15); cout << FlowSol->norm_residual(i);
      write_hist[0] << ", " << log10(FlowSol->norm_residual(i));
    }

    // Output forces
        if(run_input.calc_force!=0)
        {
            for(i=0; i< FlowSol->n_dims; i++)
            {
                cout.width(15);
                cout << FlowSol->inv_force(i) + FlowSol->vis_force(i);
                write_hist[0] << ", " << FlowSol->inv_force(i) + FlowSol->vis_force(i);
            }

            //Output lift and drag coeffs
            write_hist[0] << ", " << FlowSol->coeff_lift  << ", " << FlowSol->coeff_drag;
        }
            //Output integral diagnostic quantities
            for(i=0; i<n_diags; i++)
                write_hist[0] << ", " << FlowSol->integral_quantities(i);

    // Output physical time
    write_hist[0] << ", " << in_time;

    // Compute execution time
    final = clock()-init;
    write_hist[0] << ", " << (double) final/(((double) CLOCKS_PER_SEC) * 60.0) << endl;
  }
}

void output::check_stability(void)
{
  int n_fields;
  int bisect_ind, file_lines;

  double c_now, dt_now;
  double a_temp, b_temp;
  double c_file, a_file, b_file;

  hf_array<double> disu_ppts_temp;

  int r_flag = 0;
  double i_tol    = 1.0e-4;
  double e_thresh = 1.5;

  bisect_ind = run_input.bis_ind;
  file_lines = run_input.file_lines;

  // check element specific data

  for(int i=0;i<FlowSol->n_ele_types;i++) {
      if (FlowSol->mesh_eles(i)->get_n_eles()!=0) {

          n_fields = FlowSol->mesh_eles(i)->get_n_fields();

          disu_ppts_temp.setup(FlowSol->mesh_eles(i)->get_n_ppts_per_ele(),n_fields);

          for(int j=0;j<FlowSol->mesh_eles(i)->get_n_eles();j++)
            {
              FlowSol->mesh_eles(i)->calc_disu_ppts(j,disu_ppts_temp);

              for(int k=0;k<FlowSol->mesh_eles(i)->get_n_ppts_per_ele();k++)
                {
                  for(int l=0;l<n_fields;l++)
                    {
                      if ( isnan(disu_ppts_temp(k,l)) || (abs(disu_ppts_temp(k,l))> e_thresh) ) {
                          r_flag = 1;
                        }
                    }
                }
            }
        }
    }

  //HACK
  c_now   = run_input.c_tet;
  dt_now  = run_input.dt;


  if( r_flag==0 )
    {
      a_temp = dt_now;
      b_temp = run_input.b_init;
    }
  else
    {
      a_temp = run_input.a_init;
      b_temp = dt_now;
    }


  //file input
  ifstream read_time;
  read_time.open("time_step.dat",ios::in);
  read_time.precision(12);

  //file output
  ofstream write_time;
  write_time.open("temp.dat",ios::out);
  write_time.precision(12);

  if(bisect_ind > 0)
    {
      for(int i=0; i<file_lines; i++)
        {
          read_time >> c_file >> a_file >> b_file;

          cout << c_file << " " << a_file << " " << b_file << endl;

          if(i == (file_lines-1))
            {
              cout << "Writing to time step file ..." << endl;
              write_time << c_now << " " << a_temp << " " << b_temp << endl;

              read_time.close();
              write_time.close();

              remove("time_step.dat");
              rename("temp.dat","time_step.dat");
            }
          else
            {
              write_time << c_file << " " << a_file << " " << b_file << endl;
            }
        }
    }


  if(bisect_ind==0)
    {
      for(int i=0; i<file_lines; i++)
        {
          read_time >> c_file >> a_file >> b_file;
          write_time << c_file << " " << a_file << " " << b_file << endl;
        }

      cout << "Writing to time step file ..." << endl;
      write_time << c_now << " " << a_temp << " " << b_temp << endl;

      read_time.close();
      write_time.close();

      remove("time_step.dat");
      rename("temp.dat","time_step.dat");
    }

  if( (abs(b_temp - a_temp)/(0.5*(b_temp + a_temp))) < i_tol )
    exit(1);

  if(r_flag>0)
    exit(0);

}

#ifdef _GPU
void output::CopyGPUCPU(void)
{
  // copy solution to cpu

  for(int i=0;i<FlowSol->n_ele_types;i++)
  {
    if (FlowSol->mesh_eles(i)->get_n_eles()!=0)
    {
      if (run_input.motion)
        FlowSol->mesh_eles(i)->cp_transforms_gpu_cpu();
      FlowSol->mesh_eles(i)->cp_disu_upts_gpu_cpu();
      if (FlowSol->viscous==1)
      {
        FlowSol->mesh_eles(i)->cp_grad_disu_upts_gpu_cpu();
      }

      if(run_input.shock_cap)
      {
        FlowSol->mesh_eles(i)->cp_sensor_gpu_cpu();
      }
    }
  }
}
#endif

void output::calc_connectivity(hf_array<int> &out_conn, int ele_type, int &start_index)
{
  int i, j, k;
  int n_eles, n_peles_per_ele, n_ppts_per_ele, n_verts_per_ele;
  hf_array<int> temp_con;
  
  n_eles = FlowSol->mesh_eles(ele_type)->get_n_eles();
  n_peles_per_ele = FlowSol->mesh_eles(ele_type)->get_n_peles_per_ele();
  n_ppts_per_ele = FlowSol->mesh_eles(ele_type)->get_n_ppts_per_ele();
  n_verts_per_ele = FlowSol->mesh_eles(ele_type)->get_n_verts_per_ele();
  temp_con = FlowSol->mesh_eles(ele_type)->get_connectivity_plot();
  out_conn.setup(n_verts_per_ele, n_peles_per_ele, n_eles);

  for (int j = 0; j < n_eles; j++)
    for (int k = 0; k < n_peles_per_ele; k++)
      for (int i = 0; i < n_verts_per_ele; i++)
        out_conn(i, k, j) = temp_con(i, k) + j * n_ppts_per_ele + start_index;

  start_index += n_eles * n_ppts_per_ele;
}
