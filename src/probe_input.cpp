/*!
 * \file probe_input.cpp
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
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <algorithm>
// Used for making sub-directories
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../include/global.h"
#include "../include/mesh.h"
#include "../include/mesh_reader.h"
#include "../include/param_reader.h"
#ifdef _HDF5
#include "hdf5.h"
#endif

using namespace std;
probe_input::probe_input()
{
    //ctor
}

probe_input::~probe_input()
{
    //dtor
}
void probe_input::setup(char *fileNameC, struct solution *FlowSol, int rank)
{
    if (rank == 0)
        cout << endl
             << "---------------------- Setting up probes ---------------------" << endl;
    fileNameS.assign(fileNameC);
    n_dims = FlowSol->n_dims;//simulation dimension
    read_probe_input(rank);
    set_probe_connectivity(FlowSol, rank);
#ifdef _HDF5
    create_probe_hdf5(rank);
#else
    create_folder(rank);
#endif
}

#ifdef _HDF5
void probe_input::create_probe_hdf5(int rank)
{
    int ct = 0; //counter for local probe
    double sample_dt;//sample interval in sec
    hid_t fid, dataset_id, attr_id, dataspace_id, datatype;//hdf5 types
    hsize_t dim[2];
    set2n_probe.setup(probe_name.get_dim(0));//set of probeto local number of probes
    set2n_probe.initialize_to_zero();

    //rank 0 dimensionalize coord, area,sample interval if needed
    if (rank == 0)
    {
        if (run_input.viscous && run_input.equation == 0)
        {
            transform(pos_probe_global.get_ptr_cpu(), pos_probe_global.get_ptr_cpu() + n_dims * n_probe_global,
                      pos_probe_global.get_ptr_cpu(), [](double x) { return x * run_input.L_ref; });
            transform(surf_area.begin(), surf_area.end(),
                      surf_area.begin(), [](double x) { return x * run_input.L_ref * run_input.L_ref; });
            sample_dt = run_input.dt * probe_freq * run_input.time_ref;
        }
        else
            sample_dt = run_input.dt * probe_freq;
    }

    for (int i = 0; i < probe_name.get_dim(0); i++) //for each set of probe
    {
        if (rank == 0)
        {
            /*! master node create hdf5 files*/
            string temp_probe_fname = probe_name(i) + ".h5";
            struct stat st = {0};

            if (stat(temp_probe_fname.c_str(), &st) == -1)//file not exist                                                                     //if not exist
            {
                fid = H5Fcreate(temp_probe_fname.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT); //creat file if not exist
                if (fid < 0)
                    FatalError("Failed to create probe data file");
                //write sample freq
                dataspace_id = H5Screate(H5S_SCALAR);
                attr_id = H5Acreate(fid, "dt", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
                H5Awrite(attr_id, H5T_NATIVE_DOUBLE, &sample_dt);
                H5Aclose(attr_id);
                //create final time
                attr_id = H5Acreate(fid, "fnl_time", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
                H5Aclose(attr_id);
                H5Sclose(dataspace_id);
                //create probe fields
                dim[0] = probe_fields.get_dim(0);
                const char **temp_field = new const char *[dim[0]];
                for (int j = 0; j < probe_fields.get_dim(0); j++)
                    temp_field[j] = probe_fields(j).c_str();
                dataspace_id = H5Screate_simple(1, dim, NULL);
                datatype = H5Tcopy(H5T_C_S1);
                H5Tset_size(datatype, H5T_VARIABLE);
                attr_id = H5Acreate(fid, "fields", datatype, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
                H5Awrite(attr_id, datatype, temp_field);
                H5Aclose(attr_id);
                H5Tclose(datatype);
                H5Sclose(dataspace_id);
                delete[] temp_field;

                //write coord
                dim[0] = probe_start(i + 1) - probe_start(i); //number of points for this set of probe
                dim[1] = n_dims;
                dataspace_id = H5Screate_simple(2, dim, NULL);
                dataset_id = H5Dcreate2(fid, "coord", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, pos_probe_global.get_ptr_cpu() + n_dims * probe_start(i));
                H5Sclose(dataspace_id);
                H5Dclose(dataset_id);
                //write face normal and area
                if (probe_surf_flag(i))
                {
                    //normal
                    dataspace_id = H5Screate_simple(2, dim, NULL);
                    dataset_id = H5Dcreate2(fid, "normal", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                    H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, surf_normal.data() + n_dims * (probe_start(i) - surf_offset));
                    H5Sclose(dataspace_id);
                    H5Dclose(dataset_id);
                    //area
                    dataspace_id = H5Screate_simple(1, dim, NULL);
                    dataset_id = H5Dcreate2(fid, "area", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                    H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, surf_area.data() + probe_start(i) - surf_offset);
                    H5Sclose(dataspace_id);
                    H5Dclose(dataset_id);
                }
                //close file
                H5Fclose(fid);
            }
        }
        //setup set2n_probe array
        while (ct < n_probe)
        {
            if (p2global_p[ct] < probe_start(i + 1))
            {
                set2n_probe(i)++;
                ct++;
            }
            else
            {
                break;
            }
        }
    }
#ifdef _MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif // _MPI
    //clear up memory
    surf_normal.clear();
    surf_area.clear();
    pos_probe_global.setup(1);
}
#endif

#ifndef _HDF5
void probe_input::create_folder(int rank)
{
    /*! master node create a directory*/
    if (rank == 0)
    {
        struct stat st = {0};
        for (int i = 0; i < probe_name.get_dim(0); i++)
            if (stat(probe_name(i).c_str(), &st) == -1)
                mkdir(probe_name(i).c_str(), 0755);
    }
#ifdef _MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif // _MPI

    //write header if needed
    char probe_data[256];
    ofstream wt_probe;
    string folder;
    int file_idx;

    //dimensionalize if needed
    if (run_input.viscous && run_input.equation == 0)
    {
        transform(pos_probe_global.get_ptr_cpu(), pos_probe_global.get_ptr_cpu() + n_dims * n_probe_global,
                  pos_probe_global.get_ptr_cpu(), [](double x) { return x * run_input.L_ref; });
        transform(surf_area.begin(), surf_area.end(),
                  surf_area.begin(), [](double x) { return x * run_input.L_ref * run_input.L_ref; });
    }

    for (int i=0; i<n_probe; i++) //loop over every local probe point i
    {
        bool surf_flag;
        //set the folder name
        for (int id = 0; id < probe_name.get_dim(0); id++) //loop over each set of probe
        {
          if (p2global_p[i] < probe_start[id + 1] && p2global_p[i] >= probe_start[id])
          {
            folder = probe_name[id];
            surf_flag = probe_surf_flag[id];
            file_idx = p2global_p[i] - probe_start[id];
            break;
          }
        }

        //check if file exist
        struct stat st = {0};
        sprintf(probe_data, "%s/%s_%.06d.dat", folder.c_str(), folder.c_str(), file_idx); //generate file name

        if (stat(probe_data, &st) == -1) //if doesn't exist write headers
        {
            wt_probe.open(probe_data, ios_base::out | ios_base::app); //open file
            if (!wt_probe.is_open())
            {
                FatalError("Cannont open input file for reading.");
            }
            //use normal notation
            wt_probe.unsetf(ios::floatfield);
            wt_probe << "NOTE: ALL OUTPUTS ARE DIMENSIONAL IN SI UNITS" << endl;
            wt_probe << "Probe position" << endl;
            wt_probe << setw(20) << setprecision(10) <<pos_probe_global(0, p2global_p[i])
                     << setw(20) << setprecision(10) << pos_probe_global(1, p2global_p[i]);
            if (n_dims == 3)
                wt_probe << setw(20) << setprecision(10) << pos_probe_global(2, p2global_p[i]) << endl;
            else
                wt_probe << endl;

            /*! write surface information*/
            if (surf_flag)
            {
                wt_probe << "Surface normal" << endl;
                wt_probe << setw(20) << setprecision(10) << surf_normal[(p2global_p[i] - surf_offset) * n_dims]
                         << setw(20) << setprecision(10) << surf_normal[(p2global_p[i] - surf_offset) * n_dims + 1];
                if (n_dims == 3)
                    wt_probe << setw(20) << setprecision(10) << surf_normal[(p2global_p[i] - surf_offset) * n_dims + 2] << endl;
                else
                    wt_probe << endl;

                wt_probe << "Surface area" << endl;
                    wt_probe << setw(20) << setprecision(10) << surf_area[p2global_p[i] - surf_offset] << endl;
            }

            /*! write field titles*/
            wt_probe << setw(20) << "time";
            for (int j = 0; j < n_probe_fields; j++)
                wt_probe << setw(20) << probe_fields(j);
            wt_probe << endl;
            wt_probe.close();
        }
    }
    //clear up memory
    surf_normal.clear();
    surf_area.clear();
    pos_probe_global.setup(1);
}
#endif

void probe_input::read_probe_input(int rank)
{
    param_reader probf(fileNameS);
    probf.openFile();

    /*!----------read probe input parameters ------------*/
    probf.getVectorValue("probe_fields", probe_fields); //name of probe fields
    n_probe_fields = probe_fields.get_dim(0);
    for (int i = 0; i < n_probe_fields; i++)
    {
        std::transform(probe_fields(i).begin(), probe_fields(i).end(),
                       probe_fields(i).begin(), ::tolower);
    }
    probf.getScalarValue("probe_freq", probe_freq);
    probf.getScalarValue("probe_source_file", probe_source_file);
    probf.closeFile();

    /*!----------calculate probes coordinates ------------*/

    if (run_input.probe == 1) //read script
    {
        read_probe_script(probe_source_file);

        if (rank == 0)
        {
            for (int nm = 0; nm < probe_name.get_dim(0); nm++)
                cout << probe_name(nm) << " loaded." << endl;
        }
    }
    else if (run_input.probe == 2) //probes on gambit mesh surface/in volume
    {
        set_probe_mesh(probe_source_file);
        if (rank == 0)
            cout << probe_name(0) << " loaded." << endl;
    }
    else
    {
        FatalError("Probe type not implemented");
    }
}

void probe_input::set_probe_connectivity(struct solution *FlowSol, int rank)
{
    if (rank == 0)
        cout << "Setting up probe points connectivity.." << flush;

    int temp_p2c;
    hf_array<double> temp_pos(n_dims);
    for (int j = 0; j < n_probe_global; j++) //for each global probe
    {
        for (int i = 0; i < FlowSol->n_ele_types; i++) //for each element type
        {
            if (FlowSol->mesh_eles(i)->get_n_eles() != 0)
            {
                for (int k = 0; k < n_dims; k++)
                    temp_pos(k) = pos_probe_global(k, j);
                temp_p2c = FlowSol->mesh_eles(i)->calc_p2c(temp_pos);

                if (temp_p2c != -1) //if inside this type of elements
                {
                    p2c.push_back(temp_p2c);
                    p2t.push_back(i);
                    p2global_p.push_back(j);
                    break;
                }
            }
        }
    }
    n_probe = p2global_p.size(); //number of local probe point before eliminating repeating points

#ifdef _MPI
    //allgather number of probe on each processor
    hf_array<int> kprocs(FlowSol->nproc);
    MPI_Allgather(&n_probe, 1, MPI_INT, kprocs.get_ptr_cpu(), 1, MPI_INT, MPI_COMM_WORLD);

    //broadcast list of p2global_p to other processors
    hf_array<int> p2global_p_buffer;
    for (int i = 0; i < FlowSol->nproc; i++)
    {
        p2global_p_buffer.setup(kprocs(i));

        if (i == FlowSol->rank) //load sending buffer
            copy(p2global_p.begin(), p2global_p.end(), p2global_p_buffer.get_ptr_cpu());

        MPI_Bcast(p2global_p_buffer.get_ptr_cpu(), kprocs(i), MPI_INT, i, MPI_COMM_WORLD);

        if (i < FlowSol->rank) //eliminate repeating element only if source has lower rank
        {
            vector<int> intersect; //intection of remote and local p2global_p
            set_intersection(p2global_p.begin(), p2global_p.end(),
                             p2global_p_buffer.get_ptr_cpu(), p2global_p_buffer.get_ptr_cpu() + kprocs(i),
                             back_inserter(intersect));
            for (size_t j = 0; j < intersect.size(); j++)
            {
                int id = index_locate_int(intersect[j], p2global_p.data(), p2global_p.size());
                p2global_p.erase(p2global_p.begin() + id);
                p2c.erase(p2c.begin() + id);
                p2t.erase(p2t.begin() + id);
            }
        }
    }
    n_probe = p2global_p.size();//update local number of probe

#endif

    if (rank == 0)
        cout << "done" << endl;
    //setup probe location in reference domain.
    if (rank == 0)
        cout << "Calculating location of probes in reference domain.." << flush;
    set_loc_probepts(FlowSol);

    if (rank == 0)
        cout << "done" << endl;
}

void probe_input::read_probe_script(string filename)
{
    //macros to skip white spaces and compare syntax
#define SKIP_SPACE(A, B) \
    do                   \
    {                    \
        A.get(B);        \
    } while ((B == ' ' || B == '\n' || B == '\r') && !A.eof());

#define COMP_SYN(VAR, SYN)          \
    if (VAR != SYN)                 \
        FatalError("Syntax error")

    //declare strings and buffers
    ifstream script_f;
    string kwd, name0, name1;
    char dlm;
    //declare arrays to store global positions of points for all types of probes
    vector<hf_array<double> > pos_vol;
    vector<hf_array<double> > pos_surf;
    vector<hf_array<double> > pos_line;
    vector<hf_array<double> > pos_point;
    //declare arrays to store name and start index of all types of probes
    vector<int> vol_start;
    vector<int> surf_start;
    vector<int> line_start;
    vector<int> point_start;

    vector<string> vol_name;
    vector<string> surf_name;
    vector<string> line_name;
    //declare start index for each type of probe
    int vol_kstart = 0;
    int surf_kstart = 0;
    int line_kstart = 0;
    int point_kstart = 0;
    //read file
    script_f.open(filename);
    if (!script_f)
        FatalError("Unable to open file");

    while (script_f >> kwd)//while not end of file, read new group
    {
         //read keywords
         if(kwd=="volume")//if is volume
         {
             //store name
             script_f >> name0;
             //look for "{"
             SKIP_SPACE(script_f, dlm);
             COMP_SYN(dlm, '{');
             //initialize volume block
             vol_start.push_back(vol_kstart); //log start index
             vol_name.push_back(name0);

             while (1) //read contents in volume block
             {
                 script_f >> name1;
                 if (name1 == "cube")
                 {
                     hf_array<int> n_xyz(3);
                     n_xyz(2) = 1;
                     hf_array<double> origin(n_dims);
                     hf_array<double> d_xyz(n_dims);

                     for (int ct = 0; ct < 3; ct++) //read 3 groups of parameters
                     {
                         SKIP_SPACE(script_f, dlm);
                         COMP_SYN(dlm, '(');
                         if (ct == 0) //start point
                             for (int i = 0; i < n_dims; i++)
                                 script_f >> origin(i);
                         else if (ct == 1) //n points each direction
                             for (int i = 0; i < n_dims; i++)
                                 script_f >> n_xyz(i);
                         else //interval in each direction
                             for (int i = 0; i < n_dims; i++)
                                 script_f >> d_xyz(i);
                         //look for ")"
                         SKIP_SPACE(script_f, dlm);
                         COMP_SYN(dlm, ')');
                     }
                     //load positions of probes into pos_vol
                     set_probe_cube(origin, n_xyz, d_xyz, pos_vol);
                 }
                 else //element not support
                 {
                     printf("%s volume is not supported", name1.c_str());
                     FatalError("exiting")
                 }

                 //detect end of block
                 SKIP_SPACE(script_f, dlm);
                 if (script_f.eof()) //if end of file
                 {
                     FatalError("Syntax Error, Expecting '}' ");
                 }
                 else
                 {
                     if (dlm == '}') //if encounter "}"
                         break;
                     else
                         script_f.seekg(-1, script_f.cur);
                 }
            } //end of block
            if ((int)pos_vol.size() != vol_kstart)
                vol_kstart = pos_vol.size(); //update new start index of volume
            else
                FatalError("No volume probes read!");
         }
        else if (kwd == "surface") //if is surface
        {
            if (n_dims == 2)
                FatalError("2D simulation doesn't support 3D surface");

            //store name
            script_f >> name0;

            //look for "{"
            SKIP_SPACE(script_f, dlm);
            COMP_SYN(dlm, '{');

            //initialize surface block
            surf_start.push_back(surf_kstart);//log start index
            surf_name.push_back(name0);

            while (1) //read contents in surf block
            {
                script_f >> name1;
                if (name1 == "circle")
                {
                    hf_array<double> cent(n_dims);
                    hf_array<double> ori(n_dims);
                    double radius;
                    int n_layer;

                    for (int ct = 0; ct < 3; ct++) //read 3 groups of parameters
                    {
                        SKIP_SPACE(script_f, dlm);
                        COMP_SYN(dlm, '(');
                        if (ct == 0) //centroid
                            for (int i = 0; i < n_dims; i++)
                                script_f >> cent(i);
                        else if (ct == 1) //orientation
                            for (int i = 0; i < n_dims; i++)
                                script_f >> ori(i);
                        else //other
                            script_f >> radius >> n_layer;
                        //look for ")"
                        SKIP_SPACE(script_f, dlm);
                        COMP_SYN(dlm, ')');
                    }
                    //load positions of probes into pos_surf
                    set_probe_circle(cent, ori, radius, n_layer, surf_normal, surf_area, pos_surf);
                }
                else if (name1 == "cone") 
                {
                    hf_array<double> cent(n_dims);
                    hf_array<double> ori(n_dims);
                    double radius_0, radius_1, len;
                    int n_layer_r, n_layer_l;

                    for (int ct = 0; ct < 4; ct++) //read 4 groups of parameters
                    {
                        SKIP_SPACE(script_f, dlm);
                        COMP_SYN(dlm, '(');
                        if (ct == 0)
                            for (int i = 0; i < n_dims; i++)
                                script_f >> cent(i);
                        else if (ct == 1)
                            for (int i = 0; i < n_dims; i++)
                                script_f >> ori(i);
                        else if (ct == 2)
                            script_f >> radius_0 >> radius_1 >> n_layer_r;
                        else
                            script_f >> len >> n_layer_l;

                        //look for ")"
                        SKIP_SPACE(script_f, dlm);
                        COMP_SYN(dlm, ')')
                    }
                    //load positions of probes into pos_surf
                    set_probe_cone(cent, ori, radius_0, radius_1, n_layer_r,
                                   len, n_layer_l, surf_normal, surf_area, pos_surf);
                }
                else //element not support
                {
                    printf("%s surface is not supported", name1.c_str());
                    FatalError("exiting")
                }

                //detect end of block
                SKIP_SPACE(script_f, dlm);
                if (script_f.eof()) //if end of file
                {
                    FatalError("Syntax Error, Expecting '}' ");
                }
                else
                {
                    if (dlm == '}') //if encounter "}"
                        break;
                    else
                        script_f.seekg(-1, script_f.cur);
                }
            } //end of block

            if ((int)pos_surf.size() != surf_kstart)
                surf_kstart = pos_surf.size(); //update new start index of surf
            else
                FatalError("No surface probes read!");
        }
        else if (kwd == "line") //read a line
        {
            int npt_line;
            double init_incre;
            hf_array<double> p_0(n_dims), p_1(n_dims);

            //store name
            script_f >> name0;

            //initialize surface block
            line_start.push_back(line_kstart);//log start index
            line_name.push_back(name0);

            for (int ct = 0; ct < 3; ct++)
            {
                SKIP_SPACE(script_f, dlm);
                COMP_SYN(dlm, '(');
                if (ct == 0)
                    for (int i = 0; i < n_dims; i++)
                        script_f >> p_0(i);
                else if (ct == 1)
                    for (int i = 0; i < n_dims; i++)
                        script_f >> p_1(i);
                else
                    script_f >> init_incre >> npt_line;
                //look for ")"
                SKIP_SPACE(script_f, dlm);
                COMP_SYN(dlm, ')');
            }
            set_probe_line(p_0, p_1, init_incre, npt_line, pos_line);

            if ((int)pos_line.size() != line_kstart)
                line_kstart = pos_line.size(); //update new start index of line
            else
                FatalError("No line probes read!");
        }
        else if (kwd=="point")
        {
            //detect beginning of block
            SKIP_SPACE(script_f, dlm);
            COMP_SYN(dlm, '{');

            point_start.push_back(point_kstart);//log start index

            while (1) //read points
            {
                //read coordinate
                SKIP_SPACE(script_f, dlm);
                COMP_SYN(dlm, '('); //look for start of coordinate

                hf_array<double> temp_pos(n_dims);
                for (int i = 0; i < n_dims; i++)
                    script_f >> temp_pos(i);

                SKIP_SPACE(script_f, dlm);
                COMP_SYN(dlm, ')'); //look for end of coordinate

                pos_point.push_back(temp_pos);

                //detect end of block
                SKIP_SPACE(script_f, dlm);
                if (script_f.eof()) //if end of file
                {
                    FatalError("Syntax Error, Expecting '}' ");
                }
                else
                {
                    if (dlm == '}') //if encounter "}"
                        break;
                    else
                        script_f.seekg(-1, script_f.cur);
                }
            }
            point_kstart = pos_point.size();//update new start index
        }
        else
        {
            FatalError("Type of probe input not implemented yet");
        }
    }
    //close file
    script_f.close();

    //combine them into pos_probe_global array(first surf then line finally points)
    n_probe_global = vol_kstart + surf_kstart + line_kstart + point_kstart;
    pos_probe_global.setup(n_dims, n_probe_global);
    surf_offset = vol_kstart;

    //copy positions to global position array
    for (size_t i = 0; i < pos_vol.size(); i++)
        for (int j = 0; j < n_dims; j++)
            pos_probe_global(j, i) = pos_vol[i][j];
    for (size_t i = 0; i < pos_surf.size(); i++)
        for (int j = 0; j < n_dims; j++)
            pos_probe_global(j, i + vol_kstart) = pos_surf[i][j];
    for (size_t i = 0; i < pos_line.size(); i++)
        for (int j = 0; j < n_dims; j++)
            pos_probe_global(j, i + vol_kstart + surf_kstart) = pos_line[i][j];
    for (size_t i = 0; i < pos_point.size(); i++)
        for (int j = 0; j < n_dims; j++)
            pos_probe_global(j, i + vol_kstart + surf_kstart + line_kstart) = pos_point[i][j];

    //copy start arrays to global start array and set probe type flag and names
    probe_start.setup((int)(vol_start.size() + surf_start.size() + line_start.size() + point_start.size() + 1));
    probe_surf_flag.setup(probe_start.get_dim(0) - 1);
    probe_name.setup(probe_start.get_dim(0) - 1);

    int ct = 0;
    for (size_t i = 0; i < vol_start.size(); i++)
    {
        probe_surf_flag(ct) = false;
        probe_name(ct) = vol_name[i];
        probe_start(ct++) = vol_start[i];
    }
    for (size_t i = 0; i < surf_start.size(); i++)
    {
        probe_surf_flag(ct) = true;
        probe_name(ct) = surf_name[i];
        probe_start(ct++) = surf_start[i] + vol_kstart;
    }
    for (size_t i = 0; i < line_start.size(); i++)
    {
        probe_surf_flag(ct) = false;
        probe_name(ct) = line_name[i];
        probe_start(ct++) = line_start[i] + vol_kstart + surf_kstart;
    }
    if (point_start.size())
    {
        probe_surf_flag(ct) = false;
        probe_name(ct) = string("points");
        probe_start(ct++) = point_start[0] + vol_kstart + surf_kstart + line_kstart;
    }
    probe_start(ct) = n_probe_global; //add total number of probes to the array

#undef SKIP_SPACE
#undef COMP_SYN
}

void probe_input::set_probe_line(hf_array<double> &in_p0, hf_array<double> &in_p1, const double in_init_incre,
                                 const int in_n_pts, vector<hf_array<double> > &out_pos_line)
{
    //calculate growth rate
    double l_length = 0.;
    for (int i = 0; i < n_dims; i++)
    {
        l_length += pow(in_p1(i) - in_p0(i), 2);
    }
    l_length = sqrt(l_length);

    double growth_rate;
    if ((l_length / in_init_incre) != (in_n_pts - 1)) //need iteratively solve for growth rate
    {
        double growth_rate_old = 10.;
        double jacob;
        double fx_n;
        double tol = 1e-10;
        if (l_length / in_init_incre < (in_n_pts - 1))
            growth_rate = 0.1;
        else
            growth_rate = 5; //initialze to be greater than 2
        //find growth rate use newton's method
        while (fabs(growth_rate - growth_rate_old) > tol)
        {
            growth_rate_old = growth_rate;
            jacob = in_init_incre * ((in_n_pts - 2.) * pow(growth_rate, in_n_pts) - (in_n_pts - 1.) * pow(growth_rate, in_n_pts - 1.) + growth_rate) / (pow(growth_rate - 1., 2.) * growth_rate);
            fx_n = l_length - in_init_incre * (pow(growth_rate, in_n_pts - 1) - 1.) / (growth_rate - 1.);
            growth_rate += fx_n / jacob;
        }

        if (std::isnan(growth_rate))
            FatalError("Growth rate NaN!");

        //calculate probe coordiantes
        hf_array<double> temp_pos(n_dims);
        for (int i = 0; i < in_n_pts; i++)
        {
            for (int j = 0; j < n_dims; j++)
                temp_pos(j) = in_p0(j) + in_init_incre * (pow(growth_rate, (double)i) - 1.) / (growth_rate - 1.) / l_length * (in_p1(j) - in_p0(j));
            out_pos_line.push_back(temp_pos);
        }
    }
    else //equidistance
    {
        growth_rate = 1.0;
        hf_array<double> temp_pos(n_dims);
        for (int i = 0; i < in_n_pts; i++)
        {
            for (int j = 0; j < n_dims; j++)
                temp_pos(j) = in_p0(j) + i * in_init_incre / l_length * (in_p1(j) - in_p0(j));
            out_pos_line.push_back(temp_pos);
        }
    }
}

void probe_input::set_probe_circle(hf_array<double> &in_cent, hf_array<double> &in_ori, const double in_r, const int n_layer,
                                   vector<double> &out_normal, vector<double> &out_area,
                                   vector<hf_array<double> > &out_pos_circle)
{
    //calculate number of cell center points for the face
    int n_cell = 6 * pow(n_layer, 2);
    //calculate number of vertex points on the face
    int n_v = (6 + 6 * n_layer) * n_layer / 2 + 1;
    //initialize connectivity
    hf_array<double> probe_xv(n_v, n_dims);
    hf_array<int> probe_c2v(n_cell, 3); //tri element
    hf_array<int> nv_per_vlayer(n_layer + 1);
    //set up nv_per_vlayer
    for (int i = 0; i < n_layer + 1; i++)
        nv_per_vlayer(i) = ((i == 0) ? 1 : 6 * i);
    //set up vertex coordinate assume origin is at 0 0 0 and face to x positive
    int ct = 0;                                 //set counter
    for (int ivl = 0; ivl < n_layer + 1; ivl++) //for each vertex layer from vertex layer 0 to n_layer
    {
        for (int iv = 0; iv < nv_per_vlayer(ivl); iv++) //for each vertex in that layer, start from y=in_cent(1),+z dir
        {
            probe_xv(ct, 0) = 0;
            probe_xv(ct, 1) = sin((double)iv / (double)nv_per_vlayer(ivl) * (2 * pi)) * ivl * in_r / (double)n_layer;
            probe_xv(ct, 2) = cos((double)iv / (double)nv_per_vlayer(ivl) * (2 * pi)) * ivl * in_r / (double)n_layer;
            ct++;
        }
    }

    //rotate and translate to the final position
    //1. setup rotational matrix
    hf_array<double> rot_y(n_dims, n_dims);
    hf_array<double> rot_z(n_dims, n_dims);
    rot_y.initialize_to_zero();
    if (sqrt(pow(in_ori(0), 2.) + pow(in_ori(2), 2.)) == 0)//no rotation about y axis
    {
        rot_y(0, 0) = 1;
        rot_y(0, 2) = 0;
    }
    else
    {
        rot_y(0, 0) = in_ori(0) / sqrt(pow(in_ori(0), 2.) + pow(in_ori(2), 2.));
        rot_y(0, 2) = -in_ori(2) / sqrt(pow(in_ori(0), 2.) + pow(in_ori(2), 2.));//inverse direction of positive axis (right hand rule)
    }
    rot_y(1, 1) = 1;
    rot_y(2, 0) = -rot_y(0, 2);
    rot_y(2, 2) = rot_y(0, 0);

    rot_z.initialize_to_zero();
    rot_z(0, 0) = cos(asin(in_ori(1) / sqrt(pow(in_ori(0), 2.) + pow(in_ori(1), 2.) + pow(in_ori(2), 2.))));
    rot_z(0, 1) = -in_ori(1) / sqrt(pow(in_ori(0), 2.) + pow(in_ori(1), 2.) + pow(in_ori(2), 2.));
    rot_z(1, 0) = -rot_z(0, 1);
    rot_z(1, 1) = rot_z(0, 0);
    rot_z(2, 2) = 1;
    //2. assemble final rotational matrix
    hf_array<double> transf = mult_arrays(rot_y, rot_z);
    transf = transpose_array(transf);//(y*z)^T
    //3. transform the vertex coordinate
    probe_xv = mult_arrays(probe_xv, transf);//A*(y*z)^T=y*z*A^T
    for (int i = 0; i < n_v; i++)
        for (int m = 0; m < n_dims; m++)
            probe_xv(i, m) += in_cent(m);

    //set up cell connectivity
    ct = 0;                              //reset counter
    for (int il = 0; il < n_layer; il++) //for each layer from 0 to n_layer-1
    {
        int ths_ly_beg = ((il == 0) ? 0 : 1) + (6 * (il - 1) * il / 2);
        int ths_ly_end = ths_ly_beg + nv_per_vlayer(il) - 1;
        int nx_ly_beg = ths_ly_end + 1;
        for (int is = 0; is < 6; is++) //for each section
        {
            int sec_beg = ths_ly_beg + is * il;      //beginning of the section
            for (int ic1 = 0; ic1 < (1 + il); ic1++) //for each downside triangle in that section, counter clockwise
            {
                int cell_beg = sec_beg + ic1;
                probe_c2v(ct, 0) = ths_ly_beg + ((cell_beg - ths_ly_beg) % nv_per_vlayer(il));
                probe_c2v(ct, 1) = nx_ly_beg + ((cell_beg + nv_per_vlayer(il) + 1 + is - nx_ly_beg) % nv_per_vlayer(il + 1));
                probe_c2v(ct, 2) = cell_beg + nv_per_vlayer(il) + is;
                ct++;
            }
            for (int ic2 = 0; ic2 < il; ic2++) //for each upside triangle in that section, counter clockwise
            {
                int cell_beg = sec_beg + ic2;
                probe_c2v(ct, 0) = cell_beg;
                probe_c2v(ct, 1) = ths_ly_beg + ((cell_beg + 1 - ths_ly_beg) % nv_per_vlayer(il));
                probe_c2v(ct, 2) = cell_beg + nv_per_vlayer(il) + 1 + is;
                ct++;
            }
        }
    }

    //calculate cell centroid
    hf_array<double> temp_pos(n_dims, 3);
    hf_array<double> temp_pos_centroid;
    for (int i = 0; i < n_cell; i++)
    {
        //store coordinates of all vertex belongs to this cell
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < n_dims; k++)
                temp_pos(k, j) = probe_xv(probe_c2v(i, j), k);
        temp_pos_centroid = calc_centroid(temp_pos);
        out_pos_circle.push_back(temp_pos_centroid);
    }

    //calculate face normal/area
    hf_array<double> temp_vec(n_dims);
    hf_array<double> temp_vec2(n_dims);
    hf_array<double> temp_normal;
    double temp_area;
    double temp_length;
    for (int i = 0; i < n_cell; i++)
    {
        for (int k = 0; k < n_dims; k++) //every dimension
        {
            temp_vec(k) = probe_xv(probe_c2v(i, 1), k) - probe_xv(probe_c2v(i, 0), k);
            temp_vec2(k) = probe_xv(probe_c2v(i, 2), k) - probe_xv(probe_c2v(i, 1), k);
        }
        temp_normal = cross_prod_3d(temp_vec, temp_vec2);

        //calculate area
        temp_length = 0.0;
        for (int k = 0; k < n_dims; k++)
            temp_length += temp_normal(k) * temp_normal(k);
        temp_length = sqrt(temp_length);
        temp_area = 0.5 * temp_length;
        out_area.push_back(temp_area);

        //normalize the normal vector
        for (int j = 0; j < n_dims; j++)
        {
            temp_normal(j) /= temp_length;
            out_normal.push_back(temp_normal(j));
        }
    }
}

void probe_input::set_probe_cone(hf_array<double> &in_cent0, hf_array<double> &in_ori, double r0, const double r1,
                                 const int n_layer_r, const double in_l,
                                 const int n_layer_l, vector<double> &out_normal,
                                 vector<double> &out_area, vector<hf_array<double> > &out_pos_cone)
{
    //calculate number of cell center points for the face
    int n_cell = n_layer_l * n_layer_r * 2;
    //calculate number of vertex points on the face
    int n_v = n_layer_r * (n_layer_l + 1);
    //initialize connectivity
    hf_array<double> probe_xv(n_v, n_dims);
    hf_array<int> probe_c2v(n_cell, 3); //tri element

    //set up vertex coordinate assume origin is at 0 0 0 and face to x positive
    int ct = 0;                                   //set counter
    for (int ivl = 0; ivl < n_layer_l + 1; ivl++) //for each axial vertex layer
    {
        for (int iv = 0; iv < n_layer_r; iv++) //for each circumference vertex
        {
            probe_xv(ct, 0) = in_l * (double)ivl / (double)n_layer_l;
            probe_xv(ct, 1) = sin((double)iv / (double)n_layer_r * (2 * pi)) * (r0 + (double)ivl / (double)n_layer_l * (r1 - r0));
            probe_xv(ct, 2) = cos((double)iv / (double)n_layer_r * (2 * pi)) * (r0 + (double)ivl / (double)n_layer_l * (r1 - r0));
            ct++;
        }
    }

    //rotate and translate to the final position
    //1. setup rotational matrix
    hf_array<double> rot_y(n_dims, n_dims);
    hf_array<double> rot_z(n_dims, n_dims);
    rot_y.initialize_to_zero();
    if (sqrt(pow(in_ori(0), 2.) + pow(in_ori(2), 2.)) == 0)
    {
        rot_y(0, 0) = 1;
        rot_y(0, 2) = 0;
    }
    else
    {
        rot_y(0, 0) = in_ori(0) / sqrt(pow(in_ori(0), 2.) + pow(in_ori(2), 2.));
        rot_y(0, 2) = -in_ori(2) / sqrt(pow(in_ori(0), 2.) + pow(in_ori(2), 2.));
    }
    rot_y(1, 1) = 1;
    rot_y(2, 0) = -rot_y(0, 2);
    rot_y(2, 2) = rot_y(0, 0);

    rot_z.initialize_to_zero();
    rot_z(0, 0) = cos(asin(in_ori(1) / sqrt(pow(in_ori(0), 2.) + pow(in_ori(1), 2.) + pow(in_ori(2), 2.))));
    rot_z(0, 1) = -in_ori(1) / sqrt(pow(in_ori(0), 2.) + pow(in_ori(1), 2.) + pow(in_ori(2), 2.));
    rot_z(1, 0) = -rot_z(0, 1);
    rot_z(1, 1) = rot_z(0, 0);
    rot_z(2, 2) = 1;
    //2. assemble final rotational matrix
    hf_array<double> transf = mult_arrays(rot_y, rot_z);
    transf = transpose_array(transf);
    //3. transform the vertex coordinate
    probe_xv = mult_arrays(probe_xv, transf);
    for (int i = 0; i < n_v; i++)
        for (int m = 0; m < n_dims; m++)
            probe_xv(i, m) += in_cent0(m);

    //set up cell connectivity
    ct = 0;                                //reset counter
    for (int il = 0; il < n_layer_l; il++) //for each axial layer of cell
    {
        int ths_ly_beg = il * n_layer_r;
        int ths_ly_end = ths_ly_beg + n_layer_r - 1;
        int nx_ly_beg = ths_ly_end + 1;

        for (int ic1 = 0; ic1 < n_layer_r; ic1++) //for each downside triangle
        {
            int cell_beg = ths_ly_beg + ic1;
            probe_c2v(ct, 0) = cell_beg;
            probe_c2v(ct, 1) = cell_beg + n_layer_r;
            probe_c2v(ct, 2) = nx_ly_beg + ((cell_beg + n_layer_r + 1 - nx_ly_beg) % n_layer_r);
            ct++;
        }
        for (int ic2 = 0; ic2 < n_layer_r; ic2++) //for each upside triangle
        {
            int cell_beg = ths_ly_beg + ic2;
            probe_c2v(ct, 0) = cell_beg;
            probe_c2v(ct, 1) = nx_ly_beg + ((cell_beg + n_layer_r + 1 - nx_ly_beg) % n_layer_r);
            probe_c2v(ct, 2) = ths_ly_beg + ((cell_beg + 1 - ths_ly_beg) % n_layer_r);
            ct++;
        }
    }
    //calculate cell centroid
    hf_array<double> temp_pos(n_dims, 3);
    hf_array<double> temp_pos_centroid;
    for (int i = 0; i < n_cell; i++)
    {
        //store coordinates of all vertex belongs to this cell
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < n_dims; k++)
                temp_pos(k, j) = probe_xv(probe_c2v(i, j), k);
        temp_pos_centroid = calc_centroid(temp_pos);
        out_pos_cone.push_back(temp_pos_centroid);
    }

    //calculate face normal/area
    hf_array<double> temp_vec(n_dims);
    hf_array<double> temp_vec2(n_dims);
    hf_array<double> temp_normal;
    double temp_area;
    double temp_length;
    for (int i = 0; i < n_cell; i++)
    {
        for (int k = 0; k < n_dims; k++) //every dimension
        {
            temp_vec(k) = probe_xv(probe_c2v(i, 1), k) - probe_xv(probe_c2v(i, 0), k);
            temp_vec2(k) = probe_xv(probe_c2v(i, 2), k) - probe_xv(probe_c2v(i, 1), k);
        }
        temp_normal = cross_prod_3d(temp_vec, temp_vec2);

        //calculate area
        temp_length = 0.0;
        for (int k = 0; k < n_dims; k++)
            temp_length += temp_normal(k) * temp_normal(k);
        temp_length = sqrt(temp_length);
        temp_area = 0.5 * temp_length;
        out_area.push_back(temp_area);

        //normalize the normal vector
        for (int j = 0; j < n_dims; j++)
        {
            temp_normal(j) /= temp_length;
            out_normal.push_back(temp_normal(j));
        }
    }
}

void probe_input::set_probe_cube(hf_array<double> &in_origin, hf_array<int> &in_n_xyz, hf_array<double> &in_d_xyz,
                                 vector<hf_array<double> > &out_pos_cube)
{
    hf_array<double> temp_pos(n_dims);
    hf_array<int> ct(3);

    for (ct[2] = 0; ct[2] < in_n_xyz[2]; ct[2]++)
        for (ct[1] = 0; ct[1] < in_n_xyz[1]; ct[1]++)
            for (ct[0] = 0; ct[0] < in_n_xyz[0]; ct[0]++)
            {
                for (int l = 0; l < n_dims; l++)
                    temp_pos[l] = in_origin[l] + ct[l] * in_d_xyz[l];
                out_pos_cube.push_back(temp_pos);
            }
}

void probe_input::set_probe_mesh(string filename) //be able to read 3D sufaces, 2D planes and 3D volumes
{
    mesh probe_msh;
    int mesh_dims,ele_dims;
    mesh_reader probe_mr(filename, &probe_msh);

    probe_mr.partial_read_connectivity(0, probe_msh.num_cells_global);
    probe_msh.create_iv2ivg();
    probe_mr.read_vertices();
    n_probe_global = probe_msh.num_cells;
    ele_dims = probe_msh.n_ele_dims; //mesh element dimension(surf or volume)
    mesh_dims = probe_msh.n_dims;    //mesh dimension(2D/3D)

    probe_start.setup(2);
    probe_start(0) = 0;
    probe_start(1) = n_probe_global;

    if (mesh_dims > n_dims)
        FatalError("Mesh for probe cannot have a higher dimension than simulation");

    /*! calculate face centroid and copy it to pos_probe*/
    pos_probe_global.setup(n_dims, n_probe_global);
    hf_array<double> temp_pos;
    hf_array<double> temp_pos_centroid;
    for (int i = 0; i < n_probe_global; i++)
    {
        //store coordinates of all vertex belongs to this cell
        temp_pos.setup(n_dims, probe_msh.c2n_v(i));
        temp_pos.initialize_to_zero();
        for (int j = 0; j < probe_msh.c2n_v(i); j++)
            for (int k = 0; k < mesh_dims; k++) //up to probe mesh coord dimension
                temp_pos(k, j) = probe_msh.xv(probe_msh.c2v(i, j), k);
        temp_pos_centroid = calc_centroid(temp_pos);
        for (int j = 0; j < n_dims; j++)
            pos_probe_global(j, i) = temp_pos_centroid(j);
    }

    /*! calculate face normals and area*/
    probe_surf_flag.setup(1);
    probe_surf_flag(0) = false;
    surf_offset=0;
    if (ele_dims == 2 && n_dims == 3) //if surface mesh and 3D simulation
    {
        probe_surf_flag(0) = true;
        //calculate face normal
        hf_array<double> temp_vec(n_dims);
        hf_array<double> temp_vec2(n_dims);
        hf_array<double> temp_normal;
        temp_vec.initialize_to_zero();
        temp_vec2.initialize_to_zero();
        double temp_length;
        for (int i = 0; i < n_probe_global; i++)
        {
            for (int k = 0; k < mesh_dims; k++) //every mesh dimension
            {
                temp_vec(k) = probe_msh.xv(probe_msh.c2v(i, 1), k) - probe_msh.xv(probe_msh.c2v(i, 0), k);
                temp_vec2(k) = probe_msh.xv(probe_msh.c2v(i, 2), k) - probe_msh.xv(probe_msh.c2v(i, 1), k);
            }
            temp_normal = cross_prod_3d(temp_vec, temp_vec2);

            //normalize the normal vector
            temp_length = 0.0;
            for (int j = 0; j < n_dims; j++)
                temp_length += temp_normal(j) * temp_normal(j);
            temp_length = sqrt(temp_length);
            for (int j = 0; j < n_dims; j++)
            {
                temp_normal(j) /= temp_length;
                surf_normal.push_back(temp_normal(j));
            }
        }

        //calculate face area
        surf_area.assign(n_probe_global, 0.0);
        double temp_area;
        for (int i = 0; i < n_probe_global; i++)
        {
            for (int j = 0; j < probe_msh.ctype(i) + 1; j++) //1->2 part
            {
                if ((probe_msh.ctype(i) == 1 && probe_msh.c2n_v(i) > 4) || (probe_msh.ctype(i) == 0 && probe_msh.c2n_v(i) > 3))
                    FatalError("2nd order surf not supported for area calculation");
               
                for (int k = 0; k < mesh_dims; k++) //every mesh dimension
                {
                    temp_vec(k) = probe_msh.xv(probe_msh.c2v(i, 1 + j), k) - probe_msh.xv(probe_msh.c2v(i, j), k);
                    temp_vec2(k) = probe_msh.xv(probe_msh.c2v(i, 2 + j), k) - probe_msh.xv(probe_msh.c2v(i, 1 + j), k);
                }
                temp_normal = cross_prod_3d(temp_vec, temp_vec2);
                temp_area = 0.0;
                for (int k = 0; k < n_dims; k++)
                    temp_area += temp_normal(k) * temp_normal(k);
                temp_area = 0.5 * sqrt(temp_area);
                surf_area[i] += temp_area;
            }
        }
    }

    //extract mesh file basename
    const size_t last_slash_idx = probe_source_file.find_last_of("\\/");
    if (last_slash_idx != string::npos)
        probe_source_file.erase(0, last_slash_idx + 1);
    // Remove extension if present.
    const size_t period_idx = probe_source_file.rfind('.');
    if (period_idx != string::npos)
        probe_source_file.erase(period_idx);

    probe_name.setup(1);
    probe_name(0) = probe_source_file;
}

void probe_input::set_loc_probepts(struct solution *FlowSol)
{
    loc_probe.setup(n_dims, n_probe);
    hf_array<double> temp_pos(n_dims);
    hf_array<double> temp_loc(n_dims);
    for (int i = 0; i < n_probe; i++) //loop over all probes
    {
        for (int j = 0; j < n_dims; j++)
            temp_pos(j) = pos_probe_global(j, p2global_p[i]);
        FlowSol->mesh_eles(p2t[i])->pos_to_loc(temp_pos, p2c[i], temp_loc);
        //copy to loc_probe
        for (int j = 0; j < n_dims; j++)
            loc_probe(j, i) = temp_loc(j);
    }
}
/*! END */
