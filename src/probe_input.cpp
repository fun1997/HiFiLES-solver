/*!
 * \file probe_input.cpp
 * \author - Original code: SD++ developed by Patrice Castonguay, Antony Jameson,
 *                          Peter Vincent, David Williams (alphabetical by surname).
  *         - Current development: Weiqi Shen
                                  University of Florida
 * \version 0.1.0
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
#include "../include/probe_input.h"
#include "../include/hf_array.h"
#include "../include/global.h"
#include "../include/mesh.h"
#include "../include/mesh_reader.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <algorithm>
// Used for making sub-directories
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

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
    n_dims = FlowSol->n_dims;
    read_probe_input(rank);
    set_probe_connectivity(FlowSol, rank);
    create_folder(rank);
}

void probe_input::create_folder(int rank)
{
    /*! master node create a directory to store .dat*/
    if (rank == 0)
    {
        struct stat st = {0};
        if (run_input.probe == 2) //from script
        {
            for (auto &temp_folder : run_probe.surf_name)
                if (stat(temp_folder.c_str(), &st) == -1)
                    mkdir(temp_folder.c_str(), 0755);
            for (auto &temp_folder : run_probe.line_name)
                if (stat(temp_folder.c_str(), &st) == -1)
                    mkdir(temp_folder.c_str(), 0755);
        }
        else //other kinds
        {
            if (stat("probes", &st) == -1)
            {
                mkdir("probes", 0755);
            }
        }
    }
#ifdef _MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif // _MPI
}

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

    /*!----------calculate probes coordinates ------------*/
    if (run_input.probe == 1) //point probes
    {
        hf_array<double> probe_x;
        hf_array<double> probe_y;
        hf_array<double> probe_z;

        probf.getVectorValueOptional("probe_x", probe_x);
        probf.getVectorValueOptional("probe_y", probe_y);
        n_probe = probe_x.get_dim(0); //set number of probes
        if (n_probe != probe_y.get_dim(0))
            FatalError("Probe coordinate data don't agree!\n");
        pos_probe.setup(n_dims, n_probe);
        for (int i = 0; i < n_probe; i++)
        {
            pos_probe(0, i) = probe_x(i);
            pos_probe(1, i) = probe_y(i);
        }
        if (n_dims == 3)
        {
            probf.getVectorValueOptional("probe_z", probe_z);
            if (n_probe != probe_z.get_dim(0))
                FatalError("Probe coordinate data don't agree!\n");
            for (int i = 0; i < n_probe; i++)
            {
                pos_probe(2, i) = probe_z(i);
            }
        }
    }
    else if (run_input.probe == 2) //read script
    {
        probf.getScalarValue("probe_source_file", probe_source_file);
        read_probe_script(probe_source_file);

        if (rank == 0)
        {
            for (auto nm : surf_name)
                cout << "Surface: " << nm << " loaded." << endl;

            for (auto nm : line_name)
                cout << "Line: " << nm << " loaded." << endl;
        }
    }
    else if (run_input.probe == 3) //probes on mesh surface/in volume
    {
        probf.getScalarValue("probe_source_file", probe_source_file);
        set_probe_mesh(probe_source_file);
    }
    else
    {
        FatalError("Probe type not implemented");
    }
    probf.closeFile();
}

void probe_input::set_probe_connectivity(struct solution *FlowSol, int rank)
{
    p2c.setup(n_probe);
    p2t.setup(n_probe);
    p2c.initialize_to_value(-1);
    p2t.initialize_to_value(-1);

    if (rank == 0)
        cout << "Setting up probe points connectivity.." << flush;

    for (int i = 0; i < FlowSol->n_ele_types; i++) //for each element type
    {
        if (FlowSol->mesh_eles(i)->get_n_eles() != 0)
        {
            for (int j = 0; j < n_probe; j++) //for each probe
            {
                if (p2c(j) == -1) //if not inside another type of elements
                {
                    int temp_p2c;
                    hf_array<double> temp_pos(n_dims);
                    for (int k = 0; k < n_dims; k++)
                        temp_pos(k) = pos_probe(k, j);
                    temp_p2c = FlowSol->mesh_eles(i)->calc_p2c(temp_pos);

                    if (temp_p2c != -1) //if inside this type of elements
                    {
                        p2c(j) = temp_p2c;
                        p2t(j) = i;
                    }
                }
            }
        }
    }

#ifdef _MPI
    //MPI_Barrier(MPI_COMM_WORLD);
    hf_array<int> p2cglobe(n_probe, FlowSol->nproc);
    MPI_Allgather(p2c.get_ptr_cpu(), n_probe, MPI_INT, p2cglobe.get_ptr_cpu(), n_probe, MPI_INT, MPI_COMM_WORLD);
    //MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < n_probe; i++) //for each probe
    {
        for (int j = 0; j < rank; j++) //loop over all processors before this one
        {
            if (p2c(i) != -1 && p2cglobe(i, j) != -1) //there's a conflict
            {
                p2c(i) = -1;
                p2t(i) = -1;
                break;
            }
        }
    }
#endif

    if (rank == 0)
        cout << "done" << endl;
    //setup probe location in reference domain.
    if (rank == 0)
        cout << "Calculating location of probes in reference domain.." << flush;
    set_loc_probepts(FlowSol);

    /*
    for(int i=0; i<n_probe; i++)
       if(p2c(i)!=-1)
           cout<<"probe "<<i<<" is found in local element No."<<p2c(i)<<
               ", element type: "<<p2t(i)<<", rank: "<<rank<<endl;
    FatalError("Test end!")
    */
    if (rank == 0)
        cout << "done" << endl;
}

void probe_input::read_probe_script(string filename)
{
    //macro to skip space
#define SKIP_SPACE(A, B)                       \
    A.get(B);                                  \
    while (B == ' ' || B == '\n' || B == '\r') \
        A.get(B);

#define COMP_SYN(VAR, SYN)          \
    if (VAR!=SYN)        \
    {                               \
        FatalError("Syntax error"); \
    }
    //declare strings and buffers
    ifstream script_f;
    string temp_name;
    char dlm;
    //declare number of points for both type
    vector<hf_array<double> > pos_line;
    vector<hf_array<double> > pos_surf;
    //start index for each type of block
    int surf_kstart = 0;
    int line_kstart = 0;
    //read file
    script_f.open(filename);
    if (!script_f)
        FatalError("Unable to open file");

    while (!script_f.eof())
    {
        script_f >> temp_name; //read keyword

        if (temp_name == "surf") //is surface
        {
            if (n_dims == 2)
                FatalError("2D simulation doesn't support 3D surface");
            //store name
            script_f >> temp_name;

            //look for "{"
            SKIP_SPACE(script_f, dlm);
            COMP_SYN(dlm, '{');

            //initialize surface block
            surf_start.push_back(surf_kstart);
            surf_name.push_back(temp_name);

            while (!script_f.eof()) //read contents in surf block
            {
                script_f >> temp_name;
                if (temp_name == "circle")
                {
                    hf_array<double> cent(n_dims);
                    hf_array<double> ori(n_dims);
                    double radius, n_layer;

                    for (int ct = 0; ct < 3; ct++) //read 2 group
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
                else if (temp_name == "cone") 
                {
                    hf_array<double> cent(n_dims);
                    hf_array<double> ori(n_dims);
                    double radius_0, radius_1, len, n_layer_r, n_layer_l;

                    for (int ct = 0; ct < 4; ct++) //read 2 group
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
                    printf("%s surface is not supported", temp_name.c_str());
                    FatalError("exiting")
                }

                SKIP_SPACE(script_f, dlm);
                if (dlm == '}') //test if end of block
                    break;
                else
                script_f.seekg(-1,script_f.cur);
            } //end of block

            surf_kstart = pos_surf.size();              //update the start index of surf
            if (surf_kstart == *(surf_start.end() - 1)) //if empty block
                FatalError("Empty block");
        }
        else if (temp_name == "line") //read a line
        {
            int npt_line;
            double init_incre;
            hf_array<double> p_0(n_dims), p_1(n_dims);

            //store name
            script_f >> temp_name;

            //initialize surface block
            line_start.push_back(line_kstart);
            line_name.push_back(temp_name);

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
            line_kstart = pos_line.size(); //update start index of line
        }
    }
    //close file
    script_f.close();

    //set total number of probes for each type(next start index)
    surf_start.push_back(surf_kstart);
    line_start.push_back(line_kstart);

    //combine them into pos_probe array(first surf then line)
    n_probe = surf_kstart + line_kstart;
    pos_probe.setup(n_dims, n_probe);
    //copy surface position
    for (int i = 0; i < pos_surf.size(); i++)
        for (int j = 0; j < n_dims; j++)
            pos_probe(j, i) = pos_surf[i][j];
    //copy line position
    for (int i = 0; i < pos_line.size(); i++)
        for (int j = 0; j < n_dims; j++)
            pos_probe(j, i + surf_kstart) = pos_line[i][j];
    //line local start index become global start index
    for (auto &id : line_start)
        id += surf_kstart;
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
        for (int i = 0; i < in_n_pts; i++)
        {
            hf_array<double> temp_pos(n_dims);
            for (int j = 0; j < n_dims; j++)
                temp_pos(j) = in_p0(j) + in_init_incre * (pow(growth_rate, (double)i) - 1.) / (growth_rate - 1.) / l_length * (in_p1(j) - in_p0(j));
            out_pos_line.push_back(temp_pos);
        }
    }
    else //equidistance
    {
        growth_rate = 1.0;
        for (int i = 0; i < in_n_pts; i++)
        {
            hf_array<double> temp_pos(n_dims);
            for (int j = 0; j < n_dims; j++)
                temp_pos(j) = in_p0(j) + i * in_init_incre / l_length * (in_p1(j) - in_p0(j));
            out_pos_line.push_back(temp_pos);
        }
    }
}

void probe_input::set_probe_circle(hf_array<double> &in_cent, hf_array<double> &in_ori, const double in_r, const int n_layer,
                                   vector<hf_array<double> > &out_normal, vector<double> &out_area,
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
            for (int ic1 = 0; ic1 < (1 + il); ic1++) //for each downside triangle in that section
            {
                int cell_beg = sec_beg + ic1;
                probe_c2v(ct, 0) = ths_ly_beg + ((cell_beg - ths_ly_beg) % nv_per_vlayer(il));
                probe_c2v(ct, 1) = nx_ly_beg + ((cell_beg + nv_per_vlayer(il) + 1 + is - nx_ly_beg) % nv_per_vlayer(il + 1));
                probe_c2v(ct, 2) = cell_beg + nv_per_vlayer(il) + is;
                ct++;
            }
            for (int ic2 = 0; ic2 < il; ic2++) //for each upside triangle in that section
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
    for (int i = 0; i < n_cell; i++)
    {
        //store coordinates of all vertex belongs to this cell
        hf_array<double> temp_pos(n_dims, 3);
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < n_dims; k++)
                temp_pos(k, j) = probe_xv(probe_c2v(i, j), k);
        hf_array<double> temp_pos_centroid;
        temp_pos_centroid = calc_centroid(temp_pos);
        out_pos_circle.push_back(temp_pos_centroid);
    }
    //calculate face normal/area
    for (int i = 0; i < n_cell; i++)
    {
        hf_array<double> temp_vec(n_dims);
        hf_array<double> temp_vec2(n_dims);
        hf_array<double> temp_normal;
        for (int k = 0; k < n_dims; k++) //every dimension
        {
            temp_vec(k) = probe_xv(probe_c2v(i, 1), k) - probe_xv(probe_c2v(i, 0), k);
            temp_vec2(k) = probe_xv(probe_c2v(i, 2), k) - probe_xv(probe_c2v(i, 1), k);
        }
        temp_normal = cross_prod_3d(temp_vec, temp_vec2);

        //calculate area
        double temp_area;
        double temp_length = 0.0;
        for (int k = 0; k < n_dims; k++)
            temp_length += temp_normal(k) * temp_normal(k);
        temp_length = sqrt(temp_length);
        temp_area = 0.5 * temp_length;
        out_area.push_back(temp_area);

        //normalize the normal vector
        for (int j = 0; j < n_dims; j++)
            temp_normal(j) /= temp_length;
        out_normal.push_back(temp_normal);
    }
}

void probe_input::set_probe_cone(hf_array<double> &in_cent0, hf_array<double> &in_ori, double r0, const double r1,
                                 const int n_layer_r, const double in_l,
                                 const int n_layer_l, vector<hf_array<double> > &out_normal,
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
    for (int i = 0; i < n_cell; i++)
    {
        //store coordinates of all vertex belongs to this cell
        hf_array<double> temp_pos(n_dims, 3);
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < n_dims; k++)
                temp_pos(k, j) = probe_xv(probe_c2v(i, j), k);
        hf_array<double> temp_pos_centroid;
        temp_pos_centroid = calc_centroid(temp_pos);
        out_pos_cone.push_back(temp_pos_centroid);
    }
    //calculate face normal/area
    for (int i = 0; i < n_cell; i++)
    {
        hf_array<double> temp_vec(n_dims);
        hf_array<double> temp_vec2(n_dims);
        hf_array<double> temp_normal;
        for (int k = 0; k < n_dims; k++) //every dimension
        {
            temp_vec(k) = probe_xv(probe_c2v(i, 1), k) - probe_xv(probe_c2v(i, 0), k);
            temp_vec2(k) = probe_xv(probe_c2v(i, 2), k) - probe_xv(probe_c2v(i, 1), k);
        }
        temp_normal = cross_prod_3d(temp_vec, temp_vec2);

        //calculate area
        double temp_area;
        double temp_length = 0.0;
        for (int k = 0; k < n_dims; k++)
            temp_length += temp_normal(k) * temp_normal(k);
        temp_length = sqrt(temp_length);
        temp_area = 0.5 * temp_length;
        out_area.push_back(temp_area);

        //normalize the normal vector
        for (int j = 0; j < n_dims; j++)
            temp_normal(j) /= temp_length;
        out_normal.push_back(temp_normal);
    }
}

void probe_input::set_probe_mesh(string filename) //be able to read 3D sufaces, 2D planes and 3D volumes
{
    mesh probe_msh;
    if (filename.compare(filename.size() - 3, 3, "neu")) //for now only gambit is supported
        FatalError("Only gambit neutral file format is supported!");

    mesh_reader probe_mr(filename, &probe_msh);

    probe_mr.partial_read_connectivity(0, probe_msh.num_cells_global);
    probe_msh.create_iv2ivg();
    probe_mr.read_vertices();
    n_probe = probe_msh.num_cells;
    ele_dims = probe_msh.n_ele_dims; //mesh element dimension(surf or volume)
    mesh_dims = probe_msh.n_dims;    //mesh dimension(2D/3D)

    if (mesh_dims > n_dims)
        FatalError("Mesh for probe cannot have a higher dimension than simulation");

    /*! calculate face centroid and copy it to pos_probe*/
    pos_probe.setup(n_dims, n_probe);
    for (int i = 0; i < n_probe; i++)
    {
        //store coordinates of all vertex belongs to this cell
        hf_array<double> temp_pos(n_dims, probe_msh.c2n_v(i));
        temp_pos.initialize_to_zero();
        for (int j = 0; j < probe_msh.c2n_v(i); j++)
            for (int k = 0; k < mesh_dims; k++) //up to probe mesh coord dimension
                temp_pos(k, j) = probe_msh.xv(probe_msh.c2v(i, j), k);
        hf_array<double> temp_pos_centroid;
        temp_pos_centroid = calc_centroid(temp_pos);
        for (int j = 0; j < n_dims; j++)
            pos_probe(j, i) = temp_pos_centroid(j);
    }

    /*! calculate face normals and area*/
    if (ele_dims == 2 && n_dims == 3) //if surface mesh and 3D simulation
    {
        //calculate face normal
        for (int i = 0; i < n_probe; i++)
        {
            hf_array<double> temp_vec(n_dims);
            hf_array<double> temp_vec2(n_dims);
            hf_array<double> temp_normal;
            for (int k = 0; k < n_dims; k++) //every dimension
            {
                temp_vec(k) = probe_msh.xv(probe_msh.c2v(i, 1), k) - probe_msh.xv(probe_msh.c2v(i, 0), k);
                temp_vec2(k) = probe_msh.xv(probe_msh.c2v(i, 2), k) - probe_msh.xv(probe_msh.c2v(i, 1), k);
            }
            temp_normal = cross_prod_3d(temp_vec, temp_vec2);

            //normalize the normal vector
            double temp_length = 0.0;
            for (int j = 0; j < n_dims; j++)
                temp_length += temp_normal(j) * temp_normal(j);
            temp_length = sqrt(temp_length);
            for (int j = 0; j < n_dims; j++)
                temp_normal(j) /= temp_length;
            surf_normal.push_back(temp_normal);
        }

        //calculate face area
        surf_area.assign(n_probe, 0.0);
        for (int i = 0; i < n_probe; i++)
        {
            hf_array<double> temp_vec(n_dims);
            hf_array<double> temp_vec2(n_dims);
            hf_array<double> temp_normal;
            for (int j = 0; j < probe_msh.ctype(i) + 1; j++) //1->2 part
            {
                if ((probe_msh.ctype(i) == 1 && probe_msh.c2n_v(i) > 4) || (probe_msh.ctype(i) == 0 && probe_msh.c2n_v(i) > 3))
                    FatalError("2nd order surf not supported for area calculation");
                for (int k = 0; k < n_dims; k++) //every dimension
                {
                    temp_vec(k) = probe_msh.xv(probe_msh.c2v(i, 1 + j), k) - probe_msh.xv(probe_msh.c2v(i, j), k);
                    temp_vec2(k) = probe_msh.xv(probe_msh.c2v(i, 2 + j), k) - probe_msh.xv(probe_msh.c2v(i, 1 + j), k);
                }
                temp_normal = cross_prod_3d(temp_vec, temp_vec2);
                double temp_area = 0.0;
                for (int k = 0; k < n_dims; k++)
                    temp_area += temp_normal(k) * temp_normal(k);
                temp_area = 0.5 * sqrt(temp_area);
                surf_area[i] += temp_area;
            }
        }
    }
}

void probe_input::set_loc_probepts(struct solution *FlowSol)
{
    loc_probe.setup(n_dims, n_probe);
    hf_array<double> temp_pos(n_dims);
    hf_array<double> temp_loc(n_dims);
    for (int i = 0; i < n_probe; i++) //loop over all probes
    {
        if (p2c(i) != -1) //if probe belongs to this processor
        {
            for (int j = 0; j < n_dims; j++)
                temp_pos(j) = pos_probe(j, i);
            FlowSol->mesh_eles(p2t(i))->pos_to_loc(temp_pos, p2c(i), temp_loc);
            //copy to loc_probe
            for (int j = 0; j < n_dims; j++)
                loc_probe(j, i) = temp_loc(j);
        }
        else
        {
            for (int j = 0; j < n_dims; j++)
                loc_probe(j, i) = 0;
        }
    }
}
/*! END */
