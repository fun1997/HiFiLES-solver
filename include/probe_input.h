/*!
 * \file probe_input.h
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
#include <vector>
#include <map>
#include "error.h"
#include "hf_array.h"
#include "solution.h"

enum PROBE_TYPE
{
    VOLUME_PROBE = 0,
    SURFACE_PROBE = 1,
    LINE_PROBE = 2,
    POINT_PROBE = 3
};

class probe_input
{
public:
    probe_input();
    ~probe_input();
    //basic inputs
    hf_array<string> probe_fields;
    int probe_freq;
    int n_probe_fields;
    string probe_source_file;//probe source filename

    //global data
    int n_probe_global;//global number of probes(all sets)
    hf_array<double> pos_probe_global;
    hf_array<string> probe_name; //set2name in order v->s->l->p/m
    hf_array<bool> probe_surf_flag;//set2flag in order v->s->l->p/m
    hf_array<int> probe_start;  //set2start in order v->s->l->p/m

    //local data
    int n_probe;//local number of probes
    #ifdef _HDF5
    hf_array<int> set2n_probe; //set2nprobe in order v->s->l->p/m
    #endif
    vector<int> p2c;//local probe point to cell number(local typewise)
    vector<int> p2t;//local probe point to cell type
    vector<int> p2global_p;//local probe point index to gloabl probal point index
    hf_array<double> loc_probe;//location of local probe point

    //surface params(global)
    int surf_offset;
    vector<double> surf_normal; //column major n_dims*n_surf
    vector<double> surf_area;


    //entrance
    void setup(char *fileNameC,struct solution* FlowSol, int rank);
private:
    void read_probe_input(int rank);
    void set_probe_connectivity(struct solution* FlowSol,int rank);
#ifdef _HDF5
    void create_probe_hdf5(int rank);
#else
    void create_folder(int rank);
#endif

    void read_probe_script(string filename);
    void set_probe_line(hf_array<double>& in_p0, hf_array<double>& in_p1,const double in_init_incre,
                        const int in_n_pts,vector<hf_array<double> > &out_pos_line);
    //read in start index and return number of points
    void set_probe_circle(hf_array<double> &in_cent, hf_array<double> &in_ori, const double in_r, const int n_layer,
                          vector<double> &out_normal, vector<double> &out_area,
                          vector<hf_array<double> > &out_pos_circle);

    void set_probe_cone(hf_array<double> &in_cent0, hf_array<double> &in_ori, double r0, const double r1,
                        const int n_layer_r, const double in_l,
                        const int n_layer_l, vector<double> &out_normal,
                        vector<double> &out_area, vector<hf_array<double> > &out_pos_cone);
                        
    void set_probe_cube(hf_array<double> &in_origin, hf_array<int> &in_n_xyz, hf_array<double> &in_d_xyz,
                        vector<hf_array<double> > &out_pos_cube);

    void set_probe_mesh(string filename);

    void set_loc_probepts(struct solution* FlowSol);

    int n_dims;//simulation dimension
    string fileNameS;//main input filename
};
