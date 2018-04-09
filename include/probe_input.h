/*!
 * \file probe_input.h
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
#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include "array.h"
#include "input.h"
#include "solution.h"
#include "funcs.h"

class probe_input
{
public:
    probe_input();
    ~probe_input();
    //basic inputs
    array<string> probe_fields;
    int n_probe;
    int prob_freq;
    int probe_layout;
    int n_probe_fields;
    array<double> pos_probe;
    //point source
    array<double> probe_x;
    array<double> probe_y;
    array<double> probe_z;
    //line source
    array<double> p_0;//start point coord
    array<double> p_1;//end point coord
    double growth_rate;
    double init_incre;
    //gambit surface
    #define MAX_V_PER_C 27
    array<double> surf_normal;//surface normals
    array<double> surf_area;//surface area
    bool output_normal;
    //connetivity
    array<int> p2c;//probe point to cell number(local typewise)
    array<int> p2t;//probe point to cell type
    array<double> loc_probe;
    //entrance
    void setup(string filenameS,struct solution* FlowSol, int rank);
protected:
    void read_probe_input(string filename, int rank);
    void set_probe_connectivity(struct solution* FlowSol,int rank);
    void set_probe_gambit(string filename);
    void set_loc_probepts(struct solution* FlowSol);
private:
    int n_dims;
    string neu_file;
};
