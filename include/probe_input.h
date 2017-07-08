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

class probe_input
{
public:
    probe_input();
    ~probe_input();
    array<string> probe_fields;
    int probe_layout;
    array<double> probe_init_cord;
    array<double> growth_rate;
    array<double> init_incre;
    int probe_dim_x;
    int probe_dim_y;
    int probe_dim_z;
    int n_probe_fields;
    array<double> probe_x;
    array<double> probe_y;
    array<double> probe_z;
    array<double> probe_pos;
    int n_probe;
    int prob_freq;
    array<int> p2c;
    array<int> p2t;

    void setup(string filenameS,int in_dim, int rank);
    void read_probe_input(string filename, int rank);
    void set_probe_connection(struct solution* FlowSol,int rank);
protected:

private:
        int n_dims;
};
