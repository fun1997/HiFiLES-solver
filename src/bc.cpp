/*!
 * \file bc.h
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
#include "../include/bc.h"

using namespace std;

bc::bc():use_wm(0)
{
}
bc::~bc()
{
}

void bc::setup(const string &in_bc_name) //setup the maps
{
    bc_type2flag["sub_in_simp"] = 0;  // Subsonic inflow simple (free pressure) //
    bc_type2flag["sub_out_simp"] = 1; // Subsonic outflow simple (fixed pressure) //
    bc_type2flag["sub_in_char"] = 2;  // Subsonic inflow characteristic //
    bc_type2flag["sub_out_char"] = 3; // Subsonic outflow characteristic //
    bc_type2flag["sup_in"] = 4;       // Supersonic inflow //
    bc_type2flag["sup_out"] = 5;      // Supersonic outflow //
    bc_type2flag["slip_wall"] = 6;    // Slip wall //
    bc_type2flag["cyclic"] = 7;        // Cyclic//
    bc_type2flag["isotherm_wall"] = 8;    // Isothermal, no-slip wall //
    bc_type2flag["adiabat_wall"] = 9;    // Adiabatic, no-slip wall //
    bc_type2flag["char"] = 10;           // Characteristic //
    bc_type2flag["slip_wall_dual"] = 11; // Dual consistent BC //
    bc_type2flag["ad_wall"] = 12;        // Advection, Advection-Diffusion Boundary Conditions //

//reverse map
    map<string, int>::iterator it;
    for (it = bc_type2flag.begin(); it != bc_type2flag.end(); it++)
        bc_flag2type[it->second] = it->first;

    bc_name=in_bc_name;    
}

//get functions
int bc::get_bc_flag(void)
{
    return bc_flag;
}
string bc::get_bc_type(void)
{
    return bc_flag2type.find(bc_flag)->second;
}

string bc::get_bc_name(void)
{
    return bc_name;
}

//set functions
int bc::set_bc_flag(string &in_type) //read from input and set flag, transform the input to lower case
{
    std::transform(in_type.begin(), in_type.end(), in_type.begin(), ::tolower);
    map<string, int>::iterator it = bc_type2flag.find(in_type);
    if (it != bc_type2flag.end())
    {
        bc_flag = it->second;
        return 0;
    }
    else
        return -1;
}

    