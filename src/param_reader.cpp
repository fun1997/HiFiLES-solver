/*!
* \file - param_reader.cpp
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
#include "../include/param_reader.h"

using namespace std;

param_reader::param_reader()
{
    
}

param_reader::param_reader(string fileName)
{
    this->fileName = fileName;
}

param_reader::~param_reader()
{
    if (optFile.is_open()) optFile.close();
}

void param_reader::setFile(string fileName)
{
    this->fileName = fileName;
}

void param_reader::openFile(void)
{
    optFile.open(fileName.c_str(), ifstream::in);
}

void param_reader::closeFile()
{
    optFile.close();
}
