/*!
 * \file error.h
 * \brief _____________________________
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

#include <stdio.h>
/********************************************************
 * Prints the error message, the stack trace, and exits
 * ******************************************************/
#ifdef OMPI_MPI_H
#define FatalError(s)                                             \
  {                                                               \
    printf("Fatal error '%s' at %s:%d\n", s, __FILE__, __LINE__); \
    MPI_Abort(MPI_COMM_WORLD, 1);                                 \
  }
#else
#define FatalError(s)                                             \
  {                                                               \
    printf("Fatal error '%s' at %s:%d\n", s, __FILE__, __LINE__); \
    exit(1);                                                      \
  }
#endif
