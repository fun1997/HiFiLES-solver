/*!
 * \file global.h
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

#pragma once

#include "input.h"
#include "probe_input.h"
#define   MAX_PROCESSOR_AVAILABLE 5000
/*! input 'run_input' has global scope */
extern input run_input;
extern probe_input run_probe;
/*! double 'pi' has global scope */
extern const double pi;

//definitions
#define MAX_V_PER_F 4
#define MAX_F_PER_C 6
#define MAX_E_PER_C 12
#define MAX_V_PER_C 27


/** enumeration for cell type */
enum CTYPE {
    TRI     = 0,
    QUAD    = 1,
    TET     = 2,
    PRISM   = 3,
    HEX     = 4,
    PYRAMID = 5
};

/** enumeration for boundary conditions */
enum BCFLAG {
  SUB_IN_SIMP   = 0,
  SUB_OUT_SIMP  = 1,
  SUB_IN_CHAR   = 2,
  SUB_OUT_CHAR  = 3,
  SUP_IN        = 4,
  SUP_OUT       = 5,
  SLIP_WALL     = 6,
  CYCLIC        = 7,
  ISOTHERM_FIX  = 8,
  ADIABAT_FIX   = 9,
  CHAR          = 10,
  SLIP_WALL_DUAL= 11,
  AD_WALL       = 12
};

/*! environment variable specifying location of HiFiLES repository */
extern const char* HIFILES_DIR;