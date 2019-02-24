/*!
 * \file flux.h
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

#include "hf_array.h"

/*! calculate inviscid flux in 2D */
void calc_invf_2d(hf_array<double>& in_u, hf_array<double>& out_f);

/*! calculate inviscid flux in 3D */
void calc_invf_3d(hf_array<double>& in_u, hf_array<double>& out_f);

/*! calculate viscous flux in 2D */
void calc_visf_2d(hf_array<double>& in_u, hf_array<double>& in_grad_u, hf_array<double>& out_f);

/*! calculate viscous flux in 3D */
void calc_visf_3d(hf_array<double>& in_u, hf_array<double>& in_grad_u, hf_array<double>& out_f);
