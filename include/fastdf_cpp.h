/*******************************************************************************
 * This file is part of Mitos.
 * Copyright (c) 2020 Willem Elbers (whe@willemelbers.com)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#ifndef FASTDF_CPP_H
#define FASTDF_CPP_H

#ifdef __cplusplus
extern "C" {
#endif

#include "input.h"
#include "output.h"
#include "message.h"
#include "delta_f.h"
#include "header.h"
#include "perturb_data.h"
#include "perturb_spline.h"
#include "particle.h"
#include "titles.h"
#include "cosmology.h"
#include "fft.h"
#include "fft_kernels.h"
#include "relativity.h"
#include "mesh_grav.h"

/* Did we compile with CLASS? */
#ifdef WITH_CLASS
#include "classex.h"
#endif

int run_fastdf(struct params *pars, struct units *us);

#ifdef __cplusplus
}
#endif

#endif
