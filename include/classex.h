/*******************************************************************************
 * This file is part of FastDF.
 * Copyright (c) 2022 Willem Elbers (whe@willemelbers.com)
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

/* Did we compile with CLASS? */
#ifdef WITH_CLASS

#ifndef CLASSEX_H
#define CLASSEX_H

#include <hdf5.h>
#include "../include/perturb_data.h"
#include "../include/input.h"

int run_class(struct perturb_data *data, struct units *us, 
              struct perturb_params *ptpars, char *ini_filename);


#endif
#endif
