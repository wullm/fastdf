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

#ifndef OUTPUT_H
#define OUTPUT_H

#include <hdf5.h>

/* General methods */
hid_t openFile(const char *fname);
hid_t createFile(const char *fname);
int writeFieldHeader(double boxlen, hid_t h_file);

/* Methods for contiguous arrays (analogous to MPI versions in output_mpi.h) */
int writeFieldFile(const double *box, int N, double box_len, const char *fname);
int writeFieldData(const double *box, hid_t h_file);

#endif
