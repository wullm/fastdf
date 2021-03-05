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

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../include/output.h"
#include "../include/fft.h"



hid_t openFile(const char *fname) {
    /* Open the hdf5 file */
    hid_t h_file = H5Fopen(fname, H5F_ACC_RDWR, H5P_DEFAULT);
    return h_file;
}

hid_t createFile(const char *fname) {
    /* Create the hdf5 file */
    hid_t h_file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    return h_file;
}

int writeFieldHeader(double boxlen, hid_t h_file) {
    /* Create the Header group */
    hid_t h_grp = H5Gcreate(h_file, "/Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Create dataspace for BoxSize attribute */
    const hsize_t arank = 1;
    const hsize_t adims[1] = {3}; //3D space
    hid_t h_aspace = H5Screate_simple(arank, adims, NULL);

    /* Create the BoxSize attribute and write the data */
    hid_t h_attr = H5Acreate1(h_grp, "BoxSize", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    double boxsize[3] = {boxlen, boxlen, boxlen};
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, boxsize);

    /* Close the attribute, corresponding dataspace, and the Header group */
    H5Aclose(h_attr);
    H5Sclose(h_aspace);
    H5Gclose(h_grp);

    return 0;
}

int writeFieldFile(const double *box, int N, double boxlen, const char *fname) {
    /* Create the hdf5 file */
    hid_t h_file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    /* Create the Header group */
    hid_t h_grp = H5Gcreate(h_file, "/Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Create dataspace for BoxSize attribute */
    const hsize_t arank = 1;
    const hsize_t adims[1] = {3}; //3D space
    hid_t h_aspace = H5Screate_simple(arank, adims, NULL);

    /* Create the BoxSize attribute and write the data */
    hid_t h_attr = H5Acreate1(h_grp, "BoxSize", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    double boxsize[3] = {boxlen, boxlen, boxlen};
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, boxsize);

    /* Close the attribute, corresponding dataspace, and the Header group */
    H5Aclose(h_attr);
    H5Sclose(h_aspace);
    H5Gclose(h_grp);

    /* Create the Field group */
    h_grp = H5Gcreate(h_file, "/Field", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Create dataspace for the field */
    const hsize_t frank = 3;
    const hsize_t fdims[3] = {N, N, N}; //3D space
    hid_t h_fspace = H5Screate_simple(frank, fdims, NULL);

    /* Create the dataset for the field */
    hid_t h_data = H5Dcreate(h_grp, "Field", H5T_NATIVE_DOUBLE, h_fspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Write the data */
    H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_fspace, h_fspace, H5P_DEFAULT, box);

    /* Close the dataset, corresponding dataspace, and the Field group */
    H5Dclose(h_data);
    H5Sclose(h_fspace);
    H5Gclose(h_grp);

    /* Close the file */
    H5Fclose(h_file);

    return 0;
}

/* Write 3-dimensional grid data in an existing hdf5 file, without any checks,
 * and assuming the dataspace is correct. */
int writeFieldData(const double *box, hid_t h_file) {
    /* Open the Header group */
    hid_t h_grp = H5Gopen(h_file, "Field", H5P_DEFAULT);

    /* Open the Field dataset */
    hid_t h_data = H5Dopen2(h_grp, "Field", H5P_DEFAULT);

    /* Get the file dataspace */
    hid_t h_space = H5Dget_space(h_data);

    /* Write the data */
    hid_t h_err = H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_space, h_space, H5P_DEFAULT, box);

    if (h_err < 0) {
        printf("Error: writing hdf5 data.\n");
        return 1;
    }

    /* Close the dataset, corresponding dataspace, and the Field group */
    H5Dclose(h_data);
    H5Sclose(h_space);
    H5Gclose(h_grp);

    return 0;
}
