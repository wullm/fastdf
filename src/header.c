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
#include <string.h>
#include <math.h>
#include "../include/header.h"

int writeHeaderAttributes(struct params *pars, struct units *us,
                          long long int Npart_local, long long int Npart_total,
                          hid_t h_file) {

    /* Create the Header group */
    hid_t h_grp = H5Gcreate(h_file, "/Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Retrieve the number of ranks */
    int MPI_Rank_Count;
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);

    /* Create dataspace for BoxSize attribute */
    const hsize_t arank = 1;
    const hsize_t adims_three[1] = {3}; //3D space
    hid_t h_aspace = H5Screate_simple(arank, adims_three, NULL);

    /* Create the BoxSize attribute and write the data */
    hid_t h_attr = H5Acreate1(h_grp, "BoxSize", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    double boxlen = pars->BoxLen;
    double boxsize[3] = {boxlen, boxlen, boxlen};
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, boxsize);
    H5Aclose(h_attr);

    /* Change dataspace dimensions to scalar value attributes */
    const hsize_t adims_single[1] = {1};
    H5Sset_extent_simple(h_aspace, arank, adims_single, NULL);

    /* Create the Dimension attribute and write the data */
    int dimension = 3;
    h_attr = H5Acreate1(h_grp, "Dimension", H5T_NATIVE_INT, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_INT, &dimension);
    H5Aclose(h_attr);

    /* Create the Redshift attribute and write the data */
    double z_final = 1./pars->ScaleFactorEnd - 1;
    h_attr = H5Acreate1(h_grp, "Redshift", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, &z_final);
    H5Aclose(h_attr);

    /* Create the Flag_Entropy_ICs attribute and write the data */
    int flag_entropy = 0;
    h_attr = H5Acreate1(h_grp, "Flag_Entropy_ICs", H5T_NATIVE_INT, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_INT, &flag_entropy);
    H5Aclose(h_attr);

    /* Create the NumFilesPerSnapshot attribute and write the data */
    int num_files_per_snapshot = pars->DistributedFiles ? MPI_Rank_Count : 1;
    h_attr = H5Acreate1(h_grp, "NumFilesPerSnapshot", H5T_NATIVE_INT, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_INT, &num_files_per_snapshot);
    H5Aclose(h_attr);

    /* Change dataspace dimensions to particle type attributes */
    const hsize_t adims_pt[1] = {7}; //particle type 0-6
    H5Sset_extent_simple(h_aspace, arank, adims_pt, NULL);

    /* Collect particle type attributes using the ExportNames */
    long long int numparts_local[7] = {0, 0, 0, 0, 0, 0, Npart_local};
    long long int numparts_total[7] = {0, 0, 0, 0, 0, 0, Npart_total};
    long long int numparts_high_word[7] = {0, 0, 0, 0, 0, 0, Npart_total >> 32};
    double mass_table[7] = {0., 0., 0., 0., 0., 0., 0.};

    /* Create the NumPart_ThisFile attribute and write the data */
    h_attr = H5Acreate1(h_grp, "NumPart_ThisFile", H5T_NATIVE_LONG, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_LONG, numparts_local);
    H5Aclose(h_attr);

    /* Create the NumPart_Total attribute and write the data */
    h_attr = H5Acreate1(h_grp, "NumPart_Total", H5T_NATIVE_LONG, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_LONG, numparts_total);
    H5Aclose(h_attr);

    /* Create the NumPart_Total_HighWord attribute and write the data */
    h_attr = H5Acreate1(h_grp, "NumPart_Total_HighWord", H5T_NATIVE_LONG, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_LONG, numparts_high_word);
    H5Aclose(h_attr);

    /* Create the MassTable attribute and write the data */
    h_attr = H5Acreate1(h_grp, "MassTable", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, mass_table);
    H5Aclose(h_attr);

    /* Close the attribute dataspace */
    H5Sclose(h_aspace);

    /* Close the Header group */
    H5Gclose(h_grp);

    /* Create the Cosmology group */
    h_grp = H5Gcreate(h_file, "/Cosmology", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Create dataspace for scalar value attributes */
    h_aspace = H5Screate_simple(arank, adims_single, NULL);

    /* Create the Redshift attribute and write the data */
    h_attr = H5Acreate1(h_grp, "Redshift", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, &z_final);
    H5Aclose(h_attr);

    /* Close the attribute dataspace */
    H5Sclose(h_aspace);

    /* Close the Cosmology group */
    H5Gclose(h_grp);

    /* Create the Units group */
    h_grp = H5Gcreate(h_file, "/Units", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Create dataspace for scalar value attributes */
    h_aspace = H5Screate_simple(arank, adims_single, NULL);

    /* Determine the units used */
    double unit_mass_cgs = us->UnitMassKilogram * 1000;
    double unit_length_cgs = us->UnitLengthMetres * 100;
    double unit_time_cgs = us->UnitTimeSeconds;
    double unit_temperature_cgs = us->UnitTemperatureKelvin;
    double unit_current_cgs = us->UnitCurrentAmpere;

    /* Write the internal unit system */
    h_attr = H5Acreate1(h_grp, "Unit mass in cgs (U_M)", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, &unit_mass_cgs);
    H5Aclose(h_attr);

    h_attr = H5Acreate1(h_grp, "Unit length in cgs (U_L)", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, &unit_length_cgs);
    H5Aclose(h_attr);

    h_attr = H5Acreate1(h_grp, "Unit time in cgs (U_t)", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, &unit_time_cgs);
    H5Aclose(h_attr);

    h_attr = H5Acreate1(h_grp, "Unit temperature in cgs (U_T)", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, &unit_temperature_cgs);
    H5Aclose(h_attr);

    h_attr = H5Acreate1(h_grp, "Unit current in cgs (U_I)", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, &unit_current_cgs);
    H5Aclose(h_attr);

    /* Close the attribute dataspace */
    H5Sclose(h_aspace);

    /* Close the Cosmology group */
    H5Gclose(h_grp);


    return 0;
}
