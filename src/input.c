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
#include <hdf5.h>
#include <assert.h>
#include <math.h>
#include "../include/input.h"

int readParams(struct params *pars, const char *fname) {
     pars->FirstID = ini_getl("Simulation", "FirstID", 0, fname);

     /* Desired particle numbers */
     pars->CubeRootNumber = ini_getl("Simulation", "CubeRootNumber", 0, fname);
     pars->NumPartGenerate = pars->CubeRootNumber * pars->CubeRootNumber * pars->CubeRootNumber;

     pars->ScaleFactorBegin = ini_getd("Simulation", "ScaleFactorBegin", 0.005, fname);
     pars->ScaleFactorEnd = ini_getd("Simulation", "ScaleFactorEnd", 0.01, fname);
     pars->ScaleFactorStep = ini_getd("Simulation", "ScaleFactorStep", 0.05, fname);
     pars->RecomputeTrigger = ini_getd("Simulation", "RecomputeTrigger", 0.01, fname);
     pars->RecomputeScaleRef = ini_getd("Simulation", "RecomputeScaleRef", 0.0, fname);
     pars->CubeRootNumber = ini_getl("Simulation", "CubeRootNumber", 0, fname);
     pars->InvertField = ini_getd("Box", "InvertField", 0, fname);

     pars->AlternativeEquations = ini_getl("Simulation", "AlternativeEquations", 0, fname);

     pars->OutputFields = ini_getl("Output", "OutputFields", 1, fname);
     
     
     pars->BackwardMode = ini_getl("Simulation", "BackwardMode", 0, fname);
     pars->CentralRatio = ini_getd("Simulation", "CentralRatio", 0.0, fname);
     pars->CentralRadius = ini_getd("Simulation", "CentralRadius", 0.0, fname);
     pars->ScaleFactorTarget = ini_getd("Simulation", "ScaleFactorTarget", 1.0, fname);

     /* Parameters of the primordial power spectrum (optional) */
     pars->NormalizeGaussianField = ini_getl("PrimordialSpectrum", "NormalizeGaussianField", 0, fname);
     pars->AssumeMonofonicNormalization = ini_getl("PrimordialSpectrum", "AssumeMonofonicNormalization", 0, fname);
     pars->PrimordialScalarAmplitude = ini_getd("PrimordialSpectrum", "ScalarAmplitude", 2e-9, fname);
     pars->PrimordialSpectralIndex = ini_getd("PrimordialSpectrum", "SpectralIndex", 0.96, fname);
     pars->PrimordialPivotScale = ini_getd("PrimordialSpectrum", "PivotScale", 0.05, fname);

     /* Read strings */
     int len = DEFAULT_STRING_LENGTH;
     pars->OutputDirectory = malloc(len);
     pars->Name = malloc(len);
     pars->ExportName = malloc(len);
     pars->InputDirectory = malloc(len);
     pars->InputFilename = malloc(len);
     pars->OutputFilename = malloc(len);
     pars->PerturbFile = malloc(len);
     pars->GaussianRandomFieldFile = malloc(len);
     pars->TransferFunctionDensity = malloc(len);
     pars->Gauge = malloc(len);
     ini_gets("Output", "Directory", "./output", pars->OutputDirectory, len, fname);
     ini_gets("Simulation", "Name", "No Name", pars->Name, len, fname);
     ini_gets("Simulation", "Gauge", "Newtonian", pars->Gauge, len, fname);
     ini_gets("Input", "Directory", "./input", pars->InputDirectory, len, fname);
     ini_gets("Input", "Filename", "particles.hdf5", pars->InputFilename, len, fname);
     ini_gets("Output", "Filename", "particles.hdf5", pars->OutputFilename, len, fname);
     ini_gets("Output", "ExportName", "PartType6", pars->ExportName, len, fname);
     ini_gets("PerturbData", "File", "", pars->PerturbFile, len, fname);
     ini_gets("PerturbData", "TransferFunctionDensity", "", pars->TransferFunctionDensity, len, fname);
     ini_gets("Box", "GaussianRandomFieldFile", "", pars->GaussianRandomFieldFile, len, fname);

     return 0;
}

int readUnits(struct units *us, const char *fname) {
    /* Internal units */
    us->UnitLengthMetres = ini_getd("Units", "UnitLengthMetres", 1.0, fname);
    us->UnitTimeSeconds = ini_getd("Units", "UnitTimeSeconds", 1.0, fname);
    us->UnitMassKilogram = ini_getd("Units", "UnitMassKilogram", 1.0, fname);
    us->UnitTemperatureKelvin = ini_getd("Units", "UnitTemperatureKelvin", 1.0, fname);
    us->UnitCurrentAmpere = ini_getd("Units", "UnitCurrentAmpere", 1.0, fname);

    /* Some physical constants */
    us->SpeedOfLight = SPEED_OF_LIGHT_METRES_SECONDS * us->UnitTimeSeconds
                        / us->UnitLengthMetres;
    us->GravityG = GRAVITY_G_SI_UNITS * us->UnitTimeSeconds * us->UnitTimeSeconds
                    / us->UnitLengthMetres / us->UnitLengthMetres / us->UnitLengthMetres
                    * us->UnitMassKilogram; // m^3 / kg / s^2 to internal
    us->hPlanck = PLANCK_CONST_SI_UNITS / us->UnitMassKilogram / us->UnitLengthMetres
                    / us->UnitLengthMetres * us->UnitTimeSeconds; //J*s = kg*m^2/s
    us->kBoltzmann = BOLTZMANN_CONST_SI_UNITS / us->UnitMassKilogram / us->UnitLengthMetres
                    / us->UnitLengthMetres * us->UnitTimeSeconds * us->UnitTimeSeconds
                    * us->UnitTemperatureKelvin; //J/K = kg*m^2/s^2/K
    us->ElectronVolt = ELECTRONVOLT_SI_UNITS / us->UnitMassKilogram / us->UnitLengthMetres
                    / us->UnitLengthMetres * us->UnitTimeSeconds
                    * us->UnitTimeSeconds; // J = kg*m^2/s^2

    return 0;
}


int cleanParams(struct params *pars) {
    free(pars->OutputDirectory);
    free(pars->Name);
    free(pars->ExportName);
    free(pars->InputDirectory);
    free(pars->InputFilename);
    free(pars->OutputFilename);
    free(pars->PerturbFile);
    free(pars->GaussianRandomFieldFile);
    free(pars->TransferFunctionDensity);
    free(pars->Gauge);

    return 0;
}

/* Read 3D box from disk, allocating memory and storing the grid dimensions */
int readFieldFile(double **box, int *N, double *box_len, const char *fname) {
    /* Open the hdf5 file */
    hid_t h_file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);

    /* Open the Header group */
    hid_t h_grp = H5Gopen(h_file, "Header", H5P_DEFAULT);

    /* Read the size of the field */
    hid_t h_attr, h_err;
    double boxsize[3];

    /* Open and read out the attribute */
    h_attr = H5Aopen(h_grp, "BoxSize", H5P_DEFAULT);
    h_err = H5Aread(h_attr, H5T_NATIVE_DOUBLE, &boxsize);
    if (h_err < 0) {
        printf("Error reading hdf5 attribute '%s'.\n", "BoxSize");
        return 1;
    }

    /* It should be a cube */
    assert(boxsize[0] == boxsize[1]);
    assert(boxsize[1] == boxsize[2]);
    *box_len = boxsize[0];

    /* Close the attribute, and the Header group */
    H5Aclose(h_attr);
    H5Gclose(h_grp);

    /* Open the Field group */
    h_grp = H5Gopen(h_file, "Field", H5P_DEFAULT);

    /* Open the Field dataset */
    hid_t h_data = H5Dopen2(h_grp, "Field", H5P_DEFAULT);

    /* Open the dataspace and fetch the grid dimensions */
    hid_t h_space = H5Dget_space(h_data);
    int ndims = H5Sget_simple_extent_ndims(h_space);
    hsize_t *dims = malloc(ndims * sizeof(hsize_t));
    H5Sget_simple_extent_dims(h_space, dims, NULL);
    int read_N = dims[0];

    /* We should be in 3D */
    if (ndims != 3) {
        printf("Number of dimensions %d != 3.\n", ndims);
        return 2;
    }
    /* It should be a cube (but allow for padding in the last dimension) */
    if (read_N != dims[1] || (read_N != dims[2] && (read_N+2) != dims[2])) {
        printf("Non-cubic grid size (%lld, %lld, %lld).\n", dims[0], dims[1], dims[2]);
        return 2;
    }
    /* Store the grid size */
    *N = read_N;

    /* Allocate the array (without padding) */
    *box = malloc(read_N * read_N * read_N * sizeof(double));

    /* The hyperslab that should be read (needed in case of padding) */
    const hsize_t space_rank = 3;
    const hsize_t space_dims[3] = {read_N, read_N, read_N}; //3D space

    /* Offset of the hyperslab */
    const hsize_t space_offset[3] = {0, 0, 0};

    /* Create memory space for the chunk */
    hid_t h_memspace = H5Screate_simple(space_rank, space_dims, NULL);
    H5Sselect_hyperslab(h_space, H5S_SELECT_SET, space_offset, NULL, space_dims, NULL);

    /* Read out the data */
    h_err = H5Dread(h_data, H5T_NATIVE_DOUBLE, h_memspace, h_space, H5P_DEFAULT, *box);
    if (h_err < 0) {
        printf("Error reading hdf5 file '%s'.\n", fname);
        return 1;
    }

    /* Close the dataspaces and dataset */
    H5Sclose(h_memspace);
    H5Sclose(h_space);
    H5Dclose(h_data);

    /* Close the Field group */
    H5Gclose(h_grp);

    /* Close the file */
    H5Fclose(h_file);

    /* Free memory */
    free(dims);

    return 0;
}

int readFieldFile_MPI(double **box, int *N, double *box_len, MPI_Comm comm,
                      const char *fname) {

    /* Property list for MPI file access */
    hid_t prop_faxs = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(prop_faxs, comm, MPI_INFO_NULL);

    /* Open the hdf5 file */
    hid_t h_file = H5Fopen(fname, H5F_ACC_RDONLY, prop_faxs);
    H5Pclose(prop_faxs);

    /* Open the Header group */
    hid_t h_grp = H5Gopen(h_file, "Header", H5P_DEFAULT);

    /* Read the size of the field */
    hid_t h_attr, h_err;
    double boxsize[3];

    /* Open and read out the attribute */
    h_attr = H5Aopen(h_grp, "BoxSize", H5P_DEFAULT);
    h_err = H5Aread(h_attr, H5T_NATIVE_DOUBLE, &boxsize);
    if (h_err < 0) {
        printf("Error reading hdf5 attribute '%s'.\n", "BoxSize");
        return 1;
    }

    /* It should be a cube */
    assert(boxsize[0] == boxsize[1]);
    assert(boxsize[1] == boxsize[2]);
    *box_len = boxsize[0];

    /* Close the attribute, and the Header group */
    H5Aclose(h_attr);
    H5Gclose(h_grp);

    /* Open the Field group */
    h_grp = H5Gopen(h_file, "Field", H5P_DEFAULT);

    /* Open the Field dataset */
    hid_t h_data = H5Dopen2(h_grp, "Field", H5P_DEFAULT);

    /* Get the file dataspace */
    hid_t h_space = H5Dget_space(h_data);

    /* Open the dataspace and fetch the grid dimensions */
    int ndims = H5Sget_simple_extent_ndims(h_space);
    hsize_t *dims = malloc(ndims * sizeof(hsize_t));
    H5Sget_simple_extent_dims(h_space, dims, NULL);
    int read_N = dims[0];

    /* We should be in 3D */
    if (ndims != 3) {
        printf("Number of dimensions %d != 3.\n", ndims);
        return 2;
    }
    /* It should be a cube (but allow for padding in the last dimension) */
    if (read_N != dims[1] || (read_N != dims[2] && (read_N+2) != dims[2])) {
        printf("Non-cubic grid size (%lld, %lld, %lld).\n", dims[0], dims[1], dims[2]);
        return 2;
    }
    /* Store the grid size */
    *N = read_N;

    /* Allocate the array (without padding) */
    *box = malloc(read_N * read_N * read_N * sizeof(double));

    /* The chunk in question */
    const hsize_t chunk_rank = 3;
    const hsize_t chunk_dims[3] = {read_N, read_N, read_N};

    /* Offset of the chunk inside the grid */
    const hsize_t chunk_offset[3] = {0, 0, 0};

    /* Create memory space for the chunk */
    hid_t h_memspace = H5Screate_simple(chunk_rank, chunk_dims, NULL);
    H5Sselect_hyperslab(h_space, H5S_SELECT_SET, chunk_offset, NULL, chunk_dims, NULL);

    /* Read the data */
    h_err = H5Dread(h_data, H5T_NATIVE_DOUBLE, h_memspace, h_space, H5P_DEFAULT, *box);
    if (h_err < 0) {
        printf("Error: reading chunk of hdf5 data.\n");
        return 1;
    }

    /* Close the dataset, corresponding dataspace, and the Field group */
    H5Dclose(h_data);
    H5Sclose(h_space);
    H5Sclose(h_memspace);
    H5Gclose(h_grp);

    /* Close the file */
    H5Fclose(h_file);

    return 0;
}

// hid_t openFile_MPI(MPI_Comm comm, const char *fname) {
//     /* Property list for MPI file access */
//     hid_t prop_faxs = H5Pcreate(H5P_FILE_ACCESS);
//     H5Pset_fapl_mpio(prop_faxs, comm, MPI_INFO_NULL);
//
//     /* Open the hdf5 file */
//     hid_t h_file = H5Fopen(fname, H5F_ACC_RDWR, prop_faxs);
//     H5Pclose(prop_faxs);
//
//     return h_file;
// }
