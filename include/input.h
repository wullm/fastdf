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

#ifndef INPUT_H
#define INPUT_H

#define DEFAULT_STRING_LENGTH 150

#define KM_METRES 1000
#define MPC_METRES 3.085677581282e22

#define SPEED_OF_LIGHT_METRES_SECONDS 2.99792e8
#define GRAVITY_G_SI_UNITS 6.67428e-11 // m^3 / kg / s^2
#define PLANCK_CONST_SI_UNITS 6.62607015e-34 //J s
#define BOLTZMANN_CONST_SI_UNITS 1.380649e-23 //J / K
#define ELECTRONVOLT_SI_UNITS 1.602176634e-19 // J

/* The .ini parser library is minIni */
#include "../parser/minIni.h"
// #include "../include/output.h"

#include <mpi.h>

struct params {

    /* Box parameters */
    int GridSize;
    double BoxLen;
    long long int CubeRootNumber;
    long long int NumPartGenerate;
    long long int FirstID;
    char *GaussianRandomFieldFile;
    char *GaussianRandomFieldDataset;
    char InvertField;

    /* CLASS parameter file names */
    char *ClassIniFile;

    /* Parameters of the primordial power spectrum (optional) */
    char NormalizeGaussianField;
    char AssumeMonofonicNormalization;
    double PrimordialScalarAmplitude;
    double PrimordialSpectralIndex;
    double PrimordialPivotScale;
    double PrimordialRunning;
    double PrimordialRunningSecond;

    /* Simulation parameters */
    char *Name;
    char *PerturbFile;
    char *TransferFunctionDensity;
    char *Gauge;
    double ScaleFactorBegin;
    double ScaleFactorEnd;
    double ScaleFactorStep;
    double RecomputeTrigger;
    double RecomputeScaleRef;
    char InterpolationOrder;

    /* Alternative Equation of Motion Mode */
    char AlternativeEquations;

    /* Use non-symplectic formulation of equations of motion */
    char NonSymplecticEquations;

    /* Output parameters */
    char *OutputDirectory;
    char *OutputFilename;
    char *InputDirectory;
    char *InputFilename;
    char *ExportName;
    char *VelocityType;
    char OutputFields;
    char IncludeHubbleFactors;
    char DistributedFiles;
    char Verbosity;

    /* MPI rank (generated automatically) */
    int rank;
};

struct units {
    double UnitLengthMetres;
    double UnitTimeSeconds;
    double UnitMassKilogram;
    double UnitTemperatureKelvin;
    double UnitCurrentAmpere;

    /* Physical constants in internal units */
    double SpeedOfLight;
    double GravityG;
    double hPlanck;
    double kBoltzmann;
    double ElectronVolt;
};

int initParams(struct params *pars);
int readParams(struct params *parser, const char *fname);
int readUnits(struct units *us, const char *fname);
int setPhysicalConstants(struct units *us);

int cleanParams(struct params *parser);

int readFieldFile(double **box, int *N, double *box_len, const char *fname);
int readFieldFileDataSet(double **box, int *N, double *box_len,
                         const char *fname, const char *dset_name);

int readFieldFile_MPI(double **box, int *N, double *box_len, MPI_Comm comm,
                      const char *fname);
int fileExists(const char *fname);
int groupExists(const char *fname, const char *group_name);

static inline void generateFieldFilename(const struct params *pars, char *fname,
                                         const char *Identifier, const char *title,
                                         const char *extra) {
    sprintf(fname, "%s/%s_%s%s.%s", pars->OutputDirectory, title, extra,
            Identifier, "hdf5");
}


#endif
