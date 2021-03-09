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

#ifndef PERTURB_DATA_H
#define PERTURB_DATA_H

#include "input.h"

/* Data structure containing all cosmological perturbation transfer functions
 * T(k, log_tau) as a function of wavenumber and logarithm of conformal time.
 */
struct perturb_data {
    /* Number of wavenumbers */
    int k_size;
    /* Number of time bins */
    int tau_size;
    /* Number of transfer functions */
    int n_functions;
    /* The array of transfer functions (k_size * tau_size * n_functions) */
    double *delta;
    /* Vector of wavenumbers (k_size) */
    double *k;
    /* Vector of logarithmic conformal times (tau_size) */
    double *log_tau;
    /* Vector of corresponding redshifts (tau_size) */
    double *redshift;
    /* Vector of corresponding Hubble rate (tau_size) */
    double *Hubble_H;
    /* Titles of the transfer functions */
    char **titles;
    /* Vector of background densities (tau_size * n_functions) */
    double *Omega;
};

/* Extra cosmological parameters that can be extracted from perturb files */
struct perturb_params {
    int N_ncdm; //number of non-cold dark matter species (neutrinos)
    double *M_ncdm_eV; //masses of the ncdm particles in eV
    double *T_ncdm; //temperatures as fraction of T_CMB
    double T_CMB; //temperature in U_T
    double h; //Hubble parameter

    /* NB: Individual Omegas of fluid components are stored in ptdat */
    double Omega_lambda;
    double Omega_k;
    double Omega_m;
    double Omega_b;
    double Omega_ur;
};

/* Read the perturbation data from file */
int readPerturb(struct params *pars, struct units *us, struct perturb_data *pt);

/* Clean up the memory */
int cleanPerturb(struct perturb_data *pt);

/* Read extra cosmological parameters from file */
int readPerturbParams(struct params *pars, struct units *us,
                      struct perturb_params *ptpars);

/* Clean up the memory */
int cleanPerturbParams(struct perturb_params *ptpars);

/* Unit conversion factor for transfer functions, depending on the title. */
double unitConversionFactor(const char *title, double unit_length_factor,
                            double unit_time_factor);

/* Merge two transfer functions (e.g. cdm & barons) into one fluid */
int mergeTransferFunctions(struct perturb_data *pt, char *title_a, char *title_b,
                           double weight_a, double weight_b);
int mergeBackgroundDensities(struct perturb_data *pt, char *title_a, char *title_b,
                             double weight_a, double weight_b);

#endif
