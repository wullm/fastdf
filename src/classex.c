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

#include <stdio.h>
#include "../include/classex.h"

#ifdef WITH_CLASS
#include <class.h>

int readPerturbData(struct perturb_data *data, struct units *us,
                    struct perturbations *pt, struct background *ba) {
    /* Try getting a source */
    int index_md = pt->index_md_scalars;  // scalar mode
    int index_ic = 0;                     // index of the initial condition
    int index_tp;                         // type of source function

    /* Size of the perturbations */
    size_t k_size = pt->k_size[index_md];
    size_t tau_size = pt->tau_size;

    /* The vectors to be transfered from the CLASS structs to the data struct */
    const int NumDesiredFunctions = 3;
    int ClassPerturbIndices[3] = {pt->index_tp_phi, pt->index_tp_psi, pt->index_tp_delta_ncdm1};
    int ClassBackgroundIndices[3] = {-1, -1, ba->index_bg_rho_ncdm1};
    char DesiredFunctions[3][50] = {"phi", "psi", "d_ncdm[0]"};

    /* The number of transfer functions to be read */
    const size_t n_functions = NumDesiredFunctions;
    
    /* Little Hubble h */
    // const double h = ba->h;
    
    /* CLASS to internal units conversion factor */
    const double unit_length_factor = MPC_METRES / us->UnitLengthMetres;
    const double unit_time_factor = unit_length_factor / us->SpeedOfLight;
    
    /* Vector of the wavenumbers */
    data->k_size = k_size;
    data->k = (double *)calloc(k_size, sizeof(double));
    
    /* Vector of the conformal times at which the perturbation is sampled */
    data->tau_size = tau_size;
    data->log_tau = (double *)calloc(tau_size, sizeof(double));
    data->redshift = (double *)calloc(tau_size, sizeof(double));
    data->Hubble_H = (double *)calloc(tau_size, sizeof(double));
    
    /* The number of transfer functions to be read */
    data->n_functions = n_functions;
    
    /* Vector with the transfer functions T(tau, k) */
    data->delta = (double *)calloc(n_functions * k_size * tau_size, sizeof(double));
    
    /* Vector of background quantities at each time Omega(tau) */
    data->Omega = (double *)calloc(n_functions * tau_size, sizeof(double));
    
    /* Read out the log conformal times */
    for (size_t index_tau = 0; index_tau < tau_size; index_tau++) {
        /* Convert tau from Mpc to U_T */
        double tau = pt->tau_sampling[index_tau] * unit_time_factor;
        data->log_tau[index_tau] = log(tau);
    }
    
    /* Read out the wavenumbers */
    for (size_t index_k = 0; index_k < k_size; index_k++) {
        /* Note: CLASS exports transfer function files with k in h/Mpc,
         * but internally it uses 1/Mpc. */
    
         /* Convert k from 1/Mpc to 1/U_L */
        double k = pt->k[index_md][index_k] / unit_length_factor;
        data->k[index_k] = k;
    }
    
    /* Transfer the titles */
    data->titles = malloc(data->n_functions * sizeof(char*));
    int j = 0;
    for (int i=0; i<NumDesiredFunctions; i++) {
        if (ClassPerturbIndices[i] >= 0) { //has a matching index
            data->titles[j] = malloc(strlen(DesiredFunctions[i]) + 1);
            strcpy(data->titles[j], DesiredFunctions[i]);
            j++;
        }
    }
        
    /* The index for data->delta, not CLASS index, nor index in titles string */
    int index_func = 0;
    /* Convert and store the transfer functions */
    for (size_t i = 0; i < NumDesiredFunctions; i++) {
        /* Ignore functions that have no matching CLASS index */
        if (ClassPerturbIndices[i] < 0) continue;
    
        /* Determine the unit conversion factor */
        char *title = DesiredFunctions[i];
        double unit_factor = unitConversionFactor(title, unit_length_factor, unit_time_factor);
    
        printf("Unit conversion factor for '%s' is %f\n", title, unit_factor);
    
        /* For each timestep and wavenumber */
        for (size_t index_tau = 0; index_tau < tau_size; index_tau++) {
            for (size_t index_k = 0; index_k < k_size; index_k++) {
                /* Transfer the corresponding data */
                index_tp = ClassPerturbIndices[i];  // CLASS index
                double p = pt->sources[index_md][index_ic * pt->tp_size[index_md] +
                                                index_tp][index_tau * k_size + index_k];
    
                /* Convert transfer functions from CLASS format to CAMB/HeWon/dexm/
                *  Eisenstein-Hu format by multiplying by -1/k^2.
                */
                double k = pt->k[index_md][index_k] / unit_length_factor;
                double T = -p/k/k;
    
                /* Convert from CLASS units to output units */
                T *= unit_factor;
    
                data->delta[tau_size * k_size * index_func + k_size * index_tau + index_k] = T;
            }
        }
        index_func++;
    }
    
    printf("\n");
    
    /* Finally, we also want to get the redshifts and background densities.
     * To do this, we need to let CLASS populate a vector of background
     * quantities at each time step.
     */
    
    /* Allocate array for background quantities */
    double *pvecback = malloc(ba->bg_size * sizeof(double));
    int last_index; //not used, but is output by CLASS
    
    /* Read out the redshifts and background densities */
    for (size_t index_tau = 0; index_tau < tau_size; index_tau++) {
        /* Conformal time in Mpc/c (the internal time unit in CLASS) */
        double tau = pt->tau_sampling[index_tau];
    
        /* Make CLASS evaluate background quantities at this time*/
        background_at_tau(ba, tau, ba->bg_size, 0,
                          &last_index, pvecback);
    
        /* The scale-factor and redshift */
        double a = pvecback[ba->index_bg_a];
        double z = 1./a - 1.;
    
        /* The critical density at this redshift in CLASS units */
        double rho_crit = pvecback[ba->index_bg_rho_crit];
    
        /* Retrieve background quantities in CLASS units */
        double H = pvecback[ba->index_bg_H];
        double H_prime = pvecback[ba->index_bg_H_prime];
        double D = pvecback[ba->index_bg_D];
        double f = pvecback[ba->index_bg_f];
        double D_prime = f * a * H * D;
        double a_prime = a*a*H;
    
        /* Store the redshift */
        data->redshift[index_tau] = z;
    
        /* The Hubble constant in 1/U_T and its conformal derivative in 1/U_T^2 */
        data->Hubble_H[index_tau] = H / unit_time_factor;
        
        /* Read out the background densities */
        index_func = 0;
        for (size_t i = 0; i < NumDesiredFunctions; i++) {
            /* Ignore functions that have no matching CLASS perturbation index */
            if (ClassPerturbIndices[i] < 0) continue;
        
            /* For functions that also have a CLASS background index */
            if (ClassBackgroundIndices[i] >= 0) {
                /* The density corresponding to this function (CLASS units) */
                double rho = pvecback[ClassBackgroundIndices[i]];
        
                /* Fraction of the critical density (dimensionless) */
                double Omega = rho/rho_crit;
        
                /* Store the dimensionless background density */
                data->Omega[tau_size * index_func + index_tau] = Omega;
            } else {
                //We just store zero, to keep it simple. The memory was calloc'ed.
            }
        
            index_func++;
        }
    }
    
    /* Done with the CLASS background vector */
    free(pvecback);
    
    printf("\n");
    printf("The perturbations are sampled at %zu * %zu points.\n", k_size, tau_size);

    return 0;
}

int run_class(struct perturb_data *data, struct units *us, 
              struct perturb_params *ptpars, char *ini_filename) {

    /* Define the CLASS structures */
    struct precision pr;  /* for precision parameters */
    struct background ba; /* for cosmological background */
    struct thermodynamics th;     /* for thermodynamics */
    struct perturbations pt;   /* for source functions */
    struct transfer ptr;   /* for transfer functions */
    struct primordial ppr;  /* for primordial functions */
    struct harmonic phr;  /* for harmonic functions */
    struct fourier pfo;   /* for fourier */
    struct lensing ple;   /* for lensing */
    struct distortions psd;    /* for distortions */
    struct output op;     /* for output files */
    ErrorMsg errmsg;      /* for CLASS-specific error messages */
    
    // char inifile[50] = "DES3_300_approx.ini";
    // char ini_filename[50] = "input_class_parameters.ini";
    
    /* If no class .ini file was specified, stop. */
    if (ini_filename[0] == '\0') {
        printf("No CLASS file specified!\n");
        return 1;
    }
    
    printf("Running CLASS on parameter file '%s'.\n\n", ini_filename);
            
    int class_argc = 2;
    char *class_argv[] = {"", ini_filename, ""};

    if (input_init(class_argc, class_argv, &pr, &ba, &th, &pt,
                   &ptr, &ppr, &phr, &pfo, &ple, &psd, &op, errmsg) == _FAILURE_) {
        printf("Error running input_init_from_arguments \n=>%s\n", errmsg);
    }

    if (background_init(&pr, &ba) == _FAILURE_) {
        printf("Error running background_init \n%s\n", ba.error_message);
    }

    if (thermodynamics_init(&pr, &ba, &th) == _FAILURE_) {
        printf("Error in thermodynamics_init \n%s\n", th.error_message);
    }

    if (perturbations_init(&pr, &ba, &th, &pt) == _FAILURE_) {
        printf("Error in perturbations_init \n%s\n", pt.error_message);
    }

    printf("\n");
    
    /* Read perturb data */
    readPerturbData(data, us, &pt, &ba);
    
    printf("We have read out %d functions.\n", data->n_functions);
    
    /* Retrieve the number of ncdm species and their masses in eV */
    ptpars->N_ncdm = ba.N_ncdm;
    ptpars->M_ncdm_eV = malloc(ptpars->N_ncdm * sizeof(double));
    for (int i = 0; i < ptpars->N_ncdm; i++) {
        ptpars->M_ncdm_eV[i] = ba.m_ncdm_in_eV[i];
    }
    
    /* Retrieve temperatures (ncdm temperatures in units of T_CMB) */
    ptpars->T_CMB = ba.T_cmb * us->UnitTemperatureKelvin;
    ptpars->T_ncdm = malloc(ptpars->N_ncdm * sizeof(double));
    for (int i = 0; i < ptpars->N_ncdm; i++) {
        ptpars->T_ncdm[i] = ba.T_ncdm[i];
    }
    
    /* Retrieve other background parameters */    
    ptpars->h = ba.h;
    ptpars->Omega_lambda = ba.Omega0_lambda;
    ptpars->Omega_k = ba.Omega0_k;
    ptpars->Omega_m = ba.Omega0_m;
    ptpars->Omega_b = ba.Omega0_b;
    ptpars->Omega_ur = ba.Omega0_ur;

    /* Here neutrinos do not contribute to Omega_m, but in CLASS they do */
    for (int i=0; i<ptpars->N_ncdm; i++) {
        ptpars->Omega_m -= ba.Omega0_ncdm[i];
    }
    
    printf("\nShutting CLASS down again.\n");
    
    /* Pre-empt segfault in CLASS if there is no interacting dark radiation */
    if (ba.has_idr == _FALSE_) {
        pt.alpha_idm_dr = (double *)malloc(0);
        pt.beta_idr = (double *)malloc(0);
    }

    /* Close CLASS again */
    if (perturbations_free(&pt) == _FAILURE_) {
        printf("Error in freeing class memory \n%s\n", pt.error_message);
        return 1;
    }

    if (thermodynamics_free(&th) == _FAILURE_) {
        printf("Error in thermodynamics_free \n%s\n", th.error_message);
        return 1;
    }

    if (background_free(&ba) == _FAILURE_) {
        printf("Error in background_free \n%s\n", ba.error_message);
        return 1;
    }
    
    
    
    return 0;
}

#endif