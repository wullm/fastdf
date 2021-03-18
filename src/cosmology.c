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
#include <assert.h>
#include <math.h>
#include <gsl/gsl_integration.h>
#include "../include/cosmology.h"

int cosmo_table_length = 100;

const size_t workspace_size = 100000;

int readCosmology(struct cosmology *cosmo, struct units *us, hid_t h_file) {
    /* Store reference to units struct */
    cosmo->units = us;

    /* Check if the Cosmology group exists */
    hid_t h_status = H5Eset_auto1(NULL, NULL);  //turn off error printing
    h_status = H5Gget_objinfo(h_file, "/Cosmology", 0, NULL);

    /* If the group exists. */
    if (h_status == 0) {
        /* Open the Cosmology group */
        hid_t h_grp = H5Gopen(h_file, "Cosmology", H5P_DEFAULT);

        hid_t h_attr, h_err;

        /* Read the Omega_m attribute */
        h_attr = H5Aopen(h_grp, "Omega_m", H5P_DEFAULT);
        h_err = H5Aread(h_attr, H5T_NATIVE_DOUBLE, &cosmo->Omega_m);
        H5Aclose(h_attr);
        assert(h_err >= 0);

        /* Read the Omega_r attribute */
        h_attr = H5Aopen(h_grp, "Omega_r", H5P_DEFAULT);
        h_err = H5Aread(h_attr, H5T_NATIVE_DOUBLE, &cosmo->Omega_r);
        H5Aclose(h_attr);
        assert(h_err >= 0);

        /* Read the Omega_k attribute */
        h_attr = H5Aopen(h_grp, "Omega_k", H5P_DEFAULT);
        h_err = H5Aread(h_attr, H5T_NATIVE_DOUBLE, &cosmo->Omega_k);
        H5Aclose(h_attr);
        assert(h_err >= 0);

        /* Read the Omega_lambda attribute */
        h_attr = H5Aopen(h_grp, "Omega_lambda", H5P_DEFAULT);
        h_err = H5Aread(h_attr, H5T_NATIVE_DOUBLE, &cosmo->Omega_lambda);
        H5Aclose(h_attr);
        assert(h_err >= 0);

        /* Read the h attribute */
        h_attr = H5Aopen(h_grp, "h", H5P_DEFAULT);
        h_err = H5Aread(h_attr, H5T_NATIVE_DOUBLE, &cosmo->h);
        H5Aclose(h_attr);
        assert(h_err >= 0);

        /* Read the a_beg attribute */
        h_attr = H5Aopen(h_grp, "a_beg", H5P_DEFAULT);
        h_err = H5Aread(h_attr, H5T_NATIVE_DOUBLE, &cosmo->a_begin);
        H5Aclose(h_attr);
        assert(h_err >= 0);

        /* Read the a_end attribute */
        h_attr = H5Aopen(h_grp, "a_end", H5P_DEFAULT);
        h_err = H5Aread(h_attr, H5T_NATIVE_DOUBLE, &cosmo->a_end);
        H5Aclose(h_attr);
        assert(h_err >= 0);

        /* Read the h attribute */
        h_attr = H5Aopen(h_grp, "T_nu", H5P_DEFAULT);
        h_err = H5Aread(h_attr, H5T_NATIVE_DOUBLE, &cosmo->T_nu);
        H5Aclose(h_attr);
        assert(h_err >= 0);

        /* Read the N_nu attribute */
        h_attr = H5Aopen(h_grp, "N_nu", H5P_DEFAULT);
        h_err = H5Aread(h_attr, H5T_NATIVE_INT, &cosmo->N_nu);
        H5Aclose(h_attr);
        assert(h_err >= 0);

        if (cosmo->N_nu > 0) {
            /* Allocate memory for the neutrino mass array */
            cosmo->M_nu = malloc(cosmo->N_nu * sizeof(double));

            /* Read the N_nu attribute */
            h_attr = H5Aopen(h_grp, "M_nu", H5P_DEFAULT);
            h_err = H5Aread(h_attr, H5T_NATIVE_DOUBLE, cosmo->M_nu);
            H5Aclose(h_attr);
            assert(h_err >= 0);
        }

        /* Convert h to H_0 in internal units */
        cosmo->H_0 = cosmo->h * 100 * KM_METRES / MPC_METRES * us->UnitTimeSeconds;
        cosmo->log_a_begin = log(cosmo->a_begin);
        cosmo->log_a_end = log(cosmo->a_end);


        intregateCosmologyTables(cosmo);

        printf("The value was %f\n", cosmo->a_end);

        /* Close the Cosmology group */
        H5Gclose(h_grp);
    } else {
        printf("Error: Cosmology group does not exist.\n");
        return 1;
    }

    return 0;
}

int cleanCosmology(struct cosmology *cosmo) {
    if (cosmo->N_nu > 0) {
        free(cosmo->M_nu);
    }
    free(cosmo->drift_factor_table);
    free(cosmo->kick_factor_table);
    free(cosmo->log_a_table);
    return 0;
}

/* Hubble constant at redshift z */
double E_z(double a, struct cosmology *cosmo) {
    double a_inv = 1./a;
    double a_inv2 = a_inv * a_inv;
    double a_inv3 = a_inv2 * a_inv;
    double a_inv4 = a_inv2 * a_inv2;

    double Omega_m = cosmo->Omega_m;
    double Omega_r = cosmo->Omega_r;
    double Omega_k = cosmo->Omega_k;
    double Omega_lambda = cosmo->Omega_lambda;

    return sqrt(Omega_r * a_inv4 +
                Omega_m * a_inv3 +
                Omega_k * a_inv2 +
                Omega_lambda);
}

double drift_integrand(double a, void *params) {
    struct cosmology *cosmo = (struct cosmology*) params;

    double a_inv = 1./a;
    double H_0 = cosmo->H_0;
    double H = H_0 * E_z(a, cosmo);
    return (1. / H) * a_inv * a_inv * a_inv;
}

double kick_integrand(double a, void *params) {
    struct cosmology *cosmo = (struct cosmology*) params;

    double a_inv = 1./a;
    double H_0 = cosmo->H_0;
    double H = H_0 * E_z(a, cosmo);
    return (1. / H) * a_inv * a_inv;
}

int intregateCosmologyTables(struct cosmology *cosmo) {
    /* Allocate tables */
    cosmo->drift_factor_table = malloc(cosmo_table_length * sizeof(double));
    cosmo->kick_factor_table = malloc(cosmo_table_length * sizeof(double));
    cosmo->log_a_table = malloc(cosmo_table_length * sizeof(double));

    /* Create a table of scale-factors between a_begin and a_end */
    double log_a_begin = log(cosmo->a_begin);
    double log_a_end = log(cosmo->a_end);
    double log_a_step = (log_a_end - log_a_begin) /
                            ((double) cosmo_table_length);
    for (int i=0; i<cosmo_table_length; i++) {
        cosmo->log_a_table[i] = log_a_begin + (i + 1) * log_a_step;
    }

    /* Initalise the GSL workspace */
    gsl_integration_workspace *space =
      gsl_integration_workspace_alloc(workspace_size);

    double result, abserr;

    /* Compute the drift factor table */
    gsl_function F = {&drift_integrand, cosmo};
    for (int i = 0; i < cosmo_table_length; i++) {
        gsl_integration_qag(&F, cosmo->a_begin, exp(cosmo->log_a_table[i]), 0,
                            1.0e-10, workspace_size, GSL_INTEG_GAUSS61, space,
                            &result, &abserr);

        cosmo->drift_factor_table[i] = result;
    }

    /* Compute the kick factor table */
    F.function = &kick_integrand;
    for (int i = 0; i < cosmo_table_length; i++) {
        gsl_integration_qag(&F, cosmo->a_begin, exp(cosmo->log_a_table[i]), 0,
                            1.0e-10, workspace_size, GSL_INTEG_GAUSS61, space,
                            &result, &abserr);

        cosmo->kick_factor_table[i] = result;
    }

    /* Free the GSL workspace */
    gsl_integration_workspace_free(space);

    return 0;
}

static inline double interp_table(const double *restrict y_table,
                                  const double *restrict x_table,
                                  const double x, const double x_min,
                                  const double x_max) {

  /* Recover the range of interest in the log-a tabel */
  const double delta_x = ((double)cosmo_table_length) / (x_max - x_min);
  const double xx = (x - x_min) * delta_x;
  const int ii = (int)xx;
  return y_table[ii - 1] + (y_table[ii] - y_table[ii - 1]) * (xx - ii);
}

double get_drift_factor(const struct cosmology *cosmo, const double log_a_start,
                       const double log_a_end) {

  const double int0 = interp_table(cosmo->drift_factor_table,
                                   cosmo->log_a_table, log_a_start,
                                   cosmo->log_a_begin, cosmo->log_a_end);
  const double int1 = interp_table(cosmo->drift_factor_table,
                                   cosmo->log_a_table, log_a_end,
                                   cosmo->log_a_begin, cosmo->log_a_end);

  return int1 - int0;
}

double get_kick_factor(const struct cosmology *cosmo, const double log_a_start,
                       const double log_a_end) {

  const double int0 = interp_table(cosmo->kick_factor_table,
                                   cosmo->log_a_table, log_a_start,
                                   cosmo->log_a_begin, cosmo->log_a_end);
  const double int1 = interp_table(cosmo->kick_factor_table,
                                   cosmo->log_a_table, log_a_end,
                                   cosmo->log_a_begin, cosmo->log_a_end);

  return int1 - int0;
}

/* Fermi-Dirac distribution function */
double f0(double q) {
    double ksi = 0; //potential
    return 1.0/pow(2*M_PI,3)*(1./(exp(q-ksi)+1.) +1./(exp(q+ksi)+1.));
}

double ncdm_density_integrand(double p, void *params) {
    double *pars = (double *) params;
    double a = pars[0];
    double m = pars[1];
    double T = pars[2];

    double eps = hypot(p, a*m);
    double f = f0(p/T);
    double p2 = p * p;

    return p2 * eps * f;
}

double ncdm_pressure_integrand(double p, void *params) {
    double *pars = (double *) params;
    double a = pars[0];
    double m = pars[1];
    double T = pars[2];

    double eps = hypot(p, a*m);
    double f = f0(p/T);
    double p2 = p * p;

    return p2 * p2 / (3 * eps) * f;
}

double ncdm_background_density(double a, double m, double T) {
    /* Initalise the GSL workspace */
    gsl_integration_workspace *space =
      gsl_integration_workspace_alloc(workspace_size);

    double result, abserr;

    /* Compute the drift factor table */
    double pars[3] = {a, m, T};
    gsl_function F = {&ncdm_density_integrand, pars};
    gsl_integration_qag(&F, 0, T * 100, 0,
                        1.0e-10, workspace_size, GSL_INTEG_GAUSS61, space,
                        &result, &abserr);

    /* Free the GSL workspace */
    gsl_integration_workspace_free(space);

    return result;
}

double ncdm_background_pressure(double a, double m, double T) {
    /* Initalise the GSL workspace */
    gsl_integration_workspace *space =
      gsl_integration_workspace_alloc(workspace_size);

    double result, abserr;

    /* Compute the drift factor table */
    double pars[3] = {a, m, T};
    gsl_function F = {&ncdm_pressure_integrand, pars};
    gsl_integration_qag(&F, 0, T * 100, 0,
                        1.0e-10, workspace_size, GSL_INTEG_GAUSS61, space,
                        &result, &abserr);

    /* Free the GSL workspace */
    gsl_integration_workspace_free(space);

    return result;
}

double ncdm_isentropic_ratio(double a, double m, double T) {
    double T1 = T * (1 - 0.001);
    double T2 = T * (1 + 0.001);
    double rho1 = ncdm_background_density(a, m, T1);
    double rho2 = ncdm_background_density(a, m, T2);

    double rho = ncdm_background_density(a, m, T);

    double drho = rho2 - rho1;
    double dT = T2 - T1;

    return (drho / rho) * (T / dT);
}

double ncdm_equation_of_state(double a, double m, double T) {
    double rho = ncdm_background_density(a, m, T);
    double P = ncdm_background_pressure(a, m, T);

    return P / rho;
}
