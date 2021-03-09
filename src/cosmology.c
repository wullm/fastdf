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
#include "../include/titles.h"

int cosmo_table_length = 1000;

const size_t workspace_size = 100000;

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
    // double a_inv = 1./a;
    // double a_inv2 = a_inv * a_inv;
    // double a_inv3 = a_inv2 * a_inv;
    // double a_inv4 = a_inv2 * a_inv2;
    //
    // double Omega_m = cosmo->Omega_m;
    // double Omega_r = cosmo->Omega_r;
    // double Omega_k = cosmo->Omega_k;
    // double Omega_lambda = cosmo->Omega_lambda;
    //
    // return sqrt(Omega_r * a_inv4 +
    //             Omega_m * a_inv3 +
    //             Omega_k * a_inv2 +
    //             Omega_lambda);

    double z = 1./a - 1;
    double log_tau = perturbLogTauAtRedshift(cosmo->spline, z);

    return perturbHubbleAtLogTau(cosmo->spline, log_tau) / cosmo->H_0;


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
    double log_a_begin = cosmo->log_a_begin;
    double log_a_end = cosmo->log_a_end;
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
                            1.0e-8, workspace_size, GSL_INTEG_GAUSS61, space,
                            &result, &abserr);

        cosmo->drift_factor_table[i] = result;
    }

    /* Compute the kick factor table */
    F.function = &kick_integrand;
    for (int i = 0; i < cosmo_table_length; i++) {
        gsl_integration_qag(&F, cosmo->a_begin, exp(cosmo->log_a_table[i]), 0,
                            1.0e-8, workspace_size, GSL_INTEG_GAUSS61, space,
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
