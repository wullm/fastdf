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
