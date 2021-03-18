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

#ifndef COSMOLOGY_H
#define COSMOLOGY_H

#include <hdf5.h>
#include "input.h"

struct cosmology {
    double h;
    double H_0;
    double rho_crit; //not user-specified, but inferred from h
    double Omega_m;
    double Omega_r;
    double Omega_k;
    double Omega_lambda;
    double T_nu;
    int N_nu;
    double *M_nu;
    double a_begin;
    double a_end;
    double log_a_begin;
    double log_a_end;

    double *drift_factor_table;
    double *kick_factor_table;
    double *log_a_table;

    struct units *units;
};

int readCosmology(struct cosmology *cosmo, struct units *us, hid_t h_file);
int cleanCosmology(struct cosmology *cosmo);

double E_z(double a, struct cosmology *cosmo);
int intregateCosmologyTables(struct cosmology *cosmo);
double get_drift_factor(const struct cosmology *cosmo, const double log_a_start,
                       const double log_a_end);
double get_kick_factor(const struct cosmology *cosmo, const double log_a_start,
                       const double log_a_end);

double ncdm_isentropic_ratio(double a, double m, double T);
double ncdm_equation_of_state(double a, double m, double T);

#endif
