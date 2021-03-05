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

#ifndef PERTURB_SPLINE_H
#define PERTURB_SPLINE_H

#define DEFAULT_K_ACC_TABLE_SIZE 1000

#include "perturb_data.h"

struct perturb_spline {
    /* Address of the perturbation data */
    const struct perturb_data *ptdat;

    /* Search table for interpolation acceleration in the k direction */
    double *k_acc_table;

    /* Size of the k acceleration table */
    int k_acc_table_size;
};

/* Initialize the perturbation spline */
int initPerturbSpline(struct perturb_spline *spline, int k_acc_size,
                      const struct perturb_data *ptdat);


/* Clean the memory */
int cleanPerturbSpline(struct perturb_spline *spline);

/* Find index along the time direction (0 <= u <= 1 between index and index+1) */
int perturbSplineFindTau(const struct perturb_spline *spline, double log_tau,
                         int *index, double *u);

/* Find index along the k direction (0 <= u <= 1 between index and index+1) */
int perturbSplineFindK(const struct perturb_spline *spline, double k, int *index,
                       double *u);

/* Bilinear interpolation of the desired transfer function */
double perturbSplineInterp(const struct perturb_spline *spline, int k_index,
                           int tau_index, double u_k, double u_tau,
                           int index_src);

/* Container function for simple bilinear interpolation */
double perturbSplineInterp0(const struct perturb_spline *spline, double k,
                            double log_tau, int index_src);

/* Linear interpolation of the redshift vector */
double perturbRedshiftAtLogTau(const struct perturb_spline *spline, double log_tau);

/* Linear interpolation of the log_tau vector */
double perturbLogTauAtRedshift(const struct perturb_spline *spline, double redshift);

/* Linear interpolation of the background density vector */
double perturbDensityAtLogTau(const struct perturb_spline *spline, double log_tau,
                              int index_src);


#endif
