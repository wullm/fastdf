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
#include <math.h>
#include <gsl/gsl_spline2d.h>
#include <gsl/gsl_spline.h>

#include "../include/perturb_spline.h"

/* Initialize the perturbation spline */
int initPerturbSpline(struct perturb_spline *spline, int k_acc_size,
                      const struct perturb_data *ptdat) {

    /* Store the address to the perturbation data */
    spline->ptdat = ptdat;

    /* Allocate the k search table */
    spline->k_acc_table = malloc(k_acc_size * sizeof(double));
    spline->k_acc_table_size = k_acc_size;

    if (spline->k_acc_table == NULL) return 1;

    /* Bounding values for the larger table */
    int k_size = ptdat->k_size;
    double k_min = ptdat->k[0];
    double k_max = ptdat->k[k_size-1];

    /* Make the index table */
    for (int i=0; i<k_acc_size; i++) {
        double u = (double) i/k_acc_size;
        double v = k_min + u * (k_max - k_min);

        /* Find the largest bin such that w > k */
        double maxJ = 0;
        for(int j=0; j<k_size; j++) {
            if (ptdat->k[j] < v) {
                maxJ = j;
            }
        }
        spline->k_acc_table[i] = maxJ;
    }

    return 0;
}

/* Clean the memory */
int cleanPerturbSpline(struct perturb_spline *spline) {
    free(spline->k_acc_table);

    return 0;
}

/* Find index along the time direction (0 <= u <= 1 between index and index+1) */
int perturbSplineFindTau(const struct perturb_spline *spline, double log_tau,
                         int *index, double *u) {

    /* Number of bins */
    int tau_size = spline->ptdat->tau_size;

    /* Quickly return if we are in the first or last bin */
    if (log_tau < spline->ptdat->log_tau[0]) {
        *index = 0;
        *u = 0.f;
        return 0;
    } else if (log_tau >= spline->ptdat->log_tau[tau_size - 1]) {
        *index = tau_size - 2;
        *u = 1.f;
        return 0;
    }

    /* Find i such that log_tau[i] <= log_tau */
    for (int i=1; i<tau_size; i++) {
        if (spline->ptdat->log_tau[i] >= log_tau) {
          *index = i-1;
          break;
        }
    }

    /* Find the bounding values */
    double left = spline->ptdat->log_tau[*index];
    double right = spline->ptdat->log_tau[*index + 1];

    /* Calculate the ratio (X - X_left) / (X_right - X_left) */
    *u = (log_tau - left) / (right - left);

    return 0;
}

/* Find index along the k direction (0 <= u <= 1 between index and index+1) */
int perturbSplineFindK(const struct perturb_spline *spline, double k, int *index,
                       double *u) {

    /* Bounding values for the larger table */
    int k_acc_table_size = spline->k_acc_table_size;
    int k_size = spline->ptdat->k_size;
    double k_min = spline->ptdat->k[0];
    double k_max = spline->ptdat->k[k_size-1];

    if (k > k_max) {
      *index = k_size - 2;
      *u = 1.0;
      return 0;
    }

    /* Quickly find a starting index using the indexed seach */
    double v = k;
    double w = (v - k_min) / (k_max - k_min);
    int J = floor(w * k_acc_table_size);
    int start = spline->k_acc_table[J < k_acc_table_size ? J : k_acc_table_size - 1];

    /* Search in the k vector */
    int i;
    for (i = start; i < k_size; i++) {
        if (k >= spline->ptdat->k[i] && k <= spline->ptdat->k[i + 1]) break;
    }

    /* We found the index */
    *index = i;

    /* Find the bounding values */
    double left = spline->ptdat->k[*index];
    double right = spline->ptdat->k[*index + 1];

    /* Calculate the ratio (X - X_left) / (X_right - X_left) */
    *u = (k - left) / (right - left);

    return 0;
}

/* Bilinear interpolation of the desired transfer function */
double perturbSplineInterp(const struct perturb_spline *spline, int k_index,
                           int tau_index, double u_k, double u_tau,
                           int index_src) {

    /* Bounding values for the larger table */
    int k_size = spline->ptdat->k_size;
    int tau_size = spline->ptdat->tau_size;

    /* Select the desired transfer function */
    double *arr = spline->ptdat->delta + index_src * k_size * tau_size;

    /* Retrieve the bounding values */
    double T11 = arr[k_size * tau_index + k_index];
    double T21 = arr[k_size * tau_index + k_index + 1];
    double T12 = arr[k_size * (tau_index + 1) + k_index];
    double T22 = arr[k_size * (tau_index + 1) + k_index + 1];

    return (1 - u_tau) * ((1 - u_k) * T11 + u_k * T21)
               + u_tau * ((1 - u_k) * T12 + u_k * T22);
}

/* Container function for simple bilinear interpolation */
double perturbSplineInterp0(const struct perturb_spline *spline, double k,
                            double log_tau, int index_src) {

    /* Indices in the k and tau directions */
    int k_index = 0, tau_index = 0;
    /* Spacing (0 <= u <= 1) between subsequent indices in both directions */
    double u_k, u_tau;

    /* Find the indices and spacings */
    perturbSplineFindTau(spline, log_tau, &tau_index, &u_tau);
    perturbSplineFindK(spline, k, &k_index, &u_k);

    /* Do the interpolation */
    return perturbSplineInterp(spline, k_index, tau_index, u_k, u_tau, index_src);
}

double perturbRedshiftAtLogTau(const struct perturb_spline *spline, double log_tau) {
    /* Indices in the tau directions */
    int tau_index = 0;
    /* Spacing (0 <= u <= 1) between subsequent indices */
    double u_tau;

    /* Find the index and spacing */
    perturbSplineFindTau(spline, log_tau, &tau_index, &u_tau);

    return (1 - u_tau) * spline->ptdat->redshift[tau_index]
               + u_tau * spline->ptdat->redshift[tau_index + 1];
}

double perturbLogTauAtRedshift(const struct perturb_spline *spline, double redshift) {

    /* Number of time/redshift bins */
    int tau_size = spline->ptdat->tau_size;

    /* Quickly return if we are in the first or last bin */
    if (redshift > spline->ptdat->redshift[0]) {
        return spline->ptdat->log_tau[0];
    } else if (redshift <= spline->ptdat->redshift[tau_size - 1]) {
        return spline->ptdat->log_tau[tau_size - 1];
    }

    /* Find i such that redshift[i] >= redshift */
    int index;
    for (index=tau_size - 1; index>0; index--) {
        if (spline->ptdat->redshift[index] >= redshift) break;
    }

    /* Find the bounding values */
    double left = spline->ptdat->redshift[index];
    double right = spline->ptdat->redshift[index + 1];

    /* Calculate the ratio (X - X_left) / (X_right - X_left) */
    double u = (redshift - left) / (right - left);

    return (1 - u) * spline->ptdat->log_tau[index]
              + u  * spline->ptdat->log_tau[index + 1];
}

/* Linear interpolation of the background density vector */
double perturbDensityAtLogTau(const struct perturb_spline *spline, double log_tau,
                              int index_src) {

    /* Number of tau indices */
    int tau_size = spline->ptdat->tau_size;

    /* Indices in the tau directions */
    int tau_index = 0;
    /* Spacing (0 <= u <= 1) between subsequent indices */
    double u_tau;

    /* Find the index and spacing */
    perturbSplineFindTau(spline, log_tau, &tau_index, &u_tau);

    /* Select the desired background density vector */
    double *arr = spline->ptdat->Omega + index_src * tau_size;

    return (1 - u_tau) * arr[tau_index]
               + u_tau * arr[tau_index + 1];

}
