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

#include <hdf5.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../include/defines.h"
#include "../include/fft.h"

/* Compute the 3D wavevector (kx,ky,kz) and its length k */
void fft_wavevector(int x, int y, int z, int N, double delta_k, double *kx,
                    double *ky, double *kz, double *k) {
    *kx = (x > N/2) ? (x - N)*delta_k : x*delta_k;
    *ky = (y > N/2) ? (y - N)*delta_k : y*delta_k;
    *kz = (z > N/2) ? (z - N)*delta_k : z*delta_k;
    *k = sqrt((*kx)*(*kx) + (*ky)*(*ky) + (*kz)*(*kz));
}

/* Normalize the complex array after transforming to momentum space */
int fft_normalize_r2c(fftw_complex *arr, int N, double boxlen) {
    const double boxvol = boxlen*boxlen*boxlen;
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<=N/2; z++) {
                arr[row_major_half(x, y, z, N)] *= boxvol/(N*N*N);
            }
        }
    }

    return 0;
}

/* Normalize the real array after transforming to configuration space */
int fft_normalize_c2r(double *arr, int N, double boxlen) {
    const double boxvol = boxlen*boxlen*boxlen;
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<N; z++) {
                arr[row_major(x, y, z, N)] /= boxvol;
            }
        }
    }

    return 0;
}

/* Execute an FFTW plan */
void fft_execute(fftw_plan plan) {
    fftw_execute(plan);
}

/* Apply a kernel to a 3D array after transforming to momentum space */
int fft_apply_kernel(fftw_complex *write, const fftw_complex *read, int N,
                     double boxlen, void (*compute)(struct kernel* the_kernel),
                     const void *params) {
    const double dk = 2 * M_PI / boxlen;

    #pragma omp parallel for
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<=N/2; z++) {
                /* Calculate the wavevector */
                double kx,ky,kz,k;
                fft_wavevector(x, y, z, N, dk, &kx, &ky, &kz, &k);

                /* Compute the kernel */
                struct kernel the_kernel = {kx, ky, kz, k, 0.f, params};
                compute(&the_kernel);

                /* Apply the kernel */
                const int id = row_major_half(x,y,z,N);
                write[id] = read[id] * the_kernel.kern;
            }
        }
    }

    return 0;
}
