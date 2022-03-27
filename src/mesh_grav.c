/*******************************************************************************
 * This file is part of FastDF.
 * Copyright (c) 2021 Willem Elbers (whe@willemelbers.com)
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

#include <math.h>
#include <stdlib.h>
#include "../include/mesh_grav.h"
#include "../include/fft.h"

/* Direct nearest grid point interpolation */
static inline double fastNGP(const double *box, int N, int i, int j, int k) {

    return box[row_major(i, j, k, N)];
}

/* Direct cloud in cell interpolation */
static inline double fastCIC(const double *box, int N, int i, int j, int k,
                             double dx, double dy, double dz, double tx,
                             double ty, double tz) {

    return box[row_major(i, j, k, N)] * tx * ty * tz
         + box[row_major(i, j, k+1, N)] * tx * ty * dz
         + box[row_major(i, j+1, k, N)] * tx * dy * tz
         + box[row_major(i, j+1, k+1, N)] * tx * dy * dz
         + box[row_major(i+1, j, k, N)] * dx * ty * tz
         + box[row_major(i+1, j, k+1, N)] * dx * ty * dz
         + box[row_major(i+1, j+1, k, N)] * dx * dy * tz
         + box[row_major(i+1, j+1, k+1, N)] * dx * dy * dz;
}

/* Nearest grid point interpolation */
double gridNGP(const double *box, int N, double boxlen, double x, double y,
               double z) {

    /* Physical length to grid conversion factor */
    double fac = N / boxlen;

    /* Convert to float grid dimensions */
    double X = x * fac;
    double Y = y * fac;
    double Z = z * fac;

    /* Integer grid position (floor is necessary to handle negatives) */
    int iX = floor(X);
    int iY = floor(Y);
    int iZ = floor(Z);

    return fastNGP(box, N, iX, iY, iZ);
}

/* Cloud in cell interpolation */
double gridCIC(const double *box, int N, double boxlen, double x, double y,
               double z) {

    /* Physical length to grid conversion factor */
    double fac = N / boxlen;

    /* Convert to float grid dimensions */
    double X = x * fac;
    double Y = y * fac;
    double Z = z * fac;

    /* Integer grid position (floor is necessary to handle negatives) */
    int iX = floor(X);
    int iY = floor(Y);
    int iZ = floor(Z);

    /* Displacements from grid corner */
    double dx = X - iX;
    double dy = Y - iY;
    double dz = Z - iZ;

    double tx = 1.0 - dx;
    double ty = 1.0 - dy;
    double tz = 1.0 - dz;

    return fastCIC(box, N, iX, iY, iZ, dx, dy, dz, tx, ty, tz);
}

/* Generic grid interpolation method */
double gridInterp(const double *box, int N, double boxlen, double x, double y,
                  double z, int order) {

    if (order == 1) {
        return gridNGP(box, N, boxlen, x, y, z);
    } else if (order == 2) {
        return gridCIC(box, N, boxlen, x, y, z);
    } else {
        printf("Error: unsupported interpolation order.\n");
        exit(1);
    }
}

/* Compute the acceleration from the potential grid using NGP interpolation */
void accelNGP(const double *box, int N, double boxlen, double *x, double *a) {

    /* Physical length to grid conversion factor */
    double fac = N / boxlen;
    double fac_over_2 = fac / 2;

    /* Convert to float grid dimensions */
    double X = x[0] * fac;
    double Y = x[1] * fac;
    double Z = x[2] * fac;

    /* Integer grid position (floor is necessary to handle negatives) */
    int iX = floor(X);
    int iY = floor(Y);
    int iZ = floor(Z);

    a[0] = 0.0;
    a[0] += fastNGP(box, N, iX + 1, iY, iZ);
    a[0] -= fastNGP(box, N, iX - 1, iY, iZ);
    a[0] *= fac_over_2;

    a[1] = 0.0;
    a[1] += fastNGP(box, N, iX, iY + 1, iZ);
    a[1] -= fastNGP(box, N, iX, iY - 1, iZ);
    a[1] *= fac_over_2;

    a[2] = 0.0;
    a[2] += fastNGP(box, N, iX, iY, iZ + 1);
    a[2] -= fastNGP(box, N, iX, iY, iZ - 1);
    a[2] *= fac_over_2;
}

/* Compute the acceleration from the potential grid using CIC interpolation */
void accelCIC(const double *box, int N, double boxlen, double *x, double *a) {

    /* Physical length to grid conversion factor */
    double fac = N / boxlen;
    double fac_over_12 = fac / 12;

    /* Convert to float grid dimensions */
    double X = x[0] * fac;
    double Y = x[1] * fac;
    double Z = x[2] * fac;

    /* Integer grid position (floor is necessary to handle negatives) */
    int iX = floor(X);
    int iY = floor(Y);
    int iZ = floor(Z);

    /* Displacements from grid corner */
    double dx = X - iX;
    double dy = Y - iY;
    double dz = Z - iZ;
    double tx = 1.0 - dx;
    double ty = 1.0 - dy;
    double tz = 1.0 - dz;

    a[0] = 0.0;
    a[0] -= fastCIC(box, N, iX + 2, iY, iZ, dx, dy, dz, tx, ty, tz);
    a[0] += fastCIC(box, N, iX + 1, iY, iZ, dx, dy, dz, tx, ty, tz) * 8.;
    a[0] -= fastCIC(box, N, iX - 1, iY, iZ, dx, dy, dz, tx, ty, tz) * 8.;
    a[0] += fastCIC(box, N, iX - 2, iY, iZ, dx, dy, dz, tx, ty, tz);
    a[0] *= fac_over_12;

    a[1] = 0.0;
    a[1] -= fastCIC(box, N, iX, iY + 2, iZ, dx, dy, dz, tx, ty, tz);
    a[1] += fastCIC(box, N, iX, iY + 1, iZ, dx, dy, dz, tx, ty, tz) * 8.;
    a[1] -= fastCIC(box, N, iX, iY - 1, iZ, dx, dy, dz, tx, ty, tz) * 8.;
    a[1] += fastCIC(box, N, iX, iY - 2, iZ, dx, dy, dz, tx, ty, tz);
    a[1] *= fac_over_12;

    a[2] = 0.0;
    a[2] -= fastCIC(box, N, iX, iY, iZ + 2, dx, dy, dz, tx, ty, tz);
    a[2] += fastCIC(box, N, iX, iY, iZ + 1, dx, dy, dz, tx, ty, tz) * 8.;
    a[2] -= fastCIC(box, N, iX, iY, iZ - 1, dx, dy, dz, tx, ty, tz) * 8.;
    a[2] += fastCIC(box, N, iX, iY, iZ - 2, dx, dy, dz, tx, ty, tz);
    a[2] *= fac_over_12;
}

/* Generic grid interpolation method for the acceleration vector */
void accelInterp(const double *box, int N, double boxlen, double *x, double *a,
                 int order) {

    if (order == 1) {
        return accelNGP(box, N, boxlen, x, a);
    } else if (order == 2) {
        return accelCIC(box, N, boxlen, x, a);
    } else {
        printf("Error: unsupported interpolation order.\n");
        exit(1);
    }
}