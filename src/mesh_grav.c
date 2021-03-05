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
#include "../include/mesh_grav.h"
#include "../include/fft.h"

/* Cloud in cell interpolation */
double gridCIC(const double *box, int N, double boxlen, double x, double y,
               double z) {
    /* Convert to float grid dimensions */
    double X = x*N/boxlen;
    double Y = y*N/boxlen;
    double Z = z*N/boxlen;

    /* Integer grid position */
    int iX = (int) floor(X);
    int iY = (int) floor(Y);
    int iZ = (int) floor(Z);


    /* Intepolate the necessary fields with CIC or TSC */
    double lookLength = 1.0;
    int lookLftX = (int) floor((X-iX) - lookLength);
    int lookRgtX = (int) floor((X-iX) + lookLength);
    int lookLftY = (int) floor((Y-iY) - lookLength);
    int lookRgtY = (int) floor((Y-iY) + lookLength);
    int lookLftZ = (int) floor((Z-iZ) - lookLength);
    int lookRgtZ = (int) floor((Z-iZ) + lookLength);

    /* Accumulate */
    double sum = 0;
    for (int i=lookLftX; i<=lookRgtX; i++) {
        for (int j=lookLftY; j<=lookRgtY; j++) {
            for (int k=lookLftZ; k<=lookRgtZ; k++) {
                double xx = fabs(X - (iX+i));
                double yy = fabs(Y - (iY+j));
                double zz = fabs(Z - (iZ+k));

                double part_x = xx <= 1 ? 1-xx : 0;
                double part_y = yy <= 1 ? 1-yy : 0;
                double part_z = zz <= 1 ? 1-zz : 0;

                sum += box[row_major(iX+i, iY+j, iZ+k, N)] * (part_x*part_y*part_z);
            }
        }
    }

    return sum;
}


/* Compute the acceleration from the potential grid using CIC interpolation */
void accelCIC(const double *box, int N, double boxlen, double *x, double *a) {

    double fac = boxlen / N;

    a[0] = 0.0;
    a[0] -= gridCIC(box, N, boxlen, x[0] + 2 * fac, x[1], x[2]);
    a[0] += gridCIC(box, N, boxlen, x[0] + 1 * fac, x[1], x[2]) * 8;
    a[0] -= gridCIC(box, N, boxlen, x[0] - 1 * fac, x[1], x[2]) * 8;
    a[0] += gridCIC(box, N, boxlen, x[0] - 2 * fac, x[1], x[2]);
    a[0] /= 12 * fac;

    a[1] = 0.0;
    a[1] -= gridCIC(box, N, boxlen, x[0], x[1] + 2 * fac, x[2]);
    a[1] += gridCIC(box, N, boxlen, x[0], x[1] + 1 * fac, x[2]) * 8;
    a[1] -= gridCIC(box, N, boxlen, x[0], x[1] - 1 * fac, x[2]) * 8;
    a[1] += gridCIC(box, N, boxlen, x[0], x[1] - 2 * fac, x[2]);
    a[1] /= 12 * fac;

    a[2] = 0.0;
    a[2] -= gridCIC(box, N, boxlen, x[0], x[1], x[2] + 2 * fac);
    a[2] += gridCIC(box, N, boxlen, x[0], x[1], x[2] + 1 * fac) * 8;
    a[2] -= gridCIC(box, N, boxlen, x[0], x[1], x[2] - 1 * fac) * 8;
    a[2] += gridCIC(box, N, boxlen, x[0], x[1], x[2] - 2 * fac);
    a[2] /= 12 * fac;
}
