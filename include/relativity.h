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

#ifndef RELATIVITY_H
#define RELATIVITY_H

static inline double relativity_kick(double V, double a, struct units *us) {
    double c = us->SpeedOfLight;
    double v = V / (a * c);

    return (2 * v * v + 1.0) / hypot(v, 1.);
}

static inline double relativity_drift(double V, double a, struct units *us) {
    double c = us->SpeedOfLight;
    double v = V / (a * c);

    return 1.0 / hypot(v, 1.);
}

static inline double relativity_energy(double V, double a, struct units *us) {
    double c = us->SpeedOfLight;
    double v = V / (a * c);

    return 1.0 / hypot(v, 1.);
}

static inline double relativity_extra(double V, double a, struct units *us) {
    double c = us->SpeedOfLight;
    double v = V / (a * c);
    double v2 = v * v;

    return v2 / (v2 + 1);
}

#endif
