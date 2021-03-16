/**
 *  Copyright (c) 2021 Willem Elbers (whe@willemelbers.com)
 *
 *  @file phase_space.h
 *  @brief Computes weights for neutrino particles according to
 *  the delta-f method (arXiv:2010.07321).
 */

/* Standard headers */
#include <math.h>
#include <stdint.h>

/* Lookup table for Fermi-Dirac transform */
#include "fermi_dirac.h"
#include "input.h"

/* Internal unit system */
struct internal_units {
    const double length_unit_metres;
    const double time_unit_seconds;
    const double mass_unit_kg;
    const double temperature_unit_kelvin;
};

/* Physical constants in internal units */
struct phys_const {
    const double speed_of_light;
    const double boltzmann_constant;
    const double reduced_planck_constant;
    const double electronvolt;
    const double neutrino_temperature;
    const double neutrino_temperature_eV;
};

/* Riemann function zeta(3) */
#define M_ZETA_3 1.2020569031595942853997

static inline double hypot3(double x, double y, double z) {
    return hypot(x, hypot(y, z));
}

/* Pseudo-random number generator */
static inline uint64_t splitmix64(uint64_t *state) {
    uint64_t result = *state;

    *state = result + 0x9E3779B97f4A7C15;
    result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
    result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
    return result ^ (result >> 31);
}

/* Generate a uniform variable on the open unit interval */
static inline double sampleUniform(uint64_t *state) {
    const uint64_t A = splitmix64(state);
    const double RM = (double)UINT64_MAX + 1;
    return ((double)A + 0.5) / RM;
}

/* Generate standard normal variable with the Box-Mueller transform */
static inline double sampleGaussian(uint64_t *state) {
    /* Generate random integers */
    const uint64_t A = splitmix64(state);
    const uint64_t B = splitmix64(state);
    const double RMax = (double)UINT64_MAX + 1;

    /* Map the random integers to the open (!) unit interval */
    const double u = ((double)A + 0.5) / RMax;
    const double v = ((double)B + 0.5) / RMax;

    /* Map to two Gaussians (the second is not used - inefficient) */
    const double z0 = sqrt(-2 * log(u)) * cos(2 * M_PI * v);
    // double z1 = sqrt(-2 * log(u)) * sin(2 * M_PI * v);

    return z0;
}

/**
 * @brief Calculate the present-day momentum in electronvolts. We can use
 * energy units because E = a*sqrt(p^2 + m^2) = a*p, since p/m >> 1 at
 * decoupling. Note that this quantity is constant in a homogeneous Universe.
 *
 * @param v Array of 3 velocity components (a^2 dx/dt, where x is comoving)
 * @param m_eV Neutrino mass in electronvolts
 * @param c Speed of light
 */
static inline double fermi_dirac_momentum(double *v, double m_eV, double c) {
    const double u = hypot3(v[0], v[1], v[2]);
    const double p_eV = u;

    return p_eV;
}

/**
 * @brief Calculate the neutrino density at the particle's location in phase
 * space, according to the 0th order background model: f_0(x,p,t).
 *
 * @param p_eV Present-day momentum in electronvolts
 * @param T_eV Present-day temperature in electronvolts
 */
static inline double fermi_dirac_density(double p_eV, double T_eV) {
    return 1.0 / (exp(p_eV / T_eV) + 1.0);
}

/**
 * @brief Initialize a neutrino particle with a random Fermi-Dirac momentum
 * and uniform random position.
 *
 * @param seed A persistent unique seed for each neutrino particle
 * @param m_eV Neutrino mass in electronvolts
 * @param v Array of 3 velocity components (a^2 dx/dt, where x is comoving)
 * @param x Array of 3 position components (comoving position)
 * @param w Reference to the weight of the particle
 * @param boxlen Comoving physical sidelength of the box
 * @param us Container of units physical constants
 */
static void init_neutrino_particle(uint64_t seed, double m_eV, double *v,
                                   double *x, double *w, double boxlen,
                                   const struct units *us, double T_eV) {

    /* Retrieve physical constants */
    const double c = us->SpeedOfLight;

    /* A unique uniform random number for this neutrino */
    const double z = sampleUniform(&seed);
    /* The corresponding initial Fermi-Dirac momentum */
    const double p_eV = fermi_dirac_transform(z) * T_eV;
    /* The corresponding velocity */
    const double u = p_eV;

    /* Generate a random point uniformly on the sphere */
    double nx = sampleGaussian(&seed);
    double ny = sampleGaussian(&seed);
    double nz = sampleGaussian(&seed);

    /* Normalize */
    double n = hypot3(nx, ny, nz);
    if (n > 0) {
        nx /= n;
        ny /= n;
        nz /= n;
    }

    /* Set the velocity in that direction */
    v[0] = u * nx;
    v[1] = u * ny;
    v[2] = u * nz;

    /* Generate a random position */
    x[0] = boxlen * sampleUniform(&seed);
    x[1] = boxlen * sampleUniform(&seed);
    x[2] = boxlen * sampleUniform(&seed);

    /* Initially, the weight is zero (dithering may be necessary here) */
    *w = 0;
}
