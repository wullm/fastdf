/*******************************************************************************
 * This file is part of FastDF.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hdf5.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>

#include "../include/fastdf.h"

int main(int argc, char *argv[]) {
    if (argc == 1) {
        printf("No parameter file specified.\n");
        return 0;
    }

    /* Read options */
    const char *fname = argv[1];
    printf("The parameter file is %s\n", fname);

    /* Timer */
    struct timeval time_stop, time_start;
    gettimeofday(&time_start, NULL);

    struct params pars;
    struct units us;
    struct cosmology cosmo = {0};
    struct perturb_data ptdat;
    struct perturb_spline spline;
    struct perturb_params ptpars;

    /* No MPI for now */
    int rank = 0;
    pars.rank = rank;

    /* Read parameter file for parameters, units */
    readParams(&pars, fname);
    readUnits(&us, fname);

    /* Read the perturbation data file */
    readPerturb(&pars, &us, &ptdat);
    readPerturbParams(&pars, &us, &ptpars);

    /* Store cosmological parameters */
    cosmo.a_begin = pars.ScaleFactorBegin;
    cosmo.a_end = pars.ScaleFactorEnd;
    cosmo.log_a_begin = log(cosmo.a_begin);
    cosmo.log_a_end = log(cosmo.a_end);

    cosmo.h = ptpars.h;
    cosmo.H_0 = cosmo.h * 100 * KM_METRES / MPC_METRES * us.UnitTimeSeconds;
    cosmo.Omega_m = ptpars.Omega_m;
    cosmo.Omega_k = ptpars.Omega_k;
    cosmo.Omega_lambda = ptpars.Omega_lambda;
    cosmo.Omega_r = ptpars.Omega_ur;

    /* Compute cosmological tables (kick and drift factors) */
    intregateCosmologyTables(&cosmo);

    /* Package physical constants */
    const double m_eV = ptpars.M_ncdm_eV[0];
    const double T_nu = ptpars.T_ncdm[0] * ptpars.T_CMB;
    const double T_eV = T_nu * us.kBoltzmann / us.ElectronVolt;
    struct phys_const phys_const = { us.SpeedOfLight, us.kBoltzmann, us.hPlanck / (2*M_PI), us.ElectronVolt, T_nu, T_eV};

    /* Initialize the interpolation spline for the perturbation data */
    initPerturbSpline(&spline, DEFAULT_K_ACC_TABLE_SIZE, &ptdat);

    header(rank, "Simulation parameters");
    printf("We want %lld (%d^3) particles\n", pars.NumPartGenerate, pars.CubeRootNumber);
    printf("a_begin = %.3e (z = %.2f)\n", cosmo.a_begin, 1./cosmo.a_begin - 1);
    printf("a_end = %.3e (z = %.2f)\n", cosmo.a_end, 1./cosmo.a_end - 1);

    /* Read the Gaussian random field */
    double *box;
    double BoxLen;
    int N;

    header(rank, "Random phases");
    message(rank, "Reading Gaussian random field from %s.\n", pars.GaussianRandomFieldFile);

    readFieldFile(&box, &N, &BoxLen, pars.GaussianRandomFieldFile);

    printf("BoxLen = %f\n", BoxLen);
    printf("GridSize = %d\n", N);

    /* Fourier transform the Gaussian random field */
    fftw_complex *fbox = malloc(N*N*N*sizeof(fftw_complex));
    fftw_complex *fgrf = malloc(N*N*N*sizeof(fftw_complex));
    fftw_plan r2c = fftw_plan_dft_r2c_3d(N, N, N, box, fbox, FFTW_ESTIMATE);
    fft_execute(r2c);
    fft_normalize_r2c(fbox, N, BoxLen);
    fftw_destroy_plan(r2c);

    /* Make a copy of the complex Gaussian random field */
    memcpy(fgrf, fbox, N*N*N*sizeof(fftw_complex));

    /* Compute neutrino mass factor */
    const double mass_factor = neutrino_mass_factor(pars.NumPartGenerate, BoxLen, &phys_const);


    /* Compute the particle mass in internal units */
    const double particle_mass = m_eV / mass_factor;

    header(rank, "Mass factors");
    printf("Neutrino mass factor = %e\n", mass_factor);
    printf("Neutrino mass is %f eV\n", m_eV);
    printf("Particle mass is %f\n", particle_mass);

    /* Store the Box Length */
    pars.BoxLen = BoxLen;

    /* The particles to be generated */
    struct particle_ext *genparts = malloc(sizeof(struct particle_ext) *
                                            pars.NumPartGenerate);

    header(rank, "Generating pre-initial conditions");
    printf("T_eV = %e\n", T_eV);
    printf("ID of first particle = %lld\n", pars.FirstID);

    /* Generate random neutrino particles */
    for (int i=0; i<pars.NumPartGenerate; i++) {
        struct particle_ext *p = &genparts[i];

        /* Set the ID of the particle */
        long long id = i + pars.FirstID;

        /* Generate random particle velocity and position */
        init_neutrino_particle(id, m_eV, p->v, p->x, &p->mass, BoxLen, &us, T_eV);

        /* Compute the momentum in eV */
        const double p_eV = fermi_dirac_momentum(p->v, m_eV, us.SpeedOfLight);
        const double f_i = fermi_dirac_density(p_eV, T_eV);

        /* Compute initial phase space density */
        p->f_i = f_i;
    }

    printf("Done with pre-initial conditions.\n");

    header(rank, "Initiating geodesic integration.");

    /* Prepare integration */
    double a_begin = cosmo.a_begin;
    double a_end = cosmo.a_end;
    double a_factor = 1.0 + pars.ScaleFactorStep;

    int MAX_ITER = (log(a_end) - log(a_begin))/log(a_factor);

    printf("Step size %.4f\n", a_factor-1);
    printf("Doing %d iterations\n", MAX_ITER);
    printf("\n");

    /* Start at the beginning */
    double a = a_begin;

    message(rank, "ITER]\t a\t z\t I\n");

    /* The main loop */
    for (int ITER = 0; ITER < MAX_ITER; ITER++) {
        /* Compute the current redshift and log conformal time */
        double z = 1./a - 1;
        double log_tau = perturbLogTauAtRedshift(&spline, z);

        /* Determine the next scale factor */
        double a_next;
        if (ITER < MAX_ITER - 1) {
            a_next = a * a_factor;
        } else {
            a_next = a_end;
        }

        /* Collect the I statistic */
        double I_df = 0;

        /* Retrieve physical constants */
        const double potential_factor = a / us.GravityG;
        const double c = us.SpeedOfLight;

        /* Find the interpolation index along the time dimension */
        int tau_index; //greatest lower bound bin index
        double u_tau; //spacing between subsequent bins
        perturbSplineFindTau(&spline, log_tau, &tau_index, &u_tau);

        /* The indices of the potential transfer function */
        int index_phi = findTitle(ptdat.titles, "phi", ptdat.n_functions);

        /* Package the perturbation theory interpolation spline parameters */
        struct spline_params sp = {&spline, index_phi, tau_index, u_tau};

        /* Apply the transfer function (read only fgrf, output into fbox) */
        fft_apply_kernel(fbox, fgrf, N, BoxLen, kernel_transfer_function, &sp);

        /* Fourier transform to real space */
        fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fbox, box, FFTW_ESTIMATE);
        fft_execute(c2r);
        fft_normalize_c2r(box, N, BoxLen);
        fftw_destroy_plan(c2r);

        /* Multiply by the potential factor */
        for (int i=0; i<N*N*N; i++) {
            box[i] *= potential_factor;
        }

        char pot_fname[50];
        sprintf(pot_fname, "phi_%d.hdf5", ITER);
        writeFieldFile(box, N, BoxLen, pot_fname);

        /* Fetch the cosmological kick and drift factors */
        double kick_factor = get_kick_factor(&cosmo, log(a), log(a_next));
        double drift_factor = get_drift_factor(&cosmo, log(a), log(a_next));

        /* Integrate the particles */
        #pragma omp parallel for reduction(+:I_df)
        for (int i=0; i<pars.NumPartGenerate; i++) {
            struct particle_ext *p = &genparts[i];

            /* Get the accelerations by computing the gradient of phi */
            double acc[3];
            accelCIC(box, N, BoxLen, p->x, acc);

            /* Fetch the relativistic correction factors */
            double relat_kick_correction = relativity_kick(p->v, a, &us);
            double relat_drift_correction = relativity_drift(p->v, a, &us);

            /* Compute the overall kick and drift step sizes */
            double kick = kick_factor * relat_kick_correction * us.GravityG;
            double drift = drift_factor * relat_drift_correction;

            /* Execute kick */
            p->v[0] += acc[0] * kick;
            p->v[1] += acc[1] * kick;
            p->v[2] += acc[2] * kick;

            /* Execute delta-f step */
            double p_eV = fermi_dirac_momentum(p->v, m_eV, c);
            double f = fermi_dirac_density(p_eV, T_eV);
            double w = (f - p->f_i)/p->f_i;
            I_df += w*w;

            /* Execute drift */
            p->x[0] += p->v[0] * drift;
            p->x[1] += p->v[1] * drift;
            p->x[2] += p->v[2] * drift;
        }

        /* Step forward */
        a = a_next;

        /* Compute summary statistic */
        I_df *= 0.5 / pars.NumPartGenerate;

        message(rank, "%04d] %f %f %e\n", ITER, a, 1./a-1, I_df);
    }

    /* Free memory */
    free(box);
    free(fgrf);
    free(fbox);

    /* Final operations before writing the particles to disk */
    for (int i=0; i<pars.NumPartGenerate; i++) {
        struct particle_ext *p = &genparts[i];

        /* Ensure that particles wrap */
        p->x[0] = fwrap(p->x[0], BoxLen);
        p->x[1] = fwrap(p->x[1], BoxLen);
        p->x[2] = fwrap(p->x[2], BoxLen);

        /* Update the mass (needs to happen before converting the velocities!)*/
        double p_eV = fermi_dirac_momentum(p->v, m_eV, us.SpeedOfLight);
        double f = fermi_dirac_density(p_eV, T_eV);
        double w = (f - p->f_i)/p->f_i;
        p->mass = particle_mass * w;

        /* Convert to peculiar velocities */
        p->v[0] /= a_end;
        p->v[1] /= a_end;
        p->v[2] /= a_end;
    }


    header(rank, "Prepare output");

    char *out_fname = pars.OutputFilename;
    message(rank, "Writing output to %s.\n", out_fname);

    /* Create the output file */
    hid_t h_out_file = createFile(out_fname);

    /* Writing attributes into the Header & Cosmology groups */
    int err = writeHeaderAttributes(&pars, &cosmo, &us, pars.NumPartGenerate, h_out_file);
    if (err > 0) exit(1);

    /* The ExportName */
    const char *ExportName = pars.ExportName;

    /* The particle group in the output file */
    hid_t h_grp;

    /* Datsets */
    hid_t h_data;

    /* Vector dataspace (e.g. positions, velocities) */
    const hsize_t vrank = 2;
    const hsize_t vdims[2] = {pars.NumPartGenerate, 3};
    hid_t h_vspace = H5Screate_simple(vrank, vdims, NULL);

    /* Scalar dataspace (e.g. masses, particle ids) */
    const hsize_t srank = 1;
    const hsize_t sdims[1] = {pars.NumPartGenerate};
    hid_t h_sspace = H5Screate_simple(srank, sdims, NULL);

    /* Create the particle group in the output file */
    printf("Creating Group '%s' with %lld particles.\n", ExportName, pars.NumPartGenerate);
    h_grp = H5Gcreate(h_out_file, ExportName, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Coordinates (use vector space) */
    h_data = H5Dcreate(h_grp, "Coordinates", H5T_NATIVE_DOUBLE, h_vspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dclose(h_data);

    /* Velocities (use vector space) */
    h_data = H5Dcreate(h_grp, "Velocities", H5T_NATIVE_DOUBLE, h_vspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dclose(h_data);

    /* Masses (use scalar space) */
    h_data = H5Dcreate(h_grp, "Masses", H5T_NATIVE_DOUBLE, h_sspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dclose(h_data);

    /* Particle IDs (use scalar space) */
    h_data = H5Dcreate(h_grp, "ParticleIDs", H5T_NATIVE_LLONG, h_sspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dclose(h_data);

    /* Create vector & scalar datapsace for smaller chunks of data */
    const hsize_t ch_vdims[2] = {pars.NumPartGenerate, 3};
    const hsize_t ch_sdims[2] = {pars.NumPartGenerate};
    hid_t h_ch_vspace = H5Screate_simple(vrank, ch_vdims, NULL);
    hid_t h_ch_sspace = H5Screate_simple(srank, ch_sdims, NULL);

    /* The start of this chunk, in the overall vector & scalar spaces */
    const hsize_t start_in_group = 0;
    const hsize_t vstart[2] = {start_in_group, 0}; //always with the "x" coordinate
    const hsize_t sstart[1] = {start_in_group};

    /* Choose the corresponding hyperslabs inside the overall spaces */
    H5Sselect_hyperslab(h_vspace, H5S_SELECT_SET, vstart, NULL, ch_vdims, NULL);
    H5Sselect_hyperslab(h_sspace, H5S_SELECT_SET, sstart, NULL, ch_sdims, NULL);

    /* Unpack particle data into contiguous arrays */
    double *coords = malloc(3 * pars.NumPartGenerate * sizeof(double));
    double *vels = malloc(3 * pars.NumPartGenerate * sizeof(double));
    double *masses = malloc(1 * pars.NumPartGenerate * sizeof(double));
    long long *ids = malloc(1 * pars.NumPartGenerate * sizeof(long long));
    for (int i=0; i<pars.NumPartGenerate; i++) {
        coords[i * 3 + 0] = genparts[i].x[0];
        coords[i * 3 + 1] = genparts[i].x[1];
        coords[i * 3 + 2] = genparts[i].x[2];
        vels[i * 3 + 0] = genparts[i].v[0];
        vels[i * 3 + 1] = genparts[i].v[1];
        vels[i * 3 + 2] = genparts[i].v[2];
        masses[i] = genparts[i].mass;
        ids[i] = pars.FirstID + i;
    }

    /* Write coordinate data (vector) */
    h_data = H5Dopen(h_grp, "Coordinates", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_ch_vspace, h_vspace, H5P_DEFAULT, coords);
    H5Dclose(h_data);
    free(coords);

    /* Write velocity data (vector) */
    h_data = H5Dopen(h_grp, "Velocities", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_ch_vspace, h_vspace, H5P_DEFAULT, vels);
    H5Dclose(h_data);
    free(vels);

    /* Write mass data (scalar) */
    h_data = H5Dopen(h_grp, "Masses", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_ch_sspace, h_sspace, H5P_DEFAULT, masses);
    H5Dclose(h_data);
    free(masses);

    /* Write particle id data (scalar) */
    h_data = H5Dopen(h_grp, "ParticleIDs", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_LLONG, h_ch_sspace, h_sspace, H5P_DEFAULT, ids);
    H5Dclose(h_data);
    free(ids);

    /* Close the chunk-sized scalar and vector dataspaces */
    H5Sclose(h_ch_vspace);
    H5Sclose(h_ch_sspace);

    /* Close the group */
    H5Gclose(h_grp);

    /* Close the file */
    H5Fclose(h_out_file);

    /* Free the particles array */
    free(genparts);

    /* Clean up */
    cleanParams(&pars);
    cleanCosmology(&cosmo);
    cleanPerturb(&ptdat);
    cleanPerturbParams(&ptpars);

    /* Release the interpolation splines */
    cleanPerturbSpline(&spline);

    /* Timer */
    gettimeofday(&time_stop, NULL);
    long unsigned microsec = (time_stop.tv_sec - time_start.tv_sec) * 1000000
                           + time_stop.tv_usec - time_start.tv_usec;
    message(rank, "\nTime elapsed: %.5f s\n", microsec/1e6);
}
