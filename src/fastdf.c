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

/* Fermi-Dirac distribution function */
double f0(double q) {
    double ksi = 0; //potential
    return 1.0/pow(2*M_PI,3)*(1./(exp(q-ksi)+1.) +1./(exp(q+ksi)+1.));
}

/* Logarithmic derivative of distribution function */
double compute_dlnf0_dlnq(double q, double h) {
    double df0_dq = 0, dlnf0_dlnq;

    df0_dq += (1./12.) * f0(q - 2*h);
    df0_dq -= (8./12.) * f0(q - 1*h);
    df0_dq += (8./12.) * f0(q + 1*h);
    df0_dq -= (1./12.) * f0(q + 2*h);
    df0_dq /= h;

    double f0_eval = f0(q);
    if (fabs(f0_eval) > 0) {
        dlnf0_dlnq = q/f0_eval * df0_dq;
    } else {
        dlnf0_dlnq = -q;
    }

    return dlnf0_dlnq;
}

int main(int argc, char *argv[]) {
    if (argc == 1) {
        printf("No parameter file specified.\n");
        return 0;
    }

    /* Initialize MPI for distributed memory parallelization */
    MPI_Init(&argc, &argv);
    fftw_mpi_init();

    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);

    /* Read options */
    const char *fname = argv[1];
    header(rank, "FastDF Neutrino Initial Condition Generator");
    message(rank, "The parameter file is %s\n", fname);

    /* Timer */
    struct timeval time_stop, time_start;
    gettimeofday(&time_start, NULL);

    struct params pars;
    struct units us;
    struct cosmology cosmo = {0};
    struct perturb_data ptdat;
    struct perturb_spline spline;
    struct perturb_params ptpars;

    /* Store the MPI rank */
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
    // cosmo.Omega_r = ptpars.Omega_ur;
    cosmo.Omega_r = 1 - cosmo.Omega_m - cosmo.Omega_k - cosmo.Omega_lambda;
    cosmo.rho_crit = 3 * cosmo.H_0 * cosmo.H_0 / (8. * M_PI * us.GravityG);

    /* Compute cosmological tables (kick and drift factors) */
    intregateCosmologyTables(&cosmo);

    /* Package physical constants */
    const double m_eV = ptpars.M_ncdm_eV[0];
    const double T_nu = ptpars.T_ncdm[0] * ptpars.T_CMB;
    const double T_eV = T_nu * us.kBoltzmann / us.ElectronVolt;

    /* Initialize the interpolation spline for the perturbation data */
    initPerturbSpline(&spline, DEFAULT_K_ACC_TABLE_SIZE, &ptdat);

    header(rank, "Simulation parameters");
    message(rank, "We want %lld (%d^3) particles\n", pars.NumPartGenerate, pars.CubeRootNumber);
    message(rank, "a_begin = %.3e (z = %.2f)\n", cosmo.a_begin, 1./cosmo.a_begin - 1);
    message(rank, "a_end = %.3e (z = %.2f)\n", cosmo.a_end, 1./cosmo.a_end - 1);

    /* Read the Gaussian random field on each MPI rank */
    double *box;
    double BoxLen;
    int N;

    header(rank, "Random phases");
    message(rank, "Reading Gaussian random field from %s.\n", pars.GaussianRandomFieldFile);

    readFieldFile_MPI(&box, &N, &BoxLen, MPI_COMM_WORLD, pars.GaussianRandomFieldFile);

    message(rank, "BoxLen = %.2f U_L\n", BoxLen);
    message(rank, "GridSize = %d\n", N);

    /* Fourier transform the Gaussian random field */
    fftw_complex *fbox = malloc(N*N*N*sizeof(fftw_complex));
    fftw_complex *fgrf = malloc(N*N*N*sizeof(fftw_complex));
    fftw_plan r2c = fftw_plan_dft_r2c_3d(N, N, N, box, fbox, FFTW_ESTIMATE);
    fft_execute(r2c);
    fft_normalize_r2c(fbox, N, BoxLen);
    fftw_destroy_plan(r2c);

    /* Make a copy of the complex Gaussian random field */
    memcpy(fgrf, fbox, N*N*N*sizeof(fftw_complex));

    /* Find the relevant density title among the transfer functions */
    char *title = pars.TransferFunctionDensity;
    int index_src = findTitle(ptdat.titles, title, ptdat.n_functions);
    if (index_src < 0) {
        message(rank, "Error: transfer function '%s' not found (%d).\n", title, index_src);
        return 1;
    }

    /* The index of the present-day, corresponds to the last index in the array */
    int today_index = ptdat.tau_size - 1;

    /* Find the present-day density, as fraction of the critical density */
    const double box_vol = BoxLen * BoxLen * BoxLen;
    const double Omega = ptdat.Omega[ptdat.tau_size * index_src + today_index];
    const double rho = Omega * cosmo.rho_crit;
    const double particle_mass = rho * box_vol / pars.NumPartGenerate;

    header(rank, "Mass factors");
    message(rank, "Neutrino mass is %f eV\n", m_eV);
    message(rank, "Particle mass is %f U_M\n", particle_mass);

    /* Store the Box Length */
    pars.BoxLen = BoxLen;

    /* Determine the number of particle to be generated on each rank */
    header(rank, "Particle distribution");

    /* The particles are also generated from a grid with dimension M^3 */
    int M = pars.CubeRootNumber;

    /* Determine what particles belong to this slice */
    double fac = (double) M / MPI_Rank_Count;
    int X_min = ceil(rank * fac);
    int X_max = ceil((rank + 1) * fac);
    int MX = X_max - X_min;
    long long localParticleNumber = MX * M * M;
    long long localFirstNumber = X_min * M * M;

    /* Check that all particles have been assigned to a node */
    long long totalParticlesAssigned;
    MPI_Allreduce(&localParticleNumber, &totalParticlesAssigned, 1,
                   MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    assert(totalParticlesAssigned == M * M * M);
    printf("%03d: Local particles [%04d, %04d], first = %lld, last = %lld, total = %lld\n", rank, X_min, X_max, localFirstNumber, localFirstNumber + localParticleNumber - 1, totalParticlesAssigned);

    /* The particles to be generated on this node */
    struct particle_ext *genparts = malloc(sizeof(struct particle_ext) *
                                    localParticleNumber);

    /* ID of the first particle on this node */
    long long firstID = pars.FirstID + localFirstNumber;

    MPI_Barrier(MPI_COMM_WORLD);

    double z_begin = 1./cosmo.a_begin - 1;
    double log_tau_begin = perturbLogTauAtRedshift(&spline, z_begin);
    double tau_begin = exp(log_tau_begin);

    header(rank, "Generating pre-initial grids");
    {


        /* Find the interpolation index along the time dimension */
        int tau_index; //greatest lower bound bin index
        double u_tau; //spacing between subsequent bins
        perturbSplineFindTau(&spline, log_tau_begin, &tau_index, &u_tau);

        /* The indices of the potential transfer function */
        int index_psi = findTitle(ptdat.titles, "psi", ptdat.n_functions);

        /* Package the perturbation theory interpolation spline parameters */
        struct spline_params sp = {&spline, index_psi, tau_index, u_tau};

        /* Apply the transfer function (read only fgrf, output into fbox) */
        fft_apply_kernel(fbox, fgrf, N, BoxLen, kernel_transfer_function, &sp);

        /* Fourier transform to real space */
        fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fbox, box, FFTW_ESTIMATE);
        fft_execute(c2r);
        fft_normalize_c2r(box, N, BoxLen);
        fftw_destroy_plan(c2r);

        /* Retrieve  physical constant */
        const double c = us.SpeedOfLight;

        /* Multiply by the appropriate factor */
        for (int i=0; i<N*N*N; i++) {
            box[i] *= -2 / (c * c);
        }

        if (rank == 0 && pars.OutputFields) {
            char dnu_fname[50];
            sprintf(dnu_fname, "%s/ic_dnu.hdf5", pars.OutputDirectory);
            writeFieldFile(box, N, BoxLen, dnu_fname);
        }

    }

    header(rank, "Generating pre-initial conditions");
    message(rank, "ID of first particle = %lld\n", firstID);
    message(rank, "T_nu = %e eV\n", T_eV);

    /* Retrieve  physical constant */
    const double c = us.SpeedOfLight;

    /* Generate random neutrino particles */
    for (int i=0; i<localParticleNumber; i++) {
        struct particle_ext *p = &genparts[i];

        /* Set the ID of the particle */
        uint64_t id = i + firstID;

        /* Generate random particle velocity and position */
        init_neutrino_particle(id, m_eV, p->v, p->x, &p->mass, BoxLen, &us, T_eV);

        /* Compute the momentum in eV */
        const double p_eV = fermi_dirac_momentum(p->v, m_eV, us.SpeedOfLight);

        if (i==0)
        message(rank, "First random momentum = %e eV\n", p_eV);

        /* Determine the density perturbation at this point */
        double dnu = gridCIC(box, N, BoxLen, p->x[0], p->x[1], p->x[2]);

        /* The local temperature perturbation dT/T */
        double deltaT = dnu/4;

        /* Apply the perturbation */
        p->v[0] *= 1 + deltaT;
        p->v[1] *= 1 + deltaT;
        p->v[2] *= 1 + deltaT;

        const double f_i = fermi_dirac_density(p_eV, T_eV);

        /* Compute initial phase space density */
        p->f_i = f_i;

        /* Compute the magnitude of the initial velocity */
        p->v_i = hypot3(p->v[0], p->v[1], p->v[2]);

    }

    message(rank, "Done with pre-initial conditions.\n");

    header(rank, "Initiating geodesic integration.");

    /* Prepare integration */
    double a_begin = cosmo.a_begin;
    double a_end = cosmo.a_end;
    double a_factor = 1.0 + pars.ScaleFactorStep;

    int MAX_ITER = (log(a_end) - log(a_begin))/log(a_factor) + 1;

    message(rank, "Step size %.4f\n", a_factor-1);
    message(rank, "Doing %d iterations\n", MAX_ITER);
    message(rank, "\n");

    /* Start at the beginning */
    double a = a_begin;

    message(rank, "ITER]\t a\t z\t I\n");

    /* The main loop */
    for (int ITER = 0; ITER < MAX_ITER; ITER++) {

        /* Determine the next scale factor */
        double a_next;
        if (ITER < MAX_ITER - 1) {
            a_next = a * a_factor;
        } else {
            a_next = a_end;
        }

        /* Compute the current redshift and log conformal time */
        double z = 1./a - 1;
        double log_tau = perturbLogTauAtRedshift(&spline, z);

        /* Determine the half-step scale factor */
        double a_half = sqrt(a_next * a);

        /* Find the next and half-step conformal times */
        double z_next = 1./a_next - 1;
        double z_half = 1./a_half - 1;
        double log_tau_next = perturbLogTauAtRedshift(&spline, z_next);
        double log_tau_half = perturbLogTauAtRedshift(&spline, z_half);
        double dtau1 = exp(log_tau_half) - exp(log_tau);
        double dtau2 = exp(log_tau_next) - exp(log_tau_half);
        double dtau = dtau1 + dtau2;

        /* Find the interpolation index along the time dimension */
        int tau_index; //greatest lower bound bin index
        double u_tau; //spacing between subsequent bins
        perturbSplineFindTau(&spline, log_tau, &tau_index, &u_tau);

        /* The index of the potential transfer function psi */
        int index_psi = findTitle(ptdat.titles, "psi", ptdat.n_functions);

        /* Package the perturbation theory interpolation spline parameters */
        struct spline_params sp = {&spline, index_psi, tau_index, u_tau};

        /* Apply the transfer function (read only fgrf, output into fbox) */
        fft_apply_kernel(fbox, fgrf, N, BoxLen, kernel_transfer_function, &sp);

        /* Fourier transform to real space */
        fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fbox, box, FFTW_ESTIMATE);
        fft_execute(c2r);
        fft_normalize_c2r(box, N, BoxLen);
        fftw_destroy_plan(c2r);

        if (rank == 0 && pars.OutputFields) {
            char psi_fname[50];
            sprintf(psi_fname, "%s/psi_%d.hdf5", pars.OutputDirectory, ITER);
            writeFieldFile(box, N, BoxLen, psi_fname);
        }

        /* Integrate the particles */
        #pragma omp parallel for
        for (int i=0; i<localParticleNumber; i++) {
            struct particle_ext *p = &genparts[i];

            /* Get the acceleration from the scalar potential psi */
            double acc[3];
            accelCIC(box, N, BoxLen, p->x, acc);

            /* Also fetch the value of the potential at the particle position */
            double psi = gridCIC(box, N, BoxLen, p->x[0], p->x[1], p->x[2]);
            double psi_c2 = psi / (c * c);

            /* Fetch the relativistic correction factors */
            double q = p->v_i;
            double epsfac = hypot(q, a * m_eV);
            double epsfac_inv = 1. / epsfac;
            double drift_factor = (3 - q * q * epsfac_inv * epsfac_inv) * psi_c2;

            /* Compute kick and drift factors */
            double kick = epsfac / c;
            double drift = epsfac_inv * (1.0 + drift_factor) * c;

            /* Execute first kick */
            p->v[0] -= acc[0] * kick * dtau1;
            p->v[1] -= acc[1] * kick * dtau1;
            p->v[2] -= acc[2] * kick * dtau1;

            /* Execute drift */
            p->x[0] += p->v[0] * drift * dtau;
            p->x[1] += p->v[1] * drift * dtau;
            p->x[2] += p->v[2] * drift * dtau;
        }

        /* Next, we will compute the potential at the half-step time */

        /* Find the interpolation index along the time dimension */
        perturbSplineFindTau(&spline, log_tau_half, &tau_index, &u_tau);

        /* Package the perturbation theory interpolation spline parameters */
        struct spline_params sp2 = {&spline, index_psi, tau_index, u_tau};

        /* Apply the transfer function (read only fgrf, output into fbox) */
        fft_apply_kernel(fbox, fgrf, N, BoxLen, kernel_transfer_function, &sp2);

        /* Fourier transform to real space */
        c2r = fftw_plan_dft_c2r_3d(N, N, N, fbox, box, FFTW_ESTIMATE);
        fft_execute(c2r);
        fft_normalize_c2r(box, N, BoxLen);
        fftw_destroy_plan(c2r);

        if (rank == 0 && pars.OutputFields) {
            char psi_fname[50];
            sprintf(psi_fname, "%s/psi_%db.hdf5", pars.OutputDirectory, ITER);
            writeFieldFile(box, N, BoxLen, psi_fname);
        }

        /* Integrate the particles */
        #pragma omp parallel for
        for (int i=0; i<localParticleNumber; i++) {
            struct particle_ext *p = &genparts[i];

            /* Get the acceleration from the scalar potential psi */
            double acc[3];
            accelCIC(box, N, BoxLen, p->x, acc);

            /* Fetch the relativistic correction factors */
            double q = p->v_i;
            double epsfac = hypot(q, a_half * m_eV);

            /* Compute kick factor */
            double kick = epsfac / c;

            /* Execute second kick */
            p->v[0] -= acc[0] * kick * dtau2;
            p->v[1] -= acc[1] * kick * dtau2;
            p->v[2] -= acc[2] * kick * dtau2;
        }

        /* Step forward */
        a = a_next;

        /* Compute weights for fraction of particles (diagnostics only) */
        int weight_compute_invfreq = 1000;

        /* Collect the I statistic */
        if (rank == 0) {
            double I_df = 0;

            #pragma omp parallel for reduction(+:I_df)
            for (int i=0; i<localParticleNumber; i+=weight_compute_invfreq) {
                struct particle_ext *p = &genparts[i];

                double p_eV = fermi_dirac_momentum(p->v, m_eV, c);
                double f = fermi_dirac_density(p_eV, T_eV);
                double w = (p->f_i - f)/p->f_i;
                I_df += w*w;
            }

            /* Compute summary statistic */
            I_df *= 0.5 / pars.NumPartGenerate * weight_compute_invfreq;

            message(rank, "%04d] %.2e %.2e %e\n", ITER, a, 1./a-1, I_df);
        }
    }

    /* Free memory */
    free(box);
    free(fgrf);
    free(fbox);

    /* Final operations before writing the particles to disk */
    #pragma omp parallel for
    for (int i=0; i<localParticleNumber; i++) {
        struct particle_ext *p = &genparts[i];

        /* Ensure that particles wrap */
        p->x[0] = fwrap(p->x[0], BoxLen);
        p->x[1] = fwrap(p->x[1], BoxLen);
        p->x[2] = fwrap(p->x[2], BoxLen);

        /* Update the mass (needs to happen before converting the velocities!)*/
        double p_eV = fermi_dirac_momentum(p->v, m_eV, us.SpeedOfLight);
        double f = fermi_dirac_density(p_eV, T_eV);
        double eps_eV = hypot(p_eV/a_end, m_eV);
        double eps = particle_mass / m_eV * eps_eV;
        double w = (p->f_i - f)/p->f_i;
        p->mass = eps * w;

        /* Convert to peculiar velocities */
        p->v[0] /= a_end;
        p->v[1] /= a_end;
        p->v[2] /= a_end;
    }

    header(rank, "Prepare output");

    char out_fname[200];
    sprintf(out_fname, "%s/%s", pars.OutputDirectory, pars.OutputFilename);
    message(rank, "Writing output to %s.\n", out_fname);

    if (rank == 0) {
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

        /* Close the group */
        H5Gclose(h_grp);

        /* Close the file */
        H5Fclose(h_out_file);
    }

    /* Ensure that all nodes are at the final time step */
    double a_min;
    MPI_Allreduce(&a, &a_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    /* Now open the file in parallel mode */
    hid_t h_out_file = openFile_MPI(MPI_COMM_WORLD, out_fname);

    /* The particle group in the output file */
    hid_t h_grp = H5Gopen(h_out_file, pars.ExportName, H5P_DEFAULT);

    /* Datsets */
    hid_t h_data;

    /* Vector dataspace (e.g. positions, velocities) */
    const hsize_t vrank = 2;
    h_data = H5Dopen2(h_grp, "Velocities", H5P_DEFAULT);
    hid_t h_vspace = H5Dget_space(h_data);
    H5Dclose(h_data);

    /* Scalar dataspace (e.g. masses, particle ids) */
    const hsize_t srank = 1;
    h_data = H5Dopen2(h_grp, "ParticleIDs", H5P_DEFAULT);
    hid_t h_sspace = H5Dget_space(h_data);
    H5Dclose(h_data);

    /* Create vector & scalar datapsace for smaller chunks of data */
    const hsize_t ch_vdims[2] = {localParticleNumber, 3};
    const hsize_t ch_sdims[2] = {localParticleNumber};
    hid_t h_ch_vspace = H5Screate_simple(vrank, ch_vdims, NULL);
    hid_t h_ch_sspace = H5Screate_simple(srank, ch_sdims, NULL);

    /* The start of this chunk, in the overall vector & scalar spaces */
    const hsize_t start_in_group = localFirstNumber;
    const hsize_t vstart[2] = {start_in_group, 0}; //always with the "x" coordinate
    const hsize_t sstart[1] = {start_in_group};

    /* Choose the corresponding hyperslabs inside the overall spaces */
    H5Sselect_hyperslab(h_vspace, H5S_SELECT_SET, vstart, NULL, ch_vdims, NULL);
    H5Sselect_hyperslab(h_sspace, H5S_SELECT_SET, sstart, NULL, ch_sdims, NULL);

    /* Unpack particle data into contiguous arrays */
    double *coords = malloc(3 * localParticleNumber * sizeof(double));
    double *vels = malloc(3 * localParticleNumber * sizeof(double));
    double *masses = malloc(1 * localParticleNumber * sizeof(double));
    long long *ids = malloc(1 * localParticleNumber * sizeof(long long));
    for (int i=0; i<localParticleNumber; i++) {
        coords[i * 3 + 0] = genparts[i].x[0];
        coords[i * 3 + 1] = genparts[i].x[1];
        coords[i * 3 + 2] = genparts[i].x[2];
        vels[i * 3 + 0] = genparts[i].v[0];
        vels[i * 3 + 1] = genparts[i].v[1];
        vels[i * 3 + 2] = genparts[i].v[2];
        masses[i] = genparts[i].mass;
        ids[i] = firstID + i;
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

    /* Done with MPI parallelization */
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

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
