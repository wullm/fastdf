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

// Returns the number of local particles generated
long long run_fastdf(struct params *pars, struct units *us) {
    
    /* Initialize MPI for distributed memory parallelization */
    fftw_mpi_init();

    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);

    /* The level of verbosity */
    const int verbosity = pars->Verbosity;
    const int verbosity_low = rank + (verbosity == 0); // low priority
    const int verbosity_high = rank; // high priority

    /* Read options */
    header(verbosity_high, "FastDF Neutrino Initial Condition Generator");

    struct perturb_data ptdat;
    struct perturb_spline spline;
    struct perturb_params ptpars;

    /* Store the MPI rank */
    pars->rank = rank;

    /* The base output file name */
    char out_fname[200];
    sprintf(out_fname, "%s", pars->OutputFilename);

    /* The file name for the current rank */
    char out_fname_local[220];
    if (pars->DistributedFiles && MPI_Rank_Count > 1) {
        sprintf(out_fname_local, "%s.%d", out_fname, pars->rank);
    } else {
        sprintf(out_fname_local, "%s", out_fname);
    }

    /* Check if this file already exists */
    int file_exists = fileExists(out_fname_local);

    /* The ExportName for the neutrino particles */
    const char *ExportName = pars->ExportName;
    int group_exists = groupExists(out_fname_local, ExportName);

    if (file_exists && group_exists) {
        printf("Error: file '%s' already exists and has a neutrino particle group.\n", out_fname_local);
        exit(1);
    } else if (file_exists && !group_exists) {
        message(verbosity_low, "Appending neutrino particles to '%s'.\n", out_fname_local);
    }

    /* Check if the user specified a perturbation data file or if CLASS
     * is to be run. If so, FastDF must be compiled with CLASS. */
    if (pars->PerturbFile[0] != '\0') {
        /* Read the perturbation data file */
        readPerturb(pars, us, &ptdat);
        readPerturbParams(pars, us, &ptpars);
    } else {
#ifdef WITH_CLASS
        /* Run CLASS */
        run_class(&ptdat, us, &ptpars, pars->ClassIniFile, /* verbose = */ pars->rank == 0);
#else
        printf("\n");
        printf("Error: Not compiled with CLASS.\n");
        printf("You could reconfigure with \"./configure --with-class=/your/class\" and then \n");
        printf("\"make clean && make\". Alternatively, you could provide a perturbation vector file.\n");
        return 1;
#endif
    }

    /* Integration limits */
    double a_begin = pars->ScaleFactorBegin;
    double a_end = pars->ScaleFactorEnd;
    double a_factor = 1.0 + pars->ScaleFactorStep;

    const double h = ptpars.h;
    const double H_0 = h * 100 * KM_METRES / MPC_METRES * us->UnitTimeSeconds;
    const double rho_crit = 3.0 * H_0 * H_0 / (8. * M_PI * us->GravityG);

    header(verbosity_low, "Neutrino species");

    /* Package physical constants */
    const int N_nu = ptpars.N_ncdm;
    double m_eV[N_nu];
    double T_nu[N_nu];
    double T_eV[N_nu];
    for (int i = 0; i < N_nu; i++) {
        m_eV[i] = ptpars.M_ncdm_eV[i];
        T_nu[i] = ptpars.T_ncdm[i] * ptpars.T_CMB;
        T_eV[i] = T_nu[i] * us->kBoltzmann / us->ElectronVolt;

        message(verbosity_low, "[%d] (m, T) = (%g eV, %g K)\n", i, m_eV[i], T_nu[i]);
    }

    /* Retrieve further physical constant */
    const double c = us->SpeedOfLight;
    const double inv_c = 1.0 / c;
    const double inv_c2 = inv_c * inv_c;

    /* Initialize the interpolation spline for the perturbation data */
    initPerturbSpline(&spline, DEFAULT_K_ACC_TABLE_SIZE, &ptdat);

    header(verbosity_low, "Simulation parameters");
    message(verbosity_low, "We want %lld (%d^3) particles\n", pars->NumPartGenerate, pars->CubeRootNumber);

    /* Check if we recognize the output gauge */
    int gauge_Nbody = 0;
    if (strcmp(pars->Gauge, "Newtonian") == 0 ||
        strcmp(pars->Gauge, "newtonian") == 0) {
        message(verbosity_low, "Output gauge: Newtonian\n");
    } else if (strcmp(pars->Gauge, "N-body") == 0 ||
               strcmp(pars->Gauge, "n-body") == 0 ||
               strcmp(pars->Gauge, "nbody") == 0) {
        message(verbosity_low, "Output gauge: N-body\n");
        gauge_Nbody = 1;
    } else {
        printf("Error: unknown output gauge '%s'.\n", pars->Gauge);
        exit(1);
    }

    /* Check if we recognize the output velocity type */
    int velocity_type = 0;
    if (strcmp(pars->VelocityType, "peculiar") == 0 ||
        strcmp(pars->VelocityType, "Peculiar") == 0) {
        message(verbosity_low, "Output velocities: peculiar (a*dx/dt)\n");
    } else if (strcmp(pars->VelocityType, "Gadget") == 0 ||
               strcmp(pars->VelocityType, "gadget") == 0) {
        message(verbosity_low, "Output velocities: Gadget (a^.5*dx/dt)\n");
        velocity_type = 1;
    } else {
        printf("Error: unknown output velocity type '%s'.\n", pars->VelocityType);
        exit(1);
    }

    if (pars->IncludeHubbleFactors) {
        printf("Output coordinates in U_L h^-1.\n");
        printf("Output masses in U_M h^-1.\n");
    }

    message(verbosity_low, "a_begin = %.3e (z = %.2f)\n", a_begin, 1./a_begin - 1);
    message(verbosity_low, "a_end = %.3e (z = %.2f)\n", a_end, 1./a_end - 1);

    const char use_nonsymplectic_eom = pars->NonSymplecticEquations;
    if (use_nonsymplectic_eom) {
        message(verbosity_low, "Using non-symplectic equations of motion!\n");
    }

    const char use_alternative_eom = pars->AlternativeEquations;
    if (use_alternative_eom) {
        message(verbosity_low, "\n\n");
        message(verbosity_low, "WARNING: Using alternative equations of motion!");
        message(verbosity_low, "\n\n");
    }

    /* Read the Gaussian random field on each MPI rank */
    double *box;
    double BoxLen = 0;
    int N;

    header(verbosity_low, "Random phases");
    message(verbosity_low, "Reading Gaussian random field from %s (dataset %s).\n", pars->GaussianRandomFieldFile, pars->GaussianRandomFieldDataset);

    readFieldFileDataSet(&box, &N, &BoxLen, pars->GaussianRandomFieldFile, pars->GaussianRandomFieldDataset);
    // readFieldFile(&box, &N, &BoxLen, pars->GaussianRandomFieldFile);
    // readFieldFile_MPI(&box, &N, &BoxLen, MPI_COMM_WORLD, pars->GaussianRandomFieldFile);

    /* Override the box length? */
    if (BoxLen > 0.) {
        pars->BoxLen = BoxLen;
    } else {
        /* Otherwise, look for a user-specified value */
        BoxLen = pars->BoxLen;
    }

    message(verbosity_low, "BoxLen = %.2f U_L\n", BoxLen);
    message(verbosity_low, "GridSize = %d\n", N);

    /* The order of grid interpolation to be used in the main loop */
    const int grid_interp_order = pars->InterpolationOrder;
    if (grid_interp_order == 1) {
        message(verbosity_low, "Interpolation = %s\n", "NGP");
    } else if (grid_interp_order == 2) {
        message(verbosity_low, "Interpolation = %s\n", "CIC");
    } else {
        printf("Unsupported grid interpolation order (use 1 = NGP or 2 = CIC).\n");
        exit(1);
    }

    /* Do we want to invert the field for paired simulations? */
    if (pars->InvertField) {
        message(verbosity_low, "Inverting input field.\n");
        for (int i = 0; i < N * N * N; i++) {
            box[i] *= -1.0;
        }
    }

    /* Compute the mean of the input field */
    double mean = 0.;
    for (int i = 0; i < N * N * N; i++) {
        mean += box[i];
    }
    mean /= (double) (N * N * N);

    /* Compute the variance of the input field */
    double var = 0.;
    for (int i = 0; i < N * N * N; i++) {
        var += (box[i] - mean) * (box[i] - mean);
    }
    var /= (double) (N * N * N);

    message(verbosity_low, "Mean = %.10g\n", mean);
    message(verbosity_low, "Variance = %.10g\n", var);

    /* Do we want to normalize the Gaussian field? */
    if (pars->NormalizeGaussianField) {
        message(verbosity_low, "\nNormalizing the input field.\n");

        /* Set the variance to 1.0, then apply the correct normalization */
        const double inv_sdev = 1.0 / sqrt(var);
        const double norm = inv_sdev * pow(N / BoxLen, 1.5);
        message(verbosity_low, "Normalizing by %g.\n", norm);

        for (int i = 0; i < N * N * N; i++) {
            box[i] *= norm;
        }
    }

    /* Do we want to correct the Monofonic normalization? */
    if (pars->AssumeMonofonicNormalization) {
        message(verbosity_low, "\nUndoing the monofonIC normalization.\n");

        /* Apply the correct normalization, relative to monofonIC */
        const double norm = pow(h / (2.0 * M_PI), 1.5);
        message(verbosity_low, "Normalizing by %g.\n", norm);

        for (int i = 0; i < N * N * N; i++) {
            box[i] *= norm;
        }
    }

    /* Fourier transform the Gaussian random field */
    fftw_complex *fbox = malloc(N*N*N*sizeof(fftw_complex));
    fftw_complex *fgrf = malloc(N*N*N*sizeof(fftw_complex));
    fftw_plan r2c = fftw_plan_dft_r2c_3d(N, N, N, box, fbox, FFTW_ESTIMATE);
    fft_execute(r2c);
    fft_normalize_r2c(fbox, N, BoxLen);
    fftw_destroy_plan(r2c);

    /* Do we need to apply the primordial power spectrum? */
    if (pars->NormalizeGaussianField || pars->AssumeMonofonicNormalization) {
        struct power_spectrum ps;
        ps.A_s = pars->PrimordialScalarAmplitude;
        ps.n_s = pars->PrimordialSpectralIndex;
        ps.k_pivot = pars->PrimordialPivotScale;
        ps.alpha_s = pars->PrimordialRunning;
        ps.beta_s = pars->PrimordialRunningSecond;

        message(verbosity_low, "\nApplying primodial power spectrum with\n");
        message(verbosity_low, "A_s = %.5e\n", ps.A_s);
        message(verbosity_low, "n_s = %.5g\n", ps.n_s);
        message(verbosity_low, "k_pivot = %.5g\n", ps.k_pivot);
        message(verbosity_low, "alpha_s = %.5g\n", ps.alpha_s);
        message(verbosity_low, "beta_s = %.5g\n", ps.beta_s);

        /* Apply the bare power spectrum to fbox */
        fft_apply_kernel(fbox, fbox, N, BoxLen, kernel_power_no_transfer, &ps);
    }

    /* Make a copy of the complex Gaussian random field */
    memcpy(fgrf, fbox, N*N*N*sizeof(fftw_complex));

    /* If we are outputting in N-body gauge, check that all the necessary
     * transfer functions are there before doing the integrtion. */
    if (gauge_Nbody) {
        /* The indices of the necessary transfer function */
        int index_hdot = findTitle(ptdat.titles, "h_prime", ptdat.n_functions);
        int index_etadot = findTitle(ptdat.titles, "eta_prime", ptdat.n_functions);
        int index_Nbshift = findTitle(ptdat.titles, "delta_shift_Nb_m", ptdat.n_functions);
        int index_HTNbp = findTitle(ptdat.titles, "H_T_Nb_prime", ptdat.n_functions);
        if (index_hdot < 0 || index_etadot < 0 || index_Nbshift < 0 || index_HTNbp < 0) {
            printf("Error: required transfer function not found (%d, %d, %d, %d).\n", index_hdot, index_etadot, index_Nbshift, index_HTNbp);
            return 1;
        }
    }

    /* The index of the present day, corresponds to the last index in the array */
    int today_index = ptdat.tau_size - 1;

    /* Find the present-day density for each neutrino species, as fraction of the critical density */
    const double box_vol = BoxLen * BoxLen * BoxLen;
    double particle_mass[N_nu];
    for (int i = 0; i < N_nu; i++) {
        /* Generate the title */
        char *title = malloc(DEFAULT_STRING_LENGTH);
        if (pars->UseExplicitTransferFunctionTitle) {
            title = pars->TransferFunctionDensity;
        } else {
            sprintf(title, "d_ncdm[%d]", i);
        }

        int index_src = findTitle(ptdat.titles, title, ptdat.n_functions);
        if (index_src < 0) {
            printf("Error: required transfer function not found '%s' (%d).\n", title, index_src);
            return 1;
        }
        free(title);

        const double Omega = ptdat.Omega[ptdat.tau_size * index_src + today_index];
        const double rho = Omega * rho_crit;
        particle_mass[i] = rho * box_vol / pars->NumPartGenerate * N_nu;
    }

    /* Store the Box Length */
    pars->BoxLen = BoxLen;

    /* Determine the number of particle to be generated on each rank */
    header(verbosity_low, "Particle distribution");

    /* The particle number is M^3 */
    long long M = pars->CubeRootNumber;

    /* Determine what particles belong to this slice */
    double fac = (double) M / MPI_Rank_Count;
    long long X_min = ceil(rank * fac);
    long long X_max = ceil((rank + 1) * fac);
    long long MX = X_max - X_min;
    long long localParticleNumber = MX * M * M;
    long long localFirstNumber = X_min * M * M;

    /* Check that all particles have been assigned to a node */
    long long totalParticlesAssigned;
    MPI_Allreduce(&localParticleNumber, &totalParticlesAssigned, 1,
                   MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    assert(totalParticlesAssigned == M * M * M);
    if (verbosity)
    printf("%03d: Local particles [%04lld, %04lld], first = %lld, last = %lld, total = %lld\n", rank, X_min, X_max, localFirstNumber, localFirstNumber + localParticleNumber - 1, totalParticlesAssigned);

    /* Check that we are not exceeding the HDF5 parallel write limit */
    if (localParticleNumber * 3 * sizeof(double) > HDF5_PARALLEL_LIMIT) {
        printf("\nError: Exceeding 2GB limit on parallel HDF5 writes. Increase the number of MPI tasks.\n");
        exit(1);
    }

    /* The particles to be generated on this node */
    struct particle_ext *genparts = calloc(localParticleNumber,
                                           sizeof(struct particle_ext));

    /* ID of the first particle on this node */
    long long firstID = pars->FirstID + localFirstNumber;

    MPI_Barrier(MPI_COMM_WORLD);

    double z_begin = 1./a_begin - 1;
    double log_tau_begin = perturbLogTauAtRedshift(&spline, z_begin);

    /* Allocate boxes for the initial conditions */
    double *box_dic = malloc(N*N*N*sizeof(double));
    double *box_tic = malloc(N*N*N*sizeof(double));


    header(verbosity_low, "Generating pre-initial conditions");
    message(verbosity_low, "Generating pre-initial grids.\n");
    {

        /* Find the interpolation index along the time dimension */
        int tau_index; //greatest lower bound bin index
        double u_tau; //spacing between subsequent bins
        perturbSplineFindTau(&spline, log_tau_begin, &tau_index, &u_tau);

        /* The indices of the potential transfer function */
        int index_psi = findTitle(ptdat.titles, "psi", ptdat.n_functions);
        if (index_psi < 0) {
            printf("Error: transfer function '%s' not found (%d).\n", "psi", index_psi);
            return 1;
        }

        /* Package the perturbation theory interpolation spline parameters */
        struct spline_params sp = {&spline, index_psi, tau_index, u_tau};

        /* Apply the transfer function (read only fgrf, output into fbox) */
        fft_apply_kernel(fbox, fgrf, N, BoxLen, kernel_transfer_function, &sp);

        /* Fourier transform to real space */
        fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fbox, box, FFTW_ESTIMATE);
        fft_execute(c2r);
        fft_normalize_c2r(box, N, BoxLen);
        fftw_destroy_plan(c2r);

        /* Copy into the boxes for the density and velocity perturbations */
        memcpy(box_dic, box, N*N*N*sizeof(double));
        memcpy(box_tic, box, N*N*N*sizeof(double));

        /* Multiply by the appropriate factor */
        for (int i=0; i<N*N*N; i++) {
            box_dic[i] *= -2 * inv_c2;
            box_tic[i] *= 0.5 * exp(log_tau_begin);
        }

        if (rank == 0 && pars->OutputFields) {
            char dnu_fname[50];
            sprintf(dnu_fname, "%s/ic_dnu.hdf5", pars->OutputDirectory);
            writeFieldFile(box_dic, N, BoxLen, dnu_fname);

            char tnu_fname[50];
            sprintf(tnu_fname, "%s/ic_tnu.hdf5", pars->OutputDirectory);
            writeFieldFile(box_tic, N, BoxLen, tnu_fname);
        }

    }

    message(verbosity_low, "ID of first particle = %lld\n", firstID);
    message(verbosity_low, "T_nu = %e eV\n", T_eV);

    /* Generate random neutrino particles */
    for (long long i=0; i<localParticleNumber; i++) {
        struct particle_ext *p = &genparts[i];

        /* Set the ID of the particle */
        uint64_t id = i + firstID;

        /* We cycle through the neutrino species based on the seed */
        int i_ncdm = id % N_nu;

        /* Generate random particle velocity and position */
        init_neutrino_particle(id, m_eV[i_ncdm], p->v, p->x, &p->mass, BoxLen, us, T_eV[i_ncdm]);

        /* Compute the momentum in eV */
        const double p_eV = fermi_dirac_momentum(p->v, m_eV[i_ncdm], us->SpeedOfLight);
        const double f_i = fermi_dirac_density(p_eV, T_eV[i_ncdm]);

        if (i==0)
        message(verbosity_low, "First random momentum = %e eV\n", p_eV);

        /* Determine the density perturbation at this point */
        double dnu = gridCIC(box_dic, N, BoxLen, p->x[0], p->x[1], p->x[2]);

        /* The local temperature perturbation dT/T */
        double deltaT = dnu/4;

        /* Determine the initial velocity perturbation at this point */
        double vel[3];
        accelCIC(box_tic, N, BoxLen, p->x, vel);

        /* Apply the perturbation */
        p->v[0] *= 1.0 + deltaT;
        p->v[1] *= 1.0 + deltaT;
        p->v[2] *= 1.0 + deltaT;

        /* The current energy */
        double eps_eV = hypot(p_eV/a_begin, m_eV[i_ncdm]);

        /* Apply the velocity perturbation */
        p->v[0] += vel[0] * inv_c * eps_eV * a_begin;
        p->v[1] += vel[1] * inv_c * eps_eV * a_begin;
        p->v[2] += vel[2] * inv_c * eps_eV * a_begin;

        /* Compute initial phase space density */
        p->f_i = f_i;

        /* Compute the magnitude of the initial velocity */
        p->v_i[0] = p->v[0];
        p->v_i[1] = p->v[1];
        p->v_i[2] = p->v[2];
        p->v_i_mag = hypot3(p->v[0], p->v[1], p->v[2]);
    }

    /* Free the boxes used for the initial conditions */
    free(box_dic);
    free(box_tic);

    message(verbosity_low, "Done with pre-initial conditions.\n");

    header(verbosity_high, "Initiating neutrino geodesic integration.");

    /* Prepare integration */
    int MAX_ITER = (log(a_end) - log(a_begin))/log(a_factor) + 1;

    message(verbosity_high, "Step size %.4f\n", a_factor-1);
    message(verbosity_high, "Doing %d iterations\n", MAX_ITER);
    message(verbosity_high, "\n");

    /* We will re-compute the potentials when they have changed a certain amount */
    const double recompute_trigger = pars->RecomputeTrigger;
    double k_ref;  // 1 / U_L (reference scale)
    if (pars->RecomputeScaleRef <= 0.0) {
        /* Set equal to the smallest scale that will be needed */
        k_ref = 2.0 * M_PI * sqrt(3.0) * N / BoxLen;
    } else {
        k_ref = pars->RecomputeScaleRef;
    }

    message(verbosity_low, "Recompute trigger: %f\n", recompute_trigger);
    message(verbosity_low, "Recompute k_ref: %f 1/U_L\n", k_ref);
    message(verbosity_low, "\n");

    /* Start at the beginning */
    double a = a_begin;

    message(verbosity_high, "     ITER a         z         I            recompute\n");

    /* Allocate grids for the potentials (phi and psi) at two different times
     * t0 <= t < t1. We will evaluate at time t using linear interpolation. */
    double *box_phi0 = malloc(N*N*N*sizeof(double));
    double *box_phi1 = malloc(N*N*N*sizeof(double));
    double *box_psi0 = malloc(N*N*N*sizeof(double));
    double *box_psi1 = malloc(N*N*N*sizeof(double));

    /* The interpolated grids at time t */
    double *box_phi = malloc(N*N*N*sizeof(double));
    double *box_psi = malloc(N*N*N*sizeof(double));

    /* Time derivative at time t */
    double *box_phi_dot = malloc(N*N*N*sizeof(double));

    /* Complex grids that will be used */
    fftw_complex *fbox_phi = malloc(N*N*N*sizeof(fftw_complex));
    fftw_complex *fbox_psi = malloc(N*N*N*sizeof(fftw_complex));

    /* Transer functions evaluated at a reference scale during the last update */
    double Tk_phi = 0.0;
    double Tk_psi = 0.0;

    /* Find the k index and spacing at the reference scale */
    int k_ref_index;
    double u_ref_k;
    perturbSplineFindK(&spline, k_ref, &k_ref_index, &u_ref_k);

    /* Time and interpolation indices of the last major update */
    double log_tau_major_prev = 0.;
    int tau_idx_major_prev;
    double u_major_prev;

    /* Time and interpolation indices of the next major update */
    double log_tau_major_next = 0.;
    int tau_idx_major_next;
    double u_major_next;

    /* The main loop */
    for (int ITER = 0; ITER < MAX_ITER; ITER++) {

        /* Determine the next scale factor */
        double a_next;
        if (ITER == 0) {
            a_next = a; //start with a step that does nothing
        } else if (ITER < MAX_ITER - 1) {
            a_next = a * a_factor;
        } else {
            a_next = a_end;
        }

        /* Compute the current redshift and log conformal time */
        double z = 1./a - 1.;
        double log_tau = perturbLogTauAtRedshift(&spline, z);

        /* Determine the half-step scale factor */
        double a_half = sqrt(a_next * a);

        /* Find the next and half-step conformal times */
        double z_next = 1./a_next - 1.;
        double z_half = 1./a_half - 1.;
        double log_tau_next = perturbLogTauAtRedshift(&spline, z_next);
        double log_tau_half = perturbLogTauAtRedshift(&spline, z_half);
        double dtau1 = exp(log_tau_half) - exp(log_tau);
        double dtau2 = exp(log_tau_next) - exp(log_tau_half);
        double dtau = dtau1 + dtau2;

        /* Find the interpolation indices along the time dimension */
        int tau_idx_curr, tau_idx_half, tau_idx_next;
        double u_curr, u_half, u_next;
        perturbSplineFindTau(&spline, log_tau, &tau_idx_curr, &u_curr);
        perturbSplineFindTau(&spline, log_tau_half, &tau_idx_half, &u_half);
        perturbSplineFindTau(&spline, log_tau_next, &tau_idx_next, &u_next);

        /* A flag indicating whether we recomputed the potentials in this step */
        int recompute = 0;

        /* The inddices of the potential transfer functions psi and phi */
        int index_psi = findTitle(ptdat.titles, "psi", ptdat.n_functions);
        int index_phi = findTitle(ptdat.titles, "phi", ptdat.n_functions);
        if (index_psi < 0 || index_phi < 0) {
            printf("Error: transfer function '%s' or '%s' not found (%d, %d).\n", "psi", "phi", index_psi, index_phi);
            return 1;
        }

        /* Evaluate the transfer functions at the reference scale */
        double Tk_phi_curr = perturbSplineInterp(&spline, k_ref_index, tau_idx_curr, u_ref_k, u_curr, index_phi);
        double Tk_psi_curr = perturbSplineInterp(&spline, k_ref_index, tau_idx_curr, u_ref_k, u_curr, index_psi);
        /* Relative change since the last major update */
        double Delta_Tk_phi = fabs((Tk_phi - Tk_phi_curr) / (Tk_phi + Tk_phi_curr));
        double Delta_Tk_psi = fabs((Tk_psi - Tk_psi_curr) / (Tk_psi + Tk_psi_curr));

        /* Do we need to recompute the potential grids? */
        if (ITER < 2 || ITER == MAX_ITER - 1 ||
            Delta_Tk_phi > recompute_trigger ||
            Delta_Tk_psi > recompute_trigger) {

            /* Find the time of the next major recompute step */
            double a_major_next = a;
            for (int j = ITER + 1; j < MAX_ITER; j++) {

                /* Step forward */
                if (j < MAX_ITER - 1) {
                    a_major_next = a_major_next * a_factor;
                } else {
                    a_major_next = a_end;
                }

                /* Compute the corresponding redshift and log conformal time */
                double z_major_next = 1./a_major_next - 1;
                log_tau_major_next = perturbLogTauAtRedshift(&spline, z_major_next);

                /* Find the interpolation indices along the time dimension */
                perturbSplineFindTau(&spline, log_tau_major_next, &tau_idx_major_next, &u_major_next);

                /* Evaluate the transfer functions */
                double Tk_phi_sub = perturbSplineInterp(&spline, k_ref_index, tau_idx_major_next, u_ref_k, u_major_next, index_phi);
                double Tk_psi_sub = perturbSplineInterp(&spline, k_ref_index, tau_idx_major_next, u_ref_k, u_major_next, index_psi);
                /* Relative change compared to the current time step (main outer loop) */
                double Delta_Tk_phi_sub = fabs((Tk_phi_curr - Tk_phi_sub) / (Tk_phi_curr + Tk_phi_sub));
                double Delta_Tk_psi_sub = fabs((Tk_psi_curr - Tk_psi_sub) / (Tk_psi_curr + Tk_psi_sub));

                /* Stop if we have found the next recompute step */
                if (j == MAX_ITER - 1 || Delta_Tk_phi_sub > recompute_trigger || Delta_Tk_psi_sub > recompute_trigger)
                    break;
            }

            /* The last major update was now */
            log_tau_major_prev = log_tau;
            perturbSplineFindTau(&spline, log_tau_major_prev, &tau_idx_major_prev, &u_major_prev);

            /* We are recomputing the potentials */
            recompute = 1;

            /* Update the reference transfer functions */
            Tk_phi = Tk_phi_curr;
            Tk_psi = Tk_psi_curr;

            /* Copy the current boxes at t=t1 to the slots for t=t0 */
            if (ITER > 0) {
                memcpy(box_phi0, box_phi1, N*N*N*sizeof(double));
                memcpy(box_psi0, box_psi1, N*N*N*sizeof(double));
            }

            /* Now compute the potentials at the next major time */

            /* Package the perturbation theory interpolation spline parameters */
            struct spline_params sp_phi_curr = {&spline, index_phi, tau_idx_major_next, u_major_next};
            struct spline_params sp_psi_curr = {&spline, index_psi, tau_idx_major_next, u_major_next};

            /* Apply the transfer function (read only fgrf, output into fbox) */
            fft_apply_kernel(fbox_phi, fgrf, N, BoxLen, kernel_transfer_function, &sp_phi_curr);
            fft_apply_kernel(fbox_psi, fgrf, N, BoxLen, kernel_transfer_function, &sp_psi_curr);

            /* Fourier transform to real space */
            fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fbox_phi, box_phi1, FFTW_ESTIMATE);
            fft_execute(c2r);
            fft_normalize_c2r(box_phi1, N, BoxLen);
            fftw_destroy_plan(c2r);

            fftw_plan c2r2 = fftw_plan_dft_c2r_3d(N, N, N, fbox_psi, box_psi1, FFTW_ESTIMATE);
            fft_execute(c2r2);
            fft_normalize_c2r(box_psi1, N, BoxLen);
            fftw_destroy_plan(c2r2);
        }

        /* Skip the particle integration during the first step */
        if (ITER == 0)
            continue;

        /* Linearly interpolate the potentials to the current time step */
        double o = (log_tau - log_tau_major_prev) / (log_tau_major_next - log_tau_major_prev);
        for (int i=0; i<N*N*N; i++) {
            box_phi[i] = o * box_phi1[i] + (1.0 - o) * box_phi0[i];
            box_psi[i] = o * box_psi1[i] + (1.0 - o) * box_psi0[i];
        }

        /* Compute the potential derivative grid */
        double inv_delta_tau = 1.0 / (exp(log_tau_major_next) - exp(log_tau_major_prev));
        for (int i=0; i<N*N*N; i++) {
            box_phi_dot[i] = (box_phi1[i] - box_phi0[i]) * inv_delta_tau;
        }

        if (rank == 0 && ((pars->OutputFields > 1 && recompute) || (pars->OutputFields > 2 && ITER % 10 == 0))) {
            char psi_fname[50];
            sprintf(psi_fname, "%s/psi_%d.hdf5", pars->OutputDirectory, ITER);
            writeFieldFile(box_psi, N, BoxLen, psi_fname);

            char phi_fname[50];
            sprintf(phi_fname, "%s/phi_%d.hdf5", pars->OutputDirectory, ITER);
            writeFieldFile(box_phi, N, BoxLen, phi_fname);

            char phi_dot_fname[50];
            sprintf(phi_dot_fname, "%s/phi_dot_%d.hdf5", pars->OutputDirectory, ITER);
            writeFieldFile(box_phi_dot, N, BoxLen, phi_dot_fname);
        }

        /* Integrate the particles */
        #pragma omp parallel for
        for (long long i=0; i<localParticleNumber; i++) {
            struct particle_ext *p = &genparts[i];

            /* We cycle through the neutrino species based on the seed */
            uint64_t id = i + firstID;
            int i_ncdm = id % N_nu;

            /* Get the acceleration from the scalar potential psi */
            double acc_psi[3];
            accelInterp(box_psi, N, BoxLen, p->x, acc_psi, grid_interp_order);

            /* Get the acceleration from the scalar potential phi */
            double acc_phi[3];
            accelInterp(box_phi, N, BoxLen, p->x, acc_phi, grid_interp_order);

            /* Also fetch the time derivative of phi at the particle position */
            double phi_dot = gridInterp(box_phi_dot, N, BoxLen, p->x[0], p->x[1], p->x[2], grid_interp_order);
            double phi_dot_c2 = phi_dot * inv_c2;

            /* The momentum vector & length to be used in the kick equation */
            double *q_vec, q_mag;

            if (use_nonsymplectic_eom) {
                /* Use the current momentum for the non-symplectic version */
                q_vec = p->v;
                q_mag = hypot3(p->v[0], p->v[1], p->v[2]);
            } else {
                /* Use the initial momentum for the symplectic formulation */
                q_vec = p->v_i;
                q_mag = p->v_i_mag;
            }

            /* Inner product of momentum with acc_phi */
            double vacx = q_vec[0] * acc_phi[0];
            double vacy = q_vec[1] * acc_phi[1];
            double vacz = q_vec[2] * acc_phi[2];
            double vac = vacx + vacy + vacz;

            /* Compute the relativistic correction factors */
            double q2 = q_mag * q_mag;
            double epsfac = hypot(q_mag, a * m_eV[i_ncdm]);
            double epsfac_inv = 1. / epsfac;

            /* Compute kick and drift factors */
            double kick_psi = epsfac * inv_c;
            double kick_phi = epsfac_inv * inv_c;
            double drift = epsfac_inv * c;

            if (use_nonsymplectic_eom) {
                /* Fetch the value of the potential at the particle position */
                double psi = gridInterp(box_psi, N, BoxLen, p->x[0], p->x[1], p->x[2], grid_interp_order);
                double phi = gridInterp(box_phi, N, BoxLen, p->x[0], p->x[1], p->x[2], grid_interp_order);
                double psi_c2 = psi * inv_c2;
                double phi_c2 = phi * inv_c2;

                /* Apply the non-symplectic relativistic drift factor */
                if (use_alternative_eom) {
                    drift *= (1.0 + psi_c2 + phi_c2 * (2.0 - q2 * epsfac_inv * epsfac_inv));
                } else {
                    drift *= (1.0 + psi_c2 + phi_c2);
                }
            }

            if (use_alternative_eom) {
                /* Zero out the anti-symmetric term */
                vac = 0.;
                /* Zero out the potential derivative term */
                phi_dot_c2 = 0;
            }

            /* Execute first gradient term */
            p->v[0] -= acc_psi[0] * kick_psi * dtau1;
            p->v[1] -= acc_psi[1] * kick_psi * dtau1;
            p->v[2] -= acc_psi[2] * kick_psi * dtau1;

            /* Add anti-symmetric gradient term (only rotates) */
            p->v[0] -= (q2 * acc_phi[0] - q_vec[0] * vac) * kick_phi * dtau1;
            p->v[1] -= (q2 * acc_phi[1] - q_vec[1] * vac) * kick_phi * dtau1;
            p->v[2] -= (q2 * acc_phi[2] - q_vec[2] * vac) * kick_phi * dtau1;

            /* Add potential derivative term */
            p->v[0] += q_vec[0] * phi_dot_c2 * dtau1;
            p->v[1] += q_vec[1] * phi_dot_c2 * dtau1;
            p->v[2] += q_vec[2] * phi_dot_c2 * dtau1;

            /* Execute drift (only one drift, so use dtau = dtau1 + dtau2) */
            p->x[0] += p->v[0] * drift * dtau;
            p->x[1] += p->v[1] * drift * dtau;
            p->x[2] += p->v[2] * drift * dtau;
        }

        /* Linearly interpolate the potentials to the half-time step */
        double o2 = (log_tau_half - log_tau_major_prev) / (log_tau_major_next - log_tau_major_prev);
        for (int i=0; i<N*N*N; i++) {
            box_phi[i] = o2 * box_phi1[i] + (1.0 - o2) * box_phi0[i];
            box_psi[i] = o2 * box_psi1[i] + (1.0 - o2) * box_psi0[i];
        }

        /* Integrate the particles during the second half-step */
        #pragma omp parallel for
        for (long long i=0; i<localParticleNumber; i++) {
            struct particle_ext *p = &genparts[i];

            /* We cycle through the neutrino species based on the seed */
            uint64_t id = i + firstID;
            int i_ncdm = id % N_nu;

            /* Get the acceleration from the scalar potential psi */
            double acc_psi[3];
            accelInterp(box_psi, N, BoxLen, p->x, acc_psi, grid_interp_order);

            /* Get the acceleration from the scalar potential phi */
            double acc_phi[3];
            accelInterp(box_phi, N, BoxLen, p->x, acc_phi, grid_interp_order);

            /* Also fetch the time derivative of phi at the particle position */
            double phi_dot = gridInterp(box_phi_dot, N, BoxLen, p->x[0], p->x[1], p->x[2], grid_interp_order);
            double phi_dot_c2 = phi_dot * inv_c2;

            /* The momentum vector & length to be used in the kick equation */
            double *q_vec, q_mag;

            if (use_nonsymplectic_eom) {
                /* Use the current momentum for the non-symplectic version */
                q_vec = p->v;
                q_mag = hypot3(p->v[0], p->v[1], p->v[2]);
            } else {
                /* Use the initial momentum for the symplectic formulation */
                q_vec = p->v_i;
                q_mag = p->v_i_mag;
            }

            /* Inner product of momentum with acc_phi */
            double vacx = q_vec[0] * acc_phi[0];
            double vacy = q_vec[1] * acc_phi[1];
            double vacz = q_vec[2] * acc_phi[2];
            double vac = vacx + vacy + vacz;

            /* Compute the relativistic correction factors */
            double q2 = q_mag * q_mag;
            double epsfac = hypot(q_mag, a * m_eV[i_ncdm]);
            double epsfac_inv = 1. / epsfac;

            /* Compute kick factors */
            double kick_psi = epsfac * inv_c;
            double kick_phi = epsfac_inv * inv_c;

            if (use_alternative_eom) {
                /* Zero out the anti-symmetric term */
                vac = 0.;
                /* Zero out the potential derivative term */
                phi_dot_c2 = 0;
            }

            /* Execute first gradient term */
            p->v[0] -= acc_psi[0] * kick_psi * dtau2;
            p->v[1] -= acc_psi[1] * kick_psi * dtau2;
            p->v[2] -= acc_psi[2] * kick_psi * dtau2;

            /* Add anti-symmetric gradient term (only rotates) */
            p->v[0] -= (q2 * acc_phi[0] - q_vec[0] * vac) * kick_phi * dtau2;
            p->v[1] -= (q2 * acc_phi[1] - q_vec[1] * vac) * kick_phi * dtau2;
            p->v[2] -= (q2 * acc_phi[2] - q_vec[2] * vac) * kick_phi * dtau2;

            /* Add potential derivative term */
            p->v[0] += q_vec[0] * phi_dot_c2 * dtau2;
            p->v[1] += q_vec[1] * phi_dot_c2 * dtau2;
            p->v[2] += q_vec[2] * phi_dot_c2 * dtau2;
        }

        /* Step forward */
        a = a_next;

        /* Compute weights for fraction of particles (diagnostics only) */
        long long weight_compute_invfreq = 1000;

        /* Collect the I statistic */
        if (rank == 0) {
            double I_df = 0;

            #pragma omp parallel for reduction(+:I_df)
            for (long long i=0; i<localParticleNumber; i+=weight_compute_invfreq) {
                struct particle_ext *p = &genparts[i];

                /* We cycle through the neutrino species based on the seed */
                uint64_t id = i + firstID;
                int i_ncdm = id % N_nu;

                double p_eV = fermi_dirac_momentum(p->v, m_eV[i_ncdm], c);
                double f = fermi_dirac_density(p_eV, T_eV[i_ncdm]);
                double w = (p->f_i - f)/p->f_i;
                I_df += w*w;
            }

            /* Compute summary statistic */
            I_df *= 0.5 / pars->NumPartGenerate * weight_compute_invfreq;

            int percent = (100.0 * (ITER + 1))/MAX_ITER;

            message(verbosity_high, "%3d%% %-4d %.3e %.3e %e %d\n", percent, ITER, a, 1./a-1, I_df, recompute);
        }
    }

    /* Free grids that are no longer needed */
    free(fbox_phi);
    free(fbox_psi);
    free(box_phi);
    free(box_psi);
    free(box_phi0);
    free(box_phi1);
    free(box_psi0);
    free(box_psi1);
    free(box_phi_dot);

    if (gauge_Nbody) {

        /* Allocate boxes for the (density and flux) gauge transformations */
        double *box_dshift = malloc(N*N*N*sizeof(double));
        double *box_tshift = malloc(N*N*N*sizeof(double));

        header(verbosity_low, "Generating N-body gauge transformation grid(s)");

        /* We do this for each neutrino species separately */
        for (int i_ncdm = 0; i_ncdm < N_nu; i_ncdm++) {

            message(verbosity_low, "Working on species %d/%d\n", i_ncdm, N_nu);

            /* Compute the isentropic ratio and equation of state at a_end */
            const double isen_ncdm = ncdm_isentropic_ratio(a_end, m_eV[i_ncdm], T_eV[i_ncdm]);
            const double w_ncdm = ncdm_equation_of_state(a_end, m_eV[i_ncdm], T_eV[i_ncdm]);
            message(verbosity_low, "Isentropic ratio = %f at a_end = %e\n", isen_ncdm, a_end);
            message(verbosity_low, "Equation of state = %f at a_end = %e\n", w_ncdm, a_end);

            {
                /* Final time at which to execute the gauge transformation */
                double z_end = 1./a_end - 1;
                double log_tau_end = perturbLogTauAtRedshift(&spline, z_end);

                /* Central difference */
                double a_min = a_end / 1.001;
                double z_min =  1./a_min - 1;
                double log_tau_min = perturbLogTauAtRedshift(&spline, z_min);
                double a_plus = a_end * 1.001;
                double z_plus =  1./a_plus - 1;
                double log_tau_plus = perturbLogTauAtRedshift(&spline, z_plus);

                /* Find the interpolation index along the time dimension */
                int tau_index; //greatest lower bound bin index
                double u_tau; //spacing between subsequent bins
                perturbSplineFindTau(&spline, log_tau_end, &tau_index, &u_tau);

                /* Generate the title */
                char *title = malloc(DEFAULT_STRING_LENGTH);
                if (pars->UseExplicitTransferFunctionTitle) {
                    title = pars->TransferFunctionDensity;
                } else {
                    sprintf(title, "d_ncdm[%d]", i_ncdm);
                }

                /* The indices of the necessary transfer function */
                int index_hdot = findTitle(ptdat.titles, "h_prime", ptdat.n_functions);
                int index_etadot = findTitle(ptdat.titles, "eta_prime", ptdat.n_functions);
                int index_Nbshift = findTitle(ptdat.titles, "delta_shift_Nb_m", ptdat.n_functions);
                int index_ncdm = findTitle(ptdat.titles, title, ptdat.n_functions);
                int index_HTNbp = findTitle(ptdat.titles, "H_T_Nb_prime", ptdat.n_functions);
                if (index_hdot < 0 || index_etadot < 0 || index_Nbshift < 0 ||
                    index_ncdm < 0 || index_HTNbp < 0) {
                    printf("Error: required transfer function not found (%d, %d, %d, %d, %d).\n", index_hdot, index_etadot, index_Nbshift, index_ncdm, index_HTNbp);
                    return 1;
                }
                free(title);

                /* Package the perturbation theory interpolation spline parameters */
                struct spline_params sp_hdot = {&spline, index_hdot, tau_index, u_tau};
                struct spline_params sp_etadot = {&spline, index_etadot, tau_index, u_tau};
                struct spline_params sp_Nbshift = {&spline, index_Nbshift, tau_index, u_tau};
                struct spline_params sp_HTNbp = {&spline, index_HTNbp, tau_index, u_tau};

                /* Allocate boxes for the complex terms needed for the gauge transformation */
                fftw_complex *fbox_alpha = malloc(N*N*N*sizeof(fftw_complex));
                fftw_complex *fbox_hdot = malloc(N*N*N*sizeof(fftw_complex));
                fftw_complex *fbox_etadot = malloc(N*N*N*sizeof(fftw_complex));
                fftw_complex *fbox_Nbshift = malloc(N*N*N*sizeof(fftw_complex));

                /* Apply the transfer function (read only fgrf, output into fbox) */
                fft_apply_kernel(fbox_hdot, fgrf, N, BoxLen, kernel_transfer_function, &sp_hdot);
                fft_apply_kernel(fbox_etadot, fgrf, N, BoxLen, kernel_transfer_function, &sp_etadot);
                fft_apply_kernel(fbox_Nbshift, fgrf, N, BoxLen, kernel_transfer_function, &sp_Nbshift);

                /* Compute alpha * k^2 = h_dot + 6 * eta_dot */
                for (int i=0; i<N*N*(N/2+1); i++) {
                    fbox_alpha[i] = fbox_hdot[i] + 6.0 * fbox_etadot[i];
                }

                /* Apply the inverse Poisson kernel -1/k^2 */
                fft_apply_kernel(fbox_alpha, fbox_alpha, N, BoxLen, kernel_inv_poisson, NULL);

                /* Free grids that are no longer needed */
                free(fbox_hdot);
                free(fbox_etadot);

                /* Compute the conformal time derivative of the background density */
                double Omega_nu = perturbDensityAtLogTau(&spline, log_tau_end, index_ncdm);
                double Omega_nu_min = perturbDensityAtLogTau(&spline, log_tau_min, index_ncdm);
                double Omega_nu_plus = perturbDensityAtLogTau(&spline, log_tau_plus, index_ncdm);

                double H = perturbHubbleAtLogTau(&spline, log_tau_end);
                double H_min = perturbHubbleAtLogTau(&spline, log_tau_min);
                double H_plus = perturbHubbleAtLogTau(&spline, log_tau_plus);

                double rho_crit_central = rho_crit * (H * H) / (H_0 * H_0);
                double rho_crit_min = rho_crit * (H_min * H_min) / (H_0 * H_0);
                double rho_crit_plus = rho_crit * (H_plus * H_plus) / (H_0 * H_0);

                double rho_nu = Omega_nu * rho_crit_central;
                double rho_nu_min = Omega_nu_min * rho_crit_min;
                double rho_nu_plus = Omega_nu_plus * rho_crit_plus;

                double dtau = exp(log_tau_plus) - exp(log_tau_min);
                double rho_dot = (rho_nu_plus - rho_nu_min) / dtau;
                double rho_dot_rho = rho_dot / rho_nu;
                double rho_dot_rho_c4 = rho_dot_rho / (c * c * c * c);

                /* Compute the total density gauge shift */
                for (int i=0; i<N*N*N; i++) {
                    fbox[i] = -0.5 * fbox_alpha[i] * rho_dot_rho_c4 - (1.0 + w_ncdm) * fbox_Nbshift[i];
                }

                /* Free grids that are no longer needed */
                free(fbox_alpha);
                free(fbox_Nbshift);

                /* Fourier transform to real space */
                fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fbox, box_dshift, FFTW_ESTIMATE);
                fft_execute(c2r);
                fft_normalize_c2r(box_dshift, N, BoxLen);
                fftw_destroy_plan(c2r);

                if (rank == 0 && pars->OutputFields) {
                    char density_fname[50];
                    sprintf(density_fname, "%s/gauge_dshift_%d.hdf5", pars->OutputDirectory, i_ncdm);
                    writeFieldFile(box_dshift, N, BoxLen, density_fname);
                }

                /* Allocate new complex grid */
                fftw_complex *fbox_HTNbp = malloc(N*N*N*sizeof(fftw_complex));

                /* Apply the transfer function for H_T_Nb_prime */
                fft_apply_kernel(fbox_HTNbp, fgrf, N, BoxLen, kernel_transfer_function, &sp_HTNbp);

                /* Apply the inverse Poisson kernel -1/k^2 */
                fft_apply_kernel(fbox_HTNbp, fbox_HTNbp, N, BoxLen, kernel_inv_poisson, NULL);

                /* Fourier transform to real space */
                fftw_plan c2r3 = fftw_plan_dft_c2r_3d(N, N, N, fbox_HTNbp, box_tshift, FFTW_ESTIMATE);
                fft_execute(c2r3);
                fft_normalize_c2r(box_tshift, N, BoxLen);
                fftw_destroy_plan(c2r3);

                if (rank == 0 && pars->OutputFields) {
                    char vshift_fname[50];
                    sprintf(vshift_fname, "%s/gauge_vshift_%d.hdf5", pars->OutputDirectory, i_ncdm);
                    writeFieldFile(box_tshift, N, BoxLen, vshift_fname);
                }

                /* Free the remaining complex grid */
                free(fbox_HTNbp);
            }

            message(verbosity_low, "Applying N-body gauge transformation to the particles of species %d.\n", i_ncdm);

            /* Perform the gauge transformation */
            #pragma omp parallel for
            for (long long i=0; i<localParticleNumber; i++) {
                struct particle_ext *p = &genparts[i];

                /* We cycle through the neutrino species based on the seed */
                uint64_t id = i + firstID;
                if (i_ncdm != id % N_nu) continue;

                /* Determine the gauge shift at this point */
                double delta = gridCIC(box_dshift, N, BoxLen, p->x[0], p->x[1], p->x[2]);

                /* The equivalent temperature perturbation dT/T */
                double deltaT = delta/isen_ncdm;

                /* Determine the velocity shift at this point */
                double vel[3];
                accelCIC(box_tshift, N, BoxLen, p->x, vel);

                /* Apply the density shift */
                p->v[0] *= 1 - deltaT;
                p->v[1] *= 1 - deltaT;
                p->v[2] *= 1 - deltaT;

                /* The current energy */
                double p_eV = fermi_dirac_momentum(p->v, m_eV[i_ncdm], us->SpeedOfLight);
                double eps_eV = hypot(p_eV/a_end, m_eV[i_ncdm]);

                /* Apply the velocity shift */
                p->v[0] += vel[0] * inv_c * eps_eV * a_end;
                p->v[1] += vel[1] * inv_c * eps_eV * a_end;
                p->v[2] += vel[2] * inv_c * eps_eV * a_end;
            }
        }

        /* Free the gauge transformation boxes */
        free(box_dshift);
        free(box_tshift);

        message(verbosity_low, "\n");
    }

    /* Free memory */
    free(box);
    free(fgrf);
    free(fbox);

    /* Compute energies, masses, and weights and store them in contiguous arrays */
    double *energies = malloc(1 * localParticleNumber * sizeof(double));
    double *masses = malloc(1 * localParticleNumber * sizeof(double));
    double *weights = malloc(1 * localParticleNumber * sizeof(double));
    double *phaseDensities = malloc(1 * localParticleNumber * sizeof(double));
    for (long long i=0; i<localParticleNumber; i++) {
        struct particle_ext *p = &genparts[i];

        /* We cycle through the neutrino species based on the seed */
        uint64_t id = i + firstID;
        int i_ncdm = id % N_nu;

        /* Compute the energy & weight (needs to happen before converting the velocities!)*/
        double p_eV = fermi_dirac_momentum(p->v, m_eV[i_ncdm], us->SpeedOfLight);
        double eps_eV = hypot(p_eV/a_end, m_eV[i_ncdm]);
        double eps = particle_mass[i_ncdm] / m_eV[i_ncdm] * eps_eV;
        double f = fermi_dirac_density(p_eV, T_eV[i_ncdm]);
        double w = (p->f_i - f)/p->f_i;
        // p->mass = particle_mass * w;

        energies[i] = eps;
        masses[i] = particle_mass[i_ncdm];
        weights[i] = w;
        phaseDensities[i] = p->f_i;

        /* Include Hubble factors for Gadget ICs? */
        if (pars->IncludeHubbleFactors) {
            energies[i] *= h;
            masses[i] *= h;
        }
    }

    /* Final operations before writing the particles to disk */
    #pragma omp parallel for
    for (long long i=0; i<localParticleNumber; i++) {
        struct particle_ext *p = &genparts[i];

        /* We cycle through the neutrino species based on the seed */
        uint64_t id = i + firstID;
        int i_ncdm = id % N_nu;

        /* Ensure that particles wrap */
        p->x[0] = fwrap(p->x[0], BoxLen);
        p->x[1] = fwrap(p->x[1], BoxLen);
        p->x[2] = fwrap(p->x[2], BoxLen);

        /* Convert momenta to velocities */
        p->v[0] *= c / m_eV[i_ncdm];
        p->v[1] *= c / m_eV[i_ncdm];
        p->v[2] *= c / m_eV[i_ncdm];

        /* Convert to peculiar velocities */
        p->v[0] /= a_end;
        p->v[1] /= a_end;
        p->v[2] /= a_end;

        /* Possibly convert to Gadget velocities */
        if (velocity_type == 1) {
            p->v[0] /= sqrt(a_end);
            p->v[1] /= sqrt(a_end);
            p->v[2] /= sqrt(a_end);
        }

        /* Include Hubble factors for Gadget ICs? */
        if (pars->IncludeHubbleFactors) {
            p->x[0] *= h;
            p->x[1] *= h;
            p->x[2] *= h;
        }
    }

    header(verbosity_high, "Prepare neutrino output");

    if (rank == 0 || pars->DistributedFiles) {
        /* The number of particles in the file */
        long long parts_in_file = pars->DistributedFiles ? localParticleNumber : pars->NumPartGenerate;

        /* Create the output file if it does not exist */
        hid_t h_out_file;
        if (!fileExists(out_fname_local)) {
            /* Create the file */
            h_out_file = createFile(out_fname_local);

            /* Writing attributes into the Header & Cosmology groups */
            int err = writeHeaderAttributes(pars, us, parts_in_file, pars->NumPartGenerate, h_out_file);
            if (err > 0) exit(1);
        } else {
            /* Otherwise, open the file in read & write mode */
            h_out_file = H5Fopen(out_fname_local, H5F_ACC_RDWR , H5P_DEFAULT);
        }

        /* The ExportName */
        const char *ExportName = pars->ExportName;

        /* The particle group in the output file */
        hid_t h_grp;

        /* Datsets */
        hid_t h_data;

        /* Vector dataspace (e.g. positions, velocities) */
        const hsize_t vrank = 2;
        const hsize_t vdims[2] = {parts_in_file, 3};
        hid_t h_vspace = H5Screate_simple(vrank, vdims, NULL);

        /* Scalar dataspace (e.g. masses, particle ids) */
        const hsize_t srank = 1;
        const hsize_t sdims[1] = {parts_in_file};
        hid_t h_sspace = H5Screate_simple(srank, sdims, NULL);

        /* Set chunking for vectors */
        hid_t h_prop_vec = H5Pcreate(H5P_DATASET_CREATE);
        const hsize_t vchunk[2] = {HDF5_CHUNK_SIZE, 3};
        H5Pset_chunk(h_prop_vec, vrank, vchunk);

        /* Set chunking for scalars */
        hid_t h_prop_sca = H5Pcreate(H5P_DATASET_CREATE);
        const hsize_t schunk[1] = {HDF5_CHUNK_SIZE};
        H5Pset_chunk(h_prop_sca, srank, schunk);

        /* Create the particle group in the output file */
        message(verbosity_low, "Creating Group '%s' with %lld particles.\n", ExportName, pars->NumPartGenerate);
        h_grp = H5Gcreate(h_out_file, ExportName, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        /* Coordinates (use vector space) */
        h_data = H5Dcreate(h_grp, "Coordinates", H5T_NATIVE_DOUBLE, h_vspace, H5P_DEFAULT, h_prop_vec, H5P_DEFAULT);
        H5Dclose(h_data);

        /* Velocities (use vector space) */
        h_data = H5Dcreate(h_grp, "Velocities", H5T_NATIVE_DOUBLE, h_vspace, H5P_DEFAULT, h_prop_vec, H5P_DEFAULT);
        H5Dclose(h_data);

        /* Particle IDs (use scalar space) */
        h_data = H5Dcreate(h_grp, "ParticleIDs", H5T_NATIVE_LLONG, h_sspace, H5P_DEFAULT, h_prop_sca, H5P_DEFAULT);
        H5Dclose(h_data);

        /* Masses (use scalar space) */
        h_data = H5Dcreate(h_grp, "Masses", H5T_NATIVE_DOUBLE, h_sspace, H5P_DEFAULT, h_prop_sca, H5P_DEFAULT);
        H5Dclose(h_data);

        /* Neutrino (delta-f) weights (use scalar space) */
        h_data = H5Dcreate(h_grp, "Weights", H5T_NATIVE_DOUBLE, h_sspace, H5P_DEFAULT, h_prop_sca, H5P_DEFAULT);
        H5Dclose(h_data);
        
        /* Initial unperturbed phase-space densities (use scalar space) */
        h_data = H5Dcreate(h_grp, "PhaseSpaceDensities", H5T_NATIVE_DOUBLE, h_sspace, H5P_DEFAULT, h_prop_sca, H5P_DEFAULT);
        H5Dclose(h_data);

        /* Energies (use scalar space) */
        h_data = H5Dcreate(h_grp, "Energies", H5T_NATIVE_DOUBLE, h_sspace, H5P_DEFAULT, h_prop_sca, H5P_DEFAULT);
        H5Dclose(h_data);

        /* Close the group */
        H5Gclose(h_grp);

        /* Close the file */
        H5Fclose(h_out_file);
    }

    /* Ensure that all nodes are at the final time step */
    double a_min;
    MPI_Allreduce(&a, &a_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    message(verbosity_high, "Writing output to %s.\n", out_fname);

    /* Now open the file in parallel mode if possible */
#ifdef H5_HAVE_PARALLEL
    hid_t h_out_file;
    if (pars->DistributedFiles) {
        h_out_file = openFile(out_fname_local);
    } else {
        h_out_file = openFile_MPI(MPI_COMM_WORLD, out_fname_local);
    }
#else
    hid_t h_out_file = openFile(out_fname_local);
#endif

    /* The particle group in the output file */
    hid_t h_grp = H5Gopen(h_out_file, pars->ExportName, H5P_DEFAULT);

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
    const hsize_t ch_sdims[1] = {localParticleNumber};
    hid_t h_ch_vspace = H5Screate_simple(vrank, ch_vdims, NULL);
    hid_t h_ch_sspace = H5Screate_simple(srank, ch_sdims, NULL);

    /* The start of this chunk, in the overall vector & scalar spaces */
    const hsize_t start_in_group = pars->DistributedFiles ? 0 : localFirstNumber;
    const hsize_t vstart[2] = {start_in_group, 0}; //always with the "x" coordinate
    const hsize_t sstart[1] = {start_in_group};

    /* Choose the corresponding hyperslabs inside the overall spaces */
    H5Sselect_hyperslab(h_vspace, H5S_SELECT_SET, vstart, NULL, ch_vdims, NULL);
    H5Sselect_hyperslab(h_sspace, H5S_SELECT_SET, sstart, NULL, ch_sdims, NULL);

    /* Unpack the remaining particle data into contiguous arrays */
    double *coords = malloc(3 * localParticleNumber * sizeof(double));
    double *vels = malloc(3 * localParticleNumber * sizeof(double));
    long long *ids = malloc(1 * localParticleNumber * sizeof(long long));
    for (long long i=0; i<localParticleNumber; i++) {
        coords[i * 3 + 0] = genparts[i].x[0];
        coords[i * 3 + 1] = genparts[i].x[1];
        coords[i * 3 + 2] = genparts[i].x[2];
        vels[i * 3 + 0] = genparts[i].v[0];
        vels[i * 3 + 1] = genparts[i].v[1];
        vels[i * 3 + 2] = genparts[i].v[2];
        ids[i] = firstID + i;
    }

    message(verbosity_high, "Writing Coordinates.\n");

    /* Write coordinate data (vector) */
    h_data = H5Dopen(h_grp, "Coordinates", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_ch_vspace, h_vspace, H5P_DEFAULT, coords);
    H5Dclose(h_data);
    free(coords);

    message(verbosity_high, "Writing Velocities.\n");

    /* Write velocity data (vector) */
    h_data = H5Dopen(h_grp, "Velocities", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_ch_vspace, h_vspace, H5P_DEFAULT, vels);
    H5Dclose(h_data);
    free(vels);

    message(verbosity_high, "Writing ParticleIDs.\n");

    /* Write particle id data (scalar) */
    h_data = H5Dopen(h_grp, "ParticleIDs", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_LLONG, h_ch_sspace, h_sspace, H5P_DEFAULT, ids);
    H5Dclose(h_data);
    free(ids);

    message(verbosity_high, "Writing Masses.\n");

    /* Write mass data (scalar) */
    h_data = H5Dopen(h_grp, "Masses", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_ch_sspace, h_sspace, H5P_DEFAULT, masses);
    H5Dclose(h_data);
    free(masses);

    message(verbosity_high, "Writing Weights.\n");

    /* Write delta-f weight data (scalar) */
    h_data = H5Dopen(h_grp, "Weights", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_ch_sspace, h_sspace, H5P_DEFAULT, weights);
    H5Dclose(h_data);
    free(weights);

    message(verbosity_high, "Writing PhaseSpaceDensities.\n");

    /* Write initial unperturbed phase-space density data (scalar) */
    h_data = H5Dopen(h_grp, "PhaseSpaceDensities", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_ch_sspace, h_sspace, H5P_DEFAULT, phaseDensities);
    H5Dclose(h_data);
    free(phaseDensities);

    message(verbosity_high, "Writing Energies.\n");

    /* Write energy data (scalar) */
    h_data = H5Dopen(h_grp, "Energies", H5P_DEFAULT);
    H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_ch_sspace, h_sspace, H5P_DEFAULT, energies);
    H5Dclose(h_data);
    free(energies);

    message(verbosity_high, "\n");
    message(verbosity_high, "Done with writing on rank 0.\n");

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

    /* Clean up */
    cleanParams(pars);
    cleanPerturb(&ptdat);
    cleanPerturbParams(&ptpars);

    /* Release the interpolation splines */
    cleanPerturbSpline(&spline);
    
    return localParticleNumber;
}
