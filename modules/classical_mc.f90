!
!   classical_mc.f90
!
! ===========================================================================
! MODULE classical_mc
!   Classical Heisenberg Monte Carlo simulation on the f-site sublattice.
!
!   Model: Classical spin Hamiltonian
!     E = -J_mc * sum_{<i,j>} S_i . S_j,   |S_i| = S_mag (fixed magnitude)
!
!   Algorithm: Metropolis single-spin updates.
!     P_accept = min(1, exp(-dE / (k_B * T)))
!   New spin proposed by Marsaglia method (uniform on unit sphere).
!
!   Weiss mean-field formula for J_mc (mc_weiss_mean_field=.true.):
!     J_mc = J_eff * S_mag / z
!   where z = coordination number from neighbor list.
!   This gives mean-field Curie temperature: T_c ~ J_eff * S_mag^2 / 3
!
!   Output: mvec_vs_T(3, n_temps) — thermal average of m_vec = (1/N)*sum_i S_i/S_mag
!   (fractional vector magnetization, |mvec| in [0,1])
!
!   Caller computes: S_eff_vec(:, iT) = S_mag * mvec_vs_T(:, iT)
!
!   [Ref: Janke, Monte Carlo Simulations of Spin Systems, Springer (1996)]
!   [Ref: Metropolis et al., J. Chem. Phys. 21, 1087 (1953)]
!
!   MPI parallelization:
!     Temperature loop is distributed via distribute_calc(n_temps).
!     Works with para_serial.f90 (serial stub) on laptop builds.
!
! ===========================================================================
MODULE classical_mc
  !
  use constants, only : dp
  use para,      only : distribute_calc, first_idx, last_idx, para_merge_real, &
                        inode, para_barrier
  !
  implicit none
  !
  TYPE mc_lattice
    integer :: n_sites          ! total number of spins in supercell
    integer :: n_neighbors_max  ! max neighbors per site
    integer, allocatable :: neighbor_list(:,:)  ! (n_neighbors_max, n_sites)
    integer, allocatable :: n_nn(:)             ! actual neighbor count per site
    real(dp), allocatable :: spin(:,:)          ! (3, n_sites) unit spin vectors
    real(dp) :: J_mc   ! exchange coupling (>0: ferromagnetic)
    real(dp) :: S_mag  ! spin magnitude |S|
  END TYPE mc_lattice
  !
CONTAINS
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE mc_init(mc, n_sites, J_mc_in, S_mag_in)
    !
    ! Allocate mc_lattice and initialize spins randomly.
    !
    TYPE(mc_lattice), intent(out) :: mc
    integer,  intent(in) :: n_sites
    real(dp), intent(in) :: J_mc_in, S_mag_in
    !
    integer :: ii
    !
    mc%n_sites         = n_sites
    mc%n_neighbors_max = 12  ! enough for FCC/HCP; will be trimmed by n_nn
    mc%J_mc            = J_mc_in
    mc%S_mag           = S_mag_in
    !
    allocate(mc%spin(3, n_sites))
    allocate(mc%neighbor_list(mc%n_neighbors_max, n_sites))
    allocate(mc%n_nn(n_sites))
    !
    mc%neighbor_list = 0
    mc%n_nn          = 0
    !
    ! Random initial spins on unit sphere
    do ii = 1, n_sites
      call mc_random_spin(mc%spin(:, ii))
    enddo
    !
  END SUBROUTINE mc_init
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE mc_finalize(mc)
    !
    TYPE(mc_lattice), intent(inout) :: mc
    !
    if (allocated(mc%spin))          deallocate(mc%spin)
    if (allocated(mc%neighbor_list)) deallocate(mc%neighbor_list)
    if (allocated(mc%n_nn))          deallocate(mc%n_nn)
    !
  END SUBROUTINE mc_finalize
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE mc_random_spin(spin)
    !
    ! Generate a uniformly random unit vector on S^2 using the Marsaglia method.
    ! Marsaglia (1972): pick (u,v) uniform in unit disk, then
    !   s = u^2 + v^2
    !   spin = (2u*sqrt(1-s), 2v*sqrt(1-s), 1-2s)
    !
    real(dp), dimension(3), intent(out) :: spin
    !
    real(dp) :: u, v, s
    !
    do
      call random_number(u); u = 2.0_dp*u - 1.0_dp
      call random_number(v); v = 2.0_dp*v - 1.0_dp
      s = u*u + v*v
      if (s < 1.0_dp) exit
    enddo
    !
    spin(1) = 2.0_dp * u * sqrt(1.0_dp - s)
    spin(2) = 2.0_dp * v * sqrt(1.0_dp - s)
    spin(3) = 1.0_dp - 2.0_dp * s
    !
  END SUBROUTINE mc_random_spin
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE mc_build_neighbors(mc, frac_pos, n_uc_sites, avec, nx, ny, nz, cutoff)
    !
    ! Build periodic neighbor list for a supercell of nx x ny x nz unit cells,
    ! each containing n_uc_sites f-sites at fractional positions frac_pos(3, n_uc_sites).
    !
    ! Site indexing: site (ix, iy, iz, isite) -> global index
    !   idx = (iz*ny*nx + iy*nx + ix) * n_uc_sites + isite
    !
    TYPE(mc_lattice), intent(inout) :: mc
    integer, intent(in)  :: n_uc_sites, nx, ny, nz
    real(dp), dimension(3, n_uc_sites), intent(in) :: frac_pos
    real(dp), dimension(3, 3), intent(in)  :: avec  ! lattice vectors (columns)
    real(dp), intent(in) :: cutoff  ! distance cutoff in Angstrom
    !
    integer :: ia, ib, ic, is, ja, jb, jc, js, idx_i, idx_j
    integer :: da, db, dc
    real(dp), dimension(3) :: ri_cart, rj_cart, diff
    real(dp) :: dist
    !
    mc%n_nn(:) = 0
    !
    do ic = 0, nz-1
      do ib = 0, ny-1
        do ia = 0, nx-1
          do is = 1, n_uc_sites
            !
            idx_i = ((ic*ny + ib)*nx + ia) * n_uc_sites + is
            ! Cartesian position of site i
            ri_cart = matmul(avec, frac_pos(:,is) + [real(ia,dp), real(ib,dp), real(ic,dp)])
            !
            ! Loop over neighboring unit cells (±2 shell in each direction)
            do dc = -2, 2
              do db = -2, 2
                do da = -2, 2
                  do js = 1, n_uc_sites
                    !
                    ! PBC unit cell indices
                    jc = mod(ic + dc + 2*nz, nz)
                    jb = mod(ib + db + 2*ny, ny)
                    ja = mod(ia + da + 2*nx, nx)
                    !
                    idx_j = ((jc*ny + jb)*nx + ja) * n_uc_sites + js
                    !
                    if (idx_j == idx_i) cycle  ! skip self
                    !
                    rj_cart = matmul(avec, frac_pos(:,js) + &
                              [real(ja,dp), real(jb,dp), real(jc,dp)])
                    diff = ri_cart - rj_cart
                    dist = sqrt(sum(diff**2))
                    !
                    if (dist < cutoff + 1.0d-6) then
                      if (mc%n_nn(idx_i) < mc%n_neighbors_max) then
                        mc%n_nn(idx_i) = mc%n_nn(idx_i) + 1
                        mc%neighbor_list(mc%n_nn(idx_i), idx_i) = idx_j
                      endif
                    endif
                    !
                  enddo
                enddo
              enddo
            enddo
            !
          enddo
        enddo
      enddo
    enddo
    !
  END SUBROUTINE mc_build_neighbors
  !
  ! ---------------------------------------------------------------------------
  FUNCTION mc_local_energy(mc, isite) RESULT(E)
    !
    ! Local energy of site isite:
    !   E = -J_mc * S_mag^2 * sum_{j in NN(i)} S_i . S_j
    !
    TYPE(mc_lattice), intent(in) :: mc
    integer, intent(in) :: isite
    real(dp) :: E
    !
    integer :: jj, jsite
    real(dp) :: sdot
    !
    sdot = 0.0_dp
    do jj = 1, mc%n_nn(isite)
      jsite = mc%neighbor_list(jj, isite)
      sdot  = sdot + dot_product(mc%spin(:, isite), mc%spin(:, jsite))
    enddo
    E = -mc%J_mc * mc%S_mag * mc%S_mag * sdot
    !
  END FUNCTION mc_local_energy
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE mc_sweep(mc, temperature, n_accepted)
    !
    ! Single-spin Metropolis sweep: n_sites attempted updates.
    ! temperature in eV.
    !
    TYPE(mc_lattice), intent(inout) :: mc
    real(dp), intent(in)  :: temperature
    integer,  intent(out) :: n_accepted
    !
    integer :: ii, isite, jj, jsite
    real(dp) :: dE, u
    real(dp), dimension(3) :: spin_old, spin_new
    real(dp) :: sdot_old, sdot_new
    !
    n_accepted = 0
    !
    do ii = 1, mc%n_sites
      !
      call random_number(u)
      isite = int(u * mc%n_sites) + 1
      if (isite > mc%n_sites) isite = mc%n_sites
      !
      spin_old = mc%spin(:, isite)
      call mc_random_spin(spin_new)
      !
      ! Compute dE from neighbor interactions only:
      ! dE = -J*S^2 * sum_{j in NN} (S_new - S_old) . S_j
      dE = 0.0_dp
      do jj = 1, mc%n_nn(isite)
        jsite = mc%neighbor_list(jj, isite)
        sdot_old = dot_product(spin_old, mc%spin(:, jsite))
        sdot_new = dot_product(spin_new, mc%spin(:, jsite))
        dE = dE - mc%J_mc * mc%S_mag * mc%S_mag * (sdot_new - sdot_old)
      enddo
      !
      ! Metropolis acceptance
      if (dE <= 0.0_dp) then
        mc%spin(:, isite) = spin_new
        n_accepted = n_accepted + 1
      else if (temperature > 1.0d-12) then
        call random_number(u)
        if (u < exp(-dE / temperature)) then
          mc%spin(:, isite) = spin_new
          n_accepted = n_accepted + 1
        endif
      endif
      !
    enddo
    !
  END SUBROUTINE mc_sweep
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE mc_thermalize(mc, temperature, n_therm)
    !
    TYPE(mc_lattice), intent(inout) :: mc
    real(dp), intent(in) :: temperature
    integer,  intent(in) :: n_therm
    !
    integer :: ii, n_acc
    !
    do ii = 1, n_therm
      call mc_sweep(mc, temperature, n_acc)
    enddo
    !
  END SUBROUTINE mc_thermalize
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE mc_measure_magnetization(mc, mvec_out)
    !
    ! Compute average spin direction: mvec = (1/N) * sum_i spin(:,i)
    ! Returns the 3-component fractional magnetization vector.
    ! |mvec| is in [0,1]; S_eff = S_mag * mvec
    !
    TYPE(mc_lattice), intent(in) :: mc
    real(dp), dimension(3), intent(out) :: mvec_out
    !
    integer :: ii
    !
    mvec_out = 0.0_dp
    do ii = 1, mc%n_sites
      mvec_out = mvec_out + mc%spin(:, ii)
    enddo
    mvec_out = mvec_out / real(mc%n_sites, dp)
    !
  END SUBROUTINE mc_measure_magnetization
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE classical_mc_run(J_mc_in, S_mag_in, frac_pos, n_f_sites, avec, &
                               T_start, T_step, T_end, mvec_vs_T, n_temps, &
                               mc_supercell_in)
    !
    ! Main MC driver: temperature sweep from T_start to T_end with step T_step.
    !
    ! Inputs:
    !   J_mc_in    : exchange coupling between f-sites (eV)
    !   S_mag_in   : local spin magnitude |S|
    !   frac_pos   : (3, n_f_sites) fractional coordinates of f-sites in unit cell
    !   n_f_sites  : number of f-sites per unit cell
    !   avec       : (3,3) lattice vectors (columns = a1,a2,a3)
    !   T_start, T_step, T_end : temperature range in eV
    !   mc_supercell_in(3) : (NX, NY, NZ) supercell. 0 = auto-detect from avec:
    !     if |a_i| > 2 * min(|a_j|, |a_k|), it is assumed vacuum → N_i = 1
    !     otherwise N_i = 10 (standard thermodynamic limit for 2D/3D).
    !
    ! Output:
    !   mvec_vs_T(3, n_temps) : fractional magnetization vector at each temperature
    !   n_temps               : number of temperature points
    !
    use constants, only : stdout
    !
    real(dp), intent(in)  :: J_mc_in, S_mag_in
    integer,  intent(in)  :: n_f_sites
    real(dp), dimension(3, n_f_sites), intent(in) :: frac_pos
    real(dp), dimension(3, 3), intent(in) :: avec
    real(dp), intent(in)  :: T_start, T_step, T_end
    real(dp), allocatable, intent(out) :: mvec_vs_T(:,:)
    integer,  intent(out) :: n_temps
    integer,  dimension(3), intent(in) :: mc_supercell_in  ! (NX,NY,NZ), 0=auto
    !
    integer, parameter :: N_DEFAULT = 10  ! default supercell per active direction
    integer, parameter :: N_THERM = 5000, N_MEAS = 10000, MEAS_EVERY = 10
    !
    TYPE(mc_lattice) :: mc
    integer :: n_total_sites, iT, imeas, n_acc, ii
    integer :: NX, NY, NZ
    real(dp) :: T_now, cutoff
    real(dp), dimension(3) :: mvec_tmp, mvec_acc
    real(dp) :: dist_nn, len_a1, len_a2, len_a3, len_min, u
    real(dp), dimension(3) :: r1_cart, r2_cart
    real(dp), allocatable :: mvec_local(:,:)
    integer :: n_local_temps, iT_start, iT_end
    !
    ! Determine supercell dimensions: user override or auto-detect from lattice
    len_a1 = sqrt(sum(avec(:,1)**2))
    len_a2 = sqrt(sum(avec(:,2)**2))
    len_a3 = sqrt(sum(avec(:,3)**2))
    !
    if (mc_supercell_in(1) > 0) then
      NX = mc_supercell_in(1)
    else
      len_min = min(len_a2, len_a3)
      NX = merge(1, N_DEFAULT, len_a1 > 2.0_dp * len_min)
    endif
    if (mc_supercell_in(2) > 0) then
      NY = mc_supercell_in(2)
    else
      len_min = min(len_a1, len_a3)
      NY = merge(1, N_DEFAULT, len_a2 > 2.0_dp * len_min)
    endif
    if (mc_supercell_in(3) > 0) then
      NZ = mc_supercell_in(3)
    else
      len_min = min(len_a1, len_a2)
      NZ = merge(1, N_DEFAULT, len_a3 > 2.0_dp * len_min)
    endif
    !
    ! Determine number of temperature points
    if (T_step < 1.0d-12) then
      ! Single temperature point
      n_temps = 1
    else
      n_temps = int((T_end - T_start) / T_step) + 1
    endif
    if (n_temps < 1) n_temps = 1
    !
    allocate(mvec_vs_T(3, n_temps))
    !
    n_total_sites = NX * NY * NZ * n_f_sites
    call mc_init(mc, n_total_sites, J_mc_in, S_mag_in)
    !
    ! Estimate nearest-neighbor distance from first two distinct f-sites
    ! (or same site in neighboring unit cells if n_f_sites=1)
    if (n_f_sites > 1) then
      r1_cart = matmul(avec, frac_pos(:,1))
      r2_cart = matmul(avec, frac_pos(:,2))
      dist_nn = sqrt(sum((r1_cart - r2_cart)**2))
    else
      ! Distance to periodic image in a1 direction
      r1_cart = matmul(avec, frac_pos(:,1))
      r2_cart = matmul(avec, frac_pos(:,1) + [1.0_dp, 0.0_dp, 0.0_dp])
      dist_nn = sqrt(sum((r1_cart - r2_cart)**2))
    endif
    cutoff = dist_nn * 1.5_dp  ! 1.5x nearest-neighbor = NN only
    !
    call mc_build_neighbors(mc, frac_pos, n_f_sites, avec, NX, NY, NZ, cutoff)
    !
    write(stdout, '(A,1I8,A,1I3,A,1F8.4,A)') &
          "  # MC: ", n_total_sites, " spins, z=", mc%n_nn(1), &
          ", cutoff=", cutoff, " Ang"
    write(stdout, '(A,3I4,A)') "  # MC supercell: ", NX, NY, NZ, " (auto or user)"
    !
    ! Distribute temperature points across MPI ranks
    call distribute_calc(n_temps)
    iT_start = first_idx
    iT_end = last_idx
    n_local_temps = last_idx - first_idx + 1
    !
    ! Allocate local result array
    allocate(mvec_local(3, n_local_temps))
    mvec_local = 0.0_dp
    !
    do iT = iT_start, iT_end
      !
      if (n_temps == 1) then
        T_now = T_start
      else
        T_now = T_start + real(iT-1, dp) * T_step
      endif
      !
      if (T_now < 1.0d-12) then
        ! T=0: perfect ferromagnetic order along z
        do ii = 1, n_total_sites
          mc%spin(:, ii) = [0.0_dp, 0.0_dp, 1.0_dp]
        enddo
        mvec_local(:, iT - iT_start + 1) = [0.0_dp, 0.0_dp, 1.0_dp]
        cycle
      endif
      !
      ! Thermalize
      call mc_thermalize(mc, T_now, N_THERM)
      !
      ! Measure
      mvec_acc = 0.0_dp
      do imeas = 1, N_MEAS
      call mc_sweep(mc, T_now, n_acc)
        if (mod(imeas, MEAS_EVERY) == 0) then
          call mc_measure_magnetization(mc, mvec_tmp)
          mvec_acc = mvec_acc + mvec_tmp
        endif
      enddo
      mvec_local(:, iT - iT_start + 1) = mvec_acc / real(N_MEAS/MEAS_EVERY, dp)
      !
      if (inode == 0 .and. (mod(iT, 20) == 1 .or. iT == n_temps)) then
        write(stdout, '(A,1F8.3,A,1F8.4)') &
              "    MC T=", T_now, " eV   |<S>|/S =", &
              sqrt(sum(mvec_local(:,iT - iT_start + 1)**2))
      endif
      !
    enddo
    !
    ! Gather results to all ranks via para_merge_real
    call para_merge_real(mvec_local, 3 * n_local_temps)
    !
    ! Master (inode=0) copies local results to full array
    if (inode == 0) then
      do iT = 1, n_temps
        mvec_vs_T(:, iT) = mvec_local(:, iT)
      enddo
    endif
    call para_barrier()
    !
    ! Broadcast full result to all ranks
    call para_merge_real(mvec_vs_T, 3 * n_temps)
    !
    deallocate(mvec_local)
    call mc_finalize(mc)
    !
  END SUBROUTINE classical_mc_run
  !
END MODULE classical_mc
