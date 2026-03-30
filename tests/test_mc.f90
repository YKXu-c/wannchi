!
! test_mc.f90 — Unit test for classical_mc.f90 module
!
! Runs MC on a 10x10 square Heisenberg lattice, J=1.0 eV, S=1.0.
! Temperature sweep: T = 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0 eV
!
! PASS criteria: m(T=0) > 0.9 and m(T=5) < 0.5
!
PROGRAM test_mc
  !
  use constants,     only : dp, stdout
  use classical_mc,  only : mc_lattice, mc_init, mc_finalize, &
                            mc_build_neighbors, mc_random_spin, &
                            mc_sweep, mc_thermalize, &
                            mc_measure_magnetization
  !
  implicit none
  !
  integer, parameter  :: N = 10       ! 10x10 lattice
  integer, parameter  :: N_THERM = 500
  integer, parameter  :: N_MEAS  = 500
  real(dp), parameter :: J_MC = 1.0_dp
  real(dp), parameter :: S_MAG = 1.0_dp
  !
  TYPE(mc_lattice) :: mc
  integer :: n_sites, iT, imeas, n_acc, ii
  real(dp) :: T_now, m_now
  real(dp), dimension(3) :: mvec_tmp, mvec_acc
  !
  ! Square lattice: 1 site per unit cell, a1=(1,0,0), a2=(0,1,0), a3=(0,0,10)
  real(dp), dimension(3,1) :: frac_pos
  real(dp), dimension(3,3) :: avec
  real(dp) :: cutoff
  !
  integer, parameter :: NTEMPS = 7
  real(dp) :: temps(NTEMPS)
  real(dp) :: mags(NTEMPS)
  logical  :: passed
  !
  write(stdout, '(A)') "============================================"
  write(stdout, '(A)') "TEST: Classical MC (square Heisenberg)"
  write(stdout, '(A)') "============================================"
  write(stdout, '(A,I3,A,I3,A,F6.2,A,F6.2)') &
        "  Lattice: ", N, "x", N, ", J=", J_MC, ", S=", S_MAG
  write(stdout, '(A)') ""
  !
  ! Setup square lattice
  frac_pos(:,1) = [0.0_dp, 0.0_dp, 0.0_dp]
  avec = 0.0_dp
  avec(1,1) = 1.0_dp  ! a1 = (1,0,0)
  avec(2,2) = 1.0_dp  ! a2 = (0,1,0)
  avec(3,3) = 10.0_dp ! a3 = (0,0,10)  (large z to make it 2D)
  cutoff = 1.5_dp  ! NN distance = 1.0, cutoff = 1.5
  !
  n_sites = N * N * 1 * 1  ! nx*ny*nz*n_uc_sites
  !
  temps = [0.0_dp, 0.5_dp, 1.0_dp, 1.5_dp, 2.0_dp, 3.0_dp, 5.0_dp]
  !
  write(stdout, '(A)') "  T (eV)    <|m|>"
  write(stdout, '(A)') "  ------    -----"
  !
  do iT = 1, NTEMPS
    T_now = temps(iT)
    !
    ! Initialize MC lattice fresh for each temperature
    call mc_init(mc, n_sites, J_MC, S_MAG)
    call mc_build_neighbors(mc, frac_pos, 1, avec, N, N, 1, cutoff)
    !
    if (T_now < 1.0d-12) then
      ! T=0: perfect order
      do ii = 1, n_sites
        mc%spin(:, ii) = [0.0_dp, 0.0_dp, 1.0_dp]
      enddo
      mags(iT) = 1.0_dp
    else
      ! Thermalize
      call mc_thermalize(mc, T_now, N_THERM)
      !
      ! Measure
      mvec_acc = 0.0_dp
      do imeas = 1, N_MEAS
        call mc_sweep(mc, T_now, n_acc)
        call mc_measure_magnetization(mc, mvec_tmp)
        mvec_acc = mvec_acc + mvec_tmp
      enddo
      mvec_acc = mvec_acc / real(N_MEAS, dp)
      mags(iT) = sqrt(sum(mvec_acc**2))
    endif
    !
    write(stdout, '(A,F6.2,A,F8.4)') "  ", T_now, "      ", mags(iT)
    !
    call mc_finalize(mc)
  enddo
  !
  write(stdout, '(A)') ""
  !
  passed = (mags(1) > 0.9_dp .and. mags(NTEMPS) < 0.5_dp)
  !
  if (passed) then
    write(stdout, '(A)') "  RESULT: PASS"
  else
    write(stdout, '(A,F8.4,A,F8.4)') "  RESULT: FAIL  m(0)=", mags(1), &
          "  m(5)=", mags(NTEMPS)
  endif
  !
END PROGRAM test_mc
