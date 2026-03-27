MODULE input
  !
  use constants
  !
  implicit none
  !
  character(len=80) seed
  ! SeedName
  !
  real(dp) mu, beta
  ! Fermi level and System temperature
  !
  integer nqpt
  ! Number of qpts
  !
  real(dp), dimension(:, :), allocatable :: qvec
  ! qvectors: qvec(3, iq)
  !
  integer nnu
  ! Number of frequencies to be calculated
  !
  real(dp) emin, emax
  ! (In case of spectrum calculation)
  ! emin, emax and nnu determines nu
  !
  real(dp) eps
  ! Input infinitesmal
  !
  complex(dp), dimension(:), allocatable :: nu
  ! In case of Matsubara calculation
  ! nu is determined by beta and nnu
  !
  logical spectra_calc
  ! Calculate on real axis? (spectra calculation)
  !
  logical trace_only
  ! Calculate only trace of Chi
  !
  logical ff_only
  ! Calculate only interaction part
  !
  logical use_lehman
  ! Use Lehman Representation To Calculate
  !
  logical fast_calc
  ! Use fast algorithm (more memory required)
  !
  integer npade
  ! Number of Poles in Pade Summation
  !
  namelist /CONTROL/ use_lehman, trace_only, ff_only, fast_calc, eps, nnu, emin, emax, npade
  namelist /SYSTEM/ spectra_calc, seed, beta, mu
  !
  ! --- wanneff_JS parameters (added for wanneff_JS module) ---
  character(len=80) :: seedbare = ''
  logical  :: eff_js = .false.
  logical  :: eff_mc = .false.
  real(dp) :: mc_temperature(3) = (/0.0_dp, 0.5_dp, 300.0_dp/) ! T_start, T_step, T_end (eV)
  logical  :: mc_weiss_mean_field = .true.
  real(dp) :: J_mc = 0.0_dp
  logical  :: J_TENSOR = .false.
  real(dp) :: tol_Jeff = 1.0d-2
  integer  :: J_R_range(6) = (/0, 0, 0, 0, 0, 0/) ! Rx_min,Rx_max,Ry_min,Ry_max,Rz_min,Rz_max
                                                    ! All zeros = use same R-grid as seedbare hr
  integer  :: bayes_niter = 50
  real(dp) :: J_bounds(2) = (/0.0_dp, 10.0_dp/)   ! J >= 0 breaks sign degeneracy with S
  real(dp) :: S_bounds(2) = (/-5.0_dp,  5.0_dp/)  ! each S_x, S_y, S_z component
  integer  :: mc_supercell(3) = (/0, 0, 0/)        ! MC supercell (0 = auto-detect from avec)
  real(dp) :: sigma_broadening = 0.05_dp           ! Lorentzian broadening for sigma_xx (eV)
  logical  :: berry_curvature_output = .false.     ! Write Berry curvature k-map for seed/bare/eff
  ! Optional explicit FF orbital indices (for wanneff_JS - CC = all other seed indices)
  ! Fixed size array - use n_ff_orbital_indices to know how many are actually used
  integer, parameter :: max_ff_indices = 100
  integer :: n_ff_orbital_indices = 0  ! count of actual FF indices
  integer, dimension(max_ff_indices) :: ff_orbital_indices = 0
  !
  namelist /EFFJS/ seedbare, eff_js, eff_mc, mc_temperature, &
                   mc_weiss_mean_field, J_mc,                 &
                   J_TENSOR, tol_Jeff, J_R_range,             &
                   bayes_niter, J_bounds, S_bounds, mc_supercell, &
                   sigma_broadening, berry_curvature_output, &
                   n_ff_orbital_indices, ff_orbital_indices
  !
CONTAINS
  !
 SUBROUTINE read_input(codename)
  !
  !
  use constants, only : dp, eps4, eps9, eps12, fin, fout
  use para,      only : inode, para_sync_int, para_sync_real
  !
  implicit none
  !
  character(*) codename
  !
  character(len=80) line
  !
  integer, dimension(8)  :: tt_int
  real(dp), dimension(5) :: tt_real
  integer ii
  !
  ! Example INPUT FILE:
  ! &SYSTEM
  !   seed='wannier90',
  !   beta=2000.d0,
  !   mu=0.d0,
  !   spectra_calc = .false.
  ! /
  ! &CONTROL
  !   use_lehman=.false.
  !   trace_only=.false.
  !   ff_only  = .true.
  !   fast_calc=.true.
  !   npade=80
  !   nnu=1
  !   ! eps=0.001  ! Only Used for Spectra
  !   ! emin=0.0   !
  !   ! emax=0.0
  ! /
  !
  ! DEFAULT:
  !
  seed='wannier90'
  ! Default seed name
  !
  beta=1e7
  ! Very low T calculation (~0K)
  !
  mu=0.d0
  ! Default Fermi level 
  !
  spectra_calc = .false.
  ! Calculate on imaginary frequency
  !
  use_lehman = .false.
  ! Use G*G algorithm
  !
  trace_only = .false.
  ! Only calculate trace
  !
  ff_only    = .true.
  ! Calculate only the interaction part
  !
  fast_calc = .true.
  ! Use more memory
  !
  npade=80
  ! 80 Pade Poles
  !
  nnu=1
  ! Calculate only single frequency
  !
  eps=eps6
  ! Default small imaginary part
  !
  emin=0.0
  ! Default at Ef
  !
  emax=0.0
  ! 
  ! logical spectra_calc, use_lehman, trace_only, ff_only, fast_calc
  ! real    eps, emin, emax, beta, mu
  ! integer nnu
  !
  if (inode.eq.0) then
    ! Read Structure Input
    !
    open(unit=fin, file=trim(codename)//".inp")
    !
    read(nml=SYSTEM, unit=fin)
    read(nml=CONTROL, unit=fin)
    !
    close(unit=fin)
    !
    tt_int(:)=0
    !
    if (spectra_calc) tt_int(1) = 1
    if (use_lehman)   tt_int(2) = 1
    if (trace_only)   tt_int(3) = 1
    if (ff_only)      tt_int(4) = 1
    if (fast_calc)    tt_int(5) = 1
    tt_int(6) = nnu
    tt_int(7) = npade
    !
    tt_real(:)=(/eps, emin, emax, beta, mu/);
    !
  endif
  !
  call para_sync_int(tt_int, 7)
  !
  spectra_calc = (tt_int(1).eq.1)
  use_lehman   = (tt_int(2).eq.1)
  trace_only   = (tt_int(3).eq.1)
  ff_only      = (tt_int(4).eq.1)
  fast_calc    = (tt_int(5).eq.1)
  nnu          =  tt_int(6)
  npade        =  tt_int(7)
  !
  call para_sync_real(tt_real, 5)
  eps  = tt_real(1)
  emin = tt_real(2)
  emax = tt_real(3)
  beta = tt_real(4)
  mu   = tt_real(5)
  !
  if (beta<0) beta=1.d7
  if (eps>eps4.or.eps<eps12) eps=eps9
  !
  allocate(nu(nnu))
  !
  if (spectra_calc) then
    !
    if (nnu>1) then
      do ii=1, nnu
        nu(ii)=emin+(emax-emin)*(ii-1)/(nnu-1)
      enddo
    else
      nu(1)=emin
    endif
    !
    nu(:)=nu(:)+eps*cmplx_i
    !
  else
    !
    do ii=1, nnu
      nu(ii)=twopi*(ii-1)*cmplx_i/beta
    enddo
    !
  endif
  !
 END SUBROUTINE
  !
 SUBROUTINE read_qpoints()
  !
  use constants, only : dp, fin
  use para,      only : inode, para_sync_int, para_sync_real
  !
  implicit none
  !
  integer mode
  integer nq1, nq2, nq3
  integer iq1, iq2, iq3
  real(dp), dimension(3) :: tq1, tq2, tq0
  integer, dimension(4) :: tt
  !
  if (inode.eq.0) then
    !
    open(unit=fin, file="QPOINTS")
    !
    ! QPOINTS Example 1
    !   0           ! Single Point Calculation
    !  0.5 0.5 0.5  ! Qvec
    !
    ! QPOINTS Example 2
    !   1           ! Line mode
    !   1  48       !  nseg   ninterpolate
    ! 0.0  0.0  0.0    0.5  0.0  0.0  ! seg 1: Q1  Q2
    !
    ! QPOINTS Example 3
    !   2           ! Plane mode
    !  48  48       ! nint1  nint2
    ! 0.0  0.0  0.0 ! Vertex
    ! 1.0  0.0  0.0 ! Direction 1
    ! 0.0  1.0  0.0 ! Direction 2
    !
    ! QPOINTS Example 4
    !   3           ! Full BZ
    !  48  48  48   ! nint1  nint2  nint3
    !
    read(fin, *) mode
    !
    if (mode.eq.0) then
      nqpt=1
      nq1=1
      nq2=1
      nq3=1
    elseif (mode.eq.1) then
      nq3=1
      read(fin, *) nq1, nq2
      nqpt=nq1*(nq2+1)
    elseif (mode.eq.2) then
      nq3=1
      read(fin, *) nq1, nq2
      nqpt=(nq1+1)*(nq2+1)
    elseif (mode.eq.3) then
      read(fin, *) nq1, nq2, nq3
      nqpt=nq1*nq2*nq3
    endif
    !
    tt(1)=nq1
    tt(2)=nq2
    tt(3)=nq3
    tt(4)=nqpt 
    !
  endif
  !
  call para_sync_int(tt, 4)
  !
  nq1=tt(1)
  nq2=tt(2)
  nq3=tt(3)
  nqpt=tt(4)
  !
  allocate(qvec(3, nqpt))
  !
  if (inode.eq.0) then
    !
    if (mode.eq.0) then
      read(fin, *) qvec(:, 1)
    elseif (mode.eq.1) then
      do iq1=0, nq1-1
        read(fin, *) tq1(:), tq2(:)
        do iq2=0, nq2
          qvec(:, iq1*(nq2+1)+iq2+1)=tq1(:) + (iq2*1.d0)/nq2*(tq2(:)-tq1(:))
        enddo
      enddo
    elseif (mode.eq.2) then
      read(fin, *) tq0
      read(fin, *) tq1
      read(fin, *) tq2
      !
      do iq1=0, nq1
        do iq2=0, nq2
          qvec(:, iq1*(nq2+1)+iq2+1)=tq0 + (iq1*1.d0)/nq1*tq1 + (iq2*1.d0)/nq2*tq2
        enddo
      enddo
    elseif (mode.eq.3) then
      do iq1=0, nq1-1
        do iq2=0, nq2-1
          do iq3=0, nq3-1
            qvec(1, iq1*nq2*nq3+iq2*nq3+iq3+1) = iq1*1.d0/nq1
            qvec(2, iq1*nq2*nq3+iq2*nq3+iq3+1) = iq2*1.d0/nq2
            qvec(3, iq1*nq2*nq3+iq2*nq3+iq3+1) = iq3*1.d0/nq3
          enddo
        enddo
      enddo
    endif
    !
    close(fin)
    !
  endif
  !
  call para_sync_real(qvec, nqpt*3)
  !
 END SUBROUTINE

  !
 SUBROUTINE finalize_input
  !
  implicit none
  !
  if (allocated(nu))   deallocate(nu)
  if (allocated(qvec)) deallocate(qvec)
  !
 END SUBROUTINE
  !
  ! --- Added for wanneff_JS: read &EFFJS namelist ---
 SUBROUTINE read_effjs_input(codename)
  !
  ! Read &EFFJS namelist from {codename}.inp (e.g. 'wanneff').
  ! Also re-reads &SYSTEM and &CONTROL for convenience.
  ! Call AFTER read_input to overwrite with wanneff-specific values.
  !
  use constants, only : fin, dp
  use para,      only : inode, para_sync_int, para_sync_real, para_sync0
  !
  implicit none
  !
  character(*), intent(in) :: codename
  !
  integer, dimension(12) :: tt_int
  real(dp), dimension(13) :: tt_real
  integer :: ii
  !
  if (inode .eq. 0) then
    open(unit=fin, file=trim(codename)//'.inp')
    read(nml=SYSTEM,  unit=fin)
    read(nml=CONTROL, unit=fin)
    read(nml=EFFJS,   unit=fin)
    close(unit=fin)
    !
    tt_int(1)  = merge(1, 0, eff_js)
    tt_int(2)  = merge(1, 0, eff_mc)
    tt_int(3)  = merge(1, 0, mc_weiss_mean_field)
    tt_int(4)  = merge(1, 0, J_TENSOR)
    tt_int(5)  = bayes_niter
    tt_int(6:11) = J_R_range(1:6)
    tt_int(12) = nnu
    tt_real(1:3) = mc_temperature(1:3)
    tt_real(4)   = J_mc
    tt_real(5)   = tol_Jeff
    tt_real(6:7) = J_bounds(1:2)
    tt_real(8:9) = S_bounds(1:2)
    tt_real(10)  = mu
    tt_real(11)  = beta
    tt_real(12)  = emin
    tt_real(13)  = emax
  endif
  !
  call para_sync_int(tt_int,  12)
  call para_sync_real(tt_real, 13)
  !
  eff_js             = (tt_int(1) == 1)
  eff_mc             = (tt_int(2) == 1)
  mc_weiss_mean_field= (tt_int(3) == 1)
  J_TENSOR           = (tt_int(4) == 1)
  bayes_niter        =  tt_int(5)
  J_R_range(1:6)     =  tt_int(6:11)
  nnu                =  tt_int(12)
  mc_temperature(1:3)=  tt_real(1:3)
  J_mc               =  tt_real(4)
  tol_Jeff           =  tt_real(5)
  J_bounds(1:2)      =  tt_real(6:7)
  S_bounds(1:2)      =  tt_real(8:9)
  mu                 =  tt_real(10)
  beta               =  tt_real(11)
  emin               =  tt_real(12)
  emax               =  tt_real(13)
  !
  ! Sync character seedbare via integer array of ASCII codes
  call para_sync_character(seedbare)
  !
 END SUBROUTINE read_effjs_input
  !
  ! --- Helper: broadcast character variable ---
 SUBROUTINE para_sync_character(str)
  !
  use para, only : inode, para_sync_int
  !
  implicit none
  !
  character(len=80), intent(inout) :: str
  !
  integer, dimension(80) :: codes
  integer :: ii
  !
  if (inode .eq. 0) then
    do ii = 1, 80
      codes(ii) = ichar(str(ii:ii))
    enddo
  endif
  call para_sync_int(codes, 80)
  if (inode .ne. 0) then
    do ii = 1, 80
      str(ii:ii) = char(codes(ii))
    enddo
  endif
  !
 END SUBROUTINE para_sync_character
  !
END MODULE
