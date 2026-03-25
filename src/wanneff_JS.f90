!
!   wanneff_JS.f90
!
! ===========================================================================
! MODULE wanneff_js_mod + PROGRAM WannEffJS
!
!   Extract effective J-S Kondo exchange coupling from Wannier Hamiltonians.
!
!   Physical model:
!     H_seed = H_CC + H_FF + H_CF   (full system with f-electrons)
!     H_eff_CC(k) = -[G(k,0)_CC]^{-1}  (downfolded CC Hamiltonian at omega=0)
!       where G_full(k,0) = (0 - H_seed(k))^{-1}
!     H_bare(k)   = Wannier HR of the seedbare (conduction-only) system
!
!   Exchange term (S = 3-component local moment vector):
!     H_JS(k) = (sum_R J(R)*exp(ik.R)) * (S.sigma)/2
!     H_spin = J(k)/2 * [[S_z, S_x-i*S_y], [S_x+i*S_y, -S_z]] (x) I_{n_c}
!
!   Objective (eigenvalue L2 norm):
!     L(J,S) = (1/N_k) sum_k ||sort(eig(H_bare(k)+H_JS(k))) - sort(eig(H_eff_CC(k)))||^2
!
!   Scalar mode (J_TENSOR=.false.): Bayesian optimize (J_0, S_x, S_y, S_z) -- 4 params
!   Tensor mode (J_TENSOR=.true.):  Bayesian optimize (J(R_1),...,J(R_n), S_x, S_y, S_z)
!                                    -- (n_jrpt+3) params
!   After Bayesian: apply tol_Jeff (prune small J(R) to 0).
!
!   Outputs:
!     {seedbare}_hr_0K.dat  -- effective HR at T=0
!     {seedbare}_hr_{T}K.dat -- at each MC temperature
!
!   [Ref: Kotliar et al., Rev. Mod. Phys. 78, 865 (2006) -- downfolding]
!   [Ref: Coleman, Introduction to Many-Body Physics (2015) -- Kondo lattice]
!   [Ref: Korotin, Mazurenko et al. -- exchange coupling from Wannier]
! ===========================================================================

MODULE wanneff_js_mod
  !
  use constants,  only : dp, cmplx_0, cmplx_i, stdout, fin, eps6
  use wanndata,   only : wannham, calc_hk, finalize_wann
  use linalgwrap,  only : invmat, eigen
  !
  implicit none
  !
  ! Module-level shared data for Bayesian callback
  integer :: g_norb_bare, g_norb_cc, g_nkirr, g_n_jrpt
  complex(dp), allocatable :: g_hk_eff_cc(:,:,:)   ! (norb_cc, norb_cc, nkirr)
  TYPE(wannham), pointer :: g_ham_bare => null()
  real(dp), allocatable :: g_kvec(:,:)              ! (3, nkirr)
  real(dp), allocatable :: g_rvec_J(:,:)            ! (3, n_jrpt) R-vectors for J
  logical :: g_J_TENSOR
  !
CONTAINS
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE downfold_to_cc(hk_cc, hk_full, norb_full, norb_cc, cc_idx)
    !
    ! Static downfolding: extract effective CC Hamiltonian from full H(k) at omega=0.
    !
    ! G_full(k,0) = -H_full(k)^{-1}      [omega=0, H already shifted to E_F=0]
    ! G_CC = G_full[CC block]
    ! H_eff_CC = -G_CC^{-1}
    !
    integer, intent(in) :: norb_full, norb_cc
    integer, dimension(norb_cc), intent(in) :: cc_idx
    complex(dp), dimension(norb_full, norb_full), intent(in) :: hk_full
    complex(dp), dimension(norb_cc, norb_cc), intent(out) :: hk_cc
    !
    complex(dp), dimension(norb_full, norb_full) :: Gfull
    complex(dp), dimension(norb_cc, norb_cc) :: Gcc
    integer :: ii, jj
    !
    ! G = -H^{-1}
    Gfull = -hk_full
    call invmat(Gfull, norb_full)
    !
    ! Extract CC block
    do ii = 1, norb_cc
      do jj = 1, norb_cc
        Gcc(ii, jj) = Gfull(cc_idx(ii), cc_idx(jj))
      enddo
    enddo
    !
    ! H_eff_CC = -G_CC^{-1}
    hk_cc = -Gcc
    call invmat(hk_cc, norb_cc)
    hk_cc = -hk_cc
    !
  END SUBROUTINE downfold_to_cc
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE determine_cc_indices(cc_idx, norb_full, norb_bare, &
                                   nbasis_full, nbasis_bare, nsite_full, nsite_bare)
    !
    ! Determine which orbital indices (1..norb_full) are CC orbitals.
    ! Strategy: compare site nbasis. Sites in seedbare match sites in seed
    ! with the same orbital count; any extra in seed are FF orbitals.
    !
    ! ASSUMPTIONS (must hold for correct results):
    !   1. Site ordering in seed.pos and seedbare.pos must match (site i in seed
    !      corresponds to site i in seedbare). The walk pairs sites sequentially.
    !   2. At each shared site, CC orbitals come FIRST (lower Wannier90 band index),
    !      and extra (FF) orbitals come after (indices > nbasis_bare(i)).
    !      This depends on the projection block order in the Wannier90 input.
    !   3. Same nbasis count at a site implies same orbital character.
    !      Cases where count matches but character differs (e.g., seed has d+f,
    !      seedbare has d+s) will silently mislabel orbitals.
    ! If these assumptions may not hold, use the 'cc_orbital_indices' override
    ! in &EFFJS namelist (future feature) to specify CC indices explicitly.
    !
    ! For spinor: CC orbitals = first n_c orbital indices for spin-up, then n_c for spin-down.
    ! The FF orbitals are those present in seed but not seedbare (by site/orbital count).
    !
    integer, intent(in) :: norb_full, norb_bare, nsite_full, nsite_bare
    integer, dimension(nsite_full), intent(in) :: nbasis_full
    integer, dimension(nsite_bare), intent(in) :: nbasis_bare
    integer, dimension(norb_bare), intent(out) :: cc_idx
    !
    integer :: ii, jj, g_orb, cc_count
    logical, dimension(norb_full) :: is_cc
    !
    ! Mark which global orbitals are CC (conduction) by comparing site-by-site
    ! For spinor: norb = 2 * n_spatial, first half spin-up, second half spin-down
    ! norb_full = 2 * n_spatial_full, norb_bare = 2 * n_spatial_bare
    !
    ! Simple strategy: CC orbitals are those NOT associated with f-sites.
    ! The f-site orbitals are the EXTRA orbitals in seed (not in seedbare).
    ! We find them by walking through sites: if a site has more orbitals in
    ! seed than in seedbare, the extras are FF.
    !
    is_cc(:) = .true.
    g_orb = 0
    do ii = 1, min(nsite_full, nsite_bare)
      ! For this site, mark all as CC (up to seedbare nbasis count)
      ! Extra orbitals (beyond seedbare) are FF
      do jj = 1, nbasis_full(ii)
        g_orb = g_orb + 1
        if (jj > nbasis_bare(ii)) then
          is_cc(g_orb) = .false.  ! FF orbital
        endif
      enddo
    enddo
    ! Sites in seed not in seedbare -> all FF
    do ii = min(nsite_full, nsite_bare)+1, nsite_full
      do jj = 1, nbasis_full(ii)
        g_orb = g_orb + 1
        is_cc(g_orb) = .false.
      enddo
    enddo
    !
    ! For spinor: also mark the spin-down FF orbitals
    ! The spin-down block starts at norb_full/2 + 1
    if (norb_full > g_orb) then
      do ii = 1, g_orb
        is_cc(g_orb + ii) = is_cc(ii)
      enddo
    endif
    !
    ! Collect CC indices
    cc_count = 0
    do ii = 1, norb_full
      if (is_cc(ii)) then
        cc_count = cc_count + 1
        if (cc_count <= norb_bare) cc_idx(cc_count) = ii
      endif
    enddo
    !
  END SUBROUTINE determine_cc_indices
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE add_js_coupling(ham_out, norb_bare, J_val, Svec, irpt)
    !
    ! Add J*(S.sigma)/2 exchange coupling to ham_out%hr at R-point irpt.
    !
    ! For n_c = norb_bare/2 spatial orbitals (spinor ordering):
    !   orbitals 1..n_c     : spin-up
    !   orbitals n_c+1..norb: spin-down
    !
    ! Exchange matrix (per orbital io):
    !   hr(io,     io,     irpt) += J * S_z / 2
    !   hr(io+n_c, io+n_c, irpt) -= J * S_z / 2
    !   hr(io,     io+n_c, irpt) += J * (S_x - i*S_y) / 2
    !   hr(io+n_c, io,     irpt) += J * (S_x + i*S_y) / 2
    !
    TYPE(wannham), intent(inout) :: ham_out
    integer,  intent(in) :: norb_bare, irpt
    real(dp), intent(in) :: J_val
    real(dp), dimension(3), intent(in) :: Svec  ! (S_x, S_y, S_z)
    !
    integer :: io, n_c
    complex(dp) :: Jsp, Jsm, Jsz, Jhalf
    !
    n_c  = norb_bare / 2
    !
    ! J/2 * each component
    Jsz  = cmplx(J_val * Svec(3) / 2.0_dp, 0.0_dp, KIND=dp)
    Jsp  = cmplx(J_val * Svec(1) / 2.0_dp, -J_val * Svec(2) / 2.0_dp, KIND=dp)  ! S_x - i*S_y
    Jsm  = cmplx(J_val * Svec(1) / 2.0_dp,  J_val * Svec(2) / 2.0_dp, KIND=dp)  ! S_x + i*S_y
    !
    do io = 1, n_c
      ham_out%hr(io,     io,     irpt) = ham_out%hr(io,     io,     irpt) + Jsz
      ham_out%hr(io+n_c, io+n_c, irpt) = ham_out%hr(io+n_c, io+n_c, irpt) - Jsz
      ham_out%hr(io,     io+n_c, irpt) = ham_out%hr(io,     io+n_c, irpt) + Jsp
      ham_out%hr(io+n_c, io,     irpt) = ham_out%hr(io+n_c, io,     irpt) + Jsm
    enddo
    !
  END SUBROUTINE add_js_coupling
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE js_objective(params, n_params, val)
    !
    ! Callback for bayesian_optimize: compute L2 eigenvalue mismatch.
    !
    ! Scalar mode (g_n_jrpt == 0): params = (J_0, S_x, S_y, S_z)
    ! Tensor mode: params = (J(R_1),...,J(R_n), S_x, S_y, S_z)
    !
    use linalgwrap, only : eigen
    !
    integer,  intent(in) :: n_params
    real(dp), dimension(n_params), intent(in) :: params
    real(dp), intent(out) :: val
    !
    integer :: ik, n_jrpt_local
    real(dp) :: J_0
    real(dp), dimension(3) :: Svec
    complex(dp), allocatable :: hk_bare(:,:), hk_tmp1(:,:), hk_tmp2(:,:)
    real(dp),    allocatable :: eig_trial(:), eig_eff(:)
    !
    n_jrpt_local = g_n_jrpt
    !
    ! Unpack parameters
    if (.not. g_J_TENSOR) then
      J_0      = params(1)
      Svec(1:3)= params(2:4)
    else
      J_0      = 0.0_dp
      Svec(1:3)= params(n_jrpt_local+1:n_jrpt_local+3)
    endif
    !
    allocate(hk_bare(g_norb_bare, g_norb_bare))
    allocate(hk_tmp1(g_norb_bare, g_norb_bare))
    allocate(hk_tmp2(g_norb_cc, g_norb_cc))
    allocate(eig_trial(g_norb_bare))
    allocate(eig_eff(g_norb_cc))
    !
    val = 0.0_dp
    !
    do ik = 1, g_nkirr
      !
      ! H_bare(k) + H_JS(k)
      call calc_hk(hk_bare, g_ham_bare, g_kvec(:, ik))
      hk_tmp1 = hk_bare
      !
      if (.not. g_J_TENSOR) then
        call add_js_coupling_kspace(hk_tmp1, g_norb_bare, J_0, Svec)
      else
        call add_js_coupling_tensor_kspace(hk_tmp1, g_norb_bare, params(1:n_jrpt_local), &
                                            Svec, n_jrpt_local, g_rvec_J, g_kvec(:,ik))
      endif
      !
      ! Eigenvalues of H_trial (heigen overwrites hk_tmp1 with eigenvectors)
      call eigen(eig_trial, hk_tmp1, g_norb_bare)
      !
      ! Eigenvalues of H_eff_CC (separate copy to avoid aliasing)
      hk_tmp2 = g_hk_eff_cc(:, :, ik)
      call eigen(eig_eff, hk_tmp2, g_norb_cc)
      !
      ! L2 norm of sorted eigenvalue differences
      ! Both are already sorted ascending by heigen
      val = val + sum((eig_trial - eig_eff)**2)
      !
    enddo
    !
    val = val / real(g_nkirr, dp)
    !
    deallocate(hk_bare, hk_tmp1, hk_tmp2, eig_trial, eig_eff)
    !
  END SUBROUTINE js_objective
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE add_js_coupling_kspace(hk, norb, J_val, Svec)
    !
    ! Add J*(S.sigma)/2 directly to k-space Hamiltonian (scalar R=0 case).
    !
    complex(dp), dimension(norb, norb), intent(inout) :: hk
    integer,  intent(in) :: norb
    real(dp), intent(in) :: J_val
    real(dp), dimension(3), intent(in) :: Svec
    !
    integer :: io, n_c
    complex(dp) :: Jsz, Jsp, Jsm
    !
    n_c = norb / 2
    Jsz = cmplx(J_val * Svec(3) / 2.0_dp, 0.0_dp, KIND=dp)
    Jsp = cmplx(J_val * Svec(1) / 2.0_dp, -J_val * Svec(2) / 2.0_dp, KIND=dp)
    Jsm = cmplx(J_val * Svec(1) / 2.0_dp,  J_val * Svec(2) / 2.0_dp, KIND=dp)
    !
    do io = 1, n_c
      hk(io,     io)     = hk(io,     io)     + Jsz
      hk(io+n_c, io+n_c) = hk(io+n_c, io+n_c) - Jsz
      hk(io,     io+n_c) = hk(io,     io+n_c) + Jsp
      hk(io+n_c, io)     = hk(io+n_c, io)     + Jsm
    enddo
    !
  END SUBROUTINE add_js_coupling_kspace
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE add_js_coupling_tensor_kspace(hk, norb, jeff_R, Svec, n_jrpt, rvec_J, kvec)
    !
    ! Add tensor JS coupling: sum_R J(R)*exp(ik.R) * (S.sigma)/2
    !
    use constants, only : twopi
    !
    complex(dp), dimension(norb, norb), intent(inout) :: hk
    integer,  intent(in) :: norb, n_jrpt
    real(dp), dimension(n_jrpt), intent(in) :: jeff_R
    real(dp), dimension(3), intent(in) :: Svec, kvec
    real(dp), dimension(3, n_jrpt), intent(in) :: rvec_J
    !
    integer :: ir, io, n_c
    real(dp) :: rdotk, Jk_re, Jk_im
    complex(dp) :: Jk
    complex(dp) :: Jsz, Jsp, Jsm
    !
    n_c = norb / 2
    !
    ! Compute J(k) = sum_R J(R) * exp(i*2pi*k.R)
    Jk = cmplx_0
    do ir = 1, n_jrpt
      rdotk = sum(kvec(:) * rvec_J(:, ir)) * twopi
      Jk = Jk + jeff_R(ir) * cmplx(cos(rdotk), sin(rdotk), KIND=dp)
    enddo
    !
    Jsz = Jk * cmplx(Svec(3) / 2.0_dp, 0.0_dp, KIND=dp)
    Jsp = Jk * cmplx(Svec(1) / 2.0_dp, -Svec(2) / 2.0_dp, KIND=dp)
    Jsm = Jk * cmplx(Svec(1) / 2.0_dp,  Svec(2) / 2.0_dp, KIND=dp)
    !
    do io = 1, n_c
      hk(io,     io)     = hk(io,     io)     + Jsz
      hk(io+n_c, io+n_c) = hk(io+n_c, io+n_c) - Jsz
      hk(io,     io+n_c) = hk(io,     io+n_c) + Jsp
      hk(io+n_c, io)     = hk(io+n_c, io)     + Jsm
    enddo
    !
  END SUBROUTINE add_js_coupling_tensor_kspace
  !
  ! ---------------------------------------------------------------------------
  ! Wrapper matching bayesian_optimize callback interface
  FUNCTION js_objective_callback(params, n) RESULT(val)
    use constants, only : dp
    integer,  intent(in) :: n
    real(dp), dimension(n), intent(in) :: params
    real(dp) :: val
    call js_objective(params, n, val)
  END FUNCTION js_objective_callback
  !
END MODULE wanneff_js_mod

! ===========================================================================
! PROGRAM WannEffJS
! ===========================================================================
PROGRAM WannEffJS
  !
  use constants,       only : stdout, dp, cmplx_0, fin
  use para,            only : init_para, inode, finalize_para, distribute_calc, &
                               first_idx, last_idx, para_merge_cmplx, para_sync_logical, &
                               para_sync0
  use wanndata,        only : wannham, read_ham, wannham_shift_ef, calc_hk, &
                               write_ham, finalize_wann
  use lattice,         only : read_posfile, read_kmesh, ham, nkirr, kvec, kwt, &
                               avec, nsite, nbasis, spinor, xat, finalize_lattice_kmesh, &
                               finalize_lattice_structure, finalize_lattice_ham, &
                               find_ws
  use input,           only : read_input, read_effjs_input, finalize_input, &
                               mu, seed, seedbare, eff_js, eff_mc, &
                               mc_temperature, mc_weiss_mean_field, J_mc, &
                               J_TENSOR, tol_Jeff, J_R_range, bayes_niter, &
                               J_bounds, S_bounds, mc_supercell, sigma_broadening, &
                               berry_curvature_output, &
                               read_qpoints, nqpt, qvec
  use bayesian,        only : bayesian_optimize
  use classical_mc,    only : classical_mc_run
  use transp_calc,     only : calc_sigma_xy, calc_sigma_xx, calc_berry_curvature_kmap
  use wannlog,         only : log_init, log_start, log_stop, log_msg, log_print_summary
  use linalgwrap,      only : eigen
  use wanneff_js_mod
  !
  implicit none
  !
  TYPE(wannham), TARGET :: ham_bare   ! seedbare (conduction only)
  TYPE(wannham) :: ham_out    ! output effective Hamiltonian
  ! Note: global 'ham' from lattice module is used as ham_full (full system with f)
  !
  integer :: norb_full, norb_bare, norb_cc
  integer, allocatable :: cc_idx(:)
  integer, allocatable :: nbasis_full(:), nbasis_bare(:)
  integer :: nsite_full, nsite_bare, nsite_f
  !
  complex(dp), allocatable :: hk_full(:,:), hk_eff_cc(:,:)
  real(dp), allocatable :: rvec_J(:,:)
  integer,  allocatable :: wt_J(:)
  integer :: n_jrpt
  !
  real(dp), dimension(:), allocatable :: params_opt
  real(dp), dimension(:,:), allocatable :: bounds_bayes
  real(dp), dimension(3) :: Svec_opt
  real(dp) :: J_opt, S_mag_opt, J_mc_used
  real(dp), allocatable :: jeff_R(:)
  !
  real(dp), allocatable :: mvec_vs_T(:,:)
  integer :: n_temps, iT
  real(dp), dimension(3) :: S_eff_vec
  real(dp) :: T_now, sigma_xy, sigma_xx
  real(dp), allocatable :: omega_kmap_seed(:), omega_kmap_bare(:), omega_kmap_eff(:)
  !
  real(dp), allocatable :: frac_pos_f(:,:)
  integer :: ik, ii, jj, n_pruned, io_tmp, iq_js
  character(len=120) :: fname_out
  ! Eigenvalue comparison variables (JS_result.dat)
  complex(dp), allocatable :: hk_js_seed(:,:), hk_js_eff(:,:)
  real(dp),    allocatable :: eig_js_seed(:),  eig_js_eff(:)
  CALL init_para('WannEffJS')
  CALL read_effjs_input('wanneff')
  !
  if (inode .eq. 0) CALL log_init()
  if (inode .eq. 0) CALL log_msg('Program WannEffJS started')
  !
  if (inode .eq. 0) then
    write(stdout, *) "========================================"
    write(stdout, *) " WannEffJS: J-S Kondo coupling fitter"
    write(stdout, *) "========================================"
    write(stdout, '(A,A)') "  seed     = ", trim(seed)
    write(stdout, '(A,A)') "  seedbare = ", trim(seedbare)
    write(stdout, '(A,L)') "  eff_js   = ", eff_js
    write(stdout, '(A,L)') "  eff_mc   = ", eff_mc
    write(stdout, '(A,L)') "  J_TENSOR = ", J_TENSOR
  endif
  !
  ! ---- Read seed (full system) into GLOBAL ham (lattice module variable) ----
  ! IMPORTANT: read_posfile sets ham%tau on the global lattice%ham,
  ! so we use the global ham directly for the full system.
  CALL read_ham(ham, seed)
  CALL read_posfile(trim(seed)//'.pos')  ! sets ham%tau AND avec, nsite, nbasis
  CALL wannham_shift_ef(ham, mu)
  nsite_full = nsite
  allocate(nbasis_full(nsite_full))
  nbasis_full = nbasis(1:nsite_full)
  norb_full = ham%norb
  !
  ! ---- Read seedbare (conduction system) into local ham_bare ----
  CALL read_ham(ham_bare, seedbare)
  norb_bare = ham_bare%norb
  norb_cc   = norb_bare
  !
  ! Read seedbare pos to get nbasis/xat for CC tau assignment
  CALL finalize_lattice_structure       ! deallocate nbasis/xat/zat before re-reading
  CALL read_posfile(trim(seedbare)//'.pos')  ! reads xat, nbasis for seedbare
  nsite_bare = nsite
  allocate(nbasis_bare(nsite_bare))
  nbasis_bare = nbasis(1:nsite_bare)
  !
  ! Manually assign tau to ham_bare using xat from seedbare pos
  ! (read_posfile sets global ham%tau, but ham_bare is a separate variable)
  ham_bare%tau(:,:) = 0.0_dp
  ii = 1
  do jj = 1, nsite_bare
    do io_tmp = 1, nbasis_bare(jj)   ! orbital within site
      ham_bare%tau(:, ii) = xat(:, jj)
      ii = ii + 1
    enddo
  enddo
  if (spinor) then
    ! Duplicate for spin-down block
    do jj = 1, nsite_bare
      do io_tmp = 1, nbasis_bare(jj)
        ham_bare%tau(:, ii) = xat(:, jj)
        ii = ii + 1
      enddo
    enddo
  endif
  !
  ! Restore seed pos for k-mesh (avec must reflect seed lattice)
  CALL finalize_lattice_structure
  CALL read_posfile(trim(seed)//'.pos')  ! restores avec, nsite, nbasis for seed
  !
  ! ---- K-mesh ----
  CALL read_kmesh('IBZKPT')
  !
  ! ---- CC indices ----
  allocate(cc_idx(norb_cc))
  CALL determine_cc_indices(cc_idx, norb_full, norb_bare, &
                              nbasis_full, nbasis_bare, nsite_full, nsite_bare)
  !
  if (inode .eq. 0) then
    write(stdout, '(A,1I5)') "  norb_full = ", norb_full
    write(stdout, '(A,1I5)') "  norb_bare = ", norb_bare
    write(stdout, '(A,1I5)') "  norb_cc   = ", norb_cc
    write(stdout, '(A,1I5)') "  n_kpts    = ", nkirr
    write(stdout, *) "  CC orbital indices: "
    write(stdout, '(10I5)') cc_idx
  endif
  !
  ! ---- Set up J(R) R-grid (tensor mode) ----
  if (J_TENSOR) then
    if (all(J_R_range == 0)) then
      ! Use same R-grid as ham_bare
      n_jrpt = ham_bare%nrpt
      allocate(rvec_J(3, n_jrpt), wt_J(n_jrpt))
      rvec_J = ham_bare%rvec
      wt_J   = nint(ham_bare%weight)
    else
      ! Custom range: build WS grid
      ! find_ws from lattice module uses module-level avec
      n_jrpt = -1
      call find_ws(n_jrpt, J_R_range(2), J_R_range(4), J_R_range(6))
      allocate(rvec_J(3, n_jrpt), wt_J(n_jrpt))
      call find_ws(n_jrpt, J_R_range(2), J_R_range(4), J_R_range(6), rvec_J, wt_J)
      ! wt_J weights already set by find_ws
    endif
    if (inode .eq. 0) then
      write(stdout, '(A,1I5,A)') "  J_TENSOR: n_jrpt = ", n_jrpt, " R-vectors for J"
    endif
  else
    n_jrpt = 0
  endif
  !
  ! ---- Downfolding loop ----
  ! Use global 'ham' (the full system hamiltonian) for calc_hk
  if (inode .eq. 0) CALL log_start('downfolding')
  allocate(hk_full(norb_full, norb_full))
  allocate(hk_eff_cc(norb_cc, norb_cc))
  allocate(g_hk_eff_cc(norb_cc, norb_cc, nkirr))
  g_hk_eff_cc = cmplx_0
  !
  CALL distribute_calc(nkirr)
  !
  do ik = first_idx, last_idx
    CALL calc_hk(hk_full, ham, kvec(:, ik))   ! use global ham
    CALL downfold_to_cc(hk_eff_cc, hk_full, norb_full, norb_cc, cc_idx)
    g_hk_eff_cc(:, :, ik) = hk_eff_cc
  enddo
  !
  CALL para_merge_cmplx(g_hk_eff_cc, norb_cc * norb_cc * nkirr)
  !
  if (inode .eq. 0) then
    write(stdout, *) "  Downfolding done."
    CALL log_stop('downfolding')
  endif
  !
  ! ---- Set up module-level globals for Bayesian callback ----
  g_norb_bare = norb_bare
  g_norb_cc   = norb_cc
  g_nkirr     = nkirr
  g_n_jrpt    = n_jrpt
  g_J_TENSOR  = J_TENSOR
  g_ham_bare  => ham_bare
  allocate(g_kvec(3, nkirr))
  g_kvec = kvec(:, 1:nkirr)
  if (J_TENSOR .and. n_jrpt > 0) then
    allocate(g_rvec_J(3, n_jrpt))
    g_rvec_J = rvec_J
  endif
  !
  ! ---- Bayesian Optimization ----
  if (eff_js .and. inode .eq. 0) then
    CALL log_start('bayesian_optimize')
    !
    if (.not. J_TENSOR) then
      ! Scalar mode: 4 parameters (J_0, S_x, S_y, S_z)
      allocate(bounds_bayes(2, 4), params_opt(4))
      bounds_bayes(1, 1) = J_bounds(1)
      bounds_bayes(2, 1) = J_bounds(2)
      bounds_bayes(1, 2:4) = S_bounds(1)
      bounds_bayes(2, 2:4) = S_bounds(2)
      !
      write(stdout, *) "  Bayesian optimization: scalar J (4 params)"
      CALL bayesian_optimize(js_objective_callback, bounds_bayes, 4, params_opt, bayes_niter)
      !
      J_opt      = params_opt(1)
      Svec_opt   = params_opt(2:4)
      !
    else
      ! Tensor mode: (n_jrpt + 3) parameters
      allocate(bounds_bayes(2, n_jrpt+3), params_opt(n_jrpt+3))
      bounds_bayes(1, 1:n_jrpt) = J_bounds(1)
      bounds_bayes(2, 1:n_jrpt) = J_bounds(2)
      bounds_bayes(1, n_jrpt+1:n_jrpt+3) = S_bounds(1)
      bounds_bayes(2, n_jrpt+1:n_jrpt+3) = S_bounds(2)
      !
      write(stdout, '(A,1I5,A)') "  Bayesian optimization: tensor J (", n_jrpt+3, " params)"
      CALL bayesian_optimize(js_objective_callback, bounds_bayes, n_jrpt+3, params_opt, bayes_niter)
      !
      allocate(jeff_R(n_jrpt))
      jeff_R   = params_opt(1:n_jrpt)
      Svec_opt = params_opt(n_jrpt+1:n_jrpt+3)
      J_opt    = 1.0_dp  ! not used in tensor mode
      !
      ! Apply tol_Jeff pruning
      n_pruned = 0
      do ii = 1, n_jrpt
        if (abs(jeff_R(ii)) < tol_Jeff) then
          jeff_R(ii) = 0.0_dp
          n_pruned = n_pruned + 1
        endif
      enddo
      write(stdout, '(A,1I5,A,1I5,A)') "  tol_Jeff pruning: ", n_pruned, " of ", &
            n_jrpt, " J(R) set to zero"
      !
    endif
    !
    S_mag_opt = sqrt(sum(Svec_opt**2))
    write(stdout, '(A,1F10.5)') "  J_opt   = ", J_opt
    write(stdout, '(A,3F10.5)') "  Svec_opt= ", Svec_opt
    write(stdout, '(A,1F10.5)') "  |S|_opt = ", S_mag_opt
    !
    ! ---- Write seed_JS.output ----
    write(fname_out, '(A,A)') trim(seed), '_JS.output'
    open(unit=99, file=trim(fname_out), status='replace')
    write(99, '(A)')        '# wanneff_JS output'
    write(99, '(A,A)')      '# seed     = ', trim(seed)
    write(99, '(A,A)')      '# seedbare = ', trim(seedbare)
    write(99, '(A,L1)')     '# J_TENSOR = ', J_TENSOR
    write(99, '(A,I6)')     '# bayes_niter = ', bayes_niter
    write(99, '(A,ES10.4)') '# tol_Jeff    = ', tol_Jeff
    write(99, '(A)')        ''
    if (.not. J_TENSOR) then
      write(99, '(A,F12.6)')  'J_opt  ', J_opt
    endif
    write(99, '(A,F12.6)')  'S_x    ', Svec_opt(1)
    write(99, '(A,F12.6)')  'S_y    ', Svec_opt(2)
    write(99, '(A,F12.6)')  'S_z    ', Svec_opt(3)
    write(99, '(A,F12.6)')  'S_mag  ', S_mag_opt
    if (J_TENSOR .and. n_jrpt > 0) then
      write(99, '(A,I6)')   'n_jrpt ', n_jrpt
      write(99, '(A)')      ''
      write(99, '(A)')      '# R1   R2   R3        J(R)'
      do ii = 1, n_jrpt
        write(99, '(3I5,F14.8)') nint(rvec_J(:,ii)), jeff_R(ii)
      enddo
    endif
    close(99)
    write(stdout, '(A,A)')  '  JS output written: ', trim(fname_out)
    CALL log_stop('bayesian_optimize')
    !
  endif
  !
  ! ---- Construct output Hamiltonian at T=0 ----
  if (eff_js .and. inode .eq. 0) then
    !
    ! Deep copy ham_bare to ham_out
    ham_out%norb = ham_bare%norb
    ham_out%nrpt = ham_bare%nrpt
    ham_out%r000 = ham_bare%r000
    allocate(ham_out%hr(ham_out%norb, ham_out%norb, ham_out%nrpt))
    allocate(ham_out%weight(ham_out%nrpt))
    allocate(ham_out%rvec(3, ham_out%nrpt))
    allocate(ham_out%tau(3, ham_out%norb))
    ham_out%hr     = ham_bare%hr
    ham_out%weight = ham_bare%weight
    ham_out%rvec   = ham_bare%rvec
    ham_out%tau    = ham_bare%tau
    !
    if (.not. J_TENSOR) then
      ! Scalar: add at R=0 only
      CALL add_js_coupling(ham_out, norb_bare, J_opt, Svec_opt, ham_out%r000)
    else
      ! Tensor: add at each R-point (J_R_range grid)
      do ii = 1, n_jrpt
        if (abs(jeff_R(ii)) < eps6) cycle
        ! Find the corresponding R-point in ham_out
        do jj = 1, ham_out%nrpt
          if (all(abs(ham_out%rvec(:,jj) - rvec_J(:,ii)) < 0.5_dp)) then
            CALL add_js_coupling(ham_out, norb_bare, jeff_R(ii), Svec_opt, jj)
            exit
          endif
        enddo
      enddo
    endif
    !
    ! Write 0K output
    write(fname_out, '(A,A)') trim(seedbare), '_hr_0K'
    CALL write_ham(ham_out, trim(fname_out))
    !
    ! Compute AHC at T=0 using existing k-mesh
    CALL log_start('transport_calc')
    CALL calc_sigma_xy(sigma_xy, ham_out, kvec(:,1:nkirr), kwt(1:nkirr), nkirr, 0.0_dp, 0.0_dp)
    CALL calc_sigma_xx(sigma_xx, ham_out, kvec(:,1:nkirr), kwt(1:nkirr), nkirr, 0.0_dp, 0.0_dp, sigma_broadening)
    write(stdout, '(A,F12.6,A)') "  AHC sigma_xy(0K) = ", sigma_xy, " e^2/h"
    write(stdout, '(A,F12.6,A)') "  sigma_xx(0K)     = ", sigma_xx, " e^2/h"
    CALL log_stop('transport_calc')
    !
    ! ---- Write seed_JS_result.dat: eigenvalue comparison along band path ----
    CALL read_qpoints   ! reads QPOINTS file -> nqpt, qvec
    allocate(hk_js_seed(norb_full, norb_full), eig_js_seed(norb_full))
    allocate(hk_js_eff(norb_bare,  norb_bare),  eig_js_eff(norb_bare))
    write(fname_out, '(A,A)') trim(seed), '_JS_result.dat'
    open(unit=97, file=trim(fname_out), status='replace')
    write(97, '(A,I5,A,I5)') '# norb_seed=', norb_full, '  norb_eff=', norb_bare
    write(97, '(A)') '# q_x       q_y       q_z       eig_seed(1..N) eig_eff(1..M)'
    do iq_js = 1, nqpt
      call calc_hk(hk_js_seed, ham,     qvec(:, iq_js))
      call eigen(eig_js_seed, hk_js_seed, norb_full)
      call calc_hk(hk_js_eff,  ham_out, qvec(:, iq_js))
      call eigen(eig_js_eff,  hk_js_eff,  norb_bare)
      write(97, '(3F10.5,100F10.4)') qvec(:, iq_js), eig_js_seed, eig_js_eff
    enddo
    close(97)
    deallocate(hk_js_seed, eig_js_seed, hk_js_eff, eig_js_eff)
    write(stdout, '(A,A)') '  JS result written: ', trim(fname_out)
    !
  endif
  !
  ! ---- Optional Berry curvature k-map output ----
  if (eff_js .and. berry_curvature_output .and. inode .eq. 0) then
    !
    allocate(omega_kmap_seed(nkirr), omega_kmap_bare(nkirr), omega_kmap_eff(nkirr))
    CALL calc_berry_curvature_kmap(omega_kmap_seed, ham,      kvec(:,1:nkirr), kwt(1:nkirr), nkirr, 0.0_dp, 0.0_dp)
    CALL calc_berry_curvature_kmap(omega_kmap_bare, ham_bare, kvec(:,1:nkirr), kwt(1:nkirr), nkirr, 0.0_dp, 0.0_dp)
    CALL calc_berry_curvature_kmap(omega_kmap_eff,  ham_out,  kvec(:,1:nkirr), kwt(1:nkirr), nkirr, 0.0_dp, 0.0_dp)
    !
    write(fname_out, '(A,A)') trim(seed), '_berry_curvature.dat'
    open(unit=96, file=trim(fname_out), status='replace')
    write(96, '(A)') '# k_x       k_y       k_z       Omega_seed  Omega_bare  Omega_eff(0K)'
    do ik = 1, nkirr
      write(96, '(3F10.5,3F14.6)') kvec(:,ik), omega_kmap_seed(ik), omega_kmap_bare(ik), omega_kmap_eff(ik)
    enddo
    close(96)
    deallocate(omega_kmap_seed, omega_kmap_bare, omega_kmap_eff)
    write(stdout, '(A,A)') '  Berry curvature written: ', trim(fname_out)
    !
  endif
  !
  ! ---- Monte Carlo temperature sweep ----
  if (eff_mc .and. eff_js .and. inode .eq. 0) then
    !
    ! Open transport vs temperature output file
    open(unit=98, file=trim(seed)//'_transport_vs_T.dat', status='replace')
    write(98, '(A)') '# T(K)  sigma_xy(e^2/h)  sigma_xx(e^2/h)'
    write(98, '(F10.2,2F14.6)') 0.0_dp, sigma_xy, sigma_xx  ! T=0 entry (computed above)
    !
    ! Get f-site positions: sites in seed but not in seedbare
    nsite_f = nsite_full - nsite_bare
    if (nsite_f < 1) nsite_f = 1  ! fallback: use last site
    allocate(frac_pos_f(3, max(1, nsite_f)))
    do ii = 1, max(1, nsite_f)
      frac_pos_f(:, ii) = xat(:, min(nsite_bare + ii, nsite_full))
    enddo
    !
    ! Determine J_mc from Weiss mean-field if requested
    if (mc_weiss_mean_field) then
      ! T_c ~ J_eff * |S|^2 / 3 (MFT), J_mc = J_eff * S_mag / z
      ! We use a rough z=6 if unknown (typical NN coordination)
      J_mc_used = J_opt * S_mag_opt / 6.0_dp
      write(stdout, '(A,1F10.5,A)') "  Weiss mean-field J_mc = ", J_mc_used, " eV"
    else
      J_mc_used = J_mc
      write(stdout, '(A,1F10.5,A)') "  Manual J_mc = ", J_mc_used, " eV"
    endif
    !
    write(stdout, '(A,1F8.3,A,1F8.3,A,1F8.3,A)') "  MC: T from ", &
          mc_temperature(1), " to ", mc_temperature(3), " step ", mc_temperature(2), " eV"
    CALL log_start('classical_mc')
    !
    CALL classical_mc_run(J_mc_used, S_mag_opt, &
                           frac_pos_f, max(1, nsite_f), avec, &
                           mc_temperature(1), mc_temperature(2), mc_temperature(3), &
                           mvec_vs_T, n_temps, mc_supercell)
    !
    ! Write output for each temperature
    do iT = 1, n_temps
      S_eff_vec = S_mag_opt * mvec_vs_T(:, iT)
      !
      if (mc_temperature(2) < 1.0d-12) then
        T_now = mc_temperature(1)
      else
        T_now = mc_temperature(1) + real(iT-1, dp) * mc_temperature(2)
      endif
      !
      ! Reconstruct ham_out with temperature-dependent S
      ham_out%hr = ham_bare%hr
      !
      if (.not. J_TENSOR) then
        CALL add_js_coupling(ham_out, norb_bare, J_opt, S_eff_vec, ham_out%r000)
      else
        do ii = 1, n_jrpt
          if (abs(jeff_R(ii)) < eps6) cycle
          do jj = 1, ham_out%nrpt
            if (all(abs(ham_out%rvec(:,jj) - rvec_J(:,ii)) < 0.5_dp)) then
              CALL add_js_coupling(ham_out, norb_bare, jeff_R(ii), S_eff_vec, jj)
              exit
            endif
          enddo
        enddo
      endif
      !
      ! Write temperature-labeled output (integer K label from eV: T_K = T_eV * 11604)
      write(fname_out, '(A,A,I0,A)') trim(seedbare), '_hr_', &
            nint(T_now * 11604.522_dp), 'K'
      CALL write_ham(ham_out, trim(fname_out))
      !
      ! Compute transport at this temperature using existing k-mesh
      CALL calc_sigma_xy(sigma_xy, ham_out, kvec(:,1:nkirr), kwt(1:nkirr), nkirr, 0.0_dp, 0.0_dp)
      CALL calc_sigma_xx(sigma_xx, ham_out, kvec(:,1:nkirr), kwt(1:nkirr), nkirr, 0.0_dp, 0.0_dp, sigma_broadening)
      write(98, '(F10.2,2F14.6)') T_now * 11604.522_dp, sigma_xy, sigma_xx
      !
    enddo
    !
    close(98)
    deallocate(frac_pos_f, mvec_vs_T)
    CALL log_stop('classical_mc')
    !
  endif
  !
  ! ---- Cleanup ----
  if (allocated(cc_idx))            deallocate(cc_idx)
  if (allocated(hk_full))           deallocate(hk_full)
  if (allocated(hk_eff_cc))         deallocate(hk_eff_cc)
  if (allocated(g_hk_eff_cc))       deallocate(g_hk_eff_cc)
  if (allocated(g_kvec))            deallocate(g_kvec)
  if (allocated(g_rvec_J))          deallocate(g_rvec_J)
  if (allocated(rvec_J))            deallocate(rvec_J)
  if (allocated(wt_J))              deallocate(wt_J)
  if (allocated(params_opt))        deallocate(params_opt)
  if (allocated(bounds_bayes))      deallocate(bounds_bayes)
  if (allocated(jeff_R))            deallocate(jeff_R)
  if (allocated(omega_kmap_seed))   deallocate(omega_kmap_seed)
  if (allocated(omega_kmap_bare))   deallocate(omega_kmap_bare)
  if (allocated(omega_kmap_eff))    deallocate(omega_kmap_eff)
  if (allocated(nbasis_full))  deallocate(nbasis_full)
  if (allocated(nbasis_bare))  deallocate(nbasis_bare)
  !
  CALL finalize_lattice_ham           ! finalizes global ham (= ham_full)
  CALL finalize_wann(ham_bare, .true.)
  CALL finalize_wann(ham_out,  .true.)
  CALL finalize_lattice_kmesh
  CALL finalize_lattice_structure
  CALL finalize_input
  if (inode .eq. 0) CALL log_print_summary()
  CALL finalize_para
  !
  if (inode .eq. 0) write(stdout, *) "WannEffJS done."
  !
END PROGRAM WannEffJS
