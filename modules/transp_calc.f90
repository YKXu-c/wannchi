!
!   transp_calc.f90
!
! ===========================================================================
! MODULE transp_calc
!   Transport properties from Wannier HR:
!     - Anomalous Hall Conductivity (AHC, sigma_xy) via Berry curvature
!     - Longitudinal DC conductivity (sigma_xx) via Kubo-Greenwood
!     - Velocity matrices (analytically from HR)
!
!   AHC formula (Kubo, Berry curvature form):
!     sigma_xy = -(e^2/h) * (1/N_k) * sum_k sum_n f_n(k) * Omega_n^{xy}(k)
!
!   Berry curvature per band (sum-over-states):
!     Omega_n^{xy}(k) = -2 Im sum_{m/=n} <n,k|v_x|m,k><m,k|v_y|n,k> / (E_n-E_m)^2
!
!   Velocity matrix (analytically from HR, NO numerical k-derivatives):
!     [v_alpha]_{io,jo} = sum_R [i*2pi*Rtilde_alpha * exp(i*2pi*k.R)/w(R)]
!                           * conj(phase_io) * phase_jo * hr(io,jo,R)
!     Rtilde_alpha = R_alpha + tau_{jo,alpha} - tau_{io,alpha}
!     phase_io = exp(i*2pi*k.tau_io)
!
!   This is the exact analytic derivative of calc_hk in wanndata.f90.
!   [Ref: Wang et al., PRB 74, 195118 (2006) - AHC via Wannier interpolation]
!   [Ref: Yao et al., PRL 92, 037204 (2004)]
!   [Ref: Xiao, Chang, Niu, Rev. Mod. Phys. 82, 1959 (2010)]
!
!   Units:
!     sigma_xy in units of e^2/h for 2D (nk3=1).
!     For 3D: multiply by 1/c_z where c_z is the z-periodicity.
!
!   MPI parallelization:
!     k-point loops are distributed via distribute_calc(nk).
!     Works with para_serial.f90 (serial stub) on laptop builds.
!
! ===========================================================================
MODULE transp_calc
  !
  use constants,  only : dp, twopi, cmplx_0, cmplx_i, eps6
  use wanndata,   only : wannham, calc_hk
  use linalgwrap, only : eigen
  use para,       only : distribute_calc, first_idx, last_idx, para_merge_real0, &
                        para_merge_cmplx0, para_merge_real, inode
  !
  implicit none
  !
CONTAINS
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE calc_velocity(v_alpha, ham, kvec, alpha)
    !
    ! Compute velocity matrix v_alpha(k) analytically from HR:
    !   v_alpha(io,jo) = sum_R [i*2pi*(R_alpha + tau_jo_alpha - tau_io_alpha)]
    !                    * exp(i*2pi*k.R) / w(R)
    !                    * conj(phase_io) * phase_jo * hr(io,jo,R)
    !
    ! This is the exact derivative dH(k)/dk_alpha from the Bloch sum formula.
    !
    TYPE(wannham), intent(in) :: ham
    real(dp), dimension(3), intent(in) :: kvec
    integer, intent(in) :: alpha   ! 1=x, 2=y, 3=z
    complex(dp), dimension(ham%norb, ham%norb), intent(out) :: v_alpha
    !
    integer :: ir, io, jo
    real(dp) :: rdotk, ktau, rtilde_alpha
    complex(dp) :: fact, orbfac
    complex(dp), dimension(ham%norb) :: phase
    !
    ! Phase factors phase(io) = exp(i*2pi*k.tau_io)
    do io = 1, ham%norb
      ktau = sum(kvec(:) * ham%tau(:, io)) * twopi
      phase(io) = cmplx(cos(ktau), sin(ktau), KIND=dp)
    enddo
    !
    v_alpha(:,:) = cmplx_0
    !
    do ir = 1, ham%nrpt
      rdotk = sum(kvec(:) * ham%rvec(:, ir)) * twopi
      fact  = cmplx(cos(rdotk), sin(rdotk), KIND=dp) / ham%weight(ir)
      !
      do io = 1, ham%norb
        do jo = 1, ham%norb
          !
          ! Rtilde_alpha = R_alpha + tau_jo_alpha - tau_io_alpha
          rtilde_alpha = ham%rvec(alpha, ir) + ham%tau(alpha, jo) - ham%tau(alpha, io)
          !
          ! Prefactor: i * 2pi * Rtilde_alpha (in fractional units)
          ! Combined with fact and orbital phase:
          orbfac = cmplx_i * twopi * rtilde_alpha * fact * &
                   conjg(phase(io)) * phase(jo)
          !
          v_alpha(io, jo) = v_alpha(io, jo) + orbfac * ham%hr(io, jo, ir)
          !
        enddo
      enddo
    enddo
    !
  END SUBROUTINE calc_velocity
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE calc_berry_curvature(omega_n, eigvec, vx, vy, eig, norb)
    !
    ! Compute Berry curvature per band from eigenstates and velocity matrices.
    !
    ! omega_n(n) = -2 Im sum_{m/=n} Vx_{nm} * Vy_{mn} / (E_n - E_m)^2
    !
    ! where Vx_{nm} = <n|vx|m> in the eigenbasis:
    !   vx_band = eigvec^dag . vx_orb . eigvec   (via ZGEMM)
    !
    ! eigvec: on input, contains the eigenvectors (columns) from heigen.
    ! eig:    eigenvalues from heigen (sorted ascending).
    !
    integer, intent(in) :: norb
    complex(dp), dimension(norb, norb), intent(in) :: eigvec, vx, vy
    real(dp),    dimension(norb), intent(in) :: eig
    real(dp),    dimension(norb), intent(out) :: omega_n
    !
    complex(dp), dimension(norb, norb) :: vx_band, vy_band, tmp
    complex(dp) :: numer
    real(dp) :: dE2
    integer :: n, m
    complex(dp), parameter :: zone  = cmplx(1.0_dp, 0.0_dp, KIND=dp)
    complex(dp), parameter :: zzero = cmplx(0.0_dp, 0.0_dp, KIND=dp)
    !
    ! Transform vx, vy to eigenbasis:
    !   vx_band = eigvec^H . vx . eigvec
    ! Using ZGEMM: tmp = vx . eigvec first, then vx_band = eigvec^H . tmp
    call zgemm('N', 'N', norb, norb, norb, zone, vx, norb, eigvec, norb, zzero, tmp, norb)
    call zgemm('C', 'N', norb, norb, norb, zone, eigvec, norb, tmp, norb, zzero, vx_band, norb)
    !
    call zgemm('N', 'N', norb, norb, norb, zone, vy, norb, eigvec, norb, zzero, tmp, norb)
    call zgemm('C', 'N', norb, norb, norb, zone, eigvec, norb, tmp, norb, zzero, vy_band, norb)
    !
    omega_n(:) = 0.0_dp
    !
    do n = 1, norb
      do m = 1, norb
        if (m == n) cycle
        dE2 = (eig(n) - eig(m))**2
        if (dE2 < eps6*eps6) cycle   ! skip near-degenerate
        numer = vx_band(n, m) * vy_band(m, n)
        omega_n(n) = omega_n(n) - 2.0_dp * aimag(numer) / dE2
      enddo
    enddo
    !
  END SUBROUTINE calc_berry_curvature
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE fermi_func(f, eig, mu, temperature, norb)
    !
    ! Compute Fermi-Dirac occupations.
    ! kT < eps6: use step function (T=0 limit).
    !
    integer,  intent(in)  :: norb
    real(dp), dimension(norb), intent(in) :: eig
    real(dp), intent(in)  :: mu, temperature
    real(dp), dimension(norb), intent(out) :: f
    !
    integer :: ii
    real(dp) :: x
    !
    do ii = 1, norb
      if (temperature < eps6) then
        f(ii) = merge(1.0_dp, 0.0_dp, eig(ii) <= mu)
      else
        x = (eig(ii) - mu) / temperature
        if (x > 500.0_dp) then
          f(ii) = 0.0_dp
        elseif (x < -500.0_dp) then
          f(ii) = 1.0_dp
        else
          f(ii) = 1.0_dp / (exp(x) + 1.0_dp)
        endif
      endif
    enddo
    !
  END SUBROUTINE fermi_func
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE calc_sigma_xy(sigma_xy, ham, kvec_all, kwt_all, nk, mu_chem, temperature)
    !
    ! Compute anomalous Hall conductivity sigma_xy.
    !
    ! sigma_xy = -(e^2/h) * (1/N_k) * sum_k sum_n f_n(k) * Omega_n^{xy}(k)
    !
    ! For 2D (nk3=1): result in units of e^2/h.
    ! kwt_all should sum to 1 (normalized weights).
    !
    use constants, only : stdout
    !
    TYPE(wannham), intent(in) :: ham
    integer, intent(in)  :: nk
    real(dp), dimension(3, nk), intent(in) :: kvec_all
    real(dp), dimension(nk),    intent(in) :: kwt_all
    real(dp), intent(in)  :: mu_chem, temperature
    real(dp), intent(out) :: sigma_xy
    !
    integer :: ik, norb
    complex(dp), allocatable :: hk(:,:), vx(:,:), vy(:,:)
    real(dp),    allocatable :: eig(:), omega_n(:), f_occ(:)
    real(dp) :: sigma_acc
    !
    norb = ham%norb
    allocate(hk(norb, norb), vx(norb, norb), vy(norb, norb))
    allocate(eig(norb), omega_n(norb), f_occ(norb))
    !
    ! Distribute k-points across MPI ranks
    call distribute_calc(nk)
    !
    sigma_acc = 0.0_dp
    !
    do ik = first_idx, last_idx
      !
      ! H(k): calc_hk overwrites hk
      call calc_hk(hk, ham, kvec_all(:, ik))
      !
      ! Diagonalize (heigen overwrites hk with eigenvectors, returns eig)
      call eigen(eig, hk, norb)  ! hk -> eigvec (columns), eig -> eigenvalues
      !
      ! Velocity matrices (analytical, no derivatives)
      call calc_velocity(vx, ham, kvec_all(:, ik), 1)
      call calc_velocity(vy, ham, kvec_all(:, ik), 2)
      !
      ! Berry curvature using eigenvectors stored in hk
      call calc_berry_curvature(omega_n, hk, vx, vy, eig, norb)
      !
      ! Fermi occupations
      call fermi_func(f_occ, eig, mu_chem, temperature, norb)
      !
      ! Accumulate: sigma += sum_n f_n * Omega_n * weight
      sigma_acc = sigma_acc + sum(f_occ * omega_n) * kwt_all(ik)
      !
    enddo
    !
    ! Merge results from all MPI ranks
    call para_merge_real0(sigma_acc)
    !
    ! Prefactor: -e^2/h (units where e^2/h = 1, caller multiplies physical constants)
    ! sigma_xy in units of e^2/h (dimensionless * e^2/h)
    ! The BZ integral is: (1/A_BZ) * sum_k sum_n f_n * Omega_n * (area per k-point)
    ! With normalized weights (sum kwt = 1), this is already normalized.
    sigma_xy = -sigma_acc / (twopi * twopi)
    ! Factor 1/(2pi)^2 from converting fractional k to 2D BZ integral in units of e^2/h
    ! Note: Omega_n is computed in units where k is in fractional coords (0 to 1),
    !       so Omega [frac^2] -> Omega [Ang^2] requires dividing by (2pi/a)^2 for absolute units.
    !       For comparative/dimensionless use within a model, sigma_xy gives the
    !       topological contribution in units of e^2/h when summed properly.
    !
    deallocate(hk, vx, vy, eig, omega_n, f_occ)
    !
  END SUBROUTINE calc_sigma_xy
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE calc_berry_curvature_kmap(omega_kmap, ham, kvec_all, kwt_all, nk, mu_chem, temperature)
    !
    ! Compute Berry curvature Omega_xy(k) summed over occupied bands at each k.
    ! Used for visualization / debug plotting.
    !
    TYPE(wannham), intent(in) :: ham
    integer, intent(in)  :: nk
    real(dp), dimension(3, nk), intent(in) :: kvec_all
    real(dp), dimension(nk),    intent(in) :: kwt_all
    real(dp), intent(in)  :: mu_chem, temperature
    real(dp), dimension(nk), intent(out) :: omega_kmap
    !
    integer :: ik, norb
    complex(dp), allocatable :: hk(:,:), vx(:,:), vy(:,:)
    real(dp),    allocatable :: eig(:), omega_n(:), f_occ(:)
    !
    norb = ham%norb
    allocate(hk(norb, norb), vx(norb, norb), vy(norb, norb))
    allocate(eig(norb), omega_n(norb), f_occ(norb))
    !
    ! Distribute k-points across MPI ranks
    call distribute_calc(nk)
    !
    ! Initialize output array (only ranks with indices will contribute via merge)
    omega_kmap = 0.0_dp
    !
    do ik = first_idx, last_idx
      call calc_hk(hk, ham, kvec_all(:, ik))
      call eigen(eig, hk, norb)
      call calc_velocity(vx, ham, kvec_all(:, ik), 1)
      call calc_velocity(vy, ham, kvec_all(:, ik), 2)
      call calc_berry_curvature(omega_n, hk, vx, vy, eig, norb)
      call fermi_func(f_occ, eig, mu_chem, temperature, norb)
      omega_kmap(ik) = sum(f_occ * omega_n)
    enddo
    !
    ! Merge results from all MPI ranks
    call para_merge_real(omega_kmap, nk)
    !
    deallocate(hk, vx, vy, eig, omega_n, f_occ)
    !
  END SUBROUTINE calc_berry_curvature_kmap
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE calc_sigma_xx(sigma_xx, ham, kvec_all, kwt_all, nk, mu_chem, temperature, broadening)
    !
    ! Longitudinal DC conductivity via Green's function Kubo bubble.
    !
    ! sigma_xx = (1/N_k) Σ_k Tr[ v_x · G(k, μ+iη) · v_x · G(k, μ+iη) ]
    !
    ! where G(k, μ+iη) = (μ+iη - H(k))^{-1}  (retarded Green's function at E_F)
    !
    ! This is the exact single-particle Kubo formula for DC conductivity,
    ! naturally including both intra- and inter-band contributions via the
    ! full Green's function. No separation into Drude + interband needed.
    !
    ! broadening = η in eV. Physically η = ℏ/(2τ) where τ is the scattering time.
    !
    ! [Ref: Kubo, J. Phys. Soc. Jpn. 12, 570 (1957)]
    ! [Ref: Bastin et al., J. Phys. Chem. Solids 32, 1811 (1971)]
    !
    TYPE(wannham), intent(in) :: ham
    integer, intent(in)  :: nk
    real(dp), dimension(3, nk), intent(in) :: kvec_all
    real(dp), dimension(nk),    intent(in) :: kwt_all
    real(dp), intent(in)  :: mu_chem, temperature, broadening
    real(dp), intent(out) :: sigma_xx
    !
    integer :: ik, norb, ii
    complex(dp), allocatable :: hk(:,:), vx(:,:), gf(:,:), tmp1(:,:), tmp2(:,:)
    complex(dp) :: sigma_acc, w_cmplx
    complex(dp), parameter :: zone  = cmplx(1.0_dp, 0.0_dp, KIND=dp)
    complex(dp), parameter :: zzero = cmplx(0.0_dp, 0.0_dp, KIND=dp)
    !
    external :: calc_g0
    !
    norb = ham%norb
    allocate(hk(norb, norb), vx(norb, norb))
    allocate(gf(norb, norb), tmp1(norb, norb), tmp2(norb, norb))
    !
    ! Distribute k-points across MPI ranks
    call distribute_calc(nk)
    !
    ! Retarded Green's function evaluated at E_F + i*eta
    w_cmplx = cmplx(mu_chem, broadening, KIND=dp)
    !
    sigma_acc = cmplx_0
    !
    do ik = first_idx, last_idx
      !
      call calc_hk(hk, ham, kvec_all(:, ik))
      !
      ! G(k, μ+iη) = (μ+iη - H(k))^{-1}
      call calc_g0(gf, hk, w_cmplx, norb, .false.)
      !
      ! v_x(k) — analytic from HR
      call calc_velocity(vx, ham, kvec_all(:, ik), 1)
      !
      ! Kubo bubble: Tr[ v_x · G · v_x · G ]
      ! tmp1 = v_x · G
      call zgemm('N', 'N', norb, norb, norb, zone, vx, norb, gf, norb, zzero, tmp1, norb)
      ! tmp2 = tmp1 · v_x = v_x · G · v_x
      call zgemm('N', 'N', norb, norb, norb, zone, tmp1, norb, vx, norb, zzero, tmp2, norb)
      ! product = tmp2 · G = v_x · G · v_x · G
      call zgemm('N', 'N', norb, norb, norb, zone, tmp2, norb, gf, norb, zzero, tmp1, norb)
      !
      ! Accumulate Tr(product) × w_k
      sigma_acc = sigma_acc + sum([(tmp1(ii, ii), ii=1,norb)]) * kwt_all(ik)
      !
    enddo
    !
    ! Merge results from all MPI ranks
    call para_merge_cmplx0(sigma_acc)
    !
    ! Extract imaginary part: σ_xx = -Im(bubble) / (2π)²
    ! The imaginary part of the retarded bubble gives the dissipative conductivity.
    sigma_xx = -aimag(sigma_acc) / (twopi * twopi)
    !
    deallocate(hk, vx, gf, tmp1, tmp2)
    !
  END SUBROUTINE calc_sigma_xx
  !
END MODULE transp_calc
