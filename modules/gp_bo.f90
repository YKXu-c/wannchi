!
!   gp_bo.f90
!
! ===========================================================================
! MODULE gp_bo
!   Gaussian Process Bayesian Optimization using Expected Improvement (EI).
!
!   Physical context:
!     Minimize L2 eigenvalue mismatch between H_eff_CC(k) and
!     H_bare(k) + J*(S.sigma)/2 over the Brillouin zone.
!
! ===========================================================================
!
!   APPLICABILITY:
!     - Low to medium dimensional problems (n_params <= 20-30)
!     - When sample efficiency is critical (uses GP surrogate model)
!     - When uncertainty quantification is needed (EI provides confidence)
!     - For smooth, continuous objective functions
!
!   NOT SUITABLE FOR:
!     - High-dimensional problems (n_params > 30) - suffers from curse of dimensionality
!     - Non-smooth, discontinuous objective functions
!     - Very expensive objective functions (GP updates are O(n^3))
!
!   Algorithm:
!     1. Latin Hypercube initialization (n_init = max(5, n_params))
!     2. Build GP surrogate from observations
!     3. EI acquisition + gradient refinement to select next candidate
!     4. Update GP with new observation
!     5. Periodic length-scale re-optimization for high-dim problems
!
!   Key parameters:
!     - n_cand: max(2000, 50*n_params) random candidates per iteration
!     - noise_var: 1e-4 (increased for eigenvalue-based objectives)
!     - Length scale: re-optimized every 10 iterations for n_params > 10
!
!   References:
!     - Mockus, Bayesian Approach to Global Optimization (1989)
!     - Brochu, Cora, de Freitas, arXiv:1012.2599 (2010)
!     - Bergstra & Bengio, JMLR 13, 281 (2012)
!
! ===========================================================================
MODULE gp_bo
  !
  use constants, only : dp, twopi
  use linalgwrap, only : invmat
  !
  implicit none
  !
  TYPE gp_model
    integer :: n_train    ! number of observations so far
    integer :: n_params   ! dimension of parameter space
    real(dp), allocatable :: x_train(:,:)  ! (n_params, n_train)
    real(dp), allocatable :: y_train(:)    ! objective values (n_train)
    real(dp) :: length_scale               ! RBF kernel length scale l
    real(dp) :: signal_var                 ! sigma_f^2
    real(dp) :: noise_var                  ! sigma_n^2 (jitter)
    real(dp), allocatable :: K_inv(:,:)    ! (K+sigma_n^2*I)^{-1}, (n_train x n_train)
    real(dp), allocatable :: alpha(:)      ! K_inv * y_train (n_train)
  END TYPE gp_model
  !
CONTAINS
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE gp_init(gp, n_params, ls, sv, nv)
    !
    TYPE(gp_model), intent(out) :: gp
    integer,  intent(in) :: n_params
    real(dp), intent(in) :: ls, sv, nv
    !
    gp%n_train      = 0
    gp%n_params     = n_params
    gp%length_scale = ls
    gp%signal_var   = sv
    gp%noise_var    = nv
    !
  END SUBROUTINE gp_init
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE gp_finalize(gp)
    !
    TYPE(gp_model), intent(inout) :: gp
    !
    if (allocated(gp%x_train)) deallocate(gp%x_train)
    if (allocated(gp%y_train)) deallocate(gp%y_train)
    if (allocated(gp%K_inv))   deallocate(gp%K_inv)
    if (allocated(gp%alpha))   deallocate(gp%alpha)
    gp%n_train = 0
    !
  END SUBROUTINE gp_finalize
  !
  ! ---------------------------------------------------------------------------
  FUNCTION rbf_kernel(x1, x2, n, ls, sv) RESULT(k)
    !
    ! Squared exponential (RBF) kernel:
    !   k(x1,x2) = sv * exp( -||x1-x2||^2 / (2*ls^2) )
    !
    integer,  intent(in) :: n
    real(dp), dimension(n), intent(in) :: x1, x2
    real(dp), intent(in) :: ls, sv
    real(dp) :: k
    !
    real(dp) :: sqdist
    !
    sqdist = sum((x1 - x2)**2)
    k = sv * exp(-sqdist / (2.0_dp * ls**2))
    !
  END FUNCTION rbf_kernel
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE gp_update(gp, x_new, y_new)
    !
    ! Update GP with a new observation (x_new, y_new).
    ! Rebuilds K_inv and alpha in place.
    !
    TYPE(gp_model), intent(inout) :: gp
    real(dp), dimension(gp%n_params), intent(in) :: x_new
    real(dp), intent(in) :: y_new
    !
    integer :: n, ii, jj
    real(dp), allocatable :: Kmat(:,:), K_new(:)
    !
    n = gp%n_train
    !
    ! Expand x_train and y_train
    if (allocated(gp%x_train)) then
      ! Check if we need to reallocate (n_train = 0 case handled by gp_init)
      if (n == 0) then
        allocate(gp%x_train(gp%n_params, 1), gp%y_train(1), &
                gp%K_inv(1,1), gp%alpha(1))
      else
        ! Store old data properly
        allocate(Kmat(gp%n_params, n), K_new(n))
        Kmat = gp%x_train(:, 1:n)
        K_new = gp%alpha(1:n)
        !
        ! Reallocate
        deallocate(gp%x_train, gp%y_train, gp%K_inv, gp%alpha)
        allocate(gp%x_train(gp%n_params, n+1), gp%y_train(n+1), &
                gp%K_inv(n+1, n+1), gp%alpha(n+1))
        !
        ! Copy old data back
        gp%x_train(:, 1:n) = Kmat
        gp%y_train(1:n) = K_new
        deallocate(Kmat, K_new)
      endif
    else
      allocate(gp%x_train(gp%n_params, 1), gp%y_train(1), &
              gp%K_inv(1,1), gp%alpha(1))
    endif
    !
    ! Add new observation
    gp%n_train = n + 1
    gp%x_train(:, n+1) = x_new
    gp%y_train(n+1)   = y_new
    !
    ! Build covariance matrix K(n+1, n+1)
    do jj = 1, gp%n_train
      do ii = 1, gp%n_train
        gp%K_inv(ii, jj) = rbf_kernel(gp%x_train(:, ii), gp%x_train(:, jj), &
                            gp%n_params, gp%length_scale, gp%signal_var)
      enddo
      ! Ensure diagonal is positive for numerical stability
      if (gp%K_inv(ii, ii) <= 0.0_dp) gp%K_inv(ii, ii) = gp%signal_var
      gp%K_inv(ii, ii) = gp%K_inv(ii, ii) + gp%noise_var
    enddo
    !
    ! Add small jitter to diagonal for numerical stability if needed
    do ii = 1, gp%n_train
      if (gp%K_inv(ii, ii) < gp%noise_var) gp%K_inv(ii, ii) = gp%noise_var * 10.0_dp
    enddo
    !
    ! Compute K_inv via LU decomposition
    call invmat(gp%K_inv, gp%n_train)
    !
    ! Check for NaN in K_inv - if found, reset GP
    if (any(gp%K_inv /= gp%K_inv)) then
      ! K_inv contains NaN - GP update failed, keep old state
      gp%n_train = n  ! revert n_train
      return
    endif
    !
    ! Compute alpha = K_inv * y
    gp%alpha = matmul(gp%K_inv, gp%y_train(1:gp%n_train))
    !
    ! Final check for NaN in alpha
    if (any(gp%alpha /= gp%alpha)) then
      gp%n_train = n  ! revert n_train
      return
    endif
    !
  END SUBROUTINE gp_update
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE gp_predict(gp, x_test, mu_out, sigma_out)
    !
    ! Predict mean and variance at x_test using GP model.
    !
    TYPE(gp_model), intent(in) :: gp
    real(dp), dimension(gp%n_params), intent(in) :: x_test
    real(dp), intent(out) :: mu_out, sigma_out
    !
    integer :: ii
    real(dp) :: k_star(gp%n_train), k_ss
    !
    ! Self-kernel k(x_test, x_test)
    k_ss = gp%signal_var
    !
    ! Compute k* = [k(x_test, x_i)] for i=1..n_train
    do ii = 1, gp%n_train
      k_star(ii) = rbf_kernel(x_test, gp%x_train(:, ii), &
                   gp%n_params, gp%length_scale, gp%signal_var)
    enddo
    !
    ! GP mean: mu = k* . alpha
    mu_out = dot_product(k_star, gp%alpha(1:gp%n_train))
    !
    ! Check for NaN in mu_out - use signal_var as fallback
    if (mu_out /= mu_out) mu_out = gp%signal_var  ! NaN fallback
    !
    ! GP variance: sigma^2 = k(x_test,x_test) - k* . K_inv . k*
    sigma_out = k_ss - dot_product(k_star, &
                     matmul(gp%K_inv(1:gp%n_train, 1:gp%n_train), k_star))
    !
    ! Numerical safety checks
    if (sigma_out <= 0.0_dp) then
      sigma_out = 1.0d-8  ! positive floor
    else if (sigma_out > gp%signal_var) then
      sigma_out = gp%signal_var  ! cannot exceed signal variance
    endif
    !
  END SUBROUTINE gp_predict
  !
  ! ---------------------------------------------------------------------------
  FUNCTION normal_cdf(x) RESULT(p)
    !
    ! Standard normal CDF: Phi(x) = 0.5 * erfc(-x/sqrt(2))
    !
    real(dp), intent(in) :: x
    real(dp) :: p
    !
    p = 0.5_dp * erfc(-x / sqrt(2.0_dp))
    !
  END FUNCTION normal_cdf
  !
  ! ---------------------------------------------------------------------------
  FUNCTION normal_pdf(x) RESULT(phi)
    !
    ! Standard normal PDF: phi(x) = (1/sqrt(2*pi)) * exp(-x^2/2)
    !
    real(dp), intent(in) :: x
    real(dp) :: phi
    !
    real(dp), parameter :: SQRT_2PI = sqrt(2.0_dp * 3.141592653589793_dp)
    !
    phi = exp(-0.5_dp * x*x) / SQRT_2PI
    !
  END FUNCTION normal_pdf
  !
  ! ---------------------------------------------------------------------------
  FUNCTION expected_improvement(mu, sigma, y_best) RESULT(ei)
    !
    ! Expected Improvement acquisition function:
    !   EI = (y_best - mu) * Phi(z) + sigma * phi(z)
    !   where z = (y_best - mu) / sigma
    !
    real(dp), intent(in) :: mu, sigma, y_best
    real(dp) :: ei
    !
    real(dp) :: z
    !
    ! Handle invalid inputs
    if (mu /= mu .or. sigma /= sigma .or. y_best /= y_best) then
      ei = 0.0_dp
      return
    endif
    !
    if (sigma < 1.0d-10) then
      ei = max(y_best - mu, 0.0_dp)
      return
    endif
    !
    z = (y_best - mu) / sigma
    ! Guard against overflow in exponential
    if (abs(z) > 50.0_dp) then
      if (z > 0.0_dp) then
        ei = 0.0_dp  ! EI -> 0 for very large positive z
      else
        ei = (y_best - mu) + sigma * normal_pdf(z)
      endif
    else
      ei = (y_best - mu) * normal_cdf(z) + sigma * normal_pdf(z)
    endif
    !
    ! Guard against NaN/Inf
    if (ei /= ei .or. ei < 0.0_dp) ei = 0.0_dp
    !
  END FUNCTION expected_improvement
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE latin_hypercube(samples, n_samples, n_params, bounds)
    !
    ! Generate Latin Hypercube samples in [bounds(1), bounds(2)]^n_params.
    !
    integer, intent(in) :: n_samples, n_params
    real(dp), dimension(2, n_params), intent(in) :: bounds
    real(dp), dimension(n_params, n_samples), intent(out) :: samples
    !
    integer :: i, j, k
    real(dp) :: u
    real(dp), dimension(n_params, n_samples) :: grid
    !
    do i = 1, n_params
      do j = 1, n_samples
        grid(i, j) = real(j-1, dp) / real(n_samples-1, dp)
      enddo
    enddo
    !
    ! Shuffle each dimension independently (Fisher-Yates)
    do i = 1, n_params
      do j = n_samples, 2, -1
        call random_number(u)
        k = 1 + floor(u * real(j, dp))
        ! Swap grid(i,j) with grid(i,k)
        u = grid(i, j)
        grid(i, j) = grid(i, k)
        grid(i, k) = u
      enddo
    enddo
    !
    ! Map to bounds and add jitter
    do i = 1, n_params
      do j = 1, n_samples
        call random_number(u)
        samples(i, j) = bounds(1, i) + (grid(i, j) + (u - 0.5_dp) / real(n_samples, dp)) * &
                        (bounds(2, i) - bounds(1, i))
        samples(i, j) = max(bounds(1, i), min(bounds(2, i), samples(i, j)))
      enddo
    enddo
    !
  END SUBROUTINE latin_hypercube
  !
  ! ---------------------------------------------------------------------------
  FUNCTION upper_confidence_bound(mu, sigma, kappa) RESULT(ucb)
    !
    ! UCB acquisition: mu + kappa * sigma
    !
    real(dp), intent(in) :: mu, sigma, kappa
    real(dp) :: ucb
    !
    ucb = mu + kappa * sigma
    !
  END FUNCTION upper_confidence_bound
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE gp_log_marginal_likelihood(lml, gp)
    !
    ! Compute GP log marginal likelihood:
    !   LML = -0.5 * y^T * K^{-1} * y - 0.5 * log|K| - n/2 * log(2*pi)
    !   Only depends on K (kernel hyperparameters), not on x directly.
    !
    TYPE(gp_model), intent(inout) :: gp
    real(dp), intent(out) :: lml
    !
    integer :: ii, jj, n
    real(dp) :: data_fit, log_det_K
    real(dp), allocatable :: Kmat(:,:)
    !
    n = gp%n_train
    if (n < 2) then
      lml = 0.0_dp
      return
    endif
    !
    ! Rebuild K matrix with current hyperparameters
    allocate(Kmat(n, n))
    do jj = 1, n
      do ii = 1, n
        Kmat(ii, jj) = rbf_kernel(gp%x_train(:, ii), gp%x_train(:, jj), &
                            gp%n_params, gp%length_scale, gp%signal_var)
      enddo
      Kmat(ii, ii) = Kmat(ii, ii) + gp%noise_var
    enddo
    !
    ! Compute K_inv via Cholesky (overwrite Kmat in place)
    call invmat(Kmat, n)
    !
    ! log|K| = -log|K^{-1}| (Cholesky gives L, K = L*L^T, but we have K_inv directly)
    log_det_K = 0.0_dp
    do ii = 1, n
      log_det_K = log_det_K + log(Kmat(ii, ii))
    enddo
    log_det_K = -2.0_dp * log_det_K
    !
    ! data_fit = 0.5 * y^T * K^{-1} * y
    data_fit = 0.5_dp * dot_product(gp%y_train(1:n), &
                         matmul(Kmat, gp%y_train(1:n)))
    !
    lml = data_fit - 0.5_dp * log_det_K &
          - 0.5_dp * real(gp%n_train, dp) * log(twopi)
    !
    deallocate(Kmat)
    !
  END SUBROUTINE gp_log_marginal_likelihood
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE gp_optimize_ls(gp, bounds, n_params)
    !
    ! Optimize GP length scale via golden-section search on marginal likelihood.
    ! Searches ls in [0.01*avg_range, 5.0*avg_range]. Updates gp%length_scale in place.
    !
    TYPE(gp_model), intent(inout) :: gp
    integer, intent(in) :: n_params
    real(dp), dimension(2, n_params), intent(in) :: bounds
    !
    integer :: ii
    real(dp) :: ls_lo, ls_hi, ls_mid1, ls_mid2, lml1, lml2
    real(dp) :: ls_saved, avg_range
    integer, parameter :: N_GS = 15  ! golden-section iterations
    real(dp), parameter :: phi = 0.618033988749895_dp  ! golden ratio
    !
    avg_range = 0.0_dp
    do ii = 1, n_params
      avg_range = avg_range + (bounds(2,ii) - bounds(1,ii))
    enddo
    avg_range = avg_range / real(n_params, dp)
    !
    ls_lo = avg_range * 0.01_dp   ! widened minimum search range
    ls_hi = avg_range * 5.0_dp    ! widened maximum search range
    if (ls_lo < 1.0d-6) ls_lo = 1.0d-6
    !
    ! Golden section search: maximize marginal likelihood over ls
    do ii = 1, N_GS
      ls_mid1 = ls_hi - phi * (ls_hi - ls_lo)
      ls_mid2 = ls_lo + phi * (ls_hi - ls_lo)
      !
      ls_saved = gp%length_scale
      gp%length_scale = ls_mid1
      call rebuild_K_inv(gp)
      call gp_log_marginal_likelihood(lml1, gp)
      !
      gp%length_scale = ls_mid2
      call rebuild_K_inv(gp)
      call gp_log_marginal_likelihood(lml2, gp)
      !
      if (lml1 > lml2) then
        ls_hi = ls_mid2
      else
        ls_lo = ls_mid1
      endif
    enddo
    !
    ! Set optimal length scale
    gp%length_scale = 0.5_dp * (ls_lo + ls_hi)
    call rebuild_K_inv(gp)
    !
  END SUBROUTINE gp_optimize_ls
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE rebuild_K_inv(gp)
    !
    ! Rebuild K_inv and alpha from current x_train, y_train, length_scale.
    !
    TYPE(gp_model), intent(inout) :: gp
    !
    integer :: ii, jj, n
    !
    n = gp%n_train
    if (n < 1) return
    !
    if (.not. allocated(gp%K_inv)) allocate(gp%K_inv(n, n))
    if (.not. allocated(gp%alpha)) allocate(gp%alpha(n))
    !
    ! Build K matrix
    do jj = 1, n
      do ii = 1, n
        gp%K_inv(ii, jj) = rbf_kernel(gp%x_train(:, ii), gp%x_train(:, jj), &
                            gp%n_params, gp%length_scale, gp%signal_var)
      enddo
      gp%K_inv(ii, ii) = gp%K_inv(ii, ii) + gp%noise_var
    enddo
    !
    ! Invert
    call invmat(gp%K_inv, gp%n_train)
    !
    ! Compute alpha
    gp%alpha = matmul(gp%K_inv, gp%y_train(1:n))
    !
  END SUBROUTINE rebuild_K_inv
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE refine_ei(x_out, x0, gp, y_best, n_params, bounds, n_steps)
    !
    ! Refine initial point x0 via gradient ascent on EI.
    !
    TYPE(gp_model), intent(in) :: gp
    integer, intent(in) :: n_params, n_steps
    real(dp), dimension(n_params), intent(in) :: x0
    real(dp), intent(in) :: y_best
    real(dp), dimension(2, n_params), intent(in) :: bounds
    real(dp), dimension(n_params), intent(out) :: x_out
    !
    integer :: step, ip
    real(dp), dimension(n_params) :: x_cur, x_p, x_m, gradient
    real(dp) :: step_size, h, ei_p, ei_m, mu_p, mu_m, sig_p, sig_m
    real(dp) :: avg_range
    !
    avg_range = 0.0_dp
    do ip = 1, n_params
      avg_range = avg_range + (bounds(2,ip) - bounds(1,ip))
    enddo
    avg_range = avg_range / real(n_params, dp)
    step_size = avg_range * 0.02_dp   ! 2% of avg range per step
    !
    x_cur = x0
    do step = 1, n_steps
      do ip = 1, n_params
        h = max(1.0d-4 * (bounds(2,ip) - bounds(1,ip)), 1.0d-8)
        x_p = x_cur; x_m = x_cur
        x_p(ip) = min(x_cur(ip) + h, bounds(2,ip))
        x_m(ip) = max(x_cur(ip) - h, bounds(1,ip))
        call gp_predict(gp, x_p, mu_p, sig_p)
        call gp_predict(gp, x_m, mu_m, sig_m)
        ei_p = expected_improvement(mu_p, sig_p, y_best)
        ei_m = expected_improvement(mu_m, sig_m, y_best)
        gradient(ip) = (ei_p - ei_m) / (x_p(ip) - x_m(ip) + 1.0d-14)
      enddo
      do ip = 1, n_params
        x_cur(ip) = x_cur(ip) + step_size * gradient(ip)
        x_cur(ip) = max(bounds(1,ip), min(bounds(2,ip), x_cur(ip)))
      enddo
    enddo
    !
    x_out = x_cur
    !
  END SUBROUTINE refine_ei
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE swap_int(a, b)
    integer, intent(inout) :: a, b
    integer :: tmp
    tmp = a; a = b; b = tmp
  END SUBROUTINE swap_int
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE bayesian_optimize(objective_func, bounds, n_params, result, n_iter)
    !
    ! Main Bayesian Optimization loop (improved with multi-start EI + GP LS opt).
    !
    use constants, only : stdout
    !
    interface
      function objective_func(params, n) result(val)
        use constants, only : dp
        integer,  intent(in) :: n
        real(dp), dimension(n), intent(in) :: params
        real(dp) :: val
      end function
    end interface
    !
    integer,  intent(in)  :: n_params, n_iter
    real(dp), dimension(2, n_params), intent(in)  :: bounds
    real(dp), dimension(n_params),    intent(out) :: result
    !
    TYPE(gp_model) :: gp
    integer :: n_init, n_cand, ii, jj, ip, best_idx, kr
    real(dp) :: y_val, y_best, ei_val, ei_best
    real(dp) :: mu_pred, sigma_pred
    real(dp), allocatable :: x_init(:,:)
    real(dp), allocatable :: x_cand(:,:)
    real(dp), allocatable :: x_next(:), x_refined(:)
    real(dp), allocatable :: x_all(:,:)
    real(dp), allocatable :: y_all(:)
    real(dp), allocatable :: ei_cand(:)    ! EI at each candidate
    integer,  allocatable :: top_idx(:)   ! indices of top-K candidates
    real(dp) :: u, ls_init
    integer, parameter :: K_REFINE = 5    ! top-K candidates to refine
    integer, parameter :: N_REFINE = 20   ! gradient steps per refinement
    !
    ! Hyperparameters
    ls_init = 0.0_dp
    do ii = 1, n_params
      ls_init = ls_init + (bounds(2,ii) - bounds(1,ii))
    enddo
    ls_init = ls_init / real(n_params, dp) * 0.5_dp
    if (ls_init < 1.0d-6) ls_init = 1.0d-6
    !
    call gp_init(gp, n_params, ls_init, 1.0_dp, 1.0d-2)  ! high noise for eigenvalue problems
    !
    n_init = max(5, n_params)
    n_cand = max(2000, 50*n_params)   ! increased for high-dimensional problems
    !
    allocate(x_init(n_params, n_init))
    allocate(x_next(n_params), x_refined(n_params))
    allocate(x_all(n_params, n_init + n_iter))
    allocate(y_all(n_init + n_iter))
    allocate(x_cand(n_params, n_cand))
    allocate(ei_cand(n_cand))
    allocate(top_idx(K_REFINE))
    !
    ! --- Phase 1: Latin Hypercube initialization ---
    call latin_hypercube(x_init, n_init, n_params, bounds)
    !
    y_best   = 1.0d30
    best_idx = 1
    !
    do ii = 1, n_init
      y_val = objective_func(x_init(:,ii), n_params)
      call gp_update(gp, x_init(:,ii), y_val)
      x_all(:, ii) = x_init(:, ii)
      y_all(ii)    = y_val
      if (y_val < y_best) then
        y_best   = y_val
        best_idx = ii
      endif
      if (mod(ii, 5) == 0) then
        write(stdout, '(A,1I4,A,1G14.6)') "  # Bayes init ", ii, "  f_best=", y_best
      endif
    enddo
    !
    ! Optimize GP length scale after initialization
    call gp_optimize_ls(gp, bounds, n_params)
    !
    ! --- Phase 2: Bayesian optimization loop (multi-start EI refinement) ---
    do ii = 1, n_iter
      !
      ! Sample N_cand random candidates
      do jj = 1, n_cand
        do ip = 1, n_params
          call random_number(u)
          x_cand(ip, jj) = bounds(1, ip) + (bounds(2, ip) - bounds(1, ip)) * u
        enddo
      enddo
      !
      ! Evaluate EI at all candidates, find top-K
      ei_best = -1.0_dp
      x_next  = x_cand(:, 1)
      do jj = 1, n_cand
        call gp_predict(gp, x_cand(:,jj), mu_pred, sigma_pred)
        ei_cand(jj) = expected_improvement(mu_pred, sigma_pred, y_best)
        if (ei_cand(jj) > ei_best) then
          ei_best = ei_cand(jj)
          x_next  = x_cand(:, jj)
        endif
      enddo
      !
      ! Select top-K candidates for local refinement.
      top_idx(1) = 1
      do kr = 1, n_cand
        if (ei_cand(kr) > ei_cand(top_idx(1))) top_idx(1) = kr
      enddo
      do kr = 2, K_REFINE
        top_idx(kr) = 1 + (kr-1) * (n_cand / K_REFINE)
        if (top_idx(kr) > n_cand) top_idx(kr) = n_cand
      enddo
      !
      ! Refine top-K candidates with gradient ascent on EI
      do kr = 1, K_REFINE
        call refine_ei(x_refined, x_cand(:, top_idx(kr)), gp, y_best, n_params, bounds, N_REFINE)
        call gp_predict(gp, x_refined, mu_pred, sigma_pred)
        ei_val = expected_improvement(mu_pred, sigma_pred, y_best)
        if (ei_val > ei_best) then
          ei_best = ei_val
          x_next  = x_refined
        endif
      enddo
      !
      ! Evaluate objective at best candidate
      y_val = objective_func(x_next, n_params)
      call gp_update(gp, x_next, y_val)
      x_all(:, n_init+ii) = x_next
      y_all(n_init+ii)    = y_val
      !
      if (y_val < y_best) then
        y_best   = y_val
        best_idx = n_init + ii
      endif
      !
      if (mod(ii, 10) == 0 .or. ii == n_iter) then
        write(stdout, '(A,1I4,A,G14.6,A,G14.6)') "  # Bayes iter ", ii, &
              "  EI=", ei_best, "  f_best=", y_best
        ! Re-optimize GP length scale periodically for high-dimensional problems
        if (n_params > 10 .and. mod(ii, 10) == 0) then
          call gp_optimize_ls(gp, bounds, n_params)
        endif
      else
        write(stdout, '(A,1I4,A,G14.6,A,G14.6)') "  # Bayes iter ", ii, &
              "  L2=", y_val, "  best=", y_best
      endif
      !
    enddo
    !
    ! Return best parameters found
    result = x_all(:, best_idx)
    !
    write(stdout, '(A,G14.6)') "  # Bayesian optimization done. f_best = ", y_best
    !
    deallocate(x_init, x_cand, x_next, x_refined, x_all, y_all, ei_cand, top_idx)
    call gp_finalize(gp)
    !
  END SUBROUTINE bayesian_optimize
  !
END MODULE gp_bo
