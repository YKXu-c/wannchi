!
!   bayesian.f90
!
! ===========================================================================
! MODULE bayesian
!   Gaussian Process Bayesian Optimization for J-S coupling parameter fitting.
!
!   Physical context:
!     Minimize L2 eigenvalue mismatch between H_eff_CC(k) and
!     H_bare(k) + J*(S.sigma)/2 over the Brillouin zone.
!
!   Algorithm: Gaussian Process surrogate with RBF kernel +
!              Expected Improvement (EI) acquisition function.
!     [Ref: Mockus, Bayesian Approach to Global Optimization (1989)]
!     [Ref: Brochu, Cora, de Freitas, arXiv:1012.2599 (2010)]
!
!   For J_TENSOR=.false.: optimize (J_0, S_x, S_y, S_z), 4 parameters.
!   For J_TENSOR=.true.:  optimize (J(R_1),...,J(R_n), S_x, S_y, S_z),
!                          (n+3) parameters. EI maximized via random sampling
!                          with N_cand = max(1000, 50*n_params) candidates.
!                         After convergence: prune |J(R)| < tol_Jeff to 0.
!                         [Ref: Bergstra & Bengio, JMLR 13, 281 (2012)]
!
!   Kernel (squared exponential / RBF):
!     k(x,x') = sigma_f^2 * exp(-||x-x'||^2 / (2*l^2)) + sigma_n^2*delta(x,x')
!
!   GP prediction at x*:
!     mu(x*) = k(x*,X) * (K+sigma_n^2*I)^{-1} * y
!     sigma^2(x*) = k(x*,x*) - k(x*,X) * (K+sigma_n^2*I)^{-1} * k(X,x*)
!
!   Expected Improvement:
!     EI(x*) = (y_best - mu(x*)) * Phi(z) + sigma(x*) * phi(z)
!     z = (y_best - mu(x*)) / sigma(x*)
!     Phi = normal CDF via 0.5*erfc(-z/sqrt(2))
!     phi = normal PDF = (1/sqrt(2*pi))*exp(-z^2/2)
!
!   Note: K^{-1} computed via dinvmat from linalgwrap.
! ===========================================================================
MODULE bayesian
  !
  use constants, only : dp
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
    k = sv * exp(-0.5_dp * sqdist / (ls * ls))
    !
  END FUNCTION rbf_kernel
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE gp_update(gp, x_new, y_new)
    !
    ! Add one observation (x_new, y_new) and rebuild K_inv, alpha.
    !
    TYPE(gp_model), intent(inout) :: gp
    real(dp), dimension(:), intent(in) :: x_new
    real(dp), intent(in)               :: y_new
    !
    integer :: n_old, n_new, ii, jj
    real(dp), allocatable :: x_tmp(:,:), y_tmp(:)
    real(dp), allocatable :: Kmat(:,:)
    !
    n_old = gp%n_train
    n_new = n_old + 1
    !
    ! Grow training data arrays
    allocate(x_tmp(gp%n_params, n_new), y_tmp(n_new))
    if (n_old > 0) then
      x_tmp(:, 1:n_old) = gp%x_train(:, 1:n_old)
      y_tmp(1:n_old)    = gp%y_train(1:n_old)
    endif
    x_tmp(:, n_new) = x_new
    y_tmp(n_new)    = y_new
    !
    if (allocated(gp%x_train)) deallocate(gp%x_train)
    if (allocated(gp%y_train)) deallocate(gp%y_train)
    allocate(gp%x_train(gp%n_params, n_new))
    allocate(gp%y_train(n_new))
    gp%x_train = x_tmp
    gp%y_train = y_tmp
    deallocate(x_tmp, y_tmp)
    gp%n_train = n_new
    !
    ! Build kernel matrix K + sigma_n^2 * I
    allocate(Kmat(n_new, n_new))
    do ii = 1, n_new
      do jj = 1, n_new
        Kmat(ii, jj) = rbf_kernel(gp%x_train(:,ii), gp%x_train(:,jj), &
                                   gp%n_params, gp%length_scale, gp%signal_var)
      enddo
      Kmat(ii, ii) = Kmat(ii, ii) + gp%noise_var
    enddo
    !
    ! Invert kernel matrix
    if (allocated(gp%K_inv)) deallocate(gp%K_inv)
    if (allocated(gp%alpha)) deallocate(gp%alpha)
    allocate(gp%K_inv(n_new, n_new))
    allocate(gp%alpha(n_new))
    gp%K_inv = Kmat
    call invmat(gp%K_inv, n_new)  ! in-place inversion via dinvmat
    !
    ! alpha = K_inv * y
    gp%alpha = matmul(gp%K_inv, gp%y_train)
    !
    deallocate(Kmat)
    !
  END SUBROUTINE gp_update
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE gp_predict(gp, x_test, mu_out, sigma_out)
    !
    ! Predict GP mean and standard deviation at x_test.
    !
    TYPE(gp_model), intent(in) :: gp
    real(dp), dimension(:), intent(in) :: x_test
    real(dp), intent(out) :: mu_out, sigma_out
    !
    integer :: ii
    real(dp) :: k_star_star
    real(dp), allocatable :: k_star(:)
    real(dp) :: var_out
    !
    if (gp%n_train == 0) then
      mu_out    = 0.0_dp
      sigma_out = sqrt(gp%signal_var)
      return
    endif
    !
    allocate(k_star(gp%n_train))
    do ii = 1, gp%n_train
      k_star(ii) = rbf_kernel(x_test, gp%x_train(:,ii), &
                               gp%n_params, gp%length_scale, gp%signal_var)
    enddo
    !
    k_star_star = rbf_kernel(x_test, x_test, gp%n_params, gp%length_scale, gp%signal_var)
    !
    ! mu = k* . alpha
    mu_out = dot_product(k_star, gp%alpha)
    !
    ! sigma^2 = k** - k* . K_inv . k*
    var_out = k_star_star - dot_product(k_star, matmul(gp%K_inv, k_star))
    if (var_out < 0.0_dp) var_out = 0.0_dp
    sigma_out = sqrt(var_out)
    !
    deallocate(k_star)
    !
  END SUBROUTINE gp_predict
  !
  ! ---------------------------------------------------------------------------
  FUNCTION normal_cdf(x) RESULT(p)
    !
    ! Standard normal CDF via complementary error function:
    !   Phi(x) = 0.5 * erfc(-x / sqrt(2))
    !
    real(dp), intent(in) :: x
    real(dp) :: p
    !
    real(dp), parameter :: sqrt2 = 1.41421356237309504_dp
    p = 0.5_dp * erfc(-x / sqrt2)
    !
  END FUNCTION normal_cdf
  !
  ! ---------------------------------------------------------------------------
  FUNCTION normal_pdf(x) RESULT(phi)
    !
    ! Standard normal PDF:
    !   phi(x) = (1/sqrt(2*pi)) * exp(-x^2/2)
    !
    use constants, only : twopi
    !
    real(dp), intent(in) :: x
    real(dp) :: phi
    !
    phi = (1.0_dp / sqrt(twopi)) * exp(-0.5_dp * x * x)
    !
  END FUNCTION normal_pdf
  !
  ! ---------------------------------------------------------------------------
  FUNCTION expected_improvement(mu, sigma, y_best) RESULT(ei)
    !
    ! Expected Improvement acquisition function:
    !   EI(x) = (y_best - mu) * Phi(z) + sigma * phi(z)
    !   z = (y_best - mu) / sigma
    ! where we MINIMISE the objective, so y_best = min observed value.
    !
    real(dp), intent(in) :: mu, sigma, y_best
    real(dp) :: ei
    !
    real(dp) :: z
    !
    if (sigma < 1.0d-10) then
      ei = 0.0_dp
      return
    endif
    !
    z = (y_best - mu) / sigma
    ei = (y_best - mu) * normal_cdf(z) + sigma * normal_pdf(z)
    if (ei < 0.0_dp) ei = 0.0_dp
    !
  END FUNCTION expected_improvement
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE latin_hypercube(samples, n_samples, n_params, bounds)
    !
    ! Generate n_samples points via Latin Hypercube Sampling in [bounds(1,p), bounds(2,p)]
    !
    integer,  intent(in)  :: n_samples, n_params
    real(dp), dimension(2, n_params), intent(in) :: bounds
    real(dp), dimension(n_params, n_samples), intent(out) :: samples
    !
    integer :: ip, is, iswap
    real(dp) :: u
    integer, allocatable :: perm(:)
    !
    allocate(perm(n_samples))
    !
    do ip = 1, n_params
      ! Initialize permutation identity
      do is = 1, n_samples
        perm(is) = is
      enddo
      ! Fisher-Yates random permutation
      do is = n_samples, 2, -1
        call random_number(u)
        iswap = int(u * is) + 1
        if (iswap > is) iswap = is
        call swap_int(perm(is), perm(iswap))
      enddo
      ! Assign stratified sample: stratum (perm(is)-1)/n_samples + rand/n_samples
      do is = 1, n_samples
        call random_number(u)
        samples(ip, is) = bounds(1, ip) + (bounds(2, ip) - bounds(1, ip)) * &
                           (real(perm(is)-1, dp) + u) / real(n_samples, dp)
      enddo
    enddo
    !
    deallocate(perm)
    !
  END SUBROUTINE latin_hypercube
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
    ! Main Bayesian Optimization loop.
    !
    ! Inputs:
    !   objective_func : callback f(params, n) -> real, function to minimize
    !   bounds         : (2, n_params) array of lower/upper bounds per parameter
    !   n_params       : dimension of parameter space
    !   n_iter         : number of Bayesian iterations after initialization
    ! Output:
    !   result         : (n_params) best parameters found
    !
    ! Algorithm:
    !   1. n_init = max(5, n_params) Latin Hypercube initial samples
    !   2. For iter = 1..n_iter:
    !      a. Predict GP over N_cand = max(1000, 50*n_params) random candidates
    !      b. Select x_next = argmax EI
    !      c. Evaluate objective_func(x_next)
    !      d. Update GP with new observation
    !   3. Return x with minimum observed f(x)
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
    integer :: n_init, n_cand, ii, jj, best_idx
    real(dp) :: y_val, y_best, ei_val, ei_best
    real(dp) :: mu_pred, sigma_pred
    real(dp), allocatable :: x_init(:,:)  ! (n_params, n_init)
    real(dp), allocatable :: x_cand(:,:)  ! (n_params, n_cand)
    real(dp), allocatable :: x_next(:)    ! (n_params)
    real(dp), allocatable :: y_all(:)     ! all observed values
    real(dp), allocatable :: x_all(:,:)   ! all observed x
    real(dp) :: u
    !
    ! Hyperparameters: length_scale scaled to average bound range,
    ! signal_var=1, noise_var=1e-6 (small jitter for stability)
    real(dp) :: ls_init
    ls_init = 0.0_dp
    do ii = 1, n_params
      ls_init = ls_init + (bounds(2,ii) - bounds(1,ii))
    enddo
    ls_init = ls_init / real(n_params, dp) * 0.5_dp
    if (ls_init < 1.0d-6) ls_init = 1.0d-6
    !
    call gp_init(gp, n_params, ls_init, 1.0_dp, 1.0d-6)
    !
    n_init = max(5, n_params)
    n_cand = max(1000, 50*n_params)
    !
    allocate(x_init(n_params, n_init))
    allocate(x_next(n_params))
    allocate(x_all(n_params, n_init + n_iter))
    allocate(y_all(n_init + n_iter))
    !
    ! --- Phase 1: Latin Hypercube initialization ---
    call latin_hypercube(x_init, n_init, n_params, bounds)
    !
    y_best  = 1.0d30
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
    ! --- Phase 2: Bayesian optimization loop ---
    allocate(x_cand(n_params, n_cand))
    !
    do ii = 1, n_iter
      !
      ! Sample N_cand random candidates uniformly within bounds
      do jj = 1, n_cand
        do ii = 1, n_params
          call random_number(u)
          x_cand(ii, jj) = bounds(1, ii) + &
                            (bounds(2, ii) - bounds(1, ii)) * u
        enddo
      enddo
      !
      ! Find x_next = argmax EI over candidates
      ei_best  = -1.0_dp
      x_next   = x_cand(:, 1)
      do jj = 1, n_cand
        call gp_predict(gp, x_cand(:,jj), mu_pred, sigma_pred)
        ei_val = expected_improvement(mu_pred, sigma_pred, y_best)
        if (ei_val > ei_best) then
          ei_best = ei_val
          x_next  = x_cand(:, jj)
        endif
      enddo
      !
      ! Evaluate objective at x_next
      y_val = objective_func(x_next, n_params)
      call gp_update(gp, x_next, y_val)
      x_all(:, n_init+ii) = x_next
      y_all(n_init+ii)    = y_val
      !
      if (y_val < y_best) then
        y_best = y_val
        best_idx = n_init + ii
      endif
      !
      if (mod(ii, 10) == 0 .or. ii == n_iter) then
        write(stdout, '(A,1I4,A,1G14.6,A,1G14.6)') "  # Bayes iter ", ii, &
              "  EI=", ei_best, "  f_best=", y_best
      endif
      !
    enddo
    !
    ! Return best parameters found
    result = x_all(:, best_idx)
    !
    write(stdout, '(A,1G14.6)') "  # Bayesian optimization done. f_best = ", y_best
    !
    deallocate(x_init, x_cand, x_next, x_all, y_all)
    call gp_finalize(gp)
    !
  END SUBROUTINE bayesian_optimize
  !
END MODULE bayesian
