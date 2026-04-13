#ifndef __CMA_ES
#define __CMA_ES

! ===========================================================================
! MODULE cma_es
!   CMA-ES: Covariance Matrix Adaptation Evolution Strategy (Standard)
!
!   Standard algorithm: Hansen & Ostermeier, Evolutionary Computation 9(2), 159 (2001)
!   With: rank-mu update, evolution paths pc (for C) and ps (for sigma),
!         full covariance matrix, eigendecomposition each generation.
!
!   Sampling: x ~ N(m, sigma^2 * C)
!     via eigendecomposition C = B * D^2 * B^T
!     x = m + sigma * B * D * z,  z ~ N(0, I)
!
!   Covariance update (full rank):
!     C = (1-c1-cmu)*C + c1*pc*pc^T + cmu*sum_{i=1..mu} w_i*y_i*y_i^T
!
!   References:
!     - Hansen & Ostermeier, EC 9(2), 159 (2001)
!     - Hansen et al., JMLR 20, 1 (2019) — review
! ===========================================================================

module cma_es

  use constants, only : stdout, dp
  use para,      only : distribute_calc, first_idx, last_idx, &
                        para_merge_real, inode, nnode

  implicit none

  private
  public :: cmaes_optimize

contains

  ! ===========================================================================
  SUBROUTINE cmaes_optimize(objective_func, bounds, n_params, result, n_iter, sigma_init)
    !
    ! Standard CMA-ES optimizer entry point.
    !
    ! Input:
    !   objective_func : external function f(params, n) -> scalar (minimize)
    !   bounds(2, n_params) : lower(1,:) and upper(2,:)
    !   n_params : dimension of parameter space
    !   n_iter : maximum number of iterations
    !   sigma_init : optional initial step-size (default 0.02)
    !
    ! Output:
    !   result(n_params) : best solution found
    !
    real(dp), intent(in), optional :: sigma_init
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
    real(dp), dimension(2, n_params), intent(in) :: bounds
    real(dp), dimension(n_params),    intent(out) :: result
    !
    ! ---- CMA-ES state ----
    real(dp) :: xmean(n_params)              ! mean of distribution
    real(dp) :: sigma                         ! global step-size (scalar)
    real(dp) :: C(n_params, n_params)       ! covariance matrix (symmetric)
    real(dp) :: B(n_params, n_params)       ! eigenvectors of C
    real(dp) :: D(n_params)                 ! sqrt(eigenvalues) of C
    real(dp) :: pc(n_params)                ! evolution path for C
    real(dp) :: ps(n_params)                ! evolution path for sigma
    !
    ! ---- Workspace ----
    real(dp), allocatable :: x_pop(:,:)     ! (n_params, lambda) population
    real(dp) :: fitness(200)                  ! up to 200 (max lambda)
    real(dp) :: y_sel(n_params, 200)        ! (x_i - mean) / sigma for selected (max mu)
    real(dp) :: y_w(n_params)               ! weighted mean of selected (centered)
    real(dp) :: z_tmp(n_params)             ! N(0,1) sample
    real(dp) :: work(n_params)              ! general workspace
    real(dp) :: eigen_work(max(1, 3*n_params)) ! workspace for dsyev
    integer  :: lwork
    real(dp) :: y_best, u, sqrt_term, norm_ps
    real(dp) :: w(200)               ! recombination weights (max mu=100)
    real(dp) :: mu_eff               ! variance-effectiveness of mu
    real(dp) :: cc, cs, c1, cmu, damps, chiN
    real(dp) :: sigma_arg               ! overflow-protected sigma update argument
    integer  :: nfe, gen, kk, k, j, best_idx, info, ipop, lambda, mu
    real(dp) :: best_individual(n_params)  ! saved best individual
    real(dp), parameter :: TOL = 1.0d-8
    !
    real(dp), external :: dnrm2
    ! =========================================================================
    ! (1) Initialize strategy parameters (Hansen 2019, weighted recombination)
    ! =========================================================================
    if (n_params <= 20) then
      lambda = max(20, 4 + int(3.0_dp * log(real(n_params, dp))))
    else if (n_params <= 100) then
      lambda = max(50, int(real(n_params, dp) / 2.0_dp))
    else
      lambda = max(100, n_params)
    endif
    mu = lambda / 2
    !
    ! Equal weights (mu_eff = mu for equal weights)
    mu_eff = real(mu, dp)
    do j = 1, mu
      w(j) = 1.0_dp / real(mu, dp)
    enddo
    !
    ! Learning rates (Hansen 2019, symmetric)
    cc   = (4.0_dp + mu_eff) / (real(n_params, dp) + 4.0_dp + 2.0_dp * mu_eff)
    cs   = (mu_eff + 2.0_dp) / (real(n_params, dp) + mu_eff + 5.0_dp)
    c1   = 2.0_dp / ((real(n_params, dp) + 1.3_dp)**2 + mu_eff)
    cmu  = min(1.0_dp - c1, &
               2.0_dp * (mu_eff - 2.0_dp + 1.0_dp/mu_eff) / &
               ((real(n_params, dp) + 2.0_dp)**2 + mu_eff))
    damps = (1.0_dp + 2.0_dp * max(0.0_dp, sqrt(mu_eff) - 1.0_dp)) / cs &
            + 0.0_dp
    chiN = sqrt(real(n_params, dp)) * &
           (1.0_dp - 1.0_dp / (4.0_dp * real(n_params, dp)) &
                    + 1.0_dp / (21.0_dp * real(n_params, dp)**2))
    !
    ! =========================================================================
    ! (2) Initialize state
    ! =========================================================================
    ! Mean at center of bounds
    do j = 1, n_params
      xmean(j) = 0.5_dp * (bounds(1, j) + bounds(2, j))
    enddo
    !
    ! Initial step-size: use provided value or default 0.02
    if (present(sigma_init)) then
      sigma = sigma_init
    else
      sigma = 0.02_dp
    endif
    !
    ! Initial covariance: identity (isotropic)
    C = 0.0_dp
    do j = 1, n_params
      C(j, j) = 1.0_dp
    enddo
    B = C
    D = 1.0_dp
    pc = 0.0_dp
    ps = 0.0_dp
    !
    ! Allocate population
    allocate(x_pop(n_params, lambda))
    !
    y_best = 1.0d30
    nfe = 0
    best_idx = 1
    lwork = max(1, 3*n_params)
    best_individual = xmean  ! initial best guess
    !
    write(stdout, '(A,I6,A,I6,A,I6)') &
          "  CMA-ES: n_params=", n_params, " lambda=", lambda, " mu=", mu
    write(stdout, '(A,4(G10.3,A))') &
          "  cs=", cs, "  cc=", cc, "  c1=", c1, "  cmu=", cmu
    write(stdout, '(A,G10.3,A,G10.3)') &
          "  damps=", damps, "  chiN=", chiN
    !
    ! =========================================================================
    ! (3) Main CMA-ES loop
    ! =========================================================================
    main_loop: do gen = 1, n_iter
      !
      ! ---- (3a) Sample lambda individuals from N(m, sigma^2 * C) ----
      ! x_k = m + sigma * B * D * z_k,   z_k ~ N(0, I)
      if (inode == 0) then
        do k = 1, lambda
          !
          ! Sample z_k ~ N(0, I) via Box-Muller (polar form)
          do j = 1, n_params
            call random_number(u)
            u = max(u, 1.0d-20)
            z_tmp(j) = sqrt(-2.0_dp * log(u))
            call random_number(u)
            z_tmp(j) = z_tmp(j) * cos(6.28318530718_dp * u)
          enddo
          !
          ! y = D * z  (scaling by sqrt(eigenvalue))
          do j = 1, n_params
            work(j) = D(j) * z_tmp(j)
          enddo
          !
          ! x = m + sigma * B * y  via dgemv
          call dgemv('N', n_params, n_params, &
                     sigma, B, n_params, work, 1, &
                     0.0_dp, x_pop(1, k), 1)
          !
          ! Add mean
          do j = 1, n_params
            x_pop(j, k) = xmean(j) + x_pop(j, k)
          enddo
        enddo
        !
        ! Re-sample any out-of-bounds individuals (reflection)
        do k = 1, lambda
          do j = 1, n_params
            if (x_pop(j, k) < bounds(1, j)) then
              x_pop(j, k) = bounds(1, j) + (bounds(1, j) - x_pop(j, k))
              x_pop(j, k) = min(x_pop(j, k), bounds(2, j))
            else if (x_pop(j, k) > bounds(2, j)) then
              x_pop(j, k) = bounds(2, j) - (x_pop(j, k) - bounds(2, j))
              x_pop(j, k) = max(x_pop(j, k), bounds(1, j))
            endif
          enddo
        enddo
      endif
      !
      ! ---- (3b) Distribute fitness evaluations across MPI ranks ----
      call distribute_calc(lambda)
      fitness(1:lambda) = 0.0_dp
      !
      do k = first_idx, last_idx
        fitness(k) = objective_func(x_pop(1, k), n_params)
        nfe = nfe + 1
      enddo
      !
      ! Merge from all ranks
      call para_merge_real(fitness, lambda)
      !
      ! ---- (3c) Sort by fitness (ascending) ----
      do j = 1, lambda - 1
        do k = j + 1, lambda
          if (fitness(k) < fitness(j)) then
            u = fitness(j); fitness(j) = fitness(k); fitness(k) = u
            work = x_pop(:, j)
            x_pop(:, j) = x_pop(:, k)
            x_pop(:, k) = work
          endif
        enddo
      enddo
      !
      if (fitness(1) < y_best) then
        y_best = fitness(1)
        best_idx = 1
        best_individual = x_pop(:, 1)  ! save the best individual
      endif
      !
      ! ---- (3d) Compute centered selected individuals: y_sel = (x_i - m) / sigma ----
      do k = 1, mu
        do j = 1, n_params
          y_sel(j, k) = (x_pop(j, k) - xmean(j)) / sigma
        enddo
      enddo
      !
      ! ---- (3e) Weighted mean of centered selected individuals ----
      y_w = 0.0_dp
      do k = 1, mu
        do j = 1, n_params
          y_w(j) = y_w(j) + w(k) * y_sel(j, k)
        enddo
      enddo
      !
      ! ---- (3f) Update evolution path pc (for covariance) ----
      ! pc = (1 - cc) * pc + sqrt(cc * (2 - cc) * mu_eff) * y_w
      sqrt_term = sqrt(cc * (2.0_dp - cc) * mu_eff)
      do j = 1, n_params
        pc(j) = (1.0_dp - cc) * pc(j) + sqrt_term * y_w(j)
      enddo
      !
      ! ---- (3g) Update evolution path ps (for sigma) ----
      ! ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mu_eff) * B * D^{-1} * y_w
      do j = 1, n_params
        work(j) = y_w(j) / max(D(j), 1.0d-10)
      enddo
      call dgemv('N', n_params, n_params, &
                 1.0_dp, B, n_params, work, 1, &
                 0.0_dp, y_w, 1)
      sqrt_term = sqrt(cs * (2.0_dp - cs) * mu_eff)
      do j = 1, n_params
        ps(j) = (1.0_dp - cs) * ps(j) + sqrt_term * y_w(j)
      enddo
      norm_ps = dnrm2(n_params, ps, 1)
      !
      ! ---- (3h) Update covariance matrix C ----
      do j = 1, n_params
        do k = 1, n_params
          C(j, k) = (1.0_dp - c1 - cmu) * C(j, k)
        enddo
      enddo
      call dger(n_params, n_params, c1, pc, 1, pc, 1, C, n_params)
      do k = 1, mu
        call dger(n_params, n_params, cmu * w(k), y_sel(1, k), 1, y_sel(1, k), 1, C, n_params)
      enddo
      !
      ! ---- (3i) Ensure C is symmetric and positive definite ----
      do j = 1, n_params
        C(j, j) = max(C(j, j), 1.0d-8)
        do k = j + 1, n_params
          C(j, k) = 0.5_dp * (C(j, k) + C(k, j))
          C(k, j) = C(j, k)
        enddo
      enddo
      !
      ! ---- (3j) Eigendecomposition: C = B * D^2 * B^T ----
      call dsyev('V', 'U', n_params, C, n_params, D, eigen_work, lwork, info)
      if (info /= 0) then
        write(stdout, '(A,I5)') "  CMA-ES WARNING: dsyev failed, info=", info
        D = 1.0_dp
        C = 0.0_dp
        do j = 1, n_params; C(j, j) = 1.0_dp; enddo
      endif
      B = C
      D = sqrt(max(D, 0.0_dp))
      !
      ! ---- (3k) Update step-size sigma ----
      sigma_arg = ((norm_ps / chiN) - 1.0_dp) * cs / damps
      sigma_arg = max(-2.0_dp, min(sigma_arg, 2.0_dp)) ! limit growth: sigma changes by at most exp(2)≈7
      sigma = sigma * exp(sigma_arg)
      sigma = max(1.0d-10, min(sigma, 1.0d4))
      !
      ! ---- (3l) Update mean ----
      do j = 1, n_params
        xmean(j) = xmean(j) + sigma * pc(j)
      enddo
      ! Soft bound: keep mean within bounds
      do j = 1, n_params
        xmean(j) = max(bounds(1, j), min(bounds(2, j), xmean(j)))
      enddo
      !
      ! ---- Progress output (every 10 generations) ----
      if (mod(gen, 10) == 0 .or. gen == n_iter) then
        write(stdout, '(A,I5,A,G14.6,A,G14.6,A,G14.6)') &
              "  CMA-ES gen ", gen, "  f_best=", y_best, &
              "  sigma=", sigma, "  ||ps||=", norm_ps
      endif
      !
      ! ---- Convergence check ----
      if (y_best < TOL) exit main_loop
      !
    enddo main_loop
    !
    ! Return best individual
    result = best_individual
    write(stdout, '(A,G14.6,A,I8)') &
          "  CMA-ES done. f_best=", y_best, "  nfe=", nfe
    !
    deallocate(x_pop)
    !
  END SUBROUTINE cmaes_optimize

#endif

end module cma_es
