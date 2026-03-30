#ifndef __CMA_ES
#define __CMA_ES

! ===========================================================================
! MODULE cma_es
!   CMA-ES: Covariance Matrix Adaptation Evolution Strategy
!
!   APPLICABILITY:
!     - High-dimensional continuous optimization (n_params >= 20)
!     - Non-convex, multi-modal objective functions
!     - Black-box optimization without gradient information
!     - When GP-based BO is too slow or doesn't converge
!
!   NOT SUITABLE FOR:
!     - Very low dimensional problems (n_params < 10) - overkill
!     - Discrete/categorical parameters
!     - When sample efficiency is critical (population-based, many evaluations)
!
!   Algorithm (simplified axis-parallel version):
!     1. Initialize mean at center of bounds, sigma ~ 30% of range
!     2. Sample lambda offspring from axis-aligned Gaussian
!     3. Evaluate fitness, select top mu parents
!     4. Update mean, evolution paths (pc, ps)
!     5. Update step size sigma based on ps norm vs expected
!     6. Update diagonal covariance via weighted variance of parents
!     7. Repeat until convergence or max iterations
!
!   Key parameters:
!     - lambda: population size (20-124 for high-dim, depends on n_params)
!     - mu: number of parents (lambda/2)
!     - sigma: step size (adapts during evolution)
!
!   References:
!     - Hansen & Ostermeier, Evolutionary Computation 9(2), 2001
!     - Hansen et al., JMLR 20, 2019 (review)
! ===========================================================================

module cma_es

use constants, only : stdout, dp

implicit none

private
public :: cmaes_optimize

contains

  ! ===========================================================================
  SUBROUTINE cmaes_optimize(objective_func, bounds, n_params, result, n_iter)
    !
    ! CMA-ES optimizer entry point.
    !
    ! Input:
    !   objective_func - external function f(params, n) -> real value (minimize)
    !   bounds(2, n_params) - lower(1,:) and upper(2,:) bounds
    !   n_params - number of parameters to optimize
    !   n_iter - maximum number of iterations
    !
    ! Output:
    !   result(n_params) - best solution found
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
    ! CMA-ES internal state
    real(dp), allocatable :: mean(:)      ! mean of distribution
    real(dp), allocatable :: sigma(:)     ! step size (per dimension)
    real(dp), allocatable :: cov(:,:)     ! covariance matrix (diagonal only in axis-parallel version)
    real(dp), allocatable :: pc(:), ps(:) ! evolution paths
    real(dp), allocatable :: work(:)      ! workspace
    real(dp), allocatable :: x_pop(:), fitness(:)
    real(dp) :: cc, c1, cmu, damps, chiN
    integer :: lambda, mu, nfe
    integer :: ii, jj, k, best_idx
    real(dp) :: y_best, y_worst, u
    real(dp), parameter :: TOL = 1.0d-8
    !
    ! CMA-ES parameters (standard defaults)
    ! Use larger lambda for high-dimensional problems to maintain diversity
    if (n_params <= 20) then
      lambda = max(20, 4 + floor(3.0_dp * log(real(n_params, dp))))
    else if (n_params <= 100) then
      lambda = max(50, n_params / 2)  ! larger population for high-dim
    else
      lambda = max(100, n_params)  ! very large population for very high-dim
    endif
    mu = lambda / 2                                               ! parents
    !
    ! Allocation
    allocate(mean(n_params), sigma(n_params))
    allocate(cov(n_params, n_params), pc(n_params), ps(n_params))
    allocate(work(n_params), x_pop(n_params * lambda), fitness(lambda))
    !
    ! Initialize
    cc = 4.0_dp / (real(n_params, dp) + 4.0_dp)
    c1 = 2.0_dp / ((real(n_params, dp) + 1.3_dp)**2 + mu)
    cmu = min(1.0_dp - c1, 2.0_dp * (mu - 2.0_dp + 1.0_dp/mu) / &
                  ((real(n_params, dp) + 2.0_dp)**2 + mu))
    damps = 1.0_dp + 2.0_dp * max(0.0_dp, sqrt(real(mu-1,dp)) - 1.0_dp) + cc
    chiN = sqrt(real(n_params, dp)) * (1.0_dp - 1.0_dp/(4.0_dp*n_params) + 1.0_dp/(21.0_dp*n_params**2))
    !
    ! Initialize mean at center of bounds
    do ii = 1, n_params
      mean(ii) = 0.5_dp * (bounds(1, ii) + bounds(2, ii))
    enddo
    sigma = 0.3_dp * (bounds(2, :) - bounds(1, :))  ! 30% of range
    cov = 0.0_dp
    do ii = 1, n_params
      cov(ii, ii) = sigma(ii)**2
    enddo
    pc = 0.0_dp
    ps = 0.0_dp
    !
    y_best = 1.0d30
    nfe = 0
    best_idx = 1
    !
    write(stdout, '(A,I6,A)') "  CMA-ES: n_params=", n_params, " lambda=", lambda
    !
    ! CMA-ES main loop
    do ii = 1, n_iter
      !
      ! Sample lambda offspring
      do k = 1, lambda
        do jj = 1, n_params
          call random_number(u)
          x_pop((k-1)*n_params + jj) = mean(jj) + sigma(jj) * (u - 0.5_dp) * sqrt(12.0_dp)
          ! Clip to bounds
          x_pop((k-1)*n_params + jj) = max(bounds(1,jj), min(bounds(2,jj), &
                                           x_pop((k-1)*n_params + jj)))
        enddo
      enddo
      !
      ! Evaluate fitness
      do k = 1, lambda
        fitness(k) = objective_func(x_pop((k-1)*n_params+1 : k*n_params), n_params)
        nfe = nfe + 1
      enddo
      !
      ! Sort by fitness (ascending) - simple bubble sort
      do jj = 1, lambda-1
        do k = jj+1, lambda
          if (fitness(k) < fitness(jj)) then
            u = fitness(jj); fitness(jj) = fitness(k); fitness(k) = u
            work = x_pop((jj-1)*n_params+1 : jj*n_params)
            x_pop((jj-1)*n_params+1 : jj*n_params) = x_pop((k-1)*n_params+1 : k*n_params)
            x_pop((k-1)*n_params+1 : k*n_params) = work
          endif
        enddo
      enddo
      !
      if (fitness(1) < y_best) then
        y_best = fitness(1)
        best_idx = 1
      endif
      !
      ! Update mean (weighted combination of top mu)
      work = 0.0_dp
      do k = 1, mu
        work = work + x_pop((k-1)*n_params+1 : k*n_params)
      enddo
      work = work / real(mu, dp)
      !
      ! Update evolution paths
      pc = (1.0_dp - cc) * pc + sqrt(cc * (2.0_dp - cc)) * (work - mean) / sigma
      ps = (1.0_dp - 1.0_dp/damps) * ps + &
           sqrt(cc * (2.0_dp - cc)) * sqrt(real(mu, dp)) * (work - mean) / sigma
      !
      ! Update covariance matrix
      cov = (1.0_dp - c1 - cmu) * cov + &
            c1 * (outer_product(pc, pc) + (1.0_dp - 1.0_dp/(4.0_dp*n_params))*cov) + &
            cmu * (1.0_dp/mu) * outer_diag_sum(x_pop, n_params, mu, bounds)
      !
      ! Update step size (axis-parallel only for simplicity)
      sigma = sigma * exp((norm2(ps) - chiN) / (sqrt(real(n_params,dp)) * damps))
      sigma = max(0.01_dp * (bounds(2,:) - bounds(1,:)), &
                  min(0.5_dp * (bounds(2,:) - bounds(1,:)), sigma))
      !
      mean = work
      !
      if (mod(ii, 10) == 0 .or. ii == n_iter) then
        write(stdout, '(A,I5,A,G14.6)') "  CMA-ES iter ", ii, "  f_best=", y_best
      endif
      !
      ! Check convergence
      if (y_best < TOL) exit
      !
    enddo
    !
    result = x_pop(1:n_params)
    write(stdout, '(A,G14.6,A,I8)') "  CMA-ES done. f_best=", y_best, "  nfe=", nfe
    !
    deallocate(mean, sigma, cov, pc, ps, work, x_pop, fitness)
    !
  CONTAINS
    !
    pure function outer_product(a, b) result(c)
      real(dp), intent(in) :: a(:), b(:)
      real(dp) :: c(size(a), size(b))
      integer :: i, j
      do i = 1, size(a); do j = 1, size(b); c(i,j) = a(i) * b(j); enddo; enddo
    end function outer_product
    !
    function outer_diag_sum(x_arr, n, m, bnds) result(c)
      integer, intent(in) :: n, m
      real(dp), intent(in) :: x_arr(n*m), bnds(2,n)
      real(dp) :: c(n,n)
      integer :: k, jj
      c = 0.0_dp
      do k = 1, m
        do jj = 1, n
          c(jj,jj) = c(jj,jj) + (x_arr((k-1)*n+jj) - 0.5_dp*(bnds(1,jj)+bnds(2,jj)))**2
        enddo
      enddo
    end function outer_diag_sum
    !
  END SUBROUTINE cmaes_optimize

#endif

end module cma_es
