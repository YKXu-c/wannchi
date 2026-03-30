!
! test_optimization.f90 — Unit test for gp_bo and cma_es modules
!
! Optimizes the same 8D function as Python Test 2:
!   f(x) = (x1-2)^2 + (x2+1)^2 + sum(sin(xi), i=3..8)
! True minimum: f ≈ -6.0 at x ≈ (2, -1, -pi/2, ..., -pi/2)
!
! PASS criterion: f_opt < 0.5
!
PROGRAM test_optimization
  !
  use constants, only : dp, stdout
  use gp_bo,     only : bayesian_optimize
  !
  implicit none
  !
  integer, parameter :: NDIM = 8
  integer, parameter :: NITER = 50
  real(dp) :: bounds(2, NDIM)
  real(dp) :: result(NDIM)
  real(dp) :: f_opt
  integer  :: ii
  logical  :: passed
  !
  write(stdout, '(A)') "============================================"
  write(stdout, '(A)') "TEST: Bayesian optimization (8D function)"
  write(stdout, '(A)') "============================================"
  write(stdout, '(A)') "  f(x) = (x1-2)^2 + (x2+1)^2 + sum(sin(xi))"
  write(stdout, '(A)') "  True minimum: f ≈ -6.0"
  write(stdout, '(A)') ""
  !
  ! Set bounds: [-3, 3] for all dimensions
  do ii = 1, NDIM
    bounds(1, ii) = -3.0_dp
    bounds(2, ii) =  3.0_dp
  enddo
  !
  ! Run Bayesian optimization
  call bayesian_optimize(objective_8d, bounds, NDIM, result, NITER)
  !
  ! Evaluate at optimum
  f_opt = objective_8d(result, NDIM)
  !
  write(stdout, '(A)') ""
  write(stdout, '(A)') "--- Results ---"
  write(stdout, '(A,F12.6)') "  f_opt = ", f_opt
  write(stdout, '(A,4F10.4)') "  x_opt(1:4) = ", result(1:4)
  write(stdout, '(A,4F10.4)') "  x_opt(5:8) = ", result(5:8)
  !
  passed = (f_opt < 0.5_dp)
  !
  if (passed) then
    write(stdout, '(A)') "  RESULT: PASS"
  else
    write(stdout, '(A,F12.6,A)') "  RESULT: FAIL (f_opt = ", f_opt, " > 0.5)"
  endif
  !
CONTAINS
  !
  function objective_8d(params, n) result(val)
    use constants, only : dp
    integer,  intent(in) :: n
    real(dp), dimension(n), intent(in) :: params
    real(dp) :: val
    integer  :: jj
    !
    val = (params(1) - 2.0_dp)**2 + (params(2) + 1.0_dp)**2
    do jj = 3, n
      val = val + sin(params(jj))
    enddo
    !
  end function objective_8d
  !
END PROGRAM test_optimization
