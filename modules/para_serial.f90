!
! para_serial.f90 — Serial stub for para module (no MPI dependency)
!
! Drop-in replacement for para.f90 when compiling without MPI.
! All sync/merge/distribute/collect routines are no-ops.
!
MODULE para
  !
  use constants, only : dp, stdout
  !
  implicit none
  !
  integer :: inode = 0
  integer :: nnode = 1
  integer :: first_idx, last_idx
  integer, dimension(:, :), allocatable :: map
  !
  interface para_sync0
    module procedure para_sync_int0, para_sync_real0
  end interface
  !
  interface para_merge0
    module procedure para_merge_cmplx0, para_merge_real0
  end interface
  !
CONTAINS
  !
SUBROUTINE distribute_calc(nidx)
  implicit none
  integer :: nidx
  first_idx = 1
  last_idx  = nidx
END SUBROUTINE

SUBROUTINE init_para(codename)
  implicit none
  character(*) :: codename
  inode = 0
  nnode = 1
  write(stdout, *) trim(codename)//" serial ..."
END SUBROUTINE

SUBROUTINE para_barrier()
  ! no-op
END SUBROUTINE

SUBROUTINE para_sync_logical(dat)
  implicit none
  logical :: dat
  ! no-op
END SUBROUTINE

SUBROUTINE para_sync_int0(dat)
  implicit none
  integer :: dat
  ! no-op
END SUBROUTINE

SUBROUTINE para_sync_int(dat, dat_size)
  implicit none
  integer :: dat(*)
  integer :: dat_size
  ! no-op
END SUBROUTINE

SUBROUTINE para_sync_real(dat, dat_size)
  implicit none
  real(dp) :: dat(*)
  integer  :: dat_size
  ! no-op
END SUBROUTINE

SUBROUTINE para_sync_real0(dat)
  implicit none
  real(dp) :: dat
  ! no-op
END SUBROUTINE

SUBROUTINE para_sync_cmplx(dat, dat_size)
  implicit none
  complex(dp) :: dat(*)
  integer :: dat_size
  ! no-op
END SUBROUTINE

SUBROUTINE para_merge_int(dat, dat_size)
  implicit none
  integer :: dat(*)
  integer :: dat_size
  ! no-op
END SUBROUTINE

SUBROUTINE para_merge_real0(dat)
  implicit none
  real(dp) :: dat
  ! no-op
END SUBROUTINE

SUBROUTINE para_merge_real(dat, dat_size)
  implicit none
  real(dp) :: dat(*)
  integer :: dat_size
  ! no-op
END SUBROUTINE

SUBROUTINE para_merge_cmplx0(dat)
  implicit none
  complex(dp) :: dat
  ! no-op
END SUBROUTINE

SUBROUTINE para_merge_cmplx(dat, dat_size)
  implicit none
  complex(dp) :: dat(*)
  integer :: dat_size
  ! no-op
END SUBROUTINE

SUBROUTINE para_collect_int(fulldat, dat)
  implicit none
  integer :: fulldat(*)
  complex(dp) :: dat(*)
  ! no-op
END SUBROUTINE

SUBROUTINE para_collect_cmplx(fulldat, dat, blk_size)
  implicit none
  complex(dp) :: fulldat(*)
  complex(dp) :: dat(*)
  integer :: blk_size
  ! no-op
END SUBROUTINE

SUBROUTINE para_distribute_cmplx(fulldat, dat, blk_size)
  implicit none
  complex(dp) :: fulldat(*)
  complex(dp) :: dat(*)
  integer :: blk_size
  ! no-op
END SUBROUTINE

SUBROUTINE finalize_para
  ! no-op
END SUBROUTINE

END MODULE
