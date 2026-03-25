!
!   wannlog.f90
!
! ===========================================================================
! MODULE wannlog
!   Lightweight timing and logging utility for wannchi programs.
!
!   Usage: call from the main program (wanneff_JS.f90) only.
!   Wraps each computation stage with log_start/log_stop to record elapsed
!   wall-clock time. Intermediate messages can be added with log_msg.
!   At program end, log_print_summary() prints a formatted table to stdout.
!
!   Usage in main program:
!     use wannlog, only: log_init, log_start, log_stop, log_msg, log_print_summary
!     CALL log_init()
!     CALL log_start('downfolding')
!       ... computation ...
!     CALL log_stop('downfolding')
!     CALL log_print_summary()
!
!   Redirect output:  wanneff_js.x > seed_wanneff.log
!
! ===========================================================================
MODULE wannlog
  !
  use constants, only : dp, stdout
  !
  implicit none
  !
  integer, parameter :: WANNLOG_MAX_TIMERS  = 50
  integer, parameter :: WANNLOG_MAX_MSGS    = 200
  integer, parameter :: WANNLOG_LABEL_LEN   = 40
  integer, parameter :: WANNLOG_MSG_LEN     = 200
  !
  character(len=WANNLOG_LABEL_LEN) :: wl_timer_labels(WANNLOG_MAX_TIMERS)
  real(dp)  :: wl_timer_start(WANNLOG_MAX_TIMERS)
  real(dp)  :: wl_timer_elapsed(WANNLOG_MAX_TIMERS)
  integer   :: wl_timer_calls(WANNLOG_MAX_TIMERS)
  integer   :: wl_n_timers = 0
  !
  character(len=WANNLOG_MSG_LEN) :: wl_messages(WANNLOG_MAX_MSGS)
  integer   :: wl_n_messages = 0
  !
  real(dp)  :: wl_wall_start  ! absolute start time (seconds)
  !
CONTAINS
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE log_init()
    !
    ! Reset all timers and messages, record program start time.
    !
    wl_n_timers   = 0
    wl_n_messages = 0
    CALL cpu_time(wl_wall_start)
    !
    write(stdout, '(A)') ' '
    write(stdout, '(A)') ' ============================================================'
    write(stdout, '(A)') '  wannlog: timing and logging initialized'
    write(stdout, '(A)') ' ============================================================'
    !
  END SUBROUTINE log_init
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE log_start(label)
    !
    ! Start a named timer. Creates a new entry or restarts existing.
    !
    character(len=*), intent(in) :: label
    !
    integer :: ii
    real(dp) :: t_now
    !
    CALL cpu_time(t_now)
    !
    ! Search for existing timer
    do ii = 1, wl_n_timers
      if (trim(wl_timer_labels(ii)) == trim(label)) then
        wl_timer_start(ii) = t_now
        wl_timer_calls(ii) = wl_timer_calls(ii) + 1
        return
      endif
    enddo
    !
    ! New timer
    if (wl_n_timers < WANNLOG_MAX_TIMERS) then
      wl_n_timers = wl_n_timers + 1
      wl_timer_labels(wl_n_timers)  = label
      wl_timer_start(wl_n_timers)   = t_now
      wl_timer_elapsed(wl_n_timers) = 0.0_dp
      wl_timer_calls(wl_n_timers)   = 1
    endif
    !
  END SUBROUTINE log_start
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE log_stop(label)
    !
    ! Stop timer and accumulate elapsed time.
    !
    character(len=*), intent(in) :: label
    !
    integer :: ii
    real(dp) :: t_now, dt
    !
    CALL cpu_time(t_now)
    !
    do ii = 1, wl_n_timers
      if (trim(wl_timer_labels(ii)) == trim(label)) then
        dt = t_now - wl_timer_start(ii)
        if (dt < 0.0_dp) dt = 0.0_dp  ! guard against clock wraps
        wl_timer_elapsed(ii) = wl_timer_elapsed(ii) + dt
        return
      endif
    enddo
    !
    ! Timer not found — create and immediately stop it (0 elapsed)
    if (wl_n_timers < WANNLOG_MAX_TIMERS) then
      wl_n_timers = wl_n_timers + 1
      wl_timer_labels(wl_n_timers)  = label
      wl_timer_elapsed(wl_n_timers) = 0.0_dp
      wl_timer_calls(wl_n_timers)   = 0
    endif
    !
  END SUBROUTINE log_stop
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE log_msg(msg)
    !
    ! Add a message to the log buffer.
    !
    character(len=*), intent(in) :: msg
    !
    if (wl_n_messages < WANNLOG_MAX_MSGS) then
      wl_n_messages = wl_n_messages + 1
      wl_messages(wl_n_messages) = trim(msg)
    endif
    !
    ! Echo message immediately to stdout
    write(stdout, '(A,A)') '  [log] ', trim(msg)
    !
  END SUBROUTINE log_msg
  !
  ! ---------------------------------------------------------------------------
  SUBROUTINE log_print_summary()
    !
    ! Print timing table and messages to stdout.
    !
    integer :: ii
    real(dp) :: t_now, total_cpu
    !
    CALL cpu_time(t_now)
    total_cpu = t_now - wl_wall_start
    !
    write(stdout, '(A)') ' '
    write(stdout, '(A)') ' ============================================================'
    write(stdout, '(A)') '  wannlog: timing summary'
    write(stdout, '(A)') ' ============================================================'
    write(stdout, '(A,F10.3,A)') '  Total CPU time:  ', total_cpu, ' s'
    write(stdout, '(A)') ' '
    write(stdout, '(A)') '  Stage                                    Calls     CPU (s)  %Total'
    write(stdout, '(A)') '  ---------------------------------------------------------------'
    !
    do ii = 1, wl_n_timers
      write(stdout, '(A,A40,I6,F12.3,F8.1)') '  ', &
        trim(wl_timer_labels(ii)), wl_timer_calls(ii), &
        wl_timer_elapsed(ii), &
        merge(100.0_dp * wl_timer_elapsed(ii) / total_cpu, 0.0_dp, total_cpu > 1.0d-10)
    enddo
    !
    write(stdout, '(A)') '  ---------------------------------------------------------------'
    !
    if (wl_n_messages > 0) then
      write(stdout, '(A)') ' '
      write(stdout, '(A)') '  Messages:'
      do ii = 1, wl_n_messages
        write(stdout, '(A,A)') '    ', trim(wl_messages(ii))
      enddo
    endif
    !
    write(stdout, '(A)') ' ============================================================'
    !
  END SUBROUTINE log_print_summary
  !
END MODULE wannlog
