# Job driver used for the 2026-11-09 Leonardo runs (record)

- `tick.sh`: one tick on the login node (copied there as `~/qkd_driver_tick.sh`). It reads squeue/sacct, submits
  missing QML array tasks (qml-surrogate clone), resubmits `da_boost.sbatch` until all 192 DA chunks exist, then
  submits `da_merge.sbatch`; state and a log in `$WORK/driver_state`. Paths and the account are hard-coded for that run.
- `loop.sh`: Mac side; runs one tick every 120 s over short SSH connections with keepalives, so a dropped connection
  only skips a tick.
- `driver_log_2026-11-09.txt`: what the driver submitted. The first merge (63982538) ran out of memory at 123 GB and
  was resubmitted by hand on a whole node (63982773).
