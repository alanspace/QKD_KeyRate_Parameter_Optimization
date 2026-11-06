import os
import time
import subprocess
import signal
import sys

# CONFIGURATION
# Note: User specified 88478 but that process was killed. 
# In a real run, the user would provide the new PID or we'd find it.
LOG_FILE = "tier3_parallel.log"  # The log file Tier 3 writes to
TIMEOUT_SECONDS = 300             # 5 minutes without log update = STALE

def monitor(pid_to_watch):
    print(f"👀 Monitoring PID {pid_to_watch} via {LOG_FILE}...")
    last_size = -1
    last_update = time.time()

    while True:
        try:
            # Check if process is still alive
            os.kill(pid_to_watch, 0)
        except ProcessLookupError:
            print("🎯 Process finished naturally.")
            return

        try:
            if os.path.exists(LOG_FILE):
                current_size = os.path.getsize(LOG_FILE)
                
                if current_size > last_size:
                    # Log is growing, things are moving!
                    last_size = current_size
                    last_update = time.time()
                    # print(f"✅ Progress detected at {time.ctime()}")
                else:
                    # Log hasn't changed. Check how long.
                    inactive_time = time.time() - last_update
                    if inactive_time > TIMEOUT_SECONDS:
                        print(f"🚨 STALE DETECTED! No progress for {inactive_time:.0f}s. Terminating PID {pid_to_watch}...")
                        os.kill(pid_to_watch, signal.SIGKILL)
                        return # Exit watchdog
            else:
                # Log doesn't exist yet, wait a bit
                pass

            time.sleep(30) # Check every 30 seconds
            
        except Exception as e:
            print(f"Error in watchdog: {e}")
            time.sleep(5)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python watchdog.py <PID_TO_WATCH>")
        sys.exit(1)
    
    pid = int(sys.argv[1])
    monitor(pid)
