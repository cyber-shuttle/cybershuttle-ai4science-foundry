#!/usr/bin/env python3
# Run on the login node:
#   python3 watch_job.py <JOB_ID>          # watch a specific job
#   python3 watch_job.py                   # watch your most recently submitted job
#   python3 watch_job.py <JOB_ID> 30       # check every 30 s instead of 15

import glob
import os
import subprocess
import sys
import time

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                       universal_newlines=True)
    return p.stdout.strip()


def latest_job():
    ids = run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i"]).split()
    return max(ids, key=lambda x: int(x.split("_")[0])) if ids else None


def now():
    return time.strftime("%H:%M:%S")


def main():
    job = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else latest_job()
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    if not job:
        sys.exit("No jobs found in your queue. Usage: python3 watch_job.py <JOB_ID>")

    print("[{}] watching job {} (checking every {} s, Ctrl+C to stop)".format(now(), job, interval))
    last = None
    start = time.time()
    try:
        while True:
            line = run(["squeue", "-h", "-j", job, "-o", "%T|%R"])
            if not line:
                break  # job left the queue -> finished
            state, reason = (line.split("|", 1) + [""])[:2]
            if state != last:
                extra = " ({})".format(reason) if reason and reason != "None" else ""
                print("[{}] {} -> {}{}".format(now(), last or "start", state, extra), flush=True)
                last = state
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[{}] stopped watching (job {} is still {})".format(now(), job, last))
        return

    # job is out of the queue: ask sacct for the final result (it can lag a few seconds)
    final = ""
    for _ in range(6):
        final = run(["sacct", "-n", "-X", "-j", job, "-o", "State,Elapsed,ExitCode", "-P"])
        if final and not final.startswith(("RUNNING", "PENDING", "COMPLETING")):
            break
        time.sleep(5)
    mins = (time.time() - start) / 60
    print("[{}] {} -> finished after watching {:.1f} min".format(now(), last or "start", mins))
    if final:
        state, elapsed, code = (final.splitlines()[0].split("|") + ["", "", ""])[:3]
        print("           final state: {}   run time: {}   exit code: {}".format(state, elapsed, code))

    for path in glob.glob("outputs/*_{}/validation.txt".format(job)):
        summary = [l for l in open(path) if l.startswith("SUMMARY")]
        print("           {}: {}".format(path, summary[-1].strip() if summary else "(no summary line)"))
    print("\a", end="", flush=True)  # terminal beep


if __name__ == "__main__":
    main()
~
~
