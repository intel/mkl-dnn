#!/usr/bin/env python3
"""benchdnn.py - hard-timeout monitor/wrapper around the benchdnn binary.

PoC. Wraps the sibling `benchdnn` executable, forwards all arguments except the
monitor-only ones below, and watches the worker's stdout. If no output appears
for `--timeout` seconds, the shape currently in flight is considered hung (a
real hang or a too-slow job); the worker process tree is killed and the run is
resumed past the offender via benchdnn's `--start=N`. The per-run logs are
stitched into a single, renumbered output with one merged summary footer.

Monitor-only options (consumed here, NOT forwarded to benchdnn):
  --timeout=SECONDS     silence budget per shape (required to activate)
  --benchdnn-exe=PATH   path to the worker binary (default: sibling ./benchdnn)
  --monitor-verbose     print monitor diagnostics on stderr

Attribution: a hang during execution prints a `run:` line first, so the
offending shape is known exactly. A hang during *parallel* creation
(--mode-modifier=P) prints nothing per shape; the monitor then transparently
relaunches the remainder with parallel creation disabled, which makes benchdnn
print a `create:` line before each shape and pins the offender exactly. No
knowledge of the parallel-create group size (thread count) is needed.

Cross-platform: pure subprocess + a reader thread (no select on pipes), and a
process-tree kill that works on both POSIX (process group) and Windows
(taskkill /T).
"""
import os
import re
import signal
import subprocess
import sys
import threading
import queue

IS_WIN = os.name == "nt"

# Matches a benchdnn result line: "<idx>:<STATUS>[ (...)]... __REPRO: <prb>"
RESULT_RE = re.compile(r"^(\d+):([A-Z_]+)(.*?)__REPRO:\s*(.*)$")
SKIP_START_RE = re.compile(r"Skip-start option hit")


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


def status_bucket(status):
    return {
        "PASSED": ["passed"],
        "SKIPPED": ["skipped"],
        "MISTRUSTED": ["mistrusted"],
        "LISTED": ["listed"],
        "FAILED": ["failed"],
        "UNIMPLEMENTED": ["failed", "unimplemented"],
        "INVALID_ARGUMENTS": ["failed", "invalid_arguments"],
    }.get(status, ["failed"])


class Stats:
    def __init__(self):
        self.c = dict.fromkeys(
            ("tests passed skipped mistrusted unimplemented "
             "invalid_arguments failed listed timeouts").split(), 0)
        self.failed_cases = []

    def add(self, status, repro, line_no_idx):
        self.c["tests"] += 1
        for b in status_bucket(status):
            self.c[b] += 1
        if status not in ("PASSED", "SKIPPED", "MISTRUSTED", "LISTED"):
            self.failed_cases.append(f"{line_no_idx} __REPRO: {repro}")

    def add_timeout(self, repro, line_no_idx):
        self.c["tests"] += 1
        self.c["failed"] += 1
        self.c["timeouts"] += 1
        self.failed_cases.append(f"{line_no_idx} __REPRO: {repro}")

    def footer(self):
        c = self.c
        return ("tests:{tests} passed:{passed} skipped:{skipped} "
                "mistrusted:{mistrusted} unimplemented:{unimplemented} "
                "invalid_arguments:{invalid_arguments} failed:{failed} "
                "listed:{listed} timeouts:{timeouts}").format(**c)


def kill_tree(proc):
    if proc.poll() is not None:
        return
    try:
        if IS_WIN:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=10)
    except Exception:
        pass


def spawn(cmd):
    kw = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
              bufsize=1, universal_newlines=True)
    if IS_WIN:
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    return subprocess.Popen(cmd, **kw)


def reader_thread(proc, q):
    try:
        for line in iter(proc.stdout.readline, ""):
            q.put(line.rstrip("\n"))
    finally:
        q.put(None)  # EOF sentinel


def parse_monitor_args(argv):
    timeout = None
    exe = None
    mon_verbose = False
    user_verbose = 0
    forwarded = []
    for a in argv:
        if a.startswith("--timeout="):
            timeout = float(a.split("=", 1)[1])
        elif a.startswith("--benchdnn-exe="):
            exe = a.split("=", 1)[1]
        elif a == "--monitor-verbose":
            mon_verbose = True
        elif a.startswith("--verbose=") or a.startswith("-v"):
            # capture and drop user's verbose; monitor needs >= 1
            try:
                user_verbose = int(a.split("=", 1)[1] if "=" in a else a[2:])
            except ValueError:
                user_verbose = 1
        elif a.startswith("--start="):
            # monitor manages --start; drop user's (rare)
            pass
        else:
            forwarded.append(a)
    return dict(timeout=timeout, exe=exe, mon_verbose=mon_verbose,
                user_verbose=user_verbose, forwarded=forwarded)


def strip_par_create(fwd):
    """Return a copy of `fwd` with `P` removed from any --mode-modifier= value
    (dropping the option if it becomes empty), forcing sequential creation."""
    out = []
    for a in fwd:
        if a.startswith("--mode-modifier="):
            val = a.split("=", 1)[1].replace("P", "").replace("p", "")
            if val:
                out.append("--mode-modifier=" + val)
            # else: drop the now-empty modifier entirely
        else:
            out.append(a)
    return out


def main():
    cfg = parse_monitor_args(sys.argv[1:])
    here = os.path.dirname(os.path.abspath(__file__))
    exe = cfg["exe"] or os.path.join(here, "benchdnn.exe" if IS_WIN else "benchdnn")

    # No timeout -> transparent passthrough.
    if cfg["timeout"] is None:
        os.execv(exe, [exe] + cfg["forwarded"]) if not IS_WIN else \
            sys.exit(subprocess.run([exe] + cfg["forwarded"]).returncode)
        return

    timeout = cfg["timeout"]
    veff = max(cfg["user_verbose"], 1)
    fwd = cfg["forwarded"]
    log = eprint if cfg["mon_verbose"] else (lambda *a: None)

    stats = Stats()
    gidx = 0          # global renumber counter for emitted lines
    start = 0         # next problem index to run
    force_seq = False  # set after a parallel-creation hang: drop --mode-modifier=P

    def emit_result(status, tail, repro):
        nonlocal gidx
        line = f"{gidx}:{status}{tail}__REPRO: {repro}"
        print(line, flush=True)
        stats.add(status, repro, f"{gidx}:{status}")
        gidx += 1

    def emit_timeout(repro):
        nonlocal gidx
        line = f"{gidx}:FAILED (TIMEOUT {int(timeout)} sec) __REPRO: {repro}"
        print(line, flush=True)
        stats.add_timeout(repro, f"{gidx}:FAILED (TIMEOUT {int(timeout)} sec)")
        gidx += 1

    while True:
        run_fwd = strip_par_create(fwd) if force_seq else fwd
        cmd = [exe, f"--verbose={veff}", f"--start={start}"] + run_fwd
        log(f"[monitor] launch start={start} seq={force_seq}: {' '.join(cmd)}")
        proc = spawn(cmd)
        q = queue.Queue()
        threading.Thread(target=reader_thread, args=(proc, q), daemon=True).start()

        last_done = start - 1        # highest completed global idx this segment
        saw_run = False              # a run: line appeared after last result
        inflight_repro = None        # repro of shape currently being processed
        saw_summary = False
        hung = False

        while True:
            try:
                line = q.get(timeout=timeout)
            except queue.Empty:
                hung = True
                break
            if line is None:          # EOF: worker exited on its own
                break

            m = RESULT_RE.match(line)
            if m:
                idx, status, tail, repro = m.group(1, 2, 3, 4)
                if status == "SKIPPED" and SKIP_START_RE.search(tail):
                    continue          # drop resume skip-noise
                emit_result(status, tail, repro)
                last_done = int(idx)
                saw_run = False
                inflight_repro = None
            elif line.startswith("run: "):
                saw_run = True
                inflight_repro = line[len("run: "):].strip()
            elif line.startswith("create: "):
                inflight_repro = line[len("create: "):].strip()
            elif line.startswith("tests:"):
                saw_summary = True    # worker's own footer; suppress it
            else:
                pass                  # impl stats / total: / banners: suppress

        if hung:
            log(f"[monitor] TIMEOUT after {timeout}s (saw_run={saw_run}, "
                f"inflight={inflight_repro!r}, last_done={last_done})")
            kill_tree(proc)
            if saw_run or inflight_repro is not None:
                # Offender is pinned: either an exec-phase hang (run: printed)
                # or a sequential creation hang (create: printed). Skip just it.
                emit_timeout(inflight_repro or f"<index {last_done + 1}>")
                start = last_done + 2
            else:
                # Parallel-creation hang: offender not yet identifiable. Don't
                # emit/skip anything; relaunch the same remainder with parallel
                # creation disabled so the offender gets a create: line and is
                # pinned on the next pass. Good shapes in the group simply re-run.
                log("[monitor] parallel-creation hang; "
                    "falling back to sequential creation")
                force_seq = True
                start = last_done + 1
            continue                  # relaunch from new start

        rc = proc.wait()
        if not saw_summary and rc != 0:
            # Crash mid-batch (segfault etc.): treat in-flight shape as failed
            # and resume past it.
            log(f"[monitor] worker crashed rc={rc}; resuming")
            emit_timeout(inflight_repro or f"<crash near index {last_done + 1}>")
            start = last_done + 2
            continue
        break

    print("=" * 60, flush=True)
    print(stats.footer(), flush=True)
    if stats.failed_cases:
        print("= Failed/timeout cases summary =", flush=True)
        for fc in stats.failed_cases:
            print(fc, flush=True)
    sys.exit(1 if stats.c["failed"] else 0)


if __name__ == "__main__":
    main()
