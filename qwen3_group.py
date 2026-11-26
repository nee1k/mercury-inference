"""Keep a vLLM process group tied to its Serve replica's pipe."""

import ctypes
import os
import select
import signal
import subprocess
import sys
import time


def main():
    lifeline = int(sys.argv[1])
    # Become the reaper for engine processes orphaned during shutdown.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER")
    signal.signal(signal.SIGTERM, lambda *_: None)
    child = subprocess.Popen(sys.argv[2:], stdin=subprocess.DEVNULL)
    while child.poll() is None and not select.select([lifeline], [], [], 0.25)[0]:
        pass
    os.killpg(os.getpgrp(), signal.SIGTERM)
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            time.sleep(0.1)
    os.killpg(os.getpgrp(), signal.SIGKILL)


if __name__ == "__main__":
    main()
