"""Block until no process whose command line contains one of the given patterns is running."""
import sys
import time
import psutil

pats = sys.argv[1:]
while True:
    busy = []
    for p in psutil.process_iter(['cmdline']):
        try:
            cl = ' '.join(p.info['cmdline'] or [])
        except Exception:
            continue
        if any(s in cl for s in pats) and 'wait_idle' not in cl:
            busy.append(cl[-80:])
    if not busy:
        break
    time.sleep(60)
print('idle', time.ctime(), flush=True)
