#!/usr/bin/env python3
"""dropcache.py DIR...: evict every regular file under DIR from the page cache (posix_fadvise DONTNEED, no root)."""
import os, sys
n = 0
for d in sys.argv[1:]:
    for root, _, files in os.walk(d):
        for f in files:
            p = os.path.join(root, f)
            try:
                fd = os.open(p, os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd); n += 1
            except OSError:
                pass
print(f"dropped {n} files")
