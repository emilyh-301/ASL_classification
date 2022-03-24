from filelock import FileLock
from path import Path

path = Path()

with FileLock(path.var.status_lock):
    path.write(path.var.status_filepath, '1')
