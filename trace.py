import os
import time
import threading
from filelock import FileLock
from path import Path

path = Path()


class Trace:
    def status(self) -> None:
        pid = os.getpid()
        while True:
            kill = False
            with FileLock(path.var.status_lock):
                status = path.read(filepath=path.var.status_filepath)[0]
                if status[:2] == path.var.dead:
                    kill = True
            if kill:
                os.system('kill ' + str(pid))
            time.sleep(0.5)

    def start(self) -> None:
        threading.Thread(target=self.status, daemon=True).start()