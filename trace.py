import os
import time
import threading
from filelock import FileLock
from path import Path

p = Path()


class Trace:
    def status(self) -> None:
        pid = os.getpid()
        while True:
            with FileLock('status'):
                status = p.read(filepath='status.txt')[0]
                if status[:2] == '11':
                    os.system('kill ' + str(pid))
            time.sleep(0.5)

    def start(self) -> None:
        threading.Thread(target=self.status, daemon=True).start()