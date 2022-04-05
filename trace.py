import os
import time
import threading
import socket
from filelock import FileLock
from path import Path

path = Path()


# To kill all processes at the same time
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
                self.update_liveness(alive=False)
                os.system('kill ' + str(pid))
            time.sleep(0.5)

    def start(self) -> None:
        threading.Thread(target=self.status, daemon=True).start()

    def update_liveness(self, alive):
        host = socket.gethostname()
        pid = os.getpid()
        name = host + '_' + str(pid)

        with FileLock(path.var.alive_lock):
            if alive:
                path.write(filepath=path.var.alive_filepath, content=name + '\n')
            else:
                contents = path.read(filepath=path.var.alive_filepath)
                path.create(path.var.alive_filepath)
                for content in contents:
                    if content[:len(name)] != name:
                        path.write(filepath=path.var.alive_filepath, content=content + '\n')
