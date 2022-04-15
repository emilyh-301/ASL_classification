import os
import shutil
import traceback

from variable import Variable


def handle_exception(func):
    def decorate(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            print(traceback.format_exc())
    return decorate


def apply_to_all(func):
    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)):
                setattr(cls, attr, func(getattr(cls, attr)))
        return cls
    return decorate


@apply_to_all(handle_exception)
class Path:
    var = Variable()

    def copy(self, src, dst):
        shutil.copyfile(src, dst)

    def create(self, *args):
        # File
        if '.' in args[-1]:
            open(self.join(*args), 'w')
        # Directory
        else:
            os.mkdir(self.join(*args))

    def delete(self, *args):
        # File
        if '.' in args[-1]:
            os.remove(self.join(*args))
        # Directory
        else:
            shutil.rmtree(self.join(*args))

    def exists(self, *args):
        return os.path.exists(self.join(*args))

    def is_empty(self, *args):
        return len(self.listdir(*args)) == 0

    def listdir(self, *args):
        return os.listdir(self.join(*args))

    def join(self, *args):
        return os.path.join(*args)

    def read(self, filepath: str) -> list:
        with open(filepath, 'r') as f:
            contents = f.read().splitlines()
        return contents

    def write(self, filepath: str, content: str, mode='a') -> None:
        with open(filepath, mode=mode) as f:
            f.write(content)
