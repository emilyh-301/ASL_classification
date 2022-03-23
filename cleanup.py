import sys
import time
from path import Path
from filelock import FileLock

p = Path()

if len(sys.argv) == 2 and sys.argv[1] == 'kill':
    with FileLock('status'):
        p.write('status.txt', '1')
    time.sleep(3)

p.delete('Models')
p.delete('Exceptions')
p.delete('used.txt')
p.delete('status.txt')
p.delete('research.py')
p.create('Models')
p.create('Exceptions')
p.create('used.txt')
p.write('status.txt', '1')
p.copy('main.py', 'research.py')