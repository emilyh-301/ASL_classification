from path import Path

path = Path()

path.delete('Models')
path.delete('Exceptions')
path.delete('Shared')
path.delete('research.py')
path.create('Models')
path.create('Exceptions')
path.create('Shared')
path.create(path.var.used_filepath)
path.write(path.var.status_filepath, '1')
path.copy('main.py', 'research.py')