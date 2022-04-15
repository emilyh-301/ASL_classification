from path import Path

path = Path()

path.delete('Exception')
path.delete('Shared')
path.delete('research.py')

path.create('Exception')
path.create('Shared')

path.create(path.var.used_filepath)
path.create(path.var.alive_filepath)
path.write(path.var.status_filepath, '1')

path.copy('main.py', 'research.py')