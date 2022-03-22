from path import Path

p = Path()

p.delete('Models')
p.delete('Exceptions')
p.delete('used.txt')
p.create('Models')
p.create('Exceptions')
p.create('used.txt')