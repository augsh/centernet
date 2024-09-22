import os
import sys


def add_path(path):
  assert os.path.isdir(path)
  path = os.path.abspath(path)
  if path not in sys.path:
    sys.path.insert(0, path)


init_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(init_dir)
print(f'Chdir to {init_dir}')

# Add lib to PYTHONPATH
add_path('./lib')
