import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(this_dir)
print(f'Work dir: {this_dir}')

# Add lib to PYTHONPATH
lib_path = os.path.join(this_dir, 'lib')
add_path(lib_path)
