from os.path import isabs, join
import os
import sys
import subprocess
import re

try:
    subprocess.check_output(
        '{} -c "import numba"'.format(sys.executable), shell=True
    )
    print('numba available, importing jit')
    from numba import jit
except:
    print('cannot import numba, creating dummy jit definition')
    def jit(function):
        def wrapper(*args,**kwargs):
            return function(*args,**kwargs)
        return wrapper

def compose_path(file_path, reference_path):
    if reference_path and not isabs(file_path):
        file_path = join(reference_path, file_path)
    return file_path

def setGPU():
    freeGpu = subprocess.check_output('nvidia-smi -q | grep "Minor\|Processes" | grep "None" -B1 | tr -d " " | cut -d ":" -f2 | sed -n "1p"', shell=True)
    freeGpu = freeGpu.decode().strip()
    if len(freeGpu) == 0:
        print('no free GPU')
        sys.exit(1)

    os.environ['CUDA_VISIBLE_DEVICES'] = freeGpu
    print('got GPU ' + (freeGpu))

    return(int(freeGpu.strip()))
