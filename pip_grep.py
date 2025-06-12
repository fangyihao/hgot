'''
Created on Jun 11, 2025

@author: Yihao Fang
'''
import re
import subprocess
import os
command = "ls -l"  # Example command
try:
    output = subprocess.check_output('grep -r "import" --include="*.py" .', shell=True)
    libs = output.decode().strip('\n').split('\n')
    libs = set([re.sub(r'.+\.py\:', '', lib).strip() for lib in libs])
    libs = [lib for lib in libs if not lib.startswith('#')]
    libs = [lib.split(' ')[1] for lib in libs if lib.startswith('from') or lib.startswith('import')]
    libs = sorted(set([lib.split('.')[0] for lib in libs if not lib.startswith('.')]))
    files = [file.split('.')[0] for file in os.listdir() if file.endswith('.py')]
    libs = [lib for lib in libs if lib not in files]
    print('\n'.join(libs))
except subprocess.CalledProcessError as e:
    print(f"Command failed with error: {e}")



    