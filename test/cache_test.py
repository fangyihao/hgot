'''
Created on Dec. 4, 2023

@author: Yihao Fang
'''

import functools
from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory, cache_turn_on
import time
import numpy as np
#cache_turn_on = False

def timeis(func): 
    '''Decorator that reports the execution time.'''
  
    def wrap(*args, **kwargs): 
        start = time.time() 
        result = func(*args, **kwargs) 
        end = time.time() 
          
        print(func.__name__, end-start) 
        return result 
    return wrap 


@timeis
@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@CacheMemory.cache
def countdown(n): 
    print(n[0])

countdown((5,)) 
countdown((1000,)) 



rng = np.random.RandomState(42)
l = ["A", "BC"]


from joblib import Memory
location = './cachedir'
memory = Memory(location, verbose=0)

@memory.cache
def costly_compute_cached(**kwargs):
    """Simulate an expensive computation"""
    time.sleep(1)
    column_index = kwargs["column_index"]
    data = kwargs["data"]
    return data[column_index]


#costly_compute_cached = memory.cache(costly_compute_cached)
start = time.time()
data_trans = costly_compute_cached(data=l, column_index=0)
end = time.time()

print('\nThe function took {:.2f} s to compute.'.format(end - start))
print('\nThe transformed data are:\n {}'.format(data_trans))





